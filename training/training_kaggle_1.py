import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==============================================================================
# 1. CONFIGURACIÓN
# ==============================================================================
CONFIG = {
    'audio': {'sample_rate': 16000, 'frame_rate': 250},
    'model': {'n_harmonics': 96, 'n_noise_filters': 64, 'hidden_size': 128, 'gru_units': 192, 'dense_output_size': 192, 'total_params_output': 161},
    'loss': {'fft_sizes': [2048, 1024, 512, 256, 128, 64], 'hop_ratio': 0.25},
    'training': {'learning_rate': 1e-3, 'global_batch_size': 12}
}

# --- BÚSQUEDA AUTOMÁTICA DE RUTAS EN KAGGLE ---
print("\n[INFO] Buscando archivos TFRecord en Kaggle...")
TRAIN_TFRECORD = None
VAL_TFRECORD = None

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if filename == 'maestro_2009_train.tfrecord':
            TRAIN_TFRECORD = os.path.join(dirname, filename)
        elif filename == 'maestro_2009_validation.tfrecord':
            VAL_TFRECORD = os.path.join(dirname, filename)

if TRAIN_TFRECORD is None or VAL_TFRECORD is None:
    raise FileNotFoundError("¡Error! No se encontraron los archivos TFRecord. Verifica que el dataset está añadido.")

print(f"[OK] Train detectado en: {TRAIN_TFRECORD}")
print(f"[OK] Validation detectado en: {VAL_TFRECORD}")

# --- CONSTANTES ESTÁTICAS ---
N_SAMPLES = int(CONFIG['audio']['sample_rate'] * 3.0) # 48000
# ==============================================================================

# ==============================================================================
# 2. ARQUITECTURA DSP (Síntesis)
# ==============================================================================
def get_modified_sigmoid(x):
    return 2.0 * tf.math.pow(tf.math.sigmoid(x), tf.math.log(10.0)) + 1e-7

def upsample_controls(controls, n_samples):
    controls_4d = tf.expand_dims(controls, axis=2)
    upsampled_4d = tf.image.resize(controls_4d, [n_samples, 1], method=tf.image.ResizeMethod.BILINEAR)
    return tf.squeeze(upsampled_4d, axis=2)

class HarmonicSynthesizer:
    def __init__(self, sample_rate=16000, frame_rate=250, n_harmonics=96):
        self.sr = sample_rate
        self.n_harmonics = n_harmonics

    def __call__(self, f_k, amplitude, harmonics):
        # f_k viene de la red, ya convertido a Hz e inarmónico
        f_k_up = upsample_controls(f_k, N_SAMPLES)
        amp_up = upsample_controls(amplitude, N_SAMPLES)
        harm_up = upsample_controls(harmonics, N_SAMPLES)

        antialiasing_mask = tf.cast(f_k_up < (self.sr / 2.0), tf.float32)

        phases = tf.math.cumsum(f_k_up * (2.0 * np.pi / self.sr), axis=1)
        wavs = tf.math.sin(phases) * harm_up * antialiasing_mask
        
        audio = tf.reduce_sum(wavs, axis=-1, keepdims=True) * amp_up
        return tf.squeeze(audio, axis=-1)

class FilteredNoiseSynthesizer:
    def __init__(self, sample_rate=16000, frame_rate=250, n_noise_filters=64, per_replica_batch_size=6):
        self.hop_size = sample_rate // frame_rate
        self.window_size = self.hop_size * 2
        self.static_batch_size = per_replica_batch_size

    def __call__(self, noise_magnitudes):
        white_noise = tf.random.uniform([self.static_batch_size, N_SAMPLES], minval=-1.0, maxval=1.0)
        
        noise_frames = tf.signal.frame(white_noise, frame_length=self.window_size, frame_step=self.hop_size, pad_end=True)
        noise_fft = tf.signal.rfft(noise_frames)

        magnitudes_padded = tf.pad(noise_magnitudes, paddings=[[0, 0], [0, 0], [0, 1]])
        filtered_fft = noise_fft * tf.cast(magnitudes_padded, tf.complex64)
        
        filtered_frames = tf.signal.irfft(filtered_fft) * tf.signal.hann_window(self.window_size)
        audio_noise = tf.signal.overlap_and_add(filtered_frames, frame_step=self.hop_size)
        return audio_noise[:, :N_SAMPLES]

# ==============================================================================
# 3. RED NEURONAL CORE (Con Inarmonicidad)
# ==============================================================================
class InharmonicityModel(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Valores físicos empíricos para un piano
        self.alpha_B = tf.Variable(-0.0847, dtype=tf.float32, trainable=True)
        self.beta_B = tf.Variable(-5.82, dtype=tf.float32, trainable=True)
        self.alpha_T = tf.Variable(0.0926, dtype=tf.float32, trainable=True)
        self.beta_T = tf.Variable(-13.64, dtype=tf.float32, trainable=True)

    def call(self, pitch_midi):
        # Evitamos que silencios (pitch=0) rompan el cálculo
        pitch_safe = tf.where(pitch_midi > 0, pitch_midi, 1.0)
        
        term_T = tf.exp(self.alpha_T * pitch_safe + self.beta_T)
        term_B = tf.exp(self.alpha_B * pitch_safe + self.beta_B)
        return term_T + term_B

class DDSPCore(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        cfg = config['model']
        self.n_harmonics = cfg['n_harmonics']
        
        self.dense_in = tf.keras.layers.Dense(cfg['hidden_size'])
        self.leaky_in = tf.keras.layers.LeakyReLU(negative_slope=0.2)
        self.gru = tf.keras.layers.GRU(cfg['gru_units'], return_sequences=True)
        self.dense_hidden = tf.keras.layers.Dense(cfg['dense_output_size'])
        self.leaky_hidden = tf.keras.layers.LeakyReLU(negative_slope=0.2)
        
        # Inharmonicity
        self.inharmonicity = InharmonicityModel()

        # NUEVO: Inicializaciones forzadas para no colapsar al ruido
        self.amp_out = tf.keras.layers.Dense(1, bias_initializer=tf.keras.initializers.Constant(-5.0))
        self.harm_out = tf.keras.layers.Dense(self.n_harmonics, bias_initializer='zeros')
        self.noise_out = tf.keras.layers.Dense(cfg['n_noise_filters'], bias_initializer=tf.keras.initializers.Constant(-5.0))

    def call(self, pitch_midi, velocity, training=False):
        x = tf.concat([pitch_midi, velocity], axis=-1)
        x = self.leaky_hidden(self.dense_hidden(self.gru(self.leaky_in(self.dense_in(x)), training=training)))
        
        # Activaciones
        amp = get_modified_sigmoid(self.amp_out(x))
        harm_dist = get_modified_sigmoid(self.harm_out(x))
        noise_mags = get_modified_sigmoid(self.noise_out(x))
        
        # Normalizar armónicos
        harm_dist /= tf.reduce_sum(harm_dist, axis=-1, keepdims=True)
        
        # 1. Conversión de MIDI a Hz real en TensorFlow
        # Fórmula: f_hz = 440 * 2^((midi - 69) / 12)
        f0_hz = 440.0 * tf.math.pow(2.0, (pitch_midi - 69.0) / 12.0)
        # Silenciar la frecuencia si no hay nota pulsada
        f0_hz = tf.where(pitch_midi > 0, f0_hz, 0.0)
        
        # 2. Calcular inarmonicidad
        B_factor = self.inharmonicity(pitch_midi)
        k = tf.reshape(tf.range(1, self.n_harmonics + 1, dtype=tf.float32), [1, 1, self.n_harmonics])
        stretch_factor = tf.sqrt(1.0 + B_factor * tf.square(k))
        
        # 3. Matriz de frecuencias finales estiradas
        f_k = f0_hz * k * stretch_factor
        
        return {
            'amplitude': amp,
            'harmonics': harm_dist,
            'noise_magnitudes': noise_mags,
            'f_k': f_k  # Devolvemos la matriz de frecuencias listas
        }

# ==============================================================================
# 4. FUNCIÓN DE PÉRDIDA
# ==============================================================================
class MultiResolutionSpectralLoss(tf.keras.losses.Loss):
    def __init__(self, config, name="multi_resolution_spectral_loss"):
        super().__init__(name=name, reduction=tf.keras.losses.Reduction.NONE)
        self.fft_sizes = config['loss']['fft_sizes']
        self.hop_ratio = config['loss']['hop_ratio']

    def spectral_loss(self, y_true, y_pred, fft_size):
        hop_size = int(fft_size * self.hop_ratio)
        stft_true = tf.signal.stft(y_true, frame_length=fft_size, frame_step=hop_size)
        stft_pred = tf.signal.stft(y_pred, frame_length=fft_size, frame_step=hop_size)
        
        mag_true, mag_pred = tf.abs(stft_true), tf.abs(stft_pred)
        log_mag_true, log_mag_pred = tf.math.log(mag_true + 1e-7), tf.math.log(mag_pred + 1e-7)
        
        lin_loss = tf.reduce_mean(tf.abs(mag_true - mag_pred), axis=[1, 2])
        log_loss = tf.reduce_mean(tf.abs(log_mag_true - log_mag_pred), axis=[1, 2])
        
        return lin_loss + log_loss

    def call(self, y_true, y_pred):
        if len(y_true.shape) == 3: y_true = tf.squeeze(y_true, axis=-1)
        if len(y_pred.shape) == 3: y_pred = tf.squeeze(y_pred, axis=-1)
        
        total_loss = self.spectral_loss(y_true, y_pred, self.fft_sizes[0])
        for size in self.fft_sizes[1:]:
            total_loss += self.spectral_loss(y_true, y_pred, size)
            
        return total_loss

# ==============================================================================
# 5. BUCLE DE ENTRENAMIENTO DISTRIBUIDO
# ==============================================================================
def parse_tfrecord(example_proto):
    desc = {
        'pitch': tf.io.FixedLenFeature([750], tf.float32), 
        'velocity': tf.io.FixedLenFeature([750], tf.float32), 
        'audio': tf.io.FixedLenFeature([48000], tf.float32)
    }
    parsed = tf.io.parse_single_example(example_proto, desc)
    return tf.expand_dims(parsed['pitch'], -1), tf.expand_dims(parsed['velocity'], -1), parsed['audio']

def get_distributed_dataset(tfrecord_path, global_batch_size, is_training=True):
    dataset = tf.data.TFRecordDataset(tfrecord_path).map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache()
    if is_training:
        dataset = dataset.shuffle(1000).repeat()
    return dataset.batch(global_batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

def run_training():
    if not os.path.exists('/kaggle/working/checkpoints'):
        os.makedirs('/kaggle/working/checkpoints')

    strategy = tf.distribute.MirroredStrategy()
    
    global_batch_size = CONFIG['training']['global_batch_size']
    num_replicas = strategy.num_replicas_in_sync
    per_replica_batch_size = global_batch_size // num_replicas

    dist_train_dataset = strategy.experimental_distribute_dataset(get_distributed_dataset(TRAIN_TFRECORD, global_batch_size, is_training=True))
    dist_val_dataset = strategy.experimental_distribute_dataset(get_distributed_dataset(VAL_TFRECORD, global_batch_size, is_training=False))

    with strategy.scope():
        model = DDSPCore(CONFIG)
        harmonic_synth = HarmonicSynthesizer()
        noise_synth = FilteredNoiseSynthesizer(per_replica_batch_size=per_replica_batch_size)
        loss_fn = MultiResolutionSpectralLoss(CONFIG)
        optimizer = tf.keras.optimizers.Adam(learning_rate=CONFIG['training']['learning_rate'])

        def compute_loss(labels, predictions):
            return tf.nn.compute_average_loss(loss_fn(labels, predictions), global_batch_size=global_batch_size)

        print("\n[INFO] Construyendo variables (Evitando bug de MirroredStrategy)...")
        dummy_pitch = tf.zeros((1, 750, 1), dtype=tf.float32)
        dummy_velocity = tf.zeros((1, 750, 1), dtype=tf.float32)
        _ = model(dummy_pitch, dummy_velocity) 
        
        optimizer.build(model.trainable_variables) 
        print("[OK] Red y Optimizador completamente listos en memoria estática.")

    with strategy.scope():
        @tf.function
        def train_step(pitch, velocity, real_audio):
            with tf.GradientTape() as tape:
                params = model(pitch, velocity, training=True)
                # Ojo: ahora le pasamos 'f_k' en lugar de 'pitch' al HarmonicSynth
                audio_pred = harmonic_synth(params['f_k'], params['amplitude'], params['harmonics']) + noise_synth(params['noise_magnitudes'])
                loss = compute_loss(real_audio, audio_pred)
            
            gradients, _ = tf.clip_by_global_norm(tape.gradient(loss, model.trainable_variables), 3.0)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss

        @tf.function
        def distributed_train_step(inputs):
            return strategy.reduce(tf.distribute.ReduceOp.SUM, strategy.run(train_step, args=inputs), axis=None)

        @tf.function
        def val_step(pitch, velocity, real_audio):
            params = model(pitch, velocity, training=False)
            audio_pred = harmonic_synth(params['f_k'], params['amplitude'], params['harmonics']) + noise_synth(params['noise_magnitudes'])
            return compute_loss(real_audio, audio_pred)

        @tf.function
        def distributed_val_step(inputs):
            return strategy.reduce(tf.distribute.ReduceOp.SUM, strategy.run(val_step, args=inputs), axis=None)

    epochs, steps_per_epoch = 300, 500 
    historial_train_loss = []
    historial_val_loss = []
    
    best_val_loss = float('inf') 
    
    # --- VARIABLES PARA REDUCE LR ON PLATEAU ---
    patience = 10
    factor = 0.5
    min_lr = 1e-6
    wait = 0

    print(f"\n[INFO] Configuración lista: Batch Global={global_batch_size}, Lote por GPU={per_replica_batch_size}")
    print("[INFO] Iniciando Entrenamiento con Validación y TQDM...")
    train_iterator = iter(dist_train_dataset)

    for epoch in range(epochs):
        total_train_loss = 0.0
        
        with tqdm(total=steps_per_epoch, desc=f"Época {epoch + 1:03d}/{epochs}", unit="batch") as pbar:
            for _ in range(steps_per_epoch):
                total_train_loss += distributed_train_step(next(train_iterator))
                pbar.update(1)
                
            avg_train_loss = (total_train_loss / steps_per_epoch).numpy()
            historial_train_loss.append(avg_train_loss)
            
            total_val_loss = 0.0
            val_batches = 0
            for dist_inputs in dist_val_dataset:
                total_val_loss += distributed_val_step(dist_inputs)
                val_batches += 1
            avg_val_loss = (total_val_loss / val_batches).numpy() if val_batches > 0 else 0.0
            historial_val_loss.append(avg_val_loss)
            
            # Obtenemos el LR actual para mostrarlo en TQDM
            current_lr = optimizer.learning_rate.numpy()
            pbar.set_postfix({"Train Loss": f"{avg_train_loss:.4f}", "Val Loss": f"{avg_val_loss:.4f}", "LR": f"{current_lr:.1e}"})
        
        # --- LÓGICA REDUCE LR ON PLATEAU ---
        if avg_val_loss < best_val_loss and avg_val_loss > 0:
            best_val_loss = avg_val_loss
            wait = 0 # Reseteamos la paciencia al conseguir un récord
            
            best_model_path = '/kaggle/working/checkpoints/ddsp_best_model.weights.h5'
            model.save_weights(best_model_path)
            print(f"  -> [NUEVO RÉCORD] Mejor modelo guardado con Val Loss: {best_val_loss:.4f}")
        else:
            wait += 1 # Aumentamos el contador si no hay mejora
            
            if wait >= patience:
                new_lr = max(current_lr * factor, min_lr)
                if current_lr > new_lr: # Solo aplicamos si realmente baja
                    optimizer.learning_rate.assign(new_lr)
                    print(f"\n  [ATENCIÓN] Plateau detectado tras {patience} épocas. Reduciendo Learning Rate a: {new_lr:.1e}")
                wait = 0 # Reseteamos tras reducir para darle tiempo a actuar
        
        if (epoch + 1) % 20 == 0:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(historial_train_loss) + 1), historial_train_loss, label='Train Loss', color='indigo', linewidth=2)
            plt.plot(range(1, len(historial_val_loss) + 1), historial_val_loss, label='Validation Loss', color='orange', linewidth=2)
            plt.title('Curva de Aprendizaje - DDSP Piano')
            plt.xlabel('Época')
            plt.ylabel('Pérdida (L1 Magnitud + Log-Magnitud)')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.savefig('/kaggle/working/loss_curve.png', dpi=300)
            plt.close()

run_training()