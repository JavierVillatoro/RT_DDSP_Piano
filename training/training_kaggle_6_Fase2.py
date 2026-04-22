# ==============================================================================
# FASE 2 - FINE TUNING FÍSICO (INHARMONICITY & DETUNER ON)
# ==============================================================================
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

tf.keras.backend.clear_session() # Limpiamos la VRAM por si acaso

# ==============================================================================
# 1. CONFIGURACIÓN (Hiperparámetros Quirúrgicos)
# ==============================================================================
CONFIG = {
    'audio': {'sample_rate': 16000, 'frame_rate': 250},
    'model': {'n_harmonics': 96, 'n_noise_filters': 64, 'hidden_size': 128, 'gru_units': 192, 'dense_output_size': 192, 'total_params_output': 161},
    'loss': {'fft_sizes': [2048, 1024, 512, 256, 128, 64], 'hop_ratio': 0.25},
    # FASE 2: LR minúsculo (1e-5) y Batch a la mitad (6 global)
    'training': {'learning_rate': 1e-5, 'global_batch_size': 6} 
}

RESUME_TRAINING = True 

# --- BÚSQUEDA AUTOMÁTICA DE RUTAS EN KAGGLE (AHORA INCLUYE LOS PESOS) ---
print("\n[INFO] Buscando archivos TFRecord y Pesos en Kaggle...")
TRAIN_TFRECORD = None
VAL_TFRECORD = None
PRETRAINED_DIR = None

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if filename == 'maestro_2009_train_poly.tfrecord':
            TRAIN_TFRECORD = os.path.join(dirname, filename)
        elif filename == 'maestro_2009_validation_poly.tfrecord':
            VAL_TFRECORD = os.path.join(dirname, filename)
        elif filename == 'core.weights.h5': # Buscamos a nuestro experto en timbre
            PRETRAINED_DIR = dirname + '/'

if TRAIN_TFRECORD is None or VAL_TFRECORD is None:
    raise FileNotFoundError("¡Error! No se encontraron los archivos TFRecord.")

if RESUME_TRAINING and PRETRAINED_DIR is None:
    raise FileNotFoundError("¡Error! RESUME_TRAINING está en True pero no encuentro 'core.weights.h5'. ¿Seguro que añadiste el dataset de pesos al notebook?")

print(f"[OK] Train detectado en: {TRAIN_TFRECORD}")
print(f"[OK] Validation detectado en: {VAL_TFRECORD}")
if RESUME_TRAINING:
    print(f"[OK] Carpeta de Pesos detectada automáticamente en: {PRETRAINED_DIR}")

N_SAMPLES = int(CONFIG['audio']['sample_rate'] * 3.0)

# ==============================================================================
# 2. FUNCIONES AUXILIARES Y SINTETIZADORES
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

    def __call__(self, noise_magnitudes):
        white_noise = tf.random.uniform([tf.shape(noise_magnitudes)[0], N_SAMPLES], minval=-1.0, maxval=1.0)
        noise_frames = tf.signal.frame(white_noise, frame_length=self.window_size, frame_step=self.hop_size, pad_end=True)
        noise_fft = tf.signal.rfft(noise_frames)

        magnitudes_padded = tf.pad(noise_magnitudes, paddings=[[0, 0], [0, 0], [0, 1]])
        filtered_fft = noise_fft * tf.cast(magnitudes_padded, tf.complex64)
        
        filtered_frames = tf.signal.irfft(filtered_fft)
        filtered_frames = tf.roll(filtered_frames, shift=self.window_size // 2, axis=-1)
        filtered_frames = filtered_frames * tf.signal.hann_window(self.window_size)
        audio_noise = tf.signal.overlap_and_add(filtered_frames, frame_step=self.hop_size)
        return audio_noise[:, :N_SAMPLES]

# ==============================================================================
# 3. MÓDULOS DE MODELADO FÍSICO Y CONTEXTO
# ==============================================================================
class ContextNetwork(tf.keras.Model): 
    def __init__(self, context_dim=32, gru_units=64):
        super().__init__()
        self.trainable = False # FASE 2: CEREBRO APAGADO
        self.dense_in = tf.keras.layers.Dense(context_dim, activation=tf.nn.leaky_relu)
        self.gru = tf.keras.layers.GRU(gru_units, return_sequences=True)
        self.dense_out = tf.keras.layers.Dense(context_dim)

    def call(self, inputs):
        x = self.dense_in(inputs)
        x = self.gru(x)
        return self.dense_out(x)

class Detuner(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(1, kernel_initializer='zeros')
        self.trainable = True # FASE 2: FÍSICA ENCENDIDA

    def build(self, input_shape):
        self.static_detune = self.add_weight(
            name='static_detune',
            shape=(1,),
            initializer='zeros',
            trainable=True
        )
        super().build(input_shape)

    def call(self, pitch_midi):
        x = pitch_midi / 127.0
        dynamic_detune = tf.nn.tanh(self.dense(x))
        static_detune = tf.nn.tanh(self.static_detune)
        return dynamic_detune + static_detune

class TrainableReverb(tf.keras.Model): 
    def __init__(self, ir_length=24000, lambda_ir=1e-2, **kwargs): 
        super().__init__(**kwargs)
        self.ir_length = ir_length
        self.lambda_ir = lambda_ir
        self.trainable = False # FASE 2: REVERB APAGADA

    def build(self, input_shape):
        np_ir_init = np.random.normal(size=self.ir_length) * np.exp(-np.linspace(0.0, 5.0, self.ir_length))
        self.ir = self.add_weight(
            name="impulse_response",
            shape=[self.ir_length],
            initializer=tf.constant_initializer(np_ir_init),
            trainable=False
        )
        super().build(input_shape)

    def call(self, audio):
        n_fft = tf.pow(2, tf.cast(tf.math.ceil(tf.math.log(tf.cast(tf.shape(audio)[1] + self.ir_length, tf.float32)) / tf.math.log(2.0)), tf.int32))
        audio_fft = tf.signal.rfft(audio, [n_fft])
        ir_fft = tf.signal.rfft(self.ir, [n_fft])
        reverb_fft = audio_fft * ir_fft
        out = tf.signal.irfft(reverb_fft)[:, :tf.shape(audio)[1]]
        self.add_loss(self.lambda_ir * tf.reduce_sum(tf.abs(self.ir)))
        return out

class InharmonicityModel(tf.keras.layers.Layer):
    def __init__(self, l1_penalty=0.1, **kwargs):
        super().__init__(**kwargs)
        self.trainable = True # FASE 2: FÍSICA ENCENDIDA
        self.l1_penalty = l1_penalty
        
        # Valores teóricos para la pérdida L1
        self.init_alpha_B = -0.0847
        self.init_beta_B = -5.82
        self.init_alpha_T = 0.0926
        self.init_beta_T = -13.64

    def build(self, input_shape):
        self.alpha_B = self.add_weight(name='alpha_B', shape=(), initializer=tf.constant_initializer(self.init_alpha_B), trainable=True)
        self.beta_B = self.add_weight(name='beta_B', shape=(), initializer=tf.constant_initializer(self.init_beta_B), trainable=True)
        self.alpha_T = self.add_weight(name='alpha_T', shape=(), initializer=tf.constant_initializer(self.init_alpha_T), trainable=True)
        self.beta_T = self.add_weight(name='beta_T', shape=(), initializer=tf.constant_initializer(self.init_beta_T), trainable=True)
        super().build(input_shape)

    def call(self, pitch_midi):
        # FASE 2: PERDIDA L1 (Evita que la afinación se aleje de la física real)
        loss_l1 = self.l1_penalty * (
            tf.abs(self.alpha_B - self.init_alpha_B) + 
            tf.abs(self.beta_B - self.init_beta_B) + 
            tf.abs(self.alpha_T - self.init_alpha_T) + 
            tf.abs(self.beta_T - self.init_beta_T)
        )
        self.add_loss(loss_l1)

        pitch_safe = tf.where(pitch_midi > 0, pitch_midi, 1.0)
        term_T = tf.exp(self.alpha_T * pitch_safe + self.beta_T)
        term_B = tf.exp(self.alpha_B * pitch_safe + self.beta_B)
        return term_T + term_B

class DDSPCore(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        cfg = config['model']
        self.n_harmonics = cfg['n_harmonics']
        
        # FASE 2: CEREBRO APAGADO CAPA POR CAPA
        self.dense_in = tf.keras.layers.Dense(cfg['hidden_size'], trainable=False)
        self.leaky_in = tf.keras.layers.LeakyReLU(negative_slope=0.2)
        self.gru = tf.keras.layers.GRU(cfg['gru_units'], return_sequences=True, trainable=False)
        self.dense_hidden = tf.keras.layers.Dense(cfg['dense_output_size'], trainable=False)
        self.leaky_hidden = tf.keras.layers.LeakyReLU(negative_slope=0.2)
        
        # LA ÚNICA QUE QUEDA ENCENDIDA
        self.inharmonicity = InharmonicityModel()

        self.amp_out = tf.keras.layers.Dense(1, bias_initializer=tf.keras.initializers.Constant(-5.0), trainable=False)
        self.harm_out = tf.keras.layers.Dense(self.n_harmonics, bias_initializer='zeros', trainable=False)
        self.noise_out = tf.keras.layers.Dense(cfg['n_noise_filters'], bias_initializer=tf.keras.initializers.Constant(-5.0), trainable=False)

    def call(self, core_input, pitch_midi, training=False):
        x = self.leaky_hidden(self.dense_hidden(self.gru(self.leaky_in(self.dense_in(core_input)), training=training)))
        
        amp = get_modified_sigmoid(self.amp_out(x))
        harm_dist = get_modified_sigmoid(self.harm_out(x))
        noise_mags = get_modified_sigmoid(self.noise_out(x))
        
        harm_dist /= tf.reduce_sum(harm_dist, axis=-1, keepdims=True)
        
        f0_hz = 440.0 * tf.math.pow(2.0, (pitch_midi - 69.0) / 12.0)
        f0_hz = tf.where(pitch_midi > 0, f0_hz, 0.0)
        
        B_factor = self.inharmonicity(pitch_midi)
        k = tf.reshape(tf.range(1, self.n_harmonics + 1, dtype=tf.float32), [1, 1, self.n_harmonics])
        stretch_factor = tf.sqrt(1.0 + B_factor * tf.square(k))
        
        f_k = f0_hz * k * stretch_factor
        
        return {
            'amplitude': amp,
            'harmonics': harm_dist,
            'noise_magnitudes': noise_mags,
            'f_k': f_k
        }

# ==============================================================================
# 4. POLYPHONIC WRAPPER
# ==============================================================================
class PolyphonicDDSPPiano(tf.keras.Model):
    def __init__(self, config, n_voices=1, per_replica_batch_size=6):
        super().__init__()
        self.n_voices = n_voices
        self.context_net = ContextNetwork()
        self.core = DDSPCore(config)
        self.detuner = Detuner()
        
        self.harmonic_synth = HarmonicSynthesizer()
        self.noise_synth = FilteredNoiseSynthesizer(per_replica_batch_size=per_replica_batch_size)
        self.reverb = TrainableReverb()

    def call(self, inputs, training=False):
        pitches = inputs['pitches']       
        velocities = inputs['velocities'] 
        pedal = inputs['pedal']           

        global_vel = tf.reduce_sum(velocities, axis=1) 
        pitches_norm = pitches / 127.0
        active_pitches = tf.where(velocities > 0, pitches_norm, 0.0)
        global_pitch = tf.reduce_sum(active_pitches, axis=1) / (tf.reduce_sum(tf.cast(velocities > 0, tf.float32), axis=1) + 1e-7)
        
        context_input = tf.concat([pedal, global_vel, global_pitch], axis=-1)
        c_t = self.context_net(context_input) 

        audio_sum = 0.0
        
        for i in range(self.n_voices):
            v_pitch = pitches[:, i, :, :] 
            v_vel = velocities[:, i, :, :]
            v_pitch_norm = v_pitch / 127.0
            
            core_input = tf.concat([v_pitch_norm, v_vel, c_t], axis=-1)
            params = self.core(core_input, v_pitch, training=training) 
            
            delta_f = self.detuner(v_pitch)
            f_k_1 = params['f_k']
            f_k_2 = f_k_1 * tf.pow(2.0, (delta_f / 12.0))
            
            audio_voice = self.harmonic_synth(f_k_1, params['amplitude'], params['harmonics'])
            audio_voice += self.harmonic_synth(f_k_2, params['amplitude'], params['harmonics'])
            audio_voice += self.noise_synth(params['noise_magnitudes'])
            
            audio_sum += audio_voice

        return self.reverb(audio_sum)

# ==============================================================================
# 5. FUNCIÓN DE PÉRDIDA
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
# 6. BUCLE DE ENTRENAMIENTO DISTRIBUIDO
# ==============================================================================
def parse_tfrecord(example_proto):
    desc = {
        'pitch': tf.io.FixedLenFeature([6000], tf.float32), 
        'velocity': tf.io.FixedLenFeature([6000], tf.float32), 
        'pedal': tf.io.FixedLenFeature([750], tf.float32),
        'audio': tf.io.FixedLenFeature([48000], tf.float32)
    }
    parsed = tf.io.parse_single_example(example_proto, desc)
    pitch = tf.reshape(parsed['pitch'], [8, 750, 1])
    velocity = tf.reshape(parsed['velocity'], [8, 750, 1])
    pedal = tf.expand_dims(parsed['pedal'], -1) 
    return pitch, velocity, pedal, parsed['audio']

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
        model = PolyphonicDDSPPiano(CONFIG, n_voices=8, per_replica_batch_size=per_replica_batch_size)
        loss_fn = MultiResolutionSpectralLoss(CONFIG)
        optimizer = tf.keras.optimizers.Adam(learning_rate=CONFIG['training']['learning_rate'])

        def compute_loss(labels, predictions):
            return tf.nn.compute_average_loss(loss_fn(labels, predictions), global_batch_size=global_batch_size)

        print("\n[INFO] Construyendo variables (Dummy Pass)...")
        dummy_pitches = tf.zeros((1, 8, 750, 1), dtype=tf.float32)
        dummy_velocities = tf.zeros((1, 8, 750, 1), dtype=tf.float32)
        dummy_pedal = tf.zeros((1, 750, 1), dtype=tf.float32) 
        _ = model({'pitches': dummy_pitches, 'velocities': dummy_velocities, 'pedal': dummy_pedal}) 
        
        optimizer.build(model.trainable_variables) 
        print("[OK] Red y Optimizador listos.")

        if RESUME_TRAINING:
            print(f"\n[INFO] FASE 2: Cargando pesos expertos de Fase 1 desde: {PRETRAINED_DIR}")
            try:
                model.core.load_weights(os.path.join(PRETRAINED_DIR, 'core.weights.h5'))
                model.context_net.load_weights(os.path.join(PRETRAINED_DIR, 'context.weights.h5'))
                model.detuner.load_weights(os.path.join(PRETRAINED_DIR, 'detuner.weights.h5'))
                model.reverb.load_weights(os.path.join(PRETRAINED_DIR, 'reverb.weights.h5'))
                print("[ÉXITO] ¡Pesos cargados! Listo para afinar la física.")
            except Exception as e:
                # En la Fase 2, si falla la carga, paramos el programa. Empezar de cero con el cerebro congelado sería fatal.
                raise RuntimeError(f"¡ERROR FATAL! No se encontraron los pesos en {PRETRAINED_DIR}. Verifica la ruta del dataset de pesos en Kaggle. Detalle del error: {e}")

    with strategy.scope():
        @tf.function
        def train_step(pitch, velocity, pedal, real_audio): 
            inputs = {'pitches': pitch, 'velocities': velocity, 'pedal': pedal}
            with tf.GradientTape() as tape:
                audio_pred = model(inputs, training=True)
                base_loss = compute_loss(real_audio, audio_pred)
                total_loss = base_loss + tf.reduce_sum(model.losses) / global_batch_size
            gradients, _ = tf.clip_by_global_norm(tape.gradient(total_loss, model.trainable_variables), 3.0)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return total_loss

        @tf.function
        def distributed_train_step(inputs):
            return strategy.reduce(tf.distribute.ReduceOp.SUM, strategy.run(train_step, args=inputs), axis=None)

        @tf.function
        def val_step(pitch, velocity, pedal, real_audio):
            inputs = {'pitches': pitch, 'velocities': velocity, 'pedal': pedal}
            audio_pred = model(inputs, training=False)
            base_loss = compute_loss(real_audio, audio_pred)
            total_loss = base_loss + tf.reduce_sum(model.losses) / global_batch_size
            return total_loss

        @tf.function
        def distributed_val_step(inputs):
            return strategy.reduce(tf.distribute.ReduceOp.SUM, strategy.run(val_step, args=inputs), axis=None)

    # FASE 2: Ajuste fino muy corto
    epochs, steps_per_epoch = 12, 500 
    historial_train_loss = []
    historial_val_loss = []
    best_val_loss = float('inf') 

    print(f"\n[INFO] Iniciando FASE 2 (Micro-Afinación) | LR: {CONFIG['training']['learning_rate']} | Batch: {global_batch_size}")
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
            
            pbar.set_postfix({"Train Loss": f"{avg_train_loss:.4f}", "Val Loss": f"{avg_val_loss:.4f}"})
        
        # Guardado modular
        if avg_val_loss < best_val_loss and avg_val_loss > 0:
            best_val_loss = avg_val_loss
            model.core.save_weights('/kaggle/working/checkpoints/core.weights.h5')
            model.context_net.save_weights('/kaggle/working/checkpoints/context.weights.h5')
            model.detuner.save_weights('/kaggle/working/checkpoints/detuner.weights.h5')
            model.reverb.save_weights('/kaggle/working/checkpoints/reverb.weights.h5')
            print(f"  -> [NUEVO RÉCORD FASE 2] Módulos afinados guardados. Val Loss: {best_val_loss:.4f}")

    # Guardamos el ZIP de la Fase 2
    import shutil
    shutil.make_archive('/kaggle/working/pesos_ddsp_fase2', 'zip', '/kaggle/working/checkpoints')
    print("\n[INFO] ENTRENAMIENTO FASE 2 FINALIZADO. Pesos afinados comprimidos en 'pesos_ddsp_fase2.zip'.")

run_training()