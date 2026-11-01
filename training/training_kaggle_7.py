# ==============================================================================
# DDSP-PIANO  -  training_kaggle_7.py   (sustituye a Fase1 + Fase2)
#
# Que cambia respecto a training_kaggle_6_Fase1/2:
#   1. PEDAL REAL: el dataset (data/prepare_kaggle_dataset.py) integra el sustain en las notas,
#      como el paper de Renault: si sueltas la tecla con el pedal pisado, la voz sigue activa
#      hasta que se levanta el pedal. Antes el modelo solo veia el pedal como senal de contexto.
#   2. 24 voces (antes 8) y SIN voice stealing "a ciegas": los trozos con notas perdidas se descartan.
#   3. Recortes aleatorios al vuelo (4 s = 1 s de CALENTAMIENTO sin loss + 3 s con loss).
#      El calentamiento deja que la GRU "arranque" con las notas que ya estaban sonando
#      (soluciona el "impacto inicial" de los chunks y la deriva en notas largas).
#   4. Las voces van en la dimension de batch (1 sola llamada a la GRU) y la sintesis armonica
#      usa tf.recompute_grad => muchisima menos VRAM que el bucle anterior.
#   5. 1 cuerda por nota (el detuner de la Fase 2 quedo en ~0, el paper tampoco lo necesita).
#      Se puede poner N_STRINGS=2 con un desafinado FIJO en STRING_DETUNE_CENTS.
#   6. Ruido enmascarado en voces inactivas (pitch=0): el plugin las salta, ahora es identico.
#   7. Una sola fase. No hace falta Fase 2 (inharmonicidad fija a los valores teoricos).
#   8. Guarda 'best' y 'last', un training_meta.json (lo lee el exportador) y para solo
#      cuando se acaba el tiempo (MAX_HOURS) para no perder el ZIP en Kaggle.
#
# COMO USARLO EN KAGGLE:  ver training/KAGGLE_GUIA.md
# ==============================================================================
import os
import json
import time
import shutil
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# ==============================================================================
# 1. CONFIGURACION
# ==============================================================================
CONFIG = {
    'audio': {'sample_rate': 16000, 'frame_rate': 250},
    'model': {'n_harmonics': 96, 'n_noise_filters': 64, 'hidden_size': 128,
              'gru_units': 192, 'dense_output_size': 192},
    'loss': {'fft_sizes': [2048, 1024, 512, 256, 128, 64], 'hop_ratio': 0.25},
    'training': {'learning_rate': 1e-3, 'learning_rate_warmstart': 4e-4, 'global_batch_size': 12,
                 'steps_per_epoch': 250, 'max_epochs': 300, 'patience': 8, 'min_lr': 1e-6},
}

WARMUP_SEC = 1.0          # segundos al inicio de cada recorte SIN loss (la GRU se "calienta")
TRAIN_SEC = 3.0           # segundos CON loss
N_STRINGS = 1             # 1 cuerda (recomendado). Con 2: se usa STRING_DETUNE_CENTS
STRING_DETUNE_CENTS = [0.0, 3.0]
USE_RECOMPUTE = True      # ahorra VRAM. Si da error de tf.recompute_grad, ponlo en False y baja el batch
MAX_HOURS = 11.0          # Kaggle corta a las 12 h: el script se para solo antes y deja el ZIP listo
VAL_CROPS = 240           # recortes fijos de validacion (multiplo del batch global)

# Reanudar un entrenamiento anterior: pon aqui la carpeta con core.weights.h5 de tu run previo
# p.ej. '/kaggle/input/ddsp7-run1/checkpoints_last'. Si es None, arranca desde 'weights_init'
# (tus pesos de la Fase 1 anterior) si existen, y si no, desde cero.
RESUME_FROM = None
USE_WARM_START = True

LOCAL = not os.path.exists('/kaggle')
SMOKE = LOCAL or os.environ.get('DDSP_SMOKE') == '1'   # prueba rapida en CPU: 3 pasos
OUT_DIR = './out_local' if LOCAL else '/kaggle/working'
CKPT_BEST = os.path.join(OUT_DIR, 'checkpoints')
CKPT_LAST = os.path.join(OUT_DIR, 'checkpoints_last')

SR = CONFIG['audio']['sample_rate']
FRAME_RATE = CONFIG['audio']['frame_rate']
HOP = SR // FRAME_RATE
WARM_FRAMES = int(WARMUP_SEC * FRAME_RATE)
TRAIN_FRAMES = int(TRAIN_SEC * FRAME_RATE)
CROP_FRAMES = WARM_FRAMES + TRAIN_FRAMES
N_SAMPLES = CROP_FRAMES * HOP
WARM_SAMPLES = WARM_FRAMES * HOP

# ==============================================================================
# 2. LOCALIZAR DATOS Y PESOS
# ==============================================================================
def find_data_root():
    env = os.environ.get('DDSP_DATA_DIR')
    if env and os.path.exists(os.path.join(env, 'manifest.json')):
        return env
    for dirname, _, files in os.walk('/kaggle/input'):
        if 'manifest.json' in files:
            return dirname
    raise FileNotFoundError('No encuentro manifest.json. Anade el dataset de datos al notebook '
                            '(o define DDSP_DATA_DIR en local).')

def find_init_weights(data_root):
    cand = os.path.join(data_root, 'weights_init')
    if os.path.exists(os.path.join(cand, 'core.weights.h5')):
        return cand
    return None

DATA_ROOT = find_data_root()
with open(os.path.join(DATA_ROOT, 'manifest.json')) as f:
    MANIFEST = json.load(f)
N_VOICES = MANIFEST['n_voices']
print(f"[OK] Datos: {DATA_ROOT}  (voces={N_VOICES}, max_sustain={MANIFEST['max_sustain_sec']} s)")

# ==============================================================================
# 3. DATOS: recortes aleatorios al vuelo
# ==============================================================================
class TrackBank:
    def __init__(self, split):
        names = MANIFEST['splits'][split]
        if SMOKE:
            names = names[:2]
        self.tracks = []
        for n in tqdm(names, desc=f'Cargando {split}'):
            z = np.load(os.path.join(DATA_ROOT, split, n))
            tr = {k: z[k] for k in z.files}
            tr['T'] = tr['pitch'].shape[1]
            if tr['T'] > CROP_FRAMES + 10:
                self.tracks.append(tr)
        self.weights = np.array([t['T'] for t in self.tracks], dtype=np.float64)
        self.weights /= self.weights.sum()

    @staticmethod
    def make_crop(tr, s):
        e = s + CROP_FRAMES
        if tr['dropped'][s:e].any():
            return None
        if tr['pitch'][:, s + WARM_FRAMES:e].max() == 0:
            return None
        pitch = tr['pitch'][:, s:e].astype(np.float32)[..., None]
        vel = tr['vel'][:, s:e].astype(np.float32)[..., None] / 127.0
        pedal = tr['pedal'][s:e].astype(np.float32)[:, None] / 127.0
        audio = tr['audio'][s * HOP:e * HOP].astype(np.float32) / 32767.0
        return pitch, vel, pedal, audio

    def train_generator(self):
        rng = np.random.default_rng()
        while True:
            tr = self.tracks[rng.choice(len(self.tracks), p=self.weights)]
            s = int(rng.integers(0, tr['T'] - CROP_FRAMES))
            c = self.make_crop(tr, s)
            if c is not None:
                yield c

    def fixed_crops(self, n, seed=0):
        crops = []
        for tr in self.tracks:
            for s in range(0, tr['T'] - CROP_FRAMES, TRAIN_FRAMES):
                c = self.make_crop(tr, s)
                if c is not None:
                    crops.append(c)
        rng = np.random.default_rng(seed)
        rng.shuffle(crops)
        return crops[:n]

SIGNATURE = (
    tf.TensorSpec([N_VOICES, CROP_FRAMES, 1], tf.float32),
    tf.TensorSpec([N_VOICES, CROP_FRAMES, 1], tf.float32),
    tf.TensorSpec([CROP_FRAMES, 1], tf.float32),
    tf.TensorSpec([N_SAMPLES], tf.float32),
)

# ==============================================================================
# 4. SINTETIZADORES
# ==============================================================================
def get_modified_sigmoid(x):
    return 2.0 * tf.math.pow(tf.math.sigmoid(x), tf.math.log(10.0)) + 1e-7

def upsample_controls(controls, n_samples):
    controls_4d = tf.expand_dims(controls, axis=2)
    upsampled_4d = tf.image.resize(controls_4d, [n_samples, 1], method=tf.image.ResizeMethod.BILINEAR)
    return tf.squeeze(upsampled_4d, axis=2)

def harmonic_voice(f_k, amplitude, harmonics, sr=SR):
    """Una voz: suma de N_STRINGS cuerdas. f_k:[B,T,K] amplitude:[B,T,1] harmonics:[B,T,K] -> [B,N]"""
    amp_up = upsample_controls(amplitude, N_SAMPLES)
    harm_up = upsample_controls(harmonics, N_SAMPLES)
    audio = 0.0
    for s in range(N_STRINGS):
        f = f_k * (2.0 ** (STRING_DETUNE_CENTS[s] / 1200.0))
        f_up = upsample_controls(f, N_SAMPLES)
        mask = tf.cast(f_up < (sr / 2.0), tf.float32)
        phases = tf.math.cumsum(f_up * (2.0 * np.pi / sr), axis=1)
        audio += tf.reduce_sum(tf.math.sin(phases) * harm_up * mask, axis=-1)
    return audio * tf.squeeze(amp_up, axis=-1)

class FilteredNoiseSynthesizer:
    def __init__(self, sample_rate=SR, frame_rate=FRAME_RATE):
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
# 5. MODULOS (misma estructura que Fase 1 => los .h5 antiguos se pueden cargar)
# ==============================================================================
class ContextNetwork(tf.keras.Model):
    def __init__(self, context_dim=32, gru_units=64):
        super().__init__()
        self.dense_in = tf.keras.layers.Dense(context_dim, activation=tf.nn.leaky_relu)
        self.gru = tf.keras.layers.GRU(gru_units, return_sequences=True)
        self.dense_out = tf.keras.layers.Dense(context_dim)

    def call(self, inputs):
        return self.dense_out(self.gru(self.dense_in(inputs)))

class TrainableReverb(tf.keras.Model):
    def __init__(self, ir_length=24000, lambda_ir=1e-2, **kwargs):
        super().__init__(**kwargs)
        self.ir_length = ir_length
        self.lambda_ir = lambda_ir

    def build(self, input_shape):
        np_ir_init = np.random.normal(size=self.ir_length) * np.exp(-np.linspace(0.0, 5.0, self.ir_length))
        self.ir = self.add_weight(name="impulse_response", shape=[self.ir_length],
                                  initializer=tf.constant_initializer(np_ir_init), trainable=True)
        super().build(input_shape)

    def call(self, audio):
        n_fft = tf.pow(2, tf.cast(tf.math.ceil(tf.math.log(tf.cast(tf.shape(audio)[1] + self.ir_length, tf.float32)) / tf.math.log(2.0)), tf.int32))
        audio_fft = tf.signal.rfft(audio, [n_fft])
        ir_fft = tf.signal.rfft(self.ir, [n_fft])
        out = tf.signal.irfft(audio_fft * ir_fft)[:, :tf.shape(audio)[1]]
        self.add_loss(self.lambda_ir * tf.reduce_sum(tf.abs(self.ir)))
        return out

class InharmonicityModel(tf.keras.layers.Layer):
    """Valores teoricos de Rigaud et al. FIJOS (la Fase 2 no los movio y el paper tampoco)."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.trainable = False

    def build(self, input_shape):
        self.alpha_B = self.add_weight(name='alpha_B', shape=(), initializer=tf.constant_initializer(-0.0847), trainable=False)
        self.beta_B = self.add_weight(name='beta_B', shape=(), initializer=tf.constant_initializer(-5.82), trainable=False)
        self.alpha_T = self.add_weight(name='alpha_T', shape=(), initializer=tf.constant_initializer(0.0926), trainable=False)
        self.beta_T = self.add_weight(name='beta_T', shape=(), initializer=tf.constant_initializer(-13.64), trainable=False)
        super().build(input_shape)

    def call(self, pitch_midi):
        pitch_safe = tf.where(pitch_midi > 0, pitch_midi, 1.0)
        return tf.exp(self.alpha_T * pitch_safe + self.beta_T) + tf.exp(self.alpha_B * pitch_safe + self.beta_B)

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
        self.inharmonicity = InharmonicityModel()
        self.amp_out = tf.keras.layers.Dense(1, bias_initializer=tf.keras.initializers.Constant(-5.0))
        self.harm_out = tf.keras.layers.Dense(self.n_harmonics, bias_initializer='zeros')
        self.noise_out = tf.keras.layers.Dense(cfg['n_noise_filters'], bias_initializer=tf.keras.initializers.Constant(-5.0))

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
        f_k = f0_hz * k * tf.sqrt(1.0 + B_factor * tf.square(k))
        return {'amplitude': amp, 'harmonics': harm_dist, 'noise_magnitudes': noise_mags, 'f_k': f_k}

# ==============================================================================
# 6. ORQUESTADOR POLIFONICO (voces en la dimension de batch)
# ==============================================================================
class PolyphonicDDSPPiano(tf.keras.Model):
    def __init__(self, config, n_voices):
        super().__init__()
        self.n_voices = n_voices
        self.n_harmonics = config['model']['n_harmonics']
        self.context_net = ContextNetwork()
        self.core = DDSPCore(config)
        self.noise_synth = FilteredNoiseSynthesizer()
        self.reverb = TrainableReverb()
        self._synth = tf.recompute_grad(harmonic_voice) if USE_RECOMPUTE else harmonic_voice

    def call(self, inputs, training=False):
        pitches = inputs['pitches']        # [B,P,T,1]  (0..127, 0 = voz libre)
        velocities = inputs['velocities']  # [B,P,T,1]  (0..1, distinta de 0 mientras la nota esta activa)
        pedal = inputs['pedal']            # [B,T,1]
        P, K = self.n_voices, self.n_harmonics
        T = pitches.shape[2]

        global_vel = tf.reduce_sum(velocities, axis=1)
        active = tf.cast(velocities > 0, tf.float32)
        global_pitch = tf.reduce_sum(pitches / 127.0 * active, axis=1) / (tf.reduce_sum(active, axis=1) + 1e-7)
        c_t = self.context_net(tf.concat([pedal, global_vel, global_pitch], axis=-1))   # [B,T,32]
        c_rep = tf.tile(c_t[:, None], [1, P, 1, 1])                                      # [B,P,T,32]

        core_in = tf.reshape(tf.concat([pitches / 127.0, velocities, c_rep], axis=-1), [-1, T, 34])
        pitch_flat = tf.reshape(pitches, [-1, T, 1])
        params = self.core(core_in, pitch_flat, training=training)

        amp = tf.reshape(params['amplitude'], [-1, P, T, 1])
        harm = tf.reshape(params['harmonics'], [-1, P, T, K])
        f_k = tf.reshape(params['f_k'], [-1, P, T, K])
        noise_mags = params['noise_magnitudes'] * tf.cast(pitch_flat > 0, tf.float32)   # voces libres: sin ruido

        audio_sum = 0.0
        for p in range(P):
            audio_sum += self._synth(f_k[:, p], amp[:, p], harm[:, p])
        noise_audio = tf.reshape(self.noise_synth(noise_mags), [-1, P, N_SAMPLES])
        audio_sum += tf.reduce_sum(noise_audio, axis=1)
        return self.reverb(audio_sum)

# ==============================================================================
# 7. LOSS
# ==============================================================================
class MultiResolutionSpectralLoss(tf.keras.losses.Loss):
    def __init__(self, config, name="multi_resolution_spectral_loss"):
        super().__init__(name=name, reduction=tf.keras.losses.Reduction.NONE)
        self.fft_sizes = config['loss']['fft_sizes']
        self.hop_ratio = config['loss']['hop_ratio']

    def spectral_loss(self, y_true, y_pred, fft_size):
        hop_size = int(fft_size * self.hop_ratio)
        mag_true = tf.abs(tf.signal.stft(y_true, frame_length=fft_size, frame_step=hop_size))
        mag_pred = tf.abs(tf.signal.stft(y_pred, frame_length=fft_size, frame_step=hop_size))
        lin_loss = tf.reduce_mean(tf.abs(mag_true - mag_pred), axis=[1, 2])
        log_loss = tf.reduce_mean(tf.abs(tf.math.log(mag_true + 1e-7) - tf.math.log(mag_pred + 1e-7)), axis=[1, 2])
        return lin_loss + log_loss

    def call(self, y_true, y_pred):
        total = 0.0
        for size in self.fft_sizes:
            total += self.spectral_loss(y_true, y_pred, size)
        return total

# ==============================================================================
# 8. ENTRENAMIENTO
# ==============================================================================
def save_modules(model, folder):
    os.makedirs(folder, exist_ok=True)
    model.core.save_weights(os.path.join(folder, 'core.weights.h5'))
    model.context_net.save_weights(os.path.join(folder, 'context.weights.h5'))
    model.reverb.save_weights(os.path.join(folder, 'reverb.weights.h5'))

def load_modules(model, folder):
    model.core.load_weights(os.path.join(folder, 'core.weights.h5'))
    model.context_net.load_weights(os.path.join(folder, 'context.weights.h5'))
    model.reverb.load_weights(os.path.join(folder, 'reverb.weights.h5'))

def run_training():
    t_start = time.time()
    os.makedirs(CKPT_BEST, exist_ok=True)
    cfg_t = CONFIG['training']
    strategy = tf.distribute.MirroredStrategy()
    n_rep = strategy.num_replicas_in_sync
    global_batch = 2 * n_rep if SMOKE else cfg_t['global_batch_size']
    steps_per_epoch = 3 if SMOKE else cfg_t['steps_per_epoch']
    max_epochs = 1 if SMOKE else cfg_t['max_epochs']
    print(f"[INFO] Replicas={n_rep}  batch global={global_batch}  recorte={CROP_FRAMES / FRAME_RATE:.1f}s "
          f"(warmup {WARMUP_SEC}s + loss {TRAIN_SEC}s)  voces={N_VOICES}  cuerdas={N_STRINGS}")

    train_bank, val_bank = TrackBank('train'), TrackBank('validation')
    val_crops = val_bank.fixed_crops(global_batch * 2 if SMOKE else VAL_CROPS)
    val_crops = val_crops[: (len(val_crops) // global_batch) * global_batch]
    print(f"[OK] Train: {len(train_bank.tracks)} piezas | Validacion: {len(val_crops)} recortes fijos")

    train_ds = tf.data.Dataset.from_generator(train_bank.train_generator, output_signature=SIGNATURE)
    train_ds = train_ds.batch(global_batch, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_generator(lambda: iter(val_crops), output_signature=SIGNATURE)
    val_ds = val_ds.batch(global_batch, drop_remainder=True)
    dist_train = strategy.experimental_distribute_dataset(train_ds)
    dist_val = strategy.experimental_distribute_dataset(val_ds)

    with strategy.scope():
        model = PolyphonicDDSPPiano(CONFIG, n_voices=N_VOICES)
        loss_fn = MultiResolutionSpectralLoss(CONFIG)
        _ = model({'pitches': tf.zeros((1, N_VOICES, CROP_FRAMES, 1)),
                   'velocities': tf.zeros((1, N_VOICES, CROP_FRAMES, 1)),
                   'pedal': tf.zeros((1, CROP_FRAMES, 1))})

        warm = False
        init_dir = RESUME_FROM or (find_init_weights(DATA_ROOT) if USE_WARM_START else None)
        if init_dir and os.path.exists(os.path.join(init_dir, 'core.weights.h5')):
            try:
                load_modules(model, init_dir)
                warm = True
                print(f"[EXITO] Pesos iniciales cargados de: {init_dir}")
            except Exception as e:
                print(f"[AVISO] No se pudieron cargar los pesos de {init_dir}: {e}\n        Se entrena DESDE CERO.")
        else:
            print("[INFO] Sin pesos iniciales: se entrena desde cero.")

        lr = cfg_t['learning_rate_warmstart'] if warm else cfg_t['learning_rate']
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        optimizer.build(model.trainable_variables)

        def compute_loss(labels, preds):
            per_example = loss_fn(labels[:, WARM_SAMPLES:], preds[:, WARM_SAMPLES:])
            return tf.nn.compute_average_loss(per_example, global_batch_size=global_batch)

        def make_inputs(pitch, velocity, pedal):
            return {'pitches': pitch, 'velocities': velocity, 'pedal': pedal}

        @tf.function
        def train_step(pitch, velocity, pedal, real_audio):
            with tf.GradientTape() as tape:
                pred = model(make_inputs(pitch, velocity, pedal), training=True)
                total = compute_loss(real_audio, pred) + tf.reduce_sum(model.losses) / global_batch
            grads, _ = tf.clip_by_global_norm(tape.gradient(total, model.trainable_variables), 3.0)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return total

        @tf.function
        def val_step(pitch, velocity, pedal, real_audio):
            pred = model(make_inputs(pitch, velocity, pedal), training=False)
            return compute_loss(real_audio, pred) + tf.reduce_sum(model.losses) / global_batch

        @tf.function
        def dist_train_step(batch):
            return strategy.reduce(tf.distribute.ReduceOp.SUM, strategy.run(train_step, args=batch), axis=None)

        @tf.function
        def dist_val_step(batch):
            return strategy.reduce(tf.distribute.ReduceOp.SUM, strategy.run(val_step, args=batch), axis=None)

    meta = {
        'semantics_version': 2,              # 2 = el pedal alarga las notas (el plugin debe imitarlo)
        'sample_rate': SR, 'frame_rate': FRAME_RATE, 'n_voices_train': N_VOICES,
        'n_strings': N_STRINGS, 'string_detune_cents': STRING_DETUNE_CENTS[:N_STRINGS],
        'release_sec': MANIFEST['t_release_sec'], 'max_sustain_sec': MANIFEST['max_sustain_sec'],
        'fade_frames': MANIFEST['fade_frames'], 'pedal_threshold': 0.5,
        'n_harmonics': CONFIG['model']['n_harmonics'], 'n_noise_filters': CONFIG['model']['n_noise_filters'],
        'warm_started': warm,
    }

    it = iter(dist_train)
    best_val, wait, lr_now = float('inf'), 0, lr
    hist_tr, hist_va = [], []
    for epoch in range(max_epochs):
        tot = 0.0
        with tqdm(total=steps_per_epoch, desc=f"Epoca {epoch + 1:03d}", unit="batch") as pbar:
            for _ in range(steps_per_epoch):
                tot += dist_train_step(next(it))
                pbar.update(1)
            tr_loss = float((tot / steps_per_epoch).numpy())
            va_tot, nb = 0.0, 0
            for batch in dist_val:
                va_tot += dist_val_step(batch)
                nb += 1
            va_loss = float((va_tot / max(nb, 1)).numpy())
            pbar.set_postfix({"Train": f"{tr_loss:.3f}", "Val": f"{va_loss:.3f}", "LR": f"{lr_now:.1e}"})
        hist_tr.append(tr_loss); hist_va.append(va_loss)

        save_modules(model, CKPT_LAST)
        if va_loss < best_val:
            best_val, wait = va_loss, 0
            save_modules(model, CKPT_BEST)
            meta['best_val_loss'] = best_val
            meta['epoch'] = epoch + 1
            with open(os.path.join(CKPT_BEST, 'training_meta.json'), 'w') as f:
                json.dump(meta, f, indent=2)
            shutil.copy(os.path.join(CKPT_BEST, 'training_meta.json'), os.path.join(CKPT_LAST, 'training_meta.json'))
            print(f"  -> [RECORD] Val {best_val:.4f} guardado en checkpoints/")
        else:
            wait += 1
            if wait >= cfg_t['patience']:
                new_lr = max(lr_now * 0.5, cfg_t['min_lr'])
                if new_lr < lr_now:
                    optimizer.learning_rate.assign(new_lr)
                    lr_now = new_lr
                    print(f"  [LR] Plateau: LR -> {new_lr:.1e}")
                wait = 0

        elapsed_h = (time.time() - t_start) / 3600
        if elapsed_h > MAX_HOURS:
            print(f"[INFO] Limite de tiempo ({MAX_HOURS} h) alcanzado: se para y se empaqueta.")
            break

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(hist_tr, label='Train'); plt.plot(hist_va, label='Validation')
        plt.grid(True, alpha=.5); plt.legend(); plt.xlabel('Epoca'); plt.ylabel('Loss')
        plt.savefig(os.path.join(OUT_DIR, 'loss_curve.png'), dpi=150); plt.close()
    except Exception as e:  # noqa
        print('[AVISO] No se pudo dibujar la curva:', e)

    shutil.make_archive(os.path.join(OUT_DIR, 'pesos_ddsp7'), 'zip', CKPT_BEST)
    shutil.make_archive(os.path.join(OUT_DIR, 'pesos_ddsp7_last'), 'zip', CKPT_LAST)
    print(f"\n[FIN] best val={best_val:.4f}. Descarga pesos_ddsp7.zip (best) y pesos_ddsp7_last.zip (para reanudar).")

run_training()
