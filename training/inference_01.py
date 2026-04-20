import tensorflow as tf
import numpy as np
import pretty_midi
import scipy.io.wavfile
import os

# ==============================================================================
# 1. CONFIGURACIÓN (Debe coincidir con el entrenamiento)
# ==============================================================================
CONFIG = {
    'audio': {'sample_rate': 16000, 'frame_rate': 250},
    'model': {'n_harmonics': 96, 'n_noise_filters': 64, 'hidden_size': 128, 'gru_units': 192, 'dense_output_size': 192, 'total_params_output': 161},
}

# ==============================================================================
# 2. SINTETIZADORES DINÁMICOS (Para Inferencia Local)
# ==============================================================================
def get_modified_sigmoid(x):
    return 2.0 * tf.math.pow(tf.math.sigmoid(x), tf.math.log(10.0)) + 1e-7

def upsample_controls_dynamic(controls, factor):
    # Aquí sí podemos usar tamaño dinámico porque no estamos en MirroredStrategy
    n_samples = tf.shape(controls)[1] * factor
    controls_4d = tf.expand_dims(controls, axis=2)
    upsampled_4d = tf.image.resize(controls_4d, [n_samples, 1], method=tf.image.ResizeMethod.BILINEAR)
    return tf.squeeze(upsampled_4d, axis=2)

class HarmonicSynthesizerDynamic:
    def __init__(self, sample_rate=16000, frame_rate=250, n_harmonics=96):
        self.sr = sample_rate
        self.n_harmonics = n_harmonics
        self.hop_size = sample_rate // frame_rate

    def __call__(self, f0, amplitude, harmonics):
        f0_up = upsample_controls_dynamic(f0, self.hop_size)
        amp_up = upsample_controls_dynamic(amplitude, self.hop_size)
        harm_up = upsample_controls_dynamic(harmonics, self.hop_size)

        harmonic_multipliers = tf.reshape(tf.range(1, self.n_harmonics + 1, dtype=tf.float32), [1, 1, self.n_harmonics])
        harmonic_freqs = f0_up * harmonic_multipliers
        antialiasing_mask = tf.cast(harmonic_freqs < (self.sr / 2.0), tf.float32)

        phases = tf.math.cumsum(harmonic_freqs * (2.0 * np.pi / self.sr), axis=1)
        wavs = tf.math.sin(phases) * harm_up * antialiasing_mask
        
        audio = tf.reduce_sum(wavs, axis=-1, keepdims=True) * amp_up
        return tf.squeeze(audio, axis=-1)

class FilteredNoiseSynthesizerDynamic:
    def __init__(self, sample_rate=16000, frame_rate=250, n_noise_filters=64):
        self.hop_size = sample_rate // frame_rate
        self.window_size = self.hop_size * 2

    def __call__(self, noise_magnitudes):
        batch_size = tf.shape(noise_magnitudes)[0]
        n_frames = tf.shape(noise_magnitudes)[1]
        n_samples = n_frames * self.hop_size
        
        white_noise = tf.random.uniform([batch_size, n_samples], minval=-1.0, maxval=1.0)
        
        noise_frames = tf.signal.frame(white_noise, frame_length=self.window_size, frame_step=self.hop_size, pad_end=True)
        noise_fft = tf.signal.rfft(noise_frames)

        magnitudes_padded = tf.pad(noise_magnitudes, paddings=[[0, 0], [0, 0], [0, 1]])
        filtered_fft = noise_fft * tf.cast(magnitudes_padded, tf.complex64)
        
        filtered_frames = tf.signal.irfft(filtered_fft) * tf.signal.hann_window(self.window_size)
        audio_noise = tf.signal.overlap_and_add(filtered_frames, frame_step=self.hop_size)
        return audio_noise[:, :n_samples]

# ==============================================================================
# 3. RED NEURONAL CORE
# ==============================================================================
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
        self.dense_final = tf.keras.layers.Dense(cfg['total_params_output'])

    def call(self, pitch, velocity, training=False):
        x = tf.concat([pitch, velocity], axis=-1)
        x = self.leaky_hidden(self.dense_hidden(self.gru(self.leaky_in(self.dense_in(x)), training=training)))
        
        act = get_modified_sigmoid(self.dense_final(x))
        return {
            'amplitude': act[..., 0:1],
            'harmonics': act[..., 1:1 + self.n_harmonics],
            'noise_magnitudes': act[..., 1 + self.n_harmonics:]
        }

# ==============================================================================
# 4. FUNCIÓN EXTRACCIÓN MIDI A TENSORES
# ==============================================================================
def midi_to_controls(midi_path, frame_rate=250):
    print(f"[INFO] Analizando {midi_path}...")
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    
    # Extraer duración total y crear array de tiempos
    end_time = midi_data.get_end_time()
    times = np.arange(0, end_time, 1.0 / frame_rate)
    
    pitch_curve = np.zeros_like(times)
    vel_curve = np.zeros_like(times)
    
    piano_roll = midi_data.get_piano_roll(fs=frame_rate)
    
    for t_idx in range(min(len(times), piano_roll.shape[1])):
        active_pitches = np.where(piano_roll[:, t_idx] > 0)[0]
        if len(active_pitches) > 0:
            # Monofonía: Nos quedamos con la nota más aguda
            highest_pitch = active_pitches[-1]
            pitch_curve[t_idx] = highest_pitch
            vel_curve[t_idx] = piano_roll[highest_pitch, t_idx] / 127.0
            
    # Añadir dimensión de Batch y Canal -> Forma final: [1, Frames, 1]
    pitch_tensor = tf.convert_to_tensor(pitch_curve, dtype=tf.float32)[tf.newaxis, ..., tf.newaxis]
    vel_tensor = tf.convert_to_tensor(vel_curve, dtype=tf.float32)[tf.newaxis, ..., tf.newaxis]
    
    return pitch_tensor, vel_tensor

# ==============================================================================
# 5. EJECUCIÓN PRINCIPAL
# ==============================================================================
def synthesize_midi(midi_file, weights_file, output_wav):
    # 1. Extraer curvas de control
    pitch, velocity = midi_to_controls(midi_file)
    print(f"[INFO] MIDI cargado. Longitud de frames: {pitch.shape[1]}")
    
    # 2. Instanciar Modelo y Sintetizadores
    model = DDSPCore(CONFIG)
    harmonic_synth = HarmonicSynthesizerDynamic()
    noise_synth = FilteredNoiseSynthesizerDynamic()
    
    # 3. Dummy Pass (Obligatorio para cargar pesos de una subclase)
    print("[INFO] Instanciando red...")
    _ = model(tf.zeros((1, 100, 1)), tf.zeros((1, 100, 1)))
    
    # 4. Cargar el cerebro entrenado
    print(f"[INFO] Cargando pesos desde {weights_file}...")
    model.load_weights(weights_file)
    
    # 5. Inferencia (El momento de la verdad)
    print("[INFO] Sintetizando audio (esto puede tardar unos segundos)...")
    params = model(pitch, velocity, training=False)
    
    audio_harm = harmonic_synth(pitch, params['amplitude'], params['harmonics'])
    audio_noise = noise_synth(params['noise_magnitudes'])
    audio_final = audio_harm + audio_noise
    
    # Extraer el tensor a numpy array
    audio_np = tf.squeeze(audio_final).numpy()
    
    # 6. Guardar WAV
    scipy.io.wavfile.write(output_wav, 16000, audio_np)
    print(f"[ÉXITO] ¡Audio guardado en {output_wav}!")

if __name__ == "__main__":
    # RUTAS PARA QUE LO PRUEBES LUEGO
    # Asegúrate de poner una melodía simple en el archivo MIDI de prueba
    ARCHIVO_MIDI_PRUEBA = "Mompou.mid" 
    PESOS_KAGGLE = "modelo_piano.weights.h5"
    SALIDA_WAV = "resultado_ddsp.wav"
    
    if os.path.exists(ARCHIVO_MIDI_PRUEBA) and os.path.exists(PESOS_KAGGLE):
        synthesize_midi(ARCHIVO_MIDI_PRUEBA, PESOS_KAGGLE, SALIDA_WAV)
    else:
        print("[AVISO] Te faltan los archivos para probar. Descarga los pesos de Kaggle y crea un .mid de prueba.")