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
    n_samples = tf.shape(controls)[1] * factor
    controls_4d = tf.expand_dims(controls, axis=2)
    upsampled_4d = tf.image.resize(controls_4d, [n_samples, 1], method=tf.image.ResizeMethod.BILINEAR)
    return tf.squeeze(upsampled_4d, axis=2)

class HarmonicSynthesizerDynamic:
    def __init__(self, sample_rate=16000, frame_rate=250, n_harmonics=96):
        self.sr = sample_rate
        self.n_harmonics = n_harmonics
        self.hop_size = sample_rate // frame_rate

    def __call__(self, f_k, amplitude, harmonics):
        f_k_up = upsample_controls_dynamic(f_k, self.hop_size)
        amp_up = upsample_controls_dynamic(amplitude, self.hop_size)
        harm_up = upsample_controls_dynamic(harmonics, self.hop_size)

        antialiasing_mask = tf.cast(f_k_up < (self.sr / 2.0), tf.float32)

        phases = tf.math.cumsum(f_k_up * (2.0 * np.pi / self.sr), axis=1)
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
# 3. MÓDULOS DE MODELADO FÍSICO Y CONTEXTO
# ==============================================================================
class ContextNetwork(tf.keras.layers.Layer):
    def __init__(self, context_dim=32, gru_units=64):
        super().__init__()
        self.dense_in = tf.keras.layers.Dense(context_dim, activation=tf.nn.leaky_relu)
        self.gru = tf.keras.layers.GRU(gru_units, return_sequences=True)
        self.dense_out = tf.keras.layers.Dense(context_dim)

    def call(self, sustain_pedal):
        x = self.dense_in(sustain_pedal)
        x = self.gru(x)
        return self.dense_out(x)

class Detuner(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(1, kernel_initializer='zeros')

    def call(self, pitch_midi):
        x = pitch_midi / 127.0
        return tf.nn.tanh(self.dense(x))

class TrainableReverb(tf.keras.layers.Layer):
    def __init__(self, ir_length=24000, lambda_ir=1e-4):
        super().__init__()
        self.ir_length = ir_length
        self.lambda_ir = lambda_ir
        ir_init = tf.random.normal([ir_length]) * tf.exp(-tf.linspace(0.0, 5.0, ir_length))
        self.ir = tf.Variable(ir_init, trainable=True, name="impulse_response")

    def call(self, audio):
        n_fft = tf.pow(2, tf.cast(tf.math.ceil(tf.math.log(tf.cast(tf.shape(audio)[1] + self.ir_length, tf.float32)) / tf.math.log(2.0)), tf.int32))
        
        audio_fft = tf.signal.rfft(audio, [n_fft])
        ir_fft = tf.signal.rfft(self.ir, [n_fft])
        
        reverb_fft = audio_fft * ir_fft
        out = tf.signal.irfft(reverb_fft)[:, :tf.shape(audio)[1]]
        
        self.add_loss(self.lambda_ir * tf.reduce_sum(tf.abs(self.ir)))
        return out

class InharmonicityModel(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.alpha_B = tf.Variable(-0.0847, dtype=tf.float32, trainable=True)
        self.beta_B = tf.Variable(-5.82, dtype=tf.float32, trainable=True)
        self.alpha_T = tf.Variable(0.0926, dtype=tf.float32, trainable=True)
        self.beta_T = tf.Variable(-13.64, dtype=tf.float32, trainable=True)

    def call(self, pitch_midi):
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
        stretch_factor = tf.sqrt(1.0 + B_factor * tf.square(k))
        
        f_k = f0_hz * k * stretch_factor
        
        return {
            'amplitude': amp,
            'harmonics': harm_dist,
            'noise_magnitudes': noise_mags,
            'f_k': f_k
        }

# ==============================================================================
# 4. POLYPHONIC WRAPPER (El Orquestador para Inferencia)
# ==============================================================================
class PolyphonicDDSPPianoDynamic(tf.keras.Model):
    def __init__(self, config, n_voices=1):
        super().__init__()
        self.n_voices = n_voices
        self.context_net = ContextNetwork()
        self.core = DDSPCore(config)
        self.detuner = Detuner()
        
        # Usamos las versiones DYNAMIC para inferencia (sin restricción de batch/longitud)
        self.harmonic_synth = HarmonicSynthesizerDynamic()
        self.noise_synth = FilteredNoiseSynthesizerDynamic()
        self.reverb = TrainableReverb()

    def call(self, inputs, training=False):
        pitches = inputs['pitches']       
        velocities = inputs['velocities'] 
        pedal = inputs['pedal']           

        c_t = self.context_net(pedal) 

        audio_sum = 0.0
        
        for i in range(self.n_voices):
            v_pitch = pitches[:, i, :, :] 
            v_vel = velocities[:, i, :, :]
            
            core_input = tf.concat([v_pitch, v_vel, c_t], axis=-1)
            
            params = self.core(core_input, v_pitch, training=training) 
            
            delta_f = self.detuner(v_pitch)
            f_k_1 = params['f_k']
            f_k_2 = f_k_1 * tf.pow(2.0, (delta_f / 12.0))
            
            audio_voice = self.harmonic_synth(f_k_1, params['amplitude'], params['harmonics'])
            audio_voice += self.harmonic_synth(f_k_2, params['amplitude'], params['harmonics'])
            
            audio_voice += self.noise_synth(params['noise_magnitudes'])
            
            audio_sum += audio_voice

        final_audio = self.reverb(audio_sum)
        return final_audio

# ==============================================================================
# 5. FUNCIÓN EXTRACCIÓN MIDI A TENSORES
# ==============================================================================
def midi_to_controls(midi_path, frame_rate=250):
    print(f"[INFO] Analizando {midi_path}...")
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    
    end_time = midi_data.get_end_time()
    times = np.arange(0, end_time, 1.0 / frame_rate)
    
    pitch_curve = np.zeros_like(times)
    vel_curve = np.zeros_like(times)
    
    piano_roll = midi_data.get_piano_roll(fs=frame_rate)
    
    for t_idx in range(min(len(times), piano_roll.shape[1])):
        active_pitches = np.where(piano_roll[:, t_idx] > 0)[0]
        if len(active_pitches) > 0:
            highest_pitch = active_pitches[-1]
            pitch_curve[t_idx] = highest_pitch
            vel_curve[t_idx] = piano_roll[highest_pitch, t_idx] / 127.0
            
    # Forma [Batch=1, Time, Channels=1]
    pitch_tensor = tf.convert_to_tensor(pitch_curve, dtype=tf.float32)[tf.newaxis, ..., tf.newaxis]
    vel_tensor = tf.convert_to_tensor(vel_curve, dtype=tf.float32)[tf.newaxis, ..., tf.newaxis]
    
    return pitch_tensor, vel_tensor

# ==============================================================================
# 6. EJECUCIÓN PRINCIPAL
# ==============================================================================
def synthesize_midi(midi_file, weights_file, output_wav):
    # 1. Extraer curvas
    pitch, velocity = midi_to_controls(midi_file)
    print(f"[INFO] MIDI cargado. Longitud de frames: {pitch.shape[1]}")
    
    # 2. Instanciar Orquestador Dinámico (1 voz para coincidir con el entrenamiento)
    model = PolyphonicDDSPPianoDynamic(CONFIG, n_voices=1)
    
    # 3. Preparar el diccionario de entrada simulando la dimensión extra de la voz
    p_pitch = tf.expand_dims(pitch, axis=1) # [1, 1, Time, 1]
    p_vel = tf.expand_dims(velocity, axis=1) # [1, 1, Time, 1]
    dummy_pedal = tf.zeros_like(pitch)       # [1, Time, 1]
    
    inputs = {'pitches': p_pitch, 'velocities': p_vel, 'pedal': dummy_pedal}
    
    # 4. Dummy Pass
    print("[INFO] Instanciando red con Dummy Pass...")
    _ = model(inputs, training=False)
    
    # 5. Cargar Pesos
    print(f"[INFO] Cargando pesos desde {weights_file}...")
    model.load_weights(weights_file)
    
    # 6. Inferencia
    print("[INFO] Sintetizando audio (esto puede tardar unos segundos)...")
    audio_final = model(inputs, training=False)
    
    audio_np = tf.squeeze(audio_final).numpy()
    
    # 7. Guardar WAV
    scipy.io.wavfile.write(output_wav, 16000, audio_np)
    print(f"[ÉXITO] ¡Audio guardado en {output_wav}!")

if __name__ == "__main__":
    ARCHIVO_MIDI_PRUEBA = "Mompou.mid" 
    PESOS_KAGGLE = "modelo_piano.weights_02.h5" 
    SALIDA_WAV = "resultado_ddsp_reverb.wav"
    
    if os.path.exists(ARCHIVO_MIDI_PRUEBA) and os.path.exists(PESOS_KAGGLE):
        synthesize_midi(ARCHIVO_MIDI_PRUEBA, PESOS_KAGGLE, SALIDA_WAV)
    else:
        print("[AVISO] Faltan archivos en este directorio.")
        print(f" -> Buscando: {ARCHIVO_MIDI_PRUEBA} y {PESOS_KAGGLE}")