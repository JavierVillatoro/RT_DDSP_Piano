import tensorflow as tf
import numpy as np
import pretty_midi
import scipy.io.wavfile
import os
from tqdm import tqdm

# ==============================================================================
# 1. CONFIGURACIÓN 
# ==============================================================================
CONFIG = {
    'audio': {'sample_rate': 16000, 'frame_rate': 250},
    'model': {'n_harmonics': 96, 'n_noise_filters': 64, 'hidden_size': 128, 'gru_units': 192, 'dense_output_size': 192},
}

MAX_VOICES = 24          
T_RELEASE_SEC = 1.0     

# ==============================================================================
# 2. SINTETIZADORES DINÁMICOS (Inferencia Local)
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
    def __init__(self, sample_rate=16000, frame_rate=250):
        self.hop_size = sample_rate // frame_rate
        self.window_size = self.hop_size * 2

    def __call__(self, noise_magnitudes):
        batch_size = tf.shape(noise_magnitudes)[0]
        n_frames = tf.shape(noise_magnitudes)[1]
        n_samples = n_frames * self.hop_size
        
        white_noise = tf.random.uniform([batch_size, n_samples], minval=-1.0, maxval=1.0)
        noise_frames = tf.signal.frame(white_noise, frame_length=self.window_size, frame_step=self.hop_size, pad_end=True)
        noise_fft = tf.signal.rfft(noise_frames)

        magnitudes_padded = tf.pad(noise_magnitudes, [[0, 0], [0, 0], [0, 1]])
        filtered_fft = noise_fft * tf.cast(magnitudes_padded, tf.complex64)
        
        filtered_frames = tf.signal.irfft(filtered_fft)
        filtered_frames = tf.roll(filtered_frames, shift=self.window_size // 2, axis=-1)
        filtered_frames *= tf.signal.hann_window(self.window_size)
        audio_noise = tf.signal.overlap_and_add(filtered_frames, frame_step=self.hop_size)
        return audio_noise[:, :n_samples]

# ==============================================================================
# 3. MÓDULOS FÍSICOS (Espejo de la arquitectura final)
# ==============================================================================
class ContextNetwork(tf.keras.Model): 
    def __init__(self, context_dim=32, gru_units=64):
        super().__init__()
        self.dense_in = tf.keras.layers.Dense(context_dim, activation=tf.nn.leaky_relu)
        self.gru = tf.keras.layers.GRU(gru_units, return_sequences=True)
        self.dense_out = tf.keras.layers.Dense(context_dim)

    def call(self, inputs): # Recibe tensor combinado
        x = self.dense_in(inputs)
        x = self.gru(x)
        return self.dense_out(x)

class Detuner(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(1, kernel_initializer='zeros')

    def build(self, input_shape):
        self.static_detune = self.add_weight(name='static_detune', shape=(1,), initializer='zeros')
        super().build(input_shape)

    def call(self, pitch_midi):
        x = pitch_midi / 127.0
        return tf.nn.tanh(self.dense(x)) + tf.nn.tanh(self.static_detune)

class TrainableReverb(tf.keras.Model):
    def __init__(self, ir_length=24000, **kwargs):
        super().__init__(**kwargs)
        self.ir_length = ir_length

    def build(self, input_shape):
        np_ir_init = np.random.normal(size=self.ir_length) * np.exp(-np.linspace(0.0, 5.0, self.ir_length))
        self.ir = self.add_weight(name="impulse_response", shape=[self.ir_length], initializer=tf.constant_initializer(np_ir_init))
        super().build(input_shape)

    def call(self, audio):
        n_fft = tf.pow(2, tf.cast(tf.math.ceil(tf.math.log(tf.cast(tf.shape(audio)[1] + self.ir_length, tf.float32)) / tf.math.log(2.0)), tf.int32))
        audio_fft = tf.signal.rfft(audio, [n_fft])
        ir_fft = tf.signal.rfft(self.ir, [n_fft])
        reverb_fft = audio_fft * ir_fft
        out = tf.signal.irfft(reverb_fft)[:, :tf.shape(audio)[1]]
        return out # Sin cálculo de loss para inferencia

class InharmonicityModel(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.alpha_B = self.add_weight(name='alpha_B', shape=(), initializer=tf.constant_initializer(-0.0847))
        self.beta_B = self.add_weight(name='beta_B', shape=(), initializer=tf.constant_initializer(-5.82))
        self.alpha_T = self.add_weight(name='alpha_T', shape=(), initializer=tf.constant_initializer(0.0926))
        self.beta_T = self.add_weight(name='beta_T', shape=(), initializer=tf.constant_initializer(-13.64))
        super().build(input_shape)

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

    def call(self, core_input, pitch_midi):
        x = self.leaky_hidden(self.dense_hidden(self.gru(self.leaky_in(self.dense_in(core_input)))))
        
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
        
        return {'amplitude': amp, 'harmonics': harm_dist, 'noise_magnitudes': noise_mags, 'f_k': f_k}

# ==============================================================================
# 4. POLYPHONIC WRAPPER (Orquestador Inferencia)
# ==============================================================================
class PolyphonicDDSPPianoDynamic(tf.keras.Model):
    def __init__(self, config, n_voices=1):
        super().__init__()
        self.n_voices = n_voices
        self.context_net = ContextNetwork()
        self.core = DDSPCore(config)
        self.detuner = Detuner()
        
        self.harmonic_synth = HarmonicSynthesizerDynamic()
        self.noise_synth = FilteredNoiseSynthesizerDynamic()
        self.reverb = TrainableReverb()

    def call(self, inputs):
        pitches = inputs['pitches']       
        velocities = inputs['velocities'] 
        pedal = inputs['pedal']           

        # --- CONTEXTO GLOBAL ---
        global_vel = tf.reduce_sum(velocities, axis=1)
        pitches_norm = pitches / 127.0
        active_pitches = tf.where(velocities > 0, pitches_norm, 0.0)
        global_pitch = tf.reduce_sum(active_pitches, axis=1) / (tf.reduce_sum(tf.cast(velocities > 0, tf.float32), axis=1) + 1e-7)
        context_input = tf.concat([pedal, global_vel, global_pitch], axis=-1)
        c_t = self.context_net(context_input) 

        audio_sum = 0.0
        
        # --- AQUÍ ESTÁ LA MAGIA DEL TQDM ---
        # Usamos tqdm para visualizar el progreso de las 8 voces
        for i in tqdm(range(self.n_voices), desc="Sintetizando Voces", unit="voz"):
            v_pitch = pitches[:, i, :, :] 
            v_vel = velocities[:, i, :, :]
            
            # Normalización
            v_pitch_norm = v_pitch / 127.0
            core_input = tf.concat([v_pitch_norm, v_vel, c_t], axis=-1)
            
            params = self.core(core_input, v_pitch) 
            
            delta_f = self.detuner(v_pitch)
            f_k_1 = params['f_k']
            f_k_2 = f_k_1 * tf.pow(2.0, (delta_f / 12.0))
            
            audio_voice = self.harmonic_synth(f_k_1, params['amplitude'], params['harmonics'])
            audio_voice += self.harmonic_synth(f_k_2, params['amplitude'], params['harmonics'])
            audio_voice += self.noise_synth(params['noise_magnitudes'])
            
            audio_sum += audio_voice

        # Aplicamos la reverb final fuera del bucle
        return self.reverb(audio_sum)

# ==============================================================================
# 5. FUNCIÓN EXTRACCIÓN MIDI A TENSORES 
# ==============================================================================
def extract_polyphony_and_release(midi_data, total_frames):
    pitch_curves = np.zeros((MAX_VOICES, total_frames), dtype=np.float32)
    vel_curves = np.zeros((MAX_VOICES, total_frames), dtype=np.float32)
    voice_free_frame = np.zeros(MAX_VOICES, dtype=int)
    
    instrument = midi_data.instruments[0]
    notes = sorted(instrument.notes, key=lambda n: n.start)
    
    for note in notes:
        start_f = int(note.start * CONFIG['audio']['frame_rate'])
        end_f = int(note.end * CONFIG['audio']['frame_rate'])
        release_f = min(total_frames, int((note.end + T_RELEASE_SEC) * CONFIG['audio']['frame_rate']))
        
        assigned_voice = -1
        for v in range(MAX_VOICES):
            if start_f >= voice_free_frame[v]:
                assigned_voice = v
                break
                
        if assigned_voice != -1:
            pitch_curves[assigned_voice, start_f:release_f] = note.pitch
            vel_curves[assigned_voice, start_f:end_f] = note.velocity / 127.0
            voice_free_frame[assigned_voice] = release_f
            
    return pitch_curves, vel_curves

def extract_sustain_pedal(midi_data, total_frames):
    pedal_curve = np.zeros(total_frames, dtype=np.float32)
    instrument = midi_data.instruments[0]
    sustain_events = [cc for cc in instrument.control_changes if cc.number == 64]
    sustain_events = sorted(sustain_events, key=lambda c: c.time)
    
    if not sustain_events:
        return pedal_curve
        
    current_val = 0.0
    event_idx = 0
    n_events = len(sustain_events)
    
    for f in range(total_frames):
        t = f / CONFIG['audio']['frame_rate']
        while event_idx < n_events and t >= sustain_events[event_idx].time:
            current_val = sustain_events[event_idx].value / 127.0
            event_idx += 1
        pedal_curve[f] = current_val
        
    return pedal_curve

def midi_to_controls(midi_path):
    # Cargamos el MIDI
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    total_time = midi_data.get_end_time()
    # CORRECCIÓN: Usar CONFIG para el frame_rate
    total_frames = int(total_time * CONFIG['audio']['frame_rate']) + 1 
    
    # 1. Extraemos con el embudo abierto a MAX_VOICES (24)
    pitch_curves, vel_curves = extract_polyphony_and_release(midi_data, total_frames)
    pedal_curve = extract_sustain_pedal(midi_data, total_frames)
    
    # 2. Convertimos a Tensores
    p_pitch = tf.convert_to_tensor(pitch_curves, dtype=tf.float32)
    p_pitch = tf.reshape(p_pitch, [1, MAX_VOICES, total_frames, 1])
    
    p_vel = tf.convert_to_tensor(vel_curves, dtype=tf.float32)
    p_vel = tf.reshape(p_vel, [1, MAX_VOICES, total_frames, 1])
    
    p_pedal = tf.convert_to_tensor(pedal_curve, dtype=tf.float32)
    p_pedal = tf.reshape(p_pedal, [1, total_frames, 1])
    
    return p_pitch, p_vel, p_pedal

# ==============================================================================
# 6. EJECUCIÓN PRINCIPAL (GENERACIÓN POR BLOQUES CON CROSSFADE)
# ==============================================================================
def get_user_duration(total_frames, frame_rate):
    max_sec = total_frames / frame_rate
    print(f"\n[INFO] El MIDI cargado dura un total de {max_sec:.1f} segundos.")
    ans = input("¿Cuántos segundos quieres generar? (Deja en blanco para la canción COMPLETA): ")
    if ans.strip() == "":
        return total_frames
    try:
        sec = float(ans)
        return min(total_frames, int(sec * frame_rate))
    except ValueError:
        print("[AVISO] Entrada no válida. Generando la canción completa.")
        return total_frames

def synthesize_midi(midi_file, weights_folder, output_wav):
    p_pitch, p_vel, p_pedal = midi_to_controls(midi_file)
    
    total_frames_original = p_pitch.shape[2]
    frame_rate = CONFIG['audio']['frame_rate']
    sample_rate = CONFIG['audio']['sample_rate']
    
    # 1. PREGUNTAR AL USUARIO
    frames_to_generate = get_user_duration(total_frames_original, frame_rate)
    
    # Recortar tensores según lo que pidió el usuario
    p_pitch = p_pitch[:, :, :frames_to_generate, :]
    p_vel = p_vel[:, :, :frames_to_generate, :]
    p_pedal = p_pedal[:, :frames_to_generate, :]
    
    duracion_segundos = frames_to_generate / frame_rate
    print(f"\n[INFO] Preparando generación de: {duracion_segundos:.1f} segundos de audio en CPU...")
    
    # 2. CARGAR EL MODELO
    model = PolyphonicDDSPPianoDynamic(CONFIG, n_voices=MAX_VOICES)
    
    # Dummy Pass para inicializar la red
    dummy_pitch = tf.zeros((1, MAX_VOICES, 10, 1), dtype=tf.float32)
    dummy_vel = tf.zeros((1, MAX_VOICES, 10, 1), dtype=tf.float32)
    dummy_pedal = tf.zeros((1, 10, 1), dtype=tf.float32)
    _ = model({'pitches': dummy_pitch, 'velocities': dummy_vel, 'pedal': dummy_pedal})
    
    print(f"[INFO] Cargando pesos desde: {weights_folder}...")
    model.core.load_weights(os.path.join(weights_folder, "core.weights.h5"))
    model.context_net.load_weights(os.path.join(weights_folder, "context.weights.h5"))
    model.detuner.load_weights(os.path.join(weights_folder, "detuner.weights.h5"))
    model.reverb.load_weights(os.path.join(weights_folder, "reverb.weights.h5"))
    print("[OK] Red cargada. Iniciando procesamiento por bloques...\n")
    
    # 3. CONFIGURACIÓN DEL CROSSFADE (Overlap-Add)
    chunk_sec = 12.0
    step_sec = 10.0
    fade_sec = chunk_sec - step_sec  # 2.0 segundos de solapamiento
    
    chunk_frames = int(chunk_sec * frame_rate)
    step_frames = int(step_sec * frame_rate)
    
    total_samples = int((frames_to_generate / frame_rate) * sample_rate)
    audio_final = np.zeros(total_samples, dtype=np.float32)
    
    fade_samples = int(fade_sec * sample_rate)
    # Curvas de igual potencia para no perder volumen en la transición
    fade_out = np.cos(np.linspace(0, np.pi/2, fade_samples, dtype=np.float32)) ** 2
    fade_in  = np.sin(np.linspace(0, np.pi/2, fade_samples, dtype=np.float32)) ** 2

    # 4. BUCLE DE INFERENCIA
    start_frame = 0
    chunk_idx = 1
    
    # Calculamos cuántos chunks habrá para el log visual
    total_chunks = (frames_to_generate - chunk_frames) // step_frames + 2 if frames_to_generate > chunk_frames else 1

    while start_frame < frames_to_generate:
        end_frame = min(start_frame + chunk_frames, frames_to_generate)
        
        print(f"--- Procesando Bloque {chunk_idx}/{total_chunks} ({start_frame/frame_rate:.1f}s a {end_frame/frame_rate:.1f}s) ---")
        
        # Extraemos el fragmento de los tensores
        chunk_pitch = p_pitch[:, :, start_frame:end_frame, :]
        chunk_vel = p_vel[:, :, start_frame:end_frame, :]
        chunk_pedal = p_pedal[:, start_frame:end_frame, :]
        
        inputs = {'pitches': chunk_pitch, 'velocities': chunk_vel, 'pedal': chunk_pedal}
        
        # Inferencia neuronal pura
        chunk_audio_tensor = model(inputs)
        chunk_audio = tf.squeeze(chunk_audio_tensor).numpy()
        
        # Matemáticas del posicionado de audio
        start_sample = int((start_frame / frame_rate) * sample_rate)
        audio_len = len(chunk_audio)
        
        if start_frame == 0:
            # El primer bloque se pega tal cual
            audio_final[start_sample : start_sample + audio_len] = chunk_audio
        else:
            # Aplicamos Fade In al inicio del nuevo bloque (que está amnésico)
            current_fade_len = min(fade_samples, audio_len)
            chunk_audio[:current_fade_len] *= fade_in[:current_fade_len]
            
            # Aplicamos Fade Out al final del audio viejo ya renderizado
            audio_final[start_sample : start_sample + current_fade_len] *= fade_out[:current_fade_len]
            
            # Sumamos las ondas
            audio_final[start_sample : start_sample + audio_len] += chunk_audio
        
        start_frame += step_frames
        chunk_idx += 1

    # 5. GUARDADO
    print(f"\n[INFO] Normalizando audio a -1dB para evitar clipeo...")
    max_val = np.max(np.abs(audio_final))
    if max_val > 0:
        audio_final = audio_final * (0.89 / max_val)

    scipy.io.wavfile.write(output_wav, sample_rate, audio_final)
    print(f"[ÉXITO] ¡Audio ensamblado y guardado en: {output_wav}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    ARCHIVO_MIDI_PRUEBA = os.path.join(BASE_DIR, "Ondine.mid") 
    CARPETA_PESOS = os.path.join(BASE_DIR, "checkpoints_descargados_06_2") 
    SALIDA_WAV = os.path.join(BASE_DIR, "resultado_Ondine_Crossfade_full.wav")
    
    if os.path.exists(ARCHIVO_MIDI_PRUEBA) and os.path.exists(CARPETA_PESOS):
        synthesize_midi(ARCHIVO_MIDI_PRUEBA, CARPETA_PESOS, SALIDA_WAV)
    else:
        print("[AVISO] Faltan archivos en este directorio. Revisa las rutas.")