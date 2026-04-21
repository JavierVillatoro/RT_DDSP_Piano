import os
import pandas as pd
import numpy as np
import librosa
import pretty_midi
import tensorflow as tf
from tqdm import tqdm

# ==============================================================================
# 1. CONFIGURACIÓN (Sincronizada con tu red)
# ==============================================================================
SR = 16000
FRAME_RATE = 250
CHUNK_DURATION = 3.0  # Segundos
OVERLAP = 0.5         # 50% de solapamiento
SAMPLES_PER_CHUNK = int(SR * CHUNK_DURATION)
FRAMES_PER_CHUNK = int(FRAME_RATE * CHUNK_DURATION)
HOP_SAMPLES = int(SAMPLES_PER_CHUNK * (1.0 - OVERLAP))

# --- NUEVAS CONFIGURACIONES POLIFÓNICAS ---
MAX_VOICES = 8         # Número de voces simultáneas (acordes de hasta 8 notas)
T_RELEASE_SEC = 1.0     # Módulo Note Release: Mantener el pitch 1 segundo tras soltar la tecla

# ==============================================================================
# 2. FUNCIONES DE EXTRACCIÓN Y SERIALIZACIÓN
# ==============================================================================
def _float_feature(value):
    """Convierte un array (1D o ND aplanado) en un Feature de TensorFlow."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten()))

def serialize_example_poly(pitch, velocity, pedal, audio):
    """
    Crea un mensaje tf.train.Example polifónico.
    pitch y velocity ahora tienen forma [MAX_VOICES, FRAMES_PER_CHUNK]
    pedal tiene forma [FRAMES_PER_CHUNK]
    """
    feature = {
        'pitch': _float_feature(pitch),       # Aplanado: 8 x 750 = 6000
        'velocity': _float_feature(velocity), # Aplanado: 8 x 750 = 6000
        'pedal': _float_feature(pedal),       # Aplanado: 750
        'audio': _float_feature(audio),       # Aplanado: 48000
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def extract_polyphony_and_release(midi_data, total_frames):
    """Asigna notas a canales independientes, aplica Note Release y Micro-Fadeout en robos."""
    pitch_curves = np.zeros((MAX_VOICES, total_frames), dtype=np.float32)
    vel_curves = np.zeros((MAX_VOICES, total_frames), dtype=np.float32)
    voice_free_frame = np.zeros(MAX_VOICES, dtype=int)
    
    instrument = midi_data.instruments[0]
    notes = sorted(instrument.notes, key=lambda n: n.start)
    
    # 12 milisegundos de fade-out = 3 frames (a 250 Hz)
    FADE_FRAMES = 3 
    
    for note in notes:
        start_f = int(note.start * FRAME_RATE)
        end_f = int(note.end * FRAME_RATE)
        release_f = min(total_frames, int((note.end + T_RELEASE_SEC) * FRAME_RATE))
        
        assigned_voice = -1
        for v in range(MAX_VOICES):
            if start_f >= voice_free_frame[v]:
                assigned_voice = v
                break
                
        # --- VOICE STEALING + MICRO-FADEOUT ---
        if assigned_voice == -1:
            assigned_voice = np.argmin(voice_free_frame)
            
            # Calculamos dónde debe empezar a bajar el volumen
            fade_start = max(0, start_f - FADE_FRAMES)
            
            # Si hay espacio para hacer el fadeout antes de la nota nueva
            if fade_start < start_f:
                # Creamos una rampa que va desde el volumen actual hasta 0.0
                vol_actual = vel_curves[assigned_voice, fade_start]
                rampa_fadeout = np.linspace(vol_actual, 0.0, start_f - fade_start)
                vel_curves[assigned_voice, fade_start:start_f] = rampa_fadeout
                
            # Limpiamos el milisegundo exacto antes del ataque para "resetear" la GRU
            if start_f - 1 >= 0:
                vel_curves[assigned_voice, start_f - 1] = 0.0
        # --------------------------------------
        
        pitch_curves[assigned_voice, start_f:release_f] = note.pitch
        vel_curves[assigned_voice, start_f:end_f] = note.velocity / 127.0
        voice_free_frame[assigned_voice] = release_f
            
    return pitch_curves, vel_curves

def extract_sustain_pedal(midi_data, total_frames):
    """Extrae el CC64 (Sustain Pedal) como una curva continua de 0.0 a 1.0."""
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
        t = f / FRAME_RATE
        # Si llegamos al tiempo de un nuevo evento de pedal, actualizamos el valor
        while event_idx < n_events and t >= sustain_events[event_idx].time:
            current_val = sustain_events[event_idx].value / 127.0
            event_idx += 1
        pedal_curve[f] = current_val
        
    return pedal_curve

def process_track_poly(audio_path, midi_path):
    """Extrae curvas polifónicas de control y fragmentos de audio."""
    # 1. Cargar Audio
    audio, _ = librosa.load(audio_path, sr=SR, mono=True)
    
    # 2. Cargar MIDI
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    total_time = len(audio) / SR
    total_frames = int(total_time * FRAME_RATE) + 1
    
    # 3. Generar curvas complejas
    pitch_curves, vel_curves = extract_polyphony_and_release(midi_data, total_frames)
    pedal_curve = extract_sustain_pedal(midi_data, total_frames)

    # 4. Fragmentación
    fragments = []
    for start_sample in range(0, len(audio) - SAMPLES_PER_CHUNK, HOP_SAMPLES):
        end_sample = start_sample + SAMPLES_PER_CHUNK
        
        start_frame = int((start_sample / SR) * FRAME_RATE)
        end_frame = start_frame + FRAMES_PER_CHUNK
        
        if end_frame <= total_frames:
            audio_chunk = audio[start_sample:end_sample]
            pitch_chunk = pitch_curves[:, start_frame:end_frame] # Shape: [4, 750]
            vel_chunk = vel_curves[:, start_frame:end_frame]     # Shape: [4, 750]
            pedal_chunk = pedal_curve[start_frame:end_frame]     # Shape: [750]
            
            # Solo guardamos el fragmento si hay ALGUNA nota sonando en CUALQUIER voz
            if np.max(vel_chunk) > 0.01 or np.max(pedal_chunk) > 0.5:
                fragments.append((pitch_chunk, vel_chunk, pedal_chunk, audio_chunk))
                
    return fragments

def make_tfrecords_poly(dataset_root, output_dir, year=2009):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df = pd.read_csv(os.path.join(dataset_root, 'maestro-v3.0.0.csv'))
    
    for split in ['train', 'validation', 'test']:
        df_split = df[(df['year'] == year) & (df['split'] == split)]
        # EL NOMBRE AHORA TERMINA EN _poly
        record_path = os.path.join(output_dir, f"maestro_{year}_{split}_poly.tfrecord")
        
        print(f"\nGenerando {record_path}...")
        with tf.io.TFRecordWriter(record_path) as writer:
            for _, row in tqdm(df_split.iterrows(), total=len(df_split), desc=f"Split: {split}"):
                audio_file = os.path.normpath(os.path.join(dataset_root, row['audio_filename']))
                midi_file = os.path.normpath(os.path.join(dataset_root, row['midi_filename']))
                
                # Fallback MIDI (.midi -> .mid)
                if not os.path.exists(midi_file):
                    midi_file = midi_file[:-1]
                
                try:
                    fragments = process_track_poly(audio_file, midi_file)
                    for p, v, ped, a in fragments:
                        example = serialize_example_poly(p, v, ped, a)
                        writer.write(example)
                except Exception as e:
                    print(f"Error procesando {audio_file}: {e}")

if __name__ == "__main__":
    DATASET_ROOT = r"C:\Users\franc\Desktop\proyectos\DDSP\dataset\maestro-v3.0.0"
    OUTPUT_DIR = r"C:\Users\franc\Desktop\proyectos\DDSP\dataset\tfrecords_poly_2_1"
    
    make_tfrecords_poly(DATASET_ROOT, OUTPUT_DIR, year=2009)