import os
import pandas as pd
import numpy as np
import librosa
import pretty_midi
import tensorflow as tf
from tqdm import tqdm

# --- CONFIGURACIÓN (Sincronizada con config.yaml) ---
SR = 16000
FRAME_RATE = 250
CHUNK_DURATION = 3.0  # Segundos
OVERLAP = 0.5         # 50% de solapamiento
SAMPLES_PER_CHUNK = int(SR * CHUNK_DURATION)
FRAMES_PER_CHUNK = int(FRAME_RATE * CHUNK_DURATION)
HOP_SAMPLES = int(SAMPLES_PER_CHUNK * (1.0 - OVERLAP))

def _float_feature(value):
    """Convierte un array de floats en una característica de TensorFlow."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten()))

def serialize_example(pitch, velocity, audio):
    """Crea un mensaje tf.train.Example listo para escribirse a disco."""
    feature = {
        'pitch': _float_feature(pitch),       # [750,]
        'velocity': _float_feature(velocity), # [750,]
        'audio': _float_feature(audio),       # [48000,]
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def process_track(audio_path, midi_path):
    """Extrae curvas de control y fragmentos de un par Audio/MIDI."""
    # 1. Cargar Audio
    audio, _ = librosa.load(audio_path, sr=SR, mono=True)
    
    # 2. Cargar MIDI y generar curvas a 250 Hz
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    total_time = len(audio) / SR
    times = np.arange(0, total_time, 1.0 / FRAME_RATE)
    
    # Creamos arrays vacíos para Pitch y Velocity
    pitch_curve = np.zeros_like(times)
    vel_curve = np.zeros_like(times)
    
    # Extraemos la nota más aguda en cada frame para mantener la monofonía
    # (Estrategia estándar para DDSP Monophonic)
    piano_roll = midi_data.get_piano_roll(fs=FRAME_RATE) # [128, frames]
    for t_idx in range(min(len(times), piano_roll.shape[1])):
        active_pitches = np.where(piano_roll[:, t_idx] > 0)[0]
        if len(active_pitches) > 0:
            highest_pitch = active_pitches[-1]
            pitch_curve[t_idx] = highest_pitch
            vel_curve[t_idx] = piano_roll[highest_pitch, t_idx] / 127.0 # Normalizado 0-1

    # 3. Fragmentación
    fragments = []
    for start_sample in range(0, len(audio) - SAMPLES_PER_CHUNK, HOP_SAMPLES):
        end_sample = start_sample + SAMPLES_PER_CHUNK
        
        # Sincronizamos los índices de los frames (control)
        start_frame = int((start_sample / SR) * FRAME_RATE)
        end_frame = start_frame + FRAMES_PER_CHUNK
        
        if end_frame <= len(pitch_curve):
            audio_chunk = audio[start_sample:end_sample]
            pitch_chunk = pitch_curve[start_frame:end_frame]
            vel_chunk = vel_curve[start_frame:end_frame]
            
            # Solo guardamos el fragmento si hay actividad musical (evitar silencios)
            if np.max(vel_chunk) > 0.05:
                fragments.append((pitch_chunk, vel_chunk, audio_chunk))
                
    return fragments

def make_tfrecords(dataset_root, output_dir, year=2009):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df = pd.read_csv(os.path.join(dataset_root, 'maestro-v3.0.0.csv'))
    # Filtramos por el año y split deseado (ej. train)
    for split in ['train', 'validation', 'test']:
        df_split = df[(df['year'] == year) & (df['split'] == split)]
        record_path = os.path.join(output_dir, f"maestro_{year}_{split}.tfrecord")
        
        print(f"\nGenerando {record_path}...")
        with tf.io.TFRecordWriter(record_path) as writer:
            for _, row in tqdm(df_split.iterrows(), total=len(df_split), desc=f"Split: {split}"):
                audio_file = os.path.normpath(os.path.join(dataset_root, row['audio_filename']))
                midi_file = os.path.normpath(os.path.join(dataset_root, row['midi_filename']))
                
                # Fallback MIDI (.midi -> .mid)
                if not os.path.exists(midi_file):
                    midi_file = midi_file[:-1]
                
                try:
                    fragments = process_track(audio_file, midi_file)
                    for p, v, a in fragments:
                        example = serialize_example(p, v, a)
                        writer.write(example)
                except Exception as e:
                    print(f"Error procesando {audio_file}: {e}")

if __name__ == "__main__":
    DATASET_ROOT = r"C:\Users\franc\Desktop\proyectos\DDSP\dataset\maestro-v3.0.0"
    OUTPUT_DIR = r"C:\Users\franc\Desktop\proyectos\DDSP\dataset\tfrecords"
    
    make_tfrecords(DATASET_ROOT, OUTPUT_DIR, year=2009)