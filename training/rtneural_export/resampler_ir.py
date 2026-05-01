import scipy.io.wavfile as wav
import scipy.signal as signal
import numpy as np
import os

def resample_ir(input_file, output_file, target_sr=44100):
    print(f"[INFO] Leyendo archivo: {input_file}...")
    
    # 1. Leer el archivo original
    try:
        orig_sr, audio_orig = wav.read(input_file)
    except FileNotFoundError:
        print(f"¡Error! No encuentro el archivo {input_file}.")
        return

    print(f"[INFO] Sample rate original: {orig_sr} Hz")
    print(f"[INFO] Convirtiendo a: {target_sr} Hz...")

    # Si ya está a 44100, no hacemos nada
    if orig_sr == target_sr:
        print("[OK] El archivo ya está en el Sample Rate correcto.")
        return

    # 2. Calcular cuántas muestras totales tendrá el nuevo archivo
    # Ej: (24000 muestras * 44100) / 16000 = 66150 muestras nuevas
    num_samples_new = int(len(audio_orig) * float(target_sr) / orig_sr)

    # 3. Aplicar el resampleo poligásico de alta calidad
    # resample_poly es excelente para audio porque aplica un filtro anti-aliasing automático
    audio_resampled = signal.resample_poly(audio_orig, target_sr, orig_sr)

    # 4. Normalizar para evitar que la convolución explote de volumen en C++
    max_val = np.max(np.abs(audio_resampled))
    if max_val > 0:
        audio_resampled = audio_resampled / max_val

    # 5. Guardarlo en formato float32 (el mejor formato para juce::dsp::Convolution)
    wav.write(output_file, target_sr, audio_resampled.astype(np.float32))
    
    print(f"[ÉXITO] Reverb resampleada guardada en: {output_file}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT_WAV = os.path.join(BASE_DIR, "reverb_ir.wav")
    OUTPUT_WAV = os.path.join(BASE_DIR, "reverb_ir_44100.wav")
    
    resample_ir(INPUT_WAV, OUTPUT_WAV)