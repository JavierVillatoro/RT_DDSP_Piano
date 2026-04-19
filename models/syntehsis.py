import tensorflow as tf
import numpy as np

# ==============================================================================
# 1. FUNCIONES AUXILIARES
# ==============================================================================

def get_modified_sigmoid(x):
    """
    Activación Sigmoide Modificada (Paper DDSP).
    Fuerza que las magnitudes (amplitud, armónicos, ruido) sean positivas.
    
    Tensor Flow (Shapes):
    - Input: [batch, frames, channels]
    - Output: [batch, frames, channels] (Misma forma, dominio [1e-7, ~2.0])
    """
    # math.log(10.0) ≈ 2.302585
    # Aplicamos la fórmula exacta para evitar gradientes muertos cerca de cero
    return 2.0 * tf.math.pow(tf.math.sigmoid(x), tf.math.log(10.0)) + 1e-7

def upsample_controls(controls, factor):
    """
    Escala las señales de control de baja resolución (Frame Rate = 250Hz) 
    a alta resolución (Sample Rate = 16000Hz) usando interpolación bilineal.
    
    Tensor Flow (Shapes):
    - Input: [batch, frames, channels]
    - Output: [batch, frames * factor, channels]
    """
    batch_size = tf.shape(controls)[0]
    n_frames = tf.shape(controls)[1]
    n_channels = tf.shape(controls)[2]
    n_samples = n_frames * factor

    # tf.image.resize requiere tensores 4D: [batch, alto, ancho, canales]
    # Tratamos el tiempo como 'alto' y 1 como 'ancho'
    controls_4d = tf.expand_dims(controls, axis=2) # [batch, frames, 1, channels]
    
    # Interpolación bilineal a través del tiempo
    upsampled_4d = tf.image.resize(
        controls_4d, 
        [n_samples, 1], 
        method=tf.image.ResizeMethod.BILINEAR
    )
    
    # Devolvemos a 3D: [batch, samples, channels]
    return tf.squeeze(upsampled_4d, axis=2)

# ==============================================================================
# 2. SINTETIZADOR ADITIVO (ARMÓNICOS)
# ==============================================================================

class HarmonicSynthesizer:
    def __init__(self, sample_rate=16000, frame_rate=250, n_harmonics=96):
        self.sr = sample_rate
        self.fr = frame_rate
        self.n_harmonics = n_harmonics
        # Factor de upsampling: 16000 / 250 = 64 muestras por frame
        self.hop_size = self.sr // self.fr 

    def __call__(self, f0, amplitude, harmonics):
        """
        Tensor Flow (Shapes):
        - f0: [batch, frames, 1] (Frecuencia fundamental en Hz)
        - amplitude: [batch, frames, 1] (Envolvente de volumen global)
        - harmonics: [batch, frames, 96] (Distribución de energía relativa)
        """
        # 1. Upsampling de los controles [batch, frames, c] -> [batch, samples, c]
        f0_up = upsample_controls(f0, self.hop_size)
        amp_up = upsample_controls(amplitude, self.hop_size)
        harm_up = upsample_controls(harmonics, self.hop_size)

        # 2. Generar matriz multiplicadora de armónicos [1.0, 2.0, ..., 96.0]
        # Shape: [1, 1, 96]
        harmonic_multipliers = tf.range(1, self.n_harmonics + 1, dtype=tf.float32)
        harmonic_multipliers = tf.reshape(harmonic_multipliers, [1, 1, self.n_harmonics])

        # 3. Frecuencias instantáneas para cada armónico
        # f0_up * multiplicador = [batch, samples, 1] * [1, 1, 96] -> [batch, samples, 96] (Broadcasting)
        harmonic_freqs = f0_up * harmonic_multipliers

        # (Filtro Antialiasing: Silenciamos armónicos que superen la frecuencia de Nyquist)
        nyquist = self.sr / 2.0
        antialiasing_mask = tf.cast(harmonic_freqs < nyquist, tf.float32)

        # 4. Cálculo de la Fase Instantánea (Integración de la frecuencia)
        # Omega = 2 * pi * f / sr
        omega = harmonic_freqs * (2.0 * np.pi / self.sr)
        # La fase es la suma acumulativa de las variaciones angulares (omega) en el tiempo
        phases = tf.math.cumsum(omega, axis=1) # [batch, samples, 96]

        # 5. Generar Osciladores y Sumar
        # Senoides puras multiplicadas por la energía de cada armónico y la máscara antialiasing
        wavs = tf.math.sin(phases) * harm_up * antialiasing_mask # [batch, samples, 96]
        
        # Sumamos todos los armónicos en el último eje
        audio = tf.reduce_sum(wavs, axis=-1, keepdims=True) # [batch, samples, 1]
        
        # Aplicamos la amplitud (volumen) global
        audio = audio * amp_up # [batch, samples, 1]

        return tf.squeeze(audio, axis=-1) # Output: [batch, samples]

# ==============================================================================
# 3. SINTETIZADOR SUSTRACTIVO (RUIDO FILTRADO)
# ==============================================================================

class FilteredNoiseSynthesizer:
    def __init__(self, sample_rate=16000, frame_rate=250, n_noise_filters=64):
        self.sr = sample_rate
        self.fr = frame_rate
        self.n_noise_filters = n_noise_filters
        self.hop_size = self.sr // self.fr # 64 muestras
        # Para un hop de 64, usamos una ventana del doble para un buen overlap (50%)
        self.window_size = self.hop_size * 2 # 128 muestras

    def __call__(self, noise_magnitudes):
        """
        Genera ruido filtrado eficientemente multiplicando en el dominio de la frecuencia.
        
        Tensor Flow (Shapes):
        - noise_magnitudes: [batch, frames, 64]
        """
        batch_size = tf.shape(noise_magnitudes)[0]
        n_frames = tf.shape(noise_magnitudes)[1]
        n_samples = n_frames * self.hop_size

        # 1. Generar ruido blanco uniforme [-1, 1]
        # Shape: [batch, samples]
        white_noise = tf.random.uniform([batch_size, n_samples], minval=-1.0, maxval=1.0)

        # 2. Trocear el ruido en ventanas solapadas (Framing)
        # Shape: [batch, frames, 128]
        noise_frames = tf.signal.frame(
            white_noise, 
            frame_length=self.window_size, 
            frame_step=self.hop_size, 
            pad_end=True
        )

        # 3. Transformada de Fourier del Ruido
        # rfft de tamaño 128 devuelve 65 bins (Nyquist: N/2 + 1)
        # Shape: [batch, frames, 65]
        noise_fft = tf.signal.rfft(noise_frames)

        # 4. Preparar magnitudes predictadas por la red
        # La red predice 64 bins, pero rfft_size 128 requiere 65 bins.
        # Rellenamos (pad) con un 0 al final (alta frecuencia) para cuadrar las formas.
        # noise_magnitudes shape: [batch, frames, 64]
        magnitudes_padded = tf.pad(noise_magnitudes, paddings=[[0, 0], [0, 0], [0, 1]]) # [batch, frames, 65]
        
        # 5. Filtrado en dominio de la frecuencia (Multiplicación compleja)
        # Convertimos magnitudes reales a complejas y multiplicamos
        # Shape: [batch, frames, 65]
        filtered_fft = noise_fft * tf.cast(magnitudes_padded, tf.complex64)

        # 6. Vuelta al dominio del tiempo (Transformada Inversa)
        # irfft nos devuelve frames de tamaño 128
        # Shape: [batch, frames, 128]
        filtered_frames = tf.signal.irfft(filtered_fft)

        # 7. Aplicar ventana (Hann) para suavizar las transiciones
        window = tf.signal.hann_window(self.window_size)
        filtered_frames = filtered_frames * window

        # 8. Reconstruir la señal continua (Overlap and Add)
        # Suma las ventanas solapadas para recuperar [batch, samples]
        audio_noise = tf.signal.overlap_and_add(filtered_frames, frame_step=self.hop_size)

        # Cortamos por si el padding del framing generó muestras extra al final
        return audio_noise[:, :n_samples] 


# ==============================================================================
# 4. BLOQUE DE PRUEBA (Kaggle Ready)
# ==============================================================================
if __name__ == "__main__":
    print("Iniciando prueba de Síntesis DDSP Differentiable...\n")
    
    # Parámetros de simulación
    BATCH_SIZE = 1
    SECONDS = 3.0
    FRAME_RATE = 250
    SAMPLE_RATE = 16000
    FRAMES = int(SECONDS * FRAME_RATE) # 750 frames
    EXPECTED_SAMPLES = int(SECONDS * SAMPLE_RATE) # 48000 muestras

    # Generación de tensores falsos (simulando salida del modelo neuronal)
    # tf.random.normal para simular el raw output antes de la activación
    f0_hz = tf.fill([BATCH_SIZE, FRAMES, 1], 440.0) # Nota A4 constante
    raw_amp = tf.random.normal([BATCH_SIZE, FRAMES, 1])
    raw_harmonics = tf.random.normal([BATCH_SIZE, FRAMES, 96])
    raw_noise = tf.random.normal([BATCH_SIZE, FRAMES, 64])

    # Aplicamos la Sigmoide Modificada
    amp = get_modified_sigmoid(raw_amp)
    harmonics = get_modified_sigmoid(raw_harmonics)
    noise_mags = get_modified_sigmoid(raw_noise)
    
    # Normalizamos los armónicos (opcional, pero buena práctica)
    harmonics = harmonics / tf.reduce_sum(harmonics, axis=-1, keepdims=True)

    print(f"Forma de F0 de entrada: {f0_hz.shape}")
    print(f"Forma de Amplitud: {amp.shape}")
    print(f"Forma de Armónicos: {harmonics.shape}")
    print(f"Forma de Magnitudes de Ruido: {noise_mags.shape}\n")

    # Instanciamos los sintetizadores
    harmonic_synth = HarmonicSynthesizer(sample_rate=SAMPLE_RATE, frame_rate=FRAME_RATE)
    noise_synth = FilteredNoiseSynthesizer(sample_rate=SAMPLE_RATE, frame_rate=FRAME_RATE)

    # Flujo de ejecución
    audio_harm = harmonic_synth(f0=f0_hz, amplitude=amp, harmonics=harmonics)
    audio_noise = noise_synth(noise_magnitudes=noise_mags)
    
    # Audio final
    audio_final = audio_harm + audio_noise

    print("--- RESULTADOS DE SÍNTESIS ---")
    print(f"Audio Armónico Tensor Shape: {audio_harm.shape}")
    print(f"Audio Ruido Tensor Shape: {audio_noise.shape}")
    print(f"Audio Final Tensor Shape: {audio_final.shape}")
    
    # Verificación del tamaño final
    assert audio_final.shape[1] == EXPECTED_SAMPLES, \
        f"Error: Se esperaban {EXPECTED_SAMPLES} muestras, se obtuvieron {audio_final.shape[1]}"
    print(f"¡Éxito! 3 segundos a 16kHz equivalen a {audio_final.shape[1]} muestras exactas.")