import tensorflow as tf

class MultiResolutionSpectralLoss(tf.keras.losses.Loss):
    def __init__(self, config, name="multi_resolution_spectral_loss"):
        super().__init__(name=name)
        self.loss_cfg = config['loss']
        self.fft_sizes = self.loss_cfg['fft_sizes']
        self.hop_ratio = self.loss_cfg['hop_ratio']

    def spectral_loss(self, y_true, y_pred, fft_size):
        """Cálculo de pérdida L1 para un tamaño de ventana específico."""
        hop_size = int(fft_size * self.hop_ratio)
        
        # Calcular STFT
        stft_true = tf.signal.stft(y_true, frame_length=fft_size, frame_step=hop_size)
        stft_pred = tf.signal.stft(y_pred, frame_length=fft_size, frame_step=hop_size)
        
        # Magnitud lineal
        mag_true = tf.abs(stft_true)
        mag_pred = tf.abs(stft_pred)
        
        # Log-magnitud (con epsilon para estabilidad)
        log_mag_true = tf.math.log(mag_true + 1e-7)
        log_mag_pred = tf.math.log(mag_pred + 1e-7)
        
        # Distancia L1
        lin_loss = tf.reduce_mean(tf.abs(mag_true - mag_pred))
        log_loss = tf.reduce_mean(tf.abs(log_mag_true - log_mag_pred))
        
        return lin_loss + log_loss

    def call(self, y_true, y_pred):
        # Aseguramos que el audio tenga forma (Batch, Samples)
        if len(y_true.shape) == 3:
            y_true = tf.squeeze(y_true, axis=-1)
        if len(y_pred.shape) == 3:
            y_pred = tf.squeeze(y_pred, axis=-1)
            
        total_loss = 0.0
        for size in self.fft_sizes:
            total_loss += self.spectral_loss(y_true, y_pred, size)
            
        return total_loss