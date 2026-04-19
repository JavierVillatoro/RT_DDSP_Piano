import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class DDSPCore(tf.keras.Model):
    def __init__(self, config):
        super(DDSPCore, self).__init__()
        # Extraemos parámetros del config.yaml
        model_cfg = config['model']
        self.n_harmonics = model_cfg['n_harmonics']
        self.n_noise_filters = model_cfg['n_noise_filters']
        
        # Capa 1: Dense + Leaky ReLU (128 unidades)
        self.dense_in = layers.Dense(model_cfg['hidden_size'])
        self.leaky_relu_in = layers.LeakyReLU(alpha=0.2)
        
        # Capa 2: GRU unidireccional (192 unidades)
        # Importante: return_sequences=True para mantener la resolución de 250Hz
        self.gru = layers.GRU(model_cfg['gru_units'], return_sequences=True)
        
        # Capa 3: Dense + Leaky ReLU (192 unidades)
        self.dense_hidden = layers.Dense(model_cfg['dense_output_size'])
        self.leaky_relu_hidden = layers.LeakyReLU(alpha=0.2)
        
        # Capa 4: Proyección Lineal Final (161 unidades)
        self.dense_final = layers.Dense(model_cfg['total_params_output'])

    def modified_sigmoid(self, x):
        """
        Activación estricta del paper original:
        2.0 * sigmoid(x)**(log(10)) + 1e-7
        """
        # np.log(10) es aprox 2.302585
        log_10 = np.log(10.0)
        return 2.0 * tf.pow(tf.nn.sigmoid(x), log_10) + 1e-7

    def call(self, pitch, velocity, training=False):
        # Concatenar entradas: [Batch, Time, 2]
        x = tf.concat([pitch, velocity], axis=-1)
        
        # Forward pass neuronal
        x = self.dense_in(x)
        x = self.leaky_relu_in(x)
        x = self.gru(x, training=training)
        x = self.dense_hidden(x)
        x = self.leaky_relu_hidden(x)
        
        # Salida lineal
        raw_output = self.dense_final(x)
        
        # Aplicar activación sigmoide modificada para asegurar positividad
        # y evitar desvanecimiento de gradiente en amplitudes bajas
        activated_output = self.modified_sigmoid(raw_output)
        
        # Split de los parámetros para los sintetizadores
        # 1. Amplitud Global
        amplitude = activated_output[..., 0:1]
        
        # 2. Distribución Armónica (96 bins)
        harmonics = activated_output[..., 1:1 + self.n_harmonics]
        
        # 3. Magnitudes de Ruido (64 bins)
        noise_magnitudes = activated_output[..., 1 + self.n_harmonics:]
        
        return {
            'amplitude': amplitude,
            'harmonics': harmonics,
            'noise_magnitudes': noise_magnitudes
        }