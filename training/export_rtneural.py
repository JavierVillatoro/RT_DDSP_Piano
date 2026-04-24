import os
import json
import numpy as np
import scipy.io.wavfile
import tensorflow as tf

# ==============================================================================
# 1. CONFIGURACIÓN Y CLASES BASE (Solo lo necesario para cargar pesos)
# ==============================================================================
CONFIG = {
    'audio': {'sample_rate': 16000, 'frame_rate': 250},
    'model': {'n_harmonics': 96, 'n_noise_filters': 64, 'hidden_size': 128, 'gru_units': 192, 'dense_output_size': 192},
}

class ContextNetwork(tf.keras.Model): 
    def __init__(self, context_dim=32, gru_units=64):
        super().__init__()
        self.dense_in = tf.keras.layers.Dense(context_dim, activation=tf.nn.leaky_relu)
        self.gru = tf.keras.layers.GRU(gru_units, return_sequences=True)
        self.dense_out = tf.keras.layers.Dense(context_dim)

    def call(self, inputs):
        return self.dense_out(self.gru(self.dense_in(inputs)))

class Detuner(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(1, kernel_initializer='zeros')

    def build(self, input_shape):
        self.static_detune = self.add_weight(name='static_detune', shape=(1,), initializer='zeros')
        super().build(input_shape)

    def call(self, pitch_midi):
        return tf.nn.tanh(self.dense(pitch_midi / 127.0)) + tf.nn.tanh(self.static_detune)

class TrainableReverb(tf.keras.Model):
    def __init__(self, ir_length=24000, **kwargs):
        super().__init__(**kwargs)
        self.ir_length = ir_length

    def build(self, input_shape):
        np_ir_init = np.random.normal(size=self.ir_length) * np.exp(-np.linspace(0.0, 5.0, self.ir_length))
        self.ir = self.add_weight(name="impulse_response", shape=[self.ir_length], initializer=tf.constant_initializer(np_ir_init))
        super().build(input_shape)

    def call(self, audio):
        pass # No hace falta para exportar

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
        pass # No hace falta para exportar

class DDSPCore(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        cfg = config['model']
        self.dense_in = tf.keras.layers.Dense(cfg['hidden_size'])
        self.gru = tf.keras.layers.GRU(cfg['gru_units'], return_sequences=True)
        self.dense_hidden = tf.keras.layers.Dense(cfg['dense_output_size'])
        self.inharmonicity = InharmonicityModel()
        self.amp_out = tf.keras.layers.Dense(1)
        self.harm_out = tf.keras.layers.Dense(cfg['n_harmonics'])
        self.noise_out = tf.keras.layers.Dense(cfg['n_noise_filters'])

    def call(self, core_input, pitch_midi):
        return core_input # Solo necesitamos la estructura para cargar pesos

# ==============================================================================
# 2. LÓGICA DE EXPORTACIÓN
# ==============================================================================
def export_layer_to_dict(layer, layer_type="dense"):
    weights = layer.get_weights()
    if not weights: return {}
        
    export_data = {}
    if layer_type == "dense":
        w, b = weights
        export_data["kernel"] = w.flatten().tolist()
        export_data["bias"] = b.flatten().tolist()
        export_data["shape_in"] = w.shape[0]
        export_data["shape_out"] = w.shape[1]
        
    elif layer_type == "gru":
        w, u, b = weights
        export_data["kernel"] = w.flatten().tolist()
        export_data["recurrent_kernel"] = u.flatten().tolist()
        export_data["bias"] = b.flatten().tolist()
        export_data["units"] = u.shape[0]
        
    return export_data

def export_to_rtneural(weights_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    core = DDSPCore(CONFIG)
    context = ContextNetwork()
    detuner = Detuner()
    reverb = TrainableReverb()

    # Dummy pass para inicializar la memoria de los tensores
    _ = core(tf.zeros((1, 10, 34)), tf.zeros((1, 10, 1)))
    _ = context(tf.zeros((1, 10, 3)))
    _ = detuner(tf.zeros((1, 10, 1)))
    _ = reverb(tf.zeros((1, 16000)))

    print(f"[1/4] Cargando pesos .h5 desde {weights_folder}...")
    core.load_weights(os.path.join(weights_folder, "core.weights.h5"))
    context.load_weights(os.path.join(weights_folder, "context.weights.h5"))
    detuner.load_weights(os.path.join(weights_folder, "detuner.weights.h5"))
    reverb.load_weights(os.path.join(weights_folder, "reverb.weights.h5"))

    print("[2/4] Exportando Reverb a WAV...")
    ir_weights = reverb.get_weights()[0]
    scipy.io.wavfile.write(os.path.join(output_folder, "reverb_ir.wav"), 16000, ir_weights)

    print("[3/4] Exportando DDSPCore y ContextNetwork a JSON...")
    core_data = {
        "dense_in": export_layer_to_dict(core.dense_in, "dense"),
        "gru": export_layer_to_dict(core.gru, "gru"),
        "dense_hidden": export_layer_to_dict(core.dense_hidden, "dense"),
        "amp_out": export_layer_to_dict(core.amp_out, "dense"),
        "harm_out": export_layer_to_dict(core.harm_out, "dense"),
        "noise_out": export_layer_to_dict(core.noise_out, "dense"),
    }
    with open(os.path.join(output_folder, "ddsp_core_weights.json"), "w") as f:
        json.dump(core_data, f)
        
    context_data = {
        "dense_in": export_layer_to_dict(context.dense_in, "dense"),
        "gru": export_layer_to_dict(context.gru, "gru"),
        "dense_out": export_layer_to_dict(context.dense_out, "dense"),
    }
    with open(os.path.join(output_folder, "context_weights.json"), "w") as f:
        json.dump(context_data, f)

    print("[4/4] Exportando Parámetros Físicos (Detuner & Inharmonicity)...")
    detuner_data = {
        "dense": export_layer_to_dict(detuner.dense, "dense"),
        "static_detune": float(detuner.static_detune.numpy()[0])
    }
    with open(os.path.join(output_folder, "detuner_weights.json"), "w") as f:
        json.dump(detuner_data, f)
        
    # NUEVO: Exportamos los parámetros físicos del acero
    inharm_data = {
        "alpha_B": float(core.inharmonicity.alpha_B.numpy()),
        "beta_B": float(core.inharmonicity.beta_B.numpy()),
        "alpha_T": float(core.inharmonicity.alpha_T.numpy()),
        "beta_T": float(core.inharmonicity.beta_T.numpy())
    }
    with open(os.path.join(output_folder, "inharmonicity_weights.json"), "w") as f:
        json.dump(inharm_data, f)

    print(f"\n[ÉXITO] ¡Exportación completada en: {output_folder}!")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CARPETA_H5 = os.path.join(BASE_DIR, "checkpoints_descargados_06_2") 
    CARPETA_EXPORTACION = os.path.join(BASE_DIR, "rtneural_export")
    
    export_to_rtneural(CARPETA_H5, CARPETA_EXPORTACION)