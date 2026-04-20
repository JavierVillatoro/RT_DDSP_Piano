import tensorflow as tf
# Asumiendo que configuras el path correctamente o usas módulos
# from configs import config
# from data.pipeline import Maestro2009Pipeline
# from models.core import DDSPMonophonicPiano
# from models.synthesis import harmonic_synth, filtered_noise_synth
# from training.loss import multiresolution_spectral_loss

def train_distributed():
    # 1. Inicializar la Estrategia para las 2 GPUs T4
    strategy = tf.distribute.MirroredStrategy()
    print(f"GPUs disponibles: {strategy.num_replicas_in_sync}")

    # 2. Configuración de Batches
    # El batch se dividirá entre las GPUs. Si quieres 8 elementos por GPU, el global es 16.
    per_replica_batch_size = 8
    global_batch_size = per_replica_batch_size * strategy.num_replicas_in_sync

    # 3. Cargar el Dataset (Pipeline Dummy para el ejemplo)
    # pipeline = Maestro2009Pipeline(...)
    # dataset = pipeline.create_tf_dataset()
    # dataset = dataset.batch(global_batch_size, drop_remainder=True)
    
    # Dummy Dataset para compilación
    def dummy_gen():
        yield (tf.zeros((100, 1)), tf.zeros((100, 1)), tf.zeros((4000,))) # 100 frames, 4000 samples
    
    dataset = tf.data.Dataset.from_generator(dummy_gen, output_types=(tf.float32, tf.float32, tf.float32))
    dataset = dataset.repeat().batch(global_batch_size)
    
    # DISTRIBUIR EL DATASET
    dist_dataset = strategy.experimental_distribute_dataset(dataset)

    # 4. Instanciar Modelo y Optimizador DENTRO DEL SCOPE
    # Esto asegura que los pesos y variables se repliquen en las 2 GPUs.
    with strategy.scope():
        model = DDSPMonophonicPiano()
        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)

        # La pérdida debe reducirse manualmente basándose en el batch global
        def compute_loss(labels, predictions):
            per_example_loss = multiresolution_spectral_loss(labels, predictions)
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)

    # 5. Paso de Entrenamiento Distribuido
    with strategy.scope():
        @tf.function
        def train_step(pitch, velocity, audio_real):
            """Ejecutado por cada GPU en su porción de los datos."""
            with tf.GradientTape() as tape:
                # Flujo 1: Red Neuronal
                amp, harm, noise_mag = model(pitch, velocity, training=True)
                
                # Flujo 2: Módulos de Síntesis (Audio DSP)
                audio_harm = harmonic_synth(pitch, amp, harm)
                audio_noise = filtered_noise_synth(noise_mag)
                audio_pred = audio_harm + audio_noise
                
                # Flujo 3: Cálculo de la pérdida
                loss = compute_loss(audio_real, audio_pred)
            
            # Flujo 4: Gradientes
            gradients = tape.gradient(loss, model.trainable_variables)
            # Clip gradients para la GRU
            gradients, _ = tf.clip_by_global_norm(gradients, 3.0)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            return loss

        @tf.function
        def distributed_train_step(dist_inputs):
            """Distribuye el step y sincroniza los resultados."""
            pitch, velocity, audio_real = dist_inputs
            per_replica_losses = strategy.run(train_step, args=(pitch, velocity, audio_real))
            # Suma de las pérdidas de ambas GPUs
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    # 6. Bucle Principal
    epochs = 100
    steps_per_epoch = 500 # Ajustar según tamaño del dataset

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        for dist_inputs in dist_dataset.take(steps_per_epoch):
            total_loss += distributed_train_step(dist_inputs)
            num_batches += 1
            
        train_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1} | Loss: {train_loss:.4f}")

if __name__ == "__main__":
    train_distributed()