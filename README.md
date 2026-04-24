# DDSP-Piano: Síntesis de Piano mediante Modelado Físico Neuronal

Este proyecto consiste en el desarrollo de un sintetizador de piano polifónico de alta fidelidad basado en la arquitectura **DDSP (Differentiable Digital Signal Processing)**. El objetivo principal es lograr una síntesis realista que capture las sutilezas tímbricas y físicas de un piano de cola, optimizando el modelo para su ejecución en tiempo real.

##  Inspiración y Referencias
Este trabajo se basa y expande la investigación original de Google Magenta:
* **Paper Original:** *Renault, L., Mignot, R., & Roberts, A. (2022). (https://hal.science/hal-04073770v2/file/)DDSP_Piano_JAES_final.pdf
* **Dataset:** [MAESTRO Dataset (v3.0.0)](https://magenta.tensorflow.org/datasets/maestro), utilizando específicamente grabaciones y archivos MIDI del año 2009 para el entrenamiento y validación.

##  Estado del Proyecto y Resultados
El modelo se ha entrenado siguiendo un esquema de aprendizaje jerárquico por fases para garantizar la convergencia y la estabilidad física:

1.  **Fase 1 (Aprendizaje Tímbrico):** Se logró capturar la firma espectral del piano (ataque de martillos, contenido armónico y ruido de fondo). Se alcanzó una pérdida espectral multiresolución de **~6.36**.
2.  **Fase 2 (Ajuste Físico):** Entrenamiento de los parámetros de inarmonicidad y micro-desafinación (*Detuning*). Se ha logrado una integración orgánica de los batimientos acústicos (chorus natural) sin comprometer la afinación fundamental.

##  Modificaciones de Ingeniería y Mejoras
Sobre la arquitectura base del paper, se han implementado las siguientes mejoras para resolver problemas críticos detectados durante las pruebas de inferencia:

### 1. Polifonía Escalable (Dynamic Voice Allocation)
* **Problema:** El entrenamiento se limitó a 8 voces por restricciones de VRAM. En obras complejas (ej. *Ondine* de Ravel), el sistema sufría de "robo de notas" y cortes abruptos.
* **Solución:** Se ha reestructurado el orquestador polifónico mediante pesos compartidos para permitir la inferencia de hasta **24 voces simultáneas**. Esto elimina los artefactos de corte y permite interpretar piezas de alta densidad polifónica con total fidelidad.

### 2. Estabilización del Detuner (Control de Afinación)
* **Problema:** El modelo original presentaba inestabilidades que derivaban en afinaciones caóticas (sonido "honky-tonk").
* **Solución:** Se ha aplicado un "blindaje físico" mediante:
    * **Inicialización estricta a cero:** Forzado de pesos en capas densas para asegurar que el piano nace perfectamente afinado.
    * **Restricción de Excursión:** Limitación matemática de la salida del Detuner a **±5 cents** ($\pm 0.05$ semitonos). Esto permite la calidez del batimiento acústico pero impide desafinaciones audibles destructivas.

### 3. Diagnóstico de la Amnesia RNN (Drift Temporal)
* **Hallazgo:** Durante la inferencia en Python con archivos MIDI largos, se detectó una degradación tímbrica a partir de los 15-20 segundos. 
* **Diagnóstico:** Se ha identificado como un problema de gestión del *Hidden State* en entornos *stateless* (TensorFlow/Python). Al resetearse la memoria de la GRU entre bloques, la red pierde la continuidad física. 
* **Solución para VST:** Este comportamiento se ha identificado como un artefacto del laboratorio que desaparecerá en la fase de tiempo real gracias a la naturaleza *stateful* de RTNeural.

##  Pruebas Realizadas
* **Interpretaciones de prueba:** Se han renderizado con éxito obras complejas como *Mompou* y pasajes de Ravel, validando la estabilidad tímbrica y la correcta asignación de voces.
* **Métrica de éxito:** Consistencia espectral mantenida y ausencia de cortes en la fase de relajación (*note release*) de las cuerdas.

## ⏩ Fase Actual: Implementación Live con RTNeural
Tras validar la calidad sonora en el entorno de investigación (Python), el proyecto entra en su fase final de integración:
1.  **Exportación:** Conversión de los pesos entrenados (`.h5`) a formato JSON compatible con la librería **RTNeural**.
2.  **Arquitectura C++:** Implementación del motor de síntesis en un plugin VST utilizando el framework **JUCE**.
3.  **Gestión de Estado:** Aprovechamiento de la persistencia del estado oculto en RTNeural para eliminar la amnesia de la red y permitir una interpretación infinita sin degradación.

---
*Este proyecto se desarrolla como parte de un Trabajo de Fin de Grado (TFG) centrado en la aplicación de Deep Learning al Procesamiento de Señales Digitales (DSP).*