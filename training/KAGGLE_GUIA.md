# Guía Kaggle: reentrenar DDSP-Piano con pedal real y notas largas (`training_kaggle_8.py`)

> **v8 (notas largas):** recortes de 10 s (7 s de calentamiento sin loss + 3 s con loss) y sustain máximo de 10 s,
> para que la red vea graves con pedal de hasta 10 s. Cada época es más lenta que en la v7 (batch 4 = 2 por GPU, 300 pasos; con batch 8 da OOM en las T4).
> Si aun así da OOM: `global_batch_size` a 2 y/o `WARMUP_SEC = 5.0`.

## 1. Qué subir (solo UN fichero)

`dataset/ddsp8_kaggle_data.zip` (2,2 GB) — ya generado por `data/prepare_kaggle_dataset.py`.
Contiene:

| Dentro del zip | Qué es |
|---|---|
| `manifest.json` | parámetros del dataset (24 voces, release 1 s, máx. sustain 10 s...) |
| `train/ validation/ test/*.npz` | 1 fichero por pieza MAESTRO 2009: audio 16 kHz int16 + curvas a 250 Hz (pitch, velocity, pedal) con el **sustain ya integrado en las notas** |
| `weights_init/` | tus pesos actuales (`core`, `context`, `reverb` .h5) para arrancar desde ahí en vez de desde cero |

**No subas**: `dataset/tfrecords_*` (13 GB cada uno, ya no se usan), `dataset/maestro-v3.0.0`, `env/`, `old_*`.

Si algún día cambias algo en `data/prepare_kaggle_dataset.py`, regenera con
`env\Scripts\python data\prepare_kaggle_dataset.py` y vuelve a comprimir `dataset/kaggle_ddsp8`
(el zip debe tener `manifest.json` en la raíz).

## 2. Pasos en Kaggle

1. **Datasets → New Dataset**, arrastra `ddsp8_kaggle_data.zip` (Kaggle lo descomprime solo). Nombre: `ddsp8-data`. Déjalo privado.
2. **Code → New Notebook** → *Add Input* → tu dataset `ddsp8-data`.
3. *Session options*: **Accelerator = GPU T4 ×2**, Persistence = *Files only*.
4. Pega **todo** `training/training_kaggle_8.py` en una única celda (igual que con Fase1/Fase2).
5. Pulsa **Save Version → Save & Run All (Commit)**: corre en segundo plano aunque cierres el navegador (máx. 12 h; el script se para solo a las 11 h y deja los pesos empaquetados).
6. Al acabar, en la pestaña *Output* descarga:
   * `pesos_ddsp8.zip` → **mejores pesos** (los que usarás en el plugin)
   * `pesos_ddsp8_last.zip` → últimos pesos (para continuar entrenando)
   * `loss_curve.png`

Cuota: Kaggle da ~30 h de GPU a la semana. Mide el tiempo de la primera época (aparece en la barra tqdm): `150 batches/época`.

## 3. Continuar entrenando en otra sesión

1. Sube `pesos_ddsp8_last.zip` como dataset nuevo (`ddsp8-run1`) y añádelo al notebook.
2. En la celda cambia `RESUME_FROM = '/kaggle/input/ddsp8-run1'` (comprueba la ruta con `!ls /kaggle/input/ddsp8-run1`; debe verse `core.weights.h5`).
3. Ejecuta de nuevo. La LR se reinicia a 4e-4 y se reduce sola con los plateaus.

## 4. Qué esperar / si algo falla

* La *Val loss* **no es comparable** con la de la Fase 1 (6,0): ahora los recortes tienen más sustain/pedal y hay 16 voces. Lo que importa es que baje y que el audio suene bien.
* Primera época con los pesos antiguos: sobre 7–9 (es normal; el modelo aún no conoce el sustain).
* `ResourceExhausted / OOM`: baja `global_batch_size` a 6 (o 4) en `CONFIG`. Si da error con `tf.recompute_grad`, pon `USE_RECOMPUTE = False` **y** baja el batch a 4–6.
* Para cambiar a 2 cuerdas con desafinado fijo (efecto "chorus"): `N_STRINGS = 2` (el plugin lo leerá del `training_meta.json`).

## 5. Después del entrenamiento (en tu PC)

```
cd training
# 1) descomprime pesos_ddsp8.zip en training/checkpoints_ddsp8/
# 2) genera los recursos del plugin (pesos en binario + IR de la reverb)
..\env\Scripts\python export_binary.py --h5 checkpoints_ddsp8 --out ..\..\DDSPiano\Resources
# 3) compara el motor C++ con Python (ver DDSPiano/Tools/README.md)
```
