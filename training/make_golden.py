"""
Prueba de oro: guarda las salidas EXACTAS de Keras para una secuencia fija, para que el motor C++
(RTNeural) demuestre que reproduce la misma red.   ->  ddsp_render golden golden.bin ddsp_weights.bin

    python make_golden.py --h5 checkpoints_ddsp7 --out ../../DDSPiano/Resources/golden.bin
    python make_golden.py --h5 checkpoints_descargados_06_2 --out ../../DDSPiano/Resources/golden_v1.bin

Usa las clases de inference_06.py (misma arquitectura que training_kaggle_6/7).
"""
import os
import sys
import struct
import argparse
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference_06 import ContextNetwork, DDSPCore, CONFIG  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--frames', type=int, default=500)
    args = ap.parse_args()

    ctx, core = ContextNetwork(), DDSPCore(CONFIG)
    _ = ctx(tf.zeros((1, 10, 3)))
    _ = core(tf.zeros((1, 10, 34)), tf.zeros((1, 10, 1)))
    core.load_weights(os.path.join(args.h5, 'core.weights.h5'))
    ctx.load_weights(os.path.join(args.h5, 'context.weights.h5'))

    rng = np.random.default_rng(1234)
    F = args.frames
    t = np.arange(F)
    pedal = ((t // 120) % 2).astype(np.float32)                                   # pedal on/off
    gvel = np.abs(np.sin(t / 37.0)).astype(np.float32) * 1.5 + rng.random(F).astype(np.float32) * 0.2
    gpitch = (0.4 + 0.2 * np.sin(t / 53.0)).astype(np.float32)
    ctx_in = np.stack([pedal, gvel, gpitch], axis=-1).astype(np.float32)           # [F,3]
    ctx_out = ctx(ctx_in[None]).numpy()[0]                                         # [F,32]

    pitch = np.where(t < 20, 0.0, 60.0 + 12.0 * np.sin(t / 91.0)).astype(np.float32)
    vel = np.where((t > 20) & (t < 300), 0.6 + 0.3 * np.sin(t / 17.0), 0.0).astype(np.float32)
    core_in = np.concatenate([(pitch / 127.0)[:, None], vel[:, None], ctx_out], axis=-1).astype(np.float32)  # [F,34]

    x = core.leaky_in(core.dense_in(core_in[None]))
    x = core.leaky_hidden(core.dense_hidden(core.gru(x)))
    raw = tf.concat([core.amp_out(x), core.harm_out(x), core.noise_out(x)], axis=-1).numpy()[0]   # [F,161]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'wb') as f:
        f.write(b'GOLD' + struct.pack('<I', F))
        for a in (ctx_in, ctx_out, core_in, raw):
            f.write(np.ascontiguousarray(a, dtype='<f4').tobytes())
    print(f'[OK] {args.out}  ({F} frames; ctx_out |medio|={np.abs(ctx_out).mean():.3f}, raw |medio|={np.abs(raw).mean():.3f})')


if __name__ == '__main__':
    main()
