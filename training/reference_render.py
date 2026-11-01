"""
Render de REFERENCIA en Python (modelo Keras, SIN reverb) + lista de eventos para el motor C++.
Sirve para comprobar que el plugin suena igual que el modelo entrenado (ver compare_renders.py).

    python reference_render.py --midi Robert.mid --h5 checkpoints_descargados_06_2 --seconds 20 --out tmp/ref
        -> tmp/ref.wav (16 kHz, seco)   tmp/ref_events.txt

Solo para pesos de la semantica v1 (training_kaggle_6). Para v2 usa --semantics 2: las notas se alargan
con el pedal igual que en data/prepare_kaggle_dataset.py.
"""
import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import pretty_midi
import scipy.io.wavfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inference_06 as inf  # noqa: E402


def write_events(midi, seconds, path, semantics):
    inst = midi.instruments[0]
    lines = []
    for n in inst.notes:
        if n.start < seconds:
            lines.append((n.start, f"on {n.start:.6f} {n.pitch} {n.velocity / 127.0:.6f}"))
            if n.end < seconds:
                lines.append((n.end, f"off {n.end:.6f} {n.pitch}"))
    for c in inst.control_changes:
        if c.number == 64 and c.time < seconds:
            lines.append((c.time, f"ped {c.time:.6f} {c.value / 127.0:.6f}"))
    lines.sort(key=lambda x: x[0])
    with open(path, 'w') as f:
        for _, l in lines:
            f.write(l + '\n')
        f.write(f"end {seconds + inf.T_RELEASE_SEC + 1.0:.6f}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--midi', required=True)
    ap.add_argument('--h5', required=True)
    ap.add_argument('--seconds', type=float, default=20.0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--semantics', type=int, default=1)
    args = ap.parse_args()
    tf.random.set_seed(args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    midi = pretty_midi.PrettyMIDI(args.midi)
    write_events(midi, args.seconds, args.out + '_events.txt', args.semantics)

    p_pitch, p_vel, p_pedal = inf.midi_to_controls(args.midi)
    frames = int(args.seconds * inf.CONFIG['audio']['frame_rate']) + int(inf.T_RELEASE_SEC * 250) + 250
    frames = min(frames, p_pitch.shape[2])
    p_pitch, p_vel, p_pedal = p_pitch[:, :, :frames], p_vel[:, :, :frames], p_pedal[:, :frames]

    model = inf.PolyphonicDDSPPianoDynamic(inf.CONFIG, n_voices=inf.MAX_VOICES)
    _ = model({'pitches': tf.zeros((1, inf.MAX_VOICES, 10, 1)), 'velocities': tf.zeros((1, inf.MAX_VOICES, 10, 1)),
               'pedal': tf.zeros((1, 10, 1))})
    model.core.load_weights(os.path.join(args.h5, 'core.weights.h5'))
    model.context_net.load_weights(os.path.join(args.h5, 'context.weights.h5'))
    model.detuner.load_weights(os.path.join(args.h5, 'detuner.weights.h5'))
    model.reverb = lambda a: a           # render SECO: la reverb se compara aparte

    audio = tf.squeeze(model({'pitches': p_pitch, 'velocities': p_vel, 'pedal': p_pedal})).numpy()
    scipy.io.wavfile.write(args.out + '.wav', 16000, audio.astype(np.float32))
    print(f"[OK] {args.out}.wav  ({len(audio) / 16000:.1f} s, pico={np.abs(audio).max():.3f}, rms={np.sqrt((audio ** 2).mean()):.4f})")


if __name__ == '__main__':
    main()
