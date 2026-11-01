"""
Inferencia en Python para los modelos de training_kaggle_7 (semantica v2: el pedal alarga las notas).
Usa EXACTAMENTE el mismo preproceso que el entrenamiento (data/prepare_kaggle_dataset.py), asi que es la
referencia para comparar con el motor C++ y para escuchar el modelo sin abrir un DAW.

    python inference_07.py --midi Mompou.mid --h5 checkpoints_ddsp7 --seconds 30 --out resultado_v7
        --wet        aplica la reverb aprendida (por defecto sale SECO, como el motor C++ sin reverb)
        --events     escribe <out>_events.txt para ddsp_render (comparacion)
"""
import os
import sys
import argparse
import json
import numpy as np
import tensorflow as tf
import pretty_midi
import scipy.io.wavfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'data'))
import prepare_kaggle_dataset as prep   # noqa: E402
import inference_06 as inf              # noqa: E402  (clases ContextNetwork, DDSPCore, TrainableReverb)

SR, FR, HOP = 16000, 250, 64


class PianoV2(tf.keras.Model):
    def __init__(self, n_voices, n_strings=1, detune_cents=(0.0, 0.0)):
        super().__init__()
        self.P, self.K = n_voices, inf.CONFIG['model']['n_harmonics']
        self.n_strings, self.detune_cents = n_strings, detune_cents
        self.context_net = inf.ContextNetwork()
        self.core = inf.DDSPCore(inf.CONFIG)
        self.reverb = inf.TrainableReverb()
        self.noise = inf.FilteredNoiseSynthesizerDynamic()

    def harmonic(self, f_k, amp, harm):
        n = tf.shape(f_k)[1] * HOP
        amp_up = inf.upsample_controls_dynamic(amp, HOP)
        harm_up = inf.upsample_controls_dynamic(harm, HOP)
        audio = 0.0
        for s in range(self.n_strings):
            f = f_k * (2.0 ** (self.detune_cents[s] / 1200.0))
            f_up = inf.upsample_controls_dynamic(f, HOP)
            mask = tf.cast(f_up < SR / 2.0, tf.float32)
            ph = tf.math.cumsum(f_up * (2.0 * np.pi / SR), axis=1)
            audio += tf.reduce_sum(tf.sin(ph) * harm_up * mask, axis=-1)
        return audio * tf.squeeze(amp_up, -1)

    def call(self, pitches, vels, pedal, wet=False):
        P, K = self.P, self.K
        T = pitches.shape[2]
        gv = tf.reduce_sum(vels, axis=1)
        act = tf.cast(vels > 0, tf.float32)
        gp = tf.reduce_sum(pitches / 127.0 * act, axis=1) / (tf.reduce_sum(act, axis=1) + 1e-7)
        c = self.context_net(tf.concat([pedal, gv, gp], -1))
        c_rep = tf.tile(c[:, None], [1, P, 1, 1])
        core_in = tf.reshape(tf.concat([pitches / 127.0, vels, c_rep], -1), [-1, T, 34])
        pf = tf.reshape(pitches, [-1, T, 1])
        prm = self.core(core_in, pf)
        amp = tf.reshape(prm['amplitude'], [-1, P, T, 1])
        harm = tf.reshape(prm['harmonics'], [-1, P, T, K])
        fk = tf.reshape(prm['f_k'], [-1, P, T, K])
        nm = prm['noise_magnitudes'] * tf.cast(pf > 0, tf.float32)
        out = 0.0
        for p in range(P):
            out += self.harmonic(fk[:, p], amp[:, p], harm[:, p])
        out += tf.reduce_sum(tf.reshape(self.noise(nm), [-1, P, T * HOP]), axis=1)
        return self.reverb(out) if wet else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--midi', required=True)
    ap.add_argument('--h5', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seconds', type=float, default=30.0)
    ap.add_argument('--start', type=float, default=0.0)
    ap.add_argument('--wet', action='store_true')
    ap.add_argument('--events', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--chunk', type=float, default=10.0, help='segundos por bloque')
    ap.add_argument('--warm', type=float, default=1.5, help='segundos de calentamiento por bloque (se descartan)')
    args = ap.parse_args()
    tf.random.set_seed(args.seed)

    meta_path = os.path.join(args.h5, 'training_meta.json')
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    P = int(meta.get('n_voices_train', prep.N_VOICES))
    prep.N_VOICES = P
    prep.MAX_SUSTAIN_SEC = float(meta.get('max_sustain_sec', prep.MAX_SUSTAIN_SEC))

    midi = pretty_midi.PrettyMIDI(args.midi)
    total_time = midi.get_end_time() + prep.T_RELEASE_SEC + 1.0
    total_frames = int(total_time * FR) + 1
    pitch, vel, dropped = prep.build_controls(midi, total_frames, total_time)
    pedal = prep.pedal_curve(midi, total_frames)
    print(f"[INFO] voces={P}  frames ssin voz libre (notas perdidas): {dropped.mean() * 100:.1f}%")

    f0 = int(args.start * FR)
    f1 = min(total_frames, f0 + int((args.seconds + prep.T_RELEASE_SEC) * FR))
    pitch, vel, pedal = pitch[:, f0:f1], vel[:, f0:f1], pedal[f0:f1]

    model = PianoV2(P, int(meta.get('n_strings', 1)), tuple(meta.get('string_detune_cents', [0.0, 0.0])) + (0.0, 0.0))
    _ = model(tf.zeros((1, P, 10, 1)), tf.zeros((1, P, 10, 1)), tf.zeros((1, 10, 1)), wet=True)   # wet=True construye tambien la reverb
    model.core.load_weights(os.path.join(args.h5, 'core.weights.h5'))
    model.context_net.load_weights(os.path.join(args.h5, 'context.weights.h5'))
    model.reverb.load_weights(os.path.join(args.h5, 'reverb.weights.h5'))

    n_frames = pitch.shape[1]
    ch, wm = int(args.chunk * FR), int(args.warm * FR)
    pieces = []
    s = 0
    while s < n_frames:
        a = max(0, s - wm)
        e = min(n_frames, s + ch)
        p_ = tf.convert_to_tensor(pitch[None, :, a:e, None].astype(np.float32))
        v_ = tf.convert_to_tensor((vel[None, :, a:e, None] / 127.0).astype(np.float32))
        d_ = tf.convert_to_tensor((pedal[None, a:e, None] / 127.0).astype(np.float32))
        y = model(p_, v_, d_, wet=args.wet).numpy()[0]
        pieces.append(y[(s - a) * HOP:])
        print(f"  bloque {s / FR:6.1f}s - {e / FR:6.1f}s", flush=True)
        s = e
    audio = np.concatenate(pieces).astype(np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    scipy.io.wavfile.write(args.out + '.wav', SR, audio)
    print(f"[OK] {args.out}.wav  ({len(audio) / SR:.1f} s, pico={np.abs(audio).max():.3f}, rms={np.sqrt((audio ** 2).mean()):.4f})")

    if args.events:
        lines = []
        t0, t1 = args.start, args.start + args.seconds
        for n in midi.instruments[0].notes:
            if t0 <= n.start < t1:
                lines.append((n.start - t0, f"on {n.start - t0:.6f} {n.pitch} {n.velocity / 127.0:.6f}"))
            if t0 <= n.end < t1 and n.start >= t0:
                lines.append((n.end - t0, f"off {n.end - t0:.6f} {n.pitch}"))
        for c in midi.instruments[0].control_changes:
            if c.number == 64 and t0 <= c.time < t1:
                lines.append((c.time - t0, f"ped {c.time - t0:.6f} {c.value / 127.0:.6f}"))
        lines.sort(key=lambda x: x[0])
        with open(args.out + '_events.txt', 'w') as f:
            f.write('\n'.join(l for _, l in lines) + '\n')
            f.write(f"end {len(audio) / SR:.6f}\n")


if __name__ == '__main__':
    main()
