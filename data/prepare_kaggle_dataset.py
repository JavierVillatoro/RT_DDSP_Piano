"""
Prepara un dataset COMPACTO para entrenar en Kaggle (training/training_kaggle_7.py).

Diferencias respecto a make_tfrecords_poly_2.py:
  * El sustain (CC64) se integra en las notas: si el pedal esta pisado cuando sueltas la tecla,
    la nota sigue "activa" hasta que se levanta el pedal (como en el paper DDSP-Piano, Renault 2023).
    Esto es lo que permite que el modelo aprenda el pedal de verdad.
  * Una nota que se vuelve a tocar (mismo pitch) corta a la anterior en su onset.
  * Maximo sustain por nota (MAX_SUSTAIN_SEC) para no ocupar voces eternamente.
  * 24 voces (P=24; con 16 el 65% de las notas obligaba a robar voz, con 24 el 31%, casi siempre colas de release). Si faltan voces se roba la voz con la tecla ya levantada mas antigua
    (la que ya se ha apagado casi del todo); si ni asi, la nota se marca como "dropped" y
    el entrenamiento no usa los trozos que la contienen.
  * No se parte el audio en chunks: se guarda 1 fichero .npz por pieza (audio int16 a 16 kHz +
    curvas a 250 Hz). Los recortes aleatorios se hacen en Kaggle, al vuelo.

Uso (en tu PC, desde la raiz del repo):
    env\\Scripts\\python data\\prepare_kaggle_dataset.py
Salida: dataset/kaggle_ddsp7/{train,validation,test}/*.npz + manifest.json
"""
import os
import sys
import json
import bisect
import argparse
import numpy as np
import pandas as pd
import librosa
import pretty_midi

SR = 16000
FRAME_RATE = 250
HOP = SR // FRAME_RATE          # 64 muestras por frame
N_VOICES = 24
T_RELEASE_SEC = 1.0             # el pitch sigue 1 s tras acabar la nota (Note Release del paper)
MAX_SUSTAIN_SEC = 6.0           # maximo que el pedal mantiene una nota tras soltar la tecla
FADE_FRAMES = 3                 # fade de 12 ms cuando se roba una voz
PEDAL_THRESHOLD = 64


def pedal_events(midi):
    inst = midi.instruments[0]
    ev = sorted([(c.time, c.value) for c in inst.control_changes if c.number == 64])
    times = [t for t, _ in ev]
    vals = [v for _, v in ev]
    return times, vals


def pedal_is_down(times, vals, t):
    i = bisect.bisect_right(times, t) - 1
    return i >= 0 and vals[i] >= PEDAL_THRESHOLD


def next_pedal_up(times, vals, t, end_of_piece):
    """Primer instante > t en el que el pedal baja de 64 (se levanta)."""
    i = bisect.bisect_right(times, t)
    while i < len(times):
        if vals[i] < PEDAL_THRESHOLD:
            return times[i]
        i += 1
    return end_of_piece


def effective_notes(midi, total_time):
    """Devuelve lista de (start, key_end, eff_end, pitch, velocity) con sustain integrado."""
    times, vals = pedal_events(midi)
    notes = sorted(midi.instruments[0].notes, key=lambda n: (n.start, n.pitch))
    out = []
    for n in notes:
        eff = n.end
        if pedal_is_down(times, vals, n.end):
            eff = max(eff, next_pedal_up(times, vals, n.end, total_time))
        eff = min(eff, n.end + MAX_SUSTAIN_SEC)
        out.append([n.start, n.end, eff, n.pitch, n.velocity])
    # una nota repetida corta a la anterior del mismo pitch en su onset
    by_pitch = {}
    for idx, o in enumerate(out):
        by_pitch.setdefault(o[3], []).append(idx)
    for idxs in by_pitch.values():
        for a, b in zip(idxs[:-1], idxs[1:]):
            if out[b][0] < out[a][2]:
                out[a][2] = max(out[a][0], out[b][0])
                out[a][1] = min(out[a][1], out[a][2])
    return out


def build_controls(midi, total_frames, total_time):
    pitch = np.zeros((N_VOICES, total_frames), dtype=np.uint8)
    vel = np.zeros((N_VOICES, total_frames), dtype=np.uint8)
    dropped = np.zeros(total_frames, dtype=bool)
    free_frame = np.zeros(N_VOICES, dtype=int)     # frame a partir del cual la voz esta libre
    key_up_frame = np.zeros(N_VOICES, dtype=int)   # frame en el que se solto la tecla
    onset_frame = np.zeros(N_VOICES, dtype=int)
    release_f = int(T_RELEASE_SEC * FRAME_RATE)

    for start, key_end, eff_end, p, v in effective_notes(midi, total_time):
        s = int(round(start * FRAME_RATE))
        e = max(s + 1, int(round(eff_end * FRAME_RATE)))
        s = min(s, total_frames - 1)
        e = min(e, total_frames)
        r = min(total_frames, e + release_f)
        if s >= total_frames:
            continue
        voice = -1
        for k in range(N_VOICES):
            if s >= free_frame[k]:
                voice = k
                break
        if voice == -1:
            # robar la voz cuya tecla ya esta levantada y que lleva mas tiempo sonando
            cands = [k for k in range(N_VOICES) if key_up_frame[k] <= s]
            if cands:
                voice = min(cands, key=lambda k: onset_frame[k])
                fs = max(0, s - FADE_FRAMES)
                cur = int(vel[voice, fs])
                if s > fs:
                    vel[voice, fs:s] = np.linspace(cur, 0, s - fs).astype(np.uint8)
                vel[voice, s:] = 0
                pitch[voice, s:] = 0
            else:
                dropped[s:r] = True
                continue
        pitch[voice, s:r] = p
        vel[voice, s:e] = v
        free_frame[voice] = r
        key_up_frame[voice] = int(round(key_end * FRAME_RATE))
        onset_frame[voice] = s
    return pitch, vel, dropped


def pedal_curve(midi, total_frames):
    pedal = np.zeros(total_frames, dtype=np.uint8)
    times, vals = pedal_events(midi)
    if not times:
        return pedal
    idx = 0
    cur = 0
    for f in range(total_frames):
        t = f / FRAME_RATE
        while idx < len(times) and t >= times[idx]:
            cur = vals[idx]
            idx += 1
        pedal[f] = cur
    return pedal


def process_track(audio_path, midi_path):
    audio, _ = librosa.load(audio_path, sr=SR, mono=True)
    total_frames = len(audio) // HOP
    audio = audio[: total_frames * HOP]
    midi = pretty_midi.PrettyMIDI(midi_path)
    total_time = len(audio) / SR
    pitch, vel, dropped = build_controls(midi, total_frames, total_time)
    pedal = pedal_curve(midi, total_frames)
    audio16 = np.clip(np.round(audio * 32767.0), -32768, 32767).astype(np.int16)
    return audio16, pitch, vel, pedal, dropped


def main():
    global MAX_SUSTAIN_SEC
    ap = argparse.ArgumentParser()
    ap.add_argument('--maestro', default=r'dataset/maestro-v3.0.0')
    ap.add_argument('--out', default=r'dataset/kaggle_ddsp7')
    ap.add_argument('--year', type=int, default=2009)
    ap.add_argument('--limit', type=int, default=0, help='solo N piezas por split (pruebas)')
    ap.add_argument('--max-sustain', type=float, default=MAX_SUSTAIN_SEC, help='maximo sustain por nota con pedal (s)')
    args = ap.parse_args()
    MAX_SUSTAIN_SEC = args.max_sustain

    df = pd.read_csv(os.path.join(args.maestro, 'maestro-v3.0.0.csv'))
    manifest = {
        'version': 2, 'sample_rate': SR, 'frame_rate': FRAME_RATE, 'n_voices': N_VOICES,
        't_release_sec': T_RELEASE_SEC, 'max_sustain_sec': args.max_sustain,
        'fade_frames': FADE_FRAMES, 'year': args.year, 'splits': {},
    }
    for split in ['train', 'validation', 'test']:
        sub = df[(df['year'] == args.year) & (df['split'] == split)]
        if args.limit:
            sub = sub.head(args.limit)
        os.makedirs(os.path.join(args.out, split), exist_ok=True)
        names = []
        for i, (_, row) in enumerate(sub.iterrows()):
            a = os.path.normpath(os.path.join(args.maestro, row['audio_filename']))
            m = os.path.normpath(os.path.join(args.maestro, row['midi_filename']))
            if not os.path.exists(m) and os.path.exists(m[:-1]):
                m = m[:-1]
            name = f"{split}_{i:04d}.npz"
            dest = os.path.join(args.out, split, name)
            try:
                audio, pitch, vel, pedal, dropped = process_track(a, m)
            except Exception as e:  # noqa
                print(f"[ERROR] {a}: {e}")
                continue
            np.savez(dest, audio=audio, pitch=pitch, vel=vel, pedal=pedal, dropped=dropped)
            names.append(name)
            print(f"[{split} {i + 1}/{len(sub)}] {name}  {len(audio) / SR / 60:.1f} min  "
                  f"dropped={dropped.mean() * 100:.1f}%  pedal_down={(pedal >= 64).mean() * 100:.0f}%", flush=True)
        manifest['splits'][split] = names
    with open(os.path.join(args.out, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    print('[OK] Dataset listo en', args.out)


if __name__ == '__main__':
    sys.exit(main())
