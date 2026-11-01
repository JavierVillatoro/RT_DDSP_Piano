"""
Empaqueta los pesos del modelo en UN binario compacto (float32) para embeberlo en el plugin
(juce_add_binary_data) + la IR de la reverb. Sustituye a los JSON de 5 MB y a la ruta absoluta C:/...

Dos entradas posibles:
  --h5   carpeta con core/context/reverb .weights.h5 (+ training_meta.json)   [entrenamientos nuevos]
  --json carpeta rtneural_export/ ya generada por export_rtneural.py          [pesos antiguos, semantica v1]

Salida (carpeta --out):  ddsp_weights.bin   reverb_ir.wav

Formato del .bin (little endian):
  'DDSW' | u32 version=1 | u32 n_tensores
  por tensor: u16 len(nombre) | nombre | u8 ndim | u32 dims[ndim] | float32 datos (row-major)
Los kernels Dense/GRU van tal cual los guarda Keras: [entradas, salidas]; GRU bias [2, 3*unidades] (reset_after).
"""
import os
import sys
import json
import shutil
import struct
import argparse
import tempfile
import numpy as np


def load_json_dir(d):
    def rd(name):
        with open(os.path.join(d, name)) as f:
            return json.load(f)
    core, ctx, det, inh = rd('ddsp_core_weights.json'), rd('context_weights.json'), rd('detuner_weights.json') if os.path.exists(os.path.join(d, 'detuner_weights.json')) else None, rd('inharmonicity_weights.json')

    def dense(j, i, o):
        return (np.array(j['kernel'], np.float32).reshape(i, o), np.array(j['bias'], np.float32).reshape(o))

    def gru(j, i, u):
        return (np.array(j['kernel'], np.float32).reshape(i, 3 * u),
                np.array(j['recurrent_kernel'], np.float32).reshape(u, 3 * u),
                np.array(j['bias'], np.float32).reshape(2, 3 * u))
    t = {}
    t['ctx.dense_in.w'], t['ctx.dense_in.b'] = dense(ctx['dense_in'], 3, 32)
    t['ctx.gru.w'], t['ctx.gru.u'], t['ctx.gru.b'] = gru(ctx['gru'], 32, 64)
    t['ctx.dense_out.w'], t['ctx.dense_out.b'] = dense(ctx['dense_out'], 64, 32)
    t['core.dense_in.w'], t['core.dense_in.b'] = dense(core['dense_in'], 34, 128)
    t['core.gru.w'], t['core.gru.u'], t['core.gru.b'] = gru(core['gru'], 128, 192)
    t['core.dense_hidden.w'], t['core.dense_hidden.b'] = dense(core['dense_hidden'], 192, 192)
    wa, ba = dense(core['amp_out'], 192, 1)
    wh, bh = dense(core['harm_out'], 192, 96)
    wn, bn = dense(core['noise_out'], 192, 64)
    t['core.heads.w'] = np.concatenate([wa, wh, wn], axis=1)       # [192, 161] = amp(1) | harm(96) | noise(64)
    t['core.heads.b'] = np.concatenate([ba, bh, bn], axis=0)
    t['phys.inharm'] = np.array([inh['alpha_B'], inh['beta_B'], inh['alpha_T'], inh['beta_T']], np.float32)
    detune = 0.0
    if det is not None:
        detune = float(det['static_detune'])
        t['phys.detuner'] = np.array([det['dense']['kernel'][0], det['dense']['bias'][0], det['static_detune']], np.float32)
    return t, detune


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--h5')
    g.add_argument('--json')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.h5:
        import export_rtneural as ex
        tmp = tempfile.mkdtemp()
        ex.export_to_rtneural(args.h5, tmp)
        json_dir = tmp
        mp = os.path.join(args.h5, 'training_meta.json')
        tm = json.load(open(mp)) if os.path.exists(mp) else {}
    else:
        json_dir = args.json
        tm = {}
    t, detune = load_json_dir(json_dir)

    legacy = not tm  # sin training_meta.json => pesos de training_kaggle_6 (semantica v1)
    n_strings = 1 if legacy else int(tm.get('n_strings', 1))
    detunes = tm.get('string_detune_cents', [0.0, 0.0]) + [0.0, 0.0]
    meta = {
        'semantics_version': 1 if legacy else int(tm.get('semantics_version', 2)),
        'n_strings': n_strings,
        # v1 sumaba 2 cuerdas identicas (detuner ~ 0): una cuerda con ganancia 2 suena igual y cuesta la mitad
        'string_gain': 2.0 if legacy else 1.0,
        'detune_cents0': float(detunes[0]), 'detune_cents1': float(detunes[1]),
        'release_sec': float(tm.get('release_sec', 1.0)),
        'max_sustain_sec': float(tm.get('max_sustain_sec', 6.0)),
        'fade_frames': float(tm.get('fade_frames', 3)),
        'pedal_threshold': float(tm.get('pedal_threshold', 0.5)),
        'sample_rate': float(tm.get('sample_rate', 16000)),
        'frame_rate': float(tm.get('frame_rate', 250)),
        'n_harmonics': float(tm.get('n_harmonics', 96)),
        'n_noise_filters': float(tm.get('n_noise_filters', 64)),
        'n_voices_train': 8.0 if legacy else float(tm.get('n_voices_train', 24)),
        'leaky_alpha': 0.2,
        'legacy_static_detune_semitones': detune,
    }
    for k, v in meta.items():
        t['meta.' + k] = np.array([v], np.float32)

    out_bin = os.path.join(args.out, 'ddsp_weights.bin')
    with open(out_bin, 'wb') as f:
        f.write(b'DDSW')
        f.write(struct.pack('<II', 1, len(t)))
        for name, arr in t.items():
            arr = np.ascontiguousarray(arr, dtype='<f4')
            nb = name.encode()
            f.write(struct.pack('<H', len(nb)) + nb)
            f.write(struct.pack('<B', arr.ndim) + struct.pack('<' + 'I' * arr.ndim, *arr.shape))
            f.write(arr.tobytes())
    shutil.copy(os.path.join(json_dir, 'reverb_ir.wav'), os.path.join(args.out, 'reverb_ir.wav'))
    print(f"[OK] {out_bin}  ({os.path.getsize(out_bin) / 1e6:.2f} MB, {len(t)} tensores, semantica v{meta['semantics_version']}, "
          f"{n_strings} cuerda(s))")
    print(f"[OK] {os.path.join(args.out, 'reverb_ir.wav')}")


if __name__ == '__main__':
    sys.exit(main())
