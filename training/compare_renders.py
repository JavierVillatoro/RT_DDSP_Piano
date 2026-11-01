"""
Compara dos renders (referencia Python vs motor C++) con metricas robustas al ruido aleatorio:
  * diferencia de nivel global (dB)
  * correlacion de las envolventes de energia (ventanas de 20 ms)
  * distancia espectral log multi-resolucion (L1 de log-magnitud, dB) y la misma entre dos renders de
    REFERENCIA con semillas distintas (--baseline) para saber cual es el suelo debido solo al ruido.
    python compare_renders.py ref.wav cpp.wav [--baseline ref_seed1.wav]
"""
import sys
import argparse
import numpy as np
import scipy.io.wavfile
import scipy.signal


def load(p):
    sr, x = scipy.io.wavfile.read(p)
    x = x.astype(np.float64)
    if x.dtype.kind == 'i':
        x /= 32768.0
    return sr, x


def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)


def envelope(x, sr, win=0.02):
    n = int(sr * win)
    m = len(x) // n
    return np.sqrt((x[:m * n].reshape(m, n) ** 2).mean(axis=1))


def mrstft_logdist(a, b, sr, sizes=(2048, 1024, 512, 256)):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    d = []
    for s in sizes:
        _, _, A = scipy.signal.stft(a, sr, nperseg=s, noverlap=s * 3 // 4)
        _, _, B = scipy.signal.stft(b, sr, nperseg=s, noverlap=s * 3 // 4)
        la, lb = 20 * np.log10(np.abs(A) + 1e-6), 20 * np.log10(np.abs(B) + 1e-6)
        # solo donde hay energia (evita que domine el suelo de ruido numerico)
        mask = (la > la.max() - 70) | (lb > lb.max() - 70)
        d.append(np.abs(la - lb)[mask].mean())
    return float(np.mean(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ref')
    ap.add_argument('test')
    ap.add_argument('--baseline')
    args = ap.parse_args()
    sr, a = load(args.ref)
    sr2, b = load(args.test)
    assert sr == sr2, f'sample rates distintos {sr} {sr2}'
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    ea, eb = envelope(a, sr), envelope(b, sr)
    corr = float(np.corrcoef(ea, eb)[0, 1])
    print(f"Nivel global     ref={rms_db(a):7.2f} dBFS  test={rms_db(b):7.2f} dBFS  (diferencia {rms_db(b) - rms_db(a):+.2f} dB)")
    print(f"Pico             ref={np.abs(a).max():.3f}  test={np.abs(b).max():.3f}")
    print(f"Corr. envolvente {corr:.4f}   (1.0 = misma dinamica)")
    d = mrstft_logdist(a, b, sr)
    print(f"Dist. espectral  {d:.2f} dB  (ref vs test)")
    if args.baseline:
        _, c = load(args.baseline)
        db = mrstft_logdist(a, c, sr)
        print(f"Suelo por ruido  {db:.2f} dB  (ref vs ref con otra semilla)  -> exceso = {d - db:+.2f} dB")


if __name__ == '__main__':
    main()
