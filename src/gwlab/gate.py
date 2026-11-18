"""Time-domain gating of loud glitches (pre-registration stage 6, section 2).

A glitch is a stretch where the whitened strain exceeds `thresh` standard deviations. The strain is
multiplied by an inverse Tukey window: zero for +-zero_half seconds around the loudest sample and a
cosine roll-off of `taper` seconds on each side. The default numbers are those GWOSC documents for the
window the LIGO-Virgo searches applied around the L1 glitch before GW170817 (zero +-0.1 s, roll-off 0.5 s).
"""
import numpy as np

from . import const, psd

THRESH = 50.0
ZERO_HALF = 0.1
TAPER = 0.5
BLOCK = 64          # seconds whitened at a time
MIN_SEP = 1.0       # glitches closer than this are one glitch


def inverse_tukey(n, centre, fs=const.FS, zero_half=ZERO_HALF, taper=TAPER):
    """Window of n samples: 0 within +-zero_half s of sample `centre`, cosine roll-off over `taper` s, else 1."""
    t = np.abs(np.arange(n) - centre) / fs
    w = np.ones(n)
    w[t <= zero_half] = 0.0
    m = (t > zero_half) & (t < zero_half + taper)
    w[m] = 0.5 * (1 - np.cos(np.pi * (t[m] - zero_half) / taper))
    return w


def find_glitches(strain, fs=const.FS, thresh=THRESH, block=BLOCK):
    """Sample indices of glitch centres and their whitened peak amplitudes (in standard deviations).

    The file is whitened in blocks of `block` s (PSD: median Welch of the 512 s around each block, which a
    short glitch does not bias); the outer 4 s of each block are not searched and blocks overlap by 8 s.
    Blocks containing NaN are skipped.
    """
    n = block * fs
    edge = 4 * fs
    hits = []
    i0 = 0
    while i0 + n <= len(strain):
        seg = strain[i0:i0 + n]
        if not np.isnan(seg).any():
            try:
                st = psd.valid_stretches(strain, fs)
                inside = [(a, b) for a, b in st if a <= i0 and i0 + n <= b]
                _, _, inv = psd.psd_for_segment(strain, i0, n, fs, bounds=inside[0] if inside else None)
            except ValueError:
                inv = None
            if inv is not None:
                from scipy.signal.windows import tukey
                xw = psd.whiten(seg * tukey(n, 2.0 * fs / n), inv, fs)[edge:n - edge]
                sd = 1.4826 * np.median(np.abs(xw - np.median(xw)))
                a = np.abs(xw) / sd
                for j in np.flatnonzero(a > thresh):
                    hits.append((i0 + edge + j, a[j]))
        i0 += n - 2 * edge
    if not hits:
        return np.zeros(0, dtype=int), np.zeros(0)
    hits.sort()
    centres, peaks = [], []
    grp = [hits[0]]
    for h in hits[1:]:
        if h[0] - grp[-1][0] <= MIN_SEP * fs:
            grp.append(h)
        else:
            k = max(grp, key=lambda x: x[1])
            centres.append(k[0]); peaks.append(k[1])
            grp = [h]
    k = max(grp, key=lambda x: x[1])
    centres.append(k[0]); peaks.append(k[1])
    return np.array(centres, dtype=int), np.array(peaks)


def apply_gates(strain, centres, fs=const.FS, zero_half=ZERO_HALF, taper=TAPER):
    """Return a gated copy of the strain."""
    out = strain.copy()
    half = int((zero_half + taper) * fs) + 1
    for c in centres:
        a, b = max(0, c - half), min(len(out), c + half + 1)
        out[a:b] *= inverse_tukey(b - a, c - a, fs, zero_half, taper)
    return out
