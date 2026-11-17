"""A small constant-Q transform on whitened data (Gaussian frequency windows)."""
import numpy as np


def qtransform(xw, fs, freqs, q=8.0):
    """Normalised energy |X(t, f0)|^2; white unit-variance noise gives mean 1 in every row.

    Each row is the squared envelope of the data band-passed by a Gaussian window of
    width sigma_f = f0 / q centred on f0.
    """
    n = len(xw)
    xf = np.fft.rfft(xw)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    out = np.empty((len(freqs), n))
    buf = np.zeros(n, dtype=complex)
    for i, f0 in enumerate(freqs):
        w = np.exp(-0.5 * ((f - f0) / (f0 / q)) ** 2)
        buf[: len(f)] = xf * w
        z = np.fft.ifft(buf)
        out[i] = np.abs(z) ** 2 / (np.sum(w**2) / n)
    return out
