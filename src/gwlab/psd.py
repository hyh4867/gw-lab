"""PSD estimation (median Welch), inverse spectrum truncation, whitening."""
import numpy as np
from scipy import signal

from . import const


def welch_median(x, fs=const.FS, seg=const.PSD_SEG):
    """One-sided PSD by Welch's method: Hann window, 50 % overlap, median average."""
    nper = int(seg * fs)
    f, p = signal.welch(x, fs=fs, window="hann", nperseg=nper, noverlap=nper // 2,
                        average="median", detrend="constant")
    return f, p


def interpolate(f_src, p_src, n, fs=const.FS):
    """Linear interpolation of a PSD onto the rfft grid of an n-sample segment."""
    f = np.fft.rfftfreq(n, 1.0 / fs)
    return f, np.interp(f, f_src, p_src)


def inverse_spectrum_truncation(psd, n, fs=const.FS, half=const.INVSPEC_HALF, f_low=15.0):
    """Return 1/S whose time-domain response is limited to +-half seconds.

    Follows the FINDCHIRP recipe: truncate the impulse response of sqrt(1/S), then square.
    Frequencies below f_low are zeroed before truncation.
    """
    f = np.fft.rfftfreq(n, 1.0 / fs)
    inv_asd = np.zeros_like(psd)
    ok = (f >= f_low) & (psd > 0)
    inv_asd[ok] = 1.0 / np.sqrt(psd[ok])
    q = np.fft.irfft(inv_asd, n)
    k = int(half * fs)
    q[k:n - k] = 0.0
    inv_asd_t = np.fft.rfft(q)
    return np.abs(inv_asd_t) ** 2


def psd_for_segment(strain, i0, n, fs=const.FS, span=const.PSD_SPAN, bounds=None):
    """Median-Welch PSD from `span` seconds centred on the segment [i0, i0+n), kept inside the file.

    bounds: optional (a, b) sample range of gap-free data containing the segment; the PSD window is kept
    inside it (and shortened to it if the stretch is shorter than `span`).
    Returns (f, psd_interp, inv_psd_truncated) on the rfft grid of the segment.
    """
    lo, hi = (0, len(strain)) if bounds is None else bounds
    centre = i0 + n // 2
    w = min(int(span * fs), hi - lo)
    a = centre - w // 2
    a = max(lo, min(a, hi - w))
    fw, pw = welch_median(strain[a:a + w], fs)
    f, p = interpolate(fw, pw, n, fs)
    return f, p, inverse_spectrum_truncation(p, n, fs)


def valid_stretches(strain, fs=const.FS):
    """List of (a, b) sample ranges, on whole seconds, without NaN."""
    nsec = len(strain) // fs
    bad = np.isnan(strain[:nsec * fs].reshape(nsec, fs)).any(axis=1)
    out, start = [], None
    for s in range(nsec):
        if not bad[s] and start is None:
            start = s
        if bad[s] and start is not None:
            out.append((start * fs, s * fs))
            start = None
    if start is not None:
        out.append((start * fs, nsec * fs))
    return out


def whiten(x, inv_psd, fs=const.FS):
    """Whiten a (windowed) segment: unit-variance white noise where the PSD model holds."""
    xf = np.fft.rfft(x)
    return np.fft.irfft(xf * np.sqrt(inv_psd * 2.0 / fs), len(x))


def bandpass(x, f1, f2, fs=const.FS, order=4):
    sos = signal.butter(order, [f1, f2], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x)
