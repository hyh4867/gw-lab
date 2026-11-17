"""Matched filter written directly on numpy FFTs.

Conventions: x~(f_k) = dt * rfft(x)[k]; one-sided PSD S(f);
<a|b> = 4 Re sum a~ conj(b~) / S * df.
"""
import numpy as np
from scipy.signal.windows import tukey

from . import const


def band_slice(n, fs=const.FS, f_low=const.F_LOW, f_high=const.F_HIGH):
    df = fs / n
    return slice(int(np.ceil(f_low / df)), int(np.floor(f_high / df)) + 1)


def to_fd(x, fs=const.FS, taper=const.TUKEY_TAPER):
    """Tukey-window a time series and return its continuous-normalised Fourier transform."""
    n = len(x)
    w = tukey(n, 2.0 * taper * fs / n) if taper > 0 else 1.0
    return np.fft.rfft(x * w) / fs


def inner(a_f, b_f, inv_psd, df, sl=None):
    """Noise-weighted inner product <a|b> (real part)."""
    if sl is None:
        sl = slice(None)
    return 4.0 * df * np.real(np.sum(a_f[sl] * np.conj(b_f[sl]) * inv_psd[sl]))


def sigmasq(h_f, inv_psd, df, sl=None):
    if sl is None:
        sl = slice(None)
    return 4.0 * df * np.sum(np.abs(h_f[sl]) ** 2 * inv_psd[sl])


def complex_filter(s_f, h_f, inv_psd, n, fs=const.FS, sl=None):
    """z(t) = 4 int s~ conj(h~) / S exp(2 pi i f t) df for all n time samples."""
    if sl is None:
        sl = band_slice(n, fs)
    buf = np.zeros(n, dtype=np.complex128)
    buf[sl] = s_f[sl] * np.conj(h_f[sl]) * inv_psd[sl]
    return 4.0 * fs * np.fft.ifft(buf)  # ifft has 1/n; 4*df*n = 4*fs


def snr_series(s_f, h_f, inv_psd, n, fs=const.FS, sl=None):
    """Return (rho(t), z(t), sigma). rho is maximised over the template phase."""
    if sl is None:
        sl = band_slice(n, fs)
    df = fs / n
    sig = np.sqrt(sigmasq(h_f, inv_psd, df, sl))
    z = complex_filter(s_f, h_f, inv_psd, n, fs, sl)
    return np.abs(z) / sig, z, sig


def peak_parabolic(y, i):
    """Sub-sample peak position and value from a parabola through y[i-1], y[i], y[i+1]."""
    if i <= 0 or i >= len(y) - 1:
        return float(i), float(y[i])
    a, b, c = y[i - 1], y[i], y[i + 1]
    den = a - 2 * b + c
    # only interpolate at a local maximum: elsewhere the parabola extrapolates (stage-5 change log, entry 1)
    if den >= 0 or b < a or b < c:
        return float(i), float(b)
    d = 0.5 * (a - c) / den
    return i + d, b - 0.25 * (a - c) * d


def chisq(s_f, h_f, inv_psd, n, i_peak, fs=const.FS, p=const.CHISQ_BINS, sl=None):
    """Allen (2005) power chi-squared at time sample i_peak. Returns (chi2, dof)."""
    if sl is None:
        sl = band_slice(n, fs)
    df = fs / n
    k = np.arange(sl.start, sl.stop)
    w = np.abs(h_f[sl]) ** 2 * inv_psd[sl]
    cum = np.cumsum(w)
    tot = cum[-1]
    sig2 = 4.0 * df * tot
    edges = np.searchsorted(cum, tot * np.arange(1, p) / p)
    integrand = 4.0 * df * s_f[sl] * np.conj(h_f[sl]) * inv_psd[sl] * np.exp(2j * np.pi * k * i_peak / n)
    parts = np.array([seg.sum() for seg in np.split(integrand, edges)])
    z = parts.sum()
    chi2 = p / sig2 * np.sum(np.abs(parts - z / p) ** 2)
    return float(chi2), 2 * p - 2


def newsnr(rho, chi2_r):
    """Re-weighted SNR, Eq. (6) of Abbott et al. PRD 93, 122003."""
    rho = np.asarray(rho, dtype=float)
    chi2_r = np.asarray(chi2_r, dtype=float)
    out = rho.copy()
    m = chi2_r > 1
    out[m] = rho[m] / ((1 + chi2_r[m] ** 3) / 2) ** (1.0 / 6)
    return out
