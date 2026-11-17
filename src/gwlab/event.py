"""Event-level measurements with two detectors (pre-registration section 4)."""
import numpy as np
from scipy.ndimage import maximum_filter1d

from . import const, mf


def scan_bank(prep_h, prep_l, bank, v0, v1, fs=const.FS, window=const.COINC_WINDOW):
    """For every template: best network SNR with |t_H - t_L| <= window on samples [v0, v1).

    Returns a dict of arrays: net, rho_h, rho_l, i_h, i_l (sub-sample indices in the segment).
    """
    n, sl = bank.n, bank.sl
    df = fs / n
    k = len(bank.h)
    out = {key: np.zeros(k) for key in ("net", "rho_h", "rho_l", "i_h", "i_l")}
    w = int(round(window * fs))
    buf = np.zeros(n, dtype=np.complex128)
    for t in range(k):
        hk = bank.h[t].astype(np.complex128)
        rho = {}
        for name, prep in (("h", prep_h), ("l", prep_l)):
            inv = prep["inv"][sl]
            s2 = 4.0 * df * np.sum((hk.real**2 + hk.imag**2) * inv)
            buf[sl] = prep["s_f"][sl] * inv * np.conj(hk)
            z = np.fft.ifft(buf)[v0 - 1:v1 + 1]
            rho[name] = np.abs(z) * (4.0 * fs / np.sqrt(s2))
        lmax = maximum_filter1d(rho["l"], size=2 * w + 1, mode="nearest")
        tot = rho["h"] ** 2 + lmax**2
        ih = int(np.argmax(tot[1:-1])) + 1
        lo, hi = max(1, ih - w), min(len(tot) - 1, ih + w + 1)
        il = lo + int(np.argmax(rho["l"][lo:hi]))
        ph, vh = mf.peak_parabolic(rho["h"], ih)
        pl, vl = mf.peak_parabolic(rho["l"], il)
        out["net"][t] = np.sqrt(vh**2 + vl**2)
        out["rho_h"][t], out["rho_l"][t] = vh, vl
        out["i_h"][t], out["i_l"][t] = ph + v0 - 1, pl + v0 - 1
    return out


def best_fit_signal(prep, h_f, n, i_peak, fs=const.FS, sl=None):
    """Maximum-likelihood copy of the template in the data: complex amplitude z/sigma^2 at i_peak
    (may be fractional). Returns the frequency-domain signal (continuous normalisation)."""
    if sl is None:
        sl = mf.band_slice(n, fs)
    df = fs / n
    kk = np.arange(sl.start, sl.stop)
    inv = prep["inv"][sl]
    z = 4.0 * df * np.sum(prep["s_f"][sl] * np.conj(h_f[sl]) * inv * np.exp(2j * np.pi * kk * i_peak / n))
    s2 = mf.sigmasq(h_f, prep["inv"], df, sl)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    return (z / s2) * h_f * np.exp(-2j * np.pi * f * i_peak / fs)
