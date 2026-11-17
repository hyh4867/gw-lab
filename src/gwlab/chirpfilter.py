"""Chirp mass from the leading-order (Newtonian) chirp law alone (pre-registration stage 5, section 2).

The only physics used is f(t): df/dt = (96/5) pi^(8/3) (G Mc/c^3)^(5/3) f^(11/3), whose
stationary-phase transform is h~(f) ~ f^(-7/6) exp(-i (3/128) (pi G Mc f / c^3)^(-5/3)).
A one-parameter family of such chirps, limited to a fixed band, is correlated with the data and the
chirp mass with the largest SNR is the estimate. No post-Newtonian term, merger or ringdown is modelled,
so the estimate is biased; the bias and scatter are measured with injections.
"""
import numpy as np

from . import const, mf

F_LO, F_HI = 30.0, 150.0
DECIM = 4  # complex output rate fs / 4 = 1024 Hz; the band ends at 150 Hz


def mc_grid(lo=8.0, hi=80.0, step=0.0025):
    return np.exp(np.arange(np.log(lo), np.log(hi) + step / 2, step))


def newton_bank(n, fs, grid, f_lo=F_LO, f_hi=F_HI):
    df = fs / n
    k0, k1 = int(np.ceil(f_lo / df)), int(np.floor(f_hi / df)) + 1
    f = np.arange(k0, k1) * df
    h = np.array([f ** (-7.0 / 6) * np.exp(-1j * (3.0 / 128) * (np.pi * mc * const.MTSUN * f) ** (-5.0 / 3))
                  for mc in grid])
    return {"sl": slice(k0, k1), "h": h, "grid": np.asarray(grid), "n": n, "fs": fs}


def newton_f(tau, mc):
    """GW frequency (Hz) a time tau (s) before coalescence at Newtonian order; mc in solar masses."""
    return (5.0 / (256.0 * tau)) ** 0.375 * (mc * const.MTSUN) ** -0.625 / np.pi


def measure(s_f, inv, bank, t_hint, half=0.15):
    """Maximise the SNR of the Newtonian chirp over chirp mass and time within +-half s of t_hint
    (seconds from the segment start). Returns dict: mc (parabolic interpolation in log Mc), mc_grid,
    snr, t_peak, edge (True if the maximum is at the edge of the mass grid), curve (SNR per grid point)."""
    n, fs, sl, grid = bank["n"], bank["fs"], bank["sl"], bank["grid"]
    df = fs / n
    nd = n // DECIM
    w = s_f[sl] * inv[sl]
    iv = inv[sl]
    a, b = int((t_hint - half) * fs / DECIM), int((t_hint + half) * fs / DECIM) + 1
    buf = np.zeros(nd, dtype=np.complex128)
    curve = np.zeros(len(grid))
    tpk = np.zeros(len(grid))
    for i, h in enumerate(bank["h"]):
        s2 = 4.0 * df * np.sum((h.real**2 + h.imag**2) * iv)
        buf[sl] = w * np.conj(h)
        z = np.fft.ifft(buf)[a:b]
        r2 = (z.real**2 + z.imag**2) * ((4.0 * fs / DECIM) ** 2 / s2)
        j = int(np.argmax(r2))
        pj, pv = mf.peak_parabolic(r2, j)  # sub-sample peak: the output rate is only 1024 Hz
        curve[i] = np.sqrt(pv)
        tpk[i] = (a + pj) * DECIM / fs
    i = int(np.argmax(curve))
    edge = i == 0 or i == len(grid) - 1
    lmc = np.log(grid[i])
    if not edge:
        y0, y1, y2 = curve[i - 1:i + 2]
        den = y0 - 2 * y1 + y2
        if den < 0:
            lmc += 0.5 * (y0 - y2) / den * (np.log(grid[i + 1]) - np.log(grid[i]))
    return {"mc": float(np.exp(lmc)), "mc_grid": float(grid[i]), "snr": float(curve[i]), "t_peak": float(tpk[i]),
            "edge": bool(edge), "curve": curve}
