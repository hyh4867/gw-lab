"""Binary-neutron-star search with hand-written TaylorF2 templates (pre-registration stage 6).

Differences from search.py, all forced by the template length (about a minute from 30 Hz):
- 256-s segments; a template is rebuilt from f2fast.F2Basis for every use instead of being stored;
- the loop runs over templates on the outside and over all segments of a file pair on the inside, so a
  template is built once per file pair;
- rho(t) is evaluated at fs/4 = 1024 Hz (complex ifft of a quarter of the length; the band 30-1024 Hz fits).
"""
import numpy as np
from scipy import fft as sfft
from scipy.ndimage import maximum_filter1d

from . import const, f2fast, mf, search, waveforms

F_LOW = 30.0
LAYOUT = search.Layout(256, 128, 112, 240, 4)


def basis_for(layout=LAYOUT, fs=const.FS):
    n = layout.seg_dur * fs
    # the highest bin must stay below fs/decim so that the band fits in the decimated transform
    return f2fast.F2Basis(n, fs, F_LOW, min(const.F_HIGH, fs / layout.decim) - fs / n)


def reduce_prep(prep, basis):
    """Keep only what the bank loop needs from search.prepare_segment: w = s~/S on the band (complex64) and
    the cumulative sum of f^(-7/3)/S, which gives sigma^2 of any TaylorF2 template up to its cut-off."""
    sl = basis.sl
    inv = prep["inv"][sl]
    return {"w": (prep["s_f"][sl] * inv * 1e-20).astype(np.complex64),         # 1e-20: keep float32 in range
            "cw": np.cumsum(basis.amp**2 * inv) * 1e-40}


def shape(basis, m1, m2):
    """conj of the unit-amplitude template shape, f^(-7/6) exp(+i psi), zero above f_ISCO; and the cut index."""
    c, c0 = basis.coeffs(m1, m2)
    psi = c @ basis.b
    psi += c0
    e = (basis.amp * (np.cos(psi) + 1j * np.sin(psi))).astype(np.complex64)
    kcut = int(np.searchsorted(basis.f, waveforms.f_isco(m1 + m2), side="right"))
    if kcut < len(e):
        e[kcut:] = 0
    return e, kcut


def filter_bank(basis, masses, preps, layout=LAYOUT, fs=const.FS, best2=None, arg=None, k_start=0, k_stop=None,
                progress=None):
    """Max over templates [k_start, k_stop) of rho^2(t) on the valid region of every prepared segment.

    preps: list of reduce_prep results. best2 (float32) and arg (int32), shape (len(preps), nvalid), are
    updated in place so the loop can be resumed from a checkpoint."""
    n, sl, decim = basis.n, basis.sl, layout.decim
    nd = n // decim
    a, b = layout.valid_start * fs // decim, layout.valid_end * fs // decim
    if best2 is None:
        best2 = np.zeros((len(preps), b - a), dtype=np.float32)
        arg = np.zeros((len(preps), b - a), dtype=np.int32)
    df = fs / n
    buf = np.zeros(nd, dtype=np.complex64)
    k_stop = len(masses) if k_stop is None else k_stop
    for k in range(k_start, k_stop):
        e, kcut = shape(basis, masses[k, 0], masses[k, 1])
        for s, p in enumerate(preps):
            if p is None:
                continue
            buf[sl] = p["w"] * e
            z = sfft.ifft(buf)[a:b]
            # z carries 1/nd; rho^2 = |4 df nd z|^2 / (4 df cw)
            r2 = (z.real**2 + z.imag**2) * np.float32((4.0 * fs / decim) ** 2 / (4.0 * df * p["cw"][kcut - 1]))
            m = r2 > best2[s]
            best2[s][m] = r2[m]
            arg[s][m] = k
        if progress is not None:
            progress(k)
    return best2, arg


def chisq(p, e, kcut, basis, ip, nbins=const.CHISQ_BINS):
    """Allen chi-squared at sample ip (full-rate index in the segment) from the reduced arrays."""
    n, sl = basis.n, basis.sl
    df = const.FS / n
    k = np.arange(sl.start, sl.start + kcut)
    cum = p["cw"][:kcut]
    tot = cum[-1]
    edges = np.searchsorted(cum, tot * np.arange(1, nbins) / nbins)
    integrand = 4.0 * df * p["w"][:kcut].astype(np.complex128) * e[:kcut] * np.exp(2j * np.pi * k * ip / n)
    parts = np.array([seg.sum() for seg in np.split(integrand, edges)])
    z = parts.sum()
    return float(nbins / (4.0 * df * tot) * np.sum(np.abs(parts - z / nbins) ** 2)), 2 * nbins - 2


def triggers(basis, masses, p, best2_row, arg_row, i0, layout=LAYOUT, fs=const.FS, thresh=const.SNR_THRESH):
    """Clustered single-detector triggers of one segment (same rules as search.triggers_from_series)."""
    decim = layout.decim
    rho = np.sqrt(best2_row.astype(np.float64))
    half = int(const.CLUSTER_HALF * fs) // decim
    loc = maximum_filter1d(rho, size=2 * half + 1, mode="constant", cval=0.0)
    idx = np.flatnonzero((rho >= thresh) & (rho == loc))
    out = np.zeros(len(idx), dtype=search.TRIG_DTYPE)
    v0 = layout.valid_start * fs
    for j, i in enumerate(idx):
        k = int(arg_row[i])
        e, kcut = shape(basis, masses[k, 0], masses[k, 1])
        ip = v0 + i * decim
        c2, dof = chisq(p, e, kcut, basis, ip)
        out[j] = ((i0 + ip) / fs, rho[i], k, c2 / dof, 0.0)
    out["newsnr"] = mf.newsnr(out["rho"], out["chi2r"])
    return out
