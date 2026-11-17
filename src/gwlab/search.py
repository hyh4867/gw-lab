"""Bank search over one analysis segment: max-over-bank SNR, clustering, chi-squared."""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter1d

from . import const, mf, psd, waveforms


@dataclass
class Bank:
    masses: np.ndarray  # (K, 2) detector-frame m1, m2
    h: np.ndarray       # (K, nband) complex64, band-limited templates
    sl: slice
    n: int
    kind: str

    def full(self, k):
        out = np.zeros(self.n // 2 + 1, dtype=np.complex128)
        out[self.sl] = self.h[k]
        return out


def coarse_masses(lo=20.0, hi=80.0, step=2.0):
    v = np.arange(lo, hi + 1e-9, step)
    return np.array([(a, b) for a in v for b in v if a >= b])


def fine_masses(m1c, m2c, half=6.0, step=0.5):
    a = np.arange(m1c - half, m1c + half + 1e-9, step)
    b = np.arange(m2c - half, m2c + half + 1e-9, step)
    return np.array([(x, y) for x in a for y in b if x >= y and y > 0])


def build_bank(masses, n=const.SEG_DUR * const.FS, kind="imr", fs=const.FS):
    """masses: rows (m1, m2) or (m1, m2, chi1, chi2) (aligned spins, IMRPhenomD only)."""
    masses = np.asarray(masses, float)
    sl = mf.band_slice(n, fs)
    h = np.zeros((len(masses), sl.stop - sl.start), dtype=np.complex64)
    for k, row in enumerate(masses):
        m1, m2 = row[0], row[1]
        if kind == "imr":
            c1, c2 = (row[2], row[3]) if len(row) >= 4 else (0.0, 0.0)
            hf = waveforms.imrphenomd(m1, m2, n, fs, chi1=c1, chi2=c2)
        elif kind == "f2":
            hf = waveforms.taylorf2(m1, m2, n, fs)
        else:
            raise ValueError(kind)
        # scale to O(1) so complex64 keeps precision; SNR does not depend on the scale
        h[k] = (hf[sl] * 1e20).astype(np.complex64)
    return Bank(masses, h, sl, n, kind)


@dataclass(frozen=True)
class Layout:
    """Segment layout of a search. `decim` = 2 evaluates rho(t) on every second sample (complex ifft of
    half length; the band stops at 1024 Hz so nothing is lost except time resolution)."""
    seg_dur: int = const.SEG_DUR
    stride: int = const.SEG_STRIDE
    valid_start: int = const.VALID_START
    valid_end: int = const.VALID_END
    decim: int = 1

    def starts(self, nsamp, fs=const.FS):
        n = self.seg_dur * fs
        return [j * self.stride * fs for j in range((nsamp - n) // (self.stride * fs) + 1)]

    def span(self, file_dur=4096):
        """Seconds of a file covered by valid regions."""
        nseg = (file_dur - self.seg_dur) // self.stride + 1
        return (nseg - 1) * self.stride + (self.valid_end - self.valid_start)


LAYOUT_BBH = Layout()                                   # stage 0-4 layout (GW150914 bank)
LAYOUT_LOWMASS = Layout(64, 32, 16, 48, 2)              # stage 5 low-mass bank (templates up to ~11 s)


MIN_PSD_SPAN = 128  # seconds of gap-free data needed around a segment (stage 5; files with gaps)


def prepare_segment(strain, i0, n, extra=None, fs=const.FS, stretches=None):
    """Window + FFT a segment and estimate its PSD. `extra` is an optional signal added in the time domain.

    stretches: optional list from psd.valid_stretches for files with gaps; the segment must lie inside one
    stretch of at least MIN_PSD_SPAN seconds, and the PSD is estimated inside that stretch."""
    seg = strain[i0:i0 + n]
    if np.isnan(seg).any():
        return None
    bounds = None
    if stretches is not None:
        inside = [(a, b) for a, b in stretches if a <= i0 and i0 + n <= b]
        if not inside or inside[0][1] - inside[0][0] < MIN_PSD_SPAN * fs:
            return None
        bounds = inside[0]
    if extra is not None:
        seg = seg + extra
    f, p, inv = psd.psd_for_segment(strain, i0, n, fs, bounds=bounds)
    return {"s_f": mf.to_fd(seg, fs), "inv": inv, "psd": p, "f": f, "seg": seg}


def bank_filter(prep, bank, v0, v1, fs=const.FS, keep=None, decim=1):
    """Max over the bank of rho(t) on samples [v0, v1) (step `decim`). Returns (maxrho, argmax, sigmas, kept).

    keep: optional list of template indices whose full rho(t) on [v0, v1) is also returned.
    """
    n, sl = bank.n, bank.sl
    df = fs / n
    nd = n // decim
    if decim > 1 and sl.stop > nd:
        raise ValueError("band does not fit in the decimated transform")
    w = prep["s_f"][sl] * prep["inv"][sl]
    inv = prep["inv"][sl]
    a, b = v0 // decim, v1 // decim
    best2 = np.zeros(b - a)
    arg = np.zeros(b - a, dtype=np.int32)
    sig = np.zeros(len(bank.h))
    buf = np.zeros(nd, dtype=np.complex128)
    kept = {}
    for k in range(len(bank.h)):
        hk = bank.h[k].astype(np.complex128)
        s2 = 4.0 * df * np.sum((hk.real**2 + hk.imag**2) * inv)
        sig[k] = np.sqrt(s2)
        buf[sl] = w * np.conj(hk)
        z = np.fft.ifft(buf)[a:b]
        # ifft of length nd carries 1/nd; 4 * df * nd = 4 * fs / decim
        r2 = (z.real**2 + z.imag**2) * ((4.0 * fs / decim) ** 2 / s2)
        if keep is not None and k in keep:
            kept[k] = np.sqrt(r2)
        m = r2 > best2
        best2[m] = r2[m]
        arg[m] = k
    return np.sqrt(best2), arg, sig, kept


def find_triggers(prep, bank, i0, fs=const.FS, thresh=const.SNR_THRESH, v0=None, v1=None, decim=1):
    """Clustered single-detector triggers of one segment with chi-squared and re-weighted SNR.

    Returns a structured array: t (s from file start), rho, tmpl, chi2r, newsnr.
    """
    if v0 is None:
        v0 = const.VALID_START * fs
    if v1 is None:
        v1 = const.VALID_END * fs
    rho, arg, sig, _ = bank_filter(prep, bank, v0, v1, fs, decim=decim)
    return triggers_from_series(prep, bank, rho, arg, i0, v0, fs, thresh, decim)


def triggers_from_series(prep, bank, rho, arg, i0, v0, fs=const.FS, thresh=const.SNR_THRESH, decim=1):
    half = int(const.CLUSTER_HALF * fs) // decim
    loc = maximum_filter1d(rho, size=2 * half + 1, mode="constant", cval=0.0)
    idx = np.flatnonzero((rho >= thresh) & (rho == loc))
    out = np.zeros(len(idx), dtype=TRIG_DTYPE)
    for j, i in enumerate(idx):
        k = int(arg[i])
        ip = v0 + i * decim
        c2, dof = mf.chisq(prep["s_f"], bank.full(k), prep["inv"], bank.n, ip, fs, sl=bank.sl)
        out[j] = ((i0 + ip) / fs, rho[i], k, c2 / dof, 0.0)
    out["newsnr"] = mf.newsnr(out["rho"], out["chi2r"])
    return out


TRIG_DTYPE = np.dtype([("t", "f8"), ("rho", "f8"), ("tmpl", "i4"), ("chi2r", "f8"), ("newsnr", "f8")])


def cluster(trigs, half=const.CLUSTER_HALF):
    """Keep a trigger only if no louder (raw rho) trigger lies within +-half seconds."""
    if len(trigs) == 0:
        return trigs
    t = trigs[np.argsort(trigs["t"])]
    keep = np.ones(len(t), dtype=bool)
    for i in range(len(t)):
        j = i + 1
        while j < len(t) and t["t"][j] - t["t"][i] <= half:
            if t["rho"][j] > t["rho"][i]:
                keep[i] = False
            else:
                keep[j] = False
            j += 1
    return t[keep]
