"""Shared helpers for the stage-6 (GW170817) scripts: data sets A and B, gating, event segments, template scans."""
import pathlib
import sys

import numpy as np
from scipy.ndimage import maximum_filter1d

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gwlab import bns, const, detectors, event, gate, io, mf, psd, search, waveforms  # noqa: E402

FS = const.FS
RAW = ROOT / "data" / "raw"
A_START, B_START = 1187006835, 1187007040
T0 = 1187008882.4                      # catalogue time
GLITCH_DOC = 1187008881.389            # glitch time in the GWOSC documentation (comparison only)
RA = np.deg2rad(15.0 * (13 + 9 / 60 + 48.085 / 3600))          # optical counterpart, [MM] p. 5
DEC = -np.deg2rad(23 + 22 / 60 + 53.343 / 3600)
Z = 0.0099
LAY = bns.LAYOUT
N = LAY.seg_dur * FS
DETS = ("H1", "L1", "V1")


def path(dataset, det):
    if dataset == "A":
        return RAW / f"{det[0]}-{det}_GWOSC_4KHZ_R1-{A_START}-4096.hdf5"
    return RAW / f"{det[0]}-{det}_LOSC_CLN_4_V1-{B_START}-2048.hdf5"


def start(dataset):
    return A_START if dataset == "A" else B_START


def load(dataset, det, gated=False):
    """Strain and the list of gates (sample index, whitened peak) found by the automatic rule."""
    x = io.read_strain(path(dataset, det))["strain"]
    c, peak = gate.find_glitches(x)
    if gated:
        x = gate.apply_gates(x, c)
    return x, [(int(a), float(b)) for a, b in zip(c, peak)]


def segment_start(dataset, t_gps):
    """First sample of the 256-s analysis segment (pre-registration stage 6, sections 2 and 4)."""
    t = t_gps - start(dataset)
    if dataset == "A":
        return int((t - LAY.valid_start) // LAY.stride) * LAY.stride * FS
    return int(round((t - 216.0) * FS))


def prep(x, dataset, i0):
    """FFT + PSD of the analysis segment. Data B: PSD from the 512 s that end with the segment."""
    st = psd.valid_stretches(x)
    if dataset == "B":
        st = [(a, min(b, i0 + N)) for a, b in st if a < i0 + N]
    return search.prepare_segment(x, i0, N, stretches=st)


def scan_fast(basis, masses, pr_h, pr_l, c, half=0.1, window=const.COINC_WINDOW):
    """Per-template network SNR at 1024 Hz around full-rate sample c (pre-registered fast pass).
    pr_h, pr_l: bns.reduce_prep results. Returns dict of arrays net, rho_h, rho_l."""
    from scipy import fft as sfft
    decim = LAY.decim
    nd = N // decim
    a, b = int((c - half * FS) // decim), int((c + half * FS) // decim) + 1
    w = int(round(window * FS / decim))
    df = FS / N
    buf = np.zeros(nd, dtype=np.complex64)
    out = {k: np.zeros(len(masses)) for k in ("net", "rho_h", "rho_l")}
    for k, (m1, m2) in enumerate(masses):
        e, kcut = bns.shape(basis, m1, m2)
        rho = []
        for p in (pr_h, pr_l):
            buf[basis.sl] = p["w"] * e
            z = sfft.ifft(buf)[a:b]
            rho.append(np.sqrt((z.real**2 + z.imag**2) * ((4.0 * FS / decim) ** 2 / (4.0 * df * p["cw"][kcut - 1]))))
        lmax = maximum_filter1d(rho[1], size=2 * w + 1, mode="nearest")
        i = int(np.argmax(rho[0] ** 2 + lmax**2))
        out["rho_h"][k], out["rho_l"][k] = rho[0][i], lmax[i]
        out["net"][k] = np.hypot(rho[0][i], lmax[i])
    return out


def bank_of(templates, masses, sl):
    """search.Bank from full frequency-domain templates (list of arrays on the rfft grid)."""
    h = np.array([(t[sl] * 1e20).astype(np.complex64) for t in templates])
    return search.Bank(np.asarray(masses, float), h, sl, N, "custom")


def scan_full(templates, masses, sl, pr_h, pr_l, c, half=0.1):
    """Full-rate scan with parabolic interpolation (event.scan_bank) around sample c."""
    v0, v1 = c - int(half * FS), c + int(half * FS)
    return event.scan_bank(pr_h, pr_l, bank_of(templates, masses, sl), v0, v1)


def z_at(pr, h, sl, i_peak):
    """Complex matched-filter output z/sigma at a (fractional) sample of the segment."""
    df = FS / N
    kk = np.arange(sl.start, sl.stop)
    inv = pr["inv"][sl]
    z = 4.0 * df * np.sum(pr["s_f"][sl] * np.conj(h[sl]) * inv * np.exp(2j * np.pi * kk * i_peak / N))
    return z / np.sqrt(mf.sigmasq(h, pr["inv"], df, sl))


def geometric_delays(gps=T0):
    """Arrival time at each detector minus arrival at the geocentre for the counterpart position (s)."""
    lon, lat = detectors.lonlat_from_radec(RA, DEC, gps)
    return {d: detectors.delay_from_geocentre(d, lon, lat) for d in DETS}


def f2_time_and_cycles(m1, m2, f_a, f_b=None, npts=200000):
    """Time (s) and number of GW cycles between f_a and f_b (default f_ISCO) from the TaylorF2 phase:
    t(f) = psi'(f) / 2 pi, N = (1/2 pi) int f psi''(f) df."""
    if f_b is None:
        f_b = waveforms.f_isco(m1 + m2)
    f = np.geomspace(f_a, f_b, npts)
    psi = waveforms.taylorf2_phase(f, m1, m2)
    t = np.gradient(psi, f) / (2 * np.pi)
    dur = t[-1] - t[0]
    cycles = np.trapezoid(f * np.gradient(t, f), f)
    return float(dur), float(cycles)
