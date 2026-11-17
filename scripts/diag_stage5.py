"""Diagnostics without a verdict (pre-registration stage 5, sections 5 and 7).

1. Where the excess power of the whitened H1 data (RMS 1.36 in stage 4) comes from.
2. Whether the bulk O1 file and the event file hold the same strain where they overlap.
Writes results/diag_stage5.json
"""
import json
import pathlib
import sys

import numpy as np
from scipy.signal.windows import tukey

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from gwlab import const, io, psd, search  # noqa: E402

FS = const.FS
RAW = ROOT / "data" / "raw"
BANDS = [(0, 15), (15, 20), (20, 35), (35, 350), (350, 1024), (1024, 2048)]


def whitened_power(det, i0, n, f_low=15.0):
    x = io.read_strain(RAW / f"{det[0]}-{det}_GWOSC_4KHZ_R1-1126257415-4096.hdf5")["strain"]
    prep = search.prepare_segment(x, i0, n)
    win = tukey(n, 2.0 * const.TUKEY_TAPER * FS / n)
    f = np.fft.rfftfreq(n, 1.0 / FS)
    out = {}
    # (a) as in stage 1: truncated inverse PSD, PSD interpolated from 0.25 Hz to 1/32 Hz
    # (b) no truncation: divide by the interpolated PSD itself
    for label, inv in (("truncated_inverse_psd", prep["inv"]),
                       ("plain_interpolated_psd", np.where(f >= f_low, 1.0 / prep["psd"], 0.0))):
        xf = np.fft.rfft(prep["seg"] * win) * np.sqrt(inv * 2.0 / FS)
        p = np.abs(xf) ** 2
        tot = 2 * p.sum() / n**2 / np.mean(win**2)  # variance of the whitened series, corrected for the window
        bands = {}
        for a, b in BANDS:
            m = (f >= a) & (f < b)
            # a white unit-variance series has band power = bandwidth / Nyquist
            bands[f"{a}-{b}"] = {"variance": float(2 * p[m].sum() / n**2 / np.mean(win**2)),
                                 "expected_if_white": (b - a) / (FS / 2)}
        xw = psd.whiten(prep["seg"] * win, inv, FS)
        out[label] = {"rms_time_domain": float(np.std(xw[const.VALID_START * FS:const.VALID_END * FS])),
                      "variance_total": float(tot), "bands": bands}
    # how much of the excess sits in narrow lines: bins where whitened power > 10 x median
    xf = np.fft.rfft(prep["seg"] * win) * np.sqrt(prep["inv"] * 2.0 / FS)
    p = np.abs(xf) ** 2
    m = (f >= 20) & (f < 1024)
    med = np.median(p[m])
    hot = m & (p > 10 * med / np.log(2))
    out["lines_20_1024"] = {"fraction_of_bins_above_10x_mean": float(hot.sum() / m.sum()),
                            "fraction_of_power_in_those_bins": float(p[hot].sum() / p[m].sum()),
                            "expected_fraction_of_bins_gaussian": float(np.exp(-10)),
                            "loudest_frequencies_hz": [float(v) for v in f[np.argsort(np.where(m, p, 0))[::-1][:12]]]}
    return out


def overlap_check(det):
    b = io.read_strain(RAW / f"{det[0]}-{det}_LOSC_4_V1-1126256640-4096.hdf5")
    e = io.read_strain(RAW / f"{det[0]}-{det}_GWOSC_4KHZ_R1-1126257415-4096.hdf5")
    off = int((e["start"] - b["start"]) * FS)
    n = len(b["strain"]) - off
    x, y = b["strain"][off:], e["strain"][:n]
    d = x - y
    # compare in the analysis band through the PSD: spectrum of the difference relative to the data
    fw, p_d = psd.welch_median(d)
    _, p_x = psd.welch_median(y)
    m = (fw >= 20) & (fw <= 1024)
    ratio = np.sqrt(p_d[m] / p_x[m])
    return {"overlap_s": n / FS, "identical": bool(np.array_equal(x, y)), "max_abs_diff": float(np.abs(d).max()),
            "max_abs_strain": float(np.abs(y).max()), "rms_diff_over_rms_strain": float(np.std(d) / np.std(y)),
            "asd_ratio_diff_over_data_20_1024": {"median": float(np.median(ratio)), "max": float(ratio.max()),
                                                 "f_at_max_hz": float(fw[m][np.argmax(ratio)])},
            "corrcoef": float(np.corrcoef(x, y)[0, 1])}


if __name__ == "__main__":
    n = const.SEG_DUR * FS
    i0 = 127 * const.SEG_STRIDE * FS
    res = {"whitened_power_event_segment": {d: whitened_power(d, i0, n) for d in ("H1", "L1")},
           "whitened_power_offsource_segment_j20": {d: whitened_power(d, 20 * const.SEG_STRIDE * FS, n) for d in ("H1", "L1")},
           "bulk_vs_event_file": {d: overlap_check(d) for d in ("H1", "L1")}}
    json.dump(res, open(ROOT / "results" / "diag_stage5.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
