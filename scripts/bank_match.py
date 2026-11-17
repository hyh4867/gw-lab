"""Minimal match of the coarse bank: random signals in the bank's mass range against all templates.

Uses the H1 PSD of the event segment and 8-s segments. Writes results/bank_match.json.
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from gwlab import const, io, mf, search, waveforms  # noqa: E402
from run_search import strain_path  # noqa: E402

if __name__ == "__main__":
    fs, n = const.FS, 8 * const.FS
    strain = io.read_strain(strain_path("H1"))["strain"]
    prep = search.prepare_segment(strain, 2040 * fs, n)
    del strain
    bank = search.build_bank(search.coarse_masses(), n, "imr")
    sl, df = bank.sl, fs / n
    inv = prep["inv"][sl]
    H = bank.h.astype(np.complex128)
    H /= np.sqrt(4 * df * np.sum(np.abs(H) ** 2 * inv, axis=1))[:, None]
    rng = np.random.default_rng(7)
    out = []
    buf = np.zeros(n, dtype=np.complex128)
    for _ in range(300):
        a, b = rng.uniform(20, 80, 2)
        m1, m2 = max(a, b), min(a, b)
        s = waveforms.imrphenomd(m1, m2, n)[sl] * 1e20
        s /= np.sqrt(4 * df * np.sum(np.abs(s) ** 2 * inv))
        # pre-select the 12 nearest templates in (chirp mass, total mass) to save time
        d = np.hypot(waveforms.chirp_mass(*bank.masses.T) - waveforms.chirp_mass(m1, m2), 0.3 * (bank.masses.sum(1) - m1 - m2))
        best = 0.0
        for k in np.argsort(d)[:12]:
            buf[sl] = s * np.conj(H[k]) * inv
            best = max(best, 4 * fs * np.abs(np.fft.ifft(buf)).max())
        out.append((m1, m2, best))
    out = np.array(out)
    res = {"n": len(out), "min_match": float(out[:, 2].min()), "p01": float(np.percentile(out[:, 2], 1)),
           "p10": float(np.percentile(out[:, 2], 10)), "median": float(np.median(out[:, 2])),
           "worst": out[np.argmin(out[:, 2])].tolist(),
           "gw150914_like_38.9_31.6": None}
    s = waveforms.imrphenomd(38.9, 31.6, n)[sl] * 1e20
    s /= np.sqrt(4 * df * np.sum(np.abs(s) ** 2 * inv))
    best = 0.0
    for k in range(len(H)):
        buf[sl] = s * np.conj(H[k]) * inv
        best = max(best, 4 * fs * np.abs(np.fft.ifft(buf)).max())
    res["gw150914_like_38.9_31.6"] = float(best)
    json.dump(res, open(ROOT / "results" / "bank_match.json", "w"), indent=1)
    print(res)
