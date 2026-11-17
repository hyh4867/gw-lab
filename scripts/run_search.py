"""Single-detector trigger generation over the whole 4096-s file with the coarse bank.

Usage: python scripts/run_search.py [--kind imr] [--dets H1 L1]
Checkpoint: one .npy of triggers per segment in results/triggers/<kind>/<det>/.
"""
import argparse
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gwlab import const, io, search  # noqa: E402

RAW = ROOT / "data" / "raw"


def strain_path(det):
    return RAW / f"{det[0]}-{det}_GWOSC_4KHZ_R1-{const.FILE_START}-4096.hdf5"


def segment_starts(nsamp, fs=const.FS):
    n = const.SEG_DUR * fs
    return [j * const.SEG_STRIDE * fs for j in range((nsamp - n) // (const.SEG_STRIDE * fs) + 1)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="imr")
    ap.add_argument("--dets", nargs="+", default=["H1", "L1"])
    a = ap.parse_args()
    fs, n = const.FS, const.SEG_DUR * const.FS
    bank = search.build_bank(search.coarse_masses(), n, a.kind)
    print("bank", len(bank.masses), a.kind, flush=True)
    for det in a.dets:
        out = ROOT / "results" / "triggers" / a.kind / det
        out.mkdir(parents=True, exist_ok=True)
        d = io.read_strain(strain_path(det))
        strain = d["strain"]
        starts = segment_starts(len(strain))
        t0 = time.time()
        for j, i0 in enumerate(starts):
            fn = out / f"seg_{j:03d}.npy"
            if fn.exists():
                continue
            prep = search.prepare_segment(strain, i0, n)
            if prep is None:
                np.save(fn, np.zeros(0, dtype=search.TRIG_DTYPE))
                print(det, j, "skipped (NaN)", flush=True)
                continue
            trig = search.find_triggers(prep, bank, i0)
            np.save(fn, trig)
            if j % 10 == 0:
                print(det, j, len(starts), f"{time.time() - t0:.0f}s", len(trig), flush=True)
        del strain, d
    print("done", flush=True)
