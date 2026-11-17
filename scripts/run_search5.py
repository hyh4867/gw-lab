"""Stage-5 single-detector trigger generation for one 4096-s file pair (resumable, one .npy per segment).

Usage: python scripts/run_search5.py --bank coarse|lowmass --start <gps> [--dets H1 L1]
Output: results/triggers/<bank>/<gps>/<det>/seg_XXX.npy  (empty array + skipped flag for unusable segments)
"""
import argparse
import glob
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gwlab import const, io, psd, search  # noqa: E402

RAW = ROOT / "data" / "raw"


def strain_file(det, start):
    hits = sorted(glob.glob(str(RAW / f"{det[0]}-{det}_*-{start}-4096.hdf5")))
    if len(hits) != 1:
        raise FileNotFoundError(f"{det} {start}: {hits}")
    return hits[0]


def load_bank(name):
    if name == "coarse":
        lay = search.LAYOUT_BBH
        return search.build_bank(search.coarse_masses(), lay.seg_dur * const.FS, "imr"), lay
    if name == "lowmass":
        lay = search.LAYOUT_LOWMASS
        b = np.loadtxt(ROOT / "results" / "bank_lowmass.csv", delimiter=",", skiprows=1)
        par = np.column_stack([b[:, 3], b[:, 4], b[:, 2], b[:, 2]])
        return search.build_bank(par, lay.seg_dur * const.FS, "imr"), lay
    raise ValueError(name)


def trig_dir(bank, start, det):
    return ROOT / "results" / "triggers" / bank / str(start) / det


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--start", type=int, nargs="+", required=True)
    ap.add_argument("--dets", nargs="+", default=["H1", "L1"])
    a = ap.parse_args()
    fs = const.FS
    bank, lay = load_bank(a.bank)
    n = lay.seg_dur * fs
    print("bank", a.bank, len(bank.masses), flush=True)
    for start in a.start:
        for det in a.dets:
            out = trig_dir(a.bank, start, det)
            out.mkdir(parents=True, exist_ok=True)
            if (out / "done.json").exists():
                continue
            strain = io.read_strain(strain_file(det, start))["strain"]
            stretches = psd.valid_stretches(strain)
            starts = lay.starts(len(strain))
            skipped = []
            t0 = time.time()
            for j, i0 in enumerate(starts):
                fn = out / f"seg_{j:03d}.npy"
                if fn.exists():
                    if (out / f"seg_{j:03d}.skip").exists():
                        skipped.append(j)
                    continue
                prep = search.prepare_segment(strain, i0, n, stretches=stretches)
                if prep is None:
                    (out / f"seg_{j:03d}.skip").write_text("")
                    np.save(fn, np.zeros(0, dtype=search.TRIG_DTYPE))
                    skipped.append(j)
                    continue
                trig = search.find_triggers(prep, bank, i0, v0=lay.valid_start * fs, v1=lay.valid_end * fs, decim=lay.decim)
                np.save(fn, trig)
                if j % 20 == 0:
                    print(start, det, j, len(starts), f"{time.time() - t0:.0f}s", len(trig), flush=True)
            json.dump({"n_segments": len(starts), "skipped": skipped, "seconds": time.time() - t0,
                       "stretches_s": [(x / fs, y / fs) for x, y in stretches]}, open(out / "done.json", "w"))
            print(start, det, "done", f"{time.time() - t0:.0f}s", "skipped", len(skipped), flush=True)
            del strain
    print("all done", flush=True)
