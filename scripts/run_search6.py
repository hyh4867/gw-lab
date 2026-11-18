"""Stage-6 binary-neutron-star search of one 4096-s H1/L1 file pair (pre-registration stage 6, section 4).

Usage: python scripts/run_search6.py --start <gps> [<gps> ...] [--variant gated|raw]
Checkpoint: data/cache/search6/<variant>/<gps>.npz every CHUNK templates (running maximum over the bank).
Output: results/triggers/bns_<variant>/<gps>/{H1,L1}.npy and done.json (gates, skipped segments, seconds).
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from gwlab import bns, const, gate, io, psd, search  # noqa: E402
from run_search5 import strain_file  # noqa: E402

FS = const.FS
CHUNK = 250
DETS = ("H1", "L1")


def load_bank():
    b = np.loadtxt(ROOT / "results" / "bank_bns.csv", delimiter=",", skiprows=1)
    return b[:, 2:4]            # m1, m2


def load_gated(det, start, variant):
    """Strain of one file, gated if variant == 'gated'. Returns (strain, gate list)."""
    x = io.read_strain(strain_file(det, start))["strain"]
    gates = []
    if variant == "gated":
        c, peak = gate.find_glitches(x)
        x = gate.apply_gates(x, c)
        gates = [{"gps": start + ci / FS, "t_file_s": ci / FS, "peak_sigma": float(p)} for ci, p in zip(c, peak)]
    return x, gates


def prepare_file(det, start, variant, basis, lay):
    x, gates = load_gated(det, start, variant)
    stretches = psd.valid_stretches(x)
    n = lay.seg_dur * FS
    preps, skipped = [], []
    for j, i0 in enumerate(lay.starts(len(x))):
        p = search.prepare_segment(x, i0, n, stretches=stretches)
        if p is None:
            skipped.append(j)
            preps.append(None)
        else:
            preps.append(bns.reduce_prep(p, basis))
    return preps, gates, skipped


def run(start, variant, masses, basis, lay):
    out = ROOT / "results" / "triggers" / f"bns_{variant}" / str(start)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "done.json").exists():
        return
    t0 = time.time()
    preps, info = [], {}
    for det in DETS:
        p, gates, skipped = prepare_file(det, start, variant, basis, lay)
        info[det] = {"gates": gates, "skipped": skipped, "n_segments": len(p)}
        preps += p
        print(start, det, "prepared", len(p), "segments,", len(gates), "gates", f"{time.time() - t0:.0f}s", flush=True)
    ck = ROOT / "data" / "cache" / "search6" / variant / f"{start}.npz"
    ck.parent.mkdir(parents=True, exist_ok=True)
    best2 = arg = None
    k0, spent = 0, 0.0
    if ck.exists():
        d = np.load(ck)
        best2, arg, k0, spent = d["best2"], d["arg"], int(d["k_done"]), float(d["seconds"])
        print(start, "resuming at template", k0, flush=True)
    t1 = time.time()
    while k0 < len(masses):
        k1 = min(k0 + CHUNK, len(masses))
        best2, arg = bns.filter_bank(basis, masses, preps, lay, FS, best2, arg, k0, k1)
        k0 = k1
        np.savez(ck, best2=best2, arg=arg, k_done=k0, seconds=spent + time.time() - t1)
        print(start, "templates", k0, len(masses), f"{spent + time.time() - t1:.0f}s", flush=True)
    filt = spent + time.time() - t1
    nseg = info["H1"]["n_segments"]
    for d, det in enumerate(DETS):
        trig = []
        for j in range(nseg):
            p = preps[d * nseg + j]
            if p is None:
                continue
            trig.append(bns.triggers(basis, masses, p, best2[d * nseg + j], arg[d * nseg + j], j * lay.stride * FS, lay))
        trig = search.cluster(np.concatenate(trig)) if trig else np.zeros(0, dtype=search.TRIG_DTYPE)
        np.save(out / f"{det}.npy", trig)
        info[det]["n_triggers"] = int(len(trig))
    info.update({"n_templates": int(len(masses)), "seconds_filter": filt, "seconds_total": spent + time.time() - t0,
                 "variant": variant})
    json.dump(info, open(out / "done.json", "w"), indent=1)
    print(start, "done", {d: info[d]["n_triggers"] for d in DETS}, f"filter {filt:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, nargs="+", required=True)
    ap.add_argument("--variant", default="gated", choices=["gated", "raw"])
    a = ap.parse_args()
    masses = load_bank()
    lay = bns.LAYOUT
    basis = bns.basis_for(lay)
    print("bank", len(masses), flush=True)
    for s in a.start:
        run(s, a.variant, masses, basis, lay)
    print("all done", flush=True)
