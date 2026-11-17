"""Stochastic aligned-spin bank for the low-mass search (pre-registration stage 5, section 4).

Uses only the PSD of the GW150914 file (first 512 s of H1), i.e. no new data.
Writes results/bank_lowmass.csv and results/bank_lowmass.json (fitting-factor test).
"""
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from gwlab import bankgen, io, psd, waveforms  # noqa: E402
from run_search import strain_path  # noqa: E402

SEED = 20261116
REGION = (7.0, 13.0, 0.10, -0.3, 0.6)  # Mc_lo, Mc_hi, eta_lo, chi_lo, chi_hi (detector frame)
M2_MIN = 3.0
MIN_MATCH = 0.95
MAX_REJECT = 150
N_TEST = 500

if __name__ == "__main__":
    x = io.read_strain(strain_path("H1"))["strain"][:512 * 4096]
    f, p = psd.welch_median(x)
    del x
    sp = bankgen.MatchSpace(f, p)
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    P, U, nprop = bankgen.stochastic_bank(
        sp, rng, REGION, MIN_MATCH, MAX_REJECT, m2_min=M2_MIN,
        log=lambda k, n: print(k, n, f"{time.time() - t0:.0f}s", flush=True))
    m = np.array([waveforms.masses_from_mc_eta(a, b) for a, b, _ in P])
    out = np.column_stack([P, m])
    np.savetxt(ROOT / "results" / "bank_lowmass.csv", out, delimiter=",", fmt="%.6f",
               header="mc,eta,chi,m1,m2", comments="")
    print("bank", len(P), "proposals", nprop, f"{time.time() - t0:.0f}s", flush=True)
    test = []
    while len(test) < N_TEST:
        q = bankgen.draw(rng, *REGION)
        if waveforms.masses_from_mc_eta(q[0], q[1])[1] >= M2_MIN:
            test.append(q)
    ff = bankgen.fitting_factors(sp, U, P, np.array(test))
    res = {"seed": SEED, "region": REGION, "m2_min": M2_MIN, "min_match": MIN_MATCH, "max_reject": MAX_REJECT,
           "n_templates": int(len(P)), "n_proposals": int(nprop), "seconds": time.time() - t0,
           "fitting_factor_test": {"n": N_TEST, "min": float(ff.min()), "p01": float(np.percentile(ff, 1)),
                                   "p10": float(np.percentile(ff, 10)), "median": float(np.median(ff)),
                                   "frac_below_0.95": float(np.mean(ff < 0.95)), "frac_below_0.90": float(np.mean(ff < 0.90))},
           "note": "test points drawn from the proposal distribution; tau window 12 % for neighbours"}
    json.dump(res, open(ROOT / "results" / "bank_lowmass.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
