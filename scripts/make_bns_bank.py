"""Stochastic non-spinning TaylorF2 bank for the binary-neutron-star search (pre-registration stage 6).

Two passes: (1) proposals uniform in chirp time and eta until 150 consecutive rejections;
(2) top-up with proposals uniform in (m1, m2) until 300 consecutive rejections. Pass 2 was added because the
fitting-factor test of pass 1 found holes (FF 0.44) in the corners of the mass range near (1, 1) and (2, 2),
where the chirp-time proposal almost never lands. `--topup-only` starts from the pass-1 bank in
results/bank_bns_pass1.csv instead of repeating pass 1.

Uses only the PSD of the GW150914 file (first 512 s of H1, O1 noise), i.e. no stage-6 data.
Writes results/bank_bns.csv and results/bank_bns.json (fitting-factor test, also against IMRPhenomD).
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

SEED = 20261117
M_MIN, M_MAX = 1.0, 2.0                      # detector-frame component masses
F_LOW = 30.0
MIN_MATCH = 0.95
MAX_REJECT = 150
TAU_WINDOW = 0.01
N_TEST = 500
TOPUP_REJECT = 300


def uniform_masses(rng):
    a, b = rng.uniform(M_MIN, M_MAX, 2)
    m1, m2 = max(a, b), min(a, b)
    return waveforms.chirp_mass(m1, m2), m1 * m2 / (m1 + m2) ** 2, 0.0


def draw_test(rng, n):
    """Half from the chirp-time proposal, half uniform in (m1, m2)."""
    out = []
    while len(out) < n // 2:
        q = bankgen.draw(rng, *REGION)
        a, b = waveforms.masses_from_mc_eta(q[0], q[1])
        if b >= M_MIN and a <= M_MAX:
            out.append(q)
    out += [uniform_masses(rng) for _ in range(n - len(out))]
    return np.array(out)
REGION = (waveforms.chirp_mass(M_MIN, M_MIN), waveforms.chirp_mass(M_MAX, M_MAX),
          M_MIN * M_MAX / (M_MIN + M_MAX) ** 2, 0.0, 0.0)

if __name__ == "__main__":
    x = io.read_strain(strain_path("H1"))["strain"][:512 * 4096]
    f, p = psd.welch_median(x)
    del x
    sp = bankgen.MatchSpace(f, p, f_low=F_LOW, kind="f2")
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    pass1 = ROOT / "results" / "bank_bns_pass1.csv"
    if "--topup-only" in sys.argv:
        b = np.loadtxt(pass1, delimiter=",", skiprows=1)
        P = np.column_stack([b[:, 0], b[:, 1], np.zeros(len(b))])
        U = np.array([sp.unit(*q) for q in P])
        nprop = int(json.load(open(ROOT / "results" / "bank_bns_pass1.json"))["n_proposals"])
        rng = np.random.default_rng(SEED + 1)
    else:
        P, U, nprop = bankgen.stochastic_bank(
            sp, rng, REGION, MIN_MATCH, MAX_REJECT, m2_min=M_MIN, m1_max=M_MAX, tau_window=TAU_WINDOW,
            log=lambda k, n: print(k, n, f"{time.time() - t0:.0f}s", flush=True))
        np.savetxt(pass1, P[:, :2], delimiter=",", fmt="%.8f", header="mc,eta", comments="")
        rng = np.random.default_rng(SEED + 1)
    n1 = len(P)
    P, U, nprop2 = bankgen.top_up(sp, rng, P, U, uniform_masses, MIN_MATCH, TOPUP_REJECT, TAU_WINDOW,
                                  log=lambda k, n: print("top-up", k, n, f"{time.time() - t0:.0f}s", flush=True))
    m = np.array([waveforms.masses_from_mc_eta(a, b) for a, b, _ in P])
    o = np.argsort(P[:, 0])
    np.savetxt(ROOT / "results" / "bank_bns.csv", np.column_stack([P[:, :2], m])[o], delimiter=",", fmt="%.8f",
               header="mc,eta,m1,m2", comments="")
    t_bank = time.time() - t0
    print("bank", len(P), "pass 1", n1, "proposals", nprop, "+", nprop2, f"{t_bank:.0f}s", flush=True)
    test = draw_test(np.random.default_rng(SEED + 2), N_TEST)
    ff = bankgen.fitting_factors(sp, U, P, test, tau_window=TAU_WINDOW)
    # the same test points as IMRPhenomD signals (no tides, no spin): what the TaylorF2 bank loses on them
    spi = bankgen.MatchSpace(f, p, f_low=F_LOW, kind="imr")
    tau_b = waveforms.chirp_time(P[:, 0], F_LOW)
    ffi = np.zeros(100)
    for i, q in enumerate(test[::5]):
        u = spi.unit(*q)
        near = np.flatnonzero(np.abs(tau_b - waveforms.chirp_time(q[0], F_LOW)) <= TAU_WINDOW * tau_b)
        ffi[i] = max(float(sp.match(u, U[near[c:c + 256]]).max()) for c in range(0, len(near), 256))

    def summ(v):
        return {"n": int(len(v)), "min": float(v.min()), "p01": float(np.percentile(v, 1)),
                "p10": float(np.percentile(v, 10)), "median": float(np.median(v)),
                "frac_below_0.95": float(np.mean(v < 0.95)), "frac_below_0.90": float(np.mean(v < 0.90))}
    res = {"seed": SEED, "component_mass_range": [M_MIN, M_MAX], "f_low": F_LOW, "min_match": MIN_MATCH,
           "max_reject": MAX_REJECT, "topup_max_reject": TOPUP_REJECT, "tau_window": TAU_WINDOW,
           "n_templates": int(len(P)), "n_templates_pass1": int(n1), "n_proposals": int(nprop),
           "n_proposals_topup": int(nprop2), "seconds_this_run": t_bank,
           "fitting_factor_taylorf2_signals": summ(ff), "fitting_factor_imrphenomd_signals": summ(ffi),
           "psd": "O1 H1, first 512 s of the GW150914 file",
           "note": "test points: half uniform in chirp time and eta, half uniform in (m1, m2); every fifth one as IMRPhenomD"}
    json.dump(res, open(ROOT / "results" / "bank_bns.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
