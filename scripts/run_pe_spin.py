"""Auxiliary PE: own profile likelihood + emcee over (chirp mass, mass ratio, one aligned spin)
(pre-registration stage 5, section 3). Usage: python scripts/run_pe_spin.py GW150914|GW151226
Writes results/pe_spin_<event>.json and results/pe_spin_<event>_samples.npz (git-ignored).
"""
import json
import pathlib
import sys
import time

import emcee
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from gwlab import const, mf, psd, search, waveforms  # noqa: E402
import evcommon  # noqa: E402

FS = const.FS
HALF = int(round(0.005 * FS))
SEED = 20261116
CFG = {"GW150914": {"dur": 8, "before": 6.0, "mc": (20.0, 45.0), "q": (0.2, 1.0)},
       "GW151226": {"dur": 16, "before": 12.0, "mc": (9.0, 10.5), "q": (0.125, 1.0)}}
CHI = (-0.9, 0.9)


def main(name, nwalk=32, nstep=3000, burn=1000):
    cfg = CFG[name]
    best = evcommon.best(name)
    n = cfg["dur"] * FS
    t = {"H1": best["seg_start_s"] + best["i_h"] / FS, "L1": best["seg_start_s"] + best["i_l"] / FS}
    i0 = int(round((t["H1"] - cfg["before"]) * FS))
    sl = mf.band_slice(n)
    df = FS / n
    w, inv, centre = {}, {}, {}
    for det in ("H1", "L1"):
        x = evcommon.load_strain(name, det)
        p = search.prepare_segment(x, i0, n, stretches=psd.valid_stretches(x))
        del x
        w[det], inv[det] = p["s_f"][sl] * p["inv"][sl], p["inv"][sl]
        centre[det] = int(round(t[det] * FS)) - i0
    buf = np.zeros(n, dtype=np.complex128)

    def loglike(mc, q, chi):
        m1, m2 = waveforms.masses_from_mc_q(mc, q)
        h = waveforms.imrphenomd(m1, m2, n, chi1=chi, chi2=chi)[sl]
        p2 = h.real**2 + h.imag**2
        tot = 0.0
        for d in ("H1", "L1"):
            s2 = 4 * df * np.sum(p2 * inv[d])
            buf[sl] = w[d] * np.conj(h)
            z = np.fft.ifft(buf)[centre[d] - HALF:centre[d] + HALF + 1]
            tot += np.max(z.real**2 + z.imag**2) * (4 * FS) ** 2 / s2
        return 0.5 * tot

    def logpost(x):
        mc, q, chi = x
        if not (cfg["mc"][0] <= mc <= cfg["mc"][1] and cfg["q"][0] <= q <= cfg["q"][1] and CHI[0] <= chi <= CHI[1]):
            return -np.inf
        return loglike(mc, q, chi) + np.log(mc) - 1.2 * np.log(q) + 0.4 * np.log1p(q)

    rng = np.random.default_rng(SEED)
    mc0 = waveforms.chirp_mass(best["m1"], best["m2"])
    q0 = min(0.95, best["m2"] / best["m1"])
    p0 = np.column_stack([mc0 * (1 + 0.003 * rng.standard_normal(nwalk)),
                          np.clip(q0 + 0.05 * rng.standard_normal(nwalk), cfg["q"][0] + 0.01, 0.99),
                          np.clip(best["chi"] + 0.03 * rng.standard_normal(nwalk), -0.8, 0.8)])
    np.random.seed(SEED)
    sam = emcee.EnsembleSampler(nwalk, 3, logpost)
    t0 = time.time()
    state = p0
    for block in range(nstep // 500):
        state = sam.run_mcmc(state, 500, progress=False)
        print(name, (block + 1) * 500, f"{time.time() - t0:.0f}s", flush=True)
    chain = sam.get_chain(discard=burn, flat=True)
    lp = sam.get_log_prob(discard=burn, flat=True)
    try:
        tau = sam.get_autocorr_time(discard=burn, tol=0).tolist()
    except Exception:
        tau = None
    mc, q, chi = chain.T
    m1, m2 = waveforms.masses_from_mc_q(mc, q)
    np.savez(ROOT / "results" / f"pe_spin_{name}_samples.npz", mc=mc, q=q, chi=chi, m1=m1, m2=m2, logpost=lp)

    def q90(x):
        a, b, c = np.percentile(x, [5, 50, 95])
        return {"median": float(b), "lo": float(a), "hi": float(c)}

    k = int(np.argmax(lp))
    res = {"event": name, "n_samples": int(len(chain)), "autocorr_time_steps": tau, "nwalk": nwalk, "nstep": nstep, "burn": burn,
           "acceptance": float(np.mean(sam.acceptance_fraction)), "seconds": time.time() - t0,
           "mc_det": q90(mc), "q": q90(q), "chi_eff": q90(chi), "m1_det": q90(m1), "m2_det": q90(m2), "mtot_det": q90(m1 + m2),
           "p_chi_positive": float(np.mean(chi > 0)), "map": chain[k].tolist(), "loglike_at_map": float(loglike(*chain[k])),
           "loglike_at_search_best": float(loglike(mc0, best["m2"] / best["m1"], best["chi"])),
           "note": "profile likelihood (amplitude, phase, time maximised per detector), chi1 = chi2 = chi"}
    json.dump(res, open(ROOT / "results" / f"pe_spin_{name}.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
