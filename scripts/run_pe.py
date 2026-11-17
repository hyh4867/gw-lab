"""Stage 4: (chirp mass, mass ratio) with emcee, everything else maximised or fixed (pre-registration section 9)."""
import json
import pathlib
import sys

import emcee
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from gwlab import const, io, mf, plotstyle, search, waveforms  # noqa: E402
from run_search import strain_path  # noqa: E402

FS = const.FS
N = 8 * FS
HALF = int(round(0.005 * FS))
LIT = {"mc": (30.4, 28.5, 32.5), "q": (0.82, 0.62, 0.99)}


def main(nwalk=32, nstep=2000, burn=500, seed=20261115):
    ev = json.load(open(ROOT / "results" / "event.json"))
    best = ev["bank"]["imr"]["fine"]
    seg0 = ev["segment"]["start_s"]
    t_h, t_l = seg0 + best["i_h"] / FS, seg0 + best["i_l"] / FS
    i0 = int(round((t_h - 6.0) * FS))
    preps, centre = {}, {}
    for det, t in (("H1", t_h), ("L1", t_l)):
        strain = io.read_strain(strain_path(det))["strain"]
        preps[det] = search.prepare_segment(strain, i0, N)
        centre[det] = int(round(t * FS)) - i0
        del strain
    sl = mf.band_slice(N)
    df = FS / N
    buf = np.zeros(N, dtype=np.complex128)
    w = {d: preps[d]["s_f"][sl] * preps[d]["inv"][sl] for d in preps}
    inv = {d: preps[d]["inv"][sl] for d in preps}

    def loglike(mc, q):
        m1, m2 = waveforms.masses_from_mc_q(mc, q)
        h = waveforms.imrphenomd(m1, m2, N)[sl]
        p2 = h.real**2 + h.imag**2
        tot = 0.0
        for d in ("H1", "L1"):
            s2 = 4 * df * np.sum(p2 * inv[d])
            buf[sl] = w[d] * np.conj(h)
            z = np.fft.ifft(buf)[centre[d] - HALF:centre[d] + HALF + 1]
            tot += np.max(z.real**2 + z.imag**2) * (4 * FS) ** 2 / s2
        return 0.5 * tot

    def logpost(x):
        mc, q = x
        if not (20 <= mc <= 45 and 0.2 <= q <= 1):
            return -np.inf
        return loglike(mc, q) + np.log(mc) - 1.2 * np.log(q) + 0.4 * np.log1p(q)

    rng = np.random.default_rng(seed)
    p0 = np.column_stack([rng.normal(best["mc"], 0.5, nwalk), np.clip(rng.normal(0.85, 0.08, nwalk), 0.3, 0.99)])
    np.random.seed(seed)
    sam = emcee.EnsembleSampler(nwalk, 2, logpost)
    sam.run_mcmc(p0, nstep, progress=False)
    chain = sam.get_chain(discard=burn, flat=True)
    lp = sam.get_log_prob(discard=burn, flat=True)
    try:
        tau = sam.get_autocorr_time(discard=burn, tol=0).tolist()
    except Exception:
        tau = None
    mc, q = chain[:, 0], chain[:, 1]
    m1, m2 = waveforms.masses_from_mc_q(mc, q)

    def q90(x):
        a, b, c = np.percentile(x, [5, 50, 95])
        return {"median": float(b), "lo": float(a), "hi": float(c)}

    res = {"n_samples": int(len(chain)), "autocorr_time_steps": tau, "acceptance": float(np.mean(sam.acceptance_fraction)),
           "mc": q90(mc), "q": q90(q), "m1": q90(m1), "m2": q90(m2), "mtot": q90(m1 + m2),
           "max_logpost": float(np.max(lp)), "map": chain[np.argmax(lp)].tolist(),
           "loglike_at_map": float(loglike(*chain[np.argmax(lp)])), "literature": LIT,
           "loglike_at_bank_best": float(loglike(best["mc"], best["m2"] / best["m1"]))}
    res["D1_mc_median_in_lit"] = bool(LIT["mc"][1] <= res["mc"]["median"] <= LIT["mc"][2])
    res["D2_q_overlap"] = bool(res["q"]["hi"] >= LIT["q"][1] and res["q"]["lo"] <= LIT["q"][2])
    json.dump(res, open(ROOT / "results" / "pe.json", "w"), indent=1)
    print(json.dumps(res, indent=1))

    plt = plotstyle.apply()
    COL = plotstyle.COL
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8))
    axs[0].hist(mc, bins=60, color=COL["H1"], density=True)
    axs[0].axvspan(LIT["mc"][1], LIT["mc"][2], color=COL["muted"], alpha=0.25, lw=0, label="literature 90 % interval")
    axs[0].axvline(LIT["mc"][0], color=COL["model"], lw=1)
    axs[0].set(xlabel="detector-frame chirp mass (M☉)", ylabel="probability density", title="chirp mass")
    axs[0].legend(fontsize=8)
    axs[1].hist(q, bins=60, color=COL["H1"], density=True)
    axs[1].axvspan(LIT["q"][1], LIT["q"][2], color=COL["muted"], alpha=0.25, lw=0)
    axs[1].axvline(LIT["q"][0], color=COL["model"], lw=1)
    axs[1].set(xlabel="mass ratio q = m2/m1", title="mass ratio")
    axs[2].plot(m1[::10], m2[::10], ".", ms=1.5, color=COL["H1"], alpha=0.4, rasterized=True)
    axs[2].errorbar([38.9], [31.6], xerr=[[4.3], [5.6]], yerr=[[4.7], [4.2]], color=COL["model"], capsize=3, marker="o", ms=4,
                    label="literature (median, 90 %)")
    axs[2].set(xlabel="m1 (M☉, detector frame)", ylabel="m2 (M☉, detector frame)", title="component masses")
    axs[2].legend(fontsize=8)
    fig.suptitle("Two-parameter profile-likelihood posterior (spins fixed to 0): narrower than the literature by construction", fontsize=10)
    fig.tight_layout(); fig.savefig(ROOT / "docs" / "figures" / "09_pe.png"); plt.close(fig)


if __name__ == "__main__":
    main()
