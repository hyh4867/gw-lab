"""Efficiency curve from results/injections.npz and results/background.json. Writes results/injections.json."""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gwlab import plotstyle  # noqa: E402


def wilson(k, n, z=1.0):
    p = k / n
    den = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return c - h, c + h


def d50(d, e):
    for a in range(len(d) - 1):
        if e[a] >= 0.5 > e[a + 1]:
            return float(d[a] + (e[a] - 0.5) / (e[a] - e[a + 1]) * (d[a + 1] - d[a]))
    return None


def sens_range(d, e):
    """R with (4/3) pi R^3 = 4 pi int eff D^2 dD (trapezoid, eff(0) = eff at the first distance)."""
    dd = np.concatenate([[0.0], d])
    ee = np.concatenate([[e[0]], e])
    y = ee * dd**2
    v = np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(dd))
    return float((3 * v) ** (1 / 3))


if __name__ == "__main__":
    z = np.load(ROOT / "results" / "injections.npz")
    cols = list(z["cols"])
    r = z["res"][z["done"]]
    c = {k: r[:, i] for i, k in enumerate(cols)}
    np.savetxt(ROOT / "results" / "injections.csv", r, delimiter=",", header=",".join(cols), comments="", fmt="%.6g")
    thr = json.load(open(ROOT / "results" / "background.json"))["threshold_newsnr_c"]
    dists = np.unique(c["dist"])
    stat = np.where(c["found_coinc"] == 1, c["newsnr_c"], 0.0)
    opt_net = np.hypot(c["opt_h"], c["opt_l"])
    out = {"n_done": int(len(r)), "threshold_main": thr, "threshold_alt": 8.0, "curves": {}}
    curves = {}
    for name, t in (("main", thr), ("alt_rhohat8", 8.0)):
        det = stat >= t
        eff, lo, hi, ks, ns = [], [], [], [], []
        for d in dists:
            m = c["dist"] == d
            k, n = int(det[m].sum()), int(m.sum())
            a, b = wilson(k, n)
            eff.append(k / n); lo.append(a); hi.append(b); ks.append(k); ns.append(n)
        eff = np.array(eff)
        curves[name] = (eff, np.array(lo), np.array(hi))
        out["curves"][name] = {"dist_mpc": dists.tolist(), "found": ks, "n": ns, "eff": eff.tolist(),
                               "wilson68_lo": lo, "wilson68_hi": hi, "d50_mpc": d50(dists, eff),
                               "range_mpc": sens_range(dists, eff),
                               "monotonic_within_errors": bool(all(lo[i + 1] <= hi[i] for i in range(len(eff) - 1)))}
    # semi-analytic expectation: optimal SNRs only (no noise): both >= 5.5 and network >= threshold
    exp = [(float(np.mean(((c["opt_h"] >= 5.5) & (c["opt_l"] >= 5.5) & (opt_net >= thr))[c["dist"] == d]))) for d in dists]
    out["expected_from_optimal_snr"] = exp
    # C1: recovered / optimal network SNR for optimal > 10 and found
    m = (opt_net > 10) & (c["found_coinc"] == 1)
    ratio = np.hypot(c["rho_h"][m], c["rho_l"][m]) / opt_net[m]
    out["C1"] = {"n": int(m.sum()), "n_opt_gt10": int((opt_net > 10).sum()), "median_ratio": float(np.median(ratio)),
                 "p16": float(np.percentile(ratio, 16)), "p84": float(np.percentile(ratio, 84))}
    mm = c["found_coinc"] == 1
    out["timing"] = {"median_abs_dt_h_ms": float(np.median(np.abs(c["dt_h"][mm])) * 1e3)}
    out["chi2r_found_median"] = {"H1": float(np.median(c["chi_h"][mm])), "L1": float(np.median(c["chi_l"][mm]))}
    json.dump(out, open(ROOT / "results" / "injections.json", "w"), indent=1)
    print(json.dumps(out, indent=1))

    plt = plotstyle.apply()
    COL = plotstyle.COL
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4))
    ax = axs[0]
    e, lo, hi = curves["main"]
    ax.errorbar(dists, e, yerr=[e - lo, hi - e], color=COL["H1"], marker="o", ms=5, capsize=2, label=f"ρ̂c ≥ {thr:.2f} (loudest background)")
    e2, lo2, hi2 = curves["alt_rhohat8"]
    ax.errorbar(dists + 15, e2, yerr=[e2 - lo2, hi2 - e2], color=COL["L1"], marker="s", ms=4, capsize=2, ls="--", label="ρ̂c ≥ 8 (reference)")
    ax.plot(dists, exp, color=COL["muted"], ls=":", marker="x", label="expected from optimal SNR, no noise")
    ax.set(xlabel="luminosity distance (Mpc)", ylabel="detection efficiency", ylim=(-0.03, 1.03),
           title="Recovered fraction of injected (38.9, 31.6) M☉ signals")
    ax.legend(fontsize=8)
    ax = axs[1]
    ax.plot(opt_net[mm], np.hypot(c["rho_h"][mm], c["rho_l"][mm]), "o", ms=3, color=COL["H1"], alpha=0.6)
    lim = [5, max(60, 0)]
    ax.plot(lim, lim, color=COL["model"], lw=0.8)
    ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="optimal network SNR (from PSD and injected waveform)",
           ylabel="recovered network SNR", title="Recovered vs optimal SNR (coincident recoveries)")
    fig.tight_layout(); fig.savefig(ROOT / "docs" / "figures" / "08_injections.png"); plt.close(fig)
