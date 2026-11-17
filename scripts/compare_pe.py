"""Compare the stage-5 posteriors with the literature and the GWTC-1 public posterior samples
(pre-registration stage 5, sections 3 and 4). Writes results/pe_compare.json and figure 16.
"""
import json
import pathlib
import sys

import h5py
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gwlab import plotstyle, waveforms  # noqa: E402

KEYS = ["mc", "q", "chieff", "m1", "m2"]
LABEL = {"mc": "chirp mass ℳ (M☉, detector frame)", "q": "mass ratio q", "chieff": "effective spin χ_eff",
         "m1": "m1 (M☉, detector frame)", "m2": "m2 (M☉, detector frame)"}
LIT = {
    "GW150914": {"m1": (38.9, 34.6, 44.5), "m2": (31.6, 26.9, 35.8), "q": (0.82, 0.62, 0.99), "chieff": (-0.07, -0.24, 0.09),
                 "mc": (30.4, 28.5, 32.5)},
    "GW151226": {"mc": (9.72, 9.66, 9.79), "q": (0.52, 0.23, 0.92), "chieff": (0.21, 0.11, 0.41),
                 "m1": (15.6, 11.6, 24.6), "m2": (8.2, 5.7, 10.8)},
}


def gwtc1(name):
    with h5py.File(ROOT / "data" / "raw" / f"{name}_GWTC-1.hdf5", "r") as f:
        d = f["Overall_posterior"][...]
    m1, m2 = d["m1_detector_frame_Msun"], d["m2_detector_frame_Msun"]
    return {"m1": m1, "m2": m2, "mc": waveforms.chirp_mass(m1, m2), "q": m2 / m1,
            "chieff": (m1 * d["spin1"] * d["costilt1"] + m2 * d["spin2"] * d["costilt2"]) / (m1 + m2)}


def q90(x):
    a, b, c = np.percentile(x, [5, 50, 95])
    return {"median": float(b), "lo": float(a), "hi": float(c)}


def overlap(a, b):
    return bool(a["lo"] <= b["hi"] and b["lo"] <= a["hi"])


def main():
    res, samples = {}, {}
    for name in ("GW150914", "GW151226"):
        r = {}
        g = gwtc1(name)
        r["gwtc1"] = {k: q90(g[k]) for k in KEYS}
        r["gwtc1"]["n_samples"] = int(len(g["m1"]))
        ours = {}
        pb = ROOT / "results" / f"pe_bilby_{name}_samples.npz"
        if pb.exists():
            z = np.load(pb)
            ours["bilby"] = {k: z[k] for k in KEYS}
        ps = ROOT / "results" / f"pe_spin_{name}_samples.npz"
        if ps.exists():
            z = np.load(ps)
            ours["emcee"] = {"mc": z["mc"], "q": z["q"], "chieff": z["chi"], "m1": z["m1"], "m2": z["m2"]}
        for tool, s in ours.items():
            summ = {k: q90(s[k]) for k in KEYS}
            cmp4 = {}
            for k in KEYS:
                a, b = summ[k], r["gwtc1"][k]
                cmp4[k] = {"our_median_in_gwtc1_90": bool(b["lo"] <= a["median"] <= b["hi"]),
                           "gwtc1_median_in_our_90": bool(a["lo"] <= b["median"] <= a["hi"]),
                           "width_ratio_ours_over_gwtc1": (a["hi"] - a["lo"]) / (b["hi"] - b["lo"])}
                cmp4[k]["both"] = cmp4[k]["our_median_in_gwtc1_90"] and cmp4[k]["gwtc1_median_in_our_90"]
            n_ok = int(sum(c["both"] for c in cmp4.values()))
            lit = LIT[name]
            v = {}
            if name == "GW150914":
                v["F1"] = {"m1_median": summ["m1"]["median"], "m2_median": summ["m2"]["median"],
                           "range_m1": lit["m1"][1:], "range_m2": lit["m2"][1:],
                           "reproduced": bool(lit["m1"][1] <= summ["m1"]["median"] <= lit["m1"][2]
                                              and lit["m2"][1] <= summ["m2"]["median"] <= lit["m2"][2])}
                v["F2"] = {"interval90": [summ["q"]["lo"], summ["q"]["hi"]], "literature": lit["q"][1:],
                           "reproduced": bool(summ["q"]["hi"] >= lit["q"][1] and summ["q"]["lo"] <= lit["q"][2]),
                           "q_equal_1_in_90": bool(summ["q"]["hi"] >= 0.99)}
                v["F3"] = {"interval90": [summ["chieff"]["lo"], summ["chieff"]["hi"]], "literature": lit["chieff"][1:],
                           "reproduced": bool(summ["chieff"]["hi"] >= lit["chieff"][1] and summ["chieff"]["lo"] <= lit["chieff"][2])}
                v["F4"] = {"n_of_5": n_ok, "reproduced": n_ok == 5}
            else:
                v["G8"] = {"mc_median": summ["mc"]["median"], "range": lit["mc"][1:],
                           "reproduced": bool(lit["mc"][1] <= summ["mc"]["median"] <= lit["mc"][2])}
                v["G9"] = {"interval90": [summ["q"]["lo"], summ["q"]["hi"]], "literature": lit["q"][1:],
                           "reproduced": bool(summ["q"]["hi"] >= lit["q"][1] and summ["q"]["lo"] <= lit["q"][2])}
                v["G10"] = {"interval90": [summ["chieff"]["lo"], summ["chieff"]["hi"]], "literature": lit["chieff"][1:],
                            "reproduced": bool(summ["chieff"]["hi"] >= lit["chieff"][1] and summ["chieff"]["lo"] <= lit["chieff"][2]),
                            "p_chi_eff_positive": float(np.mean(s["chieff"] > 0))}
                v["G11"] = {"n_of_5": n_ok, "reproduced": n_ok == 5}
            r[tool] = {"summary": summ, "vs_gwtc1": cmp4, "verdicts": v, "n_samples": int(len(s["mc"]))}
        res[name] = r
        samples[name] = {"gwtc1": g, **ours}
    json.dump(res, open(ROOT / "results" / "pe_compare.json", "w"), indent=1)
    print(json.dumps(res, indent=1))

    plt = plotstyle.apply()
    COL = plotstyle.COL
    fig, axs = plt.subplots(2, 4, figsize=(14, 6.4))
    for r, name in enumerate(("GW150914", "GW151226")):
        s = samples[name]
        for c, k in enumerate(("mc", "q", "chieff")):
            ax = axs[r, c]
            lo = min(np.percentile(v[k], 0.5) for v in s.values())
            hi = max(np.percentile(v[k], 99.5) for v in s.values())
            bins = np.linspace(lo, hi, 60)
            ax.hist(s["gwtc1"][k], bins=bins, density=True, color=COL["muted"], alpha=0.45, label="GWTC-1 public samples")
            if "bilby" in s:
                ax.hist(s["bilby"][k], bins=bins, density=True, histtype="step", color=COL["H1"], lw=1.4, label="this work: bilby, aligned spin")
            if "emcee" in s:
                ax.hist(s["emcee"][k], bins=bins, density=True, histtype="step", color=COL["L1"], lw=1.0, ls="--", label="this work: profile likelihood")
            ax.set(xlabel=LABEL[k], yticks=[])
            if c == 0:
                ax.set_ylabel(name)
            if r == 0 and c == 1:
                ax.legend(fontsize=7, loc="upper left", frameon=False)
        ax = axs[r, 3]
        g = s["gwtc1"]
        st = max(1, len(g["m1"]) // 3000)
        ax.plot(g["m1"][::st], g["m2"][::st], ".", ms=1.5, color=COL["muted"], alpha=0.5, rasterized=True)
        if "bilby" in s:
            b = s["bilby"]
            st = max(1, len(b["m1"]) // 3000)
            ax.plot(b["m1"][::st], b["m2"][::st], ".", ms=1.5, color=COL["H1"], alpha=0.5, rasterized=True)
        ax.set(xlabel=LABEL["m1"], ylabel=LABEL["m2"])
    fig.suptitle("Posteriors compared with the GWTC-1 public samples (grey). The line of points in the mass plane is the chirp-mass degeneracy.", fontsize=10)
    fig.tight_layout(); fig.savefig(ROOT / "docs" / "figures" / "16_pe_compare.png"); plt.close(fig)


if __name__ == "__main__":
    main()
