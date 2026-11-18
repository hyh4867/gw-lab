"""GW170817: compare the stage-6 posterior with the literature and the GWTC-1 public samples (low-spin prior)
(pre-registration stage 6, section 7). Writes results/pe_compare_GW170817.json and figure 23.
"""
import json
import pathlib
import sys

import h5py
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from gwlab import plotstyle, waveforms  # noqa: E402
import ev6common as E  # noqa: E402
from run_pe_bns import sky_area_90  # noqa: E402

KEYS = ["mc", "q", "chieff", "m1", "m2"]
LABEL = {"mc": "chirp mass ℳ (M☉, detector frame)", "q": "mass ratio q", "chieff": "effective spin χ_eff"}


def gwtc1():
    with h5py.File(ROOT / "data" / "raw" / "GW170817_GWTC-1.hdf5", "r") as f:
        names = list(f.keys())
        d = f["IMRPhenomPv2NRT_lowSpin_posterior"][...]
    m1, m2 = d["m1_detector_frame_Msun"], d["m2_detector_frame_Msun"]
    out = {"m1": m1, "m2": m2, "mc": waveforms.chirp_mass(m1, m2), "q": m2 / m1,
           "chieff": (m1 * d["spin1"] * d["costilt1"] + m2 * d["spin2"] * d["costilt2"]) / (m1 + m2),
           "dl": d["luminosity_distance_Mpc"], "ra": d["right_ascension"], "dec": d["declination"]}
    return out, names


def q90(x):
    a, b, c = np.percentile(x, [5, 50, 95])
    return {"median": float(b), "lo": float(a), "hi": float(c)}


def main():
    g, names = gwtc1()
    z = np.load(ROOT / "results" / "pe_bilby_GW170817_samples.npz")
    s = {k: z[k] for k in KEYS + ["dl", "ra", "dec"]}
    summ = {k: q90(s[k]) for k in KEYS + ["dl"]}
    gs = {k: q90(g[k]) for k in KEYS + ["dl"]}
    cmp5 = {}
    for k in KEYS:
        a, b = summ[k], gs[k]
        cmp5[k] = {"our_median_in_gwtc1_90": bool(b["lo"] <= a["median"] <= b["hi"]),
                   "gwtc1_median_in_our_90": bool(a["lo"] <= b["median"] <= a["hi"]),
                   "width_ratio_ours_over_gwtc1": (a["hi"] - a["lo"]) / (b["hi"] - b["lo"])}
        cmp5[k]["both"] = cmp5[k]["our_median_in_gwtc1_90"] and cmp5[k]["gwtc1_median_in_our_90"]
    n_ok = int(sum(c["both"] for c in cmp5.values()))
    m1s, m2s = s["m1"] / (1 + E.Z), s["m2"] / (1 + E.Z)
    area, inside, pix = sky_area_90(s["ra"], s["dec"], E.RA, E.DEC)
    garea, ginside, _ = sky_area_90(g["ra"], g["dec"], E.RA, E.DEC)

    def ov(a, lo, hi):
        return bool(a["lo"] <= hi and a["hi"] >= lo)
    v = {"M3": {"interval90": [summ["mc"]["lo"], summ["mc"]["hi"]], "literature": [1.1974, 1.1976],
                "reproduced": ov(summ["mc"], 1.1974, 1.1976)},
         "M4": {"interval90": [summ["q"]["lo"], summ["q"]["hi"]], "literature": [0.7, 1.0], "reproduced": ov(summ["q"], 0.7, 1.0)},
         "M5": {"m1_source_median": float(np.median(m1s)), "m2_source_median": float(np.median(m2s)),
                "range_m1": [1.36, 1.60], "range_m2": [1.17, 1.36],
                "reproduced": bool(1.36 <= np.median(m1s) <= 1.60 and 1.17 <= np.median(m2s) <= 1.36)},
         "M6": {"interval90": [summ["dl"]["lo"], summ["dl"]["hi"]], "literature": [26.0, 48.0], "reproduced": ov(summ["dl"], 26.0, 48.0)},
         "M7": {"n_of_5": n_ok, "reproduced": n_ok == 5},
         "M8": {"counterpart_inside_90": inside, "area90_deg2": area, "pixel_deg2": pix, "reproduced": bool(inside)}}
    res = {"gwtc1_datasets_in_file": names, "gwtc1_lowspin": dict(gs, n_samples=int(len(g["m1"])),
                                                                  sky_area90_deg2_same_method=garea, counterpart_inside_90=ginside),
           "bilby": {"summary": summ, "vs_gwtc1": cmp5, "n_samples": int(len(s["mc"])),
                     "m1_source": q90(m1s), "m2_source": q90(m2s), "mtot_source": q90(m1s + m2s)},
           "verdicts": v}
    json.dump(res, open(ROOT / "results" / "pe_compare_GW170817.json", "w"), indent=1)
    print(json.dumps(res, indent=1))

    plt = plotstyle.apply()
    COL = plotstyle.COL
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.6))
    for c, k in enumerate(("mc", "q", "chieff")):
        ax = axs[c]
        lo = min(np.percentile(v_[k], 0.5) for v_ in (g, s))
        hi = max(np.percentile(v_[k], 99.5) for v_ in (g, s))
        bins = np.linspace(lo, hi, 60)
        off, sc = (1.197, 1e4) if k == "mc" else (0.0, 1.0)
        bins = (bins - off) * sc
        ax.hist((g[k] - off) * sc, bins=bins, density=True, color=COL["muted"], alpha=0.45, label="GWTC-1 public samples (low spin)")
        ax.hist((s[k] - off) * sc, bins=bins, density=True, histtype="step", color=COL["H1"], lw=1.4, label="this work: bilby, no tides")
        ax.set(xlabel="(ℳ − 1.197 M☉) × 10⁴, detector frame" if k == "mc" else LABEL[k], yticks=[])
        ax.grid(False)
        if c == 0:
            fig.legend(*ax.get_legend_handles_labels(), fontsize=8, loc="upper left", bbox_to_anchor=(0.01, 0.93), ncol=2, frameon=False)
    ax = axs[3]
    st = max(1, len(g["ra"]) // 3000)
    ax.plot(np.rad2deg(g["ra"][::st]), np.rad2deg(g["dec"][::st]), ".", ms=1.5, color=COL["muted"], alpha=0.5, rasterized=True)
    st = max(1, len(s["ra"]) // 3000)
    ax.plot(np.rad2deg(s["ra"][::st]), np.rad2deg(s["dec"][::st]), ".", ms=1.5, color=COL["H1"], alpha=0.5, rasterized=True)
    ax.plot(np.rad2deg(E.RA), np.rad2deg(E.DEC), marker="*", ms=11, color=COL["L1"], mec="#0b0b0b", mew=0.5, ls="")
    ax.invert_xaxis()
    ax.set(xlabel="right ascension (deg)", ylabel="declination (deg)")
    fig.suptitle("GW170817 posterior compared with the GWTC-1 public samples (grey). Star: optical counterpart.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(ROOT / "docs" / "figures" / "23_pe_gw170817.png"); plt.close(fig)


if __name__ == "__main__":
    main()
