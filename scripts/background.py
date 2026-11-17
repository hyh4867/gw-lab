"""Time-slide background and false-alarm rate (pre-registration section 6). Writes results/background.json."""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from gwlab import coinc, const, plotstyle  # noqa: E402
from analyze_event import load_triggers  # noqa: E402

YEAR = 365.25 * 86400
SPAN = 4096 - const.VALID_START - (const.SEG_DUR - const.VALID_END)  # 4080 s


def run(th, tl, label):
    i, j, k = coinc.slide_coincidences(th["t"] - const.VALID_START, tl["t"] - const.VALID_START, SPAN)
    stat = coinc.combined(th["newsnr"][i], tl["newsnr"][j])
    raw = coinc.combined(th["rho"][i], tl["rho"][j])
    return {"label": label, "n_h": len(th), "n_l": len(tl), "n_pairs": len(th) * len(tl),
            "n_background_coinc": len(stat), "n_slides_with_coinc": int(len(np.unique(k))),
            "loudest_newsnr_c": float(stat.max()) if len(stat) else None,
            "loudest_raw_c": float(raw.max()) if len(stat) else None}, stat


if __name__ == "__main__":
    th, tl = load_triggers("imr", "H1"), load_triggers("imr", "L1")
    zi, zj = coinc.zero_lag(th["t"], tl["t"])
    zstat = coinc.combined(th["newsnr"][zi], tl["newsnr"][zj])
    o = int(np.argmax(zstat))
    t_ev, ev_stat = float(th["t"][zi[o]]), float(zstat[o])
    ns = coinc.n_slides(SPAN)
    t_bg = ns * SPAN
    res = {"span_s": SPAN, "n_slides": ns, "t_bg_s": t_bg, "t_bg_yr": t_bg / YEAR,
           "event": {"t_file_s": t_ev, "gps": const.FILE_START + t_ev, "newsnr_c": ev_stat,
                     "raw_c": float(coinc.combined(th["rho"][zi[o]], tl["rho"][zj[o]]))},
           "n_zero_lag": len(zstat),
           "single": {d: {"n": len(t), "rho_sorted_top": np.sort(t["rho"])[::-1][:6].tolist(),
                          "newsnr_sorted_top": np.sort(t["newsnr"])[::-1][:6].tolist(),
                          "max_chi2r": float(t["chi2r"].max())} for d, t in (("H1", th), ("L1", tl))}}
    mh, ml = np.abs(th["t"] - t_ev) > 1.0, np.abs(tl["t"] - t_ev) > 1.0
    main, stat_main = run(th[mh], tl[ml], "event removed (main)")
    incl, stat_incl = run(th, tl, "event kept")
    for r, s in ((main, stat_main), (incl, stat_incl)):
        n = int(np.sum(s >= ev_stat))
        r["n_louder_than_event"] = n
        r["far_per_yr"] = n / (t_bg / YEAR) if n else None
        r["far_upper_bound_per_yr"] = (1.0 if n == 0 else None) and 1.0 / (t_bg / YEAR)
        r["far_poisson90_upper_per_yr"] = 2.303 / (t_bg / YEAR) if n == 0 else None
    res["main"], res["event_kept"] = main, incl
    res["threshold_newsnr_c"] = main["loudest_newsnr_c"]
    res["background_stats_main"] = np.sort(stat_main)[::-1].tolist()
    json.dump(res, open(ROOT / "results" / "background.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "background_stats_main"}, indent=1))

    plt = plotstyle.apply()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for s, lab, c, ls in ((stat_main, "time-slide background, event removed", plotstyle.COL["H1"], "-"),
                          (stat_incl, "time-slide background, event kept", plotstyle.COL["muted"], "--")):
        x = np.sort(s)[::-1]
        ax.step(x, np.arange(1, len(x) + 1) / (t_bg / YEAR), where="post", color=c, ls=ls, label=lab)
    ax.axvline(ev_stat, color=plotstyle.COL["L1"], lw=1.5)
    ax.text(ev_stat - 0.3, 8, f"zero-lag candidate\nρ̂c = {ev_stat:.1f}", ha="right", color="#0b0b0b", fontsize=9)
    ax.axhline(1 / (t_bg / YEAR), color="#52514e", lw=0.8, ls=":")
    ax.text(7.2, 1.25 / (t_bg / YEAR), f"1 / T_bg = 1 per {t_bg / YEAR:.2f} yr", fontsize=8, color="#52514e")
    ax.set(yscale="log", xlabel="combined re-weighted SNR ρ̂c", ylabel="cumulative rate (per year)",
           title="Background from 0.1-s time slides of 4080 s of data", xlim=(7, 25))
    ax.legend(loc="upper center")
    fig.tight_layout(); fig.savefig(ROOT / "docs" / "figures" / "07_background.png"); plt.close(fig)
