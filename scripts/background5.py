"""Stage-5 extended time-slide background (pre-registration stage 5, section 5).

Usage: python scripts/background5.py GW150914|GW151226
Writes results/background_ext_<event>.json and docs/figures/1x_background_ext_<event>.png
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from gwlab import coinc, const, plotstyle, search  # noqa: E402

YEAR = 365.25 * 86400
CFG = {
    "GW150914": {"bank": "coarse", "layout": search.LAYOUT_BBH, "event_file": 1126257415, "t0": const.T0_GW150914,
                 "fig": "10_background_ext_GW150914.png"},
    "GW151226": {"bank": "lowmass", "layout": search.LAYOUT_LOWMASS, "event_file": 1135134303, "t0": 1135136350.6,
                 "fig": "11_background_ext_GW151226.png"},
}


def load_file(bank, start, lay, legacy=False):
    """Triggers (clustered, newsnr >= 5) of both detectors and the coincident live intervals of one file."""
    trig, skipped = {}, set()
    for det in ("H1", "L1"):
        d = ROOT / "results" / "triggers" / ("imr" if legacy else f"{bank}/{start}") / det
        files = sorted(d.glob("seg_*.npy"))
        t = search.cluster(np.concatenate([np.load(f) for f in files]))
        trig[det] = t[t["newsnr"] >= const.NEWSNR_MIN]
        if not legacy:
            skipped |= set(json.load(open(d / "done.json"))["skipped"])
        nseg = len(files)
    live = coinc.merge_intervals([(j * lay.stride + lay.valid_start, j * lay.stride + lay.valid_end)
                                  for j in range(nseg) if j not in skipped])
    return trig, live, sorted(skipped)


def analyse(th, tl, span, ev_stat, label):
    i, j, k = coinc.slide_coincidences(th["ring"], tl["ring"], span)
    stat = coinc.combined(th["newsnr"][i], tl["newsnr"][j])
    ns = coinc.n_slides(span)
    t_bg = ns * span
    pairs = coinc.all_pairs_stat(th["newsnr"], tl["newsnr"])
    keep = coinc.no_reuse(stat, i, j)
    top = np.argsort(stat)[::-1][:10]
    out = {"label": label, "n_h": len(th), "n_l": len(tl), "n_pairs": int(len(pairs)), "n_slides": ns,
           "span_s": span, "t_bg_yr": t_bg / YEAR, "n_background_coinc": int(len(stat)),
           "loudest_newsnr_c": float(stat.max()), "top10": np.sort(stat)[::-1][:10].tolist(),
           "top10_distinct_h_triggers": int(len(set(i[top]))), "top10_distinct_l_triggers": int(len(set(j[top]))),
           "n_louder_than_event": int(np.sum(stat >= ev_stat)),
           "all_pairs": {"n_pairs_louder_than_event": int(np.sum(pairs >= ev_stat)),
                         "loudest_possible_pair": float(pairs.max()),
                         "rate_bound_per_yr_if_zero": coinc.all_pairs_rate(1, span) * YEAR,
                         "equivalent_t_bg_yr": 1.0 / (coinc.all_pairs_rate(1, span) * YEAR)},
           "no_reuse": {"n": int(len(keep)), "top10": np.sort(stat[keep])[::-1][:10].tolist()},
           "rates_per_yr": {}}
    n = out["n_louder_than_event"]
    out["far_per_yr"] = n / out["t_bg_yr"]
    out["far_upper_bound_per_yr"] = 1.0 / out["t_bg_yr"] if n == 0 else None
    out["far_poisson90_upper_per_yr"] = 2.303 / out["t_bg_yr"] if n == 0 else None
    for x in (8.0, 8.5, 9.0):
        out["rates_per_yr"][str(x)] = {
            "slides": float(np.sum(stat >= x) / out["t_bg_yr"]),
            "all_pairs": float(coinc.all_pairs_rate(np.sum(pairs >= x), span) * YEAR),
            "no_reuse_biased_low": float(np.sum(stat[keep] >= x) / out["t_bg_yr"])}
    return out, stat, stat[keep]


def main(name):
    cfg = CFG[name]
    lay = cfg["layout"]
    meta = json.load(open(ROOT / "data" / "meta" / "background_files.json"))[name]
    starts = sorted([cfg["event_file"]] + [c["gps_start"] for c in meta["chosen"]])
    H, L, files, pos = [], [], [], 0.0
    for fi, start in enumerate(starts):
        legacy = name == "GW150914" and start == cfg["event_file"]  # stage-3 triggers of the event file
        trig, live, skipped = load_file(cfg["bank"], start, lay, legacy)
        length = 0.0
        for det, acc in (("H1", H), ("L1", L)):
            t = trig[det]
            ring, keep, length = coinc.to_ring(t["t"], live, pos)
            rec = np.zeros(int(keep.sum()), dtype=[("ring", "f8"), ("gps", "f8"), ("newsnr", "f8"), ("rho", "f8"), ("file", "i4")])
            rec["ring"], rec["gps"], rec["newsnr"], rec["rho"], rec["file"] = ring[keep], start + t["t"][keep], t["newsnr"][keep], t["rho"][keep], fi
            acc.append(rec)
        files.append({"gps_start": start, "live_s": length, "n_skipped_segments": len(skipped),
                      "n_h": int(len(H[-1])), "n_l": int(len(L[-1])), "event_file": start == cfg["event_file"]})
        pos += length
    th, tl, span = np.concatenate(H), np.concatenate(L), pos
    # zero lag (per file, in GPS time)
    zi, zj = coinc.zero_lag(th["gps"], tl["gps"])
    zstat = coinc.combined(th["newsnr"][zi], tl["newsnr"][zj])
    order = np.argsort(zstat)[::-1]
    zero = [{"gps": float(th["gps"][zi[o]]), "newsnr_c": float(zstat[o]), "rho_h": float(th["rho"][zi[o]]),
             "rho_l": float(tl["rho"][zj[o]]), "file": int(starts[th["file"][zi[o]]])} for o in order[:8]]
    ev = zero[0]
    res = {"event": name, "files": files, "span_s": span, "zero_lag_top": zero, "n_zero_lag": int(len(zstat)),
           "loudest_is_event": bool(abs(ev["gps"] - cfg["t0"]) <= 0.1),
           "loudest_zero_lag_outside_event": next((z for z in zero if abs(z["gps"] - cfg["t0"]) > 1.0), None)}
    ev_stat = ev["newsnr_c"]
    mh, ml = np.abs(th["gps"] - ev["gps"]) > 1.0, np.abs(tl["gps"] - ev["gps"]) > 1.0
    res["main"], stat_main, stat_nr = analyse(th[mh], tl[ml], span, ev_stat, "event removed (main)")
    res["event_kept"], stat_incl, _ = analyse(th, tl, span, ev_stat, "event kept")
    # jackknife over files (event removed)
    jk = {x: [] for x in (8.0, 8.5, 9.0)}
    for fi in range(len(starts)):
        a, b = th[mh], tl[ml]
        a, b = a[a["file"] != fi], b[b["file"] != fi]
        sp = span - files[fi]["live_s"]
        # re-pack the ring without the removed file
        for arr in (a, b):
            arr["ring"] = arr["ring"] - np.where(arr["file"] > fi, files[fi]["live_s"], 0.0)
        i, j, _ = coinc.slide_coincidences(a["ring"], b["ring"], sp)
        s = coinc.combined(a["newsnr"][i], b["newsnr"][j])
        tb = coinc.n_slides(sp) * sp / YEAR
        for x in jk:
            jk[x].append(float(np.sum(s >= x) / tb))
    nf = len(starts)
    res["jackknife_by_file"] = {str(x): {"rates_per_yr": v, "full_rate_per_yr": res["main"]["rates_per_yr"][str(x)]["slides"],
                                         "se_per_yr": float(np.sqrt((nf - 1) / nf * np.sum((np.array(v) - np.mean(v)) ** 2)))}
                                for x, v in jk.items()}
    res["H_verdict_event_louder_than_all_background"] = bool(res["main"]["n_louder_than_event"] == 0)
    json.dump(res, open(ROOT / "results" / f"background_ext_{name}.json", "w"), indent=1)
    print(json.dumps(res, indent=1))

    plt = plotstyle.apply()
    t_bg = res["main"]["t_bg_yr"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for s, lab, c, ls in ((stat_main, "time slides, event removed", plotstyle.COL["H1"], "-"),
                          (stat_nr, "each trigger used once (biased low)", plotstyle.COL["muted"], "--")):
        x = np.sort(s)[::-1]
        ax.step(x, np.arange(1, len(x) + 1) / t_bg, where="post", color=c, ls=ls, label=lab)
    ax.axvline(ev_stat, color=plotstyle.COL["L1"], lw=1.5)
    ax.axhline(1 / t_bg, color="#52514e", lw=0.8, ls=":")
    lo = 7.0
    ax.text(lo + 0.15, 1.3 / t_bg, f"1 / T_bg = 1 per {t_bg:.0f} yr", fontsize=8, color="#52514e")
    ax.text(ev_stat - 0.2, 30 / t_bg, f"{name}\nρ̂c = {ev_stat:.1f}", ha="right", color="#0b0b0b", fontsize=9)
    ax.set(yscale="log", xlabel="combined re-weighted SNR ρ̂c", ylabel="cumulative rate (per year)",
           title=f"{name}: background from 0.1-s time slides of {span / 3600:.1f} h of data ({len(starts)} files)",
           xlim=(lo, max(ev_stat + 1, 14)))
    ax.legend(loc="upper center")
    fig.tight_layout(); fig.savefig(ROOT / "docs" / "figures" / cfg["fig"]); plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1])
