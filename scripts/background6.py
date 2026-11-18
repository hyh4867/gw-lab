"""Stage-6 time-slide background for GW170817 (pre-registration stage 6, section 9; the stage-5 method).

Usage: python scripts/background6.py [gated|raw]
Writes results/background_GW170817[_raw].json and docs/figures/21_background_GW170817.png (gated only).
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from gwlab import bns, coinc, const, plotstyle  # noqa: E402
from background5 import YEAR, analyse  # noqa: E402
import ev6common as E  # noqa: E402

LAY = bns.LAYOUT
FS = const.FS


def load_file(variant, start):
    d = ROOT / "results" / "triggers" / f"bns_{variant}" / str(start)
    info = json.load(open(d / "done.json"))
    skipped = set(info["H1"]["skipped"]) | set(info["L1"]["skipped"])
    live = coinc.merge_intervals([(j * LAY.stride + LAY.valid_start, j * LAY.stride + LAY.valid_end)
                                  for j in range(info["H1"]["n_segments"]) if j not in skipped])
    trig = {}
    for det in ("H1", "L1"):
        t = np.load(d / f"{det}.npy")
        trig[det] = t[t["newsnr"] >= const.NEWSNR_MIN]
    gated_s = {det: sum(2 * (0.1 + 0.5) for g in info[det]["gates"]
                        if any(a <= g["t_file_s"] < b for a, b in live)) for det in ("H1", "L1")}
    return trig, live, sorted(skipped), info, gated_s


def main(variant):
    meta = json.load(open(ROOT / "data" / "meta" / "background_files_GW170817.json"))["GW170817"]
    starts = sorted([E.A_START] + [c["gps_start"] for c in meta["chosen"]])
    if variant == "raw":
        starts = [s for s in starts if (ROOT / "results" / "triggers" / "bns_raw" / str(s) / "done.json").exists()]
    H, L, files, pos = [], [], [], 0.0
    for fi, start in enumerate(starts):
        trig, live, skipped, info, gated_s = load_file(variant, start)
        length = 0.0
        for det, acc in (("H1", H), ("L1", L)):
            t = trig[det]
            ring, keep, length = coinc.to_ring(t["t"], live, pos)
            rec = np.zeros(int(keep.sum()), dtype=[("ring", "f8"), ("gps", "f8"), ("newsnr", "f8"), ("rho", "f8"), ("file", "i4")])
            rec["ring"], rec["gps"], rec["newsnr"], rec["rho"], rec["file"] = ring[keep], start + t["t"][keep], t["newsnr"][keep], t["rho"][keep], fi
            acc.append(rec)
        files.append({"gps_start": start, "live_s": length, "n_skipped_segments": len(skipped),
                      "n_h": int(len(H[-1])), "n_l": int(len(L[-1])), "event_file": start == E.A_START,
                      "n_gates": {d: len(info[d]["gates"]) for d in ("H1", "L1")},
                      "gates": {d: info[d]["gates"] for d in ("H1", "L1")}, "seconds_touched_by_gates": gated_s,
                      "seconds_filter": info["seconds_filter"]})
        pos += length
    th, tl, span = np.concatenate(H), np.concatenate(L), pos
    zi, zj = coinc.zero_lag(th["gps"], tl["gps"])
    zstat = coinc.combined(th["newsnr"][zi], tl["newsnr"][zj])
    order = np.argsort(zstat)[::-1]
    zero = [{"gps": float(th["gps"][zi[o]]), "newsnr_c": float(zstat[o]), "rho_h": float(th["rho"][zi[o]]),
             "rho_l": float(tl["rho"][zj[o]]), "file": int(starts[th["file"][zi[o]]])} for o in order[:8]]
    res = {"event": "GW170817", "variant": variant, "files": files, "span_s": span, "zero_lag_top": zero,
           "n_zero_lag": int(len(zstat))}
    ev = next((z for z in zero if abs(z["gps"] - E.T0) <= 0.1), None)
    res["loudest_is_event"] = bool(zero and abs(zero[0]["gps"] - E.T0) <= 0.1)
    res["loudest_zero_lag_outside_event"] = next((z for z in zero if abs(z["gps"] - E.T0) > 1.0), None)
    if ev is None:
        res["note"] = "the event is not a zero-lag coincidence in this variant"
        ev_stat, t_cut = np.inf, E.T0
    else:
        ev_stat, t_cut = ev["newsnr_c"], ev["gps"]
    mh, ml = np.abs(th["gps"] - t_cut) > 1.0, np.abs(tl["gps"] - t_cut) > 1.0
    res["main"], stat_main, stat_nr = analyse(th[mh], tl[ml], span, ev_stat, "event removed (main)")
    res["event_kept"], _, _ = analyse(th, tl, span, ev_stat, "event kept")
    jk = {x: [] for x in (8.0, 8.5, 9.0)}
    for fi in range(len(starts)):
        if len(starts) < 2:
            break
        a, b = th[mh], tl[ml]
        a, b = a[a["file"] != fi].copy(), b[b["file"] != fi].copy()
        sp = span - files[fi]["live_s"]
        for arr in (a, b):
            arr["ring"] = arr["ring"] - np.where(arr["file"] > fi, files[fi]["live_s"], 0.0)
        i, j, _ = coinc.slide_coincidences(a["ring"], b["ring"], sp)
        s = coinc.combined(a["newsnr"][i], b["newsnr"][j])
        tb = coinc.n_slides(sp) * sp / YEAR
        for x in jk:
            jk[x].append(float(np.sum(s >= x) / tb))
    nf = len(starts)
    res["jackknife_by_file"] = {str(x): {"rates_per_yr": v, "full_rate_per_yr": res["main"]["rates_per_yr"][str(x)]["slides"],
                                         "se_per_yr": float(np.sqrt((nf - 1) / nf * np.sum((np.array(v) - np.mean(v)) ** 2))) if v else None}
                                for x, v in jk.items()}
    res["N1_event_louder_than_all_background"] = bool(ev is not None and res["main"]["n_louder_than_event"] == 0)
    suffix = "" if variant == "gated" else "_raw"
    json.dump(res, open(ROOT / "results" / f"background_GW170817{suffix}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "files"}, indent=1)[:6000])
    if variant != "gated" or ev is None:
        return
    plt = plotstyle.apply()
    t_bg = res["main"]["t_bg_yr"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for s, lab, c, ls in ((stat_main, "time slides, event removed", plotstyle.COL["H1"], "-"),
                          (stat_nr, "each trigger used once (biased low)", plotstyle.COL["muted"], "--")):
        x = np.sort(s)[::-1]
        ax.step(x, np.arange(1, len(x) + 1) / t_bg, where="post", color=c, ls=ls, label=lab)
    ax.axvline(ev_stat, color=plotstyle.COL["L1"], lw=1.5)
    ax.axhline(1 / t_bg, color="#52514e", lw=0.8, ls=":")
    ax.text(7.15, 1.3 / t_bg, f"1 / T_bg = 1 per {t_bg:.0f} yr", fontsize=8, color="#52514e")
    ax.text(ev_stat - 0.3, 30 / t_bg, f"GW170817\nρ̂c = {ev_stat:.1f}", ha="right", color="#0b0b0b", fontsize=9)
    ax.set(yscale="log", xlabel="combined re-weighted SNR ρ̂c", ylabel="cumulative rate (per year)",
           title=f"GW170817: background from 0.1-s time slides of {span / 3600:.1f} h of data ({len(starts)} files)",
           xlim=(7.0, ev_stat + 1))
    ax.legend(loc="upper center")
    fig.tight_layout(); fig.savefig(ROOT / "docs" / "figures" / "21_background_GW170817.png"); plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "gated")
