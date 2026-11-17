"""Compare the CNN detector with the matched filter on the same injections at the same false-alarm rate
(pre-registration stage 5, section 6). Writes results/cnn_eval.json, results/cnn_injections.csv, figure 17.

Checkpoint: data/cache/cnn_scores_<det>.npz (event-file scores) and data/cache/cnn_inj.npz.
"""
import csv
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from gwlab import cnn, coinc, const, detectors, io, plotstyle, search, waveforms  # noqa: E402
from run_search import segment_starts, strain_path  # noqa: E402

FS, N = const.FS, const.SEG_DUR * const.FS
YEAR = 365.25 * 86400
# "prereg": the pre-registered coincidence window. "wide": change-log entry 2 (reference only).
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "prereg"
COINC, SLIDE = {"prereg": (0.08, 0.2), "wide": (0.35, 0.75)}[VARIANT]
SUFFIX = "" if VARIANT == "prereg" else "_" + VARIANT
SPAN = 4080.0
N_TOP = 1000           # single-detector triggers kept per detector (loudest local maxima)
FAR_TARGET = 1.0       # per year (the "wide" variant uses 1 / T_bg of the CNN background instead)
T_EVENT = const.T0_GW150914 - const.FILE_START
M1, M2 = 38.9, 31.6
CACHE = ROOT / "data" / "cache"


def wilson(k, n, z=1.0):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def file_scores(model, det):
    fn = CACHE / f"cnn_scores_{det}.npz"
    if fn.exists():
        z = np.load(fn)
        return z["t"], z["s"]
    strain = io.read_strain(strain_path(det))["strain"]
    ts, ss = [], []
    for i0 in segment_starts(len(strain)):
        prep = search.prepare_segment(strain, i0, N)
        x = cnn.preprocess(prep["seg"], prep["inv"])
        t, s = cnn.score_series(model, x, const.VALID_START, const.VALID_END)
        ts.append(t + i0 / FS); ss.append(s)
    t, s = np.concatenate(ts), np.concatenate(ss)
    np.savez(fn, t=t, s=s)
    return t, s


def threshold_for_far(stats_desc, t_bg_yr, far):
    """Statistic x* such that at most floor(far * t_bg) background events are strictly louder."""
    k = int(np.floor(far * t_bg_yr))
    return float(stats_desc[k]), k


def main():
    import torch
    torch.set_num_threads(4)
    model = cnn.build_model()
    model.load_state_dict(torch.load(ROOT / "results" / "cnn_model.pt"))
    model.eval()
    CACHE.mkdir(parents=True, exist_ok=True)
    res = {}
    # ---- background on the event file
    trig = {}
    for det in ("H1", "L1"):
        t, s = file_scores(model, det)
        i = cnn.local_maxima(t, s)
        i = i[np.argsort(s[i])[::-1][:N_TOP]]
        trig[det] = (t[i], s[i])
        res[f"n_windows_{det}"] = int(len(t))
    th, sh = trig["H1"]
    tl, sl_ = trig["L1"]
    zi, zj = coinc.zero_lag(th, tl, COINC)
    zstat = sh[zi] + sl_[zj]
    o = np.argsort(zstat)[::-1]
    res["zero_lag_top5"] = [{"t_file_s": float(th[zi[k]]), "stat": float(zstat[k]), "logit_h": float(sh[zi[k]]),
                             "logit_l": float(sl_[zj[k]])} for k in o[:5]]
    res["event_is_loudest_zero_lag"] = bool(abs(th[zi[o[0]]] - T_EVENT) <= 0.5)
    mh, ml = np.abs(th - T_EVENT) > 1.0, np.abs(tl - T_EVENT) > 1.0
    i, j, k = coinc.slide_coincidences(th[mh] - const.VALID_START, tl[ml] - const.VALID_START, SPAN, SLIDE, COINC)
    bg = np.sort(sh[mh][i] + sl_[ml][j])[::-1]
    t_bg = coinc.n_slides(SPAN, SLIDE) * SPAN / YEAR
    far = FAR_TARGET if VARIANT == "prereg" else 1.0 / t_bg
    x_cnn, k_cnn = threshold_for_far(bg, t_bg, far)
    ev_stat = [z["stat"] for z in res["zero_lag_top5"] if abs(z["t_file_s"] - T_EVENT) <= 0.5]
    res["cnn_background"] = {"t_bg_yr": t_bg, "n_coinc": int(len(bg)), "top10": bg[:10].tolist(), "threshold": x_cnn,
                             "n_louder_allowed": k_cnn, "event_stat": ev_stat[0] if ev_stat else None,
                             "n_background_louder_than_event": int(np.sum(bg >= ev_stat[0])) if ev_stat else None,
                             "lowest_kept_trigger_logit": {"H1": float(sh.min()), "L1": float(sl_.min())}}
    mfb = json.load(open(ROOT / "results" / "background.json"))
    x_mf, k_mf = threshold_for_far(np.array(mfb["background_stats_main"]), mfb["t_bg_yr"], far)
    res["mf_background"] = {"t_bg_yr": mfb["t_bg_yr"], "threshold": x_mf, "n_louder_allowed": k_mf}
    print(json.dumps(res, indent=1), flush=True)

    # ---- the stage-3 injections again, scored by the CNN
    z = np.load(ROOT / "results" / "injections.npz")
    cols = [str(c) for c in z["cols"]]
    inj = z["res"]
    c = {name: inj[:, cols.index(name)] for name in cols}
    fn = CACHE / f"cnn_inj{SUFFIX}.npz"
    out = np.full((len(inj), 3), np.nan)   # found, stat, dt
    done = np.zeros(len(inj), dtype=bool)
    if fn.exists():
        zz = np.load(fn)
        out, done = zz["out"], zz["done"]
    if not done.all():
        strain = {d: io.read_strain(strain_path(d))["strain"] for d in ("H1", "L1")}
        starts = segment_starts(len(strain["H1"]))
        f = np.fft.rfftfreq(N, 1.0 / FS)
        t0 = time.time()
        for q in range(len(inj)):
            if done[q]:
                continue
            i0 = starts[int(c["seg"][q])]
            hp, hc = waveforms.lal_fd(M1, M2, N, FS, dist_mpc=c["dist"][q], incl=np.arccos(c["cosi"][q]), phi=c["phi"][q])
            tr = {}
            for det, tarr in (("H1", c["t_h"][q]), ("L1", c["t_l"][q])):
                fp, fc = detectors.antenna(det, c["lon"][q], c["lat"][q], c["psi"][q])
                hdet = (fp * hp + fc * hc) * np.exp(-2j * np.pi * f * (tarr - i0 / FS))
                extra = np.fft.irfft(hdet, N) * FS
                prep = search.prepare_segment(strain[det], i0, N, extra=extra)
                x = cnn.preprocess(prep["seg"], prep["inv"])
                t, s = cnn.score_series(model, x, const.VALID_START, const.VALID_END)
                m = cnn.local_maxima(t, s)
                tr[det] = (t[m] + i0 / FS, s[m])
            a, b = coinc.zero_lag(tr["H1"][0], tr["L1"][0], COINC)
            near = np.abs(tr["H1"][0][a] - c["t_h"][q]) <= (0.25 if VARIANT == "prereg" else 0.5)
            a, b = a[near], b[near]
            if len(a):
                st = tr["H1"][1][a] + tr["L1"][1][b]
                o2 = int(np.argmax(st))
                out[q] = [1, st[o2], tr["H1"][0][a[o2]] - c["t_h"][q]]
            else:
                out[q] = [0, np.nan, np.nan]
            done[q] = True
            if done.sum() % 50 == 0:
                np.savez(fn, out=out, done=done)
                print("inj", int(done.sum()), len(inj), f"{time.time() - t0:.0f}s", flush=True)
        np.savez(fn, out=out, done=done)

    det_cnn = (out[:, 0] == 1) & (out[:, 1] > x_cnn)
    det_mf = (c["found_coinc"] == 1) & (c["newsnr_c"] > x_mf)
    rows, dists = [], sorted(set(c["dist"]))
    for d in dists:
        m = c["dist"] == d
        n = int(m.sum())
        r = {"dist_mpc": float(d), "n": n}
        for lab, dd in (("cnn", det_cnn), ("mf", det_mf)):
            kq = int(dd[m].sum())
            lo, hi = wilson(kq, n)
            r[lab] = {"k": kq, "eff": kq / n, "lo": float(lo), "hi": float(hi)}
        r["both"] = int((det_cnn & det_mf)[m].sum()); r["cnn_only"] = int((det_cnn & ~det_mf)[m].sum()); r["mf_only"] = int((det_mf & ~det_cnn)[m].sum())
        rows.append(r)
    res["efficiency"] = rows

    def summary(lab):
        e = np.array([r[lab]["eff"] for r in rows]); dd = np.array(dists)
        d50 = None
        for a in range(len(dd) - 1):
            if e[a] >= 0.5 > e[a + 1]:
                d50 = float(dd[a] + (e[a] - 0.5) / (e[a] - e[a + 1]) * (dd[a + 1] - dd[a]))
        dg = np.concatenate([[0.0], dd]); eg = np.concatenate([[e[0]], e])
        vol = 4 * np.pi * np.trapezoid(eg * dg**2, dg)
        return {"d50_mpc": d50, "range_mpc": float((3 * vol / (4 * np.pi)) ** (1 / 3)), "n_detected": int(sum(r[lab]["k"] for r in rows))}
    res["summary"] = {"cnn": summary("cnn"), "mf": summary("mf"), "n_injections": int(len(inj)), "far_per_yr": far,
                      "variant": VARIANT, "coinc_window_s": COINC, "slide_step_s": SLIDE}
    res["cnn_better"] = bool(res["summary"]["cnn"]["range_mpc"] > res["summary"]["mf"]["range_mpc"])
    json.dump(res, open(ROOT / "results" / f"cnn_eval{SUFFIX}.json", "w"), indent=1)
    with open(ROOT / "results" / f"cnn_injections{SUFFIX}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dist_mpc", "cnn_found_coinc", "cnn_stat", "cnn_dt_s", "cnn_detected", "mf_newsnr_c", "mf_detected"])
        for q in range(len(inj)):
            w.writerow([c["dist"][q], int(out[q, 0]), f"{out[q, 1]:.4f}", f"{out[q, 2]:.4f}", int(det_cnn[q]),
                        f"{c['newsnr_c'][q]:.4f}", int(det_mf[q])])
    print(json.dumps({k2: res[k2] for k2 in ("efficiency", "summary", "cnn_better")}, indent=1))

    plt = plotstyle.apply()
    COL = plotstyle.COL
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    dd = np.array(dists)
    for lab, name, col, off in (("mf", "matched filter (496 templates)", COL["H1"], -12), ("cnn", "small CNN (58k parameters)", COL["L1"], 12)):
        e = np.array([r[lab]["eff"] for r in rows])
        lo = np.array([r[lab]["lo"] for r in rows]); hi = np.array([r[lab]["hi"] for r in rows])
        ax.errorbar(dd + off, e, yerr=[e - lo, hi - e], color=col, marker="o", ms=4, capsize=3, lw=1.2, label=name)
    ax.set(xlabel="luminosity distance (Mpc)", ylabel="detection efficiency", ylim=(-0.03, 1.03),
           title=f"Same 900 injections, both thresholds at a false-alarm rate of {far:.2g} per year ({VARIANT})")
    ax.legend()
    fig.tight_layout(); fig.savefig(ROOT / "docs" / "figures" / f"17_cnn_vs_mf{SUFFIX}.png"); plt.close(fig)


if __name__ == "__main__":
    main()
