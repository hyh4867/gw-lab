"""GW170817: zero-lag candidates of the BNS search, glitch handling, event measurements, template comparison
(pre-registration stage 6, sections 3-5). Writes results/event_gw170817.json and figures 18, 19.
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from gwlab import bns, coinc, const, mf, plotstyle, psd, waveforms  # noqa: E402
import ev6common as E  # noqa: E402

FS, N = E.FS, E.N
SEED = 20261117
N_FINE, N_TOP, N_IMR = 4000, 20, 200


def zero_lag(variant):
    d = ROOT / "results" / "triggers" / f"bns_{variant}" / str(E.A_START)
    th, tl = (np.load(d / f"{det}.npy") for det in ("H1", "L1"))
    th, tl = th[th["newsnr"] >= const.NEWSNR_MIN], tl[tl["newsnr"] >= const.NEWSNR_MIN]
    i, j = coinc.zero_lag(th["t"], tl["t"])
    stat = coinc.combined(th["newsnr"][i], tl["newsnr"][j])
    o = np.argsort(stat)[::-1]
    top = [{"gps_h": E.A_START + float(th["t"][i[k]]), "gps_l": E.A_START + float(tl["t"][j[k]]),
            "newsnr_c": float(stat[k]), "rho_h": float(th["rho"][i[k]]), "rho_l": float(tl["rho"][j[k]]),
            "chi2r_h": float(th["chi2r"][i[k]]), "chi2r_l": float(tl["chi2r"][j[k]]),
            "tmpl_h": int(th["tmpl"][i[k]]), "tmpl_l": int(tl["tmpl"][j[k]])} for k in o[:5]]
    info = json.load(open(d / "done.json"))
    near = {}
    for det, t in (("H1", th), ("L1", tl)):
        m = np.abs(E.A_START + t["t"] - E.T0) <= 3.0
        near[det] = [{"gps": E.A_START + float(r["t"]), "rho": float(r["rho"]), "chi2r": float(r["chi2r"]),
                      "newsnr": float(r["newsnr"])} for r in t[m]]
    return {"n_h": int(len(th)), "n_l": int(len(tl)), "n_zero_lag": int(len(stat)), "top5": top,
            "single_detector_triggers_within_3s_of_catalogue_time": near,
            "loudest_single": {det: {"gps": E.A_START + float(t["t"][np.argmax(t["newsnr"])]),
                                     "newsnr": float(t["newsnr"].max()), "rho": float(t["rho"][np.argmax(t["newsnr"])])}
                               for det, t in (("H1", th), ("L1", tl)) if len(t)},
            "gates": {det: info[det]["gates"] for det in ("H1", "L1")},
            "seconds_filter": info["seconds_filter"], "n_templates": info["n_templates"]}


def chisq_newsnr(pr, h, sl, i_peak, rho):
    c2, dof = mf.chisq(pr["s_f"], h, pr["inv"], N, i_peak, FS, sl=sl)
    return {"chi2r": c2 / dof, "newsnr": float(mf.newsnr(np.array([rho]), np.array([c2 / dof]))[0])}


def fine_bank(basis, mcb, preps, c, rng):
    """Pre-registered fine bank: 4000 random points, fast pass, top 20 at full rate. Returns best dict + extras."""
    mc = rng.uniform(mcb - 0.002, mcb + 0.002, N_FINE)
    eta = rng.uniform(2.0 / 9, 0.25, N_FINE)
    mm = np.array([waveforms.masses_from_mc_eta(a, e) for a, e in zip(mc, eta)])
    red = {d: bns.reduce_prep(preps[d], basis) for d in ("H1", "L1")}
    fast = E.scan_fast(basis, mm, red["H1"], red["L1"], c)
    top = np.argsort(fast["net"])[::-1][:N_TOP]
    full = E.scan_full([basis.full(*mm[k]) for k in top], mm[top], basis.sl, preps["H1"], preps["L1"], c)
    kb = int(np.argmax(full["net"]))
    k = int(top[kb])
    ok = fast["net"] ** 2 >= fast["net"].max() ** 2 - 4.61
    best = {"m1": float(mm[k, 0]), "m2": float(mm[k, 1]), "mc": float(mc[k]), "eta": float(eta[k]),
            "q": float(mm[k, 1] / mm[k, 0]), "mtot": float(mm[k].sum()),
            "rho_h": float(full["rho_h"][kb]), "rho_l": float(full["rho_l"][kb]), "rho_net_hl": float(full["net"][kb]),
            "i_h": float(full["i_h"][kb]), "i_l": float(full["i_l"][kb]),
            "dt_h_minus_l_ms": float((full["i_h"][kb] - full["i_l"][kb]) / FS * 1e3),
            "fast_pass_net_of_best": float(fast["net"][k]), "fast_pass_max_net": float(fast["net"].max()),
            "region": {"n": int(ok.sum()), "mc": [float(mc[ok].min()), float(mc[ok].max())],
                       "q": [float((mm[ok, 1] / mm[ok, 0]).min()), float((mm[ok, 1] / mm[ok, 0]).max())],
                       "m1": [float(mm[ok, 0].min()), float(mm[ok, 0].max())],
                       "m2": [float(mm[ok, 1].min()), float(mm[ok, 1].max())]}}
    return best, mm, fast


def virgo(pr_v, h, sl, i_l, i0, dataset):
    """|z_V| at the arrival time expected from the counterpart position, and the maximum within +-35 ms."""
    dl = E.geometric_delays()
    i_exp = i_l + (dl["V1"] - dl["L1"]) * FS
    rho_exp = float(abs(E.z_at(pr_v, h, sl, i_exp)))
    rho, _, _ = mf.snr_series(pr_v["s_f"], h, pr_v["inv"], N, FS, sl)
    a, b = int(i_exp - 0.035 * FS), int(i_exp + 0.035 * FS) + 1
    j = a + int(np.argmax(rho[a:b]))
    far = np.concatenate([rho[int(i_exp) - 20 * FS:int(i_exp) - FS], rho[int(i_exp) + FS:int(i_exp) + 15 * FS]])
    # distribution of the +-35 ms window maximum in the surrounding noise
    w = b - a
    wm = far[:len(far) // w * w].reshape(-1, w).max(axis=1)
    return {"rho_at_expected_time": rho_exp, "expected_t_v_minus_t_l_ms": (dl["V1"] - dl["L1"]) * 1e3,
            "window_max": float(rho[j]), "window_max_t_v_minus_t_l_ms": float((j - i_l) / FS * 1e3),
            "noise_window_max_median": float(np.median(wm)), "noise_window_max_p95": float(np.percentile(wm, 95)),
            "noise_fraction_of_windows_louder": float(np.mean(wm >= rho[j]))}


def main():
    res = {"search_gated": zero_lag("gated")}
    try:
        res["search_raw"] = zero_lag("raw")
    except FileNotFoundError:
        res["search_raw"] = None
    top = res["search_gated"]["top5"][0]
    t_ev = top["gps_h"]
    res["found_time_gps"] = t_ev
    res["offset_from_catalogue_s"] = t_ev - E.T0
    masses = np.loadtxt(ROOT / "results" / "bank_bns.csv", delimiter=",", skiprows=1)
    mcb = float(masses[top["tmpl_h"], 0]) if top["rho_h"] >= top["rho_l"] else float(masses[top["tmpl_l"], 0])
    res["search_best_mc"] = mcb
    print("zero lag", top, flush=True)

    basis = bns.basis_for()
    sl = basis.sl
    var = {}
    preps_all = {}
    # ---- (b) data A gated, (c) data B: fine bank each; (a) data A raw: template of (b)
    for key, dataset, gated in (("b_A_gated", "A", True), ("c_B", "B", False), ("a_A_raw", "A", False)):
        i0 = E.segment_start(dataset, t_ev)
        c = int(round((t_ev - E.start(dataset)) * FS)) - i0
        preps, gates = {}, {}
        for det in E.DETS:
            x, g = E.load(dataset, det, gated)
            gates[det] = [{"gps": E.start(dataset) + a / FS, "peak_sigma": p} for a, p in g]
            preps[det] = E.prep(x, dataset, i0)
            del x
        preps_all[key] = (preps, i0, c)
        v = {"segment_start_gps": E.start(dataset) + i0 / FS, "glitches_found": gates}
        if key != "a_A_raw":
            best, mm, fast = fine_bank(basis, mcb, preps, c, np.random.default_rng(SEED))
            v["fine"] = best
            var[key + "_mm"] = (mm, fast)
        else:
            best = dict(var["b_A_gated"]["fine"])
            h = basis.full(best["m1"], best["m2"])
            s = E.scan_full([h], [[best["m1"], best["m2"]]], sl, preps["H1"], preps["L1"], c)
            best.update({"rho_h": float(s["rho_h"][0]), "rho_l": float(s["rho_l"][0]), "rho_net_hl": float(s["net"][0]),
                         "i_h": float(s["i_h"][0]), "i_l": float(s["i_l"][0]),
                         "dt_h_minus_l_ms": float((s["i_h"][0] - s["i_l"][0]) / FS * 1e3)})
            best.pop("region", None)
            v["template_of_b"] = best
        h = basis.full(best["m1"], best["m2"])
        v["reweighted"] = {"H1": chisq_newsnr(preps["H1"], h, sl, best["i_h"], best["rho_h"]),
                           "L1": chisq_newsnr(preps["L1"], h, sl, best["i_l"], best["rho_l"])}
        v["reweighted"]["newsnr_c"] = float(np.hypot(v["reweighted"]["H1"]["newsnr"], v["reweighted"]["L1"]["newsnr"]))
        if preps["V1"] is not None:
            v["V1"] = virgo(preps["V1"], h, sl, best["i_l"], i0, dataset)
            v["rho_net_hlv"] = float(np.sqrt(best["rho_net_hl"] ** 2 + v["V1"]["rho_at_expected_time"] ** 2))
        v["gps_h"] = E.start(dataset) + (i0 + best["i_h"]) / FS
        v["gps_l"] = E.start(dataset) + (i0 + best["i_l"]) / FS
        v["sigma_1mpc"] = {d: float(np.sqrt(mf.sigmasq(h, preps[d]["inv"], FS / N, sl))) for d in E.DETS if preps[d] is not None}
        v["horizon_mpc_snr8"] = {d: s / 8.0 for d, s in v["sigma_1mpc"].items()}
        var[key] = v
        print(key, {k: best[k] for k in ("m1", "m2", "mc", "rho_h", "rho_l", "rho_net_hl", "dt_h_minus_l_ms")},
              v.get("V1", {}).get("rho_at_expected_time"), flush=True)

    # ---- (a): what the glitch does to the L1 output of the best template
    preps, i0, c = preps_all["a_A_raw"]
    best = var["b_A_gated"]["fine"]
    h = basis.full(best["m1"], best["m2"])
    rho_raw, _, _ = mf.snr_series(preps["L1"]["s_f"], h, preps["L1"]["inv"], N, FS, sl)
    lg = [g for g in var["a_A_raw"]["glitches_found"]["L1"] if abs(g["gps"] - t_ev) < 5]
    if lg:
        ig = int(round((lg[0]["gps"] - E.A_START) * FS)) - i0
        a, b = ig - 2 * FS, ig + 2 * FS
        j = a + int(np.argmax(rho_raw[a:b]))
        var["a_A_raw"]["L1_near_glitch"] = {"glitch_gps": lg[0]["gps"], "peak_sigma": lg[0]["peak_sigma"],
                                            "max_rho_within_2s": float(rho_raw[j]),
                                            "t_of_max_minus_glitch_s": float((j - ig) / FS),
                                            **chisq_newsnr(preps["L1"], h, sl, j, float(rho_raw[j]))}
    res["variants"] = {k: v for k, v in var.items() if not k.endswith("_mm")}

    # ---- verdicts J, K
    b, cc = var["b_A_gated"], var["c_B"]
    gl = [g["gps"] for g in var["b_A_gated"]["glitches_found"]["L1"]]
    j1 = min((abs(g - E.GLITCH_DOC) for g in gl), default=None)
    ratio = b["fine"]["rho_l"] / cc["fine"]["rho_l"]
    mcs = b["fine"]["mc"] / (1 + E.Z)
    dur24, _ = E.f2_time_and_cycles(b["fine"]["m1"], b["fine"]["m2"], 24.0)
    dur30, cyc30 = E.f2_time_and_cycles(b["fine"]["m1"], b["fine"]["m2"], 30.0)
    res["template_numbers"] = {"f_isco_hz": float(waveforms.f_isco(b["fine"]["mtot"])), "cycles_30hz_to_isco": cyc30,
                               "seconds_30hz_to_isco": dur30, "seconds_24hz_to_isco": dur24, "mc_source": mcs}

    def vd(v, lo, hi):
        return {"value": None if v is None else float(v), "range": [lo, hi], "reproduced": bool(v is not None and lo <= v <= hi)}
    res["verdicts"] = {
        "J1": vd(j1, 0.0, 0.05), "J2": vd(ratio, 0.84, 0.92),
        "K1": vd(res["offset_from_catalogue_s"], -0.1, 0.1),
        "K2": vd(cc["fine"]["rho_h"], 16.9, 20.7), "K3": vd(cc["fine"]["rho_l"], 23.8, 29.0),
        "K4": vd(cc["V1"]["rho_at_expected_time"], 0.8, 3.8), "K5": vd(cc["rho_net_hlv"], 29.2, 35.6),
        "K6": vd(b["fine"]["rho_net_hl"], 27.8, 34.0), "K7": vd(cc["fine"]["dt_h_minus_l_ms"], 2.8, 3.8),
        "K8": vd(b["fine"]["mc"], 1.1974, 1.1985), "K9": vd(mcs, 1.185, 1.187),
        "K10": vd(cyc30, 2400, 3600), "K11": vd(dur24, 80, 120)}
    json.dump(res, open(ROOT / "results" / "event_gw170817.json", "w"), indent=1)
    print(json.dumps(res["verdicts"], indent=1), flush=True)

    # ---- L: TaylorF2 against IMRPhenomD on data B (top 200 TaylorF2 points of the fine bank)
    preps, i0, c = preps_all["c_B"]
    mm, fast = var["c_B_mm"]
    top = np.argsort(fast["net"])[::-1][:N_IMR]
    imr = {k: [] for k in ("net", "rho_h", "rho_l", "i_h", "i_l")}
    for s0 in range(0, N_IMR, 20):
        idx = top[s0:s0 + 20]
        tm = []
        for k in idx:
            t = waveforms.imrphenomd(mm[k, 0], mm[k, 1], N, FS, f_low=bns.F_LOW)
            t[:sl.start] = 0
            t[sl.stop:] = 0
            tm.append(t)
        s = E.scan_full(tm, mm[idx], sl, preps["H1"], preps["L1"], c)
        for key in imr:
            imr[key].append(s[key])
        print("imr", s0 + 20, float(np.max(np.concatenate(imr["net"]))), flush=True)
    imr = {k: np.concatenate(v) for k, v in imr.items()}
    ki = int(np.argmax(imr["net"]))
    f2 = cc["fine"]
    res["taylorf2_vs_imrphenomd"] = {
        "data": "B", "n_points": N_IMR,
        "taylorf2": {"rho_h": f2["rho_h"], "rho_l": f2["rho_l"], "rho_net_hl": f2["rho_net_hl"], "m1": f2["m1"], "m2": f2["m2"]},
        "imrphenomd": {"rho_h": float(imr["rho_h"][ki]), "rho_l": float(imr["rho_l"][ki]), "rho_net_hl": float(imr["net"][ki]),
                       "m1": float(mm[top[ki], 0]), "m2": float(mm[top[ki], 1])},
        "ratio_h": f2["rho_h"] / float(imr["rho_h"][ki]), "ratio_l": f2["rho_l"] / float(imr["rho_l"][ki]),
        "ratio_net": f2["rho_net_hl"] / float(imr["net"][ki])}
    r = res["taylorf2_vs_imrphenomd"]
    res["verdicts"]["L1"] = {"value": [r["ratio_h"], r["ratio_l"]], "range": [0.97, None],
                             "reproduced": bool(r["ratio_h"] >= 0.97 and r["ratio_l"] >= 0.97)}
    json.dump(res, open(ROOT / "results" / "event_gw170817.json", "w"), indent=1)
    print(json.dumps(r, indent=1), flush=True)
    np.savez(ROOT / "data" / "cache" / "gw170817_fine.npz", mm_b=var["b_A_gated_mm"][0], net_b=var["b_A_gated_mm"][1]["net"],
             mm_c=mm, net_c=fast["net"])
    figures(res, preps_all, basis)


def figures(res, preps_all, basis):
    from scipy.signal import spectrogram
    from scipy.signal.windows import tukey
    plt = plotstyle.apply()
    COL = plotstyle.COL
    sl = basis.sl
    var = res["variants"]
    # figure 18: the glitch in L1 and what each treatment does to the SNR time series
    fig, axs = plt.subplots(2, 3, figsize=(13, 6.2))
    for col, (key, lab) in enumerate((("a_A_raw", "(a) data A, no treatment"), ("b_A_gated", "(b) data A, automatic gating"),
                                      ("c_B", "(c) data B, glitch subtracted by LIGO"))):
        preps, i0, c = preps_all[key]
        v = var[key]
        best = v.get("fine", v.get("template_of_b"))
        h = basis.full(best["m1"], best["m2"])
        p = preps["L1"]
        xw = psd.bandpass(psd.whiten(p["seg"] * tukey(N, 2.0 * const.TUKEY_TAPER * FS / N), p["inv"], FS), 30, 1000)
        t = (np.arange(N) - best["i_l"]) / FS
        m = (t > -2.5) & (t < 0.5)
        ax = axs[0, col]
        ax.plot(t[m], xw[m], color=COL["L1"], lw=0.5)
        ax.set(title=lab, yscale="symlog" if key == "a_A_raw" else "linear")
        if key == "a_A_raw":
            ax.set_yscale("symlog", linthresh=10)
        if col == 0:
            ax.set_ylabel("L1 whitened strain, 30-1000 Hz (σ)")
        rho, _, _ = mf.snr_series(p["s_f"], h, p["inv"], N, FS, sl)
        m2 = (t > -2.5) & (t < 0.5)
        ax = axs[1, col]
        ax.plot(t[m2], rho[m2], color=COL["L1"], lw=0.7)
        ax.set(xlabel="time from the L1 SNR peak (s)")
        ax.text(0.03, 0.92, f"ρ at the event = {best['rho_l']:.1f}", transform=ax.transAxes, fontsize=9, va="top")
        if col == 0:
            ax.set_ylabel("L1 SNR ρ(t), best template")
    fig.suptitle("GW170817 in LIGO-Livingston: a glitch 1.1 s before the merger, untreated, gated, and subtracted", fontsize=10)
    fig.tight_layout(); fig.savefig(ROOT / "docs" / "figures" / "18_gw170817_glitch.png"); plt.close(fig)

    # figure 19: spectrograms of the last 30 s (data B) and SNR time series
    preps, i0, c = preps_all["c_B"]
    best = var["c_B"]["fine"]
    h = basis.full(best["m1"], best["m2"])
    fig, axs = plt.subplots(2, 3, figsize=(13, 6.4))
    for col, (det, key) in enumerate((("H1", "i_h"), ("L1", "i_l"), ("V1", "i_l"))):
        p = preps[det]
        ip = best[key] if det != "V1" else best["i_l"] + var["c_B"]["V1"]["expected_t_v_minus_t_l_ms"] * 1e-3 * FS
        xw = psd.whiten(p["seg"] * tukey(N, 2.0 * const.TUKEY_TAPER * FS / N), p["inv"], FS)
        a, b = int(ip) - 30 * FS, int(ip) + 2 * FS
        f, tt, s = spectrogram(xw[a:b], fs=FS, nperseg=FS // 2, noverlap=FS // 2 - FS // 16, window="hann")
        k = (f >= 25) & (f <= 500)
        ax = axs[0, col]
        ax.pcolormesh(tt - 30, f[k], s[k] * FS / 2, shading="auto", vmin=0, vmax=6, cmap="viridis", rasterized=True)
        tau = np.linspace(0.02, 30, 400)
        ax.set(yscale="log", ylim=(25, 500), title=f"{det}: whitened spectrogram", xlabel="time from merger (s)")
        plotstyle.plain_log_ticks(ax, "y", [30, 50, 100, 200, 400])
        if col == 0:
            ax.set_ylabel("frequency (Hz)")
        rho, _, _ = mf.snr_series(p["s_f"], h, p["inv"], N, FS, sl)
        t = (np.arange(N) - ip) / FS
        m = np.abs(t) < 0.06
        ax = axs[1, col]
        ax.plot(t[m] * 1e3, rho[m], color=COL.get(det, COL["muted"]), lw=0.9)
        ax.axvline(0, color="#52514e", lw=0.6, ls=":")
        ax.set(xlabel="time from the SNR peak (ms)" if det != "V1" else "time from the expected arrival (ms)")
        if col == 0:
            ax.set_ylabel("SNR ρ(t), TaylorF2")
    fig.suptitle(f"GW170817 (data B): ρ = {best['rho_h']:.1f} (H1), {best['rho_l']:.1f} (L1), "
                 f"{var['c_B']['V1']['rho_at_expected_time']:.1f} (Virgo, at the expected time)", fontsize=10)
    fig.tight_layout(); fig.savefig(ROOT / "docs" / "figures" / "19_gw170817_event.png"); plt.close(fig)


def figures_only():
    """Redraw figures 18 and 19 from the saved results (the segments are prepared again)."""
    res = json.load(open(ROOT / "results" / "event_gw170817.json"))
    preps_all = {}
    for key, dataset, gated in (("b_A_gated", "A", True), ("c_B", "B", False), ("a_A_raw", "A", False)):
        i0 = int(round((res["variants"][key]["segment_start_gps"] - E.start(dataset)) * FS))
        preps = {}
        for det in E.DETS:
            x, _ = E.load(dataset, det, gated)
            preps[det] = E.prep(x, dataset, i0)
            del x
        preps_all[key] = (preps, i0, None)
    figures(res, preps_all, bns.basis_for())


if __name__ == "__main__":
    figures_only() if "--figures" in sys.argv else main()
