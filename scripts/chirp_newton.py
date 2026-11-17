"""Chirp mass from the Newtonian chirp law alone, validated and calibrated with injections
(pre-registration stage 5, section 2).

Usage: python scripts/chirp_newton.py GW150914|GW151226
Checkpoint: results/chirp_newton_<event>.npz (every 50 injections). Writes results/chirp_newton_<event>.json
and docs/figures/1x_chirp_newton_<event>.png
"""
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from gwlab import chirpfilter, const, mf, plotstyle, psd, qtransform, search, waveforms  # noqa: E402
import evcommon  # noqa: E402

FS, N = const.FS, 32 * const.FS
SEED = 20261116
N_PER = 100
CFG = {
    "GW150914": {"mc_true": [20, 25, 30, 35, 40], "q": (0.5, 1.0), "chi": (0.0, 0.0), "grid": (8.0, 80.0),
                 "lit": (30.4, 28.5, 32.5), "fig": "14_chirp_newton_GW150914.png"},
    "GW151226": {"mc_true": [8, 9, 10, 11, 12], "q": (0.25, 1.0), "chi": (-0.1, 0.5), "grid": (5.0, 20.0),
                 "lit": (9.72, 9.66, 9.79), "fig": "15_chirp_newton_GW151226.png"},
}


def segment_start(t_centre, stretches):
    """Start sample of a 32-s segment centred on t_centre (s), or None if it does not fit in gap-free data."""
    i0 = int(round((t_centre - 16.0) * FS))
    for a, b in stretches:
        if a <= i0 and i0 + N <= b and b - a >= search.MIN_PSD_SPAN * FS:
            return i0
    return None


def calibrate(true, med):
    """Inverse of the piecewise-linear median relation raw = g(true), extrapolated with the end slopes."""
    true, med = np.asarray(true, float), np.asarray(med, float)

    def inv(raw):
        raw = np.asarray(raw, float)
        out = np.interp(raw, med, true)
        lo, hi = raw < med[0], raw > med[-1]
        out = np.where(lo, true[0] + (raw - med[0]) * (true[1] - true[0]) / (med[1] - med[0]), out)
        out = np.where(hi, true[-1] + (raw - med[-1]) * (true[-1] - true[-2]) / (med[-1] - med[-2]), out)
        return out
    return inv


def main(name):
    cfg, ev = CFG[name], evcommon.EVENTS[name]
    best = evcommon.best(name)
    t_pk = {"H1": best["seg_start_s"] + best["i_h"] / FS, "L1": best["seg_start_s"] + best["i_l"] / FS}
    snr_ev = {"H1": best["rho_h"], "L1": best["rho_l"]}
    bank = chirpfilter.newton_bank(N, FS, chirpfilter.mc_grid(*cfg["grid"]))
    f = np.fft.rfftfreq(N, 1.0 / FS)
    sl = mf.band_slice(N, FS)
    ck = ROOT / "results" / f"chirp_newton_{name}.npz"
    cols = ["det", "mc_true", "q", "chi", "t_centre", "mc_raw", "snr_newton", "edge", "snr_opt"]
    n_tot = 2 * len(cfg["mc_true"]) * N_PER
    inj = np.full((n_tot, len(cols)), np.nan)
    done = np.zeros(n_tot, dtype=bool)
    if ck.exists():
        z = np.load(ck)
        inj, done = z["inj"], z["done"]
    event_res, qmaps = {}, {}
    t0 = time.time()
    for d_i, det in enumerate(("H1", "L1")):
        strain = evcommon.load_strain(name, det)
        stretches = psd.valid_stretches(strain)
        # the event itself
        i0 = segment_start(t_pk[det], stretches)
        if i0 is None:  # event too close to the start of the data: slide the segment inside the stretch
            a, b = [s for s in stretches if s[0] <= t_pk[det] * FS < s[1]][0]
            i0 = max(a, min(int(round((t_pk[det] - 16.0) * FS)), b - N))
        prep = search.prepare_segment(strain, i0, N, stretches=stretches)
        r = chirpfilter.measure(prep["s_f"], prep["inv"], bank, t_pk[det] - i0 / FS)
        event_res[det] = {k: v for k, v in r.items() if k != "curve"}
        event_res[det]["snr_full_template"] = snr_ev[det]
        event_res[det]["curve"] = r["curve"].tolist()
        from scipy.signal.windows import tukey
        xw = psd.whiten(prep["seg"] * tukey(N, 2.0 * const.TUKEY_TAPER * FS / N), prep["inv"], FS)
        c = int(round((t_pk[det] - i0 / FS) * FS))
        w0 = c - 2 * FS
        freqs = np.geomspace(25, 500, 140)
        e = qtransform.qtransform(xw[w0:w0 + 4 * FS], FS, freqs, q=6.0)
        a_, b_ = 2 * FS - int((1.2 if name == "GW151226" else 0.3) * FS), 2 * FS + int(0.08 * FS)
        qmaps[det] = {"t": (np.arange(a_, b_) - 2 * FS) / FS, "freqs": freqs,
                      "e": e[:, a_:b_], "tc_rel": r["t_peak"] - (t_pk[det] - i0 / FS), "mc": r["mc"]}
        print(det, "event", event_res[det]["mc"], event_res[det]["snr"], flush=True)
        # injections
        rng = np.random.default_rng(SEED + d_i)
        t_file = len(strain) / FS
        k = d_i * len(cfg["mc_true"]) * N_PER
        for mc in cfg["mc_true"]:
            for _ in range(N_PER):
                while True:
                    tc = rng.uniform(16.0, t_file - 16.0)
                    if abs(tc - t_pk[det]) > 48.0 and segment_start(tc, stretches) is not None:
                        break
                q = rng.uniform(*cfg["q"])
                chi = rng.uniform(*cfg["chi"]) if cfg["chi"][1] > cfg["chi"][0] else 0.0
                t_in = 16.0 + rng.uniform(-1.0, 1.0)
                phi = rng.uniform(0, 2 * np.pi)
                if not done[k]:
                    j0 = segment_start(tc, stretches)
                    m1, m2 = waveforms.masses_from_mc_q(mc, q)
                    h = waveforms.imrphenomd(m1, m2, N, FS, chi1=chi, chi2=chi) * np.exp(1j * phi)
                    p0 = search.prepare_segment(strain, j0, N, stretches=stretches)
                    sig = np.sqrt(mf.sigmasq(h, p0["inv"], FS / N, sl))
                    h = h * snr_ev[det] / sig * np.exp(-2j * np.pi * f * t_in)
                    s_f = p0["s_f"] + h  # the Tukey taper is flat at the injection time
                    rr = chirpfilter.measure(s_f, p0["inv"], bank, t_in)
                    inj[k] = [d_i, mc, q, chi, tc, rr["mc"], rr["snr"], rr["edge"], snr_ev[det]]
                    done[k] = True
                    if done.sum() % 50 == 0:
                        np.savez(ck, inj=inj, done=done, cols=np.array(cols))
                        print(det, int(done.sum()), n_tot, f"{time.time() - t0:.0f}s", flush=True)
                k += 1
        del strain
    np.savez(ck, inj=inj, done=done, cols=np.array(cols))

    # calibration and verdicts
    res = {"event": name, "band_hz": [chirpfilter.F_LO, chirpfilter.F_HI], "n_per_point": N_PER, "detectors": {}}
    cal = {}
    for d_i, det in enumerate(("H1", "L1")):
        rows = inj[inj[:, 0] == d_i]
        tab, med = [], []
        for mc in cfg["mc_true"]:
            r = rows[rows[:, 1] == mc]
            okr = r[r[:, 7] == 0][:, 5]
            med.append(float(np.median(okr)))
            tab.append({"mc_true": mc, "n": int(len(r)), "n_edge": int((r[:, 7] == 1).sum()), "raw_median": med[-1],
                        "raw_p16": float(np.percentile(okr, 16)), "raw_p84": float(np.percentile(okr, 84)),
                        "ratio_median": med[-1] / mc, "snr_newton_median": float(np.median(r[:, 6])),
                        "snr_newton_over_optimal": float(np.median(r[:, 6]) / snr_ev[det])})
        mono = bool(np.all(np.diff(med) > 0))
        inv = calibrate(cfg["mc_true"], med) if mono else None
        if mono:
            for row, mc in zip(tab, cfg["mc_true"]):
                r = rows[(rows[:, 1] == mc) & (rows[:, 7] == 0)][:, 5]
                c = inv(r)
                row["cal_median"] = float(np.median(c))
                row["cal_scatter_rel"] = float(0.5 * (np.percentile(c, 84) - np.percentile(c, 16)) / mc)
        cal[det] = inv
        er = dict(event_res[det]); er.pop("curve")
        er["mc_raw"] = er.pop("mc")
        er["snr_newton_over_full"] = er["snr"] / snr_ev[det]
        if mono and not er["edge"]:
            er["mc_cal"] = float(inv(er["mc_raw"]))
            near = int(np.argmin(np.abs(np.array(cfg["mc_true"]) - er["mc_cal"])))
            er["scatter_rel"] = tab[near]["cal_scatter_rel"]
            er["scatter_from_mc_true"] = cfg["mc_true"][near]
        res["detectors"][det] = {"calibration": tab, "monotonic": mono, "event": er}
    n_edge = int((inj[:, 7] == 1).sum())
    h1 = res["detectors"]["H1"]
    res["validation"] = {
        "V1_monotonic_both": bool(h1["monotonic"] and res["detectors"]["L1"]["monotonic"]),
        "V2_h1_scatter_mid": h1["calibration"][2].get("cal_scatter_rel"),
        "V2_ok": bool(h1["calibration"][2].get("cal_scatter_rel", 1.0) <= 0.15),
        "V3_edge_fraction": n_edge / len(inj), "V3_ok": bool(n_edge / len(inj) <= 0.05)}
    res["validation"]["all_ok"] = bool(res["validation"]["V1_monotonic_both"] and res["validation"]["V2_ok"] and res["validation"]["V3_ok"])
    evs = [res["detectors"][d]["event"] for d in ("H1", "L1")]
    if all("mc_cal" in e for e in evs):
        w = np.array([1.0 / (e["scatter_rel"] * e["mc_cal"]) ** 2 for e in evs])
        mcc = float(np.sum(w * np.array([e["mc_cal"] for e in evs])) / w.sum())
        sig = float(1.0 / np.sqrt(w.sum()))
        res["combined"] = {"mc_cal": mcc, "sigma": sig, "sigma_rel": sig / mcc,
                           "interval90": [mcc - 1.645 * sig, mcc + 1.645 * sig]}
    lit = cfg["lit"]
    res["literature_mc_det"] = {"median": lit[0], "lo": lit[1], "hi": lit[2]}
    if name == "GW150914":
        raw = h1["event"]["mc_raw"]
        res["E1"] = {"value": raw, "range": [22.8, 38.0], "reproduced": bool(22.8 <= raw <= 38.0 and not h1["event"]["edge"])}
        if "combined" in res:
            lo, hi = res["combined"]["interval90"]
            res["E2"] = {"interval90": [lo, hi], "literature": [lit[1], lit[2]], "reproduced": bool(lo <= lit[2] and hi >= lit[1])}
    res["event_curves"] = {d: {"grid_lo_hi_step": [cfg["grid"][0], cfg["grid"][1], 0.0025], "snr": event_res[d]["curve"]} for d in event_res}
    json.dump(res, open(ROOT / "results" / f"chirp_newton_{name}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "event_curves"}, indent=1))

    plt = plotstyle.apply()
    COL = plotstyle.COL
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.2))
    ax = axs[0]
    for d_i, det in enumerate(("H1", "L1")):
        tab = res["detectors"][det]["calibration"]
        x = np.array([r["mc_true"] for r in tab]) * (1 + 0.006 * (2 * d_i - 1))
        y = np.array([r["raw_median"] for r in tab])
        ax.errorbar(x, y, yerr=[y - [r["raw_p16"] for r in tab], [r["raw_p84"] for r in tab] - y], color=COL[det],
                    marker="o", ms=4, capsize=3, lw=1, label=f"{det} injections (median, 16-84 %)")
        ax.axhline(res["detectors"][det]["event"]["mc_raw"], color=COL[det], lw=0.9, ls="--")
    lim = [cfg["mc_true"][0] * 0.85, cfg["mc_true"][-1] * 1.25]
    ax.plot(lim, lim, color="#52514e", lw=0.8, ls=":", label="no bias")
    ax.set(xlabel="true chirp mass of the injection (M☉)", ylabel="Newtonian-chirp estimate ℳ_raw (M☉)",
           title="bias and scatter (dashed: the event)", xlim=lim)
    ax.legend(fontsize=8, loc="upper left")
    for ax, det in zip(axs[1:], ("H1", "L1")):
        qm = qmaps[det]
        pc = ax.pcolormesh(qm["t"], qm["freqs"], qm["e"], shading="auto", cmap="viridis", vmin=0,
                           vmax=max(25.0, float(np.percentile(qm["e"], 99.9))), rasterized=True)
        tau = qm["tc_rel"] - qm["t"]
        ok = tau > 0
        fn = chirpfilter.newton_f(tau[ok], qm["mc"])
        m = (fn >= chirpfilter.F_LO) & (fn <= chirpfilter.F_HI)
        ax.plot(qm["t"][ok][m], fn[m], color="white", lw=1.2, ls="--")
        ax.set(yscale="log", xlabel="time from the matched-filter peak (s)", ylabel="frequency (Hz)",
               title=f"{det}: Newtonian track, ℳ_raw = {qm['mc']:.1f} M☉")
        plotstyle.plain_log_ticks(ax, "y", [30, 50, 100, 200, 400])
        ax.grid(False)
        fig.colorbar(pc, ax=ax, label="normalised energy")
    fig.suptitle(f"{name}: chirp mass from the leading-order chirp law (30-150 Hz), no post-Newtonian terms", fontsize=10)
    fig.tight_layout(); fig.savefig(ROOT / "docs" / "figures" / cfg["fig"]); plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1])
