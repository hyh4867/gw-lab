"""Stages 1-2 for the loudest zero-lag coincidence found by run_search.py.

Writes results/event.json and figures in docs/figures/.
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from gwlab import chirpfit, coinc, const, event, io, mf, plotstyle, psd, qtransform, search, waveforms  # noqa: E402
from run_search import strain_path  # noqa: E402

plt = plotstyle.apply()
COL = plotstyle.COL
FIG = ROOT / "docs" / "figures"
FS, N = const.FS, const.SEG_DUR * const.FS
Z_LIT = 0.09


def load_triggers(kind, det):
    d = ROOT / "results" / "triggers" / kind / det
    t = np.concatenate([np.load(f) for f in sorted(d.glob("seg_*.npy"))])
    t = search.cluster(t)
    return t[t["newsnr"] >= const.NEWSNR_MIN]


def loudest_zero_lag():
    th, tl = load_triggers("imr", "H1"), load_triggers("imr", "L1")
    i, j = coinc.zero_lag(th["t"], tl["t"])
    stat = coinc.combined(th["newsnr"][i], tl["newsnr"][j])
    order = np.argsort(stat)[::-1]
    rows = [{"t_h": float(th["t"][i[o]]), "t_l": float(tl["t"][j[o]]), "newsnr_c": float(stat[o]),
             "rho_h": float(th["rho"][i[o]]), "rho_l": float(tl["rho"][j[o]]),
             "chi2r_h": float(th["chi2r"][i[o]]), "chi2r_l": float(tl["chi2r"][j[o]])} for o in order]
    return rows


def region(scan, masses):
    """Templates within delta(rho_net^2) <= 4.61 of the maximum."""
    m = scan["net"] ** 2 >= scan["net"].max() ** 2 - 4.61
    mc = waveforms.chirp_mass(masses[m, 0], masses[m, 1])
    mt = masses[m].sum(axis=1)
    return {"n": int(m.sum()), "mc": [float(mc.min()), float(mc.max())], "mtot": [float(mt.min()), float(mt.max())],
            "m1": [float(masses[m, 0].min()), float(masses[m, 0].max())],
            "m2": [float(masses[m, 1].min()), float(masses[m, 1].max())]}


def summarise(scan, masses):
    k = int(np.argmax(scan["net"]))
    m1, m2 = masses[k]
    return {"m1": float(m1), "m2": float(m2), "mc": float(waveforms.chirp_mass(m1, m2)), "mtot": float(m1 + m2),
            "rho_h": float(scan["rho_h"][k]), "rho_l": float(scan["rho_l"][k]), "rho_net": float(scan["net"][k]),
            "dt_ms": float((scan["i_h"][k] - scan["i_l"][k]) / FS * 1e3),
            "i_h": float(scan["i_h"][k]), "i_l": float(scan["i_l"][k]), "region": region(scan, masses)}


def snr_near(prep, h_f, i_ref, half_s=0.05):
    rho, _, _ = mf.snr_series(prep["s_f"], h_f, prep["inv"], N, FS)
    a, b = int(i_ref - half_s * FS), int(i_ref + half_s * FS)
    i = a + int(np.argmax(rho[a:b]))
    return mf.peak_parabolic(rho, i)


def main():
    res = {}
    zl = loudest_zero_lag()
    res["zero_lag_top5"] = zl[:5]
    res["n_zero_lag"] = len(zl)
    top = zl[0]
    t_ev = top["t_h"]
    res["found_time_gps"] = const.FILE_START + t_ev
    res["offset_from_catalog_s"] = const.FILE_START + t_ev - const.T0_GW150914
    j = int((t_ev - const.VALID_START) // const.SEG_STRIDE)
    i0 = j * const.SEG_STRIDE * FS
    v0, v1 = const.VALID_START * FS, const.VALID_END * FS
    res["segment"] = {"index": j, "start_s": i0 / FS}
    print("loudest zero-lag at file time", t_ev, "segment", j, flush=True)

    strain = {d: io.read_strain(strain_path(d))["strain"] for d in ("H1", "L1")}
    prep = {d: search.prepare_segment(strain[d], i0, N) for d in ("H1", "L1")}
    tt = (i0 + np.arange(N)) / FS - t_ev  # seconds relative to the H1 trigger

    # ---------------- stage 2: banks ----------------
    out = {}
    scans = {}
    for kind in ("imr", "f2"):
        cm = search.coarse_masses()
        bank = search.build_bank(cm, N, kind)
        sc = event.scan_bank(prep["H1"], prep["L1"], bank, v0, v1)
        c = summarise(sc, cm)
        fm = search.fine_masses(c["m1"], c["m2"])
        fbank = search.build_bank(fm, N, kind)
        fsc = event.scan_bank(prep["H1"], prep["L1"], fbank, v0, v1)
        f = summarise(fsc, fm)
        out[kind] = {"coarse": c, "fine": f}
        scans[kind] = {"coarse": (cm, sc), "fine": (fm, fsc)}
        print(kind, "coarse", c["m1"], c["m2"], round(c["rho_net"], 2), "fine", f["m1"], f["m2"],
              round(f["rho_h"], 2), round(f["rho_l"], 2), round(f["rho_net"], 2), round(f["dt_ms"], 2), flush=True)
        del bank, fbank
    res["bank"] = out
    best = out["imr"]["fine"]
    best["mc_source_z009"] = best["mc"] / (1 + Z_LIT)
    np.savez(ROOT / "results" / "event_scans.npz",
             **{f"{k}_{g}_{q}": (v[0] if q == "masses" else v[1][q]) for k, d in scans.items() for g, v in d.items()
                for q in ("masses", "net", "rho_h", "rho_l", "i_h", "i_l")})

    # ---------------- merger-ringdown contribution ----------------
    m1, m2 = best["m1"], best["m2"]
    fisco = waveforms.f_isco(m1 + m2)
    f = np.fft.rfftfreq(N, 1.0 / FS)
    h_full = waveforms.imrphenomd(m1, m2, N)
    h_cut = np.where(f <= fisco, h_full, 0)
    h_f2 = waveforms.taylorf2(m1, m2, N)
    mr = {"f_isco": float(fisco), "m1": m1, "m2": m2}
    for name, h in (("imr_full", h_full), ("imr_below_isco", h_cut), ("taylorf2", h_f2)):
        vals = {}
        for d, key in (("H1", "i_h"), ("L1", "i_l")):
            _, v = snr_near(prep[d], h, best[key])
            vals[d] = float(v)
        vals["net"] = float(np.hypot(vals["H1"], vals["L1"]))
        mr[name] = vals
    mr["snr2_fraction_above_isco"] = 1 - mr["imr_below_isco"]["net"] ** 2 / mr["imr_full"]["net"] ** 2
    # expected (optimal) fraction from the template itself
    for d in ("H1", "L1"):
        df = FS / N
        sl = mf.band_slice(N)
        mr[f"expected_fraction_above_isco_{d}"] = float(
            1 - mf.sigmasq(h_cut, prep[d]["inv"], df, sl) / mf.sigmasq(h_full, prep[d]["inv"], df, sl))
    res["merger_ringdown"] = mr

    # ---------------- stage 1 + f(t) ----------------
    white, bp, tpk = {}, {}, {}
    fits_all = {"prereg": {}, "modified": {}, "all_crossings": {}}
    for d, key in (("H1", "i_h"), ("L1", "i_l")):
        from scipy.signal.windows import tukey
        win = tukey(N, 2.0 * const.TUKEY_TAPER * FS / N)
        white[d] = psd.whiten(prep[d]["seg"] * win, prep[d]["inv"])
        bp[d] = psd.bandpass(white[d], 35.0, 350.0)
        ip = best[key]
        a, b = int(ip - 0.1 * FS), int(ip + 0.1 * FS)
        ipk = a + int(np.argmax(np.abs(bp[d][a:b])))
        tpk[d] = tt[ipk]
        for name, skip in (("prereg", False), ("modified", True)):
            tm, ff = chirpfit.f_of_t(tt, bp[d], tt[ipk], skip_high=skip)
            fit = chirpfit.fit_chirp_mass(tm, ff)
            fits_all[name][d] = {"t_pk_rel_s": float(tt[ipk]), "n_points": int(len(ff)), "t": tm.tolist(),
                                 "f": ff.tolist(), "fit": None if fit is None else {k: float(v) for k, v in fit.items()}}
        m_ = (tt >= tt[ipk] - 0.2) & (tt <= tt[ipk])
        zc = chirpfit.zero_crossings(tt[m_], bp[d][m_])
        fits_all["all_crossings"][d] = {"t": (0.5 * (zc[1:] + zc[:-1])).tolist(), "f": (0.5 / np.diff(zc)).tolist()}
    res["chirp_fit_zero_crossing"] = fits_all["prereg"]
    res["chirp_fit_zero_crossing_modified"] = fits_all["modified"]
    res["zero_crossings_all"] = fits_all["all_crossings"]
    fits = fits_all["modified"]

    # Q-transform ridge (reference only)
    freqs = np.geomspace(25, 400, 120)
    qres, ridge = {}, {}
    for d in ("H1", "L1"):
        a, b = int(np.searchsorted(tt, -4.0)), int(np.searchsorted(tt, 4.0))
        e = qtransform.qtransform(white[d][a:b], FS, freqs, q=6.0)
        tq = tt[a:b]
        qres[d] = (tq, e)
        m = (tq >= tpk[d] - 0.2) & (tq <= tpk[d])
        idx = np.flatnonzero(m)[::16]
        fr = freqs[np.argmax(e[:, idx], axis=0)]
        en = e[:, idx].max(axis=0)
        ok = (fr >= 30) & (fr <= 150) & (en > 10)
        fit = chirpfit.fit_chirp_mass(tq[idx][ok], fr[ok]) if ok.sum() >= 4 else None
        ridge[d] = {"n_points": int(ok.sum()), "fit": None if fit is None else {k: float(v) for k, v in fit.items()},
                    "t": tq[idx][ok].tolist(), "f": fr[ok].tolist()}
    res["chirp_fit_q_ridge"] = ridge

    # noise numbers for the "when does the signal become visible" discussion
    vis = {}
    for d, key in (("H1", "i_h"), ("L1", "i_l")):
        hfit = event.best_fit_signal(prep[d], h_full, N, best[key])
        sig_td = np.fft.irfft(hfit, N) * FS
        sig_w = psd.whiten(sig_td, prep[d]["inv"])
        sig_bp = psd.bandpass(sig_w, 35.0, 350.0)
        quiet = (np.abs(tt) > 1) & (tt > -7) & (tt < 7)
        vis[d] = {"raw_rms": float(prep[d]["seg"].std()), "signal_peak_strain": float(np.abs(sig_td).max()),
                  "raw_peak_over_rms": float(np.abs(sig_td).max() / prep[d]["seg"].std()),
                  "white_rms": float(white[d][quiet].std()), "white_signal_peak": float(np.abs(sig_w).max()),
                  "bp_rms": float(bp[d][quiet].std()), "bp_signal_peak": float(np.abs(sig_bp).max()),
                  "bp_peak_over_rms": float(np.abs(sig_bp).max() / bp[d][quiet].std()),
                  "bp_data_peak_over_rms": float(np.abs(bp[d][np.abs(tt) < 0.1]).max() / bp[d][quiet].std())}
        vis[d]["_sig"] = (sig_td, sig_bp)
    res["visibility"] = {d: {k: v for k, v in vis[d].items() if not k.startswith("_")} for d in vis}

    json.dump(res, open(ROOT / "results" / "event.json", "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("found_time_gps", "offset_from_catalog_s", "merger_ringdown")}, indent=1))

    # ---------------- figures ----------------
    FIG.mkdir(parents=True, exist_ok=True)
    # fig 1: ASD
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for d in ("H1", "L1"):
        m = (prep[d]["f"] >= 10) & (prep[d]["f"] <= 2000)
        ax.loglog(prep[d]["f"][m], np.sqrt(prep[d]["psd"][m]), color=COL[d], lw=1, label=d)
    ax.axvspan(const.F_LOW, const.F_HIGH, color="#8a8983", alpha=0.08, lw=0)
    ax.text(22, ax.get_ylim()[1] * 0.4, "filter band 20–1024 Hz", color="#52514e", fontsize=8)
    ax.set(xlabel="frequency (Hz)", ylabel="ASD (strain / √Hz)", xlim=(10, 2000), ylim=(2e-24, 1e-19),
           title="Noise amplitude spectral density (median Welch, 512 s around the event)")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "01_asd.png"); plt.close(fig)

    # fig 2: raw vs whitened vs band-passed
    fig, axs = plt.subplots(3, 2, figsize=(10, 7), sharex=True)
    m = (tt > -0.25) & (tt < 0.08)
    for c, d in enumerate(("H1", "L1")):
        sig_td, sig_bp = vis[d]["_sig"]
        axs[0, c].plot(tt[m], prep[d]["seg"][m] * 1e18, color=COL[d])
        axs[0, c].plot(tt[m], sig_td[m] * 1e18, color=COL["model"], lw=1, label="best-fit template (same scale)")
        axs[0, c].set_title(f"{d}: raw strain (×10⁻¹⁸)")
        axs[1, c].plot(tt[m], white[d][m], color=COL[d])
        axs[1, c].set_title(f"{d}: whitened, no band-pass (σ units)")
        axs[2, c].plot(tt[m], bp[d][m], color=COL[d], label="data")
        axs[2, c].plot(tt[m], sig_bp[m], color=COL["model"], lw=1, label="best-fit template")
        axs[2, c].set_title(f"{d}: whitened + 35–350 Hz band-pass")
        axs[2, c].set_xlabel("time relative to the H1 trigger (s)")
        axs[0, c].legend(fontsize=8); axs[2, c].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "02_raw_vs_whitened.png"); plt.close(fig)

    # fig 3: Q-transform
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, d in zip(axs, ("H1", "L1")):
        tq, e = qres[d]
        m = (tq > -0.35) & (tq < 0.1)
        im = ax.pcolormesh(tq[m][::8], freqs, e[:, m][:, ::8], cmap="Blues", vmin=0, vmax=25, shading="auto", rasterized=True)
        ax.set_yscale("log"); ax.set_title(d)
        ax.set_xlabel("time relative to the H1 trigger (s)"); ax.grid(False)
        ax.set_yticks([32, 64, 128, 256]); ax.set_yticklabels(["32", "64", "128", "256"])
        ax.minorticks_off()
    axs[0].set_ylabel("frequency (Hz)")
    fig.suptitle("Constant-Q transform (Q = 6) of whitened strain", fontsize=11)
    fig.colorbar(im, ax=axs, label="normalised energy (white noise mean = 1)", pad=0.01)
    fig.savefig(FIG / "03_qtransform.png", bbox_inches="tight"); plt.close(fig)

    # fig 4: SNR time series
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    rr = {}
    for d in ("H1", "L1"):
        rr[d], _, _ = mf.snr_series(prep[d]["s_f"], h_full, prep[d]["inv"], N, FS)
    for d in ("H1", "L1"):
        axs[0].plot(tt[v0:v1], rr[d][v0:v1], color=COL[d], lw=0.7, label=d)
        m = np.abs(tt) < 0.03
        axs[1].plot(tt[m] * 1e3, rr[d][m], color=COL[d], marker="o", ms=2.5, label=d)
    axs[0].set(xlabel="time relative to the H1 trigger (s)", ylabel="matched-filter SNR ρ",
               title="16-s valid region, best IMRPhenomD template")
    axs[1].set(xlabel="time relative to the H1 trigger (ms)", title=f"zoom: L1 peak precedes H1 by {best['dt_ms']:.2f} ms")
    axs[0].legend(); axs[1].legend()
    fig.tight_layout(); fig.savefig(FIG / "04_snr_timeseries.png"); plt.close(fig)

    # fig 5: bank maps
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.3))
    for ax, g in zip(axs, ("coarse", "fine")):
        masses, sc = scans["imr"][g]
        s = ax.scatter(masses[:, 0], masses[:, 1], c=sc["net"], cmap="Blues", s=14 if g == "coarse" else 22, marker="s",
                       vmin=np.percentile(sc["net"], 5))
        k = int(np.argmax(sc["net"]))
        ax.plot(*masses[k], marker="x", color=COL["L1"], ms=9, mew=2, label="maximum")
        g1 = np.linspace(masses[:, 0].min(), masses[:, 0].max(), 100)
        g2 = np.linspace(masses[:, 1].min(), masses[:, 1].max(), 100)
        G1, G2 = np.meshgrid(g1, g2)
        mcg = np.where(G1 >= G2, waveforms.chirp_mass(G1, G2), np.nan)
        cs = ax.contour(G1, G2, mcg, levels=[28.5, 30.4, 32.5], colors="#52514e", linewidths=0.8, linestyles=["--", "-", "--"])
        ax.clabel(cs, fmt="ℳ=%.1f", fontsize=7)
        ax.set(xlabel="m1 (M☉, detector frame)", ylabel="m2 (M☉, detector frame)", title=f"{g} bank: network SNR")
        ax.grid(False); ax.legend(loc="upper left")
        fig.colorbar(s, ax=ax, label="ρ_net")
    fig.tight_layout(); fig.savefig(FIG / "05_bank_snr_map.png"); plt.close(fig)

    # fig 6: f(t) fit
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for d in ("H1", "L1"):
        r = fits[d]
        t_, f_ = np.array(r["t"]), np.array(r["f"])
        ac = res["zero_crossings_all"][d]
        axs[0].plot(ac["t"], ac["f"], "o", color=COL[d], ms=4, mfc="none", alpha=0.6)
        axs[0].plot(t_, f_, "o", color=COL[d], ms=5, label=f"{d} zero crossings (filled: used in fit)")
        axs[1].plot(t_, f_ ** (-8 / 3) * 1e5, "o", color=COL[d], ms=5)
        if r["fit"] and np.isfinite(r["fit"]["mc"]):
            x = np.linspace(t_.min(), t_.max(), 50)
            kk = (8 * np.pi) ** (8 / 3) / 5 * (r["fit"]["mc"] * const.MTSUN) ** (5 / 3)
            axs[1].plot(x, kk * (r["fit"]["tc"] - x) * 1e5, color=COL[d], lw=1,
                        label=f"{d}: ℳ = {r['fit']['mc']:.1f} ± {r['fit']['mc_err']:.1f} M☉")
    axs[0].set(xlabel="time relative to the H1 trigger (s)", ylabel="f (Hz)", title="frequency from zero crossings", ylim=(0, 400))
    axs[0].axhspan(30, 150, color="#8a8983", alpha=0.08, lw=0)
    axs[1].set(xlabel="time relative to the H1 trigger (s)", ylabel="f^(−8/3) (×10⁻⁵ Hz^(−8/3))", title="Newtonian chirp: straight line")
    axs[0].legend(); axs[1].legend()
    fig.tight_layout(); fig.savefig(FIG / "06_chirp_fit.png"); plt.close(fig)


if __name__ == "__main__":
    main()
