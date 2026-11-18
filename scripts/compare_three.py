"""Three events side by side: two black-hole mergers and one neutron-star merger
(pre-registration stage 6, sections 5 and 9). Also the new TaylorF2 measurement on GW151226.

Writes results/compare_three.json and docs/figures/24_three_events.png
"""
import json
import pathlib
import sys

import numpy as np
from scipy.signal.windows import tukey

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from gwlab import bns, const, event, mf, plotstyle, psd, search, waveforms  # noqa: E402
import ev6common as E  # noqa: E402
import evcommon  # noqa: E402
import compare_events as CE  # noqa: E402

FS = const.FS


def load(name):
    return json.load(open(ROOT / "results" / name))


def tf2_gw151226():
    """Hand-written non-spinning TaylorF2 on the stage-5 fine-bank (Mc, eta) points of GW151226."""
    ev = load("event_gw151226.json")
    b = evcommon.best("GW151226")
    t_ev = b["seg_start_s"] + b["i_h"] / FS
    prep, i0, n = evcommon.preps("GW151226", t_ev)
    c = int(round(b["i_h"]))
    v0, v1 = c - int(0.1 * FS), c + int(0.1 * FS)
    rng = np.random.default_rng(20261116)                       # the stage-5 fine bank draws, repeated
    mcb = ev["coarse"]["mc"]
    mc = rng.uniform(mcb - 0.3, mcb + 0.3, 4000)
    eta = rng.uniform(0.10, 0.25, 4000)
    mm = np.array([waveforms.masses_from_mc_eta(a, e) for a, e in zip(mc, eta)])
    acc = None
    for s in range(0, len(mm), 400):
        bank = search.build_bank(mm[s:s + 400], n, "f2")
        r = event.scan_bank(prep["H1"], prep["L1"], bank, v0, v1)
        acc = r if acc is None else {k: np.concatenate([acc[k], r[k]]) for k in r}
        del bank
    k = int(np.argmax(acc["net"]))
    return {"m1": float(mm[k, 0]), "m2": float(mm[k, 1]), "mc": float(mc[k]), "rho_h": float(acc["rho_h"][k]),
            "rho_l": float(acc["rho_l"][k]), "rho_net": float(acc["net"][k]),
            "imrphenomd_nonspinning_rho_net": ev["fine_nonspinning"]["rho_net"], "imrphenomd_aligned_spin_rho_net": ev["fine"]["rho_net"],
            "ratio_to_nonspinning_imr": float(acc["net"][k] / ev["fine_nonspinning"]["rho_net"]),
            "ratio_to_aligned_spin_imr": float(acc["net"][k] / ev["fine"]["rho_net"])}


def bbh(name, aux):
    """Cycles and time from 30 Hz to the amplitude peak, and the f(t) track, from the best template."""
    t, hc, ipk = aux["t"], aux["h"], aux["ipk"]
    ph = np.unwrap(np.angle(hc))
    finst = np.abs(np.gradient(ph, t)) / (2 * np.pi)
    i30 = CE.f_crossing(t, finst, ipk, 30.0)
    m = slice(i30, ipk + 1)
    return {"cycles_30hz_to_peak": float(abs(ph[ipk] - ph[i30]) / (2 * np.pi)), "time_30hz_to_peak_s": float(t[ipk] - t[i30])}, \
        (t[ipk] - t[m], finst[m])


def gw170817():
    ev = load("event_gw170817.json")
    v = ev["variants"]["c_B"]
    best = v["fine"]
    i0 = int(round((v["segment_start_gps"] - E.B_START) * FS))
    basis = bns.basis_for()
    sl = basis.sl
    h = basis.full(best["m1"], best["m2"])
    f = np.fft.rfftfreq(E.N, 1.0 / FS)
    out = {"m1": best["m1"], "m2": best["m2"], "mc": best["mc"], "rho_h": best["rho_h"], "rho_l": best["rho_l"],
           "rho_v": v["V1"]["rho_at_expected_time"], "rho_net": v["rho_net_hlv"], "f_isco_hz": float(waveforms.f_isco(best["mtot"]))}
    cum = {}
    for det, key in (("H1", "i_h"), ("L1", "i_l")):
        x, _ = E.load("B", det)
        p = E.prep(x, "B", i0)
        del x
        wgt = np.abs(h[sl]) ** 2 * p["inv"][sl]
        c = np.cumsum(wgt) / wgt.sum()
        cum[det] = c
        out[f"snr2_freq_{det}"] = {"f05": float(f[sl][np.searchsorted(c, 0.05)]), "f50": float(f[sl][np.searchsorted(c, 0.5)]),
                                   "f95": float(f[sl][np.searchsorted(c, 0.95)]), "fraction_above_f_isco": 0.0}
        xb = psd.bandpass(psd.whiten(p["seg"] * tukey(E.N, 2.0 * const.TUKEY_TAPER * FS / E.N), p["inv"], FS), 35, 350)
        sig = event.best_fit_signal(p, h, E.N, best[key], FS, sl)
        sb = psd.bandpass(np.fft.irfft(sig * np.sqrt(p["inv"] * 2.0 / FS) * FS, E.N), 35, 350)
        rms = float(np.std(xb[20 * FS:100 * FS]))
        out.setdefault("visibility", {})[det] = {"bp_signal_peak_over_noise_rms": float(np.abs(sb).max() / rms), "bp_noise_rms": rms}
    dur, cyc = E.f2_time_and_cycles(best["m1"], best["m2"], 30.0)
    d35, c35 = E.f2_time_and_cycles(best["m1"], best["m2"], 35.0)
    out.update({"cycles_30hz_to_isco": cyc, "time_30hz_to_isco_s": dur, "cycles_35hz_to_isco": c35, "time_35hz_to_isco_s": d35,
                "cycles_30hz_to_1024hz": E.f2_time_and_cycles(best["m1"], best["m2"], 30.0, 1024.0)[1]})
    ff = np.geomspace(30, out["f_isco_hz"], 2000)
    tau = -np.gradient(waveforms.taylorf2_phase(ff, best["m1"], best["m2"]), ff) / (2 * np.pi)
    tau = tau - tau[-1]
    return out, {"f": f[sl], "cum": cum, "track": (tau, ff)}


def main():
    res, aux, tracks = {}, {}, {}
    for name in ("GW150914", "GW151226"):
        r, a = CE.analyse(name)
        extra, tracks[name] = bbh(name, a)
        r.update(extra)
        res[name], aux[name] = r, a
        print(name, extra, flush=True)
    res["GW170817"], aux["GW170817"] = gw170817()
    tracks["GW170817"] = aux["GW170817"]["track"]
    tf2 = tf2_gw151226()
    res["GW151226"]["taylorf2"] = tf2
    print("GW151226 TaylorF2", tf2, flush=True)
    e1, e3 = load("event.json"), load("event_gw170817.json")
    r1 = e1["bank"]["f2"]["fine"]["rho_net"] / e1["bank"]["imr"]["fine"]["rho_net"]
    r3 = e3["taylorf2_vs_imrphenomd"]["ratio_net"]
    res["taylorf2_over_imrphenomd_network_snr"] = {"GW150914": r1, "GW151226": tf2["ratio_to_nonspinning_imr"], "GW170817": r3}
    res["L2"] = {"order_ok": bool(r1 < tf2["ratio_to_nonspinning_imr"] < r3)}
    # Newtonian chirp filter and posterior widths
    pe = {n: load(f"pe_bilby_{n}.json") for n in ("GW150914", "GW151226", "GW170817") if (ROOT / "results" / f"pe_bilby_{n}.json").exists()}
    for n in res:
        if not n.startswith("GW"):
            continue
        p = ROOT / "results" / f"chirp_newton_{n}.json"
        if p.exists():
            c = load(f"chirp_newton_{n}.json")
            d = c["detectors"]
            res[n]["newton"] = {"raw_h1": d["H1"]["event"]["mc_raw"], "raw_l1": d["L1"]["event"]["mc_raw"],
                                "calibrated": c.get("combined", {}).get("mc_cal"), "sigma": c.get("combined", {}).get("sigma"),
                                "snr_fraction_h1": d["H1"]["event"]["snr_newton_over_full"],
                                "snr_fraction_l1": d["L1"]["event"]["snr_newton_over_full"]}
        if n in pe:
            m = pe[n]["mc_det"]
            res[n]["pe"] = {"mc_det": m, "mc_width90_rel": (m["hi"] - m["lo"]) / m["median"], "q": pe[n]["q"],
                            "m1_det": pe[n]["m1_det"], "m2_det": pe[n]["m2_det"], "chi_eff": pe[n]["chi_eff"],
                            "distance_mpc": pe[n]["luminosity_distance_mpc"]}
            if "newton" in res[n]:
                res[n]["newton"]["raw_h1_over_pe_median"] = res[n]["newton"]["raw_h1"] / m["median"]
    res["bank_sizes"] = {"GW150914": 496, "GW151226": int(load("bank_lowmass.json")["n_templates"]),
                         "GW170817": int(load("bank_bns.json")["n_templates"])}
    json.dump(res, open(ROOT / "results" / "compare_three.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("GW")}, indent=1), flush=True)

    plt = plotstyle.apply()
    COL = plotstyle.COL
    cols = {"GW150914": COL["H1"], "GW151226": COL["L1"], "GW170817": COL["alt"]}
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.4))
    ax = axs[0]
    for n, (tau, ff) in tracks.items():
        m = tau > 1e-3
        key = "cycles_30hz_to_peak" if n != "GW170817" else "cycles_30hz_to_isco"
        ax.plot(tau[m], ff[m], color=cols[n], lw=1.6, label=f"{n}: {res[n][key]:.0f} cycles from 30 Hz")
    ax.axhline(1024, color="#52514e", lw=0.6, ls=":")
    ax.set(xscale="log", yscale="log", xlabel="time before merger (s)", ylabel="gravitational-wave frequency (Hz)",
           title="frequency against time of the best templates", xlim=(100, 1e-3), ylim=(25, 2000))
    plotstyle.plain_log_ticks(ax, "y", [30, 100, 300, 1000])
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax = axs[1]
    for n in ("GW150914", "GW151226", "GW170817"):
        ax.plot(aux[n]["f"], aux[n]["cum"]["H1"], color=cols[n], lw=1.6, label=n)
        if res[n]["f_isco_hz"] < 1024:
            ax.axvline(res[n]["f_isco_hz"], color=cols[n], lw=0.8, ls="--")
    ax.set(xscale="log", xlim=(20, 1024), xlabel="frequency (Hz)", ylabel="cumulative fraction of SNR² (H1)",
           title="where the SNR is collected (dashed: f_ISCO)")
    plotstyle.plain_log_ticks(ax, "x", [20, 50, 100, 200, 500, 1000])
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax = axs[2]
    names = ["GW150914", "GW151226", "GW170817"]
    vals = [res["taylorf2_over_imrphenomd_network_snr"][n] for n in names]
    ax.bar(range(3), vals, color=[cols[n] for n in names], width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
    ax.axhline(1.0, color="#52514e", lw=0.6, ls=":")
    ax.set(xticks=range(3), ylim=(0, 1.15), ylabel="network SNR: TaylorF2 / IMRPhenomD (non-spinning)",
           title="is an inspiral-only template enough?")
    ax.set_xticklabels(names)
    ax.grid(False)
    fig.tight_layout(); fig.savefig(ROOT / "docs" / "figures" / "24_three_events.png"); plt.close(fig)


if __name__ == "__main__":
    main()
