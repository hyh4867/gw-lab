"""GW151226: loudest zero-lag coincidence of the low-mass search, then event measurements
(pre-registration stage 5, section 4). Writes results/event_gw151226.json and figure 12.
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from gwlab import coinc, const, event, mf, plotstyle, psd, search, waveforms  # noqa: E402
import evcommon  # noqa: E402
from background5 import load_file  # noqa: E402

FS = const.FS
NAME = "GW151226"
EV = evcommon.EVENTS[NAME]
LAY = EV["layout"]
SEED = 20261116
N_FINE = 4000
CHUNK = 400


def scan_params(par, prep, n, v0, v1):
    """scan_bank over rows (m1, m2, chi1, chi2) in chunks; returns concatenated dict."""
    acc = None
    for c in range(0, len(par), CHUNK):
        bank = search.build_bank(par[c:c + CHUNK], n, "imr")
        s = event.scan_bank(prep["H1"], prep["L1"], bank, v0, v1)
        acc = s if acc is None else {k: np.concatenate([acc[k], s[k]]) for k in s}
        del bank
    return acc


def summarise(scan, par, k=None):
    k = int(np.argmax(scan["net"])) if k is None else k
    m1, m2, chi = par[k, 0], par[k, 1], par[k, 2]
    return {"m1": float(m1), "m2": float(m2), "chi": float(chi), "q": float(m2 / m1),
            "mc": float(waveforms.chirp_mass(m1, m2)), "mtot": float(m1 + m2),
            "rho_h": float(scan["rho_h"][k]), "rho_l": float(scan["rho_l"][k]), "rho_net": float(scan["net"][k]),
            "dt_ms": float((scan["i_h"][k] - scan["i_l"][k]) / FS * 1e3),
            "i_h": float(scan["i_h"][k]), "i_l": float(scan["i_l"][k])}


def main():
    res = {}
    trig, live, skipped = load_file("lowmass", EV["file"], LAY)
    th, tl = trig["H1"], trig["L1"]
    res["triggers"] = {"n_h": len(th), "n_l": len(tl), "live_intervals_s": live, "n_skipped_segments": len(skipped)}
    i, j = coinc.zero_lag(th["t"], tl["t"])
    stat = coinc.combined(th["newsnr"][i], tl["newsnr"][j])
    order = np.argsort(stat)[::-1]
    res["zero_lag_top5"] = [{"t_h": float(th["t"][i[o]]), "t_l": float(tl["t"][j[o]]), "newsnr_c": float(stat[o]),
                             "rho_h": float(th["rho"][i[o]]), "rho_l": float(tl["rho"][j[o]]),
                             "chi2r_h": float(th["chi2r"][i[o]]), "chi2r_l": float(tl["chi2r"][j[o]]),
                             "tmpl_h": int(th["tmpl"][i[o]]), "tmpl_l": int(tl["tmpl"][j[o]])} for o in order[:5]]
    res["n_zero_lag"] = int(len(stat))
    top = res["zero_lag_top5"][0]
    t_ev = top["t_h"]
    res["found_time_gps"] = EV["file"] + t_ev
    res["offset_from_catalog_s"] = EV["file"] + t_ev - EV["t0"]
    res["G1_found"] = bool(abs(res["offset_from_catalog_s"]) <= 0.1)
    print("zero lag", res["zero_lag_top5"][:2], res["offset_from_catalog_s"], flush=True)

    prep, i0, n = evcommon.preps(NAME, t_ev)
    res["segment"] = {"index": int(i0 // (LAY.stride * FS)), "start_s": i0 / FS}
    c = int(round(t_ev * FS)) - i0
    v0, v1 = c - int(0.1 * FS), c + int(0.1 * FS)

    b = np.loadtxt(ROOT / "results" / "bank_lowmass.csv", delimiter=",", skiprows=1)
    par_c = np.column_stack([b[:, 3], b[:, 4], b[:, 2], b[:, 2]])
    sc = scan_params(par_c, prep, n, v0, v1)
    res["coarse"] = summarise(sc, par_c)
    res["coarse"]["eta"] = float(b[int(np.argmax(sc["net"])), 1])
    print("coarse", res["coarse"], flush=True)

    rng = np.random.default_rng(SEED)
    mcb, chib = res["coarse"]["mc"], res["coarse"]["chi"]
    mc = rng.uniform(mcb - 0.3, mcb + 0.3, N_FINE)
    eta = rng.uniform(0.10, 0.25, N_FINE)
    chi = rng.uniform(max(-0.5, chib - 0.3), min(0.8, chib + 0.3), N_FINE)
    mm = np.array([waveforms.masses_from_mc_eta(a, e) for a, e in zip(mc, eta)])
    par_f = np.column_stack([mm[:, 0], mm[:, 1], chi, chi])
    sf = scan_params(par_f, prep, n, v0, v1)
    fine = summarise(sf, par_f)
    ok = sf["net"] ** 2 >= sf["net"].max() ** 2 - 6.25
    fine["region"] = {"n": int(ok.sum()), "mc": [float(mc[ok].min()), float(mc[ok].max())],
                      "q": [float((mm[ok, 1] / mm[ok, 0]).min()), float((mm[ok, 1] / mm[ok, 0]).max())],
                      "chi": [float(chi[ok].min()), float(chi[ok].max())],
                      "m1": [float(mm[ok, 0].min()), float(mm[ok, 0].max())],
                      "m2": [float(mm[ok, 1].min()), float(mm[ok, 1].max())]}
    fine["mc_source_z009"] = fine["mc"] / (1 + EV["z"])
    res["fine"] = fine
    print("fine", fine, flush=True)

    par_0 = np.column_stack([mm[:, 0], mm[:, 1], np.zeros(N_FINE), np.zeros(N_FINE)])
    s0 = scan_params(par_0, prep, n, v0, v1)
    res["fine_nonspinning"] = summarise(s0, par_0)
    res["snr_fraction_lost_without_spin"] = 1 - res["fine_nonspinning"]["rho_net"] / fine["rho_net"]
    print("nonspinning", res["fine_nonspinning"], flush=True)

    # chi-squared and re-weighted SNR of the best template
    h = waveforms.imrphenomd(fine["m1"], fine["m2"], n, FS, chi1=fine["chi"], chi2=fine["chi"])
    sl = mf.band_slice(n, FS)
    rw = {}
    for det, key in (("H1", "i_h"), ("L1", "i_l")):
        c2, dof = mf.chisq(prep[det]["s_f"], h, prep[det]["inv"], n, fine[key], FS, sl=sl)
        rho = fine["rho_h" if det == "H1" else "rho_l"]
        rw[det] = {"chi2r": c2 / dof, "newsnr": float(mf.newsnr(np.array([rho]), np.array([c2 / dof]))[0])}
    rw["newsnr_c"] = float(np.hypot(rw["H1"]["newsnr"], rw["L1"]["newsnr"]))
    res["reweighted"] = rw

    lit = {"G2": (fine["rho_h"], 9.45, 11.55), "G3": (fine["rho_l"], 7.11, 8.69), "G4": (fine["rho_net"], 11.7, 14.3),
           "G5": (fine["dt_ms"], 0.68, 1.52), "G6": (fine["mc"], 9.66, 9.79), "G7": (fine["mc_source_z009"], 8.6, 9.2)}
    res["verdicts"] = {k: {"value": float(v), "range": [lo, hi], "reproduced": bool(lo <= v <= hi)} for k, (v, lo, hi) in lit.items()}
    res["verdicts"]["G1"] = {"value": res["offset_from_catalog_s"], "range": [-0.1, 0.1], "reproduced": res["G1_found"]}
    json.dump(res, open(ROOT / "results" / "event_gw151226.json", "w"), indent=1)
    print(json.dumps(res["verdicts"], indent=1))

    # figure: whitened, band-passed data against the best-fit template, and the SNR time series
    plt = plotstyle.apply()
    COL = plotstyle.COL
    fig, axs = plt.subplots(2, 2, figsize=(11, 6), sharex="col")
    from scipy.signal.windows import tukey
    for r, (det, key) in enumerate((("H1", "i_h"), ("L1", "i_l"))):
        p = prep[det]
        xw = psd.bandpass(psd.whiten(p["seg"] * tukey(n, 2.0 * const.TUKEY_TAPER * FS / n), p["inv"], FS), 35, 350)
        sig = event.best_fit_signal(p, h, n, fine[key], FS, sl)
        sw = psd.bandpass(np.fft.irfft(sig * np.sqrt(p["inv"] * 2.0 / FS) * FS, n), 35, 350)
        t = (np.arange(n) - fine[key]) / FS
        m = (t > -1.0) & (t < 0.05)
        ax = axs[r, 0]
        ax.plot(t[m], xw[m], color=COL["muted"], lw=0.6, label="data (whitened, 35-350 Hz)")
        ax.plot(t[m], sw[m], color=COL[det], lw=1.0, label="best-fit template")
        ax.set(ylabel=f"{det} whitened strain")
        ax.legend(loc="upper left", fontsize=8)
        rho, _, _ = mf.snr_series(p["s_f"], h, p["inv"], n, FS, sl)
        m2 = (t > -0.25) & (t < 0.25)
        axs[r, 1].plot(t[m2], rho[m2], color=COL[det], lw=0.8)
        axs[r, 1].set(ylabel=f"{det}  SNR ρ(t)")
    axs[1, 0].set(xlabel="time from the SNR peak (s)")
    axs[1, 1].set(xlabel="time from the SNR peak (s)")
    fig.suptitle(f"GW151226: the best-fit template is buried in the noise sample by sample; "
                 f"matched filtering gives ρ = {fine['rho_h']:.1f} (H1), {fine['rho_l']:.1f} (L1)", fontsize=10)
    fig.tight_layout(); fig.savefig(ROOT / "docs" / "figures" / "12_gw151226_event.png"); plt.close(fig)


if __name__ == "__main__":
    main()
