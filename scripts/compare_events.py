"""Why GW151226 is harder to find than GW150914, in numbers (pre-registration stage 5, section 4).

Writes results/compare_events.json and docs/figures/13_two_events.png
"""
import json
import pathlib
import sys

import numpy as np
from scipy.signal.windows import tukey

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from gwlab import const, event, mf, plotstyle, psd, qtransform, waveforms  # noqa: E402
import evcommon  # noqa: E402

FS = const.FS
LIT_CYCLES = {"GW150914": (8, 6, 10), "GW151226": (55, 44, 66)}


def time_domain(h_f, n):
    """Complex (analytic) time series of a template whose merger is at t = 0 (wrapped to the middle)."""
    buf = np.zeros(n, dtype=np.complex128)
    k = len(h_f)
    buf[:k] = 2.0 * h_f
    hc = np.fft.ifft(buf) * FS
    return np.roll(hc, n // 2)


def f_crossing(t, f, amp_pk_i, f0):
    """Last time before the amplitude peak at which the instantaneous frequency is below f0."""
    i = amp_pk_i
    while i > 0 and f[i] > f0:
        i -= 1
    return i


def analyse(name):
    b = evcommon.best(name)
    t_h = b["seg_start_s"] + b["i_h"] / FS
    prep, i0, n = evcommon.preps(name, t_h)
    h = evcommon.template(b, n)
    sl = mf.band_slice(n, FS)
    f = np.fft.rfftfreq(n, 1.0 / FS)
    out = {"m1": b["m1"], "m2": b["m2"], "chi": b["chi"], "mc": float(waveforms.chirp_mass(b["m1"], b["m2"])),
           "rho_h": b["rho_h"], "rho_l": b["rho_l"], "rho_net": b["rho_net"],
           "f_isco_hz": float(waveforms.f_isco(b["m1"] + b["m2"]))}
    # (b) where the SNR comes from
    cum = {}
    for det in ("H1", "L1"):
        wgt = np.abs(h[sl]) ** 2 * prep[det]["inv"][sl]
        c = np.cumsum(wgt) / wgt.sum()
        cum[det] = c
        fr = {p: float(f[sl][np.searchsorted(c, p / 100)]) for p in (5, 50, 95)}
        out[f"snr2_freq_{det}"] = {"f05": fr[5], "f50": fr[50], "f95": fr[95],
                                   "fraction_above_f_isco": float(1 - c[np.searchsorted(f[sl], out["f_isco_hz"])])}
    # (c) cycles and time in band from the template phase
    hc = time_domain(h, n)
    t = (np.arange(n) - n // 2) / FS
    amp = np.abs(hc)
    ph = np.unwrap(np.angle(hc))
    finst = np.gradient(ph, t) / (2 * np.pi)
    finst = np.abs(finst)
    ipk = int(np.argmax(amp))
    i35 = f_crossing(t, finst, ipk, 35.0)
    i20 = f_crossing(t, finst, ipk, 22.0)
    out["f_at_amplitude_peak_hz"] = float(finst[ipk])
    out["time_35hz_to_peak_s"] = float(t[ipk] - t[i35])
    out["cycles_35hz_to_peak"] = float(abs(ph[ipk] - ph[i35]) / (2 * np.pi))
    out["time_22hz_to_peak_s"] = float(t[ipk] - t[i20])
    out["cycles_22hz_to_peak"] = float(abs(ph[ipk] - ph[i20]) / (2 * np.pi))
    f05, f95 = out["snr2_freq_H1"]["f05"], out["snr2_freq_H1"]["f95"]
    ia, ib = f_crossing(t, finst, ipk, f05), min(ipk, f_crossing(t, finst, ipk, f95) if f95 < finst[ipk] else ipk)
    out["cycles_f05_to_f95_H1"] = float(abs(ph[ib] - ph[ia]) / (2 * np.pi))
    out["time_f05_to_f95_H1_s"] = float(t[ib] - t[ia])
    lit, lo, hi = LIT_CYCLES[name]
    out["G12"] = {"value": out["cycles_35hz_to_peak"], "literature": lit, "range": [lo, hi],
                  "reproduced": bool(lo <= out["cycles_35hz_to_peak"] <= hi)}
    # (e) visibility sample by sample, (f) Q-transform peak
    vis, qm = {}, {}
    for det, key in (("H1", "i_h"), ("L1", "i_l")):
        p = prep[det]
        xw = psd.whiten(p["seg"] * tukey(n, 2.0 * const.TUKEY_TAPER * FS / n), p["inv"], FS)
        xb = psd.bandpass(xw, 35, 350)
        sig = event.best_fit_signal(p, h, n, b[key], FS, sl)
        sb = psd.bandpass(np.fft.irfft(sig * np.sqrt(p["inv"] * 2.0 / FS) * FS, n), 35, 350)
        lay = evcommon.EVENTS[name]["layout"]
        rms = float(np.std(xb[lay.valid_start * FS:lay.valid_end * FS]))
        c = int(round(b[key]))
        freqs = np.geomspace(25, 600, 150)
        e = qtransform.qtransform(xw[c - 2 * FS:c + 2 * FS], FS, freqs, q=6.0)
        near = e[:, 2 * FS - int(0.15 * FS):2 * FS + int(0.05 * FS)]
        # the same statistic in off-source windows of the same segment, to see what noise alone gives
        offs = []
        for s in range(lay.valid_start + 3, lay.valid_end - 3):
            cc = s * FS
            if abs(cc - c) < 3 * FS:
                continue
            eo = qtransform.qtransform(xw[cc - 2 * FS:cc + 2 * FS], FS, freqs, q=6.0)
            offs.append(float(eo[:, 2 * FS - int(0.15 * FS):2 * FS + int(0.05 * FS)].max()))
        vis[det] = {"bp_signal_peak_over_noise_rms": float(np.abs(sb).max() / rms), "bp_noise_rms": rms,
                    "q_energy_max_at_event": float(near.max()),
                    "q_energy_max_offsource_median": float(np.median(offs)), "q_energy_max_offsource_max": float(np.max(offs)),
                    "n_offsource_windows": len(offs)}
        a_, b_ = 2 * FS - int(1.0 * FS), 2 * FS + int(0.1 * FS)
        qm[det] = (np.arange(a_, b_) / FS - 2.0, freqs, e[:, a_:b_])
    out["visibility"] = vis
    return out, {"f": f[sl], "cum": cum, "q": qm, "t": t, "h": hc, "ipk": ipk}


def main():
    res, aux = {}, {}
    for name in ("GW150914", "GW151226"):
        res[name], aux[name] = analyse(name)
        print(name, json.dumps(res[name], indent=1), flush=True)
    for name in res:
        p = ROOT / "results" / f"chirp_newton_{name}.json"
        if p.exists():
            c = json.load(open(p))
            res[name]["newtonian_chirp_snr_over_full"] = {d: c["detectors"][d]["event"]["snr_newton_over_full"] for d in ("H1", "L1")}
    e2 = json.load(open(ROOT / "results" / "event_gw151226.json"))
    res["GW151226"]["snr_fraction_lost_without_spin"] = e2["snr_fraction_lost_without_spin"]
    json.dump(res, open(ROOT / "results" / "compare_events.json", "w"), indent=1)

    plt = plotstyle.apply()
    COL = plotstyle.COL
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 7.2))
    vmax = 60.0
    for ax, name in zip(axs[0], ("GW150914", "GW151226")):
        tq, fq, e = aux[name]["q"]["H1"]
        pc = ax.pcolormesh(tq[::4], fq, e[:, ::4], shading="auto", cmap="viridis", vmin=0, vmax=vmax, rasterized=True)
        ax.set(yscale="log", xlabel="time from the SNR peak (s)", ylabel="frequency (Hz)",
               title=f"{name}, H1 Q-transform (ρ_H1 = {res[name]['rho_h']:.1f}, peak energy {res[name]['visibility']['H1']['q_energy_max_at_event']:.0f})")
        plotstyle.plain_log_ticks(ax, "y", [30, 50, 100, 200, 400])
        fig.colorbar(pc, ax=ax, label="normalised energy (same scale)")
    ax = axs[1, 0]
    for name, ls in (("GW150914", "-"), ("GW151226", "--")):
        ax.plot(aux[name]["f"], aux[name]["cum"]["H1"], color=COL["H1"], ls=ls, label=f"{name} (H1)")
        ax.axvline(res[name]["f_isco_hz"], color="#52514e", lw=0.8, ls=ls)
    ax.set(xscale="log", xlim=(20, 1000), xlabel="frequency (Hz)", ylabel="cumulative fraction of SNR²",
           title="where the SNR is collected (vertical lines: f_ISCO)")
    plotstyle.plain_log_ticks(ax, "x", [20, 50, 100, 200, 500, 1000])
    ax.legend(loc="lower right")
    ax = axs[1, 1]
    for name, ls, c in (("GW150914", "-", COL["H1"]), ("GW151226", "-", COL["L1"])):
        t, hc, ipk = aux[name]["t"], aux[name]["h"], aux[name]["ipk"]
        m = (t > t[ipk] - 1.05) & (t < t[ipk] + 0.03)
        ax.plot(t[m] - t[ipk], hc.real[m] / np.abs(hc).max() + (1.2 if name == "GW150914" else -1.2), color=c, lw=0.6,
                label=f"{name}: {res[name]['cycles_35hz_to_peak']:.0f} cycles from 35 Hz in {res[name]['time_35hz_to_peak_s']:.2f} s")
    ax.set(xlabel="time before the amplitude peak (s)", ylabel="template strain (normalised, offset)", yticks=[],
           title="best-fit templates (not whitened)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(ROOT / "docs" / "figures" / "13_two_events.png"); plt.close(fig)


if __name__ == "__main__":
    main()
