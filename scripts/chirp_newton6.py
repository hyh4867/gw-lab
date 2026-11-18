"""GW170817: chirp mass from the Newtonian chirp law alone, validated and calibrated with injections
(pre-registration stage 6, section 6; the stage-5 method with a longer segment and a finer mass grid).

Usage: python scripts/chirp_newton6.py
Checkpoint: results/chirp_newton_GW170817.npz (every 25 injections). Writes results/chirp_newton_GW170817.json
and docs/figures/20_chirp_newton_GW170817.png
"""
import json
import pathlib
import sys
import time

import numpy as np
from scipy import fft as sfft

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from gwlab import bns, chirpfilter, const, gate, mf, plotstyle, psd, search, waveforms  # noqa: E402
import ev6common as E  # noqa: E402
from chirp_newton import calibrate  # noqa: E402

FS, N = E.FS, E.N
SEED = 20261117
N_PER = 50
MC_TRUE = [1.0, 1.1, 1.2, 1.3, 1.4]
Q_RANGE = (0.7, 1.0)
STEP, FINE_HALF, FINE_STEP = 2e-4, 4e-4, 2e-5
GRID_EVENT = (0.8, 2.0)
T_INJ = 200.0                  # merger time of an injection inside its 256-s segment
DECIM = 16                     # output rate 256 Hz; the band is 120 Hz wide
LIT = (1.1977, 1.1974, 1.1985)


class Newton:
    """Newtonian chirps on the 30-150 Hz band of an N-sample segment, heterodyned to start at 0 Hz."""

    def __init__(self):
        df = FS / N
        self.k0, k1 = int(np.ceil(chirpfilter.F_LO / df)), int(np.floor(chirpfilter.F_HI / df)) + 1
        self.sl = slice(self.k0, k1)
        f = np.arange(self.k0, k1) * df
        self.amp = f ** (-7.0 / 6)
        self.ph = (3.0 / 128) * (np.pi * const.MTSUN * f) ** (-5.0 / 3)
        self.nd = N // DECIM
        self.nb = k1 - self.k0

    def curve(self, s_f, inv, grid, t_hint, half=0.15):
        """SNR (maximised over time within +-half s of t_hint) for every chirp mass of the grid."""
        df = FS / N
        w = s_f[self.sl] * inv[self.sl] * self.amp
        s2 = 4.0 * df * np.sum(self.amp**2 * inv[self.sl])
        a, b = int((t_hint - half) * FS / DECIM), int((t_hint + half) * FS / DECIM) + 1
        buf = np.zeros(self.nd, dtype=np.complex128)
        out, tpk = np.zeros(len(grid)), np.zeros(len(grid))
        for i, mc in enumerate(grid):
            arg = self.ph * mc ** (-5.0 / 3)
            buf[:self.nb] = w * (np.cos(arg) + 1j * np.sin(arg))
            z = sfft.ifft(buf)[a:b]
            r2 = (z.real**2 + z.imag**2) * ((4.0 * FS / DECIM) ** 2 / s2)
            j = int(np.argmax(r2))
            pj, pv = mf.peak_parabolic(r2, j)
            out[i], tpk[i] = np.sqrt(pv), (a + pj) * DECIM / FS
        return out, tpk

    def measure(self, s_f, inv, lo, hi, t_hint):
        grid = np.exp(np.arange(np.log(lo), np.log(hi) + STEP / 2, STEP))
        c, _ = self.curve(s_f, inv, grid, t_hint)
        i = int(np.argmax(c))
        edge = i == 0 or i == len(grid) - 1
        g2 = grid[i] * np.exp(np.arange(-FINE_HALF, FINE_HALF + FINE_STEP / 2, FINE_STEP))
        c2, t2 = self.curve(s_f, inv, g2, t_hint)
        j = int(np.argmax(c2))
        lmc = np.log(g2[j])
        if 0 < j < len(g2) - 1:
            y0, y1, y2 = c2[j - 1:j + 2]
            den = y0 - 2 * y1 + y2
            if den < 0:
                lmc += 0.5 * (y0 - y2) / den * FINE_STEP
        return {"mc": float(np.exp(lmc)), "snr": float(c2[j]), "t_peak": float(t2[j]), "edge": bool(edge),
                "coarse_grid": grid, "coarse_curve": c, "fine_grid": g2, "fine_curve": c2}


def main():
    ev = json.load(open(ROOT / "results" / "event_gw170817.json"))
    vb = ev["variants"]["b_A_gated"]
    best = vb["fine"]
    i0 = int(round((vb["segment_start_gps"] - E.A_START) * FS))
    t_pk = {"H1": best["i_h"] / FS, "L1": best["i_l"] / FS}          # seconds in the event segment
    snr_ev = {"H1": best["rho_h"], "L1": best["rho_l"]}
    gl = [g["gps"] for g in vb["glitches_found"]["L1"] if abs(g["gps"] - vb["gps_l"]) < 5]
    gate_offset = (vb["gps_l"] - gl[0]) if gl else None               # merger minus glitch (s)
    nw = Newton()
    f = np.fft.rfftfreq(N, 1.0 / FS)
    slb = bns.basis_for().sl
    ck = ROOT / "results" / "chirp_newton_GW170817.npz"
    cols = ["det", "mc_true", "q", "t_start", "mc_raw", "snr_newton", "edge", "snr_opt"]
    n_tot = 2 * len(MC_TRUE) * N_PER
    inj, done = np.full((n_tot, len(cols)), np.nan), np.zeros(n_tot, dtype=bool)
    if ck.exists():
        z = np.load(ck)
        inj, done = z["inj"], z["done"]
    res_ev, curves = {}, {}
    t0 = time.time()
    for d_i, det in enumerate(("H1", "L1")):
        x, _ = E.load("A", det, gated=True)
        st = psd.valid_stretches(x)
        p = search.prepare_segment(x, i0, N, stretches=st)
        r = nw.measure(p["s_f"], p["inv"], *GRID_EVENT, t_pk[det])
        curves[det] = {k: r.pop(k) for k in ("coarse_grid", "coarse_curve", "fine_grid", "fine_curve")}
        r["snr_full_template"] = snr_ev[det]
        r["snr_newton_over_full"] = r["snr"] / snr_ev[det]
        r["t_peak_minus_mf_peak_s"] = r["t_peak"] - t_pk[det]
        res_ev[det] = r
        print(det, "event", r, flush=True)
        rng = np.random.default_rng(SEED + d_i)
        k = d_i * len(MC_TRUE) * N_PER
        nfile = len(x)
        for mc in MC_TRUE:
            for _ in range(N_PER):
                while True:
                    j0 = int(rng.uniform(0, nfile - N))
                    if abs((j0 + T_INJ * FS) - (i0 + t_pk[det] * FS)) > 300 * FS and not np.isnan(x[j0:j0 + N]).any():
                        break
                q = rng.uniform(*Q_RANGE)
                phi = rng.uniform(0, 2 * np.pi)
                if not done[k]:
                    m1, m2 = waveforms.masses_from_mc_q(mc, q)
                    h = waveforms.imrphenomd(m1, m2, N, FS, f_low=bns.F_LOW) * np.exp(1j * phi)
                    h[:slb.start] = 0
                    h[slb.stop:] = 0
                    p0 = search.prepare_segment(x, j0, N, stretches=st)
                    sig = np.sqrt(mf.sigmasq(h, p0["inv"], FS / N, slb))
                    h_t = np.fft.irfft(h * (snr_ev[det] / sig) * np.exp(-2j * np.pi * f * T_INJ), N) * FS
                    seg = p0["seg"] + h_t
                    if det == "L1" and gate_offset is not None:
                        seg = seg * gate.inverse_tukey(N, int(round((T_INJ - gate_offset) * FS)))
                    rr = nw.measure(mf.to_fd(seg, FS), p0["inv"], mc * np.exp(-0.05), mc * np.exp(0.05), T_INJ)
                    inj[k] = [d_i, mc, q, j0 / FS, rr["mc"], rr["snr"], rr["edge"], snr_ev[det]]
                    done[k] = True
                    if done.sum() % 25 == 0:
                        np.savez(ck, inj=inj, done=done, cols=np.array(cols))
                        print(det, int(done.sum()), n_tot, f"{time.time() - t0:.0f}s", flush=True)
                k += 1
        del x
    np.savez(ck, inj=inj, done=done, cols=np.array(cols))

    res = {"event": "GW170817", "band_hz": [chirpfilter.F_LO, chirpfilter.F_HI], "n_per_point": N_PER,
           "grid_step": STEP, "gate_offset_s": gate_offset, "detectors": {}}
    cal = {}
    for d_i, det in enumerate(("H1", "L1")):
        rows = inj[inj[:, 0] == d_i]
        tab, med = [], []
        for mc in MC_TRUE:
            r = rows[rows[:, 1] == mc]
            okr = r[r[:, 6] == 0][:, 4]
            med.append(float(np.median(okr)))
            tab.append({"mc_true": mc, "n": int(len(r)), "n_edge": int((r[:, 6] == 1).sum()), "raw_median": med[-1],
                        "ratio_median": med[-1] / mc, "ratio_p16": float(np.percentile(okr, 16) / mc),
                        "ratio_p84": float(np.percentile(okr, 84) / mc),
                        "snr_newton_over_optimal": float(np.median(r[:, 5]) / snr_ev[det])})
        mono = bool(np.all(np.diff(med) > 0))
        inv = calibrate(MC_TRUE, med) if mono else None
        if mono:
            for row, mc in zip(tab, MC_TRUE):
                c = inv(rows[(rows[:, 1] == mc) & (rows[:, 6] == 0)][:, 4])
                row["cal_median"] = float(np.median(c))
                row["cal_scatter_rel"] = float(0.5 * (np.percentile(c, 84) - np.percentile(c, 16)) / mc)
        cal[det] = inv
        e = dict(res_ev[det])
        e["mc_raw"] = e.pop("mc")
        if mono:
            e["mc_cal"] = float(inv(e["mc_raw"]))
            near = int(np.argmin(np.abs(np.array(MC_TRUE) - e["mc_cal"])))
            e["scatter_rel"], e["scatter_from_mc_true"] = tab[near]["cal_scatter_rel"], MC_TRUE[near]
        res["detectors"][det] = {"calibration": tab, "monotonic": mono, "event": e}
    hd, ld = res["detectors"]["H1"], res["detectors"]["L1"]
    edge_frac = float(np.mean(inj[:, 6] == 1))
    v2 = hd["calibration"][MC_TRUE.index(1.2)].get("cal_scatter_rel")
    res["validation"] = {"V1_monotonic_both": bool(hd["monotonic"] and ld["monotonic"]), "V2_h1_scatter_mc1.2": v2,
                         "V2_ok": bool(v2 is not None and v2 <= 0.15), "V3_edge_fraction": edge_frac, "V3_ok": bool(edge_frac <= 0.05)}
    res["validation"]["all_ok"] = bool(res["validation"]["V1_monotonic_both"] and res["validation"]["V2_ok"] and res["validation"]["V3_ok"])
    raw_h = hd["event"]["mc_raw"]
    res["M1"] = {"value": raw_h, "range": [0.898, 1.497], "reproduced": bool(0.898 <= raw_h <= 1.497)}
    if res["validation"]["V1_monotonic_both"]:
        vals = np.array([hd["event"]["mc_cal"], ld["event"]["mc_cal"]])
        s = np.array([hd["event"]["scatter_rel"], ld["event"]["scatter_rel"]]) * vals
        wgt = 1 / s**2
        mcc, sg = float(np.sum(wgt * vals) / wgt.sum()), float(wgt.sum() ** -0.5)
        lo, hi = mcc - 1.645 * sg, mcc + 1.645 * sg
        res["combined"] = {"mc_cal": mcc, "sigma": sg, "sigma_rel": sg / mcc, "interval90": [lo, hi]}
        res["M2"] = {"interval90": [lo, hi], "literature": [LIT[1], LIT[2]], "reproduced": bool(lo <= LIT[2] and hi >= LIT[1])}
    json.dump(res, open(ROOT / "results" / "chirp_newton_GW170817.json", "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("validation", "M1", "M2", "combined") if k in res}, indent=1), flush=True)

    plt = plotstyle.apply()
    COL = plotstyle.COL
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
    for det in ("H1", "L1"):
        c = curves[det]
        axs[0].plot(c["coarse_grid"], c["coarse_curve"], color=COL[det], lw=0.7, label=det)
        axs[1].plot(c["fine_grid"], c["fine_curve"], color=COL[det], lw=1.2, marker=".", ms=3, label=det)
        rows = inj[inj[:, 0] == (0 if det == "H1" else 1)]
        axs[2].plot(rows[:, 1] + (-0.008 if det == "H1" else 0.008), rows[:, 4] / rows[:, 1], ".", ms=3, color=COL[det], alpha=0.5, label=det)
        axs[2].plot(rows[0, 1] * 0 + res["detectors"][det]["event"].get("mc_cal", np.nan), res["detectors"][det]["event"]["mc_raw"] / res["detectors"][det]["event"].get("mc_cal", np.nan), "s", color=COL[det], ms=7, mec="#0b0b0b")
    axs[0].set(xlabel="chirp mass of the Newtonian chirp (M☉)", ylabel="SNR", title="whole grid 0.8-2.0 M☉", xscale="log")
    plotstyle.plain_log_ticks(axs[0], "x", [0.8, 1.0, 1.2, 1.5, 2.0])
    cmb = res.get("combined")
    axs[1].set(xlabel="chirp mass (M☉)", title="around the maximum (before calibration)")
    axs[1].ticklabel_format(axis="x", useOffset=False)
    axs[1].xaxis.set_major_locator(plt.MaxNLocator(4))
    if cmb:
        axs[1].text(0.03, 0.5, f"after calibration with injections:\n{cmb['mc_cal']:.5f} ± {cmb['sigma']:.5f} M☉\n"
                    f"literature 90 %: {LIT[1]}-{LIT[2]}", transform=axs[1].transAxes, fontsize=8, va="center")
    axs[2].set(xlabel="true chirp mass of the injection (M☉)", ylabel="raw estimate / true", title="injections (squares: the event)")
    for ax in axs[:2]:
        ax.legend()
    fig.suptitle("GW170817: chirp mass from the leading-order chirp law alone (30-150 Hz); the raw value is about 1.3 % low", fontsize=10)
    fig.tight_layout(); fig.savefig(ROOT / "docs" / "figures" / "20_chirp_newton_GW170817.png"); plt.close(fig)


if __name__ == "__main__":
    main()
