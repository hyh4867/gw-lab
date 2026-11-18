"""GW170817: rough sky localisation from the matched-filter outputs of three detectors
(pre-registration stage 6, section 8; optional, no verdict). Data B, best TaylorF2 template.

Writes results/sky_gw170817.json and docs/figures/22_sky_gw170817.png
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from gwlab import bns, const, detectors, mf, plotstyle  # noqa: E402
import ev6common as E  # noqa: E402

FS, N = E.FS, E.N
UP = 8                    # interpolate z(t) to 32768 Hz
HALF = 0.25               # seconds of z kept around the L1 peak
N_SKY = 200000
T_SCAN = 0.002            # scan of the L1 arrival time around its SNR peak (s)
SQDEG = 4 * np.pi * (180 / np.pi) ** 2


def fibonacci_sphere(n):
    k = np.arange(n) + 0.5
    dec = np.arcsin(1 - 2 * k / n)
    ra = np.mod(np.pi * (1 + 5 ** 0.5) * k, 2 * np.pi)
    return ra, dec


def antenna_vec(det, lon, lat):
    """F+, Fx (psi = 0) for arrays of Earth-fixed longitude/latitude; same convention as detectors.antenna."""
    gha = -lon
    cg, sg, cd, sd = np.cos(gha), np.sin(gha), np.cos(lat), np.sin(lat)
    X = np.stack([-sg, -cg, np.zeros_like(lon)], axis=-1)
    Y = np.stack([-cg * sd, sg * sd, cd], axis=-1)
    D = detectors.response_tensor(det)
    XD, YD = X @ D, Y @ D
    return np.sum(XD * X, -1) - np.sum(YD * Y, -1), np.sum(XD * Y, -1) + np.sum(YD * X, -1)


def upsample(z, up):
    """Fourier interpolation of a band-limited complex series."""
    n = len(z)
    zf = np.fft.fft(z)
    out = np.zeros(n * up, dtype=complex)
    out[:n // 2] = zf[:n // 2]
    out[-(n - n // 2):] = zf[n // 2:]
    return np.fft.ifft(out) * up


def coherent(y, sig, lon, lat, dets, t_gps, i_ref):
    """max over the L1 arrival time of the coherent SNR^2 for each sky point. y: dict det -> upsampled z/sigma
    (index 0 = sample i_ref - HALF*FS of the segment); sig: dict of sigma."""
    nhat = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)
    tau = {d: -(nhat @ detectors.DETECTORS[d]["loc"]) / const.C for d in dets}
    A = np.stack([np.stack([sig[d] * f for f in antenna_vec(d, lon, lat)], axis=-1) for d in dets], axis=1)  # (n, ndet, 2)
    G = np.einsum("nda,ndb->nab", A, A)
    Gi = np.linalg.inv(G + 1e-12 * np.eye(2))
    fs_up = FS * UP
    c0 = int(HALF * fs_up)
    offs = np.arange(-int(T_SCAN * fs_up), int(T_SCAN * fs_up) + 1)
    best = np.zeros(len(lon))
    for o in offs:
        yv = np.stack([y[d][np.clip(c0 + o + np.rint((tau[d] - tau["L1"]) * fs_up).astype(int), 0, len(y[d]) - 1)]
                       for d in dets], axis=1)                                   # (n, ndet) complex
        v = np.einsum("nda,nd->na", A, yv)
        r2 = np.real(np.einsum("na,nab,nb->n", np.conj(v), Gi, v))
        best = np.maximum(best, r2)
    return best


def area90(logp, value_at=None):
    p = np.exp(logp - logp.max())
    p /= p.sum()
    o = np.argsort(p)[::-1]
    k = int(np.searchsorted(np.cumsum(p[o]), 0.9)) + 1
    thr = p[o[k - 1]]
    return k * SQDEG / len(p), p, thr


def main():
    ev = json.load(open(ROOT / "results" / "event_gw170817.json"))
    v = ev["variants"]["c_B"]
    best = v["fine"]
    i0 = int(round((v["segment_start_gps"] - E.B_START) * FS))
    basis = bns.basis_for()
    h = basis.full(best["m1"], best["m2"])
    i_ref = int(round(best["i_l"]))
    y, sig, peak = {}, {}, {}
    for det in E.DETS:
        x, _ = E.load("B", det)
        p = E.prep(x, "B", i0)
        del x
        rho, z, s = mf.snr_series(p["s_f"], h, p["inv"], N, FS, basis.sl)
        a, b = i_ref - int(HALF * FS), i_ref + int(HALF * FS)
        y[det] = upsample(z[a:b] / s, UP)
        sig[det] = s
        peak[det] = float(np.abs(y[det]).max())
    t_gps = v["gps_l"]
    ra, dec = fibonacci_sphere(N_SKY)
    lon, lat = detectors.lonlat_from_radec(ra, dec, t_gps)
    res = {"data": "B", "template": {"m1": best["m1"], "m2": best["m2"]}, "n_sky_points": N_SKY,
           "pixel_deg2": SQDEG / N_SKY, "sigma_1mpc": sig, "counterpart_ra_dec_deg": [float(np.rad2deg(E.RA)), float(np.rad2deg(E.DEC))]}
    lon0, lat0 = detectors.lonlat_from_radec(np.array([E.RA]), np.array([E.DEC]), t_gps)
    maps = {}
    for label, dets in (("HLV", ("H1", "L1", "V1")), ("HL", ("H1", "L1"))):
        r2 = coherent(y, sig, lon, lat, dets, t_gps, i_ref)
        area, p, thr = area90(r2 / 2)
        r2c = float(coherent(y, sig, lon0, lat0, dets, t_gps, i_ref)[0])
        pc = np.exp(r2c / 2 - (r2 / 2).max()) / np.exp(r2 / 2 - (r2 / 2).max()).sum()
        k = int(np.argmax(r2))
        sep = np.rad2deg(np.arccos(np.clip(np.sin(dec[k]) * np.sin(E.DEC) + np.cos(dec[k]) * np.cos(E.DEC) * np.cos(ra[k] - E.RA), -1, 1)))
        res[label] = {"area90_deg2": float(area), "max_coherent_snr2": float(r2.max()), "coherent_snr2_at_counterpart": r2c,
                      "counterpart_inside_90": bool(pc >= thr), "max_ra_dec_deg": [float(np.rad2deg(ra[k])), float(np.rad2deg(dec[k]))],
                      "separation_of_maximum_from_counterpart_deg": float(sep),
                      "area50_deg2": float((np.searchsorted(np.cumsum(np.sort(p)[::-1]), 0.5) + 1) * SQDEG / N_SKY)}
        maps[label] = (p, thr)
        print(label, res[label], flush=True)
    res["incoherent_network_snr2_hlv_at_peaks"] = float(sum(peak[d] ** 2 for d in E.DETS))
    json.dump(res, open(ROOT / "results" / "sky_gw170817.json", "w"), indent=1)

    plt = plotstyle.apply()
    fig = plt.figure(figsize=(11, 4.6))
    for i, (label, title) in enumerate((("HL", "H1 + L1 only"), ("HLV", "H1 + L1 + Virgo"))):
        ax = fig.add_subplot(1, 2, i + 1, projection="mollweide")
        p, thr = maps[label]
        m = p >= thr
        x = np.pi - np.mod(ra, 2 * np.pi)          # RA increases to the left
        ax.scatter(x[m], dec[m], s=1.2, color=plotstyle.COL["H1"], rasterized=True, linewidths=0)
        ax.plot(np.pi - E.RA, E.DEC, marker="*", ms=11, color=plotstyle.COL["L1"], mec="#0b0b0b", mew=0.5, ls="")
        ax.set_title(f"{title}: 90 % region {res[label]['area90_deg2']:.0f} deg²", fontsize=10)
        ax.set_xticklabels([])
        ax.grid(True, lw=0.3)
    fig.suptitle("GW170817: sky region from matched-filter times, amplitudes and phases (star: optical counterpart)", fontsize=10)
    fig.tight_layout(); fig.savefig(ROOT / "docs" / "figures" / "22_sky_gw170817.png"); plt.close(fig)


if __name__ == "__main__":
    main()
