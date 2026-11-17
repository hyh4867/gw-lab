"""Injection campaign (pre-registration section 7). Checkpoints to results/injections.npz every 20 injections."""
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from gwlab import coinc, const, detectors, io, mf, search, waveforms  # noqa: E402
from run_search import segment_starts, strain_path  # noqa: E402

SEED = 20261115
DISTANCES = [250, 500, 750, 1000, 1250, 1500, 2000, 2500, 3000]
N_PER = 100
M1, M2 = 38.9, 31.6
T_EVENT = const.T0_GW150914 - const.FILE_START  # used only to keep injections away from the event
FS, N = const.FS, const.SEG_DUR * const.FS
OUT = ROOT / "results" / "injections.npz"
COLS = ["dist", "seg", "t_geo", "lon", "lat", "psi", "cosi", "phi", "t_h", "t_l", "opt_h", "opt_l",
        "found_coinc", "rho_h", "rho_l", "new_h", "new_l", "chi_h", "chi_l", "newsnr_c", "dt_h", "tmpl_h"]


def draw(rng, starts):
    ok = [j for j, i0 in enumerate(starts) if (i0 / FS > T_EVENT + 32) or (i0 / FS + const.SEG_DUR < T_EVENT - 32)]
    rows = []
    for d in DISTANCES:
        for _ in range(N_PER):
            j = int(rng.choice(ok))
            t_geo = starts[j] / FS + rng.uniform(const.VALID_START + 0.5, const.VALID_END - 0.5)
            rows.append([d, j, t_geo, rng.uniform(0, 2 * np.pi), np.arcsin(rng.uniform(-1, 1)), rng.uniform(0, np.pi),
                         rng.uniform(-1, 1), rng.uniform(0, 2 * np.pi)])
    return np.array(rows)


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    strain = {d: io.read_strain(strain_path(d))["strain"] for d in ("H1", "L1")}
    starts = segment_starts(len(strain["H1"]))
    params = draw(rng, starts)
    order = rng.permutation(len(params))  # interleave distances so partial runs are usable
    res = np.full((len(params), len(COLS)), np.nan)
    res[:, :8] = params
    done = np.zeros(len(params), dtype=bool)
    if OUT.exists():
        z = np.load(OUT)
        res, done = z["res"], z["done"]
    bank = search.build_bank(search.coarse_masses(), N, "imr")
    f = np.fft.rfftfreq(N, 1.0 / FS)
    t0 = time.time()
    for c, q in enumerate(order):
        if done[q]:
            continue
        dist, j, t_geo, lon, lat, psi, cosi, phi = params[q]
        j = int(j)
        i0 = starts[j]
        hp, hc = waveforms.lal_fd(M1, M2, N, FS, dist_mpc=dist, incl=np.arccos(cosi), phi=phi)
        trig, tarr, opt = {}, {}, {}
        for det in ("H1", "L1"):
            fp, fc = detectors.antenna(det, lon, lat, psi)
            tarr[det] = t_geo + detectors.delay_from_geocentre(det, lon, lat)
            hdet = (fp * hp + fc * hc) * np.exp(-2j * np.pi * f * (tarr[det] - i0 / FS))
            extra = np.fft.irfft(hdet, N) * FS
            prep = search.prepare_segment(strain[det], i0, N, extra=extra)
            opt[det] = np.sqrt(mf.sigmasq(hdet, prep["inv"], FS / N, bank.sl))
            trig[det] = search.find_triggers(prep, bank, i0)
        row = res[q]
        row[8:12] = tarr["H1"], tarr["L1"], opt["H1"], opt["L1"]
        row[12] = 0
        th, tl = trig["H1"], trig["L1"]
        th, tl = th[th["newsnr"] >= const.NEWSNR_MIN], tl[tl["newsnr"] >= const.NEWSNR_MIN]
        if len(th) and len(tl):
            a, b = coinc.zero_lag(th["t"], tl["t"])
            near = np.abs(th["t"][a] - tarr["H1"]) <= 0.05
            a, b = a[near], b[near]
            if len(a):
                s = coinc.combined(th["newsnr"][a], tl["newsnr"][b])
                o = int(np.argmax(s))
                h, l = th[a[o]], tl[b[o]]
                row[12:] = 1, h["rho"], l["rho"], h["newsnr"], l["newsnr"], h["chi2r"], l["chi2r"], s[o], h["t"] - tarr["H1"], h["tmpl"]
        done[q] = True
        if c % 20 == 19 or c == len(order) - 1:
            np.savez(OUT, res=res, done=done, cols=np.array(COLS))
            print(c + 1, len(order), f"{time.time() - t0:.0f}s", flush=True)
    print("done", flush=True)
