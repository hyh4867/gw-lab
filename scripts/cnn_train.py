"""Train the small CNN detector (pre-registration stage 5, section 6).

Training data: noise from the eight extra GW150914-era files (never the event file); signals from the
stage-3 injection population (IMRPhenomD, detector-frame 38.9 + 31.6 Msun), optimal single-detector SNR
uniform in 4-20, merger at 0.6-0.9 s of the 1-s window.
Writes data/cache/cnn_train.npz (git-ignored), results/cnn_model.pt (git-ignored), results/cnn_train.json
"""
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from gwlab import cnn, const, io, mf, psd, search, waveforms  # noqa: E402
from run_search5 import strain_file  # noqa: E402

SEED = 20261116
FS, N = const.FS, const.SEG_DUR * const.FS
M1, M2 = 38.9, 31.6
PER_SEG = 8            # noise windows and signal windows taken from one segment
N_SEG = 2500           # -> 20,000 noise + 20,000 signal windows
CACHE = ROOT / "data" / "cache" / "cnn_train.npz"


def make_data():
    rng = np.random.default_rng(SEED)
    meta = json.load(open(ROOT / "data" / "meta" / "background_files.json"))["GW150914"]
    starts = [c["gps_start"] for c in meta["chosen"]]
    lay = search.LAYOUT_BBH
    seg_starts = lay.starts(4096 * FS)
    combos = [(s, d) for s in starts for d in ("H1", "L1")]
    per_file = N_SEG // len(combos) + 1
    h0 = waveforms.imrphenomd(M1, M2, N, FS)
    f = np.fft.rfftfreq(N, 1.0 / FS)
    sl = mf.band_slice(N, FS)
    X, y, snr = [], [], []
    t0 = time.time()
    for ci, (start, det) in enumerate(combos):
        strain = io.read_strain(strain_file(det, start))["strain"]
        for j in rng.choice(len(seg_starts), size=per_file, replace=False):
            i0 = seg_starts[j]
            prep = search.prepare_segment(strain, i0, N)
            if prep is None:
                continue
            x = cnn.preprocess(prep["seg"], prep["inv"])
            sig0 = np.sqrt(mf.sigmasq(h0, prep["inv"], FS / N, sl))
            lo, hi = lay.valid_start * cnn.FS_NET, lay.valid_end * cnn.FS_NET - cnn.WIN
            for a in rng.integers(lo, hi, PER_SEG):
                X.append(x[a:a + cnn.WIN]); y.append(0); snr.append(0.0)
            for a in rng.integers(lo, hi, PER_SEG):
                rho = rng.uniform(4, 20)
                t_m = a / cnn.FS_NET + rng.uniform(0.6, 0.9)
                h = h0 * (rho / sig0) * np.exp(1j * rng.uniform(0, 2 * np.pi)) * np.exp(-2j * np.pi * f * t_m)
                s = cnn.preprocess_fd_signal(h, prep["inv"], N)
                X.append(x[a:a + cnn.WIN] + s[a:a + cnn.WIN]); y.append(1); snr.append(rho)
        del strain
        print("data", ci + 1, len(combos), len(X), f"{time.time() - t0:.0f}s", flush=True)
    X, y, snr = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(snr, dtype=np.float32)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, X=X, y=y, snr=snr)
    return X, y, snr


def main():
    import torch
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    if CACHE.exists():
        z = np.load(CACHE)
        X, y, snr = z["X"], z["y"], z["snr"]
    else:
        X, y, snr = make_data()
    rng = np.random.default_rng(SEED + 1)
    perm = rng.permutation(len(X))
    nval = len(X) // 10
    va, tr = perm[:nval], perm[nval:]
    model = cnn.build_model()
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = torch.nn.BCEWithLogitsLoss()
    Xt, yt = torch.from_numpy(X[:, None, :]), torch.from_numpy(y)
    hist = []
    t0 = time.time()
    for ep in range(15):
        model.train()
        order = rng.permutation(tr)
        tot = 0.0
        for b in range(0, len(order), 128):
            idx = torch.from_numpy(order[b:b + 128])
            opt.zero_grad()
            loss = lossf(model(Xt[idx]).squeeze(1), yt[idx])
            loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            out = torch.cat([model(Xt[torch.from_numpy(va[b:b + 512])]).squeeze(1) for b in range(0, nval, 512)]).numpy()
        yv, sv = y[va], snr[va]
        vloss = float(np.mean(np.logaddexp(0, out) - yv * out))
        acc = float(np.mean((out > 0) == (yv > 0.5)))
        thr = np.percentile(out[yv == 0], 99)  # 1 % false-positive rate per window
        tpr = {f"snr_{a}_{b}": float(np.mean(out[(yv == 1) & (sv >= a) & (sv < b)] > thr)) for a, b in ((4, 8), (8, 12), (12, 20))}
        hist.append({"epoch": ep + 1, "train_loss": tot / len(tr), "val_loss": vloss, "val_acc": acc, "tpr_at_fpr_0.01": tpr})
        print(json.dumps(hist[-1]), f"{time.time() - t0:.0f}s", flush=True)
    torch.save(model.state_dict(), ROOT / "results" / "cnn_model.pt")
    json.dump({"seed": SEED, "n_train": int(len(tr)), "n_val": int(nval), "n_parameters": int(nparam),
               "torch": torch.__version__, "history": hist, "seconds": time.time() - t0},
              open(ROOT / "results" / "cnn_train.json", "w"), indent=1)


if __name__ == "__main__":
    main()
