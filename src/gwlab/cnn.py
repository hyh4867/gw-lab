"""Small 1-D CNN detector and its preprocessing (pre-registration stage 5, section 6). Needs torch."""
import numpy as np
from scipy.signal.windows import tukey

from . import const, psd

FS_NET = 2048
WIN = FS_NET            # 1-s input window
STRIDE = FS_NET // 16   # 1/16 s
MERGER_POS = 0.75       # nominal position of the merger inside a window (s)
BAND = (20.0, 500.0)
NORM = np.sqrt((BAND[1] - BAND[0]) / (const.FS / 2))  # std of band-passed unit white noise


def preprocess(seg, inv, fs=const.FS):
    """Whiten a segment with its truncated inverse PSD, band-pass 20-500 Hz, keep every second sample,
    scale to about unit variance. Returns float32 at 2048 Hz."""
    n = len(seg)
    xw = psd.whiten(seg * tukey(n, 2.0 * const.TUKEY_TAPER * fs / n), inv, fs)
    return (psd.bandpass(xw, *BAND, fs=fs)[::fs // FS_NET] / NORM).astype(np.float32)


def preprocess_fd_signal(h_f, inv, n, fs=const.FS):
    """Same chain for a noise-free signal given in the frequency domain (continuous normalisation)."""
    sw = np.fft.irfft(h_f * fs * np.sqrt(inv * 2.0 / fs), n)
    return (psd.bandpass(sw, *BAND, fs=fs)[::fs // FS_NET] / NORM).astype(np.float32)


def build_model():
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv1d(1, 16, 16, stride=2), nn.ReLU(),
        nn.Conv1d(16, 32, 16, stride=2), nn.ReLU(),
        nn.Conv1d(32, 64, 8, stride=2), nn.ReLU(),
        nn.Conv1d(64, 64, 8, stride=2), nn.ReLU(),
        nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(64, 1))


def score_series(model, x, t_first, t_last):
    """Logit for every window whose nominal merger time (start + MERGER_POS) lies in [t_first, t_last),
    times in seconds from the start of the 2048-Hz series x. Returns (times, logits)."""
    import torch
    a0 = int(round((t_first - MERGER_POS) * FS_NET))
    starts = np.arange(a0, int(round((t_last - MERGER_POS) * FS_NET)), STRIDE)
    starts = starts[(starts >= 0) & (starts + WIN <= len(x))]
    idx = starts[:, None] + np.arange(WIN)[None, :]
    with torch.no_grad():
        out = model(torch.from_numpy(x[idx][:, None, :])).squeeze(1).numpy()
    return starts / FS_NET + MERGER_POS, out


def local_maxima(t, s, half=0.5):
    """Indices of samples that are the maximum within +-half seconds."""
    from scipy.ndimage import maximum_filter1d
    k = int(round(half / (t[1] - t[0])))
    return np.flatnonzero(s == maximum_filter1d(s, size=2 * k + 1, mode="constant", cval=-np.inf))
