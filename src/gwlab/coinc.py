"""Coincidence and time-slide background."""
import numpy as np

from . import const


def zero_lag(th, tl, window=const.COINC_WINDOW):
    """Index pairs (i, j) with |th[i] - tl[j]| <= window."""
    d = th[:, None] - tl[None, :]
    i, j = np.nonzero(np.abs(d) <= window)
    return i, j


def slide_coincidences(th, tl, span, step=const.SLIDE_STEP, window=const.COINC_WINDOW):
    """Coincidences when L1 times are circularly shifted by k*step, k = 1 .. span/step - 1.

    Times must lie in [0, span). Returns (i, j, k). Each pair can match at most one k
    because step > 2*window.
    """
    ns = int(round(span / step))
    d = np.mod(th[:, None] - tl[None, :], span)
    k = np.rint(d / step).astype(np.int64)
    resid = d - k * step
    k = np.mod(k, ns)
    ok = (np.abs(resid) <= window) & (k != 0)
    i, j = np.nonzero(ok)
    return i, j, k[i, j]


def n_slides(span, step=const.SLIDE_STEP):
    return int(round(span / step)) - 1


def combined(a, b):
    return np.sqrt(a**2 + b**2)


# ---- stage 5: several files joined into one ring; estimators that expose trigger reuse ----

def merge_intervals(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1] + 1e-9:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def to_ring(t, intervals, offset=0.0):
    """Map times to ring coordinates: the live intervals are laid end to end starting at `offset`.

    Returns (ring_time, keep_mask, length). Times outside the intervals are dropped."""
    ring = np.full(len(t), np.nan)
    pos = offset
    for a, b in intervals:
        m = (t >= a) & (t < b)
        ring[m] = pos + (t[m] - a)
        pos += b - a
    return ring, ~np.isnan(ring), pos - offset


def all_pairs_stat(sh, sl):
    """Combined statistic of every (H1, L1) trigger pair, flattened."""
    return np.sqrt(sh[:, None] ** 2 + sl[None, :] ** 2).ravel()


def all_pairs_rate(n_pairs_above, span, window=const.COINC_WINDOW):
    """Expected accidental-coincidence rate (per second) from the number of pairs above a threshold."""
    return n_pairs_above * 2.0 * window / span**2


def no_reuse(stat, i, j):
    """Greedy one-to-one selection: walk the coincidences from loudest to quietest and keep one only if
    neither of its single-detector triggers has been used. Returns indices of the kept coincidences."""
    used_h, used_l, keep = set(), set(), []
    for o in np.argsort(stat)[::-1]:
        if i[o] in used_h or j[o] in used_l:
            continue
        used_h.add(i[o]); used_l.add(j[o]); keep.append(o)
    return np.array(keep, dtype=int)
