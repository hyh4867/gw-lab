"""Chirp mass from f(t) read off zero crossings (pre-registered recipe, section 4)."""
import numpy as np

from . import const


def zero_crossings(t, x):
    s = np.signbit(x)
    i = np.flatnonzero(s[1:] != s[:-1])
    return t[i] - x[i] * (t[i + 1] - t[i]) / (x[i + 1] - x[i])


def f_of_t(t, x, t_pk, back=0.20, f_min=30.0, f_max=150.0, skip_high=False):
    """Frequencies from successive zero crossings in [t_pk - back, t_pk], walking back from t_pk
    and stopping at the first value outside [f_min, f_max]. Returns (t_mid, f) in time order.

    skip_high=False is the pre-registered recipe. skip_high=True (change log entry 1) first skips
    the points with f > f_max next to t_pk (the merger), then collects as before."""
    m = (t >= t_pk - back) & (t <= t_pk)
    zc = zero_crossings(t[m], x[m])
    tm = 0.5 * (zc[1:] + zc[:-1])
    f = 1.0 / (2.0 * np.diff(zc))
    keep = []
    start = len(f) - 1
    if skip_high:
        while start >= 0 and f[start] > f_max:
            start -= 1
    for i in range(start, -1, -1):
        if f_min <= f[i] <= f_max:
            keep.append(i)
        else:
            break
    keep = keep[::-1]
    return tm[keep], f[keep]


def fit_chirp_mass(tm, f):
    """Least-squares line f^(-8/3) = a (t_c - t); returns dict with Mc (Msun), its error, t_c."""
    if len(f) < 4:
        return None
    y = f ** (-8.0 / 3)
    A = np.vstack([tm - tm.mean(), np.ones_like(tm)]).T
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    slope, icpt = coef
    r = y - A @ coef
    dof = len(y) - 2
    s2 = (r @ r) / dof
    cov = s2 * np.linalg.inv(A.T @ A)
    if slope >= 0:
        return {"mc": np.nan, "mc_err": np.nan, "slope": slope, "n": len(f)}
    a = -slope
    k = (8 * np.pi) ** (8.0 / 3) / 5.0
    mc_s = (a / k) ** (3.0 / 5)
    mc = mc_s / const.MTSUN
    mc_err = mc * 0.6 * np.sqrt(cov[0, 0]) / a
    tc = tm.mean() + icpt / a
    ss_tot = np.sum((y - y.mean()) ** 2)
    return {"mc": mc, "mc_err": mc_err, "slope": slope, "tc": tc, "n": len(f),
            "r2": 1 - (r @ r) / ss_tot}
