"""Luminosity distance <-> redshift for flat LambdaCDM (Planck 2015 values used by GWTC-1)."""
import numpy as np

H0 = 67.9          # km / s / Mpc
OMEGA_M = 0.3065
C_KMS = 299792.458


def dl_from_z(z, n=2001):
    """Luminosity distance (Mpc) by trapezoidal integration of 1/E(z)."""
    z = np.atleast_1d(np.asarray(z, float))
    out = np.zeros(len(z))
    for i, zi in enumerate(z):
        x = np.linspace(0.0, zi, n)
        e = np.sqrt(OMEGA_M * (1 + x) ** 3 + 1 - OMEGA_M)
        out[i] = (1 + zi) * C_KMS / H0 * np.trapezoid(1.0 / e, x)
    return out


_ZG = np.linspace(0.0, 1.0, 2001)
_DG = None


def z_from_dl(dl):
    """Redshift for luminosity distance(s) in Mpc (interpolation of a table up to z = 1)."""
    global _DG
    if _DG is None:
        _DG = dl_from_z(_ZG)
    return np.interp(dl, _DG, _ZG)
