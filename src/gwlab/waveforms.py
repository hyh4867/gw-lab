"""Frequency-domain templates: hand-written TaylorF2 and IMRPhenomD via lalsimulation."""
import warnings

import numpy as np

from . import const


def chirp_mass(m1, m2):
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


def masses_from_mc_q(mc, q):
    """q = m2/m1 <= 1."""
    m1 = mc * (1 + q) ** 0.2 / q ** 0.6
    return m1, q * m1


def f_isco(mtot):
    """GW frequency at the Schwarzschild ISCO for total mass mtot (solar masses)."""
    return 1.0 / (6 ** 1.5 * np.pi * mtot * const.MTSUN)


def taylorf2_phase(f, m1, m2, tc=0.0, phic=0.0):
    """3.5PN stationary-phase phase psi(f), Eq. (3.18) of Buonanno et al. 2009 (arXiv:0907.0700).

    Non-spinning. v = (pi M f)^(1/3), nu = m1 m2 / M^2, v_lso = 6^(-1/2).
    """
    M = (m1 + m2) * const.MTSUN
    nu = m1 * m2 / (m1 + m2) ** 2
    v = (np.pi * M * f) ** (1.0 / 3)
    vlso = 6 ** -0.5
    g = const.EULER_GAMMA
    pi = np.pi
    series = (
        1
        + (20.0 / 9) * (743.0 / 336 + 11.0 / 4 * nu) * v**2
        - 16 * pi * v**3
        + 10 * (3058673.0 / 1016064 + 5429.0 / 1008 * nu + 617.0 / 144 * nu**2) * v**4
        + pi * (38645.0 / 756 - 65.0 / 9 * nu) * (1 + 3 * np.log(v / vlso)) * v**5
        + (
            11583231236531.0 / 4694215680
            - 640.0 / 3 * pi**2
            - 6848.0 * g / 21
            - 6848.0 / 21 * np.log(4 * v)
            + (-15737765635.0 / 3048192 + 2255.0 * pi**2 / 12) * nu
            + 76055.0 / 1728 * nu**2
            - 127825.0 / 1296 * nu**3
        ) * v**6
        + pi * (77096675.0 / 254016 + 378515.0 / 1512 * nu - 74045.0 / 756 * nu**2) * v**7
    )
    return 2 * pi * f * tc - phic - pi / 4 + 3.0 / (128 * nu * v**5) * series


def taylorf2(m1, m2, n, fs=const.FS, f_low=const.F_LOW, f_cut=None, dist_mpc=1.0):
    """TaylorF2 h+ for a face-on binary on the rfft grid of an n-sample segment.

    Newtonian amplitude sqrt(5/24) pi^(-2/3) (c/D) (G Mc/c^3)^(5/6) f^(-7/6), cut at f_ISCO.
    The sign of the phase follows the numpy/LAL transform convention h~(f) = int h(t) exp(-2 pi i f t) dt,
    so h~ = A f^(-7/6) exp(-i psi).
    """
    f = np.fft.rfftfreq(n, 1.0 / fs)
    if f_cut is None:
        f_cut = f_isco(m1 + m2)
    h = np.zeros(len(f), dtype=np.complex128)
    m = (f >= f_low) & (f <= f_cut)
    mc = chirp_mass(m1, m2) * const.MTSUN
    amp = np.sqrt(5.0 / 24) * np.pi ** (-2.0 / 3) * const.C / (dist_mpc * const.MPC) * mc ** (5.0 / 6)
    h[m] = amp * f[m] ** (-7.0 / 6) * np.exp(-1j * taylorf2_phase(f[m], m1, m2))
    return h


def _lal():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import lal
        import lalsimulation as ls
    return lal, ls


def lal_fd(m1, m2, n, fs=const.FS, f_low=const.F_LOW, dist_mpc=1.0, incl=0.0, phi=0.0,
           approximant="IMRPhenomD", chi1=0.0, chi2=0.0):
    """(h+, hx) from lalsimulation on the rfft grid of an n-sample segment.

    chi1, chi2 are the dimensionless spin components along the orbital angular momentum."""
    lal, ls = _lal()
    df = fs / n
    hp, hc = ls.SimInspiralChooseFDWaveform(
        m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, 0, 0, chi1, 0, 0, chi2, dist_mpc * 1e6 * lal.PC_SI,
        incl, phi, 0, 0, 0, df, f_low, fs / 2, f_low, None, getattr(ls, approximant))
    nf = n // 2 + 1
    out = []
    for h in (hp, hc):
        d = h.data.data
        a = np.zeros(nf, dtype=np.complex128)
        m = min(nf, len(d))
        a[:m] = d[:m]
        out.append(a)
    return out[0], out[1]


def imrphenomd(m1, m2, n, fs=const.FS, f_low=const.F_LOW, dist_mpc=1.0, chi1=0.0, chi2=0.0):
    """Face-on IMRPhenomD h+ (aligned spins chi1, chi2)."""
    return lal_fd(m1, m2, n, fs, f_low, dist_mpc, chi1=chi1, chi2=chi2)[0]


def masses_from_mc_eta(mc, eta):
    mtot = mc / eta ** 0.6
    d = np.sqrt(max(0.0, 1 - 4 * eta))
    return 0.5 * mtot * (1 + d), 0.5 * mtot * (1 - d)


def chirp_time(mc, f_low=const.F_LOW):
    """Newtonian time to coalescence from f_low (seconds) for chirp mass mc (solar masses)."""
    return 5.0 / 256 * (np.pi * f_low) ** (-8.0 / 3) * (mc * const.MTSUN) ** (-5.0 / 3)


def chi_eff(m1, m2, chi1, chi2):
    return (m1 * chi1 + m2 * chi2) / (m1 + m2)
