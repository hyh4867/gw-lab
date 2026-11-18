"""Fast TaylorF2 templates for long (binary-neutron-star) searches (pre-registration stage 6).

A 256-s segment has 2.5e5 frequency bins in band, so a bank of thousands of templates cannot be stored
(2 MB each). The 3.5PN phase of waveforms.taylorf2_phase is a linear combination of nine fixed functions
of frequency with mass-dependent coefficients, so each template is rebuilt from a stored basis when needed.
The result is the same function as waveforms.taylorf2 (tests/test_bns.py), not a new approximation.
"""
import numpy as np

from . import const, waveforms

_POW = (-5.0 / 3, -1.0, -2.0 / 3, -1.0 / 3)


class F2Basis:
    def __init__(self, n, fs=const.FS, f_low=30.0, f_high=const.F_HIGH):
        df = fs / n
        self.n, self.fs = n, fs
        self.sl = slice(int(np.ceil(f_low / df)), int(np.floor(f_high / df)) + 1)
        f = np.arange(self.sl.start, self.sl.stop) * df
        self.f = f
        lf = np.log(f)
        c = np.cbrt(f)
        self.b = np.stack([f ** p for p in _POW] + [lf, c, c * lf, c * c])   # (8, nband); the constant is separate
        self.amp = f ** (-7.0 / 6)

    @staticmethod
    def coeffs(m1, m2):
        """Coefficients of [f^-5/3, f^-1, f^-2/3, f^-1/3, ln f, f^1/3, f^1/3 ln f, f^2/3] and the constant."""
        M = (m1 + m2) * const.MTSUN
        nu = m1 * m2 / (m1 + m2) ** 2
        k = (np.pi * M) ** (1.0 / 3)          # v = k f^(1/3)
        pi, g, vlso = np.pi, const.EULER_GAMMA, 6 ** -0.5
        a2 = (20.0 / 9) * (743.0 / 336 + 11.0 / 4 * nu)
        a3 = -16 * pi
        a4 = 10 * (3058673.0 / 1016064 + 5429.0 / 1008 * nu + 617.0 / 144 * nu**2)
        a5 = pi * (38645.0 / 756 - 65.0 / 9 * nu)
        a6 = (11583231236531.0 / 4694215680 - 640.0 / 3 * pi**2 - 6848.0 * g / 21
              + (-15737765635.0 / 3048192 + 2255.0 * pi**2 / 12) * nu + 76055.0 / 1728 * nu**2
              - 127825.0 / 1296 * nu**3)
        a6l = -6848.0 / 21
        a7 = pi * (77096675.0 / 254016 + 378515.0 / 1512 * nu - 74045.0 / 756 * nu**2)
        pre = 3.0 / (128 * nu)
        c = pre * np.array([k**-5, a2 * k**-3, a3 * k**-2, a4 / k, a5, (a6 + a6l * np.log(4 * k)) * k,
                            a6l * k / 3, a7 * k * k])
        const_term = pre * a5 * (1 + 3 * np.log(k / vlso)) - pi / 4
        return c, const_term

    def template(self, m1, m2, dist_mpc=1.0, f_cut=None):
        """Band-limited h+ (face-on), identical to waveforms.taylorf2 on self.sl."""
        c, c0 = self.coeffs(m1, m2)
        psi = c @ self.b
        psi += c0
        if f_cut is None:
            f_cut = waveforms.f_isco(m1 + m2)
        mc = waveforms.chirp_mass(m1, m2) * const.MTSUN
        a0 = np.sqrt(5.0 / 24) * np.pi ** (-2.0 / 3) * const.C / (dist_mpc * const.MPC) * mc ** (5.0 / 6)
        h = (a0 * self.amp) * (np.cos(psi) - 1j * np.sin(psi))
        if f_cut < self.f[-1]:
            h[self.f > f_cut] = 0.0
        return h

    def full(self, m1, m2, **kw):
        out = np.zeros(self.n // 2 + 1, dtype=np.complex128)
        out[self.sl] = self.template(m1, m2, **kw)
        return out
