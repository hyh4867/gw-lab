"""Stochastic template bank for aligned-spin IMRPhenomD (pre-registration stage 5, section 4).

Templates are proposed at random and kept when their match with every template already in the
bank is below the minimal match. Parameters: detector-frame chirp mass, symmetric mass ratio eta,
and one spin chi = chi1 = chi2.
"""
import numpy as np
from scipy import fft as sfft

from . import const, waveforms


class MatchSpace:
    """Normalised, PSD-whitened templates on a coarse frequency grid; match maximised over time and phase."""

    def __init__(self, psd_f, psd, dur=16, fs=const.FS, f_low=const.F_LOW, f_high=const.F_HIGH, pad=2):
        self.n = int(dur * fs)
        self.fs = fs
        self.f_low = f_low
        f = np.fft.rfftfreq(self.n, 1.0 / fs)
        k0, k1 = int(np.ceil(f_low * dur)), int(np.floor(f_high * dur)) + 1
        self.sl = slice(k0, k1)
        self.w = 1.0 / np.sqrt(np.interp(f[self.sl], psd_f, psd))
        nb = k1 - k0
        self.nfft = pad * int(2 ** np.ceil(np.log2(nb)))

    def unit(self, mc, eta, chi):
        m1, m2 = waveforms.masses_from_mc_eta(mc, eta)
        h = waveforms.imrphenomd(m1, m2, self.n, self.fs, self.f_low, chi1=chi, chi2=chi)[self.sl] * self.w
        h = h / np.sqrt(np.sum(h.real**2 + h.imag**2))
        return h.astype(np.complex64)

    def match(self, a, bs):
        """Match of unit template a with each row of bs (2-D)."""
        prod = np.conj(a)[None, :] * bs
        z = sfft.ifft(prod, n=self.nfft, axis=1)
        return np.abs(z).max(axis=1) * self.nfft


def draw(rng, mc_lo, mc_hi, eta_lo, chi_lo, chi_hi):
    """Proposal: uniform in Newtonian chirp time (density ~ Mc^(-8/3)), eta and chi."""
    a, b = mc_hi ** (-5.0 / 3), mc_lo ** (-5.0 / 3)
    mc = rng.uniform(a, b) ** (-3.0 / 5)
    return mc, rng.uniform(eta_lo, 0.25), rng.uniform(chi_lo, chi_hi)


def stochastic_bank(space, rng, region, min_match=0.95, max_reject=400, max_templates=20000,
                    tau_window=0.12, chunk=64, log=None, m2_min=None, mtot_max=None):
    mc_lo, mc_hi, eta_lo, chi_lo, chi_hi = region
    params, units, tau = [], [], []
    rejects, n_prop = 0, 0
    U = np.zeros((0, space.sl.stop - space.sl.start), dtype=np.complex64)
    tau_arr = np.zeros(0)
    while rejects < max_reject and len(params) < max_templates:
        p = draw(rng, *region)
        m1, m2 = waveforms.masses_from_mc_eta(p[0], p[1])
        if (m2_min is not None and m2 < m2_min) or (mtot_max is not None and m1 + m2 > mtot_max):
            continue
        n_prop += 1
        u = space.unit(*p)
        t = waveforms.chirp_time(p[0], space.f_low)
        ok = True
        if len(params):
            near = np.flatnonzero(np.abs(tau_arr - t) <= tau_window * t)
            near = near[np.argsort(np.abs(tau_arr[near] - t))]
            for c in range(0, len(near), chunk):
                if space.match(u, U[near[c:c + chunk]]).max() >= min_match:
                    ok = False
                    break
        if ok:
            params.append(p)
            U = np.vstack([U, u[None, :]])
            tau_arr = np.append(tau_arr, t)
            rejects = 0
            if log and len(params) % 100 == 0:
                log(len(params), n_prop)
        else:
            rejects += 1
    return np.array(params), U, n_prop


def fitting_factors(space, U, bank_params, test_params, tau_window=0.12, chunk=256):
    tau_b = waveforms.chirp_time(bank_params[:, 0], space.f_low)
    out = np.zeros(len(test_params))
    for i, p in enumerate(test_params):
        u = space.unit(*p)
        t = waveforms.chirp_time(p[0], space.f_low)
        near = np.flatnonzero(np.abs(tau_b - t) <= tau_window * t)
        best = 0.0
        for c in range(0, len(near), chunk):
            best = max(best, float(space.match(u, U[near[c:c + chunk]]).max()))
        out[i] = best
    return out
