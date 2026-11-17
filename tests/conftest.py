import numpy as np
import pytest

from gwlab import const


def model_psd(f):
    """A smooth aLIGO-like analytic PSD (not a fit to any data); used only to colour test noise."""
    f = np.maximum(f, 5.0)
    x = f / 150.0
    return 1e-46 * (x ** -8 * 0.02 + 0.6 / x**2 + 1.0 + 0.4 * x**2)


def coloured_noise(n, fs, rng, psd_func=model_psd):
    f = np.fft.rfftfreq(n, 1.0 / fs)
    s = psd_func(f)
    amp = np.sqrt(s * fs * n / 4.0)
    xf = amp * (rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f)))
    xf[0] = 0
    return np.fft.irfft(xf, n)


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


def fd_to_td(h_f, n, fs, shift_s=0.0):
    """Time series of a continuous-normalised FD signal, delayed by shift_s seconds."""
    f = np.fft.rfftfreq(n, 1.0 / fs)
    return np.fft.irfft(h_f * np.exp(-2j * np.pi * f * shift_s), n) * fs
