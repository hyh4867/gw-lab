import numpy as np

from conftest import coloured_noise, model_psd
from gwlab import chirpfilter, const, mf, psd, waveforms

FS, N = 4096, 32 * 4096


def test_newtonian_chirp_recovers_its_own_chirp_mass():
    """A Newtonian chirp injected without noise is recovered at its chirp mass and with SNR = optimal SNR."""
    f = np.fft.rfftfreq(N, 1.0 / FS)
    inv = psd.inverse_spectrum_truncation(model_psd(f), N, FS)
    grid = chirpfilter.mc_grid()
    bank = chirpfilter.newton_bank(N, FS, grid)
    mc_true, t_in = 27.3, 15.2
    h = np.zeros(len(f), dtype=complex)
    sl = bank["sl"]
    h[sl] = 1e-21 * f[sl] ** (-7.0 / 6) * np.exp(-1j * (3.0 / 128) * (np.pi * mc_true * const.MTSUN * f[sl]) ** (-5.0 / 3))
    h *= np.exp(-2j * np.pi * f * t_in)
    opt = np.sqrt(mf.sigmasq(h, inv, FS / N, sl))
    r = chirpfilter.measure(h, inv, bank, t_in + 0.02)
    assert abs(r["mc"] / mc_true - 1) < 0.002
    assert abs(r["snr"] / opt - 1) < 0.01
    assert abs(r["t_peak"] - t_in) <= 2 * chirpfilter.DECIM / FS
    assert not r["edge"]


def test_newtonian_track_matches_chirp_law():
    """newton_f inverts tau = 5/256 (pi f)^(-8/3) (G Mc/c^3)^(-5/3)."""
    for mc in (9.7, 30.0):
        for f0 in (35.0, 100.0):
            assert abs(chirpfilter.newton_f(waveforms.chirp_time(mc, f0), mc) / f0 - 1) < 1e-12


def test_imr_signal_in_noise_gives_biased_but_stable_estimate(rng):
    """An IMRPhenomD signal at SNR 20: the Newtonian estimate is within 0.9-1.5 of the true chirp mass
    (bias is expected and is calibrated with injections; this only guards against gross failures)."""
    f = np.fft.rfftfreq(N, 1.0 / FS)
    inv = psd.inverse_spectrum_truncation(model_psd(f), N, FS)
    bank = chirpfilter.newton_bank(N, FS, chirpfilter.mc_grid())
    sl = mf.band_slice(N, FS)
    ratios = []
    for _ in range(8):
        h = waveforms.imrphenomd(36.0, 29.0, N, FS)
        h = h * 20.0 / np.sqrt(mf.sigmasq(h, inv, FS / N, sl))
        t_in = 16.0 + rng.uniform(-1, 1)
        x = coloured_noise(N, FS, rng) + np.fft.irfft(h * np.exp(-2j * np.pi * f * t_in), N) * FS
        r = chirpfilter.measure(mf.to_fd(x, FS), inv, bank, t_in)
        ratios.append(r["mc"] / waveforms.chirp_mass(36.0, 29.0))
    assert 0.9 < np.median(ratios) < 1.5
