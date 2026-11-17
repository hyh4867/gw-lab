import numpy as np

from gwlab import const, mf, psd, waveforms
from conftest import coloured_noise, fd_to_td, model_psd

FS = 4096
N = 16 * FS


def test_inner_product_white_noise_sine():
    # x(t) = A cos(2 pi f0 t) over T seconds, white PSD S0: <x|x> = (2/S0) int x^2 dt = A^2 T / S0
    n, fs, f0, A, S0 = 8 * 4096, 4096, 100.0, 3.0, 2.5
    t = np.arange(n) / fs
    xf = np.fft.rfft(A * np.cos(2 * np.pi * f0 * t)) / fs
    inv = np.full(len(xf), 1.0 / S0)
    val = mf.inner(xf, xf, inv, fs / n)
    assert abs(val / (A**2 * (n / fs) / S0) - 1) < 1e-6


def test_inner_product_orthogonal_quadratures():
    n, fs = 8 * 4096, 4096
    t = np.arange(n) / fs
    a = np.fft.rfft(np.cos(2 * np.pi * 100 * t)) / fs
    b = np.fft.rfft(np.sin(2 * np.pi * 100 * t)) / fs
    inv = np.ones(len(a))
    assert abs(mf.inner(a, b, inv, fs / n)) < 1e-6 * mf.inner(a, a, inv, fs / n)


def test_welch_median_recovers_known_psd(rng):
    x = coloured_noise(256 * FS, FS, rng)
    f, p = psd.welch_median(x, FS)
    m = (f > 30) & (f < 1000)
    ratio = p[m] / model_psd(f[m])
    # 127 half-overlapping segments: per-bin scatter ~10 %, mean over thousands of bins much tighter
    assert abs(ratio.mean() - 1) < 0.03


def test_noise_only_snr_squared_mean_is_two(rng):
    h = waveforms.taylorf2(30, 30, N, FS)
    f = np.fft.rfftfreq(N, 1 / FS)
    inv = 1.0 / model_psd(f)
    vals = []
    for _ in range(20):
        s_f = mf.to_fd(coloured_noise(N, FS, rng), FS)
        rho, _, _ = mf.snr_series(s_f, h, inv, N, FS)
        vals.append(np.mean(rho[2 * FS:-2 * FS] ** 2))
    assert abs(np.mean(vals) - 2.0) < 0.1


def test_snr_and_time_recovery(rng):
    """Inject a template of known optimal SNR; recovered SNR^2 averages rho_opt^2 + 2."""
    f = np.fft.rfftfreq(N, 1 / FS)
    inv = 1.0 / model_psd(f)
    sl = mf.band_slice(N, FS)
    h = waveforms.taylorf2(30, 30, N, FS)
    sig1 = np.sqrt(mf.sigmasq(h, inv, FS / N, sl))
    rho_opt = 12.0
    t_inj = 9.0 + 100.3 / FS  # deliberately off-sample
    sig_td = fd_to_td(h * rho_opt / sig1, N, FS, t_inj)
    rec, terr = [], []
    for _ in range(200):
        s_f = mf.to_fd(coloured_noise(N, FS, rng) + sig_td, FS)
        rho, _, _ = mf.snr_series(s_f, h, inv, N, FS, sl)
        i = int(np.argmax(rho[4 * FS:-4 * FS])) + 4 * FS
        ip, val = mf.peak_parabolic(rho, i)
        rec.append(rho[i])
        terr.append(ip / FS - t_inj)
    rec = np.array(rec)
    expect = np.sqrt(rho_opt**2 + 2)
    sem = rec.std(ddof=1) / np.sqrt(len(rec))
    assert abs(rec.mean() - expect) < 3 * sem + 0.02 * expect
    # in noise the peak time scatters by ~1/(2 pi rho sigma_f); this template stops at f_ISCO = 73 Hz,
    # so the scatter is of order 1 ms. Check unbiasedness and a loose bound.
    terr = np.array(terr)
    assert abs(terr.mean()) < 3 * terr.std(ddof=1) / np.sqrt(len(terr)) + 0.5 / FS
    assert terr.std() < 3e-3
    # noise-free: exact recovery of the optimal SNR and of the time to within one sample
    rho0, _, _ = mf.snr_series(mf.to_fd(sig_td, FS), h, inv, N, FS, sl)
    assert abs(rho0.max() / rho_opt - 1) < 2e-3
    ip0, _ = mf.peak_parabolic(rho0, int(np.argmax(rho0)))
    assert abs(ip0 / FS - t_inj) <= 1.0 / FS


def test_chisq_signal_vs_glitch(rng):
    f = np.fft.rfftfreq(N, 1 / FS)
    inv = 1.0 / model_psd(f)
    sl = mf.band_slice(N, FS)
    h = waveforms.taylorf2(30, 30, N, FS)
    sig1 = np.sqrt(mf.sigmasq(h, inv, FS / N, sl))
    sig_td = fd_to_td(h * 15.0 / sig1, N, FS, 8.0)
    c = []
    for _ in range(100):
        s_f = mf.to_fd(coloured_noise(N, FS, rng) + sig_td, FS)
        rho, _, _ = mf.snr_series(s_f, h, inv, N, FS, sl)
        i = int(np.argmax(rho[6 * FS:10 * FS])) + 6 * FS
        c2, dof = mf.chisq(s_f, h, inv, N, i, FS, sl=sl)
        c.append(c2 / dof)
    assert dof == 30
    assert abs(np.mean(c) - 1.0) < 0.15
    # sine-Gaussian glitch seen by a broadband (IMRPhenomD) template. A TaylorF2 template cut at
    # f_ISCO = 73 Hz is too narrow-band for the test to bite (chi2_r ~ 1.7 at rho = 17).
    h = waveforms.imrphenomd(30, 30, N, FS)
    t = np.arange(N) / FS
    g = 1e-20 * np.exp(-((t - 8.0) / 0.01) ** 2) * np.sin(2 * np.pi * 60 * (t - 8.0))
    s_f = mf.to_fd(coloured_noise(N, FS, rng) + g, FS)
    rho, _, _ = mf.snr_series(s_f, h, inv, N, FS, sl)
    i = int(np.argmax(rho[6 * FS:10 * FS])) + 6 * FS
    c2, dof = mf.chisq(s_f, h, inv, N, i, FS, sl=sl)
    assert rho[i] > 8
    assert c2 / dof > 3
    assert mf.newsnr(np.array([rho[i]]), np.array([c2 / dof]))[0] < rho[i]


def test_chisq_parts_sum_to_filter_output(rng):
    f = np.fft.rfftfreq(N, 1 / FS)
    inv = 1.0 / model_psd(f)
    sl = mf.band_slice(N, FS)
    h = waveforms.taylorf2(25, 20, N, FS)
    s_f = mf.to_fd(coloured_noise(N, FS, rng), FS)
    z = mf.complex_filter(s_f, h, inv, N, FS, sl)
    i = 5 * FS + 17
    k = np.arange(sl.start, sl.stop)
    direct = 4 * (FS / N) * np.sum(s_f[sl] * np.conj(h[sl]) * inv[sl] * np.exp(2j * np.pi * k * i / N))
    assert abs(direct - z[i]) < 1e-9 * abs(z[i])


def test_newsnr_formula():
    out = mf.newsnr(np.array([10.0, 10.0]), np.array([0.5, 2.0]))
    assert out[0] == 10.0
    assert abs(out[1] - 10.0 / ((1 + 8) / 2) ** (1 / 6)) < 1e-12


def test_inverse_spectrum_truncation_length():
    f = np.fft.rfftfreq(N, 1 / FS)
    inv = psd.inverse_spectrum_truncation(model_psd(f), N, FS)
    q = np.fft.irfft(inv, N)  # response of 1/S: autocorrelation of the truncated filter
    k = int(2 * const.INVSPEC_HALF * FS)
    assert np.max(np.abs(q[k + 1:N - k])) < 1e-9 * np.max(np.abs(q))
    m = (f > 30) & (f < 1000)
    assert np.max(np.abs(inv[m] * model_psd(f[m]) - 1)) < 0.05


def test_peak_parabolic_does_not_extrapolate():
    """At a sample that is not a local maximum the value is returned unchanged (no extrapolation)."""
    y = np.array([1.0, 2.0, 3.2, 4.3, 5.0, 4.0])
    assert mf.peak_parabolic(y, 2) == (2.0, 3.2)
    pos, val = mf.peak_parabolic(y, 4)
    assert 3.5 < pos < 4.5 and 5.0 <= val < 5.2
