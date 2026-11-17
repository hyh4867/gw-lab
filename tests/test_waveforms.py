import numpy as np

from gwlab import const, mf, waveforms
from conftest import model_psd

FS = 4096
N = 32 * FS


def test_chirp_mass_roundtrip():
    m1, m2 = waveforms.masses_from_mc_q(30.0, 0.8)
    assert abs(waveforms.chirp_mass(m1, m2) - 30.0) < 1e-12
    assert abs(m2 / m1 - 0.8) < 1e-12


def test_f_isco_value():
    # 4400 Hz / (M / Msun) is the textbook number
    assert abs(waveforms.f_isco(1.0) - 4397) < 5


def test_newtonian_chirp_time():
    """dpsi/df / 2pi at leading order is minus the time to coalescence 5/256 (pi f)^(-8/3) Mc^(-5/3)."""
    m1, m2 = 1.4, 1.4  # low mass: PN corrections small at low frequency
    f = np.array([20.0, 20.01])
    psi = waveforms.taylorf2_phase(f, m1, m2)
    t_f = np.diff(psi)[0] / np.diff(f)[0] / (2 * np.pi)
    mc = waveforms.chirp_mass(m1, m2) * const.MTSUN
    tau = 5.0 / 256 * (np.pi * 20.0) ** (-8.0 / 3) * mc ** (-5.0 / 3)
    assert abs(-t_f / tau - 1) < 0.02  # 1PN correction is ~1 % here


def _phase_residual(m1, m2):
    f = np.fft.rfftfreq(N, 1 / FS)
    mine = waveforms.taylorf2(m1, m2, N, FS)
    ref, _ = waveforms.lal_fd(m1, m2, N, FS, approximant="TaylorF2")
    m = (f >= 20) & (f <= waveforms.f_isco(m1 + m2)) & (np.abs(ref) > 0)
    dphi = np.unwrap(np.angle(mine[m] * np.conj(ref[m])))
    A = np.vstack([f[m], np.ones(m.sum())]).T
    coef, *_ = np.linalg.lstsq(A, dphi, rcond=None)
    amp_ratio = np.abs(mine[m]) / np.abs(ref[m])
    return np.max(np.abs(dphi - A @ coef)), amp_ratio


def test_taylorf2_matches_lalsimulation():
    """Independent check of the hand-written phase against LAL's TaylorF2 (non-spinning).

    A constant and a term linear in f (phi_c, t_c) are removed.
    """
    for m1, m2 in [(36.0, 29.0), (20.0, 20.0), (10.0, 5.0), (1.4, 1.4)]:
        res, amp = _phase_residual(m1, m2)
        assert res < 0.01, (m1, m2, res)
        assert np.max(np.abs(amp - 1)) < 1e-3, (m1, m2)


def test_imrphenomd_inspiral_amplitude_is_newtonian_at_low_f():
    f = np.fft.rfftfreq(N, 1 / FS)
    a = np.abs(waveforms.imrphenomd(10, 10, N, FS))
    b = np.abs(waveforms.taylorf2(10, 10, N, FS))
    k = int(25 * N / FS)
    assert abs(a[k] / b[k] - 1) < 0.05


def test_taylorf2_overlap_with_imrphenomd_inspiral():
    """Below ISCO, for a low-mass system, the two families should nearly agree (match > 0.95)."""
    m1, m2 = 6.0, 5.0
    f = np.fft.rfftfreq(N, 1 / FS)
    inv = 1.0 / model_psd(f)
    fcut = 0.5 * waveforms.f_isco(m1 + m2)
    sl = mf.band_slice(N, FS, 20.0, fcut)
    a = waveforms.taylorf2(m1, m2, N, FS)
    b = waveforms.imrphenomd(m1, m2, N, FS)
    z = mf.complex_filter(a, b, inv, N, FS, sl)
    df = FS / N
    match = np.abs(z).max() / np.sqrt(mf.sigmasq(a, inv, df, sl) * mf.sigmasq(b, inv, df, sl))
    assert match > 0.95


def test_cosmology_distance_redshift():
    """Low-z limit d_L ~ c z / H0 (1 + (1 - q0) z / 2) and a round trip; z(440 Mpc) ~ 0.09 as in GWTC-1."""
    from gwlab import cosmo
    z = 0.01
    q0 = 1.5 * cosmo.OMEGA_M - 1
    approx = cosmo.C_KMS * z / cosmo.H0 * (1 + 0.5 * (1 - q0) * z)
    assert abs(cosmo.dl_from_z(z)[0] / approx - 1) < 1e-3
    assert abs(cosmo.z_from_dl(cosmo.dl_from_z(0.2)[0]) - 0.2) < 1e-5
    assert abs(cosmo.z_from_dl(440.0) - 0.093) < 0.003


def test_aligned_spin_changes_phase_but_zero_spin_is_default():
    n, fs = 8 * 4096, 4096
    a = waveforms.imrphenomd(15.0, 8.0, n, fs)
    b = waveforms.imrphenomd(15.0, 8.0, n, fs, chi1=0.0, chi2=0.0)
    c = waveforms.imrphenomd(15.0, 8.0, n, fs, chi1=0.3, chi2=0.3)
    assert np.array_equal(a, b)
    k = slice(8 * 30, 8 * 300)
    overlap = abs(np.vdot(a[k], c[k])) / np.sqrt(np.vdot(a[k], a[k]).real * np.vdot(c[k], c[k]).real)
    assert overlap < 0.9
    m1, m2 = waveforms.masses_from_mc_eta(waveforms.chirp_mass(15.0, 8.0), 15.0 * 8.0 / 23.0**2)
    assert abs(m1 - 15.0) < 1e-9 and abs(m2 - 8.0) < 1e-9
    assert abs(waveforms.chi_eff(15.0, 8.0, 0.3, -0.1) - (15 * 0.3 - 8 * 0.1) / 23) < 1e-12
