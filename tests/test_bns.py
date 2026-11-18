import numpy as np

from conftest import coloured_noise, fd_to_td, model_psd
from gwlab import bns, f2fast, gate, mf, psd, search, waveforms

FS = 4096


def test_fast_taylorf2_equals_direct():
    n = 64 * FS
    b = f2fast.F2Basis(n, FS, 30.0, 1024.0)
    for m1, m2 in [(1.48, 1.27), (2.0, 1.0), (1.0, 1.0)]:
        ref = waveforms.taylorf2(m1, m2, n, FS, 30.0)[b.sl]
        got = b.template(m1, m2)
        assert np.abs(got - ref).max() < 1e-9 * np.abs(ref).max()


def test_inverse_tukey_shape():
    w = gate.inverse_tukey(4 * FS, 2 * FS, FS, 0.1, 0.5)
    assert w[2 * FS] == 0 and w[int(1.95 * FS)] == 0 and w[int(2.09 * FS)] == 0
    assert abs(w[int(2.35 * FS)] - 0.5) < 1e-3          # half-way up the 0.5 s roll-off
    assert w[int(2.7 * FS)] == 1 and w[0] == 1


def test_gate_finds_and_removes_glitch(rng):
    n = 192 * FS
    x = coloured_noise(n, FS, rng)
    ig = 100 * FS + 123
    g = x.copy()
    g[ig - 8:ig + 8] += 3e-18 * np.hanning(16) * (-1) ** np.arange(16)
    c, peak = gate.find_glitches(g, FS)
    assert len(c) == 1 and abs(c[0] - ig) <= 8 and peak[0] > 100
    assert len(gate.find_glitches(x, FS)[0]) == 0
    out = gate.apply_gates(g, c, FS)
    assert np.all(out[ig - 400:ig + 400] == 0)
    far = np.abs(np.arange(n) - ig) > 0.7 * FS
    assert np.array_equal(out[far], g[far])


def test_bns_bank_filter_matches_direct_filter_and_finds_injection(rng):
    lay = search.Layout(64, 32, 28, 60, 4)
    n = lay.seg_dur * FS
    basis = bns.basis_for(lay, FS)
    f = np.fft.rfftfreq(n, 1.0 / FS)
    inv = psd.inverse_spectrum_truncation(model_psd(f), n, FS)
    m1, m2 = 2.0, 1.9                                  # 22 s from 30 Hz: fits before the valid region
    h = basis.full(m1, m2)
    df = FS / n
    sig = np.sqrt(mf.sigmasq(h, inv, df, basis.sl))
    t_inj = 45.3
    x = coloured_noise(n, FS, rng) + fd_to_td(h * (12.0 / sig), n, FS, t_inj)
    s_f = mf.to_fd(x, FS)
    p = bns.reduce_prep({"s_f": s_f, "inv": inv}, basis)
    masses = np.array([[1.9, 1.9], [m1, m2], [2.0, 2.0]])
    best2, arg = bns.filter_bank(basis, masses, [p], lay, FS)
    rho_direct = mf.snr_series(s_f, h, inv, n, FS, basis.sl)[0][lay.valid_start * FS:lay.valid_end * FS:lay.decim]
    i = int(np.argmax(best2[0]))
    assert arg[0][i] == 1
    assert abs(np.sqrt(best2[0][i]) - rho_direct[i]) < 2e-3 * rho_direct[i]
    assert abs(lay.valid_start + i * lay.decim / FS - t_inj) < 2.0 / 1024
    assert 10.5 < np.sqrt(best2[0][i]) < 13.5
    trig = bns.triggers(basis, masses, p, best2[0], arg[0], 0, lay, FS)
    k = int(np.argmax(trig["rho"]))
    assert trig["tmpl"][k] == 1 and abs(trig["t"][k] - t_inj) < 2.0 / 1024
    assert trig["chi2r"][k] < 2.0
    c2, dof = mf.chisq(s_f, h, inv, n, int(round(trig["t"][k] * FS)), FS, sl=basis.sl)
    assert abs(trig["chi2r"][k] - c2 / dof) < 0.02 * max(1.0, c2 / dof)
