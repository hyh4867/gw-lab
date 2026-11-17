import numpy as np

from gwlab import chirpfit, coinc, const, mf, qtransform, search, waveforms
from conftest import coloured_noise, fd_to_td, model_psd


def test_slide_coincidences_match_brute_force():
    rng = np.random.default_rng(3)
    span, step, w = 100.0, 0.1, 0.015
    th = np.sort(rng.uniform(0, span, 40))
    tl = np.sort(rng.uniform(0, span, 35))
    i, j, k = coinc.slide_coincidences(th, tl, span, step, w)
    got = set(zip(i.tolist(), j.tolist(), k.tolist()))
    want = set()
    for kk in range(1, coinc.n_slides(span, step) + 1):
        shifted = np.mod(tl + kk * step, span)
        d = th[:, None] - shifted[None, :]
        d = (d + span / 2) % span - span / 2  # circular difference
        a, b = np.nonzero(np.abs(d) <= w)
        want |= {(int(x), int(y), kk) for x, y in zip(a, b)}
    assert got == want
    assert coinc.n_slides(span, step) == 999


def test_slide_hand_made():
    th = np.array([10.0, 50.0])
    tl = np.array([9.7, 49.995])  # first pair coincides at k=3; second only at zero lag
    i, j, k = coinc.slide_coincidences(th, tl, 100.0)
    trip = set(zip(i.tolist(), j.tolist(), k.tolist()))
    assert (0, 0, 3) in trip
    assert all(not (a == 1 and b == 1) for a, b, _ in trip)
    zi, zj = coinc.zero_lag(th, tl)
    assert list(zip(zi, zj)) == [(1, 1)]


def test_cluster_keeps_loudest():
    t = np.zeros(4, dtype=search.TRIG_DTYPE)
    t["t"] = [1.0, 1.2, 1.4, 5.0]
    t["rho"] = [6, 9, 7, 6]
    out = search.cluster(t)
    assert list(out["t"]) == [1.2, 5.0]


def test_bank_search_finds_injection():
    rng = np.random.default_rng(11)
    fs, n = 4096, 32 * 4096
    strain = coloured_noise(64 * fs, fs, rng)
    masses = np.array([(30.0, 30.0), (40.0, 30.0), (60.0, 50.0), (24.0, 20.0)])
    bank = search.build_bank(masses, n)
    f = np.fft.rfftfreq(n, 1 / fs)
    h = waveforms.imrphenomd(40, 30, n, fs)
    sig1 = np.sqrt(mf.sigmasq(h, 1.0 / model_psd(f), fs / n, bank.sl))
    extra = fd_to_td(h * 14.0 / sig1, n, fs, 15.25)
    prep = search.prepare_segment(strain, 16 * fs, n, extra=extra)
    trig = search.find_triggers(prep, bank, 16 * fs)
    assert len(trig) >= 1
    best = trig[np.argmax(trig["rho"])]
    assert abs(best["t"] - (16 + 15.25)) < 2.0 / fs
    assert best["tmpl"] == 1
    assert 11 < best["rho"] < 17
    assert best["chi2r"] < 2.5
    # all other triggers are noise-level
    assert np.all(np.delete(trig["rho"], np.argmax(trig["rho"])) < 7)


def test_chirp_fit_recovers_newtonian_chirp():
    fs = 4096
    mc_true = 30.0
    t = np.arange(0, 1.0, 1 / fs)
    tc = 0.8
    k = (8 * np.pi) ** (8 / 3) / 5 * (mc_true * const.MTSUN) ** (5 / 3)
    tau = np.clip(tc - t, 1e-4, None)
    f = (k * tau) ** (-3 / 8)
    phase = 2 * np.pi * np.cumsum(f) / fs
    x = np.where(t < tc, np.cos(phase), 0.0)
    tm, ff = chirpfit.f_of_t(t, x, tc - 0.004)
    fit = chirpfit.fit_chirp_mass(tm, ff)
    assert fit["n"] >= 4
    assert abs(fit["mc"] / mc_true - 1) < 0.05


def test_qtransform_white_noise_normalisation():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(8 * 4096)
    e = qtransform.qtransform(x, 4096, np.array([50.0, 100.0, 300.0]))
    assert np.all(np.abs(e.mean(axis=1) - 1) < 0.15)


def test_ring_mapping_and_all_pairs_rate():
    """Two live intervals laid end to end; all-pairs rate equals the slide expectation for uniform triggers."""
    t = np.array([5.0, 12.0, 25.0, 31.0])
    ring, keep, length = coinc.to_ring(t, [[10.0, 20.0], [30.0, 40.0]])
    assert length == 20.0
    assert keep.tolist() == [False, True, False, True]
    assert ring[1] == 2.0 and ring[3] == 11.0
    assert coinc.merge_intervals([(8, 24), (24, 40), (56, 72)]) == [[8, 40], [56, 72]]
    rng = np.random.default_rng(3)
    span = 4000.0
    th, tl = np.sort(rng.uniform(0, span, 300)), np.sort(rng.uniform(0, span, 300))
    i, j, k = coinc.slide_coincidences(th, tl, span)
    n_slides = coinc.n_slides(span)
    rate_slides = len(i) / (n_slides * span)
    rate_pairs = coinc.all_pairs_rate(len(th) * len(tl), span)
    assert abs(rate_slides / rate_pairs - 1) < 0.05


def test_no_reuse_keeps_each_trigger_once():
    stat = np.array([9.0, 8.5, 8.0, 7.0])
    i = np.array([0, 0, 1, 2])
    j = np.array([0, 1, 0, 2])
    keep = coinc.no_reuse(stat, i, j)
    assert sorted(keep.tolist()) == [0, 3]
