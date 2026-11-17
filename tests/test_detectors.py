import warnings

import numpy as np

from gwlab import detectors


def _lal():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import lal
    return lal


def test_antenna_matches_lal():
    lal = _lal()
    rng = np.random.default_rng(5)
    for det, key in (("H1", lal.LHO_4K_DETECTOR), ("L1", lal.LLO_4K_DETECTOR)):
        d = lal.CachedDetectors[key]
        assert np.allclose(d.location, detectors.DETECTORS[det]["loc"], atol=1e-3)
        assert np.allclose(d.response, detectors.response_tensor(det), atol=1e-6)
        for _ in range(50):
            lon, lat, psi = rng.uniform(0, 2 * np.pi), np.arcsin(rng.uniform(-1, 1)), rng.uniform(0, np.pi)
            fp, fc = detectors.antenna(det, lon, lat, psi)
            rp, rc = lal.ComputeDetAMResponse(d.response, lon, lat, psi, 0.0)
            assert abs(fp - rp) < 1e-5 and abs(fc - rc) < 1e-5
            dt = detectors.delay_from_geocentre(det, lon, lat)
            ref = lal.TimeDelayFromEarthCenter(d.location, lon, lat, lal.LIGOTimeGPS(0))
            # gmst at GPS 0 is not zero: compare only the magnitude bound here
            assert abs(dt) <= 6.4e6 / 299792458.0
    # explicit delay check with gmst folded in
    gps = lal.LIGOTimeGPS(1126259462)
    gmst = lal.GreenwichMeanSiderealTime(gps)
    for det, key in (("H1", lal.LHO_4K_DETECTOR), ("L1", lal.LLO_4K_DETECTOR)):
        d = lal.CachedDetectors[key]
        ra, dec = 1.3, -0.4
        ref = lal.TimeDelayFromEarthCenter(d.location, ra, dec, gps)
        mine = detectors.delay_from_geocentre(det, ra - gmst, dec)
        assert abs(ref - mine) < 1e-7


def test_max_delay_between_sites_is_10ms():
    b = detectors.DETECTORS["H1"]["loc"] - detectors.DETECTORS["L1"]["loc"]
    assert abs(np.linalg.norm(b) / 299792458.0 - 0.0100) < 2e-4


def test_antenna_power_average():
    """Sky- and polarisation-average of F+^2 + Fx^2 is 2/5 for an L-shaped interferometer."""
    rng = np.random.default_rng(6)
    v = []
    for _ in range(20000):
        fp, fc = detectors.antenna("H1", rng.uniform(0, 2 * np.pi), np.arcsin(rng.uniform(-1, 1)), rng.uniform(0, np.pi))
        v.append(fp**2 + fc**2)
    assert abs(np.mean(v) - 0.4) < 0.01
