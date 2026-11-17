"""Antenna patterns and arrival-time delays for H1 and L1 from detector tensors.

Vertex positions (m) and arm unit vectors are the values published in LAL (LALDetectors.h),
Earth-fixed frame. Tests compare the results with lal.ComputeDetAMResponse.
"""
import numpy as np

from . import const

DETECTORS = {
    "H1": {
        "loc": np.array([-2.16141492636e6, -3.83469517889e6, 4.60035022664e6]),
        "x": np.array([-0.22389266154, 0.79983062746, 0.55690487831]),
        "y": np.array([-0.91397818574, 0.02609403989, -0.40492342125]),
    },
    "L1": {
        "loc": np.array([-7.42760447238e4, -5.49628371971e6, 3.22425701744e6]),
        "x": np.array([-0.95457412153, -0.14158077340, -0.26218911324]),
        "y": np.array([0.29774156894, -0.48791033647, -0.82054461286]),
    },
}


def response_tensor(det):
    d = DETECTORS[det]
    return 0.5 * (np.outer(d["x"], d["x"]) - np.outer(d["y"], d["y"]))


def antenna(det, lon, lat, psi):
    """F+, Fx for a source at Earth-fixed longitude/latitude (rad) and polarisation angle psi.

    Same convention as XLALComputeDetAMResponse with Greenwich hour angle gha = -lon.
    """
    gha = -lon
    cg, sg = np.cos(gha), np.sin(gha)
    cd, sd = np.cos(lat), np.sin(lat)
    cp, sp = np.cos(psi), np.sin(psi)
    X = np.array([-cp * sg - sp * cg * sd, -cp * cg + sp * sg * sd, sp * cd])
    Y = np.array([sp * sg - cp * cg * sd, sp * cg + cp * sg * sd, cp * cd])
    D = response_tensor(det)
    fp = X @ D @ X - Y @ D @ Y
    fc = X @ D @ Y + Y @ D @ X
    return fp, fc


def delay_from_geocentre(det, lon, lat):
    """Arrival time at the detector minus arrival time at the Earth's centre (s)."""
    nhat = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    return -float(nhat @ DETECTORS[det]["loc"]) / const.C
