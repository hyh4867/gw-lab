"""Shared helpers for the stage-5 scripts: load the analysis segment of an event in both detectors."""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gwlab import const, io, psd, search, waveforms  # noqa: E402
from run_search5 import strain_file  # noqa: E402

FS = const.FS
EVENTS = {
    "GW150914": {"file": 1126257415, "t0": const.T0_GW150914, "layout": search.LAYOUT_BBH, "z": 0.09},
    "GW151226": {"file": 1135134303, "t0": 1135136350.6, "layout": search.LAYOUT_LOWMASS, "z": 0.09},
}


def segment_for(lay, t_file):
    """Index of the segment whose valid region contains t_file (seconds from file start)."""
    j = int((t_file - lay.valid_start) // lay.stride)
    return j, j * lay.stride * FS


def load_strain(name, det):
    return io.read_strain(strain_file(det, EVENTS[name]["file"]))["strain"]


def preps(name, t_file, extra=None):
    """Prepared segment (FFT + PSD) of both detectors around t_file. Returns (preps, i0, n)."""
    lay = EVENTS[name]["layout"]
    n = lay.seg_dur * FS
    j, i0 = segment_for(lay, t_file)
    out = {}
    for det in ("H1", "L1"):
        x = load_strain(name, det)
        out[det] = search.prepare_segment(x, i0, n, stretches=psd.valid_stretches(x),
                                          extra=None if extra is None else extra[det])
        del x
    return out, i0, n


def best(name):
    """Best-fit template and peak samples of an event from the result files.
    Returns dict: m1, m2, chi, i_h, i_l (samples in the analysis segment), seg_start_s, rho_h, rho_l."""
    if name == "GW150914":
        e = json.load(open(ROOT / "results" / "event.json"))
        b = e["bank"]["imr"]["fine"]
        return {"m1": b["m1"], "m2": b["m2"], "chi": 0.0, "i_h": b["i_h"], "i_l": b["i_l"],
                "seg_start_s": e["segment"]["start_s"], "rho_h": b["rho_h"], "rho_l": b["rho_l"], "rho_net": b["rho_net"]}
    e = json.load(open(ROOT / "results" / "event_gw151226.json"))
    b = e["fine"]
    return {"m1": b["m1"], "m2": b["m2"], "chi": b["chi"], "i_h": b["i_h"], "i_l": b["i_l"],
            "seg_start_s": e["segment"]["start_s"], "rho_h": b["rho_h"], "rho_l": b["rho_l"], "rho_net": b["rho_net"]}


def template(b, n):
    return waveforms.imrphenomd(b["m1"], b["m2"], n, FS, chi1=b["chi"], chi2=b["chi"])
