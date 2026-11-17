"""Read GWOSC hdf5 strain files."""
import hashlib

import h5py
import numpy as np


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_strain(path):
    """Return dict with strain (float64), gps start, sample interval, DQ mask (1 Hz) and bit names."""
    with h5py.File(path, "r") as f:
        ds = f["strain/Strain"]
        strain = ds[...].astype(np.float64)
        dt = float(ds.attrs["Xspacing"])
        start = float(ds.attrs["Xstart"])
        dq = f["quality/simple/DQmask"][...]
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in f["quality/simple/DQShortnames"][...]]
        meta = {k: (v[()].decode() if isinstance(v[()], bytes) else v[()]) for k, v in f["meta"].items()}
    return {"strain": strain, "start": start, "dt": dt, "dq": dq, "dq_names": names, "meta": meta}
