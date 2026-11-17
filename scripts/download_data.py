"""Download the GWOSC strain files for GW150914 into data/raw and print their SHA-256.

Usage: python scripts/download_data.py
Files are not committed. See docs/DATA.md for licence and expected hashes.
"""
import pathlib
import sys
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from gwlab.io import sha256  # noqa: E402

BASE = "https://gwosc.org/eventapi/json/GWTC-1-confident/GW150914/v3/"
FILES = [
    "H-H1_GWOSC_4KHZ_R1-1126259447-32.hdf5",
    "L-L1_GWOSC_4KHZ_R1-1126259447-32.hdf5",
    "H-H1_GWOSC_4KHZ_R1-1126257415-4096.hdf5",
    "L-L1_GWOSC_4KHZ_R1-1126257415-4096.hdf5",
]

if __name__ == "__main__":
    raw = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dest = raw / name
        if not dest.exists():
            print("downloading", name, flush=True)
            urllib.request.urlretrieve(BASE + name, dest)
        print(f"{sha256(dest)}  {dest.stat().st_size:>10d}  {name}")
