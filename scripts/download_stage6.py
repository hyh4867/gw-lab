"""Download the stage-6 data: GW170817 strain (H1, L1, V1), two extra O2 file pairs, GWTC-1 posterior samples.

Usage: python scripts/download_stage6.py   -> data/raw/*.hdf5, data/meta/stage6_files.json (sizes and SHA-256)
Files already present are not downloaded again. Run only after docs/PREREGISTRATION_STAGE6.md is committed.
"""
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gwlab import io  # noqa: E402

RAW = ROOT / "data" / "raw"
EV_A = "https://gwosc.org/eventapi/json/GWTC-1-confident/GW170817/v3/"
EV_B = "https://gwosc.org/eventapi/json/O1_O2-Preliminary/GW170817/v2/"
DCC = "https://dcc.ligo.org/public/0157/P1800370/005/"


def targets():
    out = []
    for det in ("H1", "L1", "V1"):
        name = f"{det[0]}-{det}_GWOSC_4KHZ_R1-1187006835-4096.hdf5"
        out.append(("data A: GWTC-1 release (glitch present in L1)", name, EV_A + name))
        name = f"{det[0]}-{det}_LOSC_CLN_4_V1-1187007040-2048.hdf5"
        out.append(("data B: discovery-paper release (L1 glitch subtracted by LIGO)", name, EV_B + name))
    meta = json.load(open(ROOT / "data" / "meta" / "background_files_GW170817.json"))["GW170817"]
    for c in meta["chosen"]:
        for det, f in c["files"].items():
            out.append(("background GW170817", f["url"].rsplit("/", 1)[1], f["url"]))
    out.append(("GWTC-1 posterior samples", "GW170817_GWTC-1.hdf5", DCC + "GW170817_GWTC-1.hdf5"))
    return out


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    rows = []
    for use, name, url in targets():
        path = RAW / name
        if not path.exists():
            tmp = path.with_suffix(".part")
            subprocess.run(["curl", "-sSL", "--fail", "--retry", "5", "-C", "-", "-o", str(tmp), url], check=True)
            tmp.rename(path)
        rows.append({"use": use, "file": name, "url": url, "bytes": path.stat().st_size, "sha256": io.sha256(path)})
        print(rows[-1]["bytes"], rows[-1]["sha256"], name, flush=True)
    json.dump(rows, open(ROOT / "data" / "meta" / "stage6_files.json", "w"), indent=1)
    print("done", flush=True)
