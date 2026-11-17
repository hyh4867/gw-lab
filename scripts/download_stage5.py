"""Download the stage-5 data (GW151226 event files, extra O1 files for the background, GWTC-1 posterior samples).

Usage: python scripts/download_stage5.py   -> data/raw/*.hdf5, data/meta/stage5_files.json (sizes and SHA-256)
Files already present with the expected size are not downloaded again.
"""
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gwlab import io  # noqa: E402

RAW = ROOT / "data" / "raw"
EV = "https://gwosc.org/eventapi/json/GWTC-1-confident/GW151226/v2/"
BULK = "https://gwosc.org/archive/data/O1/1126170624/"
DCC = "https://dcc.ligo.org/public/0157/P1800370/005/"


def targets():
    out = []
    for det in ("H1", "L1"):
        for start, dur in ((1135134303, 4096), (1135136335, 32)):
            name = f"{det[0]}-{det}_GWOSC_4KHZ_R1-{start}-{dur}.hdf5"
            out.append(("GW151226 event", name, EV + name))
        name = f"{det[0]}-{det}_LOSC_4_V1-1126256640-4096.hdf5"
        out.append(("overlap check (not used for background)", name, BULK + name))
    meta = json.load(open(ROOT / "data" / "meta" / "background_files.json"))
    for ev, d in meta.items():
        for c in d["chosen"]:
            for det, f in c["files"].items():
                out.append((f"background {ev}", f["url"].rsplit("/", 1)[1], f["url"]))
    for ev in ("GW150914", "GW151226"):
        out.append(("GWTC-1 posterior samples", f"{ev}_GWTC-1.hdf5", DCC + f"{ev}_GWTC-1.hdf5"))
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
    json.dump(rows, open(ROOT / "data" / "meta" / "stage5_files.json", "w"), indent=1)
    print("done", flush=True)
