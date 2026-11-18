"""Choose extra O1 strain files for the background from GWOSC metadata only (no strain is read).

Rule (pre-registration stage 5, section 5): bulk O1 4-kHz files (4096 s, GPS start on a multiple of 4096)
in which BOTH detectors have dutycycle_percent == 100 and nans_fraction == 0, that do not overlap the
event file, taken in order of increasing |file centre - event time| until N files are found.

Usage: python scripts/select_background_files.py            -> data/meta/background_files.json (stage 5)
       python scripts/select_background_files.py GW170817   -> data/meta/background_files_GW170817.json (stage 6)
"""
import json
import pathlib
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
API = "https://gwosc.org/api/v2"
EVENTS = {
    "GW150914": {"gps": 1126259462.4, "event_file_start": 1126257415, "n": 8},
    "GW151226": {"gps": 1135136350.6, "event_file_start": 1135134303, "n": 4},
}
STAGE6 = {"GW170817": {"gps": 1187008882.4, "event_file_start": 1187006835, "n": 2, "run": "O2"}}
SEARCH_HALF = 3 * 86400


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "gwlab-script"}), timeout=60) as r:
        return json.load(r)


def listing(det, a, b, run="O1"):
    out, url = [], f"{API}/runs/{run}/strain-files?detector={det}&sample-rate=4&start={a}&stop={b}"
    while url:
        j = get(url)
        out += j["results"]
        url = j["next"]
    return {r["gps_start"]: r for r in out}


if __name__ == "__main__":
    res = {}
    stage6 = len(sys.argv) > 1
    for name, ev in (STAGE6 if stage6 else EVENTS).items():
        a, b = int(ev["gps"] - SEARCH_HALF), int(ev["gps"] + SEARCH_HALF)
        lh, ll = listing("H1", a, b, ev.get("run", "O1")), listing("L1", a, b, ev.get("run", "O1"))
        common = sorted(set(lh) & set(ll), key=lambda s: abs(s + 2048 - ev["gps"]))
        e0 = ev["event_file_start"]
        chosen, rejected = [], []
        for s in common:
            if len(chosen) >= ev["n"]:
                break
            if s < e0 + 4096 and s + 4096 > e0:
                rejected.append({"gps_start": s, "why": "overlaps the event file"})
                continue
            meta = {d: get(lst[s]["detail_url"]) for d, lst in (("H1", lh), ("L1", ll))}
            ok = all(m["dutycycle_percent"] == 100.0 and m["nans_fraction"] == 0.0 for m in meta.values())
            if not ok:
                rejected.append({"gps_start": s, "why": "duty cycle below 100 %",
                                 "duty": {d: m["dutycycle_percent"] for d, m in meta.items()}})
                continue
            chosen.append({"gps_start": s, "offset_from_event_s": s + 2048 - ev["gps"],
                           "files": {d: {"url": m["hdf5_url"], "bytes": m["filesize_bytes"]} for d, m in meta.items()}})
        res[name] = {"event_gps": ev["gps"], "n_listed_h1": len(lh), "n_listed_l1": len(ll), "n_common": len(common),
                     "chosen": chosen, "rejected_before_filling": rejected}
        print(name, "common", len(common), "chosen", [c["gps_start"] for c in chosen], "rejected", len(rejected))
    out = ROOT / "data" / "meta" / ("background_files_GW170817.json" if stage6 else "background_files.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
