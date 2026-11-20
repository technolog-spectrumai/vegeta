"""Merge run files (``output/runs/arigo_run_<time>_<host>.yaml``) from several runs or machines into ONE file.

    python labs/aero/sims/arigo_merge.py output/runs/*.yaml -o arigo_merged.yaml

Each result keeps the run it came from (``source``); when two runs hold the same result (e.g. arigo1_mujoco), the
newer run's wins, and the older is listed under ``superseded``."""
import argparse
import json
import time
from pathlib import Path

import yaml


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None)
    a = ap.parse_args(argv)
    runs = []
    for f in a.files:
        d = yaml.safe_load(f.read_text()) or {}
        runs.append((d.get("run", {}).get("updated") or d.get("run", {}).get("started") or "", f.name, d))
    runs.sort()
    merged = {"merged": {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "runs": [{"file": n, **d.get("run", {})} for _, n, d in runs]},
              "results": {}, "superseded": {}}
    for _, name, d in runs:
        for key, val in (d.get("results") or {}).items():
            if key in merged["results"]:
                merged["superseded"].setdefault(key, []).append(merged["results"][key]["source"])
            if key == "arigo_perf" and key in merged["results"]:          # cards from different runs add up
                cards = {**merged["results"][key]["data"].get("cards", {}), **(val or {}).get("cards", {})}
                val = {**val, "cards": cards}
            merged["results"][key] = {"source": name, "data": val}
    out = a.out or a.files[0].parent / f"arigo_merged_{time.strftime('%Y%m%d-%H%M%S')}.yaml"
    out.write_text(yaml.safe_dump(json.loads(json.dumps(merged, default=str)), sort_keys=False, width=140))
    print(f"{len(runs)} runs, {len(merged['results'])} results -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
