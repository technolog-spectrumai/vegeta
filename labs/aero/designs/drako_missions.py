"""Drako's cargo route (helipad -> hut -> helipad -> village -> helipad) in three weathers, in parallel; per weather a
picture (``<out>/<name>.png``), the legs and deliveries (``<out>/<name>_legs.csv``, ``_deliveries.csv``) and a summary
of them all (``<out>/summary.csv``).

    python drako_missions.py --out runs/drako_missions                 # calm, breeze, strong
    python drako_missions.py --out runs/drako_missions --only calm
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

WEATHERS = {  # name: Wind kwargs
    "calm": dict(speed=0.0),
    "breeze": dict(speed=6.0, from_deg=270.0, gust=0.20, seed=1),        # a west wind, 6 m/s at 10 m, 20 % gusts
    "strong": dict(speed=12.0, from_deg=225.0, gust=0.30, seed=2),       # from the south-west, 12 m/s, 30 % gusts
}


def one(args):
    name, out = args
    import drako_scenario as ds
    t0 = time.time()
    scene = ds.Scene(wind=ds.Wind(**WEATHERS[name]))
    lab = ds.make_lab(scene)
    ep = ds.run(lab, scene)
    out = Path(out)
    legs, dels = ds.legs(ep), ds.deliveries(ep)
    legs.to_csv(out / f"{name}_legs.csv", index=False)
    dels.to_csv(out / f"{name}_deliveries.csv")
    ds.plot_mission(ep, scene, out / f"{name}.png", title=f"Drako cargo route, {name}: {WEATHERS[name]}")
    (out / f"{name}_mission.json").write_text(json.dumps(ep.log["mission"], indent=0, default=float))
    o = ep.outcome
    return {"weather": name, **WEATHERS[name], "success": o["success"], "reason": o["reason"],
            "time [min]": o["t_end"] / 60, "energy [Wh]": o["energy_Wh"], "lowest charge [%]": 100 * o["min_soc"],
            "delivered": int(dels["delivered"].notna().sum()), "packages": len(dels), "wall [s]": time.time() - t0}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", choices=list(WEATHERS), nargs="*")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    names = a.only or list(WEATHERS)
    with ProcessPoolExecutor(len(names)) as ex:
        rows = list(ex.map(one, [(n, str(out)) for n in names]))
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out / "summary.csv", index=False)
    print(df.round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
