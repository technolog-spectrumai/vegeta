"""The net dropped from above: ARIGO (and GAVILAN) hanging nose-up over a bandit throws its slingshot's aft net straight
down — onto a quadcopter hovering below it, or onto one that has landed. The launcher test's model (net_drone.py on
dev_gavilan_net, ``--light``: the 1.5 m net, four 8 g weights) with the muzzle above the target, a little behind it
(``OFFSET``), aimed at it; four shots a case with the test's scatter (0.5 degrees, 3 % speed).

    python arigo_drop_net.py --net-py PATH/TO/net_drone.py [--speeds 20,30] [--heights 3,4,5] [--out FILE]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

OFFSET = 0.5               # m: the muzzle this far behind the target horizontally (it hangs a little behind the bandit)
TARGETS = {"hovering 4 m up": 4.0, "landed": 0.15}


def load(path):
    spec = importlib.util.spec_from_file_location("net_drone", path)
    nd = importlib.util.module_from_spec(spec)
    sys.modules["net_drone"] = nd
    spec.loader.exec_module(nd)
    return nd


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net-py", required=True, type=Path)
    ap.add_argument("--speeds", default="20,30")
    ap.add_argument("--heights", default="3,4,5", help="the muzzle above the target [m]")
    ap.add_argument("--shots", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("arigo_drop_net_results.json"))
    a = ap.parse_args(argv)
    nd = load(a.net_py)
    out = {}
    for v0 in (float(x) for x in a.speeds.split(",")):
        nd.configure(light=True, drone="quad")
        nd.V0 = v0
        cal = nd.calibrate(nd.Sim())
        for tname, zt in TARGETS.items():
            for h in (float(x) for x in a.heights.split(",")):
                nd.MUZZLE = np.array([0.0, 0.0, zt + h])
                sim, rng, res = nd.Sim(), np.random.default_rng(7), []
                for _ in range(a.shots):
                    r = nd.shot(sim, cal, np.array([OFFSET, 0.0, zt]), np.zeros(3), rng)
                    res.append(bool(r["fouled"]))
                key = f"{v0:g} m/s, {tname}, {h:g} m above"
                out[key] = f"{sum(res)}/{len(res)}"
                print(f"net {key}: fouled {sum(res)}/{len(res)}", flush=True)
                a.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
