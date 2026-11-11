#!/usr/bin/env python3
"""Pumba in the forest — MuJoCo dynamics through Chiron, one movie.

    xvfb-run -a python3 labs/land/sims/pumba_forest_movie.py            # headless (pyvista needs a display)
    python3 labs/land/sims/pumba_forest_movie.py --fps 25 --width 1280 --height 720

The scenario is ``designs/pumba_forest.py``: the light Pumba drives a winding forest trail between the trees, over
a rough floor and two branches, stops at a fallen log across the trail, picks it up with the main arm's grab,
swings it off the trail, sets it down and drives on. The camera chases Pumba on the trail and swings round to
the side for the arm work. Writes ``sims/output/pumba_forest.mp4`` and ``pumba_forest.json`` (the events and the
summary).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "labs" / "land" / "designs"))

import pumba_forest as pfo  # noqa: E402

OUT = ROOT / "labs" / "land" / "sims" / "output"


def camera(ep, every: int = 1):
    """A stateful camera for ``chiron.viz.frames``: a chase view while driving, a side view of the arm while it
    works (blended over 2 s), the focus smoothed so the bumps of the trail do not shake the picture."""
    t = np.asarray(ep.log["t"])[::every]
    ev = ep.log.get("events", [])
    t_stop = next((e[0] for e in ev if e[1] == "stop"), 1e9)
    t_go = next((e[0] for e in ev if e[1] == "log"), 1e9)
    state = {"k": 0, "focus": None, "psi": 0.0}

    def cam(com):
        k = min(state["k"], len(t) - 1)
        state["k"] += 1
        c = np.asarray(com, dtype=float)
        dt = t[k] - t[k - 1] if k > 0 else 0.0
        a = 1.0 - math.exp(-dt / 0.6)                  # the focus follows the COM with a 0.6 s lag
        prev = state["focus"]
        f = c if prev is None else (1 - a) * prev + a * c
        state["focus"] = f
        if prev is not None and np.hypot(*(f - prev)[:2]) > 1e-4:   # heading of the focus' motion, smoothed
            psi = math.atan2(f[1] - prev[1], f[0] - prev[0])
            d = (psi - state["psi"] + math.pi) % (2 * math.pi) - math.pi
            state["psi"] += (1.0 - math.exp(-dt / 1.5)) * d
        cs, sn = math.cos(state["psi"]), math.sin(state["psi"])

        def along(back, side, up):                     # behind the machine on its heading: on the trail
            return f + np.array([back * cs - side * sn, back * sn + side * cs, up])
        # weight of the arm view: 0 driving, 1 while the arm works
        w = min(max((t[k] - t_stop) / 2.0, 0.0), 1.0) * min(max((t_go + 1.0 - t[k]) / 2.0, 0.0), 1.0)
        chase_pos, chase_foc = along(-4.8, -0.7, 2.7), along(1.2, 0.0, 0.2)
        arm_pos, arm_foc = f + np.array([3.9, -0.5, 2.0]), f + np.array([0.9, 0.55, 0.35])   # on the trail ahead: clear of trees
        pos = (1 - w) * chase_pos + w * arm_pos
        foc = (1 - w) * chase_foc + w * arm_foc
        return tuple(pos), tuple(foc), (0, 0, 1)

    return cam


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--duration", type=float, default=62.0)
    ap.add_argument("--out", default=str(OUT / "pumba_forest.mp4"))
    a = ap.parse_args(argv)
    from vegeta.chiron import viz

    lab, scene = pfo.make_lab(log_dt=1.0 / a.fps)
    ep = pfo.run(lab, scene, duration=a.duration)
    summ = pfo.summary(ep, scene)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Pumba went {summ['distance_m']:.1f} m; the log lifted {summ['log_lift_m']:.2f} m, from "
          f"{summ['log_start']} to {summ['log_end']} (off the trail: {summ['log_off_trail']})")
    for e in summ["events"]:
        print("  ", e)
    images = viz.frames(ep, camera=camera(ep), size=(a.width, a.height), background="#cfe3ee",
                        ground_color="#5f5a36", scenery_range=40.0, robot_color="#d9b020")
    path = viz.to_video(images, a.out, fps=a.fps)
    Path(a.out).with_suffix(".json").write_text(json.dumps({"summary": summ, "lab_events": lab.events}, indent=1,
                                                           default=str))
    print(f"wrote {path} ({len(images)} frames, {len(images) / a.fps:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
