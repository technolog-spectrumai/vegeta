#!/usr/bin/env python3
"""Velutina flies medical aid to a mountain rescue site — the reduced flight model, one movie.

    python3 scenarios/velutina_mission.py                 # hand-over at the rock face (default)
    python3 scenarios/velutina_mission.py --mode pad      # set-down on the pad
    python3 scenarios/velutina_mission.py --seconds 40 --wind 12
    python3 scenarios/velutina_mission.py --mode turbine     # Velutina v2: the wind-turbine blade inspection

The scenario is ``notebooks/designs/velutina_flight.py`` (the same one notebook 24 §7 and §10 run) with the notebook's
first design as a preset (mass budget, drag areas, the Boreas hover point and thrust limit, the 6S 8000 mAh pack): the
valley depot, a fast climb along the route, the delivery at the site, the flight home. ``--mode turbine`` flies the v2
inspection of ``velutina_inspection.py`` instead (the parked turbine, four passes per blade, the suspect-point dwells).
Writes ``scenarios/output/velutina_<mode>.mp4`` and ``velutina_<mode>.json`` (summary, events, phases).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks" / "designs"))

import velutina_flight as vf  # noqa: E402
import velutina_inspection as vi  # noqa: E402

OUT = ROOT / "scenarios" / "output"

# notebook 24's first design (recorded values; the notebook recomputes them)
PRESET = dict(mass_kg=2.60, capsule_kg=0.54, cda_axial_m2=0.0031, cda_cross_m2=0.0575, hover_power_w_sl=506.0,
              hover_thrust_n_sl=25.5, max_thrust_n_sl=100.2, battery_wh=151.0, max_speed=40.0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="hand", choices=("hand", "pad", "turbine"))
    ap.add_argument("--wind", type=float, default=8.0, help="valley wind [m/s]")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=24.0, help="length of the movie")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--wall-relative", action="store_true", help="hand-over with a wall-relative sensor and a tight hold")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    kw = dict(PRESET)
    if args.wall_relative:
        kw.update(position_noise_m=0.05, response_s=0.6, hold_gain=2.0)
    if args.mode == "turbine":
        kw.update(mass_kg=PRESET["mass_kg"] - PRESET["capsule_kg"] + 0.25, capsule_kg=0.0, position_noise_m=0.05, response_s=0.6, hold_gain=2.0, max_speed=8.0)
        ac, turbine, cam = vf.Aircraft(**kw), vi.Turbine(), vi.Camera()
        wind = vi.SiteWind(speed=args.wind, hub_height=turbine.hub_height, gust_sigma=1.5, seed=args.seed)
        wps, _ = vi.inspection_path(turbine, vi.InspectionPlan(), cam, dwell_s=3.0)
        ep = vi.follow_path(ac, turbine, wind, wps, seed=args.seed)
        movie, frames = vi.render_movie(ep, turbine, OUT / "velutina_turbine.mp4", seconds=args.seconds, fps=args.fps, progress=True)
        table = ep.table().reset_index()
    else:
        ac, terrain = vf.Aircraft(**kw), vf.Terrain()
        ep = vf.simulate(ac, terrain, vf.Wind(speed=args.wind, seed=args.seed), vf.Plan(mode=args.mode))
        movie, frames = vf.render_movie(ep, terrain, OUT / f"velutina_{args.mode}.mp4", seconds=args.seconds, fps=args.fps, progress=True)
        table = ep.phase_table().reset_index()
    doc = {"preset": kw, "mode": args.mode, "wind_m_s": args.wind, "summary": ep.summary(),
           "phases": table.round(3).to_dict(orient="records"), "movie": str(movie), "frames": len(frames)}
    (OUT / f"velutina_{args.mode}.json").write_text(json.dumps(doc, indent=1, default=float))
    s = ep.summary()
    print(f"{movie} ({len(frames)} frames); {s['duration_s'] / 60:.1f} min, {s['energy_wh']:.0f} Wh of {ac.battery_wh:.0f}; events: {[e[1] for e in ep.events]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
