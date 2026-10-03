"""MERLIN (notebook 26): the race to a fire and the sampling mission through its smoke, with the movie.

    python scenarios/merlin_mission.py                              # the fastest propulsor, a fire 20 km out
    python scenarios/merlin_mission.py --propulsion edf --distance-km 10 --heat-mw 60 --wind 7
    python scenarios/merlin_mission.py --race-only                  # the 5 / 8 / 10 / 20 / 30 km table, no movie

Builds the three MERLINs from the design files' defaults (``merlin_flight.build_merlins``: the notebook's first
design; the notebook recomputes the airframe mass from the CAD and may refine the polar with CFD), races them to fires 5, 8, 10, 20 and 30 km out (reach, return and landing times; flat ground and a fire
``--fire-elevation`` m up a mountain), flies
the chosen one through a Gaussian smoke plume and renders the movie. Writes
``scenarios/output/merlin_<propulsion>_<km>km.mp4`` and ``.json`` (the race table, the mission summary, the phases and
the fire's size from the passes).
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks" / "designs"))
import merlin_flight as mf  # noqa: E402

OUT = ROOT / "scenarios" / "output"
NAME = {"edf": "ducted fan", "tractor": "tractor propeller", "pusher": "pusher propeller"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--propulsion", choices=("fastest", "edf", "tractor", "pusher"), default="fastest")
    ap.add_argument("--distance-km", type=float, default=20.0)
    ap.add_argument("--heat-mw", type=float, default=40.0, help="the fire's heat release")
    ap.add_argument("--wind", type=float, default=5.0, help="wind speed [m/s]")
    ap.add_argument("--wind-from", type=float, default=225.0, help="wind direction, blowing from [deg]")
    ap.add_argument("--seconds", type=float, default=24.0)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--quick", action="store_true", help="coarse propulsor tables (seconds instead of a minute)")
    ap.add_argument("--fire-elevation", type=float, default=600.0, help="the mountain case: the fire this far above the launch site [m]")
    ap.add_argument("--race-only", action="store_true")
    a = ap.parse_args(argv)

    merlins = mf.build_merlins(quick=a.quick)
    import pandas as pd
    pd.set_option("display.width", 200)
    dists = sorted(set(mf.DISTANCES_KM) | {a.distance_km})
    cols = ["reach [min]", "return [min]", "landing [min]", "dash [m/s]", "dash climb [deg]", "return [m/s]", "limited by"]
    flat = mf.race_table(merlins, dists)
    print("flat forest: reach (first data by radio), return (fire to touchdown), landing (launch to touchdown)")
    print(flat[cols].round(2).to_string())
    mountain = mf.race_table(merlins, dists, fire_elevation_m=a.fire_elevation)
    print(f"\nthe fire {a.fire_elevation:.0f} m up a mountain (a gentle climb all along the dash)")
    print(mountain[cols].round(2).to_string())
    table = {"flat": flat.to_dict("index"), f"mountain_{a.fire_elevation:.0f}m": mountain.to_dict("index")}
    table = {case: {" / ".join(k): v for k, v in t.items()} for case, t in table.items()}
    kind = a.propulsion
    if kind == "fastest":
        reach = flat.xs(f"{a.distance_km:g} km", level=0)["reach [min]"].astype(float)
        kind = next(k for k, n in NAME.items() if n == reach.idxmin())
    print(f"flying the {NAME[kind]} to a fire {a.distance_km:.0f} km out")
    if a.race_only:
        return 0
    v = merlins[kind]
    race = mf.race(v["airframe"], v["unit"], a.distance_km * 1000)
    if not race["reachable"]:
        print("out of reach:", race["why"])
        return 1
    plume = mf.Plume(source_xy=(a.distance_km * 1000, 0.0), heat_mw=a.heat_mw, wind_speed=a.wind, wind_from_deg=a.wind_from)
    forest = mf.Forest()
    ep = mf.fly(v["airframe"], v["unit"], forest, plume, race_result=race)
    est = mf.source_estimate(plume, ep.passes) if len(ep.passes) >= 2 else {}
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / f"merlin_{kind}_{a.distance_km:.0f}km"
    movie, frames = mf.render_movie(ep, forest, plume, stem.with_suffix(".mp4"), seconds=a.seconds, fps=a.fps,
                                    title=f"MERLIN ({NAME[kind]}) — a fire {a.distance_km:.0f} km out", progress=True)
    doc = {"propulsion": kind, "race": table, "mission": ep.summary(), "phases": ep.phase_table().reset_index().to_dict("records"),
           "source_estimate": {k: x for k, x in est.items() if not hasattr(x, "shape")}, "fire_mw": a.heat_mw,
           "movie": str(movie), "frames": len(frames)}
    stem.with_suffix(".json").write_text(json.dumps(doc, indent=1, default=float))
    s = ep.summary()
    print(f"first pass after {s['time_to_first_pass_s'] / 60:.1f} min, {s['passes']} passes, peak CO {s['peak_ppm']:.2f} ppm, "
          f"fire estimated {est.get('heat_mw', float('nan')):.0f} MW (true {a.heat_mw:.0f}), home after {s['flight_time_s'] / 60:.0f} min, "
          f"{s['energy_wh']:.0f} Wh -> {movie}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
