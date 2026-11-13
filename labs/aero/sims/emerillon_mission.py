"""Fly ÉMERILLON ER1 and ER2 through their scenarios (``labs/aero/designs/emerillon_mission.py``) and write a figure per
flight and one summary: ``labs/aero/sims/output/emerillon_<variant>_<scenario>.png`` and ``emerillon_missions.json``.

    .venv/bin/python labs/aero/sims/emerillon_mission.py                               # both variants, all five scenarios (about a minute)
    .venv/bin/python labs/aero/sims/emerillon_mission.py --variant ER2 --scenario palace sea
    .venv/bin/python labs/aero/sims/emerillon_mission.py --ceiling                     # the altitude question: 4 km for ER1, 7 km for ER2, and the packs that get there
    .venv/bin/python labs/aero/sims/emerillon_mission.py --range                       # the range question: best-range speed, radius of action, the packs
    .venv/bin/python labs/aero/sims/emerillon_mission.py --quick                       # two short flights, nothing written (the mocks do this)

Scenarios: ``range`` (out against the wind to the radius of action and back at the best-range speed), ``wildfire`` (out to a fire, round the smoke, back), ``mountain`` (catch a target circling a mountain),
``sea`` (catch a boat in a crosswind), ``climb`` (to the ceiling target and back), ``palace`` (a fast climb over Warsaw,
a swoop, a slow landing on the roof of the Palace of Culture and Science). A point-mass simulation: times, speeds,
energy and margins — not handling. No solver, no MuJoCo.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "designs"))
import emerillon_flight as ef  # noqa: E402
import emerillon_mission as mi  # noqa: E402

OUT = HERE / "output"


def ceilings() -> dict:
    out = {}
    for v in ef.VARIANTS:
        c = mi.ceiling(v)
        print(f"\n{ef.VARIANT_NAME[v]}: target {c['target_m']:.0f} m — the economy climb takes {c['climb_wh']:.1f} Wh and {c['time_s']:.0f} s, the landing {c['land_wh']:.1f} Wh; "
              f"the pack holds {c['pack_wh']:.1f} Wh ({c['budget_wh']:.1f} for the climb with the {mi.RESERVE:.0%} reserve) → "
              f"{'REACHED' if c['reaches_target'] else 'NOT REACHED'}; with the reserve it gets to {c['energy_ceiling_m']:.0f} m; "
              f"thrust over weight up there {c['static_tw_at_target']:.1f}, {c['T_C_at_target']:.0f} C outside")
        rows = mi.ceiling_options(v)
        for r in rows:
            print(f"   {r['pack']:<40s} {r['mass [g]']:5d} g  {r['price [USD]']:7.0f} USD  {r['top speed [km/h]']:4d} km/h  climb {r['climb [Wh]']:5.1f} of {r['pack [Wh]']:5.1f} Wh"
                  f"  reaches {r['reaches [m]']:5.0f} m  {'target met' if r['target met'] else ''}{'  (as built)' if r['as built'] else ''}")
        out[v] = {"ceiling": {k: x for k, x in c.items() if k != "profile"}, "options": rows}
    return out


def ranges() -> dict:
    out = {}
    for v in ef.VARIANTS:
        r = mi.range_curve(v)
        print(f"\n{ef.VARIANT_NAME[v]}: best range {r['range_km']:.1f} km (a radius of {r['radius_km']:.1f} km) at {r['best_range_ms'] * 3.6:.0f} km/h and "
              f"{r['wh_per_km']:.2f} Wh/km, in {r['range_time_s'] / 60:.1f} min; at top speed {r['range_at_top_km']:.1f} km; "
              f"longest in the air {r['endurance_s'] / 60:.1f} min at {r['best_endurance_ms'] * 3.6:.0f} km/h (hover {r['hover_endurance_s'] / 60:.1f} min)")
        rows = mi.range_options(v)
        for o in rows:
            print(f"   {o['pack']:<40s} {o['mass [g]']:5d} g  {o['price [USD]']:7.0f} USD  range {o['range [km]']:5.1f} km  radius {o['radius [km]']:4.1f} km"
                  f"  at {o['best-range speed [km/h]']:3d} km/h  {o['endurance [min]']:4.1f} min{'  (as built)' if o['as built'] else ''}")
        out[v] = {"range": {k: x for k, x in r.items() if k != "rows"}, "options": rows}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", nargs="*", default=list(ef.VARIANTS), choices=list(ef.VARIANTS))
    ap.add_argument("--scenario", nargs="*", default=list(mi.SCENARIOS), choices=list(mi.SCENARIOS))
    ap.add_argument("--ceiling", action="store_true", help="only the altitude analysis")
    ap.add_argument("--range", action="store_true", help="only the range analysis")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    quick = args.quick or os.environ.get("VEGETA_DEBUG") == "1"
    t0 = time.time()
    if quick:
        for v, s in (("ER1", "palace"), ("ER2", "mountain")):
            f = mi.fly(v, s, dt=0.05)
            print(f"{v} {s}: {'OK' if f.ok else 'NOT OK: ' + '; '.join(f.why)} — {f.metrics['time_s']:.0f} s, {f.metrics['energy_wh']:.1f} Wh")
        return 0
    summary = {}
    if not args.range:
        summary["ceilings"] = ceilings()
    if not args.ceiling:
        summary["ranges"] = ranges()
    if not (args.ceiling or args.range):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        OUT.mkdir(parents=True, exist_ok=True)
        flights = mi.fly_all(args.variant, args.scenario)
        print()
        for r in mi.table(flights):
            print("  ".join(f"{k} {x}" for k, x in r.items() if x not in (None, "")))
        for (v, s), f in flights.items():
            fig = mi.plot(f, OUT / f"emerillon_{v.replace(' ', '_')}_{s}.png")
            plt.close(fig)
        summary["flights"] = [{**r, "events": flights[(r["variant"], r["scenario"])].events, "notes": mi.SCENARIOS[r["scenario"]].notes} for r in mi.table(flights)]
        (OUT / "emerillon_missions.json").write_text(json.dumps(summary, indent=1, default=float))
        print(f"\nwritten: {OUT}/emerillon_<variant>_<scenario>.png and emerillon_missions.json")
    print(f"{time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
