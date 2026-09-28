"""Run one scenario: build the whole machine, one CFD case with its propeller(s) as rotor disks, a particle movie.

    python scenarios/run_scenario.py air [-j N] [--dry-run] [--movie-only] [--force] ...

The machines are **preconfigured** (``PRESETS``: the designs, operating points and propellers the notebooks
currently use); reading a design exported by a notebook comes later. Output: ``scenarios/output/<scenario>/``
(``cad/``, ``case/``, ``<scenario>.mp4``, ``summary.json``). See ``scenarios/scenarios.md``.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DESIGNS = ROOT / "notebooks" / "designs"


@dataclass
class Scenario:
    name: str
    title: str
    design: str                          # "path/to/design.py:Class"
    design_params: dict
    template: str
    cfd: dict                            # CFDCase parameters
    geometry_units: str = "mm"
    stl_tolerance: float = 0.2
    movie: dict = field(default_factory=lambda: {"n": 600, "seconds": 8.0, "fps": 24})
    notes: tuple = ()
    build: Callable | None = None        # custom geometry: (design, params) -> dedalus Geometry
    force_scale: float = 1.0             # 0.5 for a double body: the real hull carries half the forces


# -- presets ---------------------------------------------------------------------------------------------------
def _polar_table(airfoil) -> list[list[float]]:
    alphas = np.unique(np.r_[np.arange(-180, -30, 10), np.arange(-30, 31, 1), np.arange(40, 181, 10)]).astype(float)
    cl, cd = airfoil.coefficients(np.radians(alphas))
    return [[float(a), float(d), float(l)] for a, l, d in zip(alphas, cl, cd)]


def air() -> Scenario:
    """The twin-motor fixed wing of ``09a_fixed_wing_design``: the preferred NACA 2415 wing, 4 deg, 14 m/s cruise,
    both 9x6 propellers at the recorded cruise rpm as counter-rotating rotor disks."""
    from vegeta import boreas, dedalus

    spec = f"{DESIGNS / 'fixed_wing.py'}:FixedWing"
    params = {"part": "aircraft", "thickness": 0.15, "angle_of_attack_deg": 4.0}
    p = dedalus.load_design(spec).resolve(**params)
    th = math.radians(p["angle_of_attack_deg"])

    def to_flow(x, y, z):                    # the aircraft is turned nose-up about +Y (right-hand rule)
        return np.array([x * math.cos(th) + z * math.sin(th), y, -x * math.sin(th) + z * math.cos(th)])

    x_prop = -p["nacelle_forward"] - 12.0     # mm: propeller plane just ahead of the nacelle nose
    d, pitch = boreas.inches(9, 6)            # the propeller of 09a Part 2 (same definitions)
    prop = boreas.Propeller.from_pitch("9x6 electric", d, pitch, blades=2, chord_root_m=0.014, chord_max_m=0.022,
                                       chord_tip_m=0.006, mass_kg=0.012, rotor_mass_kg=0.045, notes="generic planform")
    airfoil = boreas.Airfoil(name="thin cambered section", cl_alpha=2 * math.pi * 0.9, alpha0_deg=-2.5, cl_max=1.1,
                             cd0=0.02, k=0.04, source="assumed for a 9-inch blade at Re ~ 1.5e5")
    rho, v, rpm = 1.2, 14.0, 5800.0           # air; cruise speed; cruise rpm recorded from 09a (BEMT on the recorded polar)
    cfd = dict(velocity=v, kinematic_viscosity=1.5e-5, density=rho, reference_area=0.17, reference_length=0.25,
               center_of_rotation=(0.05, 0.0, 0.0), iterations=600, residual_target=1e-4,
               surface_level=4, near_level=3, wake_level=2, cells_per_length=2.0,
               disk1_center=(to_flow(x_prop, -p["nacelle_y"], 0.0) / 1000).tolist(),
               disk2_center=(to_flow(x_prop, p["nacelle_y"], 0.0) / 1000).tolist(),
               disk_axis=to_flow(-1.0, 0.0, 0.0).tolist(),
               diameter=prop.diameter, rpm=rpm, blades=prop.blades,
               blade=[[r, b, c] for r, b, c in zip(prop.r, prop.beta_deg, prop.chord)], polar=_polar_table(airfoil),
               rotation1=1, rotation2=-1, disk_level=5)
    return Scenario("air", "fixed-wing drone in cruise, both propellers running", spec, params, "aircraft_rotor_disks", cfd,
                    notes=("propellers are rotor disks (blade-element sources), not resolved blades",
                           "forces are the airframe's in the slipstream; the disks' thrust is in the solver log"))


def sub() -> Scenario:
    """The 1.2 m AUV of ``13_submarine`` at its 1.5 m/s cruise, nose upstream, its 120 mm 3-blade propeller at the
    recorded cruise rpm as a rotor disk just behind the tail tip."""
    from vegeta import boreas

    spec = f"{DESIGNS / 'submarine.py'}:Submarine"
    params = {"part": "vehicle"}
    prop = boreas.Propeller.from_pitch("120 mm 3-blade", 0.120, 0.100, blades=3, chord_root_m=0.018, chord_max_m=0.030,
                                       chord_tip_m=0.012, mass_kg=0.06, rotor_mass_kg=0.15, notes="generic planform")
    section = boreas.Airfoil(name="marine blade section", cl_alpha=5.5, alpha0_deg=-2.0, cl_max=1.0, cd0=0.02, k=0.05, source="assumed")
    rho, nu, v, rpm = 1025.0, 1.05e-6, 1.5, 910.0     # sea water; cruise; cruise rpm recorded from 13 (BEMT, wake 0.85)
    cb_x = 0.6393                                      # m: centre of buoyancy from the tail tip (recorded from 13)

    def nose_upstream(dd, p_):
        from vegeta import dedalus

        g = dd.generate(**p_)
        return dedalus.Geometry.from_cadquery(g.shape.rotate((0, 0, 0), (0, 0, 1), 180), name="submarine_nose_upstream")

    cfd = dict(velocity=v, kinematic_viscosity=nu, density=rho, reference_area=0.6936, reference_length=0.30,
               center_of_rotation=(-cb_x, 0.0, 0.0), iterations=600, residual_target=1e-4,
               cells_per_length=3.0, surface_level=4, near_level=3, wake_level=2,
               disk1_center=[0.04, 0.0, 0.0], disk_axis=[-1.0, 0.0, 0.0],        # 40 mm behind the tail tip; thrust forward
               diameter=prop.diameter, rpm=rpm, blades=prop.blades,
               blade=[[r, b, c] for r, b, c in zip(prop.r, prop.beta_deg, prop.chord)], polar=_polar_table(section),
               rotation1=1, disk_level=5)
    return Scenario("sub", "submarine at cruise, its propeller running", spec, params, "hull_rotor_disk", cfd,
                    build=nose_upstream,
                    notes=("the propeller is a rotor disk just behind the tail tip (no shaft or hub modelled)",
                           "forces are the hull's in the propeller's inflow; the disk's thrust is in the solver log"))


def boat() -> Scenario:
    """The 1 m survey boat of ``12_boat_at_sea`` at 1.5 m/s without a free surface: the hull and motor pod below the
    waterline mirrored about it (double body), the 60 mm 3-blade propeller behind the pod at the recorded cruise rpm and
    its mirror image turning the other way."""
    from vegeta import boreas

    spec = f"{DESIGNS / 'survey_boat.py'}:SurveyBoat"
    params = {"part": "hull_solid"}
    draft = 53.1                                       # mm: waterline above the keel at the loaded displacement (recorded from 12)
    prop = boreas.Propeller.from_pitch("60 mm 3-blade marine", 0.060, 0.050, blades=3, chord_root_m=0.012, chord_max_m=0.018,
                                       chord_tip_m=0.008, mass_kg=0.02, rotor_mass_kg=0.05, notes="generic planform")
    section = boreas.Airfoil(name="marine blade section", cl_alpha=5.5, alpha0_deg=-2.0, cl_max=1.0, cd0=0.02, k=0.05, source="assumed")
    rho, nu, v, rpm = 1025.0, 1.05e-6, 1.5, 2184.0     # sea water; cruise; cruise rpm recorded from 12 (BEMT, wake 0.9)

    def double_body(dd, p_):
        import cadquery as cq
        from vegeta import dedalus

        p = dd.resolve(**p_)
        hull = cq.Workplane("XY").add(dd.generate(part="hull_solid").shape)
        pod = cq.Workplane("XY").add(dd.generate(part="bracket").shape.translate((0, 0, p["depth"] * 0.55)))   # as in the boat
        big = 4 * max(p["length"], p["beam"])
        below = cq.Workplane("XY").box(big, big, big, centered=(True, True, False)).translate((0, 0, draft - big))
        under = hull.union(pod).intersect(below)
        body = under.union(under.mirror("XY", basePointVector=(0, 0, draft)))
        return dedalus.Geometry.from_cadquery(body.val().rotate((0, 0, 0), (0, 0, 1), 180), name="boat_double_body")

    # the pod (bracket) runs from x = -8 to -98 mm behind the transom, axis at z = 0.55 depth - bracket_height = -26.5 mm;
    # the propeller plane 10 mm behind it. After turning the boat bow-upstream (180 deg about z): x -> -x.
    z_prop = 0.55 * 170.0 - 120.0
    x_prop = 0.108
    cfd = dict(velocity=v, kinematic_viscosity=nu, density=rho, reference_area=2 * 0.231, reference_length=0.25,
               center_of_rotation=(-0.5, 0.0, draft / 1000), iterations=600, residual_target=1e-4,
               surface_level=4, near_level=3, wake_level=2, cells_per_length=2.0,
               disk1_center=[x_prop, 0.0, z_prop / 1000], disk2_center=[x_prop, 0.0, (2 * draft - z_prop) / 1000],
               disk_axis=[-1.0, 0.0, 0.0], diameter=prop.diameter, rpm=rpm, blades=prop.blades,
               blade=[[r, b, c] for r, b, c in zip(prop.r, prop.beta_deg, prop.chord)], polar=_polar_table(section),
               rotation1=1, rotation2=-1, disk_level=6)
    movie = {"n": 600, "seconds": 8.0, "fps": 24, "surface_z": draft / 1000,     # water below: speed colours; above: grey
             "above_label": "above the waterline: mirror image (double body, no air simulated)"}
    return Scenario("boat", "survey boat at cruise, no free surface (double body), propeller running", spec, params,
                    "aircraft_rotor_disks", cfd, build=double_body, force_scale=0.5, movie=movie,
                    notes=("no free surface: the waterline is a symmetry plane (double body); no waves, no wave drag, no trim",
                           "forces in summary.json 'real_hull' are half the double body's; the mirror disk turns the other way",
                           "the propeller is a rotor disk behind the motor pod"))


PRESETS: dict[str, Callable[[], Scenario]] = {"air": air, "sub": sub, "boat": boat}


# -- runner ------------------------------------------------------------------------------------------------------
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_geometry(sc: Scenario, out: Path) -> Path:
    from vegeta import dedalus

    dd = dedalus.load_design(sc.design)
    geom = sc.build(dd, sc.design_params) if sc.build else dd.generate(**sc.design_params)
    stl = out / "cad" / f"{sc.name}.stl"
    stl.parent.mkdir(parents=True, exist_ok=True)
    geom.export_stl(stl, tolerance=sc.stl_tolerance)
    return stl


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Whole-machine CFD with rotor-disk propellers and a particle movie")
    ap.add_argument("scenario", choices=sorted(PRESETS))
    ap.add_argument("-j", "--jobs", type=int, default=1, help="cores for the solver (MPI); default 1")
    ap.add_argument("--out", default=None, help="output directory (default scenarios/output/<scenario>)")
    ap.add_argument("--iterations", type=int, default=None, help="solver iterations (default: the preset's)")
    ap.add_argument("--seconds", type=float, default=None, help="movie length")
    ap.add_argument("--fps", type=int, default=None, help="movie frames per second")
    ap.add_argument("--movie-only", action="store_true", help="only re-render the movie from the solved case")
    ap.add_argument("--force", action="store_true", help="start again from an empty case")
    ap.add_argument("--dry-run", action="store_true", help="build the geometry, prepare the case, print the commands")
    args = ap.parse_args(argv)
    if args.jobs < 1:
        ap.error("-j must be >= 1")

    from tqdm.auto import tqdm
    from vegeta import aeromant
    from vegeta.aeromant import viz as aviz

    sc = PRESETS[args.scenario]()
    if args.iterations:
        sc.cfd["iterations"] = args.iterations
    movie = dict(sc.movie, **{k: v for k, v in (("seconds", args.seconds), ("fps", args.fps)) if v})
    out = Path(args.out) if args.out else ROOT / "scenarios" / "output" / sc.name
    if args.force and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    case_dir, movie_path = out / "case", out / f"{sc.name}.mp4"
    summary: dict[str, Any] = {"scenario": sc.name, "title": sc.title, "design": sc.design, "design_params": sc.design_params,
                               "template": sc.template, "processors": args.jobs, "started_at": utc_now(), "stages": {},
                               "notes": list(sc.notes)}
    print(f"[{sc.name}] {sc.title} -> {out}  ({args.jobs} core{'s' if args.jobs > 1 else ''})")

    stages = ["geometry", "case", "mesh + solve", "movie"]
    bar = tqdm(total=len(stages), desc=sc.name, unit="stage", position=0)

    def stage(name, fn):
        bar.set_postfix_str(name)
        t0 = time.monotonic()
        value = fn()
        summary["stages"][name] = round(time.monotonic() - t0, 1)
        bar.update(1)
        return value

    env = aeromant.OpenFOAMEnvironment.detect()
    stl = out / "cad" / f"{sc.name}.stl"
    case = None

    def make_case():
        return aeromant.CFDCase(sc.template, stl, sc.cfd, workdir=case_dir, geometry_units=sc.geometry_units, environment=env)

    if args.movie_only:
        bar.update(3)
        case = make_case()
        if not case.results().ok:
            bar.close()
            print(f"[{sc.name}] no solved case in {case_dir}: run without --movie-only first", file=sys.stderr)
            return 2
    else:
        stage("geometry", lambda: stl if stl.is_file() else build_geometry(sc, out))
        case = make_case()

        def prepare():
            # a solved case is kept; anything else is prepared again (seconds), so template fixes always reach the case
            if case.is_prepared and case.results().ok:
                return "already solved"
            r = case.prepare(overwrite=True)
            if not r.ok:
                raise RuntimeError(f"prepare failed: {r.messages}")
            return r
        stage("case", prepare)
        if args.dry_run:
            bar.close()
            print(f"[{sc.name}] dry run: case prepared in {case_dir}; the steps would be:")
            for s in case.pipeline(args.jobs):
                print("   ", " ".join(env.command(list(s.argv))) if s.kind == "openfoam" else f"(internal) {' '.join(s.argv)}")
            return 0

        def solve():
            done = case.results()
            if done.ok:
                return done
            r = case.run(progress=True, processors=args.jobs)
            if not r.ok:
                raise RuntimeError(f"CFD failed: {r.messages[:3]} (logs in {case_dir})")
            return r
        res = stage("mesh + solve", solve)
        summary["metrics"] = {k: v for k, v in res.metrics.items() if not isinstance(v, (list, dict))}
    stage("movie", lambda: aviz.animate_particles(case, movie_path, size=(1280, 720), progress=True, **movie))
    bar.close()
    if "metrics" not in summary:
        summary["metrics"] = {k: v for k, v in case.results().metrics.items() if not isinstance(v, (list, dict))}
    summary.update(finished_at=utc_now(), files={"movie": str(movie_path), "case": str(case_dir), "stl": str(stl)})
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    m = summary["metrics"]
    if sc.force_scale != 1.0:
        summary["real_hull"] = {k: m[k] * sc.force_scale for k in ("drag_force_N", "lift_force_N") if m.get(k) is not None}
        summary["real_hull"]["note"] = f"x {sc.force_scale}: {sc.notes[0]}"
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        print(f"[{sc.name}] real hull (x {sc.force_scale} of the double body): {summary['real_hull']}")
    print(f"[{sc.name}] movie {movie_path}")
    if m.get("Cl") is not None:
        print(f"[{sc.name}] Cl {m['Cl']:.3f}, Cd {m['Cd']:.4f}, lift {m.get('lift_force_N', float('nan')):.2f} N, "
              f"drag {m.get('drag_force_N', float('nan')):.2f} N, converged {m.get('converged')}, cells {m.get('mesh_cells')}")
    print(f"[{sc.name}] time per stage [s]: {summary['stages']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
