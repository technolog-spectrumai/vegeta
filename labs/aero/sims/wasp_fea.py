"""WASP's FEA — one printed arm with its motor pod (Talos / Gmsh / CalculiX). RUN BY YOU, on a machine with CalculiX.

ÉMERILLON's arm cases (``emerillon_campaigns``) on a WASP build: the arm clamped at its root pad, loaded on the pod's
motor face, all at ultimate x1.5:

- ``A``: the static full-throttle thrust of one motor;
- ``B``: a 5 g manoeuvre across the arm's thin way on the motor unit, the thrust on;
- ``C``: a tail-first landing knock, the motor unit at 8 g along the axis, no thrust;
- ``modes``: the first modes with the motor and propeller as a point mass, read against the rotor's once-per-turn
  frequency at top speed.

Printed PA6-CF as a solid, isotropic equivalent (3300 MPa, 60 MPa in the layer plane, 30 MPa across the layers:
assumed until coupons are pulled). The peak stress is a nodal value and is mesh dependent at the clamp and at the
loaded face; the hand value beside it is a cantilever's.

Writes ``labs/aero/sims/output/<build>_fea.yaml`` after every case.

    labs/aero/sims/wasp_fea.sh --list
    labs/aero/sims/wasp_fea.sh                            # WASP-1F, all four cases
    labs/aero/sims/wasp_fea.sh --cases A modes --element-size 1.0
"""
import argparse
import math
import time

import wasp_common as wc
from wasp_common import wasp

import emerillon as em  # noqa: E402
import emerillon_campaigns as ec  # noqa: E402
import emerillon_flight as ef  # noqa: E402

CASES = ("A", "B", "C", "modes")


def loads(veh) -> dict:
    T = ec.ULTIMATE * veh.static_thrust
    side = ec.ULTIMATE * ec.N_MANOEUVRE * ef.G * veh.motor_unit_kg
    knock = ec.ULTIMATE * ec.LANDING_G * ef.G * veh.motor_unit_kg
    return {"A": {"what": f"static thrust {veh.static_thrust:.2f} N x {ec.ULTIMATE}", "fx": -T, "fz": 0.0},
            "B": {"what": f"{ec.N_MANOEUVRE:g} g across the arm on the {veh.motor_unit_kg * 1e3:.1f} g motor unit, thrust on, x {ec.ULTIMATE}", "fx": -T, "fz": side},
            "C": {"what": f"landing knock, the motor unit at {ec.LANDING_G:g} g along the axis, x {ec.ULTIMATE}", "fx": knock, "fz": 0.0},
            "modes": {"what": f"first modes, {veh.motor_unit_kg * 1e3:.1f} g on the motor face (rotor 1P at top speed {veh.rpm_top / 60:.0f} Hz)"}}


def beam(veh, p: dict) -> dict:
    """The arm as a cantilever (the hand check): I = 0.036 c t^3 the thin way, 0.044 t c^3 the stiff way."""
    c, t = p["arm_chord"], p["arm_thickness"]
    span = p["arm_radius"] - em.Emerillon.layout(p)["R"]
    i_thin, i_stiff = 0.036 * c * t ** 3, 0.044 * t * c ** 3
    L = loads(veh)
    return {"span_mm": wc.num(span, 1), "stress_A_MPa": wc.num(abs(L["A"]["fx"]) * span * (c / 2) / i_stiff, 2),
            "stress_B_side_MPa": wc.num(L["B"]["fz"] * span * (t / 2) / i_thin, 2),
            "f1_thin_Hz": wc.num(math.sqrt(3 * ec.PA6CF["E"] * i_thin / (veh.motor_unit_kg * 1e-3 * span ** 3)) / (2 * math.pi), 0),
            "rotor_1P_Hz": wc.num(veh.rpm_top / 60.0, 0)}


def model(build: str, veh, name: str, element_size: float):
    from vegeta import talos
    c = wasp.VARIANTS[build]
    p = em.resolve(wasp.cad_parameters(c))
    step = wc.RUNS / "fea" / f"{wc.slug(build)}_arm.step"
    step.parent.mkdir(parents=True, exist_ok=True)
    if not step.exists():
        wasp.cad(c, part="arm").export_step(str(step))
    boxes = ec.arm_regions(p)
    m = ec.PA6CF
    mat = talos.Material(m["name"], youngs_modulus=m["E"], poissons_ratio=m["nu"], density=m["rho"], yield_strength=m["strength"], source=m["source"])
    regions = [talos.SurfacesInBox("root", boxes["root"]), talos.SurfacesInBox("mount", boxes["mount"])]
    L = loads(veh)[name]
    force = [] if name == "modes" else [talos.Force("mount", fx=L["fx"], fz=L["fz"])]
    masses = [talos.PointMass("mount", veh.motor_unit_kg * 1e-3)] if name == "modes" else []
    return talos.StructuralModel(geometry=str(step), units="mm-N-MPa", material=mat, regions=regions, supports=[talos.FixedSupport("root")],
                                 loads=force, mesh_settings=talos.MeshSettings(element_size=element_size, order=2), masses=masses)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", default=wc.DEFAULT, choices=wc.BUILDS)
    ap.add_argument("--cases", nargs="*", default=list(CASES), choices=CASES)
    ap.add_argument("--element-size", type=float, default=1.2, help="mm (the arm is about 7 mm thick)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--list", action="store_true", help="print the cases and the hand check and stop: nothing is meshed or solved")
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args(argv)
    veh = wasp.vehicle(wasp.VARIANTS[a.build])
    p = em.resolve(wasp.cad_parameters(wasp.VARIANTS[a.build]))
    hb = beam(veh, p)
    print(f"{a.build}: {veh.label}\n  arm {p['arm_chord']:.1f} x {p['arm_thickness']:.1f} mm, span {hb['span_mm']} mm; material {ec.PA6CF['name']}, {ec.PA6CF['strength']:.0f} MPa")
    print(f"  hand check (cantilever): {hb}")
    if a.list:
        for n in a.cases:
            print(f"  {n:6s} {loads(veh)[n]['what']}")
        print(f"  {len(a.cases)} cases, element size {a.element_size} mm: NOT RUN")
        return 0
    path = wc.OUT / f"{wc.slug(a.build)}_fea.yaml"
    doc = wc.load(path, a.fresh) or {"aircraft": a.build, "config": veh.label, "part": "one arm with its pod, clamped at the root pad",
                                     "material": dict(ec.PA6CF), "ultimate_factor": ec.ULTIMATE, "hand": hb, "cases": {}}
    bad = 0
    for n in a.cases:
        if doc["cases"].get(n, {}).get("status") == "solved":
            print(f"  {n}: already solved (kept)")
            continue
        t1, work = time.time(), str(wc.RUNS / "fea" / f"{wc.slug(a.build)}_{n}")
        try:
            mdl = model(a.build, veh, n, a.element_size)
            mesh = mdl.mesh(work)
            if n == "modes":
                r = mdl.solve_modes(work, n_modes=4, threads=a.threads)
                f = [float(x) for x in r.metrics["frequencies_hz"]]
                doc["cases"][n] = {"what": loads(veh)[n]["what"], "status": "solved", "frequencies_Hz": [round(x, 1) for x in f],
                                   "rotor_1P_Hz": hb["rotor_1P_Hz"], "first_mode_over_1P": wc.num(f[0] / (veh.rpm_top / 60.0), 2), "hand_f1_Hz": hb["f1_thin_Hz"]}
            else:
                r = mdl.solve(work, threads=a.threads)
                m = r.metrics
                doc["cases"][n] = {"what": loads(veh)[n]["what"], "status": "solved", "max_von_mises_MPa": wc.num(m["max_von_mises"], 2),
                                   "at_mm": [round(float(x), 1) for x in m["max_von_mises_location"]], "max_displacement_mm": wc.num(m["max_displacement"], 3),
                                   "reserve_in_plane": wc.num(ec.PA6CF["strength"] / m["max_von_mises"], 2),
                                   "reserve_across_layers": wc.num(ec.PA6CF["strength_z"] / m["max_von_mises"], 2), "messages": list(r.messages)[:3]}
            doc["cases"][n].update({"nodes": mesh.metrics["n_nodes"], "elements": mesh.metrics["n_elements"]})
        except Exception as exc:
            bad += 1
            doc["cases"][n] = {"what": loads(veh)[n]["what"], "status": "failed", "message": f"{type(exc).__name__}: {exc}"[:300]}
        doc["cases"][n]["seconds"] = round(time.time() - t1, 1)
        print(f"  {n}: {doc['cases'][n]}")
        wc.save(path, doc)
    print(f"-> {path}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
