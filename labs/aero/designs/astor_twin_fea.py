"""HAWK 2 (ASTOR-TWIN) — the wing spar at 8 g in Talos (CalculiX FEA). Prepared here, solved where CalculiX and Gmsh are.

The flight controller's g limit is 8 (``astor_v_flight.SAS``); nothing was sized for it. This sizes the one member that
carries it, the wing's spar, and checks it:

- **the beam reference** (``reference``, closed form, runs anywhere): the half wing's elliptic lift at 8 g on the
  aircraft's mass, ultimate (x ``ULTIMATE``), as a cantilever from the body's side: the bending moment and stress at
  the root and at the pivot, the tip deflection, the first bending frequency with the motor on the spar.
- **spar_8g** (FEA): one half wing's spar — the inner tube from the centre plane to the pivot, a plug through the
  pivot, the outer tube to the tip — held inside the body, the lift on six stations.
- **spar_motor_out_8g** (FEA): the same less the relief of the motor's weight (the motor torn off or not fitted).
- **spar_modes** (FEA): the spar with the motor, nacelle and propeller as a point mass at the motor station.

NOT covered (each needs its own drawing first): the pivot's lugs and lock, the skin and ribs, the tail halves' spigot,
the fin, the launcher's trunnion, flutter. The spar is one material here (the plug is carbon too).

Units mm-N-MPa, masses in tonnes. ``solve(run=False)`` reads what exists and marks the rest NOT RUN.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import astor_twin as W
import astor_v_flight as VF

G = 9.80665
ULTIMATE = 1.5
KNOCKDOWN = 0.8                    # carbon: holes, clamps, impact damage
CARBON_TUBE = dict(name="roll-wrapped carbon tube", E=70000.0, nu=0.30, rho=1.55e-9, strength=500.0,
                   source="roll-wrapped T700 tube class, quasi-axial layup (assumed; verify with the supplier): as drako_fea")
SPAR = dict(inner_od=25.0, inner_id=22.0, outer_od=16.0, outer_id=13.0, plug_mm=80.0)    # the inner tube takes the root moment with a reserve; the outer fits the tip's 20 mm of depth
MOTOR_KG = 0.46 + 0.15 + 0.07 + 0.07          # motor, controller, variable-pitch hub and propeller, nacelle (astor_twin.checks: HAND)
STATIONS = 6
MARGIN = 0.6                                   # mm around a region's box


def params():
    return W.resolve(W.variant(net_launchers=1, talon_deploy=-1.0, net_count=0))


def loads(p=None, n=None, relief=True) -> dict:
    """The half wing's lift at ``n`` g, ultimate, as forces on STATIONS equal lengths of the spar from the body's side
    to the tip (an elliptic distribution over the half span), less the motor's own weight x n at its station."""
    p = p or params()
    n = n or VF.SAS["twin"]["g_max"]
    m = VF.get("twin")["mass"]
    half = p["pivot_y"] + p["panel_span"]
    y0 = p["fuselage_diameter"] / 2
    edges = np.linspace(y0, half, STATIONS + 1)
    ell = lambda y: (y / half * math.sqrt(max(1 - (y / half) ** 2, 0.0)) + math.asin(min(y / half, 1.0))) / (math.pi / 2)     # the share of the half wing's lift inboard of y
    L_half = ULTIMATE * n * m * G / 2
    F = [L_half * (ell(b) - ell(a)) for a, b in zip(edges[:-1], edges[1:])]
    F[0] += L_half * ell(y0)                                               # the lift over the body goes in at the root station
    out = {"n": n, "mass_kg": m, "half_span_mm": half, "edges_mm": [float(e) for e in edges], "lift_N": [float(f) for f in F],
           "half_wing_lift_ultimate_N": L_half, "motor_relief_N": ULTIMATE * n * MOTOR_KG * G if relief else 0.0, "motor_y_mm": p["motor_y"]}
    return out


def reference(p=None, n=None) -> dict:
    """The spar as a cantilever beam (closed form): the numbers the FEA is read against, and the spar's sizing."""
    p = p or params()
    ld = loads(p, n)
    half, y0, yp = ld["half_span_mm"], p["fuselage_diameter"] / 2, p["pivot_y"]
    mids = [(a + b) / 2 for a, b in zip(ld["edges_mm"][:-1], ld["edges_mm"][1:])]
    moment = lambda y: sum(f * (ym - y) for f, ym in zip(ld["lift_N"], mids) if ym > y) - (ld["motor_relief_N"] * (ld["motor_y_mm"] - y) if ld["motor_y_mm"] > y else 0.0)
    I = lambda od, i: math.pi / 64 * (od ** 4 - i ** 4)
    Ii, Io = I(SPAR["inner_od"], SPAR["inner_id"]), I(SPAR["outer_od"], SPAR["outer_id"])
    s_root, s_piv = moment(y0) * SPAR["inner_od"] / 2 / Ii, moment(yp) * SPAR["outer_od"] / 2 / Io
    allow = CARBON_TUBE["strength"] * KNOCKDOWN
    # the tip deflection: the curvature M / E I integrated twice along the two tubes
    ys = np.linspace(y0, half, 400)
    k = np.array([moment(y) / (CARBON_TUBE["E"] * (Ii if y < yp else Io)) for y in ys])
    slope = np.concatenate([[0.0], np.cumsum(0.5 * (k[1:] + k[:-1]) * np.diff(ys))])
    defl = float(np.sum(0.5 * (slope[1:] + slope[:-1]) * np.diff(ys)))
    area = lambda od, i: math.pi / 4 * (od ** 2 - i ** 2)
    g_in, g_out = area(SPAR["inner_od"], SPAR["inner_id"]) * 2 * yp * 1.55e-3, area(SPAR["outer_od"], SPAR["outer_id"]) * 2 * p["panel_span"] * 1.55e-3
    # the first bending mode: the motor on the inner tube as a cantilever with a tip mass at the motor's station
    Lm = ld["motor_y_mm"] - y0
    f1 = math.sqrt(3 * CARBON_TUBE["E"] * Ii / Lm ** 3 / (MOTOR_KG * 1e-3)) / (2 * math.pi)
    return {"load_factor": ld["n"], "ultimate_factor": ULTIMATE, "mass_kg": round(ld["mass_kg"], 2), "half_wing_lift_ultimate_N": round(ld["half_wing_lift_ultimate_N"]),
            "spar": {**SPAR, "material": CARBON_TUBE["name"], "allowable_MPa": allow, "mass_both_wings_g": round(g_in + g_out),
                     "fits": {"wing_depth_at_the_root_mm": round(0.12 * p["glove_root_chord"]), "at_the_pivot_mm": round(0.12 * p["panel_root_chord"]), "at_the_tip_mm": round(0.12 * p["panel_tip_chord"])}},
            "root": {"moment_Nm": round(moment(y0) / 1000.0, 1), "stress_MPa": round(s_root), "margin": round(allow / s_root - 1, 2)},
            "pivot": {"moment_Nm": round(moment(yp) / 1000.0, 1), "stress_MPa": round(s_piv), "margin": round(allow / s_piv - 1, 2)},
            "tip_deflection_ultimate_mm": round(defl), "tip_deflection_share_of_the_half_span": round(defl / half, 3),
            "first_bending_with_the_motor_Hz": round(f1, 1),
            "note": "no relief from the wing's own weight (conservative); the lift over the body is put in at the root station"}


# ------------------------------------------------------------------------------------------- the CAD and the cases
def _tube(y0, y1, od, i):
    import cadquery as cq
    return cq.Workplane("XZ").circle(od / 2).circle(i / 2).extrude(-(y1 - y0)).translate((0, y0, 0))


def spar_cad(p=None):
    """One half wing's spar along +y from the centre plane: the inner tube in segments at the load stations (Talos
    selects whole faces), a solid plug through the pivot, the outer tube in segments. Unions are not cleaned."""
    import cadquery as cq
    p = p or params()
    ld = loads(p)
    yp, half = p["pivot_y"], ld["half_span_mm"]
    cuts = sorted(set([0.0] + [e for e in ld["edges_mm"]] + [yp, ld["motor_y_mm"] - 15.0, ld["motor_y_mm"] + 15.0]))
    out = None
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a < 1e-6:
            continue
        seg = _tube(a, b, SPAR["inner_od"], SPAR["inner_id"]) if b <= yp + 1e-6 else _tube(a, b, SPAR["outer_od"], SPAR["outer_id"])
        out = seg if out is None else out.union(seg, clean=False)
    plug = cq.Workplane("XZ").circle(SPAR["inner_id"] / 2).circle(SPAR["outer_id"] / 2).extrude(-SPAR["plug_mm"]).translate((0, yp - SPAR["plug_mm"] / 2, 0))
    return out.union(plug, clean=False), cuts


def _box(name, y0, y1, r):
    from vegeta import talos
    return talos.SurfacesInBox(name, (-r - MARGIN, y0 - MARGIN, -r - MARGIN, r + MARGIN, y1 + MARGIN, r + MARGIN))


def regions(p=None):
    """The root (inside the body: held), the load stations, the motor's station."""
    p = p or params()
    ld = loads(p)
    r = SPAR["inner_od"] / 2
    out = [_box("root", 0.0, p["fuselage_diameter"] / 2, r), _box("motor", ld["motor_y_mm"] - 15.0, ld["motor_y_mm"] + 15.0, r)]
    _, cuts = spar_cad(p)
    for k, (a, b) in enumerate(zip(ld["edges_mm"][:-1], ld["edges_mm"][1:])):
        inside = [(c0, c1) for c0, c1 in zip(cuts[:-1], cuts[1:]) if c0 >= a - 1e-6 and c1 <= b + 1e-6 and not (abs(c0 - (ld["motor_y_mm"] - 15.0)) < 1e-6)]
        c0, c1 = max(inside, key=lambda s: s[1] - s[0])                    # the longest whole segment of the station carries its force
        out.append(_box(f"S{k}", c0, c1, r))
    return out


class FEACase:
    def __init__(self, name, description, model, *, kind="static"):
        self.name, self.description, self.model, self.kind = name, description, model, kind


def export_step(shape, path):
    import cadquery as cq
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cq.exporters.export(shape, str(path))
    return path


def cases(workdir, p=None, *, element_size=3.0):
    from vegeta import talos
    p = p or params()
    shape, _ = spar_cad(p)
    step = export_step(shape, Path(workdir) / "cad" / "hawk2_spar.step")
    reg = regions(p)
    mat = talos.Material(CARBON_TUBE["name"], youngs_modulus=CARBON_TUBE["E"], poissons_ratio=CARBON_TUBE["nu"], density=CARBON_TUBE["rho"],
                         yield_strength=CARBON_TUBE["strength"], source=CARBON_TUBE["source"])
    mesh = talos.MeshSettings(element_size=element_size, order=2)
    sup = [talos.FixedSupport("root")]
    out = []
    for name, relief, text in (("spar_8g", True, "the half wing's lift at 8 g, ultimate, the motor's weight relieving it"),
                               ("spar_motor_out_8g", False, "the same without the motor's relief")):
        ld = loads(p, relief=relief)
        forces = [talos.Force(f"S{k}", fz=f) for k, f in enumerate(ld["lift_N"])]
        if relief:
            forces.append(talos.Force("motor", fz=-ld["motor_relief_N"]))
        out.append(FEACase(name, text, talos.StructuralModel(step, "mm-N-MPa", mat, reg, sup, forces, mesh, name=name)))
    out.append(FEACase("spar_modes", "the spar with the motor, hub, propeller and nacelle as a point mass", talos.StructuralModel(
        step, "mm-N-MPa", mat, reg, sup, [], mesh, masses=[talos.PointMass("motor", MOTOR_KG * 1e-3)], name="spar_modes"), kind="modes"))
    return out


def regions_select(shape, regs) -> dict:
    """How many faces each region's box holds (0: a region that would select nothing), without a mesher."""
    faces = [f.BoundingBox() for f in shape.val().Faces()]
    return {r.name: sum(r.box[0] <= b.xmin and b.xmax <= r.box[3] and r.box[1] <= b.ymin and b.ymax <= r.box[4] and r.box[2] <= b.zmin and b.zmax <= r.box[5] for b in faces)
            for r in regs}


def solve_one(case, workdir, *, run=True, threads=4, n_modes=6):
    wd = Path(workdir) / case.name
    try:
        if case.kind == "static":
            return case.model.ensure(wd, run=run, threads=threads)
        if not run and not (wd / "modes_summary.json").exists():
            return None
        if not (wd / "mesh.msh").exists():
            case.model.mesh(wd)
        return case.model.solve_modes(wd, n_modes=n_modes, threads=threads)
    except Exception as exc:                                              # a failure is a result
        from vegeta.talos.result import Result
        r = Result(kind="talos.static" if case.kind == "static" else "talos.modes", status="failed")
        r.messages.append(f"{type(exc).__name__}: {str(exc)[:200]}")
        return r


def row(case, r) -> dict:
    """One case's result against the allowable and the beam reference."""
    status = "NOT RUN" if r is None else "solved" if getattr(r, "ok", False) else ("NOT RUN" if any("NOT RUN" in m for m in getattr(r, "messages", [])) else "FAILED")
    out = {"kind": case.kind, "status": status, "description": case.description}
    if status != "solved":
        out["note"] = "" if r is None else (r.messages[-1][:200] if r.messages else "")
        return out
    if case.kind == "modes":
        f = [float(x) for x in (r.metrics.get("frequencies_hz") or [])]
        out.update({"frequencies_Hz": [round(x, 1) for x in f[:4]], "beam_reference_Hz": reference()["first_bending_with_the_motor_Hz"]})
    else:
        vm, allow = r.metrics.get("max_von_mises"), CARBON_TUBE["strength"] * KNOCKDOWN
        out.update({"peak_von_mises_MPa": None if vm is None else float(vm), "max_displacement_mm": r.metrics.get("max_displacement"), "allowable_MPa": allow,
                    "margin": None if not vm else round(allow / float(vm) - 1, 2), "beam_reference": {k: reference()[k] for k in ("root", "pivot", "tip_deflection_ultimate_mm")}})
    return out


if __name__ == "__main__":
    import yaml
    print(yaml.safe_dump(reference(), sort_keys=False, allow_unicode=True, width=140))
