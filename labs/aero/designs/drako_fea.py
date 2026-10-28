"""Drako — the arms and the frame in Talos (CalculiX FEA). Prepared here, solved where CalculiX and Gmsh are installed.

Cases (``cases``), loads ultimate (x ``ULTIMATE``) on the 15 kg maximum take-off mass:

- **arm_full_thrust**: one arm clamped over its length inside the plate sandwich, the motor mount carrying the motor's
  full catalogue thrust (13 kgf) up and the motor's reaction torque, less the motor's weight.
- **arm_modes**: the same arm with the motor and propeller as a point mass on the mount — read against the rotor's
  1/rev and 2/rev (blade passing) at hover (``rotor_rpm``).
- **frame_manoeuvre**: plates + six arms, the lower plate's underside held (where battery, cradle and package hang),
  every mount at the thrust of a 2 g pull-up at the maximum mass.
- **frame_motor_out**: motor 0 dead, its opposite motor throttled down with it for yaw trim, the other four at full
  thrust (the worst asymmetric case an arm and the plates see).
- **frame_modes**: the frame with all six motors as point masses.

Every case is checked against the cantilever beam (``arm_reference``): root stress M c / I and tip deflection
F L^3 / 3 E I. Units mm-N-MPa, masses in tonnes. ``solve(run=False)`` reads what exists and marks the rest NOT RUN;
under ``vegeta.mock`` everything returns at once with fixed numbers (the numbers mean nothing).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from drako import Drako

G = 9.81
ULTIMATE = 1.5
KNOCKDOWN = 0.8                   # carbon: holes, clamps, impact damage
CARBON_TUBE = dict(name="roll-wrapped carbon tube", E=70000.0, nu=0.30, rho=1.55e-9, strength=500.0,
                   source="roll-wrapped T700 tube class, quasi-axial layup (assumed; verify with the supplier)")
CARBON_PLATE = dict(name="carbon plate 0/90", E=50000.0, nu=0.30, rho=1.55e-9, strength=400.0,
                    source="woven 3K carbon sheet class (assumed)")
MOTOR_MASS = 0.95 + 0.15          # kg: motor + ESC + propeller
MOTOR_TORQUE = 2.5                # N m at full thrust (30-inch, 13 kgf class; catalogue order of magnitude)
MARGIN = 3.0                      # mm around region boxes


def params(**over):
    p = {q.name: q.default for q in Drako.parameters}
    p.update(over)
    return p


def talos_material(m):
    from vegeta import talos
    return talos.Material(m["name"], youngs_modulus=m["E"], poissons_ratio=m["nu"], density=m["rho"],
                          yield_strength=m["strength"], source=m["source"])


class FEACase:
    def __init__(self, name, part, description, model, *, kind="static", strength=0.0):
        self.name, self.part, self.description, self.model, self.kind, self.strength = name, part, description, model, kind, strength


# ------------------------------------------------------------------------------------------- reference numbers
def arm_geometry(p):
    R, rh = p["wheelbase"] / 2, p["plate_diameter"] / 2
    r_in = 0.55 * rh
    return dict(root=rh, r_in=r_in, length=R - rh, z=p["plate_thickness"] + p["plate_gap"] / 2)


def arm_reference(p=None, thrust_kgf=13.0):
    """Cantilever tube from the plate edge to the motor, ultimate thrust less the motor weight at the tip."""
    p = p or params()
    a = arm_geometry(p)
    ro, ri = p["arm_diameter"] / 2, p["arm_diameter"] / 2 - p["arm_wall"]
    I = math.pi / 4 * (ro ** 4 - ri ** 4)
    F = ULTIMATE * (thrust_kgf - MOTOR_MASS) * G
    M = F * a["length"]
    tau = ULTIMATE * MOTOR_TORQUE * 1e3 * ro / (2 * I)               # torsion: J = 2 I
    sigma = M * ro / I
    E = CARBON_TUBE["E"]
    k = 3 * E * I / a["length"] ** 3                                 # N/mm
    m_tip = MOTOR_MASS * 1e-3                                        # t
    f1 = math.sqrt(k / m_tip) / (2 * math.pi)
    return {"length [mm]": a["length"], "tip force, ultimate [N]": F, "root bending stress [MPa]": sigma,
            "torsion shear [MPa]": tau, "von Mises [MPa]": math.sqrt(sigma ** 2 + 3 * tau ** 2),
            "tip deflection [mm]": F / k, "allowable [MPa]": CARBON_TUBE["strength"] * KNOCKDOWN,
            "margin": CARBON_TUBE["strength"] * KNOCKDOWN / math.sqrt(sigma ** 2 + 3 * tau ** 2) - 1,
            "f1 bending, motor as point mass [Hz]": f1}


def rotor_rpm(p=None, ct=0.11):
    """Hover rpm at the design and maximum mass from T = Ct rho n^2 D^4 (Ct of a 30-inch two-blade prop)."""
    p = p or params()
    s = Drako.sizing(p)
    D = p["prop_diameter"] * 0.0254
    out = {}
    for key, m in (("design", s["tow_design_kg"]), ("max", s["mtow_max_kg"])):
        n = math.sqrt(m * G / 6 / (ct * 1.225 * D ** 4))
        out[key] = {"rpm": 60 * n, "1P [Hz]": n, "2P [Hz]": 2 * n}
    return out


# ------------------------------------------------------------------------------------------- the CAD
def arm_cad(p=None):
    p = p or params(part="arm")
    return Drako().build({**p, "part": "arm"})


def frame_cad(p=None):
    p = p or params(part="frame")
    return Drako().build({**p, "part": "frame"})


def export_step(shape, path):
    import cadquery as cq
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cq.exporters.export(shape, str(path))
    return path


def _mount_box(p, angle_deg, name):
    from vegeta import talos
    R, a = p["wheelbase"] / 2, arm_geometry(p)
    rm, ztop = p["mount_diameter"] / 2, a["z"] + p["arm_diameter"] / 2 + p["mount_thickness"]
    c, s = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    x, y = R * c, R * s
    return talos.SurfacesInBox(name, (x - rm - MARGIN, y - rm - MARGIN, ztop - MARGIN, x + rm + MARGIN, y + rm + MARGIN, ztop + MARGIN))


# ------------------------------------------------------------------------------------------- the cases
def arm_cases(workdir, p=None, *, step=None, element_size=4.0):
    from vegeta import talos
    p = p or params()
    step = Path(step) if step else export_step(arm_cad(p), Path(workdir) / "cad" / "drako_arm.step")
    a, ra = arm_geometry(p), p["arm_diameter"] / 2
    clamp = talos.SurfacesInBox("clamp", (a["r_in"] - MARGIN, -ra - MARGIN, a["z"] - ra - MARGIN,
                                          a["root"] + MARGIN, ra + MARGIN, a["z"] + ra + MARGIN))
    regions = [clamp, _mount_box(p, 0.0, "mount")]
    mat, mesh = talos_material(CARBON_TUBE), talos.MeshSettings(element_size=element_size, order=2)
    F = ULTIMATE * (13.0 - MOTOR_MASS) * G
    # the reaction torque as a couple on the mount face: +-fy over the mount's diameter is not expressible as one
    # Force on one region, so it is carried in the reference (arm_reference) and the case takes the thrust
    static = talos.StructuralModel(step, "mm-N-MPa", mat, regions, [talos.FixedSupport("clamp")],
                                   [talos.Force("mount", fz=F)], mesh, name="arm_full_thrust")
    modes = talos.StructuralModel(step, "mm-N-MPa", mat, regions, [talos.FixedSupport("clamp")], [], mesh,
                                  masses=[talos.PointMass("mount", MOTOR_MASS * 1e-3)], name="arm_modes")
    return [FEACase("arm_full_thrust", "arm", "13 kgf thrust at the mount, ultimate, arm clamped in the plates", static,
                    strength=CARBON_TUBE["strength"]),
            FEACase("arm_modes", "arm", "arm with motor + propeller as a point mass", modes, kind="modes")]


def frame_cases(workdir, p=None, *, step=None, element_size=6.0):
    from vegeta import talos
    p = p or params()
    step = Path(step) if step else export_step(frame_cad(p), Path(workdir) / "cad" / "drako_frame.step")
    regions = [talos.SurfacesOnPlane("plate_under", "z", 0.0)] + [_mount_box(p, 60 * k, f"M{k}") for k in range(6)]
    mat, mesh = talos_material(CARBON_TUBE), talos.MeshSettings(element_size=element_size, order=2)
    mmax = Drako.sizing(p)["mtow_max_kg"]
    t_pull = 2.0 * mmax * G / 6                                       # 2 g pull-up, per motor
    t_full = 13.0 * G
    w_motor = MOTOR_MASS * G
    sup = [talos.FixedSupport("plate_under")]
    man = [talos.Force(f"M{k}", fz=ULTIMATE * (t_pull - 2 * w_motor)) for k in range(6)]   # 2 g on the motor too
    out_ = {0: 0.0, 3: 0.3 * t_full}
    mo = [talos.Force(f"M{k}", fz=ULTIMATE * (out_.get(k, t_full) - w_motor)) for k in range(6)]
    masses = [talos.PointMass(f"M{k}", MOTOR_MASS * 1e-3) for k in range(6)]
    return [FEACase("frame_manoeuvre", "frame", f"2 g pull-up at {mmax:.1f} kg, ultimate", talos.StructuralModel(
                step, "mm-N-MPa", mat, regions, sup, man, mesh, name="frame_manoeuvre"), strength=CARBON_TUBE["strength"]),
            FEACase("frame_motor_out", "frame", "M0 out, M3 at 30 %, four at full thrust, ultimate", talos.StructuralModel(
                step, "mm-N-MPa", mat, regions, sup, mo, mesh, name="frame_motor_out"), strength=CARBON_TUBE["strength"]),
            FEACase("frame_modes", "frame", "frame with six motors as point masses", talos.StructuralModel(
                step, "mm-N-MPa", mat, regions, sup, [], mesh, masses=masses, name="frame_modes"), kind="modes")]


def cases(workdir, p=None, *, arm_step=None, frame_step=None):
    return arm_cases(workdir, p, step=arm_step) + frame_cases(workdir, p, step=frame_step)


# ------------------------------------------------------------------------------------------- solve and read
def solve(cases_, workdir, *, run=True, threads=4, n_modes=6):
    out = {}
    for c in cases_:
        wd = Path(workdir) / c.name
        try:
            if c.kind == "static":
                out[c.name] = c.model.ensure(wd, run=run, threads=threads)
            else:
                if not run and not (wd / "modes_summary.json").exists():
                    out[c.name] = None
                    continue
                if not (wd / "mesh.msh").exists():
                    c.model.mesh(wd)
                out[c.name] = c.model.solve_modes(wd, n_modes=n_modes, threads=threads)
        except Exception as exc:                          # a failure is a result
            from vegeta.talos.result import Result
            r = Result(kind="talos.static" if c.kind == "static" else "talos.modes", status="failed")
            r.messages.append(f"{type(exc).__name__}: {str(exc)[:200]}")
            out[c.name] = r
    return out


def _status(r):
    if r is None:
        return "NOT RUN"
    if getattr(r, "ok", False):
        return "solved"
    return "NOT RUN" if any("NOT RUN" in m for m in getattr(r, "messages", [])) else "FAILED"


def table(cases_, results, p=None):
    """One row per case: status, peak von Mises, displacement, margin on the knocked-down allowable; for modal
    cases the first frequencies and the nearest rotor harmonic at hover."""
    import pandas as pd
    rpm = rotor_rpm(p)
    harmonics = [rpm[k][h] for k in ("design", "max") for h in ("1P [Hz]", "2P [Hz]")]
    rows = {}
    for c in cases_:
        r = results.get(c.name)
        row = {"part": c.part, "kind": c.kind, "status": _status(r), "description": c.description}
        if row["status"] != "solved":
            row["note"] = "" if r is None else (r.messages[-1][:160] if r.messages else "")
        elif c.kind == "modes":
            f = list(r.metrics.get("frequencies_hz") or [])
            row.update({"f1 [Hz]": f[0] if f else np.nan, "f2 [Hz]": f[1] if len(f) > 1 else np.nan,
                        "closest rotor harmonic [Hz]": min(harmonics, key=lambda h: abs(h - f[0])) if f else np.nan})
            if f:
                row["separation"] = abs(f[0] - row["closest rotor harmonic [Hz]"]) / row["closest rotor harmonic [Hz]"]
        else:
            vm = r.metrics.get("max_von_mises")
            allow = c.strength * KNOCKDOWN
            row.update({"peak von Mises [MPa]": vm, "max displacement [mm]": r.metrics.get("max_displacement"),
                        "allowable [MPa]": allow, "margin": allow / vm - 1 if vm else np.nan})
        rows[c.name] = row
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    for k, v in arm_reference().items():
        print(f"{k:40s} {v:10.2f}")
    print(rotor_rpm())
