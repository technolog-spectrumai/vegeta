"""emerillon_campaigns — the book ``emerillon`` (code E) behind the shared workflow scripts: the same interface as
``kestrel_campaigns`` (variants, the vehicle, the fingerprint, the CFD and FEA campaigns, case builders, outcomes and
summaries), on ``emerillon`` / ``emerillon_flight``. ``side_labs/kestrel/airframe_cfd.py`` and ``structure_fea.py``
take the book by name and import this module for it (``side_labs/emerillon/*.py`` are the one-line entry points).

The chapters are the solver runs notebook 37 leaves NOT RUN:

- **E1 ``airframe-drag``** (CFD ``drag``): the solid aircraft in axial flow at half, three quarters and all of the
  variant's top speed → the drag area without propellers, and the margin it supports on the hand build-up
  (``emerillon_flight.DRAG_MARGIN`` is a guess until this runs).
- **E2 ``airframe-alpha``** (CFD ``alpha``): the body at 0–90° of incidence at 60 % of the top speed → the normal
  force the level-flight trim and the broadside braking lean on (``emerillon_flight.normal_force_area``).
- **E3 ``arm``** (FEA ``arm``): one printed PA6-CF arm with its pod, clamped at its root pad: the static full-throttle
  thrust (A), a 5 g manoeuvre across the arm's thin way with the thrust on (B), a tail-first landing knock on the
  motor (C) — all at ultimate ×1.5 — and its first modes with the motor and propeller as a point mass, read against
  the rotor's once-per-revolution frequency.

The CFD is incompressible RANS (``rans_ksst_external``): right for ER1 (Mach 0.21) and ER2 (Mach 0.31) and for
nothing faster. Nothing here runs a solver; the scripts do, on a machine that has one.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import emerillon as em
import emerillon_flight as ef
import emerillon_market as mk

BOOK = "emerillon"
SCHEMA = 1
VARIANTS = dict(ef.VARIANT_NAME)
DEFAULT_VARIANT = "ER2"
TOLERANCE = {"x_cg_mm": 2.0, "mass_frac": 0.02}
CFD_CAMPAIGNS = {
    "drag": "the solid aircraft in axial flow at 50, 75 and 100 % of the variant's top speed: the drag area and the margin on the hand build-up (3 cases)",
    "alpha": "the body at 0, 3, 6, 10, 15, 30, 60 and 90° of incidence at 60 % of the top speed: the normal force of the trim and of the broadside braking (8 cases)",
}
FEA_CAMPAIGNS = {
    "arm": "one printed arm with its pod, clamped at the root pad: static full-throttle thrust (A), a 5 g manoeuvre across the arm with the thrust on (B), "
           "a tail-first landing knock on the motor (C), ultimate x1.5, and the first modes with the motor and propeller as a point mass",
}
DRAG_FRACTIONS = (0.5, 0.75, 1.0)
ALPHAS = (0.0, 3.0, 6.0, 10.0, 15.0, 30.0, 60.0, 90.0)
ALPHA_SPEED = 0.6
QUALITY = {                                                      # as kestrel_cfd.QUALITY (the shared script reads that table for --quality)
    "screening": dict(cells_per_length=2.0, surface_level=4, near_level=3, wake_level=2, iterations=400, residual_target=1e-4),
    "fine": dict(cells_per_length=3.0, surface_level=5, near_level=4, wake_level=3, iterations=800, residual_target=1e-5),
}
# printed PA6-CF as a solid equivalent, in kestrel_fea's material shape (mm-N-MPa-tonne); assumed until coupons are pulled
PA6CF = dict(name="PA6-CF printed (in the layer plane)", E=3300.0, nu=0.35, rho=1.09e-9, strength=60.0, strength_z=30.0,
             source="assumed: about 60 % of the filament's datasheet stiffness and strength in the plane, half of that across the layers")
ULTIMATE = 1.5
N_MANOEUVRE = 5.0                                                # the manoeuvre the flight simulation asks for (emerillon_mission.ACC_MAX)
LANDING_G = 8.0                                                  # a tail-first knock: the motor unit decelerated at this many g along the axis
_VEH: dict = {}


def params(variant: str) -> dict:
    if variant not in VARIANTS:
        raise ValueError(f"variant {variant!r}: one of {list(VARIANTS)}")
    return ef.cad_parameters(ef.VARIANTS[variant])


def vehicle(variant: str):
    """What the scripts ask of a vehicle: a name, the mass, the centre of gravity — and here the configuration, its
    geometry, its top speed and its static thrust per motor."""
    if variant not in _VEH:
        params(variant)
        c = ef.VARIANTS[variant]
        b = ef.mass_budget(c)
        top = ef.top_speed(c, b.mass)
        motor, prop = mk.part(c.motor), mk.part(c.propeller)
        _VEH[variant] = SimpleNamespace(name=VARIANTS[variant], variant=variant, config=c, mass=b.mass, x_cg=float(b.properties().com[0]), z_cg=0.0,
                                        geo=ef.geometry(c), V_top=float(top["V"]), rpm_top=float(top["rpm"]),
                                        static_thrust=float(ef.drive(c, 0.5)["thrust"]) / 4,
                                        motor_unit_kg=(motor.mass_g + prop.mass_g) / 1e3)
    return _VEH[variant]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def fingerprint(variant: str, veh=None) -> dict:
    veh = veh or vehicle(variant)
    p = params(variant)
    return {"schema": SCHEMA, "name": VARIANTS[variant], "book": BOOK, "variant": variant, "config": veh.config.label(),
            "params_sha256": _sha(json.dumps(p, sort_keys=True)), "params": p, "mass_kg": round(veh.mass, 4), "x_cg_m": round(veh.x_cg, 4),
            "z_cg_m": 0.0, "V_top_m_s": round(veh.V_top, 2), "drag_margin": ef.DRAG_MARGIN, "prop_lab": {"on_file": False},
            "sources": {m: _sha(Path(em.__file__).with_name(m).read_text()) for m in ("emerillon.py", "emerillon_flight.py", "emerillon_market.py")}}


def compare(expected: dict, found: dict | None, *, blown: bool = False) -> dict:
    if not found:
        return {"level": "incompatible", "reasons": ["the run carries no design fingerprint"]}
    if found.get("params_sha256") != expected.get("params_sha256"):
        return {"level": "incompatible", "reasons": ["the CAD parameters differ"]}
    if abs(found.get("mass_kg", 0) - expected.get("mass_kg", 0)) > TOLERANCE["mass_frac"] * expected.get("mass_kg", 1):
        return {"level": "incompatible", "reasons": [f"mass {found.get('mass_kg')} against {expected.get('mass_kg')} kg"]}
    if abs(found.get("x_cg_m", 0) - expected.get("x_cg_m", 0)) * 1e3 > TOLERANCE["x_cg_mm"]:
        return {"level": "incompatible", "reasons": [f"CG {found.get('x_cg_m')} against {expected.get('x_cg_m')} m"]}
    reasons = [f"{m} changed since the run" for m, h in expected.get("sources", {}).items() if found.get("sources", {}).get(m) not in (None, h)]
    return {"level": "warnings" if reasons else "compatible", "reasons": reasons}


def list_footer(veh, kind: str) -> str:
    """The last line of a script's ``--list`` (the shared scripts print Kestrel's own without this hook)."""
    if kind == "fea":
        return (f"vehicle: {veh.name}, {veh.mass:.3f} kg; material {PA6CF['name']} {PA6CF['strength']} MPa / {PA6CF['strength_z']} MPa across layers; "
                f"ultimate x{ULTIMATE}; static thrust {veh.static_thrust:.1f} N per motor, motor unit {veh.motor_unit_kg * 1e3:.0f} g")
    return (f"vehicle: {veh.name} ({veh.config.label()}), {veh.mass:.3f} kg, CG x {veh.x_cg * 1000:.1f} mm, top speed {veh.V_top * 3.6:.0f} km/h "
            f"(Mach {veh.V_top / ef.A_SOUND:.2f}); incompressible RANS, no propellers")


# ------------------------------------------------------------------------------------------------- CFD
def cfd_specs(variant: str, campaign: str, *, veh=None) -> dict:
    if campaign not in CFD_CAMPAIGNS:
        raise ValueError(f"campaign {campaign!r}: one of {list(CFD_CAMPAIGNS)}")
    veh = veh or vehicle(variant)
    base = dict(source=campaign, kind=campaign, variant=variant, T_up=0.0, T_lo=0.0, t=0.0, props=False)
    if campaign == "drag":
        return {f"axial, {f * 100:.0f} % of the top speed, {f * veh.V_top:.1f} m/s": dict(base, alpha_deg=0.0, target_deg=0.0, V=round(f * veh.V_top, 2))
                for f in DRAG_FRACTIONS}
    V = round(ALPHA_SPEED * veh.V_top, 2)
    return {f"α {a:.0f}°, {V:.1f} m/s": dict(base, alpha_deg=a, target_deg=a, V=V) for a in ALPHAS}


def not_run_table(specs: dict):
    """The planned cases with their air (the shared script's ``--list``)."""
    import pandas as pd
    L = ef.geometry(ef.ER2)["length"]
    return pd.DataFrame({n: {"status": "NOT RUN", "α [deg]": s["alpha_deg"], "V [m/s]": s["V"], "Mach": round(s["V"] / ef.A_SOUND, 3),
                             "Re (length)": round(s["V"] * L / ef.NU), "template": "rans_ksst_external", "rotor disks": False} for n, s in specs.items()}).T


def export_stl(workdir, variant: str, alpha_deg: float) -> Path:
    """The solid aircraft pitched nose-up by ``alpha_deg`` about the nose tip (the flow stays along +x), as an STL [mm]."""
    d = Path(workdir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"emerillon_{variant}_a{alpha_deg:04.1f}.stl"
    if not path.exists():
        em.Emerillon().generate(**params(variant), angle_of_attack_deg=float(alpha_deg)).export_stl(path, tolerance=0.05, angular_tolerance=0.1)
    return path


def cfd_case(name: str, spec: dict, workdir, *, variant: str, quality: str, veh, env=None):
    """One ``rans_ksst_external`` case: coefficients on the body's frontal area, the moment about the CG on the body's
    length."""
    from vegeta import aeromant
    a = float(spec["alpha_deg"])
    stl = export_stl(Path(workdir) / "stl", variant, a)
    ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))
    prm = dict(velocity=max(spec["V"], 1.0), kinematic_viscosity=ef.NU, density=ef.RHO, reference_area=veh.geo["frontal"], reference_length=veh.geo["length"],
               center_of_rotation=(veh.x_cg * ca, 0.0, -veh.x_cg * sa), **QUALITY[quality])
    env = env or aeromant.OpenFOAMEnvironment.detect()
    return aeromant.CFDCase("rans_ksst_external", str(stl), prm, workdir=str(Path(workdir) / f"{variant}_a{a:04.1f}_V{spec['V']:.1f}_{quality}"),
                            geometry_units="mm", environment=env)


def cfd_case_key(case, quality: str) -> str:
    import kestrel_campaigns  # noqa: F401  (its _flow() puts side_labs/ on sys.path)
    from runkit import sha256_file
    return _sha(json.dumps({"template": case.template.name, "parameters": case.parameters, "units": case.geometry_units, "quality": quality,
                            "stl": sha256_file(case.geometry)}, sort_keys=True, default=str))


SANE = 50.0                                                      # coefficients on the small frontal area are large at incidence


def cfd_outcome(name: str, spec: dict, result, p=None) -> dict:
    """solved / diverged / FAILED with the coefficients on the frontal area and, in square centimetres, the drag area
    and the normal-force area the flight model uses."""
    ok = getattr(result, "ok", False)
    if result is None or not ok:
        msg = "NOT RUN" if result is None or any("NOT RUN" in m for m in getattr(result, "messages", [])) else "FAILED"
        status = getattr(result, "status", "failed") if result is not None else "failed"
        return {"status": "cancelled" if status == "cancelled" else msg, "process": "cancelled" if status == "cancelled" else ("ok" if ok else "failed"),
                "converged": None, "note": (result.messages[-1][:160] if result is not None and getattr(result, "messages", None) else ""), "metrics": {}, "artifacts": {}}
    m = result.metrics
    cl, cd, cm = (np.nan if m.get(k) is None else float(m.get(k)) for k in ("Cl", "Cd", "Cm"))
    sane = all(np.isfinite(v) and abs(v) <= SANE for v in (cl, cd, cm))
    note = "" if sane else f"diverged: |coefficients| up to {np.nanmax(np.abs([cl, cd, cm])):.3g}"
    if sane and m.get("converged") is False:
        note = f"not converged in {m.get('iterations')} iterations"
    S = ef.geometry(ef.VARIANTS[spec.get("variant", DEFAULT_VARIANT)])["frontal"] if spec.get("variant") else None
    a = math.radians(spec["alpha_deg"])
    metrics = {"CL": cl, "CD": cd, "Cm (CG, length)": cm, "CN": cl * math.cos(a) + cd * math.sin(a), "iterations": m.get("iterations"),
               "converged": m.get("converged"), "Cd scatter (last 50)": m.get("Cd_std_lastN"), "cells": m.get("mesh_cells"), "α [deg]": spec["alpha_deg"],
               "V [m/s]": spec["V"]}
    if S:
        metrics["CdA [cm²]"] = cd * S * 1e4
    metrics = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in metrics.items()}
    arts = {k: str(v) for k, v in (getattr(result, "artifacts", {}) or {}).items() if k in ("case", "summary", "log_solver", "log_snappyHexMesh", "log_checkMesh", "forces")}
    return {"status": "solved" if sane else "diverged", "process": "ok", "converged": m.get("converged"), "note": note, "metrics": metrics, "artifacts": arts}


def cfd_summary(*, variant: str, campaign: str, quality: str, specs: dict, records: dict, veh, run_root: Path, provenance: dict | None = None) -> dict:
    """The run's summary: a row per case, the drag areas against the hand build-up (``drag``: the margin the CFD
    supports) or the normal-force areas against the trim model (``alpha``), a CSV of the solved cases."""
    import pandas as pd
    S, c = veh.geo["frontal"], veh.config
    rows, recs = {}, []
    for name, spec in specs.items():
        r = records.get(name)
        m = (r.metrics if r is not None else {}) or {}
        a, V = float(spec["alpha_deg"]), float(spec["V"])
        row = {"status": r.status if r is not None else "NOT RUN", "process": r.process if r is not None else None, "converged": m.get("converged"),
               "iterations": m.get("iterations"), "cells": m.get("cells"), "alpha_deg": a, "V_m_s": V, "rotor_disks": False, "source": spec.get("source", ""),
               "CL": m.get("CL"), "CD": m.get("CD"), "Cm": m.get("Cm (CG, length)"), "CN": m.get("CN"), "note": r.note if r is not None else "",
               "duration_s": r.duration_s if r is not None else None, "dir": (r.artifacts.get("case") if r is not None else None)}
        clean = ef.drag_buildup(c, V, margin=1.0)["CdA"]
        row["CdA_buildup_clean_cm2"] = clean * 1e4
        row["CNA_model_cm2"] = ef.normal_force_area(c, math.radians(min(a, 89.0))) * 1e4
        if row["status"] == "solved" and row["CD"] is not None:
            row["CdA_cm2"], row["CNA_cm2"] = float(row["CD"]) * S * 1e4, float(row["CN"]) * S * 1e4
            if a == 0.0:
                row["margin_supported"] = float(row["CD"]) * S / clean
            recs.append({"case": name, "alpha_deg": a, "V": V, "CL": float(row["CL"]), "CD": float(row["CD"]), "Cm": float(row["Cm"]), "CN": float(row["CN"]),
                         "CdA_cm2": row["CdA_cm2"], "CNA_cm2": row["CNA_cm2"], "converged": row["converged"], "iterations": row["iterations"]})
        rows[name] = row
    df = pd.DataFrame(rows).T if rows else pd.DataFrame()
    maps = {}
    solved = df[df["status"] == "solved"] if len(df) else df
    if len(solved):
        path = run_root / f"{campaign}.csv"
        cols = [k for k in ("alpha_deg", "V_m_s", "CL", "CD", "Cm", "CN", "CdA_cm2", "CNA_cm2", "CdA_buildup_clean_cm2", "CNA_model_cm2", "converged", "iterations", "cells") if k in solved]
        solved[cols].sort_values(["alpha_deg", "V_m_s"]).to_csv(path, index_label="case")
        maps[campaign] = {"file": path.name, "columns": cols, "valid_domain": {"alpha_deg": [float(solved["alpha_deg"].min()), float(solved["alpha_deg"].max())],
                                                                              "V_m_s": [float(solved["V_m_s"].min()), float(solved["V_m_s"].max())], "n": int(len(solved))}}
    counts = {k: int((df["status"] == k).sum()) if len(df) else 0 for k in ("solved", "diverged", "FAILED", "cancelled", "NOT RUN")}
    counts["n"] = int(len(df))
    counts["converged"] = int(df["converged"].fillna(False).astype(bool).sum()) if len(df) else 0
    margins = [r["margin_supported"] for r in rows.values() if r.get("margin_supported")]
    finding = {}
    if margins:
        finding = {"margin_supported_mean": float(np.mean(margins)), "margin_assumed": ef.DRAG_MARGIN,
                   "reading": "CdA(CFD, no propellers) over the clean build-up; the assumed margin also covers the propellers' wash, cooling flow and what a print adds"}
    return {"schema": SCHEMA, "workflow": "airframe_cfd", "book": BOOK, "campaign": campaign, "description": CFD_CAMPAIGNS.get(campaign, ""), "variant": variant,
            "quality": quality, "quality_settings": QUALITY.get(quality), "design": fingerprint(variant, veh),
            "reference": {"S_ref_m2": S, "l_ref_m": veh.geo["length"], "x_cg_m": veh.x_cg, "density_kg_m3": ef.RHO, "kinematic_viscosity_m2_s": ef.NU,
                          "frame": "wind axes: CD along the flow, CL up; coefficients on the body's frontal area; Cm about the CG on the body's length; "
                                   "CN = CL cos α + CD sin α is the force normal to the body axis"},
            "counts": counts, "cases": rows, "records": recs, "maps": maps, "finding": finding, "provenance": provenance or {"source": "solver run"},
            "validation_note": "incompressible RANS without propellers; a case marked converged reached the residual target; a solved, unconverged case is indicative"}


# ------------------------------------------------------------------------------------------------- FEA
def arm_regions(p: dict) -> dict:
    """The arm part's two boxes [mm] (x0, y0, z0, x1, y1, z1): the root pad's inner face (clamped) and the pod's flat
    motor face (loaded)."""
    p = em.resolve(p)
    L = em.Emerillon.layout(p)
    r = p["pod_diameter"] / 2 + 1.0
    return {"root": (L["x_arm"] - 0.2 * p["arm_chord"], L["R"] - 0.7, -1.3 * p["arm_thickness"], L["x_arm"] + 1.2 * p["arm_chord"], L["R"] - 0.3, 1.3 * p["arm_thickness"]),
            "mount": (L["x_mount"] - 0.2, p["arm_radius"] - r, -r, L["x_mount"] + 0.2, p["arm_radius"] + r, r)}


def arm_step(workdir, variant: str) -> Path:
    d = Path(workdir) / "cad"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"emerillon_{variant}_arm.step"
    if not path.exists():
        em.Emerillon().generate(part="arm", **params(variant)).export_step(path)
    return path


def arm_beam(variant: str, veh=None) -> dict:
    """The arm as a cantilever, for the summary's hand check: the thrust bends it in its chord plane (the stiff way),
    the manoeuvre across its thickness (the thin way). Section moduli of a NACA 00xx-like solid: I ≈ 0.036 c t³ thin
    way, 0.044 t c³ stiff way (an ellipse's are 0.049)."""
    veh = veh or vehicle(variant)
    p = em.resolve(params(variant))
    c, t = p["arm_chord"], p["arm_thickness"]
    L = em.Emerillon.layout(p)
    span = p["arm_radius"] - L["R"]
    i_thin, i_stiff = 0.036 * c * t ** 3, 0.044 * t * c ** 3
    f_thrust = ULTIMATE * veh.static_thrust
    f_side = ULTIMATE * N_MANOEUVRE * ef.G * veh.motor_unit_kg
    return {"span_mm": span, "I_thin_mm4": i_thin, "I_stiff_mm4": i_stiff, "thrust_N": f_thrust, "side_N": f_side,
            "stress_thrust_MPa": f_thrust * span * (c / 2) / i_stiff, "stress_side_MPa": f_side * span * (t / 2) / i_thin,
            "deflection_side_mm": f_side * span ** 3 / (3 * PA6CF["E"] * i_thin),
            "f1_thin_Hz": math.sqrt(3 * PA6CF["E"] * i_thin / (veh.motor_unit_kg * 1e-3 * span ** 3)) / (2 * math.pi),
            "rotor_1P_Hz": veh.rpm_top / 60.0}


def fea_cases(variant: str, campaign: str, workdir, *, veh=None, element_size: float = 2.0) -> list:
    if campaign not in FEA_CAMPAIGNS:
        raise ValueError(f"campaign {campaign!r}: one of {list(FEA_CAMPAIGNS)}")
    import kestrel_fea as kfe
    from vegeta import talos
    veh = veh or vehicle(variant)
    p = em.resolve(params(variant))
    step = arm_step(workdir, variant)
    boxes = arm_regions(p)
    regions = [talos.SurfacesInBox("root", boxes["root"]), talos.SurfacesInBox("mount", boxes["mount"])]
    mat = kfe.talos_material(PA6CF)
    mesh = talos.MeshSettings(element_size=element_size, order=2)
    sup = [talos.FixedSupport("root")]
    T = ULTIMATE * veh.static_thrust
    side = ULTIMATE * N_MANOEUVRE * ef.G * veh.motor_unit_kg
    knock = ULTIMATE * LANDING_G * ef.G * veh.motor_unit_kg
    L = em.Emerillon.layout(p)

    def model(name, loads, masses=()):
        return talos.StructuralModel(step, "mm-N-MPa", mat, regions, sup, loads, mesh, masses=list(masses), name=name)

    kw = dict(strength=PA6CF["strength"], strength_z=PA6CF["strength_z"], build_axis="z", bracing="none")
    out = [kfe.FEACase("arm_A", "arm", f"static full-throttle thrust {veh.static_thrust:.1f} N on the motor face (ultimate x{ULTIMATE})",
                       model("arm_A", [talos.Force("mount", fx=-T)]), **kw),
           kfe.FEACase("arm_B", "arm", f"{N_MANOEUVRE:g} g across the arm's thin way on the {veh.motor_unit_kg * 1e3:.0f} g motor unit, the thrust on (ultimate x{ULTIMATE})",
                       model("arm_B", [talos.Force("mount", fx=-T, fz=side)]), **kw),
           kfe.FEACase("arm_C", "arm", f"tail-first landing knock: the motor unit at {LANDING_G:g} g along the axis, no thrust (ultimate x{ULTIMATE})",
                       model("arm_C", [talos.Force("mount", fx=knock)]), **kw),
           kfe.FEACase("arm_modes", "arm", f"the first modes with the motor and propeller as a point mass (rotor 1P at top speed: {veh.rpm_top / 60:.0f} Hz)",
                       model("arm_modes", [], [talos.PointMass("mount", veh.motor_unit_kg * 1e-3)]), kind="modes", bracing="none")]
    y_root, x_mount, y_pod = L["R"] + p["pad_thickness"] + 3.0, L["x_mount"] + 4.0, p["arm_radius"] - p["pod_diameter"] / 2 - 1.0
    for c in out:                                                # the nominal stress is read away from the clamp and the loaded face
        c.exclude = lambda xyz, y_root=y_root, x_mount=x_mount, y_pod=y_pod: (xyz[:, 1] < y_root) | ((xyz[:, 0] < x_mount) & (xyz[:, 1] > y_pod))
    return out


def fea_case_key(case) -> str:
    return _sha(json.dumps({"model": case.model.key, "kind": case.kind}, sort_keys=True))


def fea_outcome(case, result) -> dict:
    import kestrel_campaigns as kc
    return kc.fea_outcome(case, result)


def fea_summary(*, variant: str, campaign: str, cases: list, records: dict, veh, run_root: Path) -> dict:
    rows = {}
    for c in cases:
        r = records.get(c.name)
        m = (r.metrics if r is not None else {}) or {}
        rows[c.name] = {"status": r.status if r is not None else "NOT RUN", "process": r.process if r is not None else None, "part": c.part, "kind": c.kind,
                        "description": c.description, "strength_MPa": c.strength, "strength_z_MPa": c.strength_z,
                        **{k: m.get(k) for k in ("peak von Mises [MPa]", "nominal von Mises [MPa]", "99.5th percentile [MPa]", "max displacement [mm]", "allowable [MPa]",
                                                 "margin", "reaction [N]", "across layers [MPa]", "margin (layers)", "f1 [Hz]", "f2 [Hz]", "f3 [Hz]")},
                        "note": r.note if r is not None else "", "duration_s": r.duration_s if r is not None else None,
                        "artifacts": r.artifacts if r is not None else {}}
    beam = arm_beam(variant, veh)
    counts = {k: sum(1 for r in rows.values() if r["status"] == k) for k in ("solved", "FAILED", "NOT RUN", "cancelled")}
    counts["n"] = len(rows)
    f1 = rows.get("arm_modes", {}).get("f1 [Hz]")
    return {"schema": SCHEMA, "workflow": "structure_fea", "book": BOOK, "campaign": campaign, "description": FEA_CAMPAIGNS.get(campaign, ""), "variant": variant,
            "design": fingerprint(variant, veh), "materials": {"pa6cf": PA6CF, "ultimate": ULTIMATE},
            "reference": {"units": "mm-N-MPa (Talos); frequencies Hz", "frame": "the aircraft's: x aft, the arm along +y, z across its thickness; the root pad's inner face fixed",
                          "loads": {"static_thrust_N": veh.static_thrust, "motor_unit_kg": veh.motor_unit_kg, "manoeuvre_g": N_MANOEUVRE, "landing_g": LANDING_G}},
            "counts": counts, "cases": rows, "beam_reference": {k: float(v) for k, v in beam.items()},
            "resonance": {"rotor_1P_Hz": beam["rotor_1P_Hz"], "f1_Hz": f1, "f1_over_1P": (f1 / beam["rotor_1P_Hz"]) if f1 else None,
                          "reading": "the first arm mode should stay clear of the rotor's once-per-revolution frequency over the throttle range (a Campbell check, not done here)"},
            "validation_note": "a static case that solved is a linear elastic solution on the given mesh with assumed printed properties, not a validated strength; "
                               "the beam rule's stresses stand beside the FEA's nominal"}
