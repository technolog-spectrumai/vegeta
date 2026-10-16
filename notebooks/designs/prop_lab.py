"""prop_lab — one propeller alone in CFD, in the configurations Kestrel flies (notebook 41), and the calibration it
hands to the Kestrel notebooks.

Kestrel's airframe CFD runs its four propellers as **rotor disks** (blade-element sources fed with Boreas's blade rows
and section polar); its flight model runs on Boreas's BEMT drive tables and De Young's propeller normal force. All of
it rests on the generic thin-airfoil section Boreas assumes for a 9-inch blade. prop_lab measures the propeller
itself — the 9×6 resolved, blades and hub, in a rotating frame (Aeromant's ``rotor_mrf_static`` / ``rotor_mrf`` /
``rotor_mrf_oblique``) — at:

- the **hover** (static thrust at three speeds, the hover rpm among them),
- **axial inflow** 4–24 m/s (the cruise, the climb, the maximum-speed end: the advance-ratio sweep),
- **inflow at an angle** to the axis, 15–90° (the transition: the propeller at an angle of attack gives a normal
  force and a pitching moment of its own, and its thrust falls off differently from the axial case).

Each case is read beside the blade-element prediction at the same point. The **calibration** is two numbers for the
blade section — the lift-slope scale ``cl_scale`` and the drag scale ``cd_scale`` that make BEMT match the solved
axial and static cases in thrust and torque at once — and one for the normal force, ``cn_factor`` (the CFD's
normal-force slope over De Young's, over the oblique cases). ``export`` writes them with the whole case table to
``data/prop_lab.json``; ``kestrel_systems.propulsion`` applies the section scales (so the drive tables, ``rpm_for``
and the rotor disks' polar tables carry the measured propeller), ``kestrel_flight.prop_cn_alpha`` the normal-force
factor. A failed or diverged case is a result: listed, counted, left out of the fit.

Frame: the rotor axis is +x, the inflow comes along +x (axial) or in the x–z plane at ``inflow_angle`` (oblique:
the cross-flow along +z); the CAD propeller (``cad``) is built about z and turned onto x (``stl``). Steady MRF: no
blade passing, no tip vortex; a steady frame averages the once-per-revolution loading of an inclined inflow.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import kestrel_systems as ks

PROP_LAB_JSON = Path(__file__).resolve().parent / "data" / "prop_lab.json"
RHO, NU = ks.RHO0, 1.46e-5
QUALITY = {
    "screening": dict(cells_per_diameter=6.0, surface_level=3, near_level=2, rotor_level=2, wake_level=1, iterations=400, residual_target=1e-4),
    "fine": dict(cells_per_diameter=8.0, surface_level=4, near_level=3, rotor_level=3, wake_level=2, iterations=800, residual_target=1e-5),
}
RPM_HOVER_REF = 4.15             # N: one rotor's share of the 1.69 kg aircraft in the hover (the rpm for it comes from Boreas)
T_CRUISE_REF = 0.8               # N: one rotor's cruise thrust at 16 m/s with round struts


# ------------------------------------------------------------------------------------------------- the propeller
def geometry() -> dict:
    """The 9×6's CAD numbers from ``kestrel_systems.COMPONENTS`` (mm): the planform Boreas uses, a 16 × 10 mm hub."""
    c = ks.COMPONENTS["propeller"]
    return dict(diameter=c["diameter_in"] * 25.4, pitch=c["pitch_in"] * 25.4, blades=c["blades"], hub_diameter=16.0, hub_height=10.0, bore=5.0,
                chord_root=c["chord_root_m"] * 1000, chord_max=c["chord_max_m"] * 1000, chord_tip=c["chord_tip_m"] * 1000,
                thickness=0.10, camber=0.04, stations=12)


def cad():
    """Dedalus's parametric propeller with the 9×6's numbers (axis z, blade 1 along +x)."""
    from vegeta.dedalus.examples import Propeller
    return Propeller().generate(**geometry())


def stl(workdir, tolerance: float = 0.05) -> Path:
    """The propeller turned onto the rotor frame's axis (+x) and written as STL (mm), once per workdir."""
    from vegeta import dedalus
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    path = workdir / "prop_9x6_axis_x.stl"
    if not path.exists():
        g = cad()
        gx = dedalus.Geometry.from_cadquery(g.shape.rotate((0, 0, 0), (0, 1, 0), 90), name="prop_axis_x")
        gx.export_stl(str(path), tolerance=tolerance)
    return path


# ------------------------------------------------------------------------------------------------- blade element
def airfoil(calibrated: bool = False):
    """Boreas's section for the blade: the generic one, or the one prop_lab calibrated (``kestrel_systems.blade_airfoil``)."""
    return ks.blade_airfoil(calibrated=calibrated)


def bemt(rpm: float, V: float = 0.0, *, calibrated: bool = False):
    """Boreas's operating point of the 9×6 at ``rpm`` and axial inflow ``V``."""
    from vegeta import boreas
    sys_ = ks.propulsion()
    return boreas.bemt.solve(sys_.prop, airfoil(calibrated), rpm, V, RHO)


def rpm_hover() -> float:
    return ks.rpm_for(RPM_HOVER_REF, 0.0)


def rpm_cruise() -> float:
    return ks.rpm_for(T_CRUISE_REF, 16.0)


def deyoung_cn_alpha(thrust: float, V: float) -> float:
    """De Young's normal-force slope dC_N/dα on q·A, as the transition model uses it (``kestrel_flight.prop_cn_alpha``
    without prop_lab's factor)."""
    import kestrel_flight as kf
    return kf.prop_cn_alpha(thrust, V, calibrated=False)


# ------------------------------------------------------------------------------------------------- the configurations
def configurations(*, rpm_main: float = 8000.0, rpm_high: float = 10000.0) -> dict:
    """The cases by name: the hover at three speeds (the hover rpm among them), the axial sweep at ``rpm_main`` with
    two points at ``rpm_high`` and the cruise point at its own rpm, the oblique cases at ``rpm_main`` at two speeds."""
    out = {}
    r_h, r_c = rpm_hover(), rpm_cruise()
    for rpm, lab in ((5000.0, "5000 rpm"), (r_h, f"{r_h:.0f} rpm (the hover)"), (9000.0, "9000 rpm")):
        out[f"hover, {lab}"] = {"kind": "hover", "template": "rotor_mrf_static", "rpm": rpm, "airspeed": 0.0, "inflow_angle": 0.0}
    for V in (4.0, 8.0, 12.0, 16.0, 20.0, 24.0):
        out[f"axial, {V:g} m/s, {rpm_main:.0f} rpm"] = {"kind": "axial", "template": "rotor_mrf", "rpm": rpm_main, "airspeed": V, "inflow_angle": 0.0}
    for V in (12.0, 20.0):
        out[f"axial, {V:g} m/s, {rpm_high:.0f} rpm"] = {"kind": "axial", "template": "rotor_mrf", "rpm": rpm_high, "airspeed": V, "inflow_angle": 0.0}
    out[f"cruise, 16 m/s, {r_c:.0f} rpm"] = {"kind": "axial", "template": "rotor_mrf", "rpm": r_c, "airspeed": 16.0, "inflow_angle": 0.0}
    for V, angles in ((8.0, (15.0, 30.0, 45.0, 60.0, 90.0)), (14.0, (15.0, 30.0, 45.0, 60.0))):
        for a in angles:
            out[f"oblique, {V:g} m/s at {a:g}°, {rpm_main:.0f} rpm"] = {"kind": "oblique", "template": "rotor_mrf_oblique", "rpm": rpm_main, "airspeed": V, "inflow_angle": a}
    return out


def case(workdir, name: str, spec: dict, *, quality: str = "screening", env=None):
    """One Aeromant case (prepared, not run)."""
    from vegeta import aeromant
    pr = ks.propulsion().prop
    params = dict(rpm=spec["rpm"], diameter=pr.diameter, kinematic_viscosity=NU, density=RHO, rotation=1, **QUALITY[quality])
    if spec["template"] == "rotor_mrf":
        params["airspeed"] = spec["airspeed"]
    elif spec["template"] == "rotor_mrf_oblique":
        params.update(airspeed=spec["airspeed"], inflow_angle=spec["inflow_angle"])
    env = env or aeromant.OpenFOAMEnvironment.detect()
    tag = name.replace(", ", "_").replace(" ", "").replace("°", "deg").replace("(", "").replace(")", "")
    return aeromant.CFDCase(spec["template"], str(stl(Path(workdir) / "stl")), params, workdir=str(Path(workdir) / f"{tag}_{quality}"),
                            geometry_units="mm", environment=env)


def plan(workdir, specs: dict | None = None, *, quality: str = "screening", env=None) -> dict:
    specs = specs or configurations()
    return {n: case(workdir, n, s, quality=quality, env=env) for n, s in specs.items()}


# ------------------------------------------------------------------------------------------------- the table
SANE = 1e3                       # N, N·m: a thrust or torque beyond this from a 9-inch propeller is a diverged solver


def _note(r) -> str:
    msgs = [m for m in getattr(r, "messages", []) if m]
    return msgs[-1][:160] if msgs else ""


def table(results: dict | None, specs: dict | None = None) -> "pd.DataFrame":
    """One row per configuration: the point, the status (``solved`` / ``diverged`` / ``FAILED`` / ``NOT RUN``), the
    CFD's thrust, torque, shaft power, efficiency or figure of merit, normal force and pitching moment, beside the
    blade-element prediction at the same point (thrust, torque, power; De Young's C_Nα for the oblique cases) and
    the ratios. A failure is a result."""
    import pandas as pd
    specs = specs or configurations()
    results = results or {}
    pr = ks.propulsion().prop
    D = pr.diameter
    rows = {}
    for name, s in specs.items():
        V, a = s["airspeed"], math.radians(s["inflow_angle"])
        v_ax = V * math.cos(a)
        op = bemt(s["rpm"], v_ax)
        n = s["rpm"] / 60
        q = 0.5 * RHO * max(V, 0.1) ** 2
        A = math.pi * (D / 2) ** 2
        row = {"kind": s["kind"], "template": s["template"], "rpm": s["rpm"], "V [m/s]": V, "angle [deg]": s["inflow_angle"], "J": v_ax / (n * D),
               "Re (0.75 R)": 0.75 * D / 2 * 2 * math.pi * n * 0.9 * pr.chord[len(pr.chord) * 3 // 4] / NU,
               "BEMT T [N]": op.thrust, "BEMT Q [N·m]": op.torque, "BEMT P [W]": op.power,
               "BEMT η / FM": op.efficiency if V > 0 else op.figure_of_merit,
               "De Young CNα [1/rad]": deyoung_cn_alpha(op.thrust, V) if s["kind"] == "oblique" else np.nan}
        r = results.get(name)
        if r is None or not getattr(r, "ok", False):
            msg = "NOT RUN" if r is None or any("NOT RUN" in m for m in getattr(r, "messages", [])) else "FAILED"
            rows[name] = dict(row, status=msg, note="" if r is None else _note(r))
            continue
        m = r.metrics
        T, Q, P = (np.nan if m.get(k) is None else float(m[k]) for k in ("thrust_N", "torque_Nm", "power_W"))
        sane = all(np.isfinite(x) for x in (T, Q)) and abs(T) <= SANE and abs(Q) <= SANE
        note = "" if sane else f"diverged: |T| {abs(T):.3g} N, |Q| {abs(Q):.3g} N·m"
        if sane and m.get("converged") is False:
            note = f"not converged in {m.get('iterations')} iterations"
        N = float(m["normal_force_N"]) if m.get("normal_force_N") is not None else np.nan
        My = float(m["pitching_moment_Nm"]) if m.get("pitching_moment_Nm") is not None else np.nan
        cn_alpha = N / (q * A) / math.sin(a) if s["kind"] == "oblique" and a > 0 and np.isfinite(N) else np.nan
        rows[name] = dict(row, status="solved" if sane else "diverged", **{"CFD T [N]": T, "CFD Q [N·m]": Q, "CFD P [W]": P,
                          "CFD η / FM": m.get("efficiency") if V > 0 else m.get("figure_of_merit"),
                          "CFD N [N]": N, "CFD My [N·m]": My, "CFD CNα [1/rad]": cn_alpha,
                          "T CFD / BEMT": T / op.thrust if op.thrust else np.nan, "Q CFD / BEMT": Q / op.torque if op.torque else np.nan,
                          "CNα CFD / De Young": cn_alpha / row["De Young CNα [1/rad]"] if s["kind"] == "oblique" and row["De Young CNα [1/rad]"] else np.nan,
                          "thrust scatter": m.get("thrust_std_lastN"), "iterations": m.get("iterations"), "converged": m.get("converged"),
                          "cells": m.get("mesh_cells")}, note=note)
    return pd.DataFrame(rows).T


def summary(tab) -> dict:
    st = tab["status"].astype(str)
    out = {k: int((st == k).sum()) for k in ("solved", "diverged", "FAILED", "NOT RUN")}
    out["n"] = len(st)
    out["by kind"] = {k: int(((st == "solved") & (tab["kind"] == k)).sum()) for k in ("hover", "axial", "oblique")}
    out["line"] = (f"{out['solved']} of {out['n']} cases solved (hover {out['by kind']['hover']}, axial {out['by kind']['axial']}, "
                   f"oblique {out['by kind']['oblique']}), {out['diverged']} diverged, {out['FAILED']} failed, {out['NOT RUN']} not run")
    return out


# ------------------------------------------------------------------------------------------------- the calibration
def fit_section(tab, *, kinds=("hover", "axial")) -> dict:
    """The two scales of the blade section — ``cl_scale`` on the lift slope, ``cd_scale`` on the profile drag — that make
    BEMT reproduce the solved hover and axial cases' thrust and torque together (least squares on the log ratios).
    ``None`` when fewer than two cases solved."""
    from scipy.optimize import least_squares
    from vegeta import boreas
    sel = tab[(tab["status"] == "solved") & tab["kind"].isin(kinds)] if "CFD T [N]" in tab.columns else tab.iloc[0:0]
    if len(sel) < 2:
        return {"cl_scale": None, "cd_scale": None, "n_cases": int(len(sel)), "rms_log_ratio_before": None, "rms_log_ratio_after": None}
    sys_ = ks.propulsion()
    base = ks.blade_airfoil(calibrated=False)
    pts = [(float(r["rpm"]), float(r["V [m/s]"]) * math.cos(math.radians(float(r["angle [deg]"]))), float(r["CFD T [N]"]), float(r["CFD Q [N·m]"]))
           for _, r in sel.iterrows()]

    def resid(x):
        af = boreas.Airfoil(name=base.name, cl_alpha=base.cl_alpha * x[0], alpha0_deg=base.alpha0_deg, cl_max=base.cl_max, cd0=base.cd0 * x[1], k=base.k)
        out = []
        for rpm, v, T, Q in pts:
            op = boreas.bemt.solve(sys_.prop, af, rpm, v, RHO)
            out += [math.log(max(T, 1e-6) / max(op.thrust, 1e-6)), math.log(max(Q, 1e-9) / max(op.torque, 1e-9))]
        return np.array(out)

    r0 = resid([1.0, 1.0])
    sol = least_squares(resid, [1.0, 1.0], bounds=([0.3, 0.1], [2.0, 5.0]))
    return {"cl_scale": float(sol.x[0]), "cd_scale": float(sol.x[1]), "n_cases": len(pts),
            "rms_log_ratio_before": float(np.sqrt(np.mean(r0 ** 2))), "rms_log_ratio_after": float(np.sqrt(np.mean(sol.fun ** 2)))}


def fit_normal_force(tab) -> dict:
    """``cn_factor``: the median of the CFD's normal-force slope over De Young's across the solved oblique cases
    (``None`` without any), and the table of C_Nα against the angle and the advance ratio."""
    if "CNα CFD / De Young" not in tab.columns:
        return {"cn_factor": None, "n_cases": 0, "points": []}
    sel = tab[(tab["status"] == "solved") & (tab["kind"] == "oblique")]
    ratios = [float(v) for v in sel["CNα CFD / De Young"].astype(float) if np.isfinite(v)]
    rows = [{"angle_deg": float(r["angle [deg]"]), "V": float(r["V [m/s]"]), "J": float(r["J"]), "cn_alpha_cfd": float(r["CFD CNα [1/rad]"]),
             "cn_alpha_deyoung": float(r["De Young CNα [1/rad]"]), "thrust_ratio": float(r["T CFD / BEMT"])} for _, r in sel.iterrows()]
    return {"cn_factor": float(np.median(ratios)) if ratios else None, "n_cases": len(ratios), "points": rows}


def export(tab, path=PROP_LAB_JSON, *, meta: dict | None = None, keep_previous: bool = True) -> dict:
    """The calibration and the whole table to ``data/prop_lab.json``. A pass that solved nothing keeps a previous
    calibration (the file's statuses are refreshed under ``last_listing``)."""
    sec, nf = fit_section(tab), fit_normal_force(tab)
    rows = {str(k): {kk: (None if vv is None or (isinstance(vv, float) and not np.isfinite(vv)) else (vv.item() if hasattr(vv, "item") else vv))
                     for kk, vv in r.items()} for k, r in tab.to_dict("index").items()}
    doc = {"meta": meta or {}, "summary": {k: v for k, v in summary(tab).items() if k != "line"},
           "calibration": {"cl_scale": sec["cl_scale"], "cd_scale": sec["cd_scale"], "cn_factor": nf["cn_factor"],
                           "section_fit": sec, "normal_force_fit": nf}, "cases": rows}
    path = Path(path)
    prev = load(path)
    if keep_previous and sec["cl_scale"] is None and nf["cn_factor"] is None and prev and prev.get("calibration", {}).get("cl_scale") is not None:
        prev["last_listing"] = {"meta": meta or {}, "summary": doc["summary"]}
        path.write_text(json.dumps(prev, indent=1, default=str))
        return prev
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, default=str))
    return doc


def load(path=PROP_LAB_JSON) -> dict | None:
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def calibration(path=PROP_LAB_JSON) -> dict:
    """What Kestrel applies: ``{"cl_scale", "cd_scale", "cn_factor", "on_file": bool, "meta"}`` — unit scales and
    ``on_file`` False when there is no calibration yet."""
    doc = load(path)
    cal = (doc or {}).get("calibration", {})
    on = cal.get("cl_scale") is not None or cal.get("cn_factor") is not None
    return {"cl_scale": cal.get("cl_scale") or 1.0, "cd_scale": cal.get("cd_scale") or 1.0, "cn_factor": cal.get("cn_factor") or 1.0,
            "on_file": on, "meta": (doc or {}).get("meta", {}), "n_solved": (doc or {}).get("summary", {}).get("solved", 0)}
