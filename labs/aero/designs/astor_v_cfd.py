"""ASTOR-V / ASTOR-BOOST CFD cases (Aeromant / OpenFOAM), prepared here, RUN BY YOU (``sims/astor_cfd.sh``): the
whole airframe, propeller folded away (not in the geometry), at three wing sweeps, an angle-of-attack sweep each —
CL, CD, Cm about the CG.

What they are for: the lattice says the sliding carriage keeps the static margin at 8-13 % from 12 to 60 deg of
sweep — the CFD checks it with the body in the flow; the drag per sweep (what tucking the wing buys, what the pod
costs); the maximum lift spread (the launch and the stall). Steady RANS k-omega SST with wall functions: trends.
Reference area and chord are fixed for every sweep (``astor_v_model.S_REF``, ``C_REF``).
"""
from __future__ import annotations

import math
from pathlib import Path

import arigo1_cfd as ac
import astor_v as V
import astor_v_model as VM
from nisus_cfd import NU, QUALITY, RHO

# sweep [deg] -> (speed m/s, tail fan, angles of attack)
CASES = {12.0: (25.0, 1.0, (-4.0, 0.0, 4.0, 8.0, 10.0, 12.0, 14.0, 16.0)),
         35.0: (40.0, 1.0, (-2.0, 0.0, 2.0, 4.0, 8.0, 12.0)),
         60.0: (60.0, 0.6, (-2.0, 0.0, 1.0, 2.0, 4.0, 6.0))}
# ASTOR-TWIN's states. twin_agile (the wing as it now is: 12 % thick, 4 % camber, flaps up, tube level) and
# twin_agile_flap (every flap down 20 deg) are what the flight model flies; twin and twin_stowed are the OLD thin
# wing (8 %, 2 %) with the tube 15 deg down and level, kept so their cases stay what they were.
AGILE = {"panel_thickness": 0.12, "panel_camber": 0.04, "inboard_flap_chord": 0.30}
AGILE_FLAPS = {**AGILE, "flaperon_deg": 20.0, "inboard_flap_deg": 20.0}
THIN = {"panel_thickness": 0.08, "panel_camber": 0.02, "inboard_flap_chord": 0.0}
# THE SECTION QUESTION (2026-11-23): the 4 % cambered wing costs ~25 % more drag than the thin one from 45 m/s up (it
# flies there at about -2 deg), for 0.06 of maximum lift; the flaps are what bring the lift. Two ways to keep the flaps
# and lose the penalty: the same thickness with 2 % camber (twin_c2, twin_c2_flap), or the 4 % wing with every flap
# reflexed 6 deg up in fast flight (twin_agile_reflex: the low angles only).
C2 = {"panel_thickness": 0.12, "panel_camber": 0.02, "inboard_flap_chord": 0.30}
SECTION_STATES = {"twin_c2": C2, "twin_c2_flap": {**C2, "flaperon_deg": 20.0, "inboard_flap_deg": 20.0},
                  "twin_agile_reflex": {**AGILE, "flaperon_deg": -6.0, "inboard_flap_deg": -6.0}}
ONLY_ALPHA = {"twin_agile_reflex": (-4.0, -2.0, 0.0, 2.0), "twin2_reflex": (-4.0, -2.0, 0.0, 2.0), "twin2_crow": (0.0, 4.0, 8.0), "twin2_swept": (-2.0, 0.0, 2.0),
              "twin2_tail_up": (0.0, 8.0, 14.0), "twin2_tail_lift": (4.0, 8.0, 14.0),
              "goshawk_flap35": (0.0, 4.0, 8.0, 12.0), "goshawk_tucked": (0.0, 4.0, 8.0)}
# HAWK 2 (2026-11-24: the wing doubled, astor_twin's default geometry; the tube level). Every older "twin..." name
# above is HAWK1 (astor_twin.variant1) and stays what it was.
#   twin2         flaps up, to 60 deg (the high angles are for the nose-high flare)
#   twin2_flap    every flap down 20 deg, to 60 deg
#   twin2_reflex  every flap 6 deg up, at the fast-flight angles
#   twin2_crow    the speed brake: flaperons 20 deg up, inboard flaps 40 deg down
#   twin2_swept   the outer pieces swung back to 60 deg, flaps reflexed: does sweeping the big wing help the dash?
#   twin2_tail_up    the tail halves 8 deg leading edge down with their flaps 8 deg up: the tail's pitch-up power
#   twin2_tail_lift  flaps down 20 deg, the tail halves 6 deg leading edge up with their flaps 6 deg down: the tail lifting
HAWK2_STATES = {"twin2": {}, "twin2_flap": {"flaperon_deg": 20.0, "inboard_flap_deg": 20.0}, "twin2_reflex": {"flaperon_deg": -6.0, "inboard_flap_deg": -6.0},
                "twin2_crow": {"flaperon_deg": -20.0, "inboard_flap_deg": 40.0}, "twin2_swept": {"flaperon_deg": -6.0, "inboard_flap_deg": -6.0, "wing_sweep_deg": 60.0},
                "twin2_tail_up": {"tail_incidence_deg": -8.0, "tail_flap_deg": -8.0},
                "twin2_tail_lift": {"flaperon_deg": 20.0, "inboard_flap_deg": 20.0, "tail_incidence_deg": 6.0, "tail_flap_deg": 6.0}}
# GOSHAWK (designs/goshawk: 1.15 m, 1.7 kg, for the wood), at 12 m/s (its speed; a third of HAWK1's Reynolds number):
#   goshawk          flaps up, to 60 deg          goshawk_flap     every flap down 20 deg, to 60 deg
#   goshawk_flap35   every flap down 35 deg       goshawk_tucked   the outer pieces drawn in 70 deg
GOSHAWK_STATES = {"goshawk": {}, "goshawk_flap": {"flaperon_deg": 20.0, "inboard_flap_deg": 20.0}, "goshawk_flap35": {"flaperon_deg": 35.0, "inboard_flap_deg": 35.0},
                  "goshawk_tucked": {"wing_sweep_deg": 70.0}}
# ASTOR-X (designs/astor_x: a swept wing of 1.2 m span, four pushers on a short X, 2026-11-24). The propellers are not in
# the geometry (as for every design here); the struts and pods are. At 25 m/s.
#   x          flaps up, to 60 deg (it flies beyond the stall on its rotors: the high angles are its transition)
#   x_flap     ailerons and inboard flaps down 20 deg, to 60 deg
#   x_tail_up  the tail halves 8 deg leading edge down with their flaps 8 deg up: the tail's pitch-up power on this wing
#   x_reflex   every flap 6 deg up at the fast-flight angles
X_STATES = {"x": {}, "x_flap": {"flaperon_deg": 20.0, "inboard_flap_deg": 20.0}, "x_tail_up": {"tail_incidence_deg": -8.0, "tail_flap_deg": -8.0},
            "x_reflex": {"flaperon_deg": -6.0, "inboard_flap_deg": -6.0}}
ONLY_ALPHA.update({"x_tail_up": (0.0, 8.0, 14.0), "x_reflex": (-4.0, -2.0, 0.0, 2.0)})
# ASTOR-X2 (designs/astor_x2: the tail-sitter — ASTOR-X's wing, an X of four fins with elevons and feet, tractors on the fins):
#   x2           flaps up, to 60 deg          x2_flap      ailerons and inboard flaps down 20 deg, to 60 deg
#   x2_pitch_up  the four elevons 15 deg trailing edge up: the X tail's pitch power (the lattice says a third of ASTOR-X's)
#   x2_reflex    every wing flap 6 deg up at the fast-flight angles
X2_STATES = {"x2": {}, "x2_flap": {"flaperon_deg": 20.0, "inboard_flap_deg": 20.0}, "x2_pitch_up": {"elevon_pitch_deg": -15.0},
             "x2_reflex": {"flaperon_deg": -6.0, "inboard_flap_deg": -6.0}}
ONLY_ALPHA.update({"x2_pitch_up": (0.0, 8.0, 14.0), "x2_reflex": (-4.0, -2.0, 0.0, 2.0)})
X_ALL = {**X_STATES, **X2_STATES}
SPEED_M_S = {k: 12.0 for k in GOSHAWK_STATES}                              # a design's own CFD speed (else CASES')
DESIGNS = ("v", "boost", "t", "twin", "twin_stowed", "twin_agile", "twin_agile_flap", "twin_c2", "twin_c2_flap", "twin_agile_reflex") + tuple(HAWK2_STATES) + tuple(GOSHAWK_STATES) + tuple(X_STATES) + tuple(X2_STATES)


def speed(design, sweep) -> float:
    return SPEED_M_S.get(design, CASES[sweep][0])      # twin_stowed: ASTOR-TWIN with the launch tube level along the belly (0 deg), for its drag


def sweeps(design) -> tuple:
    """The wing states run for this design: ASTOR-TWIN's wing is fixed at 12 deg (its one state, 8 angles)."""
    return (12.0,) if design.startswith(("twin", "goshawk")) or design in X_ALL else tuple(CASES)


EXTRA_ALPHA = {"twin_agile": (-2.0, 2.0), "twin_agile_flap": (-2.0, 2.0), "twin_c2": (-2.0, 2.0), "twin_c2_flap": (-2.0, 2.0),
               "twin2": (-2.0, 2.0, 20.0, 30.0, 45.0, 60.0), "twin2_flap": (-2.0, 2.0, 20.0, 30.0, 45.0, 60.0),
               "goshawk": (-2.0, 2.0, 20.0, 30.0, 45.0, 60.0), "goshawk_flap": (-2.0, 2.0, 20.0, 30.0, 45.0, 60.0),
               "x": (-2.0, 2.0, 20.0, 24.0, 30.0, 45.0, 60.0), "x_flap": (-2.0, 2.0, 20.0, 24.0, 30.0, 45.0, 60.0),
               "x2": (-2.0, 2.0, 20.0, 24.0, 30.0, 45.0, 60.0), "x2_flap": (-2.0, 2.0, 20.0, 24.0, 30.0, 45.0, 60.0)}   # the cambered wing flies fast at about -2 deg; the flaps' lift peaks near 4


def alphas(design, sweep) -> tuple:
    """The angles of attack run for this design at this sweep."""
    if design in ONLY_ALPHA:
        return ONLY_ALPHA[design]
    return tuple(sorted(set(CASES[sweep][2]) | set(EXTRA_ALPHA.get(design, ()))))


def key(design, sweep) -> str:
    return f"{design}_sweep{sweep:.0f}"


def _cls(design):
    if design == "boost":
        import astor_boost as B
        return B.AstorBoost, B.variant
    if design == "t":
        import astor_t as T
        return T.AstorT, T.variant
    if design == "twin":                                              # the OLD thin wing, the launcher at its rest tilt (15 deg down)
        import astor_twin as W
        return W.AstorTwin, (lambda **over: W.variant1(**{"net_launchers": 1, "talon_deploy": -1.0, "net_count": 0, **THIN, **over}))
    if design in X2_STATES:
        import astor_x2 as AX2
        return AX2.AstorX2, (lambda **over: AX2.variant(**{"net_launchers": 1, "talon_deploy": -1.0, "net_count": 0, "launcher_depression_deg": 0.0, **over, **X2_STATES[design]}))
    if design in X_STATES:
        import astor_x as AX
        return AX.AstorX, (lambda **over: AX.variant(**{"net_launchers": 1, "talon_deploy": -1.0, "net_count": 0, "launcher_depression_deg": 0.0, **over, **X_STATES[design]}))
    if design in GOSHAWK_STATES:
        import goshawk as GH
        return GH.Goshawk, (lambda **over: GH.variant(**{"talon_deploy": -1.0, **over, **GOSHAWK_STATES[design]}))      # legs off: folded they are in the body's shadow
    if design in HAWK2_STATES:                                        # the given state last: export_stl passes the case's own sweep (12 deg)
        import astor_twin as W
        return W.AstorTwin, (lambda **over: W.variant(**{"net_launchers": 1, "talon_deploy": -1.0, "net_count": 0, "launcher_depression_deg": 0.0, **over, **HAWK2_STATES[design]}))
    if design in SECTION_STATES:
        import astor_twin as W
        return W.AstorTwin, (lambda **over: W.variant1(**{"net_launchers": 1, "talon_deploy": -1.0, "net_count": 0, "launcher_depression_deg": 0.0, **SECTION_STATES[design], **over}))
    if design in ("twin_agile", "twin_agile_flap"):                   # the thicker, more cambered section; and with every flap down 20 deg
        import astor_twin as W
        extra = AGILE_FLAPS if design.endswith("flap") else AGILE
        return W.AstorTwin, (lambda **over: W.variant1(**{"net_launchers": 1, "talon_deploy": -1.0, "net_count": 0, "launcher_depression_deg": 0.0, **extra, **over}))
    if design == "twin_stowed":
        import astor_twin as W
        return W.AstorTwin, (lambda **over: W.variant1(**{"net_launchers": 1, "talon_deploy": -1.0, "net_count": 0, "launcher_depression_deg": 0.0, **THIN, **over}))
    return V.AstorV, V.variant


def reference(design="v", sweep=12.0) -> dict:
    """The coefficients' reference, and the geometry the case was flown with (the corrections check it)."""
    cls, var = _cls(design)
    p = cls().resolve(**var())
    return {"S": VM.S_REF, "c": VM.C_REF, "cg_mm": [V.CG_MM, 0.0], "carriage_travel_mm": V.travel(sweep, p), "camber": p["panel_camber"]}


def l_ref(design="v", sweep=12.0) -> float:
    """The CFD's reference length [m]: the reference chord, unless the body is over 4 of them (the template's limit:
    spread, the wing is 1.08 m across on a 0.20 m chord); then the largest dimension / 3.7, and ``row`` scales Cm back
    to the reference chord."""
    cls, var = _cls(design)
    p = cls().resolve(**var())
    length = (p.get("edf_x", 0.0) + p.get("edf_length", 0.0)) or p["fuselage_length"]
    span = 2 * p["sw_half_span"] if design in X_ALL else V.wing_numbers(p, sweep)["span_mm"]
    body = max(span, max(length, p["fuselage_length"]) + 80.0) / 1000.0
    return max(VM.C_REF, body / 3.7)


def export_stl(folder, design, sweep, alpha_deg):
    import arigo_stl
    cls, var = _cls(design)
    tf = CASES[sweep][1]
    if sweep >= 50.0:
        tf = cls().resolve(**var()).get("dash_tail_fan", tf)             # the design's own tail state in the dash (the flight model's)

    def build(path):
        from vegeta import dedalus
        shape = cls().generate(**var(wing_sweep_deg=sweep, tail_fan=tf)).shape.rotate((0, 0, 0), (0, 1, 0), alpha_deg)
        dedalus.Geometry.from_cadquery(shape, name=f"astor_{design}").export_stl(str(path), tolerance=0.2)
    return arigo_stl.stl(folder, f"astor_{key(design, sweep)}", alpha_deg, build)


def case(workdir, design, sweep, *, alpha_deg=0.0, quality="screening", env=None):
    from vegeta import aeromant
    Vv = speed(design, sweep)
    ref = reference(design, sweep)
    stl, tag = export_stl(Path(workdir) / "stl", design, sweep, alpha_deg)
    params = dict(velocity=Vv, kinematic_viscosity=NU, density=RHO, reference_area=ref["S"], reference_length=l_ref(design, sweep),
                  center_of_rotation=tuple(ac.to_flow(*ref["cg_mm"], alpha_deg).tolist()), **QUALITY[quality])
    env = env or aeromant.OpenFOAMEnvironment.detect()
    return aeromant.CFDCase("rans_ksst_external", str(stl), params,
                            workdir=str(Path(workdir) / f"astor_{key(design, sweep)}{tag}_a{alpha_deg:+.0f}_V{Vv:.0f}_{quality}"),
                            geometry_units="mm", environment=env)


def lattice_row(sweep, alpha_deg, design="v") -> dict:
    """The lattice's CL and Cm about the CG at this sweep and angle (no drag: inviscid), for the comparison: on the
    design's own wing and tail."""
    cls, var = _cls(design)
    if design in X_ALL:                                               # its own wing: astor_x_flight's lattice, with the state's flaps and tail
        import astor_x_flight as XF
        st = X_ALL[design]
        r = XF.lattice(cls().resolve(**var()), alpha=alpha_deg, tail_inc=st.get("tail_incidence_deg", st.get("elevon_pitch_deg", 0.0)), flap=st.get("inboard_flap_deg", 0.0))
        return {"Cl": round(r["CL"], 4), "Cm": round(r["Cm"] + r["CL"] * V.CG_MM / 1000.0 / VM.C_REF, 4)}
    r = VM.lattice(sweep, tail_fan=CASES[sweep][1], alpha=alpha_deg, p=cls().resolve(**var()))
    return {"Cl": round(r["CL"], 4), "Cm": round(r["Cm"] + r["CL"] * V.CG_MM / 1000.0 / VM.C_REF, 4)}


def row(design, sweep, alpha_deg, result) -> dict:
    out = {"alpha_deg": alpha_deg, "V": speed(design, sweep), "lattice": lattice_row(sweep, alpha_deg, design)}
    if result is None or not getattr(result, "ok", False):
        out["status"] = "NOT RUN" if result is None or any("NOT RUN" in s for s in getattr(result, "messages", [])) else "FAILED"
        out["messages"] = [str(m)[:200] for m in list(getattr(result, "messages", []))[:3]]
        return out
    k = result.metrics
    f = lambda v: None if v is None else float(v)
    out.update({"status": "solved", "Cl": f(k.get("Cl")), "Cd": f(k.get("Cd")),
                "Cm_cg": None if k.get("Cm") is None else float(k.get("Cm")) * l_ref(design, sweep) / VM.C_REF,
                "converged": k.get("converged"), "iterations": k.get("iterations"), "cells": k.get("mesh_cells"), "mesh_ok": k.get("mesh_ok")})
    return out


def summary(rows) -> dict:
    """Per configuration: CD0, the lift slope, the static margin (dCm/dCL), the maximum lift."""
    import numpy as np
    ok = [r for r in rows if r.get("status") == "solved" and r.get("Cl") is not None]
    if len(ok) < 3:
        return {"solved": len(ok)}
    al, cl, cd, cm = (np.array([r[k] for r in ok]) for k in ("alpha_deg", "Cl", "Cd", "Cm_cg"))
    lin = np.abs(al) <= 6.0
    out = {"solved": len(ok), "CD_min": float(cd.min()), "CL_max": float(cl.max()), "alpha_CL_max_deg": float(al[int(np.argmax(cl))])}
    if lin.sum() >= 2:
        a1 = float(np.polyfit(np.radians(al[lin]), cl[lin], 1)[0])
        m1 = float(np.polyfit(cl[lin], cm[lin], 1)[0])
        out.update({"CL_alpha_per_rad": a1, "static_margin": -m1, "drag_area_cm2": float(cd.min() * VM.S_REF * 1e4)})
    return out
