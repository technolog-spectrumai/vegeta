"""ASTOR-X — the swept wing's spar and a rotor strut at 8 g in Talos (CalculiX FEA). Prepared here, solved where
CalculiX and Gmsh are (``sims/astor_fea.sh --design x``).

ASTOR-X (``designs/astor_x``) has two members ASTOR-TWIN did not: a one-piece swept wing and four short struts with a
motor on each tip. Both are sized here as carbon tubes and checked:

- **the beam reference** (``reference``, closed form, runs anywhere): the numbers the FEA is read against.
- **wing_spar_8g** (FEA): one half wing's spar, a tube along the wing's 30 % chord line, held inside the body, the half
  wing's elliptic lift at 8 g ultimate on six stations. The spar is SWEPT (about 30 deg): it is modelled as a straight
  cantilever of its own length (the half span / cos of its sweep) — the lift's arm along a swept spar is that much
  longer. The twist a swept wing puts into its root is NOT in this model (it goes into the skin and the root rib).
- **strut_8g** (FEA): one strut, a tube from the body's side to the motor: the motor's full thrust along the body
  axis and the motor's weight at 8 g across it, both ultimate, together.
- **strut_modes** (FEA): the strut with the motor, propeller and pod as a point mass on its tip — against the
  propeller's once-per-turn frequency.

NOT covered: the wing's skin, ribs and torsion; the struts' joint to the body (four tubes meet the fuselage between
the wing and the tail: a frame is needed there, not drawn); the tail, the fin, the launcher (ASTOR-TWIN's); flutter.

Units mm-N-MPa, masses in tonnes.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import astor_twin_fea as TF
import astor_x as AX
import astor_x_flight as XF

G, ULTIMATE, KNOCKDOWN, CARBON_TUBE = TF.G, TF.ULTIMATE, TF.KNOCKDOWN, TF.CARBON_TUBE
N_G = 8.0                                      # astor_v_flight.SAS["x"]["g_max"]
SPAR = dict(inner_od=25.0, inner_id=22.0, outer_od=16.0, outer_id=13.0, plug_mm=80.0, step_at=0.5, chord_line=0.30)      # ASTOR-TWIN's two tubes, the step at half the half span
STRUT = dict(od=12.0, id=9.0)                  # inside the strut's fairing: 20 % of 60 mm is 12 mm deep
MOTOR_KG = 0.23 + 0.075 + 0.02 + 0.04          # a 2 kW motor, its controller, a 9-inch propeller, the pod (HAND +-25 %)
THRUST_N = 202.3 / 4                           # each motor's share of the static thrust (astor_v_flight's table: HAND)
STATIONS = 6


def params():
    return AX.resolve()


def spar_sweep_deg(p=None) -> float:
    p = p or params()
    return math.degrees(math.atan(math.tan(math.radians(p["sw_sweep_deg"])) - SPAR["chord_line"] * (p["sw_root_chord"] - p["sw_tip_chord"]) / p["sw_half_span"]))


def loads(p=None, n=N_G, mass=None) -> dict:
    """The half wing's lift at ``n`` g, ultimate, on STATIONS equal lengths ALONG THE SPAR from the body's side to
    the tip (an elliptic distribution over the half span); positions are distances along the spar from the centre."""
    p = p or params()
    half, k = p["sw_half_span"], 1.0 / math.cos(math.radians(spar_sweep_deg(p)))
    y0 = p["fuselage_diameter"] / 2
    edges = np.linspace(y0, half, STATIONS + 1)
    ell = lambda y: (y / half * math.sqrt(max(1 - (y / half) ** 2, 0.0)) + math.asin(min(y / half, 1.0))) / (math.pi / 2)
    mass = mass or XF.MASS_KG
    L_half = ULTIMATE * n * mass * G / 2
    F = [L_half * (ell(b) - ell(a)) for a, b in zip(edges[:-1], edges[1:])]
    F[0] += L_half * ell(y0)
    return {"n": n, "mass_kg": mass, "half_span_mm": half, "spar_length_factor": k, "root_s_mm": float(y0 * k), "tip_s_mm": float(half * k),
            "step_s_mm": float(SPAR["step_at"] * half * k), "edges_s_mm": [float(e * k) for e in edges], "lift_N": [float(f) for f in F], "half_wing_lift_ultimate_N": L_half}


def strut_loads(p=None, n=N_G) -> dict:
    p = p or params()
    r0 = TF.W.AstorTwin().fuselage_radius(p, AX.AstorX()._strut_x(p) + 0.3 * p["x_strut_chord"])
    return {"root_mm": float(r0), "tip_mm": float(p["x_radius"]), "length_mm": float(p["x_radius"] - r0), "thrust_ultimate_N": ULTIMATE * THRUST_N,
            "inertia_ultimate_N": ULTIMATE * n * MOTOR_KG * G, "motor_kg": MOTOR_KG}


def _I(od, i):
    return math.pi / 64 * (od ** 4 - i ** 4)


def wing_reference(p, mass=None) -> dict:
    """The swept wing's spar as a cantilever of its own length (closed form)."""
    ld = loads(p, mass=mass)
    allow, E = CARBON_TUBE["strength"] * KNOCKDOWN, CARBON_TUBE["E"]
    mids = [(a + b) / 2 for a, b in zip(ld["edges_s_mm"][:-1], ld["edges_s_mm"][1:])]
    moment = lambda s: sum(f * (sm - s) for f, sm in zip(ld["lift_N"], mids) if sm > s)
    Ii, Io = _I(SPAR["inner_od"], SPAR["inner_id"]), _I(SPAR["outer_od"], SPAR["outer_id"])
    s0, sp, s1 = ld["root_s_mm"], ld["step_s_mm"], ld["tip_s_mm"]
    sr, ss = moment(s0) * SPAR["inner_od"] / 2 / Ii, moment(sp) * SPAR["outer_od"] / 2 / Io
    xs = np.linspace(s0, s1, 400)
    kk = np.array([moment(s) / (E * (Ii if s < sp else Io)) for s in xs])
    slope = np.concatenate([[0.0], np.cumsum(0.5 * (kk[1:] + kk[:-1]) * np.diff(xs))])
    defl = float(np.sum(0.5 * (slope[1:] + slope[:-1]) * np.diff(xs)))
    area = lambda od, i: math.pi / 4 * (od ** 2 - i ** 2)
    g_spar = (area(SPAR["inner_od"], SPAR["inner_id"]) * 2 * sp + area(SPAR["outer_od"], SPAR["outer_id"]) * 2 * (s1 - sp)) * 1.55e-3
    cl = lambda eta: p["sw_root_chord"] + (p["sw_tip_chord"] - p["sw_root_chord"]) * eta
    return {**SPAR, "sweep_deg": round(spar_sweep_deg(p), 1), "length_each_side_mm": round(s1), "mass_both_wings_g": round(g_spar),
            "half_wing_lift_ultimate_N": round(ld["half_wing_lift_ultimate_N"]),
            "fits": {"wing_depth_at_the_root_mm": round(0.12 * cl(0.0)), "at_the_step_mm": round(0.12 * cl(SPAR["step_at"])), "at_the_tip_mm": round(0.12 * cl(1.0))},
            "root": {"moment_Nm": round(moment(s0) / 1000.0, 1), "stress_MPa": round(sr), "margin": round(allow / sr - 1, 2)},
            "step": {"moment_Nm": round(moment(sp) / 1000.0, 1), "stress_MPa": round(ss), "margin": round(allow / ss - 1, 2)},
            "tip_deflection_ultimate_mm": round(defl), "tip_deflection_share_of_the_spar": round(defl / s1, 3),
            "note": "a straight cantilever of the swept spar's own length; the root twist of a swept wing is not in it"}


def reference(p=None) -> dict:
    p = p or params()
    st = strut_loads(p)
    allow, E = CARBON_TUBE["strength"] * KNOCKDOWN, CARBON_TUBE["E"]
    area = lambda od, i: math.pi / 4 * (od ** 2 - i ** 2)
    # the strut: the thrust (along the body axis) and the motor's weight at n g (across it) bend the tube about two axes
    Is, Ls = _I(STRUT["od"], STRUT["id"]), st["length_mm"]
    f_res = math.hypot(st["thrust_ultimate_N"], st["inertia_ultimate_N"])
    s_strut = f_res * Ls * STRUT["od"] / 2 / Is
    f1 = math.sqrt(3 * E * Is / Ls ** 3 / (MOTOR_KG * 1e-3)) / (2 * math.pi)
    return {"load_factor": N_G, "ultimate_factor": ULTIMATE, "mass_kg": XF.MASS_KG, "allowable_MPa": allow, "material": CARBON_TUBE["name"],
            "wing_spar": wing_reference(p),
            "strut": {**STRUT, "length_mm": round(Ls), "motor_kg": round(MOTOR_KG, 3), "thrust_ultimate_N": round(st["thrust_ultimate_N"], 1),
                      "motor_weight_at_8g_ultimate_N": round(st["inertia_ultimate_N"], 1), "root_moment_Nm": round(f_res * Ls / 1000.0, 2),
                      "stress_MPa": round(s_strut), "margin": round(allow / s_strut - 1, 2), "tip_deflection_mm": round(f_res * Ls ** 3 / (3 * E * Is), 2),
                      "first_bending_Hz": round(f1, 1), "mass_four_g": round(area(STRUT["od"], STRUT["id"]) * Ls * 4 * 1.55e-3),
                      "propeller_once_per_turn_Hz": {"at_6000_rpm": 100.0, "at_12000_rpm": 200.0},
                      "note": "the first bending frequency must stay clear of the propeller's once-per-turn over the working speeds (HAND: the speed range is not yet set)"}}


# ------------------------------------------------------------------------------------------- the CAD and the cases
def spar_cad(p=None):
    """The half wing's spar as a straight tube along +y (its own axis), in segments at the load stations."""
    import cadquery as cq
    p = p or params()
    ld = loads(p)
    sp = ld["step_s_mm"]
    cuts = sorted(set([0.0] + ld["edges_s_mm"] + [sp]))
    out = None
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a < 1e-6:
            continue
        seg = TF._tube(a, b, SPAR["inner_od"], SPAR["inner_id"]) if b <= sp + 1e-6 else TF._tube(a, b, SPAR["outer_od"], SPAR["outer_id"])
        out = seg if out is None else out.union(seg, clean=False)
    plug = cq.Workplane("XZ").circle(SPAR["inner_id"] / 2).circle(SPAR["outer_id"] / 2).extrude(-SPAR["plug_mm"]).translate((0, sp - SPAR["plug_mm"] / 2, 0))
    return out.union(plug, clean=False), cuts


def spar_regions(p=None):
    p = p or params()
    ld = loads(p)
    r = SPAR["inner_od"] / 2
    _, cuts = spar_cad(p)
    out = [TF._box("root", 0.0, ld["root_s_mm"], r)]
    for k, (a, b) in enumerate(zip(ld["edges_s_mm"][:-1], ld["edges_s_mm"][1:])):
        inside = [(c0, c1) for c0, c1 in zip(cuts[:-1], cuts[1:]) if c0 >= a - 1e-6 and c1 <= b + 1e-6]
        c0, c1 = max(inside, key=lambda s: s[1] - s[0])
        out.append(TF._box(f"S{k}", c0, c1, r))
    return out


TIP_MM = 24.0                                  # the motor's mount: the last length of the strut


def strut_cad(p=None):
    """One strut's tube along +y from the axis to the motor: the part inside the body (held), the free length, the mount."""
    st = strut_loads(p)
    cuts = [0.0, st["root_mm"], st["tip_mm"] - TIP_MM, st["tip_mm"]]
    out = None
    for a, b in zip(cuts[:-1], cuts[1:]):
        seg = TF._tube(a, b, STRUT["od"], STRUT["id"])
        out = seg if out is None else out.union(seg, clean=False)
    return out, cuts


def strut_regions(p=None):
    st = strut_loads(p)
    r = STRUT["od"] / 2
    return [TF._box("root", 0.0, st["root_mm"], r), TF._box("mount", st["tip_mm"] - TIP_MM, st["tip_mm"], r)]


FEACase, solve_one, regions_select = TF.FEACase, TF.solve_one, TF.regions_select


def cases(workdir, p=None, *, element_size=3.0):
    from vegeta import talos
    p = p or params()
    mat = talos.Material(CARBON_TUBE["name"], youngs_modulus=CARBON_TUBE["E"], poissons_ratio=CARBON_TUBE["nu"], density=CARBON_TUBE["rho"],
                         yield_strength=CARBON_TUBE["strength"], source=CARBON_TUBE["source"])
    sup = [talos.FixedSupport("root")]
    spar, _ = spar_cad(p)
    step_w = TF.export_step(spar, Path(workdir) / "cad" / "astor_x_wing_spar.step")
    ld = loads(p)
    out = [FEACase("wing_spar_8g", "the half wing's lift at 8 g, ultimate, on the swept spar as a straight cantilever of its own length",
                   talos.StructuralModel(step_w, "mm-N-MPa", mat, spar_regions(p), sup, [talos.Force(f"S{k}", fz=f) for k, f in enumerate(ld["lift_N"])],
                                         talos.MeshSettings(element_size=element_size, order=2), name="wing_spar_8g"))]
    strut, _ = strut_cad(p)
    step_s = TF.export_step(strut, Path(workdir) / "cad" / "astor_x_strut.step")
    st = strut_loads(p)
    mesh_s = talos.MeshSettings(element_size=min(element_size, 2.0), order=2)          # the strut's wall is 1.5 mm
    out.append(FEACase("strut_8g", "one strut: the motor's full thrust along the body axis and its weight at 8 g across it, ultimate",
                       talos.StructuralModel(step_s, "mm-N-MPa", mat, strut_regions(p), sup,
                                             [talos.Force("mount", fx=-st["thrust_ultimate_N"], fz=-st["inertia_ultimate_N"])], mesh_s, name="strut_8g")))
    out.append(FEACase("strut_modes", "the strut with the motor, controller, propeller and pod as a point mass on its mount",
                       talos.StructuralModel(step_s, "mm-N-MPa", mat, strut_regions(p), sup, [], mesh_s, masses=[talos.PointMass("mount", MOTOR_KG * 1e-3)], name="strut_modes"),
                       kind="modes"))
    return out


def row(case, r) -> dict:
    status = "NOT RUN" if r is None else "solved" if getattr(r, "ok", False) else ("NOT RUN" if any("NOT RUN" in m for m in getattr(r, "messages", [])) else "FAILED")
    out = {"kind": case.kind, "status": status, "description": case.description}
    if status != "solved":
        out["note"] = "" if r is None else (r.messages[-1][:200] if r.messages else "")
        return out
    ref = reference()
    if case.kind == "modes":
        f = [float(x) for x in (r.metrics.get("frequencies_hz") or [])]
        out.update({"frequencies_Hz": [round(x, 1) for x in f[:4]], "beam_reference_Hz": ref["strut"]["first_bending_Hz"], "propeller_once_per_turn_Hz": ref["strut"]["propeller_once_per_turn_Hz"]})
    else:
        vm, allow = r.metrics.get("max_von_mises"), CARBON_TUBE["strength"] * KNOCKDOWN
        part = ref["wing_spar"] if case.name.startswith("wing") else ref["strut"]
        out.update({"peak_von_mises_MPa": None if vm is None else float(vm), "max_displacement_mm": r.metrics.get("max_displacement"), "allowable_MPa": allow,
                    "margin": None if not vm else round(allow / float(vm) - 1, 2),
                    "beam_reference": {k: part[k] for k in ("root", "step", "tip_deflection_ultimate_mm", "stress_MPa", "margin", "tip_deflection_mm") if k in part}})
    return out


AIRCRAFT, PREFIX, BAR = "ASTOR-X", "astor_x", "astor-x fea"
CASE_LIST = [("wing_spar_8g", "static", "the half wing's lift at 8 g, ultimate, on the swept spar"), ("strut_8g", "static", "one strut: full thrust and the motor's weight at 8 g, ultimate"),
             ("strut_modes", "modes", "the strut with the motor as a point mass")]

if __name__ == "__main__":
    import yaml
    import arigo1_model as am
    ref = reference()
    sel = {"wing_spar": regions_select(spar_cad()[0], spar_regions()), "strut": regions_select(strut_cad()[0], strut_regions())}
    (am.DATA / "astor_x").mkdir(parents=True, exist_ok=True)
    (am.DATA / "astor_x" / "astor_x_spar_reference.yaml").write_text(yaml.safe_dump({"beam_reference": ref, "faces_selected_per_region": sel}, sort_keys=False, allow_unicode=True, width=140))
    print(yaml.safe_dump({"beam_reference": ref, "faces_selected_per_region": sel}, sort_keys=False, allow_unicode=True, width=140))
