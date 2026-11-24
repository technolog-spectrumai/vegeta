"""ASTOR-X2 (the tail-sitter) — the wing's spar and a fin's spar in Talos (CalculiX FEA). Prepared here, solved where
CalculiX and Gmsh are (``sims/astor_x2.sh``).

ASTOR-X2's fins do three jobs: they are the tail, they carry the motors, and the aircraft lands on their tips.
Each fin has a carbon tube along it (``FIN_SPAR``, inside the 10 % thick section: 16 mm fits to the tip), from the
body's side to the foot. The cases:

- **wing_spar_8g**: ASTOR-X's swept wing spar (``astor_x_fea``) at this aircraft's mass.
- **fin_flight_8g**: one fin's tube in an 8 g pull-up at full thrust, ultimate: at the motor's station the thrust
  (along the body) and the motor's weight at 8 g (its part across the fin, cos 45, and its part along it are both put
  across the tube: conservative); the fin's own air load at its limit (``FIN_CN`` at ``FIN_V`` m/s) at mid span.
- **fin_landing**: the landing on TWO feet at ``LAND_G`` g, ultimate: the ground pushes on the foot along the body's
  axis, bending the fin in its own plane. The tube ALONE takes it here; the fin's skin, a deep beam in that
  direction, is not counted (conservative: if this passes, the fin passes).
- **fin_modes**: the tube with the motor as a point mass at its station — against the propeller's once-per-turn.

NOT covered: the fins' joint to the tail cone (four tubes meet a thin cone that also holds the air vessel: a frame is
needed, not drawn); the skin; the elevons' hinges; flutter of a fin with a motor on it; the wing's torsion.

Units mm-N-MPa, masses in tonnes.
"""
from __future__ import annotations

import math
from pathlib import Path

import astor_twin_fea as TF
import astor_x2 as AX2
import astor_x_fea as XFE
import astor_x_flight as XF

G, ULTIMATE, KNOCKDOWN, CARBON_TUBE = TF.G, TF.ULTIMATE, TF.KNOCKDOWN, TF.CARBON_TUBE
N_G = 8.0
MASS_KG = XF.MASS["x2"]
FIN_SPAR = dict(od=16.0, id=13.0)
MOTOR_KG, THRUST_N = XFE.MOTOR_KG, XFE.THRUST_N
LAND_G = 3.0                                   # a firm vertical landing (HAND): 3 g on two of the four feet
FIN_CN, FIN_V = 1.0, 60.0                      # one fin at its maximum normal force at 60 m/s (HAND: the worst the elevons and a gust can ask)
MOUNT_MM, FOOT_MM = 24.0, 24.0
RHO = 1.225


def params():
    return AX2.resolve()


def fin_loads(p=None) -> dict:
    p = p or params()
    A = AX2.AstorX2()
    R, r_m = p["fin_radius"], p["x_radius"]
    r0 = A.fuselage_radius(p, p["xfin_root_le_x"] + 0.3 * p["xfin_root_chord"])
    area = 0.5 * (A._fin_chord(p, r0) + A._fin_chord(p, R)) * (R - r0) / 1e6
    air = ULTIMATE * 0.5 * RHO * FIN_V ** 2 * area * FIN_CN
    return {"root_mm": float(r0), "motor_mm": float(r_m), "tip_mm": float(R), "air_station_mm": float(0.5 * (r0 + R)), "fin_area_m2": area,
            "thrust_ultimate_N": ULTIMATE * THRUST_N, "motor_inertia_ultimate_N": ULTIMATE * N_G * MOTOR_KG * G, "air_ultimate_N": air,
            "landing_ultimate_N": ULTIMATE * LAND_G * MASS_KG * G / 2, "motor_kg": MOTOR_KG}


def reference(p=None) -> dict:
    p = p or params()
    ld = fin_loads(p)
    allow, E = CARBON_TUBE["strength"] * KNOCKDOWN, CARBON_TUBE["E"]
    I = XFE._I(FIN_SPAR["od"], FIN_SPAR["id"])
    c = FIN_SPAR["od"] / 2
    Lm, La, Lt = ld["motor_mm"] - ld["root_mm"], ld["air_station_mm"] - ld["root_mm"], ld["tip_mm"] - ld["root_mm"]
    # flight: the thrust in the fin's plane; the motor's weight and the air load across it (taken together across: conservative)
    m_in = ld["thrust_ultimate_N"] * Lm
    m_out = ld["motor_inertia_ultimate_N"] * Lm + ld["air_ultimate_N"] * La
    s_flight = math.hypot(m_in, m_out) * c / I
    s_land = ld["landing_ultimate_N"] * Lt * c / I
    f1 = math.sqrt(3 * E * I / Lm ** 3 / (MOTOR_KG * 1e-3)) / (2 * math.pi)
    area = math.pi / 4 * (FIN_SPAR["od"] ** 2 - FIN_SPAR["id"] ** 2)
    A = AX2.AstorX2()
    return {"load_factor": N_G, "ultimate_factor": ULTIMATE, "mass_kg": MASS_KG, "allowable_MPa": allow, "material": CARBON_TUBE["name"],
            "wing_spar": XFE.wing_reference(p, MASS_KG),
            "fin_spar": {**FIN_SPAR, "length_mm": round(Lt), "mass_four_g": round(area * Lt * 4 * 1.55e-3),
                         "fits": {"fin_depth_at_the_root_mm": round(p["xfin_thickness"] * A._fin_chord(p, ld["root_mm"]), 1),
                                  "at_the_motor_mm": round(p["xfin_thickness"] * A._fin_chord(p, ld["motor_mm"]), 1), "at_the_tip_mm": round(p["xfin_thickness"] * A._fin_chord(p, ld["tip_mm"]), 1)},
                         "flight_8g": {"thrust_N": round(ld["thrust_ultimate_N"], 1), "motor_weight_at_8g_N": round(ld["motor_inertia_ultimate_N"], 1), "fin_air_load_N": round(ld["air_ultimate_N"], 1),
                                       "root_moment_Nm": round(math.hypot(m_in, m_out) / 1000.0, 1), "stress_MPa": round(s_flight), "margin": round(allow / s_flight - 1, 2)},
                         "landing": {"g": LAND_G, "feet_carrying": 2, "foot_load_N": round(ld["landing_ultimate_N"], 1), "root_moment_Nm": round(ld["landing_ultimate_N"] * Lt / 1000.0, 1),
                                     "stress_MPa": round(s_land), "margin": round(allow / s_land - 1, 2),
                                     "note": "the tube alone; the fin's skin is a deep beam in this direction and is not counted"},
                         "first_bending_with_the_motor_Hz": round(f1, 1), "propeller_once_per_turn_Hz": {"at_6000_rpm": 100.0, "at_12000_rpm": 200.0}}}


def fin_cad(p=None):
    """One fin's tube along +y from the axis to the foot, cut at the body's side, the motor's mount, the air load's
    station and the foot."""
    ld = fin_loads(p)
    m, a = ld["motor_mm"], ld["air_station_mm"]
    cuts = sorted({0.0, ld["root_mm"], m - MOUNT_MM / 2, m + MOUNT_MM / 2, a - MOUNT_MM / 2, a + MOUNT_MM / 2, ld["tip_mm"] - FOOT_MM, ld["tip_mm"]})
    out = None
    for c0, c1 in zip(cuts[:-1], cuts[1:]):
        seg = TF._tube(c0, c1, FIN_SPAR["od"], FIN_SPAR["id"])
        out = seg if out is None else out.union(seg, clean=False)
    return out, cuts


def fin_regions(p=None):
    ld = fin_loads(p)
    r, m, a = FIN_SPAR["od"] / 2, ld["motor_mm"], ld["air_station_mm"]
    return [TF._box("root", 0.0, ld["root_mm"], r), TF._box("mount", m - MOUNT_MM / 2, m + MOUNT_MM / 2, r), TF._box("air", a - MOUNT_MM / 2, a + MOUNT_MM / 2, r),
            TF._box("foot", ld["tip_mm"] - FOOT_MM, ld["tip_mm"], r)]


FEACase, solve_one, regions_select = TF.FEACase, TF.solve_one, TF.regions_select
CASE_LIST = [("wing_spar_8g", "static", "the half wing's lift at 8 g, ultimate, on the swept spar"),
             ("fin_flight_8g", "static", "one fin's tube: full thrust, the motor's weight at 8 g and the fin's air load, ultimate"),
             ("fin_landing", "static", "one fin's tube alone: a 3 g landing on two feet, ultimate"),
             ("fin_modes", "modes", "the fin's tube with the motor as a point mass")]
AIRCRAFT, PREFIX, BAR = "ASTOR-X2", "astor_x2", "astor-x2 fea"


def cases(workdir, p=None, *, element_size=3.0):
    from vegeta import talos
    p = p or params()
    mat = talos.Material(CARBON_TUBE["name"], youngs_modulus=CARBON_TUBE["E"], poissons_ratio=CARBON_TUBE["nu"], density=CARBON_TUBE["rho"],
                         yield_strength=CARBON_TUBE["strength"], source=CARBON_TUBE["source"])
    sup = [talos.FixedSupport("root")]
    text = {n: t for n, _, t in CASE_LIST}
    spar, _ = XFE.spar_cad(p)
    step_w = TF.export_step(spar, Path(workdir) / "cad" / "astor_x2_wing_spar.step")
    wl = XFE.loads(p, mass=MASS_KG)
    out = [FEACase("wing_spar_8g", text["wing_spar_8g"], talos.StructuralModel(step_w, "mm-N-MPa", mat, XFE.spar_regions(p), sup,
                   [talos.Force(f"S{k}", fz=f) for k, f in enumerate(wl["lift_N"])], talos.MeshSettings(element_size=element_size, order=2), name="wing_spar_8g"))]
    fin, _ = fin_cad(p)
    step_f = TF.export_step(fin, Path(workdir) / "cad" / "astor_x2_fin_spar.step")
    ld = fin_loads(p)
    mesh = talos.MeshSettings(element_size=min(element_size, 2.0), order=2)              # the tube's wall is 1.5 mm
    reg = fin_regions(p)
    out.append(FEACase("fin_flight_8g", text["fin_flight_8g"], talos.StructuralModel(step_f, "mm-N-MPa", mat, reg, sup,
               [talos.Force("mount", fx=-ld["thrust_ultimate_N"], fz=-ld["motor_inertia_ultimate_N"]), talos.Force("air", fz=-ld["air_ultimate_N"])], mesh, name="fin_flight_8g")))
    out.append(FEACase("fin_landing", text["fin_landing"], talos.StructuralModel(step_f, "mm-N-MPa", mat, reg, sup,
               [talos.Force("foot", fx=-ld["landing_ultimate_N"])], mesh, name="fin_landing")))
    out.append(FEACase("fin_modes", text["fin_modes"], talos.StructuralModel(step_f, "mm-N-MPa", mat, reg, sup, [], mesh,
               masses=[talos.PointMass("mount", MOTOR_KG * 1e-3)], name="fin_modes"), kind="modes"))
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
        out.update({"frequencies_Hz": [round(x, 1) for x in f[:4]], "beam_reference_Hz": ref["fin_spar"]["first_bending_with_the_motor_Hz"],
                    "propeller_once_per_turn_Hz": ref["fin_spar"]["propeller_once_per_turn_Hz"]})
    else:
        vm, allow = r.metrics.get("max_von_mises"), CARBON_TUBE["strength"] * KNOCKDOWN
        beam = {k: ref["wing_spar"][k] for k in ("root", "step", "tip_deflection_ultimate_mm")} if case.name.startswith("wing") else \
            ref["fin_spar"]["landing" if case.name == "fin_landing" else "flight_8g"]
        out.update({"peak_von_mises_MPa": None if vm is None else float(vm), "max_displacement_mm": r.metrics.get("max_displacement"), "allowable_MPa": allow,
                    "margin": None if not vm else round(allow / float(vm) - 1, 2), "beam_reference": beam})
    return out


if __name__ == "__main__":
    import yaml
    import arigo1_model as am
    doc = {"beam_reference": reference(), "faces_selected_per_region": {"wing_spar": regions_select(XFE.spar_cad(params())[0], XFE.spar_regions(params())),
                                                                          "fin_spar": regions_select(fin_cad()[0], fin_regions())}}
    (am.DATA / "astor_x2").mkdir(parents=True, exist_ok=True)
    (am.DATA / "astor_x2" / "astor_x2_spar_reference.yaml").write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=140))
    print(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=140))
