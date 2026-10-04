"""The fixed wing's vibration and life (notebook 09b), lifted as it was. Wing (cells 6, 8, 15, 17, 20, 24, 25): the
operating points with their propeller unbalance, the nacelles as point masses, four unit cases, three missions, the
LW-PLA curve, the thinner-wing variant. Fuselage (cells 30, 32, 36, 38): the shell with the nose contents as a point
mass clamped at the wing, three unit cases, the full-battery survey with sharp-edged gusts.

09b analyses 09a's preferred wing, NACA 2415 (thickness 0.15, its hand-off), and compares the thinner NACA 2412: so do
these, whatever the flown aircraft's default. The notebook's globals are arguments; the hand-off's numbers come from
the fixed-wing tree. The FEA runs in ``workflows._common.unit_fea``, the life in ``life``.
"""
from __future__ import annotations

import math

from vegeta import boreas, chronos, talos

from . import wing

PREFERRED_THICKNESS, THIN_THICKNESS = 0.15, 0.12                   # 09b cell 4 (09a's preferred), cell 25 (NACA 2412)
WING_MODES, FUSELAGE_MODES = 8, 6                                   # cells 10, 34
WING_ELEMENT_MM = {"smoke": 20.0, "quick": 15.0, "full": 12.0}       # full: cell 8
FUSELAGE_ELEMENT_MM = {"smoke": 8.0, "quick": 6.0, "full": 5.0}      # full: cell 32
CURVE = talos.FatigueCurve("LW-PLA (assumed)", sigma_f=22.0, b=-0.12, ultimate=14.0, source="assumed; coupon tests needed")  # c20
USAGE = {"survey": 0.5, "patrol": 0.3, "windy_hops": 0.2}           # cell 24
N_FLIGHTS = 2000                                                     # cells 24, 41
NOSE_PARTS = ["battery 3S 5000 mAh", "flight controller + GPS", "payload (camera)", "receiver, wiring, bolts"]   # cell 30
G = 9.81


def material() -> talos.Material:
    return talos.Material(**wing.LW_PLA)                             # the same LW-PLA as 09b cell 4


# ------------------------------------------------------------------------------------------------ the wing
def prop_points(prop: boreas.Propeller, points: dict) -> dict:
    """Cell 6: rpm, thrust and unbalance force at cruise, climb and with one engine out."""
    return {name: {"rpm": pt.rpm, "thrust_N": pt.thrust, "unbalance_N": boreas.excitations(prop, pt.rpm)["unbalance_force_n"]}
            for name, pt in ((n, points[n]) for n in ("cruise", "climb", "engine_out"))}


def nacelle_mass_t(motor_kg: float, prop_kg: float) -> float:
    """Cell 6: motor + prop + ESC per nacelle, tonnes."""
    return (motor_kg + prop_kg + 0.025) * 1e-3


def lift_unit_mpa(auw_kg: float, wing_area_m2: float) -> float:
    """Cell 6: the lift pressure for n = 1 (level flight) in MPa."""
    return auw_kg * 9.81 / (wing_area_m2 * 1e6)


def wing_models(step, p: dict, nacelle_t: float, lift_unit: float, element_mm: float = 12.0):
    """Cells 8 and 15: the modal model (nacelle masses) and the unit cases ``{name: (model, level)}``."""
    regions = [wing.root_region(p["fuselage_diameter"]), talos.Surfaces("lift", wing.lower_skins(step)),
               *wing.motor_regions(p["nacelle_y"], p["nacelle_diameter"], p["nacelle_forward"])]
    masses = [talos.PointMass("motor_left", nacelle_t), talos.PointMass("motor_right", nacelle_t)]
    mesh = talos.MeshSettings(element_size=element_mm)
    mat = material()

    def model(loads, name):
        return talos.StructuralModel(step, "mm-N-MPa", mat, regions, [talos.FixedSupport("root")], loads, mesh, name=name,
                                     masses=masses)

    unit = {
        "lift":        (model([talos.Pressure("lift", lift_unit)], "lift"), 1.0),
        "thrust":      (model([talos.Force("motor_left", fx=6.0), talos.Force("motor_right", fx=6.0)], "thrust"), 6.0),
        "thrust_left": (model([talos.Force("motor_left", fx=8.0)], "thrust_left"), 8.0),
        "vib_left":    (model([talos.Force("motor_left", fz=1.0)], "vib_left"), 1.0),
    }
    return model([], "modal"), unit


def missions(PROP: dict) -> dict:
    """Cell 17 (the profile plots left out)."""
    def unb(point):
        return chronos.Excitation(f"unbalance {point}", PROP[point]["rpm"] / 60, PROP[point]["unbalance_N"], "vib_left")

    TC, TCL = PROP["cruise"]["thrust_N"], PROP["climb"]["thrust_N"]
    return {
        "survey": chronos.Mission("survey", (
            chronos.Segment("take-off + climb", 60, {"lift": 1.2, "thrust": TCL}, (unb("climb"),)),
            chronos.Segment("mapping legs", 2400, {"lift": 1.0, "thrust": TC}, (unb("cruise"),)),
            chronos.Segment("turn between legs", 8, {"lift": 1.3, "thrust": TC}, (unb("cruise"),), repeat=16),
            chronos.Segment("light turbulence", 2, {"lift": 1.25, "thrust": TC}, (unb("cruise"),), repeat=80),
            chronos.Segment("descent + landing", 90, {"lift": 0.9, "thrust": 0.3}),
            chronos.Segment("touchdown", 1, {"lift": 1.8}),
        ), "45 min mapping survey: long straight legs, gentle turns"),
        "patrol": chronos.Mission("patrol", (
            chronos.Segment("take-off + climb", 60, {"lift": 1.2, "thrust": TCL}, (unb("climb"),)),
            chronos.Segment("loiter", 1500, {"lift": 1.15, "thrust": TC}, (unb("cruise"),)),
            chronos.Segment("steep turn", 6, {"lift": 1.6, "thrust": TC}, (unb("cruise"),), repeat=60),
            chronos.Segment("evasive pull-up", 2, {"lift": 2.5, "thrust": TCL}, (unb("climb"),), repeat=4),
            chronos.Segment("engine-out drill", 30, {"lift": 1.0, "thrust_left": PROP["engine_out"]["thrust_N"]}, (unb("engine_out"),)),
            chronos.Segment("descent + landing", 90, {"lift": 0.9, "thrust": 0.3}),
            chronos.Segment("touchdown", 1, {"lift": 2.0}),
        ), "30 min patrol: continuous loiter turns, a few hard pull-ups, one engine-out drill"),
        "windy_hops": chronos.Mission("windy_hops", (
            chronos.Segment("take-off + climb", 45, {"lift": 1.3, "thrust": TCL}, (unb("climb"),), repeat=6),
            chronos.Segment("short cruise", 300, {"lift": 1.0, "thrust": TC}, (unb("cruise"),), repeat=6),
            chronos.Segment("gust", 1.5, {"lift": 1.7, "thrust": TC}, (unb("cruise"),), repeat=300),
            chronos.Segment("hard touchdown", 1, {"lift": 2.5}, repeat=6),
        ), "six short hops in gusty wind: 300 gusts, six hard touchdowns, ~35 min"),
    }


# ------------------------------------------------------------------------------------------------ the fuselage
def nose_kg(parts_g: dict) -> float:
    """Cell 30: what sits in the nose [kg]."""
    return sum(parts_g[k] for k in NOSE_PARTS) / 1000


def fuselage_mass_kg(volume_mm3: float) -> float:
    """Cell 30: the fuselage + tail shell in LW-PLA [kg] (t/mm^3 x mm^3 = t -> kg)."""
    return volume_mm3 * material().density * 1e3


def fuselage_models(step, pf: dict, nose: float, element_mm: float = 5.0):
    """Cells 30, 32 and 36: the modal model (nose contents as a point mass, clamped at the wing saddle) and the unit
    cases ``{name: (model, level)}``."""
    r_f, tt = pf["fuselage_diameter"] / 2, pf["tail_thickness"]
    x_nose, x_cyl_end = -pf["nose_length"], -pf["nose_length"] + 0.55 * pf["fuselage_length"]
    xt, tc = -pf["nose_length"] + pf["fuselage_length"] - pf["tail_chord"], pf["tail_chord"]
    x_te = min(pf["root_chord"], x_cyl_end - 1.0)                          # the fuselage faces are split at the wing LE and TE
    regions = [
        talos.SurfacesInBox("saddle", (-0.5, -r_f - 1, -2 * r_f - 1, x_te + 0.5, r_f + 1, 1.0)),            # cylinder under the wing
        talos.SurfacesInBox("nose", (x_nose - 1, -r_f - 1, -2 * r_f - 1, 0.5, r_f + 1, 1.0)),                # nose + cylinder ahead of the wing
        talos.SurfacesInBox("hstab", (xt - 0.5, -pf["tail_span"] / 2 - 1, -r_f - tt / 2 - 0.5, xt + tc + 0.5, pf["tail_span"] / 2 + 1, -r_f + tt / 2 + 0.5)),
        talos.SurfacesInBox("fin", (xt - 0.5, -tt / 2 - 0.5, -r_f - 0.5, xt + tc + 0.5, tt / 2 + 0.5, -r_f + pf["fin_height"] + 0.5)),
    ]
    masses = [talos.PointMass("nose", nose * 1e-3)]                         # tonnes in mm-N-MPa
    mesh = talos.MeshSettings(element_size=element_mm)
    mat = material()

    def fmodel(loads, name):
        return talos.StructuralModel(step, "mm-N-MPa", mat, regions, [talos.FixedSupport("saddle")], loads, mesh, name=name,
                                     masses=masses)

    unit = {
        "inertia":   (fmodel([talos.Acceleration(az=-9810.0), talos.Force("nose", fz=-nose * 9.81)], "inertia"), 1.0),
        "tail_lift": (fmodel([talos.Force("hstab", fz=5.0)], "tail_lift"), 5.0),
        "fin_side":  (fmodel([talos.Force("fin", fy=3.0)], "fin_side"), 3.0),
    }
    return fmodel([], "fuselage_modal"), unit


def long_mission(pf: dict, W: float, S: float, RHO: float, V_CRUISE: float, endurance_min: float):
    """Cell 38: the full-battery survey with sharp-edged gusts; returns ``(mission, gusts)``."""
    S_T = pf["tail_chord"] * pf["tail_span"] / 1e6                    # m^2 horizontal tail
    S_F = pf["tail_chord"] * pf["fin_height"] / 1e6                   # m^2 fin
    A_W, A_T, A_F = 2 * math.pi * 0.8, 3.5, 2.5                       # lift slopes [1/rad]: wing (AR ~6), low-AR plates (assumed)
    V = V_CRUISE

    def dn_gust(u):  return 0.5 * RHO * V * A_W * S * u / W         # wing load-factor increment
    def tail_gust(u): return 0.5 * RHO * V * A_T * S_T * u           # N on the horizontal tail
    def fin_gust(u):  return 0.5 * RHO * V * A_F * S_F * u           # N on the fin

    TAIL_TRIM = -0.03 * W                                             # N: small balancing download (assumed)
    T_FLIGHT = 0.9 * endurance_min * 60                               # s: a full battery (09a endurance, 10 % reserve)
    U_L, U_M, GUST_S = 2.0, 5.0, 0.5
    N_LIGHT, N_SIDE, N_MOD, LEGS = int(0.2 * T_FLIGHT), int(0.1 * T_FLIGHT), 20, 12
    T_LEGS = T_FLIGHT - 150 - LEGS * 8 - (N_LIGHT + N_SIDE + N_MOD) * GUST_S

    m = chronos.Mission("long_survey", (
        chronos.Segment("take-off + climb", 60, {"inertia": 1.2, "tail_lift": 1.3 * TAIL_TRIM}),
        chronos.Segment("survey legs", T_LEGS, {"inertia": 1.0, "tail_lift": TAIL_TRIM}),
        chronos.Segment("turn between legs", 8, {"inertia": 1.3, "tail_lift": 1.3 * TAIL_TRIM, "fin_side": 0.5 * fin_gust(U_L)}, repeat=LEGS),
        chronos.Segment("light vertical gust", GUST_S, {"inertia": 1 + dn_gust(U_L), "tail_lift": TAIL_TRIM + tail_gust(U_L)}, repeat=N_LIGHT),
        chronos.Segment("side gust", GUST_S, {"inertia": 1.0, "tail_lift": TAIL_TRIM, "fin_side": fin_gust(U_L)}, repeat=N_SIDE),
        chronos.Segment("moderate gust", GUST_S, {"inertia": 1 + dn_gust(U_M), "tail_lift": TAIL_TRIM + tail_gust(U_M), "fin_side": fin_gust(U_M)}, repeat=N_MOD),
        chronos.Segment("descent + landing", 90, {"inertia": 0.9, "tail_lift": TAIL_TRIM}),
        chronos.Segment("touchdown", 1, {"inertia": 3.0}),
    ), f"full-battery survey ({T_FLIGHT / 60:.0f} min): {LEGS} legs, {N_LIGHT} light and {N_MOD} moderate vertical gusts, {N_SIDE} side gusts, one touchdown")
    gusts = {"light_m_s": U_L, "moderate_m_s": U_M, "slopes_per_rad": {"wing": A_W, "tail": A_T, "fin": A_F},
             "tail_trim_N": TAIL_TRIM, "dn_light": dn_gust(U_L), "dn_moderate": dn_gust(U_M)}
    return m, gusts


def fuselage_bound(fspec: chronos.LoadSpectrum, unit_rows: dict) -> float:
    """Cell 41: every unit case at its own peak level, added (conservative) [MPa]."""
    fpeak = {}
    for b in fspec.blocks:
        fpeak[b.pattern] = max(fpeak.get(b.pattern, 0.0), abs(b.mean) + abs(b.amplitude))
    return sum(fpeak.get(k, 0.0) * row["max_von_mises_MPa"] / row["load"] for k, row in unit_rows.items())
