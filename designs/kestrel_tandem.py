"""Kestrel-Q4T — the tandem-wing Kestrel (design brief: ``kestrel_tandem.md``, notebook 46).

Four propellers on two equal wings one behind the other, a pod between them, no bracing. ``VARIANTS``:
- **Q4T1N-10** — front wing low, rear wing high (the Quickie arrangement), span 1.0 m: *the design*;
- **Q4T1P-10** — the mirror: front wing high, rear wing low;
- **Q4T1N-12** — Q4T1N at span 1.2 m (the span a horizontal start frees);
- **Q4T1-10 (flat)** — both wings in one plane: the control case (Munk: +40 % induced drag; no pitch lever in the hover).

What this module computes, and from what:
- ``layout``: the wings' stations (front leading edge, stagger, heights), the motors (tractor on each wing, the low
  wing's on pylons for the propellers' ground clearance on the skids), the pod (longer by the stagger), the keys the
  transition model, the control allocation and the drawings read.
- ``mass_table``: Kestrel's parts (``kestrel_systems.COMPONENTS``) on this layout: two wings with one Ø10×8 carbon
  spar each, four nacelles, two pylons, skids, a fin, four elevon servos, the pod shell scaled with its length; the
  battery placed for the CG the tandem's trim wants (``static_margin`` of the stagger ahead of the neutral point).
- ``aero``: the lattice (``peregrine_flight.vlm``) on the two staggered wings with the front wing's decalage: the lift
  slope, the induced-drag factor (Munk: the gap decides it, not the stagger), the neutral point, the front wing's
  share of the lift, each wing's effective slope (the rear in the front's downwash).
- ``stall``: the tandem's CL max rule — the front wing stalls first (the CG ahead of the pair's neutral point loads it
  harder), the aircraft's CL max is the section's × (front area / total) / (front share at trim).
- ``drag_buildup``: no struts; the pod's extra length, the pylons, the skids, the fin.
- ``polar`` / ``make_vehicle`` / ``cruise`` / ``takeoff_roll``: the full-range polar, the vehicle for
  ``kestrel_transition`` (four half-wing panels at their own stations: the rear ones with the downwash factor, the
  front ones with the decalage; the elevator is the rear pair), level flight trimmed by the elevator, the ground roll.
- ``cantilever``: one half-wing as a cantilever with its motor, the Ø10×8 tube: stress, deflection, the first mode.
- ``KestrelQ4T``: the CAD (Dedalus, on ``kestrel.Kestrel4Q``'s parts) for the drawings and the CFD's STL.

Units SI (metres, radians) unless named _deg / _mm; the CAD parameters in millimetres as ``kestrel.py``.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import kestrel
import kestrel_systems as ks

G = ks.G
RHO = 1.225
NU = 1.46e-5
MAIN = "Q4T1N-10"
VARIANTS = {
    "Q4T1N-10": {"span": 1.0, "chord": 0.14, "gap": 0.28, "stagger_c": 3.0, "sign": -1, "prop": "10x7", "battery_ah": 3.3, "title": "front wing low, rear high — the design"},
    "Q4T1P-10": {"span": 1.0, "chord": 0.14, "gap": 0.28, "stagger_c": 3.0, "sign": +1, "prop": "10x7", "battery_ah": 3.3, "title": "front wing high, rear low — the mirror"},
    "Q4T1N-12": {"span": 1.2, "chord": 0.14, "gap": 0.28, "stagger_c": 3.0, "sign": -1, "prop": "10x7", "battery_ah": 3.3, "title": "the design at span 1.2 m"},
    "Q4T1-10 (flat)": {"span": 1.0, "chord": 0.14, "gap": 0.0, "stagger_c": 3.0, "sign": 0, "prop": "10x7", "battery_ah": 3.3, "title": "both wings in one plane — the control case"},
}
# the fixed numbers of the family
POD_WIDTH = 0.110
X_LE_FRONT = 0.200                # the front wing's leading edge from the nose tip (the pod's nose ahead of it)
PROP_AHEAD = 0.075                # each propeller plane ahead of its wing's leading edge
MOTOR_Y = 0.350                   # the motors' spanwise station (4Q: 0.39 at span 1.0; 0.35 keeps a 10-inch disc inside the tip)
NACELLE_DIAMETER = 0.034
DECALAGE_DEG = 2.0                # the front wing set this much more nose-up than the rear: it stalls first
STATIC_MARGIN = 0.10              # of the stagger, ahead of the pair's neutral point: the CG the battery is placed for
TAIL_LENGTH = 0.100               # the pod's tail taper behind the rear wing's trailing edge
PYLON = 0.050                     # the low wing's motors raised by this much (the propellers' ground clearance)
SKID_HEIGHT = 0.100               # the skids below the pod's bottom
POD_SKIN_OVER_WING = 0.030        # the pod extends this far above the upper and below the lower wing
FIN_HEIGHT = 0.120                # a small fin on the rear wing's centre: the weathercock the tailsitter's plate gave
PROPS = {"9x6": (0.2286, 0.014), "10x7": (0.254, 0.018)}
SECTION_CL_MAX = 1.45             # a cambered NACA 2412-class section at Re ~1.5e5 (×0.9 for the built-up wing, as kestrel_fast)
WING_CL_MAX = 0.9 * SECTION_CL_MAX
ELEVON_CF = 0.28
ELEVON_TAU = 0.45
MU_ROLL = 0.08                    # rolling friction of skids on short grass
V_CRUISE = 16.0
AVIONICS_W = 13.0


def arrangement(sign: int) -> str:
    return {-1: "N (front low, rear high)", 1: "P (front high, rear low)", 0: "flat (one plane)"}[int(sign)]


# ================================================================================================= the layout
def layout(variant: str = MAIN, **over) -> dict:
    v = {**VARIANTS[variant], **over}
    b, c, g, sgn = v["span"], v["chord"], v["gap"], int(v["sign"])
    stagger = v["stagger_c"] * c
    D, m_prop = PROPS[v["prop"]]
    R = D / 2
    z_f, z_r = sgn * g / 2, -sgn * g / 2                 # P: front high; N: front low; flat: both 0
    x_le_f, x_le_r = X_LE_FRONT, X_LE_FRONT + stagger
    low = {"front": z_f <= z_r, "rear": z_r <= z_f}       # the low wing's motors sit on pylons (both, when flat)
    pylon = {"front": PYLON if low["front"] else 0.0, "rear": PYLON if low["rear"] else 0.0}
    motors = {}
    for name, (sy, wing) in {"M1": (-1, "front"), "M2": (1, "front"), "M3": (-1, "rear"), "M4": (1, "rear")}.items():
        x_le, z = (x_le_f, z_f) if wing == "front" else (x_le_r, z_r)
        motors[name] = {"x": x_le - PROP_AHEAD, "y": sy * MOTOR_Y, "z": z + pylon[wing], "wing": wing}
    pod_top, pod_bottom = max(max(z_f, z_r) + POD_SKIN_OVER_WING, POD_WIDTH / 2), min(min(z_f, z_r) - POD_SKIN_OVER_WING, -POD_WIDTH / 2)   # at least a square pod (the flat tandem)
    ground_z = pod_bottom - SKID_HEIGHT
    prop_lowest = min(mo["z"] for mo in motors.values()) - R
    pod_length = x_le_r + c + TAIL_LENGTH
    S_wing = b * c
    return {
        "variant": variant, "title": v["title"], "arrangement": arrangement(sgn), "sign": sgn, "span": b, "chord": c, "gap": g, "h": g / 2, "stagger": stagger,
        "stagger_c": v["stagger_c"], "decalage": math.radians(DECALAGE_DEG), "prop": v["prop"], "battery_ah": v["battery_ah"],
        "x_le": x_le_f, "x_te": x_le_f + c, "x_ac": x_le_f + 0.25 * c, "x_le_front": x_le_f, "x_le_rear": x_le_r, "x_ac_front": x_le_f + 0.25 * c,
        "x_ac_rear": x_le_r + 0.25 * c, "x_elevator": x_le_r, "z_front": z_f, "z_rear": z_r, "x_spar": x_le_f + 0.30 * c, "x_spar_rear": x_le_r + 0.30 * c,
        "S_wing": S_wing, "S_ref": 2 * S_wing, "AR_wing": b / c, "AR_total": b * b / (2 * S_wing), "gap_chord": g / c, "gap_span": g / b,
        "R_prop": R, "A_disc": math.pi * R ** 2, "prop_mass": m_prop, "motors": motors, "motor_y": MOTOR_Y, "prop_x": x_le_f - PROP_AHEAD, "prop_x_rear": x_le_r - PROP_AHEAD,
        "prop_ahead_of_le": PROP_AHEAD, "pylon": pylon, "prop_to_pod": MOTOR_Y - R - POD_WIDTH / 2, "prop_beyond_tip": MOTOR_Y + R - b / 2,
        "prop_ground_clearance": prop_lowest - ground_z, "pod_length": pod_length, "pod_width": POD_WIDTH, "pod_height": pod_top - pod_bottom,
        "pod_top": pod_top, "pod_bottom": pod_bottom, "ground_z": ground_z, "skid_height": SKID_HEIGHT, "fin_height": FIN_HEIGHT,
        "elevons": True, "elevon_cf": ELEVON_CF, "elevon_y_in": max(MOTOR_Y - R, POD_WIDTH / 2 + 0.02), "elevon_span_side": b / 2 - max(MOTOR_Y - R, POD_WIDTH / 2 + 0.02),
        "x_hinge": x_le_f + (1 - ELEVON_CF) * c, "x_hinge_rear": x_le_r + (1 - ELEVON_CF) * c, "S_elevon": 4 * (b / 2 - max(MOTOR_Y - R, POD_WIDTH / 2 + 0.02)) * ELEVON_CF * c,
        "bracing": "none", "posts": 0, "diagonals": 0, "hover_pitch_lever": abs(motors["M1"]["z"] - motors["M3"]["z"]),
    }


def checks(L: dict) -> list[str]:
    out = []
    if L["prop_to_pod"] < 0.025:
        out.append(f"the propeller discs clear the pod by {L['prop_to_pod'] * 1e3:.0f} mm (< 25 mm)")
    if L["prop_beyond_tip"] > 0.02:
        out.append("the propellers overhang the tips")
    if L["prop_ground_clearance"] < 0.03:
        out.append(f"the low propellers clear the ground by {L['prop_ground_clearance'] * 1e3:.0f} mm (< 30 mm): taller skids or pylons")
    if L["stagger"] < L["chord"] * 1.5:
        out.append("the rear wing's leading edge is inside 1.5 chords of the front wing's trailing edge")
    if L["sign"] == 0:
        out.append("no vertical lever between the rotors: the hover's pitch control is the slipstream elevons alone")
    return out


# ================================================================================================= masses
def mass_table(L: dict, *, battery_x: float | None = None, x_cg_target: float | None = None, payload: bool = True) -> list[dict]:
    """Every item with mass and position; the battery where it puts the CG at ``x_cg_target`` (default: ``cg_target``),
    clipped to the pod's battery bay and flagged when it is."""
    c = ks.COMPONENTS
    items = []

    def add(name, m, x, y, z, group, rod=None):
        items.append({"item": name, "mass_kg": m, "x": x, "y": y, "z": z, "group": group, "rod": rod})

    for k, mo in L["motors"].items():
        x_le = L["x_le_front"] if mo["wing"] == "front" else L["x_le_rear"]
        add(f"motor {k}", c["motor"]["mass_kg"], mo["x"] + 0.025, mo["y"], mo["z"], "propulsion")
        add(f"propeller {k}", L["prop_mass"], mo["x"], mo["y"], mo["z"], "propulsion")
        add(f"ESC {k}", c["esc"]["mass_kg"], x_le + 0.02, mo["y"], mo["z"], "propulsion")
        add(f"nacelle {k}", ks.NACELLE_KG, 0.5 * (mo["x"] + x_le + 0.6 * L["chord"]), mo["y"], mo["z"], "structure")
        if L["pylon"][mo["wing"]]:
            add(f"pylon {k}", 0.008, x_le + 0.3 * L["chord"], mo["y"], mo["z"] - L["pylon"][mo["wing"]] / 2, "structure")
    for nm, x_le, z in (("front", L["x_le_front"], L["z_front"]), ("rear", L["x_le_rear"], L["z_rear"])):
        add(f"wing {nm}", ks.WING_AREAL_KG_M2 * L["S_wing"], x_le + 0.42 * L["chord"], 0.0, z, "structure", ("y", L["span"]))
        add(f"spar {nm}", ks.SPAR_KG_M * L["span"], x_le + 0.30 * L["chord"], 0.0, z, "structure", ("y", L["span"]))
        y_mid = 0.5 * (L["elevon_y_in"] + L["span"] / 2)
        for s in (-1, 1):
            add(f"elevon servo {nm} {'L' if s < 0 else 'R'}", ks.SERVO_KG, x_le + (1 - ELEVON_CF) * L["chord"] - 0.02, s * y_mid, z, "systems")
    add("spar sockets (4)", 4 * ks.JOINT_KG, 0.5 * (L["x_spar"] + L["x_spar_rear"]), 0.0, 0.0, "structure")
    add("pod shell", ks.POD_SHELL_KG * L["pod_length"] / 0.47, 0.45 * L["pod_length"], 0.0, 0.5 * (L["pod_top"] + L["pod_bottom"]), "structure")
    add("skids (2)", 2 * 0.020, 0.45 * L["pod_length"], 0.0, L["pod_bottom"] - L["skid_height"] / 2, "structure")
    add("fin", ks.FIN_KG_M2 * 0.5 * L["fin_height"] * (0.8 * L["chord"]), L["x_le_rear"] + 0.6 * L["chord"], 0.0, L["z_rear"] + L["fin_height"] / 2, "structure")
    add("wiring", ks.WIRING_KG * 1.15, 0.5 * (L["x_le_front"] + L["x_le_rear"]), 0.0, 0.0, "systems")
    add("flight stack", c["flight_stack"]["mass_kg"], 0.55 * L["pod_length"], 0.0, 0.03, "systems")
    add("camera", c["camera"]["mass_kg"], 0.02, 0.0, 0.0, "payload")
    if payload:
        add("mission computer", c["mission_computer"]["mass_kg"], 0.40 * L["pod_length"], 0.0, 0.03, "payload")
    add("growth margin", ks.MARGIN_KG, 0.5 * (L["x_ac_front"] + L["x_ac_rear"]), 0.0, 0.0, "margin")
    mb = c["battery"]["mass_kg"] * L["battery_ah"] / c["battery"]["capacity_ah"]
    half = c["battery"]["length_m"] / 2
    lo, hi = 0.090 + half, L["pod_length"] - 0.060 - half
    clipped = False
    if battery_x is None:
        target = cg_target(L) if x_cg_target is None else x_cg_target
        m = sum(i["mass_kg"] for i in items)
        mx = sum(i["mass_kg"] * i["x"] for i in items)
        battery_x = (target * (m + mb) - mx) / mb
        if not lo <= battery_x <= hi:
            clipped = True
            battery_x = float(np.clip(battery_x, lo, hi))
    items.append({"item": "battery", "mass_kg": mb, "x": battery_x, "y": 0.0, "z": -0.02, "group": "propulsion", "rod": None, "clipped": clipped})
    return items


def cg_target(L: dict) -> float:
    """The CG the tandem's trim wants: ``STATIC_MARGIN`` of the stagger ahead of the pair's neutral point."""
    return aero(L)["x_np"] - STATIC_MARGIN * L["stagger"]


# ================================================================================================= aerodynamics
def quads(L: dict) -> list:
    b2 = L["span"] / 2
    out = []
    for x_le, z in ((L["x_le_front"], L["z_front"]), (L["x_le_rear"], L["z_rear"])):
        out.append(np.array([[x_le, -b2, z], [x_le, b2, z], [x_le + L["chord"], b2, z], [x_le + L["chord"], -b2, z]]))
    return out


_AERO_CACHE: dict = {}


def aero(L: dict, *, n_span: int = 16, n_chord: int = 3) -> dict:
    """The lattice on the two wings (front with the decalage): CLα on S_ref, k = CDi/CL² and e on the span, the neutral
    point, the front wing's share of the lift at a cruise-like α, each wing's effective slope against one wing alone
    (the rear wing's is the downwash factor the transition model uses), CL at α = 0 from the decalage."""
    import peregrine_flight as pf
    key = (L["variant"], L["span"], L["chord"], L["gap"], L["stagger"], L["sign"], n_span, n_chord)
    if key in _AERO_CACHE:
        return dict(_AERO_CACHE[key])
    qs = quads(L)
    S, c = L["S_ref"], L["chord"]
    x_ref = L["x_ac_front"]
    inc = [DECALAGE_DEG, 0.0]
    kw = dict(S_ref=S, c_ref=c, x_ref=x_ref, n_span=n_span, n_chord=n_chord)
    r0 = pf.vlm(qs, alpha_deg=0.0, incidence_deg=inc, **kw)
    r1 = pf.vlm(qs, alpha_deg=4.0, incidence_deg=inc, **kw)
    da = math.radians(4.0)
    CLa = (r1["CL"] - r0["CL"]) / da
    Cma = (r1["Cm"] - r0["Cm"]) / da
    k = (r1["CDi"] - r0["CDi"]) / (r1["CL"] ** 2 - r0["CL"] ** 2)
    x_np = x_ref - Cma / CLa * c
    lat, F0, F1 = r1["lattice"], r0["F"], r1["F"]
    front1 = lat["quad"] == 0
    dL_front = F1[front1, 2].sum() - F0[front1, 2].sum()
    dL_rear = F1[~front1, 2].sum() - F0[~front1, 2].sum()
    front_share_alpha4 = float(F1[front1, 2].sum() / F1[:, 2].sum())
    # one wing alone, the same lattice: the slopes against it
    one = pf.vlm([qs[0]], alpha_deg=4.0, S_ref=L["S_wing"], c_ref=c, x_ref=x_ref, n_span=n_span, n_chord=n_chord)
    one0 = pf.vlm([qs[0]], alpha_deg=0.0, S_ref=L["S_wing"], c_ref=c, x_ref=x_ref, n_span=n_span, n_chord=n_chord)
    CLa_one = (one["CL"] - one0["CL"]) / da
    slope_front = (dL_front / (S / 2) / da) / CLa_one
    slope_rear = (dL_rear / (S / 2) / da) / CLa_one
    AR = L["span"] ** 2 / S
    out = {"CLa": CLa, "CLa_one_wing": CLa_one, "k": k, "e": 1 / (math.pi * AR * k), "k_one_wing": one["CDi"] / one["CL"] ** 2, "x_np": x_np, "Cma": Cma,
           "CL0": r0["CL"], "Cm0": r0["Cm"], "front_share_alpha4": front_share_alpha4, "slope_front": slope_front, "slope_rear": slope_rear,
           "downwash_factor_rear": slope_rear, "x_np_c_from_front_le": (x_np - L["x_le_front"]) / c, "AR_total": AR}
    _AERO_CACHE[key] = dict(out)
    return out


def stall(L: dict, a: dict, x_cg: float) -> dict:
    """The tandem's stall rule. At trim the front wing carries ``share_f`` = (x_ac,rear − x_cg) / stagger of the lift
    (moments about the CG, both wings' lift at their quarter chords); each wing's section lift coefficient is its share
    × S / S_wing × CL. The front wing reaches ``WING_CL_MAX`` first when share_f > ½ (equal wings): the aircraft's CL
    max = WING_CL_MAX × ½ / share_f, the rear wing then at WING_CL_MAX × (1 − share_f) / share_f."""
    share_f = (L["x_ac_rear"] - x_cg) / (L["x_ac_rear"] - L["x_ac_front"])
    share_f = float(np.clip(share_f, 0.05, 0.95))
    first = "front" if share_f >= 0.5 else "rear"
    cl_max = WING_CL_MAX * 0.5 / max(share_f, 1 - share_f)
    return {"front share at trim": share_f, "stalls first": first, "CL max (aircraft)": cl_max, "rear wing cl at the front's stall": WING_CL_MAX * (1 - share_f) / share_f,
            "static margin [c]": (a["x_np"] - x_cg) / L["chord"], "static margin [stagger]": (a["x_np"] - x_cg) / L["stagger"],
            "nose-down at the stall": first == "front"}


def drag_buildup(L: dict, *, V: float = V_CRUISE, t: float = 0.12) -> dict:
    """Zero-lift drag areas [m²] and Cd0 on S_ref: two wings, the pod (its length), four nacelles, the pylons, the skids,
    the fin, the camera dome; a 10 % interference allowance. No struts."""
    import kestrel_flight as kf
    rows = {}
    ff_wing = 1 + 0.6 / 0.3 * t + 100 * t ** 4
    rows["wings (2)"] = kf._cf(V * L["chord"] / NU) * ff_wing * 2 * L["S_ref"] * (1 + 0.2 * t)
    lf, df = L["pod_length"], math.sqrt(L["pod_width"] * L["pod_height"] * 0.6)        # as the biplane's build-up
    fr = lf / df
    rows["pod"] = kf._cf(V * lf / NU) * (1 + 60 / fr ** 3 + fr / 400) * math.pi * df * lf * 0.75
    dn, ln = NACELLE_DIAMETER, PROP_AHEAD + 0.6 * L["chord"]
    rows["nacelles (4)"] = 4 * (kf._cf(V * ln / NU) * 1.3 * math.pi * dn * ln + 0.1 * math.pi * dn ** 2 / 4)
    n_pyl = sum(1 for mo in L["motors"].values() if L["pylon"][mo["wing"]])
    rows[f"pylons ({n_pyl}, faired)"] = n_pyl * kf.CD_FAIRED * PYLON * 0.012 + n_pyl * kf._cf(V * 0.08 / NU) * 2 * 0.08 * PYLON
    rows["skid legs (2, faired) and runners"] = 2 * kf.CD_FAIRED * L["skid_height"] * 0.008 + 2 * kf._cf(V * 0.5 * L["pod_length"] / NU) * 1.2 * 2 * 0.5 * L["pod_length"] * 0.012
    rows["fin"] = kf._cf(V * 0.8 * L["chord"] / NU) * 1.2 * 2 * 0.5 * L["fin_height"] * 0.8 * L["chord"]
    rows["camera dome"] = 0.4 * math.pi * 0.018 ** 2
    total = sum(rows.values()) * kf.INTERFERENCE
    return {"parts_m2": rows, "CdA_m2": total, "Cd0": total / L["S_ref"], "Cd0_wings": rows["wings (2)"] * kf.INTERFERENCE / L["S_ref"]}


def polar(L: dict, a: dict | None = None, d: dict | None = None, x_cg: float | None = None):
    """The full-range polar on S_ref: the lattice's slope, the build-up's Cd0, k over the viscous Oswald factor, the
    tandem's CL max (``stall``), the flat-plate tail, α₀ from the decalage."""
    import kestrel_flight as kf
    a = a or aero(L)
    d = d or drag_buildup(L)
    x_cg = cg_target(L) if x_cg is None else x_cg
    st = stall(L, a, x_cg)
    return kf.Polar(cl_alpha=a["CLa"], cd0=d["Cd0"], k=a["k"] / kf.OSWALD_VISCOUS, cl_max=st["CL max (aircraft)"], cd90=kf.viterna_cd90(L["AR_wing"]),
                    alpha0=-a["CL0"] / a["CLa"], source=f"tandem lattice + build-up; CL max by the front-first stall rule ({st['front share at trim']:.2f} front share)")


# ================================================================================================= the drive, the vehicle, the cruise
def drive(L: dict):
    """The drive table of the variant's propeller and pack (``kestrel_systems.drive_table_for``: cached per key)."""
    import kestrel_fast as kfa
    if L["prop"] == "9x6" and abs(L["battery_ah"] - ks.COMPONENTS["battery"]["capacity_ah"]) < 1e-9:
        return ks.drive_table()
    c = kfa.Candidate("tandem", prop=L["prop"], battery_ah=L["battery_ah"])
    return ks.drive_table_for(kfa.propulsion(c), f"{L['prop']}_{L['battery_ah']:g}Ah")


def panels(L: dict, a: dict) -> list:
    """The four half-wings for ``kestrel_transition``: each at its own station and height, blown by its own rotor
    (``up`` = the higher wing's, the differential pair the controller pitches with); the rear ones carry the downwash
    factor, the front ones the decalage."""
    hi = "front" if L["z_front"] > L["z_rear"] else "rear"
    out = []
    for wing, x_le, z in (("front", L["x_le_front"], L["z_front"]), ("rear", L["x_le_rear"], L["z_rear"])):
        for sy in (-1, 1):
            out.append({"phi": 0.0, "sy": sy, "z": z, "y0": POD_WIDTH / 2, "y1": L["span"] / 2, "y_motor": MOTOR_Y, "rotor": "up" if wing == hi else "lo",
                        "S": L["S_wing"] / 2, "x_ac": x_le + 0.25 * L["chord"], "x_le": x_le, "prop_x": x_le - PROP_AHEAD,
                        "cl_scale": a["slope_rear"] if wing == "rear" else a["slope_front"], "incidence": L["decalage"] if wing == "front" else 0.0, "wing": wing})
    return out


def make_vehicle(variant: str = MAIN, *, name: str | None = None, extra_kg: float = 0.0, x_cg: float | None = None, elevons: bool = True, L: dict | None = None):
    """The tandem for ``kestrel_transition.simulate``. The panel polar is one wing's (its own slope; the panels' factors
    scale it), the whole aircraft's CL max is the stall rule's."""
    import kestrel_flight as kf
    import kestrel_transition as kt
    L = L or layout(variant)
    a = aero(L)
    table = mass_table(L, x_cg_target=x_cg)
    if extra_kg:
        table.append({"item": "extra payload", "mass_kg": extra_kg, "x": 0.5 * (L["x_ac_front"] + L["x_ac_rear"]), "y": 0.0, "z": 0.0, "group": "payload", "rod": None})
    ci = ks.cg_inertia(table)
    d = drag_buildup(L)
    st = stall(L, a, ci["x_cg"])
    pol_panel = kf.Polar(cl_alpha=a["CLa_one_wing"], cd0=d["Cd0"], k=a["k"] / kf.OSWALD_VISCOUS, cl_max=WING_CL_MAX, cd90=kf.viterna_cd90(L["AR_wing"]), alpha0=0.0,
                         source="one wing's slope; the panels' factors make the pair")
    p_cad = {"elevons": elevons, "prop_diameter": 2e3 * L["R_prop"], "motor_y": 1e3 * MOTOR_Y, "span": 1e3 * L["span"], "chord": 1e3 * L["chord"],
             "gap": 1e3 * max(L["gap"], 0.1), "pod_length": 1e3 * L["pod_length"]}      # what kestrel_flight's propeller normal force reads (a biplane-shaped parameter set)
    veh = kt.Vehicle(name or variant, p_cad, L, ci["mass_kg"], ci["x_cg"], ci["z_cg"], ci["Iyy"], ci["Ixx"], pol_panel, drive(L),
                     elevons=elevons, panels=panels(L, a), z_rotor=L["hover_pitch_lever"] / 2, elevon_y_in=L["elevon_y_in"], lift_slope_vertical=a["CLa"])
    veh.aircraft_cl_max = st["CL max (aircraft)"]
    veh.stall_rule = st
    veh.aircraft_polar = polar(L, a, d, ci["x_cg"])
    return veh


def elevator_power(L: dict, a: dict, x_cg: float) -> dict:
    """Cm per radian of the rear pair as the elevator (and of the front pair, the other way) about the CG, on S_ref·c;
    the biplane's elevons at a fraction of a chord for scale."""
    S_e = 2 * L["elevon_span_side"] * ELEVON_CF * L["chord"]           # one wing's pair
    arm_r = (L["x_le_rear"] + 0.6 * L["chord"]) - x_cg
    arm_f = (L["x_le_front"] + 0.6 * L["chord"]) - x_cg
    cm_r = -a["CLa_one_wing"] * a["slope_rear"] * ELEVON_TAU * (S_e / L["S_ref"]) * arm_r / L["chord"]
    cm_f = -a["CLa_one_wing"] * a["slope_front"] * ELEVON_TAU * (S_e / L["S_ref"]) * arm_f / L["chord"]
    return {"Cm_delta rear pair [1/rad]": cm_r, "Cm_delta front pair [1/rad]": cm_f, "rear arm [c]": arm_r / L["chord"], "front arm [c]": arm_f / L["chord"],
            "Cm_delta both, opposed [1/rad]": cm_r - cm_f}


def cruise(veh, *, speeds=np.arange(9.0, 27.0, 1.0), avionics_W: float = AVIONICS_W):
    """Level flight at each speed, the pitch trimmed by the rear elevator (its deflection and the lift it costs), the
    four drives at the drag, their electrical power (the variant's drive table) and the endurance on the usable pack."""
    import pandas as pd
    L, d = veh.L, veh.drive
    pol = veh.aircraft_polar
    a = aero(L)
    S, c, W = L["S_ref"], L["chord"], veh.weight
    ep = elevator_power(L, a, veh.x_cg)
    Ewh = ks.COMPONENTS["battery"]["cells"] * 3.7 * L["battery_ah"] * ks.COMPONENTS["battery"]["usable"]
    sm = (a["x_np"] - veh.x_cg) / c
    rows = []
    for V in speeds:
        q = 0.5 * RHO * V * V
        CL = W / (q * S)
        if CL > pol.cl_max:
            rows.append({"V [m/s]": V, "CL": CL, "feasible": False, "note": "beyond the aircraft's CL max (the front wing stalls)"})
            continue
        alpha = CL / pol.cl_alpha + pol.alpha0
        Cm_cg = a["Cm0"] + a["Cma"] * (alpha - 0.0) + 0.0                      # about the front AC → shift to the CG
        Cm_cg = a["Cm0"] + a["Cma"] * alpha + CL * (veh.x_cg - L["x_ac_front"]) / c
        delta = -Cm_cg / ep["Cm_delta rear pair [1/rad]"]
        dCL = a["CLa_one_wing"] * a["slope_rear"] * ELEVON_TAU * (2 * L["elevon_span_side"] * ELEVON_CF * c / S) * delta
        CD = pol.cd0 + pol.k * CL ** 2 + 0.002 * abs(math.degrees(delta)) / 10.0
        D = q * S * CD
        T = D / 4
        ok = d.T_min <= T <= float(d.t_max(V))
        P = 4 * d.power(T, V) + avionics_W
        rows.append({"V [m/s]": V, "CL": CL, "α [deg]": math.degrees(alpha), "L/D": CL / CD, "drag [N]": D, "elevator trim [deg]": math.degrees(delta), "ΔCL of the trim": dCL,
                     "static margin [c]": sm, "thrust per rotor [N]": T, "within the drive": ok, "electrical [W]": P, "endurance [min]": Ewh / P * 60, "range [km]": Ewh / P * V * 3.6,
                     "feasible": True, "note": ""})
    return pd.DataFrame(rows).set_index("V [m/s]")


def takeoff_roll(veh, *, mu: float = MU_ROLL, alpha_ground_deg: float = 3.0, lift_off_factor: float = 1.15, dt: float = 0.02) -> dict:
    """The horizontal start: full thrust from rest on the skids, the wings at ``alpha_ground_deg``, rolling friction
    ``mu`` on the weight not yet carried, until ``lift_off_factor`` × the stall speed. Distance, time, the speeds."""
    L, d, pol = veh.L, veh.drive, veh.aircraft_polar
    S, W, m = L["S_ref"], veh.weight, veh.mass
    Vs = math.sqrt(2 * W / (RHO * S * pol.cl_max))
    V_lo = lift_off_factor * Vs
    CLg = min(pol.cl_alpha * (math.radians(alpha_ground_deg) - pol.alpha0), pol.cl_max)
    CDg = pol.cd0 + pol.k * CLg ** 2 + 0.01
    V = x = t = 0.0
    while V < V_lo and t < 60.0:
        q = 0.5 * RHO * V * V
        T = 4 * float(d.t_max(V))
        Lift = min(q * S * CLg, W)
        acc = (T - q * S * CDg - mu * (W - Lift)) / m
        if acc <= 0.05:
            return {"lifts off": False, "stall speed [m/s]": Vs, "lift-off speed [m/s]": V_lo, "stuck at [m/s]": V, "ground roll [m]": math.inf, "time [s]": math.inf}
        V += acc * dt
        x += V * dt
        t += dt
    return {"lifts off": True, "stall speed [m/s]": Vs, "lift-off speed [m/s]": V_lo, "ground roll [m]": x, "time [s]": t, "static T/W": 4 * float(d.t_max(0.0)) / W,
            "CL on the ground": CLg, "rolling friction": mu}


# ================================================================================================= structure
def cantilever(L: dict, *, n: float = 4.0, mass_kg: float, share_front: float = 0.66, od_mm: float = 10.0, wall_mm: float = 1.0) -> dict:
    """The front half-wing (the loaded one) as a cantilever from the pod at ``n`` g: its share of the lift, elliptic
    along the half-span, less the wing's and the motor's inertia; the Ø10×8 tube's stress, tip deflection, the first
    bending mode of the motor on the spar, the spar's reserve against the carbon allowable."""
    import kestrel_structure as kst
    Lh = L["span"] / 2 - POD_WIDTH / 2
    lift = n * mass_kg * G * share_front / 2
    motor_m = ks.COMPONENTS["motor"]["mass_kg"] + L["prop_mass"] + ks.COMPONENTS["esc"]["mass_kg"] + ks.NACELLE_KG
    wing_m = ks.WING_AREAL_KG_M2 * L["S_wing"] / 2 + ks.SPAR_KG_M * Lh
    r_mot = MOTOR_Y - POD_WIDTH / 2
    M_root = lift * (4 / (3 * math.pi)) * Lh - n * G * (wing_m * Lh / 2 + motor_m * r_mot)
    E = kst.CARBON["E"]
    od, idd = od_mm / 1000, (od_mm - 2 * wall_mm) / 1000
    I = math.pi / 64 * (od ** 4 - idd ** 4)
    A = math.pi / 4 * (od ** 2 - idd ** 2)
    sigma = abs(M_root) * od / 2 / I
    w_tip = lift * Lh ** 3 / (8 * E * I) * 0.85 - n * G * motor_m * r_mot ** 2 * (3 * Lh - r_mot) / (6 * E * I)
    m_per = A * 1550.0 + wing_m / Lh
    f1 = 1 / (2 * math.pi) * math.sqrt(3 * E * I / (r_mot ** 3 * (motor_m + 0.24 * m_per * r_mot)))
    return {"n": n, "half-wing lift [N]": lift, "root moment [N·m]": M_root, "spar": f"carbon Ø{od_mm:g}×{od_mm - 2 * wall_mm:g}", "spar stress [MPa]": sigma / 1e6,
            "spar reserve": kst.CARBON["sigma_allow"] / sigma if sigma > 0 else math.inf, "tip deflection [mm]": w_tip * 1e3, "tip deflection / half-span [%]": 100 * w_tip / Lh,
            "first bending, with the motor [Hz]": f1, "spar mass per wing [g]": A * 1550.0 * L["span"] * 1e3}


# ================================================================================================= drawings
def front_view(L: dict, *, cg=None, ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    t = 0.12 * L["chord"]
    for wing, z, col in (("rear", L["z_rear"], "#9ec5e8"), ("front", L["z_front"], "#4682b4")):
        ax.fill([-L["span"] / 2, L["span"] / 2, L["span"] / 2, -L["span"] / 2], [z - t / 2, z - t / 2, z + t / 2, z + t / 2], color=col, ec="#223", lw=0.8, alpha=0.9, label=f"{wing} wing")
    ax.add_patch(plt.Rectangle((-L["pod_width"] / 2, L["pod_bottom"]), L["pod_width"], L["pod_height"], color="#e8e8e8", ec="#223"))
    for s in (-1, 1):
        ax.plot([s * 0.03, s * 0.03], [L["pod_bottom"], L["ground_z"]], color="#555", lw=2)
    ax.axhline(L["ground_z"], color="#8b5a2b", lw=2)
    for name, mo in L["motors"].items():
        ax.add_patch(plt.Circle((mo["y"], mo["z"]), L["R_prop"], color="#f0a050" if mo["wing"] == "front" else "#c080f0", alpha=0.25))
        ax.add_patch(plt.Circle((mo["y"], mo["z"]), 0.017, color="#333"))
        if L["pylon"][mo["wing"]]:
            ax.plot([mo["y"], mo["y"]], [mo["z"] - L["pylon"][mo["wing"]], mo["z"]], color="#333", lw=3)
        ax.text(mo["y"] * 1.1, mo["z"] + 0.03, name, ha="center", fontsize=9)
    ax.plot([0, 0], [L["z_rear"], L["z_rear"] + L["fin_height"]], color="#223", lw=3)
    if cg is not None:
        ax.plot(0.0, cg["z_cg"], "x", color="red", ms=12, mew=3)
    ax.set(aspect="equal", xlabel="y [m]", ylabel="z [m]", title=f"{L['variant']}: looking at the nose ({L['arrangement']})")
    ax.legend(fontsize=8, loc="upper right")
    return ax


def side_view(L: dict, *, x_cg=None, x_np=None, ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    c = L["chord"]
    for wing, x_le, z, col in (("front", L["x_le_front"], L["z_front"], "#4682b4"), ("rear", L["x_le_rear"], L["z_rear"], "#9ec5e8")):
        ax.fill([x_le, x_le + c, x_le + c, x_le], [z - 0.06 * c, z - 0.02 * c, z + 0.06 * c, z + 0.1 * c], color=col, ec="#223", label=f"{wing} wing")
        ax.plot([x_le - PROP_AHEAD, x_le - PROP_AHEAD], [z + L["pylon"][wing] - L["R_prop"], z + L["pylon"][wing] + L["R_prop"]], color="#333", lw=2)
    ax.add_patch(plt.Rectangle((0.0, L["pod_bottom"]), L["pod_length"], L["pod_height"], color="#e8e8e8", ec="#223", alpha=0.8))
    ax.plot([0.08, L["pod_length"] - 0.1], [L["ground_z"], L["ground_z"]], color="#555", lw=3)
    ax.plot([L["x_le_rear"] + 0.3 * c, L["x_le_rear"] + c], [L["z_rear"], L["z_rear"] + L["fin_height"]], color="#223", lw=2)
    if x_cg is not None:
        ax.plot(x_cg, 0.0, "x", color="red", ms=12, mew=3, label="CG")
    if x_np is not None:
        ax.plot(x_np, 0.0, "v", color="green", ms=10, label="neutral point")
    ax.set(aspect="equal", xlabel="x [m] (nose at 0)", ylabel="z [m]", title=f"{L['variant']} from the left; stagger {L['stagger_c']:g} c, gap {L['gap']:.2f} m")
    ax.legend(fontsize=8, loc="upper right")
    return ax


# ================================================================================================= the CAD
def cad_params(variant: str = MAIN, **over) -> dict:
    """``KestrelQ4T``'s parameters [mm] for a variant (``generate(**cad_params(v), part=...)``)."""
    L = layout(variant, **over)
    return dict(span=L["span"] * 1e3, chord=L["chord"] * 1e3, gap=L["gap"] * 1e3, stagger=L["stagger"] * 1e3, sign=L["sign"], wing_le_x=L["x_le_front"] * 1e3,
                pod_length=L["pod_length"] * 1e3, pod_width=POD_WIDTH * 1e3, motor_y=MOTOR_Y * 1e3, prop_diameter=L["R_prop"] * 2e3, prop_x=L["prop_x"] * 1e3,
                pylon=PYLON * 1e3, skid_height=SKID_HEIGHT * 1e3, fin_height=FIN_HEIGHT * 1e3, decalage_deg=DECALAGE_DEG, elevons=True, elevon_inner_y=L["elevon_y_in"] * 1e3,
                bracing="none", camber=0.02, camber_pos=0.4, x_feet=L["pod_length"] * 1e3 + 1.0, nose_length=150.0, tail_length=TAIL_LENGTH * 1e3)


class KestrelQ4T(kestrel.Kestrel4Q):
    """The tandem's CAD on the biplane's parts: two wings at their stations and heights (the front with the decalage),
    the pod spanning both, tractor nacelles on each wing (the low wing's on pylons), two skids, a fin on the rear wing.
    Parts: ``aircraft`` (with the propeller discs), ``airframe`` (the CFD's body), ``wing``, ``pod``."""
    from vegeta.dedalus import Parameter as _P
    parameters = [p for p in kestrel.Kestrel4Q.parameters if p.name not in ("bracing", "strut_fairing", "post_diameter", "diagonal_diameter", "fin_height", "ventral_height", "gap")] + [
        _P("gap", 280.0, "mm", min=0, description="vertical distance between the wings' chord planes (0: one plane)"),
        _P("stagger", 420.0, "mm", min=100, description="the rear wing's leading edge behind the front wing's"),
        _P("sign", -1, "", choices=(-1, 0, 1), description="-1: front wing low (N), 1: front wing high (P), 0: one plane"),
        _P("decalage_deg", DECALAGE_DEG, "deg", min=-5, max=8, description="the front wing's incidence over the rear's"),
        _P("pylon", PYLON * 1e3, "mm", min=0, description="the low wing's motors raised by this much"),
        _P("skid_height", SKID_HEIGHT * 1e3, "mm", min=0),
        _P("fin_height", FIN_HEIGHT * 1e3, "mm", min=0, description="the fin on the rear wing's centre"),
        _P("bracing", "none", choices=("none",), description="a tandem has no bracing"),
        _P("strut_fairing", False, ""),
        _P("post_diameter", 8.0, "mm"), _P("diagonal_diameter", 6.0, "mm"), _P("ventral_height", 0.0, "mm", min=0),
    ]

    @staticmethod
    def checks(p) -> list[str]:
        return []

    def _tandem_wing(self, p, which: str):
        import cadquery as cq
        sgn = int(p["sign"])
        z = (sgn if which == "front" else -sgn) * p["gap"] / 2
        q = dict(p, wing_le_x=p["wing_le_x"] + (0.0 if which == "front" else p["stagger"]))
        w = self._wing(q, z)
        if which == "front" and p["decalage_deg"]:
            x_q = q["wing_le_x"] + 0.25 * p["chord"]
            w = w.rotate((x_q, 0, z), (x_q, 1, z), -p["decalage_deg"])       # nose-up: CadQuery's right-hand rotation about +y turns the nose up with a negative angle here
        return w, q, z

    def build(self, p):
        import cadquery as cq
        part = p["part"]
        sgn = int(p["sign"])
        if part == "wing":
            return self._tandem_wing(p, "front")[0]
        if part == "pod":
            return self._pod(p)
        wf, qf, zf = self._tandem_wing(p, "front")
        wr, qr, zr = self._tandem_wing(p, "rear")
        shape = wf.union(wr).union(self._pod(p))
        low = {"front": zf <= zr, "rear": zr <= zf}
        for which, q, z in (("front", qf, zf), ("rear", qr, zr)):
            pyl = p["pylon"] if low[which] else 0.0
            zn = z + pyl
            qn = dict(q, gap=2 * abs(zn), prop_x=q["wing_le_x"] - PROP_AHEAD * 1e3)
            for sy in (-1, 1):
                shape = shape.union(self._nacelle(qn, sy, 1 if zn >= 0 else -1, prop=(part == "aircraft" and p["show_props"])))
                if pyl:
                    shape = shape.union(cq.Workplane("XY").box(0.6 * p["chord"], 8.0, pyl + 4.0).translate((q["wing_le_x"] + 0.3 * p["chord"], sy * p["motor_y"], z + pyl / 2)))
            if p["elevons"]:
                for sy in (-1, 1):
                    shape = shape.union(self._elevon(q, z, sy))
        pod_bottom = min(min(zf, zr) - POD_SKIN_OVER_WING * 1e3, -p["pod_width"] / 2)
        for sy in (-1, 1):
            skid = cq.Workplane("XY").box(0.6 * p["pod_length"], 6.0, p["skid_height"]).translate((0.5 * p["pod_length"], sy * 30.0, pod_bottom - p["skid_height"] / 2))
            shape = shape.union(skid)
        if p["fin_height"]:
            x0 = qr["wing_le_x"] + 0.3 * p["chord"]
            x1 = qr["wing_le_x"] + p["chord"]
            fin = cq.Workplane("XZ").polyline([(x0, zr), (x1, zr), (x1, zr + p["fin_height"]), (x1 - 0.5 * p["chord"], zr + p["fin_height"])]).close().extrude(p["fin_thickness"] / 2, both=True)
            shape = shape.union(fin)
        a = p["angle_of_attack_deg"]
        if a:
            x0 = p["rotation_x"]
            shape = shape.rotate((x0, 0, 0), (x0, 1, 0), a)
        return shape

    def _pod(self, p):
        """The biplane's pod loft, stretched in height to span both wings plus the skin over them (at least square)."""
        height = max(p["gap"] + 2 * POD_SKIN_OVER_WING * 1e3, p["pod_width"])
        return kestrel.Kestrel4Q._pod(self, dict(p, gap=height / 0.98))


def export_stl(workdir, variant: str, alpha_deg: float, x_rot_m: float) -> Path:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    path = workdir / f"kestrel_{variant.split(' ')[0].replace('/', '_')}_a{alpha_deg:+05.1f}.stl"
    if not path.exists():
        g = KestrelQ4T().generate(**cad_params(variant), part="airframe", angle_of_attack_deg=float(alpha_deg), rotation_x=x_rot_m * 1000)
        g.export_stl(str(path), tolerance=0.3)
    return path


# ================================================================================================= against Kestrel-4Q (notebook 43)
def biplane_reference(path=None):
    """Kestrel-4Q's numbers as notebook 43 exported them (``kestrel_xwing.biplane_reference``): None with a loud warning
    when the report is missing."""
    import sys
    from pathlib import Path
    exp = str(Path(__file__).resolve().parent / "exp")      # the Kestrel-X experiments
    if exp not in sys.path:
        sys.path.insert(0, exp)
    import kestrel_xwing as kx
    return kx.biplane_reference(path)
