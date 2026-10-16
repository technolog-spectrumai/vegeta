"""Kestrel-4Q against an X-wing of the same motors, battery and pod (notebook 38, Part II).

The X-wing: four straight cantilever panels radiating from the pod at ±45°, a motor near the tip of each, **no
truss** — the configuration the biplane's bracing is weighed against. Seen as wings, it is two straight wings of equal
span crossing at the pod, one rolled +45° and one −45° about the pod's axis: each panel is the half-span of one of
them. The same four fixed rotors thrust along the pod's axis; standing on its tail the aircraft rests on the four
tips.

What changes against the biplane, and what this module computes:
- **Lift at a dihedral of 45°:** a panel's lift is normal to the panel; its vertical component is cos 45° of it, and the
  panel sees only cos 45° of the aircraft's angle of attack. The vertical lift slope of the four panels is half that
  of the same area laid flat — the X needs twice the area for the same cruise, or flies at twice the panel cl.
  ``VARIANTS``: ``X-A`` (the biplane's wing area, 4 × 0.5 m × 0.14 m) and ``X-B`` (twice it: 4 × 0.70 m × 0.20 m).
- ``layout``: the panels, the motors' stations (y, z at 45°: the pitch and the yaw levers are equal), the projected span,
  the keys the transition model reads (one straight line through the pod per wing, the rotor at ``motor_y`` along it).
- ``mass_table``: the biplane's parts list without the truss and the fin plate (the panels are the fins), the panels'
  spars sized as cantilevers with tip motors, the feet at the tips.
- ``aero``: the lattice (``peregrine_flight.vlm``) on the four panels as quads at ±45°: the vertical lift slope, the
  induced-drag factor on the projected span, the neutral point.
- ``drag_buildup``: no struts; the panels, the pod, the nacelles, the four wing-pod junctions.
- ``cantilever``: the panel's spar at a pull-up with its tip motor: root moment, stress, tip deflection, the first
  bending mode against the rotor's 1P.
- ``stance``: the footprint on the four tips.
- ``make_vehicle``: the X-wing for ``kestrel_transition.simulate`` (the generalised panel model: each panel at its
  dihedral and height, blown by its own rotor).

Units SI (the layout in metres), angles in radians unless named _deg.
"""
from __future__ import annotations

import math

import numpy as np

import kestrel_systems as ks

G = ks.G
RHO = 1.225
PHI_DEG = 45.0
VARIANTS = {
    "X-A (the biplane's wing area)": {"panel_length": 0.50, "chord": 0.14, "motor_r": 0.46},
    "X-B (twice the wing area)": {"panel_length": 0.70, "chord": 0.20, "motor_r": 0.66},
}
POD_WIDTH, POD_LENGTH, X_LE, PROP_X, R_PROP = 0.110, 0.430, 0.200, 0.125, 0.1143
X_FEET_MARGIN = 0.10             # the feet (the tips' rods) this far behind the trailing edge
NACELLE_DIAMETER = 0.034
SPAR_STRESS_MAX = 200e6          # the spar's working stress at n = 4 [Pa]
DEFLECTION_MAX = 0.03            # tip deflection at n = 4, fraction of the panel length
F1_MIN = 12.0                    # the motor's first bending mode on its arm [Hz]: an octave above the pitch loop's ~2 Hz


def layout(variant: str = "X-A (the biplane's wing area)", *, phi_deg: float = PHI_DEG) -> dict:
    v = VARIANTS[variant]
    Lp, c, r_m = v["panel_length"], v["chord"], v["motor_r"]
    phi = math.radians(phi_deg)
    r0 = POD_WIDTH / 2
    r1 = r0 + Lp
    cph, sph = math.cos(phi), math.sin(phi)
    motors = {}
    for name, (sy, sz) in (("M1", (-1, 1)), ("M2", (1, 1)), ("M3", (-1, -1)), ("M4", (1, -1))):
        motors[name] = {"x": PROP_X, "y": sy * r_m * cph, "z": sz * r_m * sph}
    x_te = X_LE + c
    S_panel = Lp * c
    return {
        "variant": variant, "phi": phi, "phi_deg": phi_deg, "panel_length": Lp, "chord": c, "r0": r0, "r1": r1, "motor_r": r_m,
        "span": 2 * r1, "span_projected": 2 * r1 * cph, "height": 2 * r1 * sph, "S_panel": S_panel, "S_ref": 4 * S_panel,
        "S_wing": 2 * S_panel, "S_vertical_projected": 4 * S_panel * cph, "AR_wing": 2 * r1 / c,
        "x_le": X_LE, "x_te": x_te, "x_ac": X_LE + 0.25 * c, "x_spar": X_LE + 0.30 * c, "chord_m": c,
        "motors": motors, "motor_y": r_m, "h": r_m * sph, "gap": 2 * r_m * sph,
        "R_prop": R_PROP, "A_disc": math.pi * R_PROP ** 2, "prop_x": PROP_X, "prop_ahead_of_le": X_LE - PROP_X,
        "prop_gap": 2 * r_m * sph - 2 * R_PROP, "prop_to_pod": r_m - R_PROP - r0, "prop_beyond_tip": r_m + R_PROP - r1,
        "pod_length": POD_LENGTH, "pod_width": POD_WIDTH, "pod_height": POD_WIDTH,
        "x_feet": x_te + X_FEET_MARGIN, "foot_rod_length": X_FEET_MARGIN + 0.4 * c,
        "feet": [(sy * r1 * cph, sz * r1 * sph) for sy, sz in ((-1, 1), (1, 1), (-1, -1), (1, -1))],
        "elevons": True, "elevon_cf": 0.28, "elevon_y_in": max(r_m - R_PROP, r0 + 0.02), "elevon_span_side": r1 - max(r_m - R_PROP, r0 + 0.02),
        "x_hinge": X_LE + 0.72 * c, "posts": 0, "diagonals": 0, "bracing": "none",
    }


def checks(L: dict) -> list[str]:
    out = []
    if L["prop_gap"] < 0.025:
        out.append("the propeller discs of the upper and lower rotors overlap")
    if L["prop_to_pod"] < 0.025:
        out.append("the propeller discs reach the pod")
    if L["prop_beyond_tip"] > 0.02:
        out.append("the propellers overhang the tips")
    return out


# ================================================================================================= masses
def _spar_kg_m(L: dict, n: float = 4.0) -> float:
    """A round carbon tube sized for the cantilever at ``n`` g (``cantilever``: stress, deflection and the first mode):
    its linear mass."""
    r = cantilever(L, n=n, mass_kg=1.7)
    od = r["spar OD [mm]"] / 1000
    idd = od - 2 * 0.001
    A = math.pi / 4 * (od ** 2 - idd ** 2)
    return A * 1550.0


def mass_table(L: dict, *, battery_x: float | None = None, x_cg_target: float | None = None, payload: bool = True) -> list[dict]:
    """The X-wing's parts: the biplane's bought parts and pod, four panels (balsa-composite at the biplane's areal density
    plus a spar each), four nacelles at the tips, four tip foot rods, four elevon servos, no truss, no fin plate; the
    battery placed for the CG as in ``kestrel_systems.mass_table``."""
    c = ks.COMPONENTS
    items = []

    def add(name, m, x, y, z, group, rod=None):
        items.append({"item": name, "mass_kg": m, "x": x, "y": y, "z": z, "group": group, "rod": rod})

    x_motor = L["prop_x"] + 0.025
    spar_kg_m = _spar_kg_m(L)
    for k, s in L["motors"].items():
        add(f"motor {k}", c["motor"]["mass_kg"], x_motor, s["y"], s["z"], "propulsion")
        add(f"propeller {k}", c["propeller"]["mass_kg"], L["prop_x"], s["y"], s["z"], "propulsion")
        add(f"ESC {k}", c["esc"]["mass_kg"], L["x_le"] + 0.02, s["y"], s["z"], "propulsion")
        add(f"nacelle {k}", ks.NACELLE_KG, 0.5 * (L["prop_x"] + L["x_le"] + 0.6 * L["chord"]), s["y"], s["z"], "structure")
        ang = math.atan2(s["z"], s["y"])
        r_mid = 0.5 * (L["r0"] + L["r1"])
        add(f"panel {k}", ks.WING_AREAL_KG_M2 * L["S_panel"], L["x_le"] + 0.42 * L["chord"], r_mid * math.cos(ang), r_mid * math.sin(ang), "structure")
        add(f"spar {k}", spar_kg_m * L["panel_length"], L["x_spar"], r_mid * math.cos(ang), r_mid * math.sin(ang), "structure")
        add(f"foot rod {k}", ks.FOOT_ROD_KG_M * L["foot_rod_length"], L["x_feet"] - 0.5 * L["foot_rod_length"], L["r1"] * math.cos(ang), L["r1"] * math.sin(ang), "structure")
        add(f"elevon servo {k}", ks.SERVO_KG, L["x_hinge"] - 0.02, 0.5 * (L["elevon_y_in"] + L["r1"]) * math.cos(ang),
            0.5 * (L["elevon_y_in"] + L["r1"]) * math.sin(ang), "systems")
    add("pod shell + panel roots", ks.POD_SHELL_KG + 4 * ks.JOINT_KG, 0.45 * L["pod_length"], 0.0, 0.0, "structure")
    add("wiring", ks.WIRING_KG, L["x_le"], 0.0, 0.0, "systems")
    add("flight stack", c["flight_stack"]["mass_kg"], 0.60 * L["pod_length"], 0.0, 0.03, "systems")
    add("camera", c["camera"]["mass_kg"], 0.03, 0.0, -0.01, "payload")
    if payload:
        add("mission computer", c["mission_computer"]["mass_kg"], 0.30 * L["pod_length"], 0.0, 0.03, "payload")
    add("growth margin", ks.MARGIN_KG, L["x_ac"], 0.0, 0.0, "margin")
    mb = c["battery"]["mass_kg"]
    half = c["battery"]["length_m"] / 2
    lo, hi = 0.090 + half, L["pod_length"] - 0.060 - half
    clipped = False
    if battery_x is None:
        target = L["x_le"] + ks.X_CG_TARGET_FRAC * L["chord"] if x_cg_target is None else x_cg_target
        m = sum(i["mass_kg"] for i in items)
        mx = sum(i["mass_kg"] * i["x"] for i in items)
        battery_x = (target * (m + mb) - mx) / mb
        if not lo <= battery_x <= hi:
            clipped = True
            battery_x = float(np.clip(battery_x, lo, hi))
    items.append({"item": "battery", "mass_kg": mb, "x": battery_x, "y": 0.0, "z": 0.0, "group": "propulsion", "rod": None, "clipped": clipped})
    return items


# ================================================================================================= aerodynamics
def quads(L: dict) -> list:
    """The four panels as planar quads (x aft, y right, z up) for the lattice."""
    out = []
    phi = L["phi"]
    for sy, sz in ((-1, 1), (1, 1), (-1, -1), (1, -1)):
        e = np.array([0.0, sy * math.cos(phi), sz * math.sin(phi)])          # the panel's span direction
        a, b = L["r0"] * e, L["r1"] * e
        out.append(np.array([[L["x_le"], *a[1:]], [L["x_le"], *b[1:]], [L["x_te"], *b[1:]], [L["x_te"], *a[1:]]]))
    return out


def aero(L: dict, *, n_span: int = 16, n_chord: int = 4) -> dict:
    """The lattice on the four panels: the vertical lift slope CLα on S_ref, the induced-drag factor k = CDi/CL² and
    the span efficiency on the projected span, the neutral point; against the flat biplane of the same area."""
    import peregrine_flight as pf
    qs = quads(L)
    S, c, x_ref = L["S_ref"], L["chord"], L["x_ac"]
    kw = dict(S_ref=S, c_ref=c, x_ref=x_ref, n_span=n_span, n_chord=n_chord)
    r0 = pf.vlm(qs, alpha_deg=0.0, **kw)
    r1 = pf.vlm(qs, alpha_deg=4.0, **kw)
    da = math.radians(4.0)
    CLa = (r1["CL"] - r0["CL"]) / da
    Cma = (r1["Cm"] - r0["Cm"]) / da
    k = r1["CDi"] / r1["CL"] ** 2
    b = L["span_projected"]
    AR = b ** 2 / S
    return {"CLa": CLa, "k": k, "e": 1 / (math.pi * AR * k), "AR_projected": AR, "S": S, "x_np": x_ref - Cma / CLa * c,
            "CLa_flat_estimate": None, "note": "CLa on S_ref: the vertical slope of panels at 45° is ~cos²45 = 0.5 of the flat wing's"}


def drag_buildup(L: dict, *, V: float = 16.0, t: float = 0.12) -> dict:
    import kestrel_flight as kf
    rows = {}
    ff_wing = 1 + 0.6 / 0.3 * t + 100 * t ** 4
    wet = 2 * L["S_ref"] * (1 + 0.2 * t)
    rows["panels (4)"] = kf._cf(V * L["chord"] / kf.NU) * ff_wing * wet
    lf, df = L["pod_length"], L["pod_width"]
    fr = lf / df
    rows["pod"] = kf._cf(V * lf / kf.NU) * (1 + 60 / fr ** 3 + fr / 400) * math.pi * df * lf * 0.75
    dn, ln = NACELLE_DIAMETER, (L["x_le"] + 0.6 * L["chord"] - L["prop_x"])
    rows["nacelles (4)"] = 4 * (kf._cf(V * ln / kf.NU) * 1.3 * math.pi * dn * ln + 0.1 * math.pi * dn ** 2 / 4)
    rows["foot rods (4), axial"] = 4 * 0.02 * math.pi * 0.006 ** 2 / 4 * 10
    rows["camera dome"] = 0.4 * math.pi * 0.018 ** 2
    rows["wing-pod junctions (4, acute)"] = 4 * 0.0012 * L["chord"] ** 2 / 0.14 ** 2 * 1.0     # an allowance: acute 45° junctions are worse than a T
    total = sum(rows.values()) * kf.INTERFERENCE
    return {"parts_m2": rows, "CdA_m2": total, "Cd0": total / L["S_ref"], "Cd0_on_biplane_Sref": total / 0.28}


# ================================================================================================= structure, stance
def cantilever(L: dict, *, n: float = 4.0, mass_kg: float, spar_od_mm: float | None = None, wall_mm: float = 1.0) -> dict:
    """One panel as a cantilever from the pod at load factor ``n``: its share of the lift (a quarter, elliptic along
    the panel, normal to the panel: the panel lift is the vertical share / cos φ), less the panel's and the tip
    motor's inertia; the root moment, a tube spar's stress (OD given, or the OD that keeps 200 MPa, ≤ 0.6 t c) and
    tip deflection, the first bending frequency against the rotor's 1P."""
    import kestrel_structure as kst
    Lp, c, phi = L["panel_length"], L["chord"], L["phi"]
    lift_vert = n * mass_kg * G / 4
    lift_panel = lift_vert / math.cos(phi)
    motor_m = ks.COMPONENTS["motor"]["mass_kg"] + ks.COMPONENTS["propeller"]["mass_kg"] + ks.COMPONENTS["esc"]["mass_kg"] + ks.NACELLE_KG
    panel_m = ks.WING_AREAL_KG_M2 * L["S_panel"] + 0.04 * Lp
    r_mot = L["motor_r"] - L["r0"]
    # elliptic along the panel: the centroid at 4/(3π) of the length from the root
    M_root = lift_panel * (4 / (3 * math.pi)) * Lp - n * G * (panel_m * Lp / 2 + motor_m * r_mot)
    E = kst.CARBON["E"]

    def evaluate(od):
        idd = max(od - 2 * wall_mm / 1000, 0.0)
        I = math.pi / 64 * (od ** 4 - idd ** 4)
        A = math.pi / 4 * (od ** 2 - idd ** 2)
        sigma = abs(M_root) * od / 2 / I
        w_tip = lift_panel * Lp ** 3 / (8 * E * I) * 0.85 - n * G * motor_m * r_mot ** 2 * (3 * Lp - r_mot) / (6 * E * I)
        m_per = A * 1550.0 + panel_m / Lp
        f1_tip = 1 / (2 * math.pi) * math.sqrt(3 * E * I / (r_mot ** 3 * (motor_m + 0.24 * m_per * r_mot)))   # the motor on its arm
        return {"od": od, "I": I, "A": A, "sigma": sigma, "w_tip": w_tip, "f1": f1_tip}

    od_max = 0.6 * 0.12 * c                                     # the tube inside the section's thickness
    if spar_od_mm is None:                                      # sized on stress, deflection and the first mode together
        od = 0.006
        while True:
            r = evaluate(od)
            if (r["sigma"] <= SPAR_STRESS_MAX and abs(r["w_tip"]) <= DEFLECTION_MAX * Lp and r["f1"] >= F1_MIN) or od >= od_max:
                break
            od += 0.0005
    else:
        od = spar_od_mm / 1000
    r = evaluate(od)
    return {"n": n, "panel lift [N]": lift_panel, "root moment [N·m]": M_root, "spar OD [mm]": od * 1000, "spar stress [MPa]": r["sigma"] / 1e6,
            "spar reserve": kst.CARBON["sigma_allow"] / r["sigma"] if r["sigma"] > 0 else math.inf, "tip deflection [mm]": r["w_tip"] * 1e3,
            "tip deflection / length": r["w_tip"] / Lp, "first bending, with the tip motor [Hz]": r["f1"],
            "spar mass per panel [g]": r["A"] * 1550.0 * Lp * 1e3,
            "sized by": "section depth (limit reached)" if od >= od_max - 1e-9 else
                        ("stress" if r["sigma"] > 0.9 * SPAR_STRESS_MAX else "deflection" if abs(r["w_tip"]) > 0.9 * DEFLECTION_MAX * Lp else "first mode")}


def stance(L: dict, *, mass_kg: float, x_cg: float, rho: float = RHO, cd: float = 1.1) -> dict:
    """Standing on the four tips: the footprint's least tip-over angle and the steady wind that tips it (the panels'
    projected area, the two headings)."""
    H = L["x_feet"] - x_cg
    phi = L["phi"]
    d = L["r1"] * math.sin(phi)                              # the footprint is a square of half-diagonal r1: edges at r1 sin45
    W = mass_kg * G
    h_wing = L["x_feet"] - (L["x_le"] + 0.5 * L["chord"])
    A_proj = 4 * L["S_panel"] * math.cos(phi)
    V_tip = math.sqrt(2 * W * d / (rho * cd * A_proj * h_wing))
    return {"CG height above the ground [m]": H, "least tip-over angle [deg]": math.degrees(math.atan2(d, H)),
            "footprint half-width [m]": d, "wind that tips it [m/s]": V_tip, "feet": L["feet"]}


# ================================================================================================= the vehicle
def make_vehicle(variant: str = "X-A (the biplane's wing area)", *, name: str | None = None, elevons: bool = True, extra_kg: float = 0.0,
                 pol=None, drive=None, x_cg: float | None = None):
    """The X-wing for ``kestrel_transition``: the panels at ±45° (two per rotor pair: each 'wing' is one line through
    the pod; its upper half is blown by an upper rotor, its lower half by a lower one)."""
    import kestrel_flight as kf
    import kestrel_transition as kt
    L = layout(variant)
    table = mass_table(L, x_cg_target=x_cg)
    if extra_kg:
        table.append({"item": "extra payload", "mass_kg": extra_kg, "x": L["x_ac"], "y": 0.0, "z": 0.0, "group": "payload", "rod": None})
    ci = ks.cg_inertia(table)
    if pol is None:
        a = aero(L)
        d = drag_buildup(L)
        pol = kf.Polar(cl_alpha=a["CLa"] / math.cos(L["phi"]) ** 2, cd0=d["Cd0"], k=a["k"] / kf.OSWALD_VISCOUS, cl_max=0.9 * kf.CL_MAX_2D,
                       cd90=kf.viterna_cd90(L["AR_wing"]), alpha0=0.0, source="X-wing lattice (panel-normal slope)")
    phi = L["phi"]
    panels = []
    for sy, sz in ((-1, 1), (1, 1), (-1, -1), (1, -1)):
        panels.append({"phi": phi, "sy": sy, "sz": sz, "z": sz * 0.5 * (L["r0"] + L["r1"]) * math.sin(phi), "y0": L["r0"], "y1": L["r1"],
                       "y_motor": L["motor_r"], "rotor": "up" if sz > 0 else "lo", "S": L["S_panel"]})
    return kt.Vehicle(name or variant, {"elevons": elevons}, L, ci["mass_kg"], ci["x_cg"], ci["z_cg"], ci["Iyy"], ci["Ixx"], pol,
                      drive or ks.drive_table(), elevons=elevons, panels=panels, z_rotor=L["motor_r"] * math.sin(phi),
                      elevon_y_in=L["elevon_y_in"], lift_slope_vertical=pol.cl_alpha * math.cos(phi) ** 2)


def cruise(veh, *, speeds=np.arange(10.0, 27.0, 1.0), avionics_W: float = 13.0):
    """Level flight of an X-wing vehicle (``make_vehicle``): the vertical CL on S_ref, the panels' own angle of attack
    (α / cos² φ of the vertical slope: the panel sees cos φ of α and gives cos φ of its lift), the drag (profile +
    induced on the lattice's k), the thrust split that trims the pitch with the rotors' vertical lever, the electrical
    power and endurance (as ``kestrel_flight.cruise`` for the biplane)."""
    import pandas as pd
    L, pol, drive = veh.L, veh.pol, veh.drive
    S, c = L["S_ref"], L["chord"]
    W = veh.weight
    cph = math.cos(L["phi"])
    Ewh = ks.COMPONENTS["battery"]["cells"] * 3.7 * ks.COMPONENTS["battery"]["capacity_ah"] * ks.COMPONENTS["battery"]["usable"]
    sm = (aero(L)["x_np"] - veh.x_cg) / c
    rows = []
    for V in speeds:
        q = 0.5 * RHO * V * V
        CL = W / (q * S)                                       # vertical
        alpha = CL / veh.lift_slope_vertical
        cl_panel = CL / cph                                    # each panel's own lift coefficient (normal to it)
        if cl_panel > pol.cl_max or alpha > pol.alpha_stall:
            rows.append({"V [m/s]": V, "CL vertical": CL, "panel cl": cl_panel, "feasible": False, "note": "the panels stall"})
            continue
        CD = pol.cd0 + pol.k * CL ** 2
        D = q * S * CD
        M_aero = q * S * c * (-sm * CL)
        zr = veh.z_rotor
        A = np.array([[1.0, 1.0], [-(zr - veh.z_cg), (zr + veh.z_cg)]])
        T_up, T_lo = np.linalg.solve(A, [D, -M_aero])
        ok = min(T_up, T_lo) / 2 >= drive.T_min and max(T_up, T_lo) / 2 <= float(drive.t_max(V))
        P = 2 * drive.power(T_up / 2, V) + 2 * drive.power(T_lo / 2, V) + avionics_W
        rows.append({"V [m/s]": V, "CL vertical": CL, "panel cl": cl_panel, "α [deg]": math.degrees(alpha), "L/D": CL / CD, "drag [N]": D,
                     "upper rotor [N]": T_up / 2, "lower rotor [N]": T_lo / 2, "trim by thrust": ok, "electrical [W]": P,
                     "endurance [min]": Ewh / P * 60, "range [km]": Ewh / P * V * 3.6, "feasible": True, "note": ""})
    return pd.DataFrame(rows).set_index("V [m/s]")
