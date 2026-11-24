"""GOSHAWK — a small aircraft that flies like its namesake: through a forest, after a quadcopter (2026-11-24).

PROTOHAWK (ASTOR-TWIN / HAWK 2: 8.3 kg, 1.95 m, a net launcher) hunts in the open. A northern goshawk hunts INSIDE
the wood: about 1.1 kg on 1.1 m of broad, rounded wing and a long tail, a wing loading near 5 kg/m2, short bursts,
turns of a few metres, the wings drawn in to pass a gap, a braking flare and the feet. GOSHAWK is sized to those
numbers, not scaled from PROTOHAWK:

- **Size**: span 1.50 m (the limit set), 0.45 m2 (aspect ratio 5.0: broad), ~1.9 kg, wing loading ~4.3 kg/m2.
- **Wing**: the outer pieces swing back about a pivot 250 mm out IN FLIGHT (the wrist): 1.50 m spread, about 1.0 m tucked.
  Flaperons on the outer pieces, plain flaps on the inner ones behind the propellers, the cambered section's flaps
  reflexed in fast flight (PROTOHAWK's schedule).
- **Tail**: long, 38 % of the wing, two all-moving halves with geared flaps, low in the propellers' wash;
  one fin.
- **Power**: two 600 W motors on the inner wing with 8-inch propellers: thrust at rest ~2 x the weight (it can hang
  on its propellers: the braking flare), the wing and the tail blown, yaw by differential thrust.
- **It takes with its feet**: two sprung legs under the centre of gravity (ASTOR-T's, smaller), their feet tearing
  away on a line that fouls the rotors. It carries no net launcher (GAVILAN's weighs 1 kg with one net).

The geometry is ``astor_twin.AstorTwin`` with these numbers (the same drawing code); the flight model is
``astor_v_flight`` as "goshawk": PROTOHAWK's predecessor's CFD carried over by area, aspect ratio and body size until
its own (a third of that Reynolds number: the weakest link). Every mass is a HAND estimate. ``forest()`` is what it is
for: can it turn between the trees?
"""
from __future__ import annotations

import dataclasses
import math

import astor_twin as W

SPAN_MAX_M = 1.50                   # the client's limit (2026-11-24); GOSHAWK uses all of it: the wing loading is what turns it
WING = dict(root_chord=360.0, tip_chord=240.0, half_span=750.0, le_x=120.0, joint_y=250.0, pivot_chord=0.85)
TAIL = dict(span=330.0, root=340.0, tip=180.0, root_y=24.0, z=-12.0, fin_height=180.0, fin_root=210.0, fin_taper=0.45, fin_sweep_deg=40.0)
SECTION = {"panel_thickness": 0.12, "panel_camber": 0.06}       # NACA 6412: see SECTION_CL below
# The section's maximum lift at this aircraft's Reynolds number (2e5), flaps up: vegeta.zephyr (NeuralFoil, n_crit 6,
# 2026-11-24): NACA 4412 1.34, 6412 1.52, 8412 1.75 (confidence 0.97-0.98). With a 30 % flap down 35 deg NeuralFoil gives
# 2.0-2.25 but with NO confidence (0.00): not used. The wing's maximum is the section's x a factor: 0.9 by the usual
# rule; 0.69 is what HAWK1's coarse CFD gave on the 4412 (0.92 against 1.34). Both are carried in ``forest``.
SECTION_CL = {"naca": "6412", "cl_max_flaps_up": 1.52, "Re": 2.0e5, "source": "vegeta.zephyr / NeuralFoil, n_crit 6", "wing_factor_rule": 0.9, "wing_factor_cfd": 0.69,
              "flap_gain_cfd": 0.20, "flap_gain_note": "HAWK1's CFD: + 0.20 of wing lift coefficient for every flap down 20 deg"}
# BALANCE. On the first sizing the neutral point was at ~293 mm (lattice + HAWK1's CFD correction). A tail that LIFTS in trim needs the CG
# behind the wing's own lift, which here means at or behind the neutral point: at 305 mm the tail carries 7 % but the
# aircraft, hands off, lost 60 m in 20 s (20 m with the angle-of-attack augmentation). The bird flies there with a brain.
# (Those figures are the first, 1.15 m sizing's.) At 1.50 m the neutral point is at ~315 mm and GOSHAWK is balanced at
# 300 mm: stable by ~5 %, the tail carrying ~2 %. Going aft is a flight-control job, not a drawing one.
CG_MM = 300.0                       # from the nose tip (astor_v_flight.CG_AFT_MM["goshawk"] is this less astor_v.CG_MM)
TUCK_DEG = 90.0                     # ``wing_sweep_deg`` with the outer pieces drawn in
MOTOR_W, PROP_IN, PROP_ETA, FIGURE_OF_MERIT = 600.0, 8.0, 0.72, 0.55          # per motor; HAND: a 7-inch propeller's efficiency in flight and at rest
G_MAX = 6.0                         # HAND: a small foam-and-carbon airframe; the turn in the wood needs 3-4
GEOMETRY = {**W.plan(WING, TAIL, motor_y=140.0, prop_diameter=PROP_IN * 25.4, fuselage_length=980.0, tail_pivot_x=830.0), **SECTION,
            "fuselage_diameter": 60.0, "nose_length": 110.0, "tail_cone": 300.0, "tail_end_diameter": 28.0, "nacelle_d": 30.0, "nacelle_len": 70.0,
            "camera_diameter": 26.0, "lowcam_drop": 36.0, "parachute_x": 200.0, "parachute_length": 60.0,
            "talon_hinge_x": CG_MM, "leg_y": 20.0, "leg_thigh": 60.0, "leg_shin": 85.0, "leg_shank": 65.0, "toe_length": 35.0, "claw_radius": 10.0}
PACK = {"name": "4S 2200 mAh LiPo", "g": 245.0, "Wh_nominal": 32.6, "max_A": 99.0, "source": "HAND: 110 g per Ah at 4S, 45C (+-10 %)"}
WING_KG_M2, TAIL_KG_M2, FUSE_KG_M = 1.5, 1.05, 0.20       # HAND: foam core under a carbon skin at this size (ASTOR's are 1.9, 1.05, 0.30)


_PARAMETERS = [dataclasses.replace(q, max=0.08) if q.name == "panel_camber" else q for q in W.AstorTwin.parameters]


class Goshawk(W.AstorTwin):
    """GOSHAWK: ASTOR-TWIN's drawing code at GOSHAWK's numbers, less what it does not carry (the launch rail's
    fitting, the dropped nets' camera, the parachute: it lands on its belly from a flare)."""

    parameters = _PARAMETERS                                           # ASTOR-TWIN's, the camber allowed to 8 % (NACA 6412)

    def parts(self, p) -> dict:
        out = super().parts(p)
        for k in ("rail", "belly_camera", "parachute"):
            out.pop(k, None)
        return out


def variant(**over) -> dict:
    """GOSHAWK's parameters: legs fitted and folded, no launcher, no nets."""
    return W.variant(**{**GEOMETRY, "net_launchers": 0, "net_count": 0, "talon_deploy": 0.0, **over})


def resolve(q=None) -> dict:
    return Goshawk().resolve(**dict(q or variant()))


def areas(p=None) -> dict:
    p = p or resolve()
    inner = 0.5 * p["glove_root_chord"] * (1 + p["glove_taper"]) * p["glove_span"] / 1e6
    outer = (p["panel_root_chord"] + p["panel_tip_chord"]) * p["panel_span"] / 1e6
    tail = (p["tail_panel_root"] + p["tail_panel_tip"]) * p["tail_panel_span"] / 1e6
    fin = 0.5 * p["fin_root_chord"] * (1 + p["fin_taper"]) * p["fin_height"] / 1e6
    span = 2 * (p["pivot_y"] + p["panel_span"]) / 1000.0
    return {"wing_m2": inner + outer, "inner_m2": inner, "outer_m2": outer, "tail_m2": tail, "fin_m2": fin, "span_m": span, "aspect_ratio": span ** 2 / (inner + outer),
            "tucked_span_m": 2 * (p["pivot_y"] + p["panel_pivot_chord"] * p["panel_root_chord"]) / 1000.0, "length_m": (p["tail_pivot_x"] + 0.75 * p["tail_panel_root"] + p["tail_panel_span"] * math.tan(math.radians(35.0))) / 1000.0}


def budget(p=None) -> dict:
    """The mass, item by item: every line a HAND estimate (no part is chosen)."""
    p = p or resolve()
    a = areas(p)
    items = [("wing", 1000 * WING_KG_M2 * a["wing_m2"], f"geometry {a['wing_m2']:.3f} m2 x {WING_KG_M2} kg/m2 (HAND +-25 %)"),
             ("tail halves and fin", 1000 * TAIL_KG_M2 * (a["tail_m2"] + a["fin_m2"]), f"geometry {a['tail_m2'] + a['fin_m2']:.3f} m2 x {TAIL_KG_M2} kg/m2 (HAND +-25 %)"),
             ("fuselage", 1000 * FUSE_KG_M * p["fuselage_length"] / 1000.0, f"geometry x {FUSE_KG_M} kg/m (HAND +-25 %)"),
             ("2 motors, 2806 class, 600 W", 90.0, "HAND (+-12): 2 x 45 g"), ("2 controllers, 45 A", 28.0, "HAND (+-8)"), ("2 propellers, 8 inch, guards", 30.0, "HAND (+-10)"),
             ("flight controller, GNSS, radio", 45.0, "HAND (+-12)"),
             ("vision computer and two cameras (the wood is flown by sight)", 95.0, "HAND (+-30): a 60 g module, 2 x 12 g cameras, leads"),
             ("7 servos: 2 flaperons, 2 inner flaps, 2 tail halves, rudder", 70.0, "HAND (+-14): 7 x 10 g; the tail flaps are geared, not driven"),
             ("2 wing pivots, their locks and tuck servos", 60.0, "HAND (+-20)"), ("wiring and connectors", 35.0, "HAND (+-10)"),
             ("2 legs, tear-away feet, line and drag strip", 110.0, "HAND (+-40): ASTOR-T's 220 g at about two thirds the size")]
    dry = sum(g for _, g, _ in items)
    items.append(("growth margin 10 %", 0.10 * dry, "HAND"))
    total = 1.10 * dry + PACK["g"]
    return {"total_g": round(total), "empty_g": round(total - PACK["g"]), "without_nets_g": round(total), "pack": dict(PACK),
            "items": [{"item": n, "g": round(g), "source": s} for n, g, s in items]}


def thrust_table(V_list) -> list:
    """Both propellers' thrust [N] at full power against the airspeed: momentum theory (HAND until a propeller is
    chosen): at rest T = (2 rho A)^(1/3) (FM P)^(2/3); in flight the power PROP_ETA P is thrust x (V + the induced
    speed), never above the thrust at rest."""
    A = math.pi * (PROP_IN * 0.0254 / 2) ** 2
    T0 = (2 * 1.225 * A) ** (1 / 3) * (FIGURE_OF_MERIT * MOTOR_W) ** (2 / 3)
    out = []
    for V in V_list:
        lo, hi = 0.0, T0
        for _ in range(50):
            T = 0.5 * (lo + hi)
            w = 0.5 * (-V + math.sqrt(V * V + 2 * T / (1.225 * A)))
            lo, hi = (T, hi) if T * (V + w) <= PROP_ETA * MOTOR_W else (lo, T)
        out.append(2 * min(lo, T0))
    return out


def forest(speeds=(8.0, 10.0, 12.0, 14.0, 17.0, 20.0), stems_per_ha=(400, 700, 1000)) -> dict:
    """Can it turn between the trees? Per speed, the turn's load factor, radius and quarter-turn time, three ways:

    - **wing, as the flight model has it**: every flap down, the propellers' wash at full throttle, the maximum lift
      carried over from HAWK1's coarse CFD (the pessimistic end);
    - **wing, if the section's lift holds**: NACA 6412's maximum (NeuralFoil) x 0.9, + the CFD's flap gain;
    - **on the propellers**: nose 40 deg up, past the stall: the thrust's share (T sin 40) plus the stalled wing as a flat
      plate. No measured aerodynamics there (the CFD cases to 60 deg are prepared), and no controller flies it yet.

    Against the mean distance between stems of a stand of that density (a square grid: 100 / sqrt(stems per hectare))."""
    import numpy as np
    import arigo1_model as am
    import astor_v_flight as VF
    C = VF.get("goshawk")
    F = VF.with_flap(C, 1.0)
    pw, th = C["powered"], C["thrust"]
    Wt, S = C["mass"] * am.G, C["S"]
    a_lim = math.degrees(F["alpha_stall"]) - F.get("alpha_margin_deg", VF.ALPHA_MARGIN_DEG)
    cl_model = float(np.interp(a_lim, F["cfd"]["alpha_deg"], F["cfd"]["CL"]))
    cl_section = SECTION_CL["cl_max_flaps_up"] * SECTION_CL["wing_factor_rule"] + SECTION_CL["flap_gain_cfd"]
    hi = math.radians(40.0)

    def turn(n, V):
        n = min(n, G_MAX)
        if n <= 1.02:
            return {"n": round(n, 2), "radius_m": None, "quarter_turn_s": None}
        r = V * V / (am.G * math.sqrt(n * n - 1))
        return {"n": round(n, 2), "radius_m": round(r, 1), "quarter_turn_s": round(0.5 * math.pi * r / V, 2)}
    rows = []
    for V in speeds:
        q = 0.5 * am.RHO * V * V
        T = float(np.interp(V, th["table_V"], th["table_T"]))
        vj = math.sqrt(V * V + 2 * (T / 2) / (am.RHO * pw["disc_m2"]))
        width = pw["prop_m"] * math.sqrt((V + vj) / (2 * vj))

        def blown(cl):
            b = VF.SLIP_EFF * 0.5 * am.RHO * (vj * vj - V * V) * 2 * width * pw["chord_m"] * cl
            return min(b, T / max(vj - V, 0.5) * vj * math.sin(min(math.radians(a_lim) + math.radians(12.0), hi)))     # astor_v_flight.powered's limit
        on_props = (q * S * 0.9 * 2 * math.sin(hi) * math.cos(hi) + T * math.sin(hi)) / Wt
        holds = T * math.cos(hi) >= q * S * (0.9 * 2 * math.sin(hi) ** 2 + C["CD0"])
        rows.append({"V_m_s": V, "wing_as_modelled": turn((q * S * cl_model + blown(cl_model)) / Wt, V), "wing_if_the_section_holds": turn((q * S * cl_section + blown(cl_section)) / Wt, V),
                     "on_the_propellers_40_deg": {**turn(on_props, V), "thrust_holds_the_speed": bool(holds)}})
    a = areas()
    return {"stall_m_s": {"flaps_up": round(VF.stall(C), 1), "flaps_down": round(VF.stall(F), 1),
                          "flaps_down_if_the_section_holds": round(math.sqrt(2 * Wt / (am.RHO * S * cl_section)), 1)},
            "lift_coefficient": {"as_modelled": round(cl_model, 2), "if_the_section_holds": round(cl_section, 2), "section": SECTION_CL},
            "static_thrust_over_weight": round(th["table_T"][0] / Wt, 2), "span_m": round(a["span_m"], 2), "tucked_span_m": round(a["tucked_span_m"], 2), "turns": rows,
            "mean_distance_between_stems_m": {f"{n} stems/ha": round(100.0 / math.sqrt(n), 1) for n in stems_per_ha},
            "stem_densities": "HAND: 400-1000 stems per hectare for a grown managed conifer stand; a thicket is denser",
            "reading": "a radius under the stems' distance turns round one tree; the tucked span, or the aircraft rolled on its side (0.3 m), is what passes between two",
            "aero_source": C["aero_source"]}


def checks() -> dict:
    p = resolve()
    a, b = areas(p), budget(p)
    m = b["total_g"] / 1000.0
    return {"status": "GOSHAWK (2026-11-24): a drawing, a mass budget of hand estimates and a carried-over flight model; no CFD, FEA or flight of its own",
            "size": {k: round(v, 3) for k, v in a.items()}, "mass_kg": round(m, 2), "wing_loading_kg_m2": round(m / a["wing_m2"], 1),
            "tail_share_of_the_wing": round(a["tail_m2"] / a["wing_m2"], 2), "the_bird": {"mass_kg": "0.7-1.5", "span_m": "1.0-1.2", "wing_loading_kg_m2": "about 5", "source": "HAND: a northern goshawk, from memory"},
            "propellers": {"diameter_mm": p["prop_diameter"], "clear_of_the_fuselage_side_mm": round(p["motor_y"] - p["prop_diameter"] / 2 - p["fuselage_diameter"] / 2),
                           "outboard_edge_inside_the_pivot_mm": round(p["pivot_y"] - p["motor_y"] - p["prop_diameter"] / 2), "thrust_N_at_0_10_20_m_s": [round(t, 1) for t in thrust_table([0.0, 10.0, 20.0])]},
            "budget": b}


if __name__ == "__main__":
    import yaml
    print(yaml.safe_dump(checks(), sort_keys=False, allow_unicode=True, width=150))
