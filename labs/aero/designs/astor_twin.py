"""ASTOR-TWIN — a concept: ASTOR with two wing motors on a fixed inner wing and swinging outer panels (2026-11-22).

Asked for after the catch studies: more pull than one nose propeller gives, the tube set aside for now. The layout
is the swing-wing jets' (F-111, F-14): a fixed inner wing (the "glove") carries the two motors; only the outer panels
sweep, on pivots outboard of the nacelles. Nothing turns on the nose: the camera looks ahead through clear air. The
body, the tail blades and fins, the net tube and the legs with their line are ASTOR-T's.

- **Glove**: 500 mm span, 300 -> 225 mm chord, through the fuselage's upper half.
- **Motors**: two 4 kW units in nacelles on the glove at +-190 mm, 11-inch folding propellers (15 mm clear of the
  fuselage's side); 8 kW together: ~113 N at 32 m/s against ASTOR-T's 57 N (the propulsion library).
- **Outer panels**: 180 -> 110 mm chord, 460 mm long, on pivots at +-235 mm, swept 12 deg (spread) to 60 deg (dash);
  spread span 1.38 m. The pivots do not slide: with only the outer wing sweeping, the lift moves much less than
  ASTOR-T's whole wing does (to be checked with the lattice).
- **Legs**: ASTOR-T's, under the centre; the propellers are 190 mm to each side and above them.

THIS IS A DRAWING AND A FIRST MASS, NOT YET A FLYING MODEL: no lattice, balance, stability or catch sortie has been
run on it. ``checks()`` gives the geometry and a mass estimate (HAND, marked). Frame: x aft from the nose tip, y
right, z up; mm.
"""
from __future__ import annotations

import dataclasses
import math

import cadquery as cq

import astor as A
import astor_t as T
import astor_v as V
from vegeta.dedalus import Parameter

_DEFAULTS = {**T._DEFAULTS, "propulsion": "pusher",                    # no motor on the nose (the tail motor is not drawn: _motors is this file's)
             "pivot_y": 235.0, "pivot_x": 395.0, "pivot_slides": False, "panel_span": 460.0, "panel_root_chord": 180.0, "panel_tip_chord": 110.0,
             "panel_stack": 4.0, "wing_plane_z": 8.0, "prop_diameter": 279.4, "talon_hinge_x": 380.0}
_NEW = [
    Parameter("glove_span", 500.0, "mm", min=200.0, description="the fixed inner wing, tip to tip"),
    Parameter("glove_root_chord", 300.0, "mm", min=120.0), Parameter("glove_taper", 0.75, "", min=0.3, max=1.0),
    Parameter("glove_sweep_deg", 25.0, "deg", min=0.0, max=50.0), Parameter("glove_le_x", 270.0, "mm", min=100.0, description="its leading edge at the centre plane"),
    Parameter("glove_z", 20.0, "mm", description="its plane above the fuselage axis"),
    Parameter("motor_y", 190.0, "mm", min=80.0, description="each wing motor from the centre plane"),
    Parameter("nacelle_d", 54.0, "mm", min=30.0), Parameter("nacelle_len", 130.0, "mm", min=60.0),
    Parameter("net_launchers", 0, "", min=0, max=2, description="GAVILAN's recoilless net launcher(s) under the forward belly (0: none; the legs are then left off)"),
    Parameter("launcher_depression_deg", 15.0, "deg", min=0.0, max=45.0, description="the launch tube below the body axis"),
    Parameter("launcher_muzzle_x", 70.0, "mm", description="the muzzle from the nose tip"),
]
LAUNCHER = dict(tube_mm=300.0, tube_od_mm=45.0, chamber_mm=90.0, chamber_od_mm=60.0, tank_l=1.0, tank_od_mm=86.0)      # gavilan_flight.NET's airborne piece


class AstorTwin(T.AstorT):
    """ASTOR-TWIN: a fixed inner wing with two motors, swinging outer panels, ASTOR-T's body, tail and legs."""

    parameters = [dataclasses.replace(q, default=_DEFAULTS.get(q.name, q.default)) for q in T.AstorT.parameters] + _NEW

    def _glove(self, p):
        return self._surface(p["glove_le_x"], p["glove_root_chord"], p["glove_taper"], p["glove_span"], p["glove_sweep_deg"], 0.10, "y", z0=p["glove_z"])

    def _wing_part(self, p, part):
        out = super()._wing_part(p, part)
        return out.union(self._glove(p)) if part == "fixed" else out

    def _wing(self, p):
        return super()._wing(p).union(self._glove(p))

    def _shoulder(self, p):
        """The two pivot pins at the glove's tips (no carriage: the pivots are fixed)."""
        out = None
        for s in (1, -1):
            pin = cq.Workplane("XY").circle(9.0).extrude(26.0).translate((p["pivot_x"], s * p["pivot_y"], p["wing_plane_z"] - 8.0))
            out = pin if out is None else out.union(pin)
        return out

    def _nacelle_x(self, p):
        return p["glove_le_x"] + p["motor_y"] * math.tan(math.radians(p["glove_sweep_deg"])) - 0.55 * p["nacelle_len"]

    def _motors(self, p):
        out, x0 = None, self._nacelle_x(p)
        for s in (1, -1):
            n = (cq.Workplane("YZ").circle(p["nacelle_d"] / 2).extrude(p["nacelle_len"]).translate((x0, s * p["motor_y"], p["glove_z"]))
                 .faces(">X").edges().fillet(p["nacelle_d"] * 0.35))
            hub = cq.Workplane("YZ").circle(11.0).extrude(-12.0).translate((x0, s * p["motor_y"], p["glove_z"]))
            out = n.union(hub) if out is None else out.union(n).union(hub)
        return out

    def _props(self, p):
        out, x0 = None, self._nacelle_x(p) - 12.0
        for s in (1, -1):
            d = cq.Workplane("YZ").circle(p["prop_diameter"] / 2).extrude(1.5).translate((x0, s * p["motor_y"], p["glove_z"]))
            out = d if out is None else out.union(d)
        return out

    def launcher_layout(self, p) -> list:
        """Each launcher's muzzle and axis: side by side under the forward belly, pointing forward and down."""
        n = int(p["net_launchers"])
        R = p["fuselage_diameter"] / 2
        a = math.radians(p["launcher_depression_deg"])
        ys = (0.0,) if n == 1 else (-28.0, 28.0)
        L = LAUNCHER["tube_mm"] + LAUNCHER["chamber_mm"]
        # the breech end tucked under the belly; the muzzle lower by the depression
        z_breech = -(R + LAUNCHER["chamber_od_mm"] / 2 + 3.0)
        return [{"y": y, "muzzle": (p["launcher_muzzle_x"], y, z_breech - L * math.sin(a)), "axis": (math.cos(a), 0.0, math.sin(a)), "length": L, "z_breech": z_breech}
                for y in ys[:n]]

    def _launchers(self, p):
        out = None
        a = p["launcher_depression_deg"]
        R = p["fuselage_diameter"] / 2
        for k, Lr in enumerate(self.launcher_layout(p)):
            tube = cq.Workplane("YZ").circle(LAUNCHER["tube_od_mm"] / 2).circle(LAUNCHER["tube_od_mm"] / 2 - 1.2).extrude(LAUNCHER["tube_mm"])
            chamber = (cq.Workplane("YZ").circle(LAUNCHER["chamber_od_mm"] / 2).extrude(LAUNCHER["chamber_mm"]).translate((LAUNCHER["tube_mm"], 0, 0))
                       .union(cq.Workplane("YZ").circle(19.0).extrude(40.0).translate((LAUNCHER["tube_mm"] + LAUNCHER["chamber_mm"], 0, 0))))     # the rear venturi
            one = tube.union(chamber).rotate((0, 0, 0), (0, 1, 0), -a).translate(Lr["muzzle"])       # the breech up under the belly
            rt = LAUNCHER["tank_od_mm"] / 2
            lt = LAUNCHER["tank_l"] * 1e6 / (math.pi * (rt - 2.0) ** 2)
            tank = (cq.Workplane("YZ").circle(rt).extrude(lt).edges().fillet(rt * 0.6)
                    .translate((470.0, (0.0 if len(self.launcher_layout(p)) == 1 else (-1 if k == 0 else 1) * (rt + 2.0)), -(R + rt + 2.0))))
            one = one.union(tank)
            out = one if out is None else out.union(one)
        return out

    def parts(self, p) -> dict:
        out = super().parts(p)
        if int(p["net_launchers"]) > 0:
            out.pop("talons", None)
            out["launcher"] = self._launchers(p)
        return out

    def _propeller(self, p):
        import astor_prop
        P = astor_prop.FoldingProp()
        one = P.build(P.resolve(diameter=p["prop_diameter"], fold_deg=p["prop_fold_deg"]))
        x0 = self._nacelle_x(p) - 18.0
        return one.translate((x0, p["motor_y"], p["glove_z"])).union(one.mirror("XZ").translate((x0, -p["motor_y"], p["glove_z"])))


def variant(**over) -> dict:
    q = {**V._OVERRIDE, **_DEFAULTS}
    q["tail_le_x"] = q["fuselage_length"] - 10.0 - q["htail_root_chord"]
    return {**q, **over}


def resolve(q=None) -> dict:
    return AstorTwin().resolve(**dict(q or variant()))


def launcher_checks(shots=1, v_pass=None, v_bandit=9.0) -> dict:
    """Can ASTOR-TWIN carry GAVILAN's net launcher (``gavilan_flight.NET``: a 150 g cartridge, a 2.4 m net at 40 m/s,
    captures 5-6 of 6 between 4 and 10 m from a fixed launcher at a hovering quadcopter)? Mass and stall with it, where it
    goes and what it clears, and the firing pass. The launcher's masses are that file's hand estimates."""
    import gavilan_flight as GF
    p = resolve(variant(net_launchers=shots, talon_deploy=-1.0))
    A_ = AstorTwin()
    c = checks()
    S = c["wing"]["total_m2"]
    legs_g, nets_g = 220.0 * 1.1, (2 * V.NET_G + 40.0) * 1.1                 # what comes off: the legs and line, the dropped nets and their tube
    m = c["mass"]["total_kg"] - (legs_g + nets_g) / 1000.0 + GF.net_kg(shots)
    stall = math.sqrt(2 * m * 9.80665 / (1.225 * S * 0.87))
    vp = v_pass or 1.3 * stall
    lay = A_.launcher_layout(p)
    x_prop = A_._nacelle_x(p) - 12.0
    a = math.radians(p["launcher_depression_deg"])
    lo, hi = GF.NET["range_m"]
    closing = vp - v_bandit
    I = GF.NET["cartridge_g"] / 1000.0 * GF.NET["muzzle_m_s"]              # the shot's momentum [N s]
    arm = -lay[0]["muzzle"][2] / 1000.0
    return {"launcher": {"shots": shots, "kg": round(GF.net_kg(shots), 2), "drag_area_cm2": round(GF.NET_CDA * 1e4 * shots, 1), "source": GF.NET["source"]},
            "taken_off": {"legs_and_line_g": round(legs_g), "dropped_nets_and_tube_g": round(nets_g)},
            "mass_kg": round(m, 2), "stall_spread_m_s": round(stall, 1), "wing_loading_kg_m2": round(m / S, 1),
            "thrust_over_weight_at_32_m_s": round(113.0 / (m * 9.80665), 2),
            "geometry": {"muzzle_x_mm": p["launcher_muzzle_x"], "muzzle_below_axis_mm": round(-lay[0]["muzzle"][2]), "muzzle_ahead_of_the_propellers_mm": round(x_prop - p["launcher_muzzle_x"]),
                         "breech_x_mm": round(p["launcher_muzzle_x"] + lay[0]["length"] * math.cos(a)), "legs": "left off: the breech's jet and the tank are where they fold",
                         "depression_deg": p["launcher_depression_deg"]},
            "pass": {"speed_m_s": round(vp, 1), "bandit_m_s": v_bandit, "closing_m_s": round(closing, 1), "fire_between_m": [lo, hi],
                     "time_in_range_s": round((hi - lo) / closing, 2), "net_flight_s": [round(lo / GF.NET["muzzle_m_s"], 2), round(hi / GF.NET["muzzle_m_s"], 2)],
                     "astor_above_the_bandit_m": [round(lo * math.sin(a) + arm, 1), round(hi * math.sin(a) + arm, 1)],
                     "net_half_span_m": round(GF.NET["open_span_m"] / 2, 2),
                     "net_speed_through_the_air_m_s": round(vp + GF.NET["muzzle_m_s"]), "tested_at_m_s": GF.NET["muzzle_m_s"]},
            "recoil_if_the_rear_jet_were_blocked": {"impulse_Ns": round(I, 1), "speed_change_m_s": round(I / m, 2), "note": "small on this mass: a plain closed breech would do"},
            "untested": ["the net opening when fired from an aircraft at 25 m/s (its tests: a fixed launcher, a hovering quadcopter)", "the rear jet along the belly",
                         "the balance with ~1 kg under the forward belly"]}


def checks() -> dict:
    p = resolve()
    lam = p["glove_taper"]
    S_glove = 0.5 * p["glove_root_chord"] * (1 + lam) * p["glove_span"] / 1e6
    S_pan = (p["panel_root_chord"] + p["panel_tip_chord"]) * p["panel_span"] / 1e6
    span = lambda sw: 2 * (p["pivot_y"] + p["panel_span"] * math.cos(math.radians(sw - V.SPREAD_DEG)) * math.cos(math.radians(V.SPREAD_DEG))) / 1000.0
    mt = T.budget()["total_g"]
    items = [("ASTOR-T complete (4 kW, 12S 2500 mAh, legs, nets)", mt, "astor_t.budget"),
             ("second motor, 4 kW class", 460.0, "catalogue (as ASTOR-T's)"), ("second ESC 12S 120 A", 150.0, "HAND (+-25)"),
             ("pack: 12S 5000 mAh in place of 2500 (190 A)", 760.0, "HAND: as astor.PACKS, doubled"),
             ("second folding propeller, two nacelles and mounts", 110.0, "HAND (+-40)"),
             ("glove in place of ASTOR-T's wing carriage and inner panels", 1000 * A.WING_KG_M2 * S_glove - 130.0, f"geometry {S_glove:.3f} m2 x {A.WING_KG_M2} kg/m2 less the carriage (HAND)"),
             ("growth margin 10 % of the additions", 0.0, "HAND")]
    add = sum(x[1] for x in items[1:-1])
    items[-1] = (items[-1][0], 0.10 * add, "HAND")
    m = (mt + 1.10 * add) / 1000.0
    S = S_glove + S_pan
    cl = 0.87
    return {"status": "a drawing and a first mass: no lattice, balance, stability or catch sortie yet",
            "wing": {"glove_m2": round(S_glove, 3), "outer_panels_m2": round(S_pan, 3), "total_m2": round(S, 3), "span_spread_m": round(span(12.0), 2), "span_dash_m": round(span(60.0), 2)},
            "propellers": {"diameter_mm": p["prop_diameter"], "clear_of_the_fuselage_side_mm": round(p["motor_y"] - p["prop_diameter"] / 2 - p["fuselage_diameter"] / 2),
                           "between_the_two_discs_mm": round(2 * p["motor_y"] - p["prop_diameter"]), "thrust_N_at_0_32_60_m_s": [136, 113, 84],
                           "astor_t_thrust_N_at_0_32_60_m_s": [68, 57, 42], "source": "propulsion library, 2 x prop-11x11-b2 at 4 kW"},
            "mass": {"items": [{"item": n, "g": round(g), "source": s} for n, g, s in items], "total_kg": round(m, 2)},
            "stall_spread_m_s": {"value": round(math.sqrt(2 * m * 9.80665 / (1.225 * S * cl)), 1), "note": f"CL max {cl} carried over from ASTOR-V's CFD (HAND for this wing)"},
            "wing_loading_kg_m2": round(m / S, 1), "current_A_at_8_kW": round(8000.0 / T.PACK_V)}
