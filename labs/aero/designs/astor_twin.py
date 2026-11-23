"""ASTOR-TWIN — the ASTOR product (since 2026-11-22): two wing motors on a fixed wing (the outer panels no longer swing:
held at 12 deg, see WING FIXED below),
the net launcher with three nets. It replaces ASTOR-T (tube launched, talons) as the default; it began as a concept:

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

It flies in ``astor_v_flight`` ("twin": the lattice with the fixed inner wing; drag and maximum lift carried over
from ASTOR-V's CFD until its own) and fires in ``astor_netshot``; its CFD and stability are run by ``sims/astor.sh``. ``checks()`` gives the geometry and a mass estimate (HAND, marked). Frame: x aft from the nose tip, y
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

# ONE PLAIN TAPERED WING (2026-11-23): the inner wing and the outer panels were two shapes (300 -> 225 mm at 25 deg of
# sweep, then 180 -> 110 mm at 12 deg: a step in chord and a kink in both edges). They are now two pieces of ONE
# trapezoid: WING root chord at the centre plane, tip chord at the half span, a straight leading edge at 12 deg, the
# same section and plane throughout. The area (0.2605 m2) and the span (1.382 m) are the old wing's. The joint at
# ``pivot_y`` is only where the outer piece plugs in.
WING = dict(root_chord=260.0, tip_chord=117.0, half_span=691.0, le_x=255.0, joint_y=235.0)
_chord = lambda y: WING["root_chord"] + (WING["tip_chord"] - WING["root_chord"]) * y / WING["half_span"]
_tan = math.tan(math.radians(V.SPREAD_DEG))
_PLAN = {"pivot_y": WING["joint_y"], "panel_root_chord": _chord(WING["joint_y"]), "panel_tip_chord": WING["tip_chord"], "panel_span": WING["half_span"] - WING["joint_y"],
         "panel_pivot_chord": 0.5, "pivot_x": WING["le_x"] + WING["joint_y"] * _tan + 0.5 * _chord(WING["joint_y"]), "panel_stack": 0.0, "wing_plane_z": 20.0}
# A V-TAIL (2026-11-23): ASTOR-T's tail was four surfaces made to fold into a tube (two drooped blades, two hinged fins
# under the tail). With no tube, two fixed surfaces on the body's back do both jobs: VTAIL below, sized in the flight
# model to the four surfaces' pitch and yaw stability. Their trailing edges are the ruddervators.
VTAIL = dict(dihedral_deg=38.0, span=300.0, root=150.0, tip=80.0, root_y=16.0, z=22.0)       # span: each surface, root to tip along itself
_DEFAULTS = {**T._DEFAULTS, "propulsion": "pusher", "tail_anhedral_deg": 0.0, "tail_panel_span": VTAIL["span"], "tail_panel_root": VTAIL["root"],
             "tail_panel_tip": VTAIL["tip"], "tail_pivot_y": VTAIL["root_y"], "dash_tail_fan": 1.0,                    # no motor on the nose (the tail motor is not drawn: _motors is this file's)
             "pivot_slides": False, **_PLAN, "prop_diameter": 279.4, "talon_hinge_x": 380.0, "parachute_x": 265.0}
_NEW = [
    Parameter("glove_span", 2 * WING["joint_y"], "mm", min=200.0, description="the fixed inner wing, tip to tip"),
    Parameter("glove_root_chord", WING["root_chord"], "mm", min=120.0), Parameter("glove_taper", _chord(WING["joint_y"]) / WING["root_chord"], "", min=0.3, max=1.0),
    Parameter("glove_sweep_deg", V.SPREAD_DEG, "deg", min=0.0, max=50.0), Parameter("glove_le_x", WING["le_x"], "mm", min=100.0, description="its leading edge at the centre plane"),
    Parameter("glove_z", 20.0, "mm", description="its plane above the fuselage axis"),
    Parameter("motor_y", 190.0, "mm", min=80.0, description="each wing motor from the centre plane"),
    Parameter("nacelle_d", 54.0, "mm", min=30.0), Parameter("nacelle_len", 130.0, "mm", min=60.0),
    Parameter("vtail_dihedral_deg", VTAIL["dihedral_deg"], "deg", min=20.0, max=55.0, description="each tail surface above the horizontal"),
    Parameter("vtail_z", VTAIL["z"], "mm", description="the tail surfaces' roots above the fuselage axis"),
    Parameter("net_launchers", 0, "", min=0, max=2, description="GAVILAN's recoilless net launcher(s) under the forward belly (0: none; the legs are then left off)"),
    Parameter("launcher_depression_deg", 15.0, "deg", min=0.0, max=15.0, description="the launch tube's tilt below the body axis: it moves between 0 (along the belly) and 15 deg to follow the prey"),
    Parameter("launcher_breech_x", 447.0, "mm", description="the tube's trunnion (at its breech) from the nose tip"),
]
# WING FIXED (2026-11-22): the flight model at 12 / 35 / 60 deg of sweep gave drag at 60 m/s of 20.9 / 21.1 / 21.3 N, stall
# 23.1 / 23.7 / 26.3 m/s and aileron power 1.0 / 0.68 / 0.26: sweeping bought nothing below 100 m/s once the tube was
# dropped. The panels stay at V.SPREAD_DEG; ``wing_sweep_deg`` is kept only so the drawing code is shared with ASTOR-T.
LAUNCHER = dict(tube_mm=300.0, tube_od_mm=45.0, chamber_mm=90.0, chamber_od_mm=60.0, tank_l=1.0)      # gavilan_flight.NET's airborne piece
# The air vessel is INSIDE the fuselage (2026-11-22): a 1 L vessel of the old 86 mm diameter does not go into a 70 mm
# body, so it is shaped to the tail cone, where the dropped nets' tube was, its front end above the breech it feeds.
VESSEL = dict(skin_mm=1.5, gap_mm=1.5, wall_mm=1.5, aft_of_the_tail_end_mm=25.0, cap=0.5)       # HAND: the skin, the gap for the tail's leads, the wall; cap depth / radius


class AstorTwin(T.AstorT):
    """ASTOR-TWIN: one plain tapered wing with two motors, a V-tail, ASTOR-T's body; the net launcher under the belly."""

    parameters = [dataclasses.replace(q, default=_DEFAULTS.get(q.name, q.default), min=0.0 if q.name == "panel_stack" else q.min)
                  for q in T.AstorT.parameters] + _NEW

    def _glove(self, p):
        """The inner piece of the wing: the outer pieces' section and plane, so the two meet without a step."""
        half = V._panel(p["glove_root_chord"], p["glove_root_chord"] * p["glove_taper"], p["glove_span"] / 2, p["glove_sweep_deg"], p["panel_thickness"], p["panel_camber"])
        return half.union(half.mirror("XZ")).translate((p["glove_le_x"], 0, p["glove_z"]))

    def _wing_part(self, p, part):
        out = super()._wing_part(p, part)
        return out.union(self._glove(p)) if part == "fixed" else out

    def _wing(self, p):
        return super()._wing(p).union(self._glove(p))

    def _tail(self, p):
        """The V-tail: two fixed surfaces on the body's back (no fins under the tail, nothing folds)."""
        out = None
        for s in (1, -1):
            pan = V._panel(p["tail_panel_root"], p["tail_panel_tip"], p["tail_panel_span"], 35.0, 0.08)
            pan = pan.translate((-0.25 * p["tail_panel_root"], 0, 0)).rotate((0, 0, 0), (1, 0, 0), p["vtail_dihedral_deg"])
            if s < 0:
                pan = pan.mirror("XZ")
            pan = pan.translate((p["tail_pivot_x"], s * p["tail_pivot_y"], p["vtail_z"]))
            out = pan if out is None else out.union(pan)
        return out

    def _shoulder(self, p):
        """The two joiner tubes the outer wing pieces plug onto (inside the wing: nothing shows outside)."""
        out = None
        for s in (1, -1):
            tube = cq.Workplane("XZ").circle(5.0).extrude(160.0).translate((p["pivot_x"], s * p["pivot_y"] + 80.0, p["wing_plane_z"]))
            out = tube if out is None else out.union(tube)
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
        """Each launcher's breech (its trunnion, fixed under the belly), muzzle and axis at the tube's present tilt:
        ``launcher_depression_deg`` from 0 (along the belly) to 15 deg (the muzzle down)."""
        n = int(p["net_launchers"])
        R = p["fuselage_diameter"] / 2
        a = math.radians(p["launcher_depression_deg"])
        ys = (0.0,) if n == 1 else (-28.0, 28.0)
        L = LAUNCHER["tube_mm"] + LAUNCHER["chamber_mm"]
        zb = -(R + LAUNCHER["chamber_od_mm"] / 2 + 3.0)
        return [{"y": y, "breech": (p["launcher_breech_x"], y, zb), "muzzle": (p["launcher_breech_x"] - L * math.cos(a), y, zb - L * math.sin(a)),
                 "axis": (math.cos(a), 0.0, math.sin(a)), "length": L, "z_breech": zb} for y in ys[:n]]

    def fuselage_radius(self, p, x):
        """The fuselage's outer radius at station x [mm] (the same ogives as ``_fuselage``)."""
        import numpy as np
        from emerillon import ogive
        L, R, re = p["fuselage_length"], p["fuselage_diameter"] / 2, p["tail_end_diameter"] / 2
        nose = ogive(p["nose_length"], R, 40)
        cone = ogive(p["tail_cone"], R - re, 40)[::-1]
        prof = np.vstack([nose, np.column_stack([L - cone[:, 0], cone[:, 1] + re])])
        return float(np.interp(x, prof[:, 0], prof[:, 1]))

    def vessel_layout(self, p) -> dict:
        """The air vessel inside the fuselage: its outer profile [(x, r)] following the tail cone, from the station that
        gives ``LAUNCHER['tank_l']`` inside its wall back to just ahead of the tail end."""
        import numpy as np
        v = VESSEL
        x1 = p["fuselage_length"] - v["aft_of_the_tail_end_mm"]
        ro = lambda x: self.fuselage_radius(p, x) - v["skin_mm"] - v["gap_mm"]

        def profile(x0):
            xs = np.linspace(x0, x1, 240)
            r = np.array([ro(x) for x in xs])
            for xe, sgn in ((x0, 1.0), (x1, -1.0)):                           # elliptical end caps, depth = cap x the radius there
                d = v["cap"] * ro(xe)
                u = np.clip(sgn * (xs - xe) / d, 0.0, 1.0)
                r = r * np.sqrt(1.0 - (1.0 - u) ** 2)
            return xs, r

        def litres(x0):
            xs, r = profile(x0)
            ri = np.maximum(r - v["wall_mm"], 0.0)
            return float(np.trapezoid(math.pi * ri ** 2, xs)) / 1e6

        lo, hi = 150.0, x1 - 60.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if litres(mid) > LAUNCHER["tank_l"] else (lo, mid)
        x0 = 0.5 * (lo + hi)
        xs, r = profile(x0)
        return {"x0": x0, "x1": x1, "xs": xs, "r": r, "litres": litres(x0), "centre_x": float(np.trapezoid(xs * r ** 2, xs) / np.trapezoid(r ** 2, xs)),
                "max_od": 2 * float(r.max()), "aft_od": 2 * ro(x1 - v["cap"] * ro(x1))}

    def _vessel(self, p):
        lay = self.vessel_layout(p)
        pts = [(float(lay["xs"][0]), 0.0)] + [(float(x), float(max(r, 0.05))) for x, r in zip(lay["xs"], lay["r"])] + [(float(lay["xs"][-1]), 0.0)]
        body = cq.Workplane("XY").polyline(pts).close().revolve(360.0, (0, 0, 0), (1, 0, 0))
        bx = p["launcher_breech_x"]                                           # the feed: down through the belly to the valve on the chamber
        feed = cq.Workplane("XY").circle(9.0).extrude(p["fuselage_diameter"] / 2 + 6.0).translate((bx - 30.0, 0, -(p["fuselage_diameter"] / 2 + 6.0)))
        return body.union(feed)

    def _launchers(self, p):
        out = None
        a = p["launcher_depression_deg"]
        R = p["fuselage_diameter"] / 2
        lay = self.launcher_layout(p)
        for k, Lr in enumerate(lay):
            L = Lr["length"]
            # built with the breech at the origin and the muzzle toward the nose (-x), then tilted about the trunnion
            tube = cq.Workplane("YZ").circle(LAUNCHER["tube_od_mm"] / 2).circle(LAUNCHER["tube_od_mm"] / 2 - 1.2).extrude(LAUNCHER["tube_mm"]).translate((-L, 0, 0))
            chamber = (cq.Workplane("YZ").circle(LAUNCHER["chamber_od_mm"] / 2).extrude(LAUNCHER["chamber_mm"]).translate((-LAUNCHER["chamber_mm"], 0, 0))
                       .union(cq.Workplane("YZ").circle(19.0).extrude(40.0)))                                  # the rear venturi behind the trunnion
            trunnion = cq.Workplane("XZ").circle(8.0).extrude(LAUNCHER["chamber_od_mm"] + 16.0).translate((0, (LAUNCHER["chamber_od_mm"] + 16.0) / 2, 0))
            one = tube.union(chamber).rotate((0, 0, 0), (0, 1, 0), -a).union(trunnion).translate(Lr["breech"])
            out = one if out is None else out.union(one)
        return out

    def parts(self, p) -> dict:
        out = super().parts(p)
        if int(p["net_launchers"]) > 0:
            out.pop("talons", None)
            out["launcher"] = self._launchers(p)
            out["vessel"] = self._vessel(p)                                    # inside the fuselage: seen only with the body drawn clear
            out.pop("belly_camera", None)                                      # the dropped nets' camera went with them
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


def vtail_fin_equivalent(p) -> dict:
    """The single upright fin of the V-tail's side force, for the hand lateral formulas: its height is the surfaces'
    rise, its area both surfaces' area x sin^2 of the dihedral (V-tail strip theory)."""
    g = math.radians(p["vtail_dihedral_deg"])
    S = (p["tail_panel_root"] + p["tail_panel_tip"]) * p["tail_panel_span"]                 # both surfaces [mm2]
    h = p["tail_panel_span"] * math.sin(g)
    lam = p["tail_panel_tip"] / p["tail_panel_root"]
    return {"fin_height": h, "fin_root_chord": 2 * S * math.sin(g) ** 2 / ((1 + lam) * h), "fin_taper": lam, "fin_sweep_deg": 35.0}


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
            "geometry": {"muzzle_x_mm": round(lay[0]["muzzle"][0]), "muzzle_below_axis_mm": round(-lay[0]["muzzle"][2]), "muzzle_ahead_of_the_propellers_mm": round(x_prop - lay[0]["muzzle"][0]),
                         "breech_x_mm": p["launcher_breech_x"], "tilt_travel_deg": [0.0, 15.0], "muzzle_below_axis_at_0_deg_mm": round(-lay[0]["z_breech"]), "legs": "left off: the breech's jet is where they fold",
                         "depression_deg": p["launcher_depression_deg"]},
            "pass": {"speed_m_s": round(vp, 1), "bandit_m_s": v_bandit, "closing_m_s": round(closing, 1), "fire_between_m": [lo, hi],
                     "time_in_range_s": round((hi - lo) / closing, 2), "net_flight_s": [round(lo / GF.NET["muzzle_m_s"], 2), round(hi / GF.NET["muzzle_m_s"], 2)],
                     "astor_above_the_bandit_m": [round(lo * math.sin(a) + arm, 1), round(hi * math.sin(a) + arm, 1)],
                     "net_half_span_m": round(GF.NET["open_span_m"] / 2, 2),
                     "net_speed_through_the_air_m_s": round(vp + GF.NET["muzzle_m_s"]), "tested_at_m_s": GF.NET["muzzle_m_s"]},
            "recoil_if_the_rear_jet_were_blocked": {"impulse_Ns": round(I, 1), "speed_change_m_s": round(I / m, 2), "note": "small on this mass: a plain closed breech would do"},
            "air_vessel": vessel_checks(),
            "untested": ["the net opening when fired from an aircraft at 25 m/s (its tests: a fixed launcher, a hovering quadcopter)", "the rear jet along the belly",
                         "the balance with ~1 kg under the forward belly"]}


def vessel_checks() -> dict:
    """The air vessel inside the fuselage: where it is, that it holds its litre, what it moves and what it displaces."""
    import gavilan_flight as GF
    A_ = AstorTwin()
    p = resolve(variant(net_launchers=1, talon_deploy=-1.0, net_count=0))
    lay = A_.vessel_layout(p)
    g = GF.NET["parts_g"]["tank 1.0 L, 30 bar"] * GF.NET["margin"]
    old_x = p["launcher_breech_x"] - 60.0 - 189.0 / 2                          # the 86 mm vessel that hung beside the tube: its centre
    m = one_launcher(3)["aircraft_kg"]
    dcg = g / 1000.0 * (lay["centre_x"] - old_x) / m
    pack = A.PACKS["12S 5000 mAh LiPo"][0]
    pbar = GF.NET["pressure_bar_g"]
    return {"where": "inside the fuselage, shaped to the tail cone (the dropped nets' tube was there); its front end above the breech",
            "from_x_mm": round(lay["x0"]), "to_x_mm": round(lay["x1"]), "length_mm": round(lay["x1"] - lay["x0"]), "largest_diameter_mm": round(lay["max_od"], 1),
            "aft_diameter_mm": round(lay["aft_od"], 1), "volume_l": round(lay["litres"], 3), "pressure_bar": pbar,
            "wall": {"mm": VESSEL["wall_mm"], "hoop_stress_MPa": round(pbar * 0.1 * (lay["max_od"] / 2) / VESSEL["wall_mm"], 1), "note": "aluminium at this stress has a large reserve; the mass stays gavilan_flight's 250 g (HAND)"},
            "clear_of_the_skin_mm": VESSEL["gap_mm"], "feed_to_the_valve_mm": round(p["fuselage_diameter"] / 2 + 6.0),
            "balance": {"vessel_g_with_margin": round(g), "centre_moves_aft_mm": round(lay["centre_x"] - old_x), "aircraft_cg_moves_aft_mm": round(dcg, 1),
                        "pack_moves_forward_to_hold_it_mm": round(dcg * m * 1000.0 / pack), "air_used_per_shot_g": GF.NET["air_g_per_shot"],
                        "cg_shift_per_shot_mm": round(-GF.NET["air_g_per_shot"] / 1000.0 * (lay["centre_x"] - V.CG_MM) / m, 2)},
            "drag": {"removed_cm2": 8.0, "source": "gavilan_flight.NET_CDA's 0.0008 m2 for the vessel under the belly (HAND)"},
            "displaced": ["the parachute bay: forward under the glove (station %d)" % p["parachute_x"], "the dropped nets' belly camera: removed",
                          "the tail's servo leads: in the %.1f mm gap along the vessel (HAND)" % VESSEL["gap_mm"],
                          "the two motor controllers: in the nacelles behind the motors (HAND: no room was ledgered for them in the body)"],
            "not_checked": ["a stationed balance of ASTOR-TWIN (none exists yet: the flight model holds the CG at %d mm)" % V.CG_MM,
                            "the V-tail's root fittings at station %d sit on the body's back over the vessel (HAND)" % p["tail_pivot_x"],
                            "the pack's own fit in the 70 mm body"]}


MAGAZINE = {"reload_g": (0.0, 180.0, 220.0),        # HAND: a breech feed for the second cartridge, a longer one for the third (+-60)
            "gas_g": (0.0, 320.0, 320.0),           # HAND: a 0.2 L, 300 bar bottle (70 g of air; a shot takes 19 g), its regulator and line, to refill the 1 L vessel
            "p_hit": {"straight": 0.9, "weave": 0.6, "jink": 0.15}}     # HAND, from twelve ASTOR-T passes against the net's 1.5 m half span: NOT a simulation


def one_launcher(nets=1, v_bandit=9.0) -> dict:
    """ONE launcher (one tube, valve and vessel) with 1, 2 or 3 net cartridges: the mass with the reload feed and the gas
    for the later shots, the aircraft with it, and what the extra nets are worth (a hit chance per shot per kind of
    quadcopter, HAND, over astor_hunt's population; one pass per net)."""
    import gavilan_flight as GF
    import astor_hunt as H
    parts = sum(GF.NET["parts_g"].values()) + (nets - 1) * GF.NET["cartridge_g"] + MAGAZINE["reload_g"][nets - 1] + MAGAZINE["gas_g"][nets - 1]
    kg = parts * GF.NET["margin"] / 1000.0
    c = checks()
    S = c["wing"]["total_m2"]
    m = c["mass"]["total_kg"] - (220.0 + 2 * V.NET_G + 40.0) * 1.1 / 1000.0 + kg
    stall = math.sqrt(2 * m * 9.80665 / (1.225 * S * 0.87))
    win = sum(w * (1 - (1 - MAGAZINE["p_hit"][b]) ** nets) for b, w in H.POPULATION["behaviour"])
    return {"nets": nets, "launcher_kg": round(kg, 2), "against_separate_launchers_kg": round(GF.net_kg(nets), 2), "aircraft_kg": round(m, 2),
            "stall_spread_m_s": round(stall, 1), "thrust_over_weight_at_32_m_s": round(113.0 / (m * 9.80665), 2),
            "closing_on_the_bandit_m_s": round(1.3 * stall - v_bandit, 1), "seconds_in_range": round(6.0 / (1.3 * stall - v_bandit), 2),
            "time_for_all_passes_s": 15 + 40 * (nets - 1), "win_rate_estimate": round(win, 2),
            "win_rate_note": "the hit chances per shot are HAND guesses; a bandit netted is counted a win"}


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
             ("wing fixed at 12 deg: the two sweep servos and linkages off", -60.0, "astor_v.budget's item"),
             ("outer panels on plug-in joiners (off for transport)", 30.0, "HAND (+-15)"),
             ("V-tail: the two fin hinges, springs and locks off", -18.0, "astor_t.budget's item"),
             ("V-tail: the tail blades' pivots and fold actuator off", -40.0, "astor_v.budget's item"),
             ("V-tail: two ruddervator servos, hinges and root fittings", 36.0, "HAND (+-12): 2 x 12 g servos + 12 g; the surfaces' area is the old four's (0.069 against 0.070 m2)"),
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
            "tail": {"kind": "V", "dihedral_deg": p["vtail_dihedral_deg"], "each_surface_mm": {"span": p["tail_panel_span"], "root": p["tail_panel_root"], "tip": p["tail_panel_tip"]},
                     "area_both_m2": round((p["tail_panel_root"] + p["tail_panel_tip"]) * p["tail_panel_span"] / 1e6, 4), "rise_above_the_root_mm": round(p["tail_panel_span"] * math.sin(math.radians(p["vtail_dihedral_deg"]))),
                     "controls": "ruddervators: together pitch, opposed yaw; roll is the ailerons' alone"},
            "stall_spread_m_s": {"value": round(math.sqrt(2 * m * 9.80665 / (1.225 * S * cl)), 1), "note": f"CL max {cl} carried over from ASTOR-V's CFD (HAND for this wing)"},
            "wing_loading_kg_m2": round(m / S, 1), "current_A_at_8_kW": round(8000.0 / T.PACK_V)}
