"""ASTOR-V and ASTOR-BOOST in flight: the full coefficient set at any wing sweep, and the simulator that flies them
(``SimV``: the wing sweeps on its carriage, the tail fan is the elevator and — its two panels apart — the ailerons,
the nose propeller can stop and fold, ASTOR-BOOST's fan can push).

**Where each number comes from** (marked in ``C["sources"]``):

- *longitudinal* (lift slope, lift at zero angle, moment about the CG, the tail fan's lift and moment per degree, the
  neutral point): ``astor_v_model``'s vortex lattice on the panels as they lie at that sweep, carriage position
  included;
- *lateral, damping, inertias*: ``arigo1_model.build`` (the family's hand formulas) on an equivalent conventional
  wing of the same span, area and sweep (capped at 35 deg: the formula's range), with ASTOR-V's fin and tail;
- *roll control*: ailerons on the outer panels (the lattice with the flapped strips at incidence, x 0.5 flap
  effectiveness, x cos^2 of the sweep) together with the tail panels moved apart (tailerons: the lattice's tail lift
  per degree at the panels' arm);
  ``astor_v_model`` found differential sweep too weak (8 deg ~ 1 deg of aileron);
- *drag*: drag areas (HAND until the CFD): the body's, the wing's by its exposed area, the pod's for ASTOR-BOOST, the
  stopped propeller's folded blades; *maximum lift*: 0.95 spread, falling with the cosine of the sweep (HAND);
- *thrust*: the propulsor library — the 9 x 11 propeller as a tractor at 1.4 kW; ASTOR-BOOST's 70 mm fan at 1.2 kW;
- *CFD* (``data/astor_v_cfd_screening.yaml``, when ``sims/astor_cfd.sh`` has run): per design and sweep it corrects
  the zero-lift drag, the maximum lift and its angle, the lift slope and the neutral point.

All in ``arigo1_model``'s conventions, so ``arigo1_model.aero``, ``trim`` and ``arigo1_mujoco``'s autopilot work.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

import arigo1_model as am
import arigo1_mujoco as amj
import astor as A
import astor_model as AM
import astor_v as V
import astor_v_model as VM

TWINS = ("twin", "twin1", "goshawk")        # + GOSHAWK (designs/goshawk: the same drawing and model code at its own numbers)
_TW = ("twin", "twin1")                   # ASTOR-TWIN: HAWK 2 (the doubled wing, the design) and HAWK1 (the aircraft before it)
TAIL_FLAP_GEAR = 1.0                        # the tail flaps move this many degrees per degree of the tail half's own angle
VP_STATIC, VP_BLEND_M_S = 1.62, 60.0        # HAND: HAWK 2's variable-pitch propellers (see coefficients)
FIXED_WING = TWINS                      # designs whose wing does not swing in flight: held at SWEEPS[0] (ASTOR-TWIN since 2026-11-22)
SWEEPS = (12.0, 35.0, 60.0)                 # the states the model is built at; between them it is interpolated
BODY_CDA = 0.0030                           # HAND: fuselage, shoulder carriage, fin, tail fan, camera blister [m2] (+-0.0008)
WING_CD = 0.010                             # HAND: the wing panels' zero-lift drag on their exposed area
POD_CDA = 0.0009                            # HAND: ASTOR-BOOST's pod and pylon, fan windmilling [m2] (+-0.0004)
FOLDED_PROP_CDA = 0.0002                    # HAND: the stopped, folded blades
CL_MAX_SPREAD = 0.95                        # HAND (+-0.1) until the CFD
FAN_W = 1200.0
TAIL_Y_M = 0.085                            # the tail panels' lift centre from the centre plane (HAND, from the geometry)


def design(name: str):
    """(resolved parameters, mass budget) of a design: "v" (ASTOR-V) or "boost" (ASTOR-BOOST)."""
    if name == "boost":
        import astor_boost as B
        p = B.resolve()
        return p, B.budget(p)
    if name == "t":
        import astor_t as T
        p = T.resolve()
        return p, T.budget(p)
    if name == "goshawk":
        import goshawk as GH
        p = GH.resolve()
        return p, GH.budget(p)
    if name in TWINS:                                                        # ASTOR-TWIN with one net launcher and three nets (astor_twin)
        import astor_twin as W
        p = W.resolve((W.variant1 if name == "twin1" else W.variant)(net_launchers=1, talon_deploy=-1.0, net_count=0))
        g, wh, amps, src = A.PACKS["12S 5000 mAh LiPo"]
        tot = (W.MASS1_KG if name == "twin1" else W.one_launcher(3)["aircraft_kg"]) * 1000.0
        return p, {"total_g": tot, "empty_g": tot - g, "without_nets_g": tot - 3 * 150.0, "pack": {"name": "12S 5000 mAh LiPo", "g": g, "Wh_nominal": wh, "max_A": amps, "source": src},
                   "items": [{"item": "ASTOR-TWIN with one launcher and three nets, less the pack", "g": round(tot - g), "source": "astor_twin.one_launcher(3): HAND"}]}
    p = V.resolve()
    return p, V.budget(p)


def cfd_config(name: str, sweep: float):
    import yaml
    f = am.DATA / "astor_v_cfd_screening.yaml"
    if not f.exists():
        return None
    d = yaml.safe_load(f.read_text()) or {}
    return (d.get("configurations") or {}).get(f"{name}_sweep{sweep:.0f}")


@lru_cache(maxsize=None)
def coefficients(name: str = "twin", sweep: float = 12.0) -> dict:
    import arigo_propulsion as ap
    p, mb = design(name)
    m = mb["total_g"] / 1000.0
    wn = V.wing_numbers(p, sweep)
    b = max(wn["span_mm"], 520.0) / 1000.0
    S = wn["exposed_area_m2"] + 0.07 * 0.21                                  # the exposed panels + the carry-over across the body
    glove = 0.0
    if name in TWINS:                                                        # + the fixed inner wing
        glove = 0.5 * p["glove_root_chord"] * (1 + p["glove_taper"]) * p["glove_span"] / 1e6
        S = wn["exposed_area_m2"] + glove
    # the equivalent conventional wing for the hand formulas (lateral, damping, inertias)
    lam = p["panel_tip_chord"] / p["panel_root_chord"]
    fin = p                                                                  # ASTOR-T: its two splayed fins as one upright fin of the same side force
    import astor_t as T
    if name == "t" or (name in TWINS and p.get("fin_splay_deg", 0.0) > 0 and not p.get("single_fin")):
        fin = T.fin_equivalent(p)
    tail_y = TAIL_Y_M
    if p.get("single_fin"):                                                  # ASTOR-TWIN: one upright fin, two all-moving tail halves
        lt = p["tail_panel_tip"] / p["tail_panel_root"]
        tail_y = (p["tail_pivot_y"] + p["tail_panel_span"] * (1 + 2 * lt) / (3 * (1 + lt))) / 1000.0     # each half's area centre from the centre plane
    if p.get("vtail_dihedral_deg", 0.0) > 0:                                 # a V-tail: the upright fin of its side force (its area x sin^2 of the dihedral)
        import astor_twin as W
        fin = W.vtail_fin_equivalent(p)
    cr = 2 * S / (b * (1 + lam)) * 1000.0
    q = A.variant(span=b * 1000.0, root_chord=cr, taper=lam, le_sweep_deg=min(sweep, 35.0), dihedral_deg=0.0,
                  wing_x=V.pivot_x(p, sweep) - 0.3 * cr, cg_x=V.CG_MM, propulsion="tractor_lowcam",
                  fuselage_length=p["fuselage_length"], tail_cone=p["tail_cone"], tail_end_diameter=p["tail_end_diameter"],
                  htail_span=2 * (p["tail_pivot_y"] + p["tail_panel_span"] * math.cos(math.radians(35.0))), htail_root_chord=p["tail_panel_root"],
                  htail_taper=p["tail_panel_tip"] / p["tail_panel_root"], htail_sweep_deg=35.0, htail_z=p.get("tail_z", 6.0),
                  fin_height=fin["fin_height"], fin_root_chord=fin["fin_root_chord"], fin_taper=fin.get("fin_taper", p["fin_taper"]), fin_sweep_deg=fin.get("fin_sweep_deg", p["fin_sweep_deg"]),
                  lowcam_drop=p["lowcam_drop"])
    q["tail_le_x"] = p["tail_pivot_x"] - 0.25 * p["tail_panel_root"]
    pe = A.resolve(q)
    L = A.Astor.layout(pe)
    power = A.MOTOR_W
    if name == "t":
        power = T.MOTOR_W
    if name in TWINS:                                                        # two 4 kW motors, an 11-inch propeller each
        sel = {**AM.propulsor(m / 2 if name == "twin1" else 3.46, 4000.0, (11.0,)), "units": 2}
    else:
        sel = AM.propulsor(m, power, (T.PROP_IN,)) if name == "t" else AM.propulsor(m, power)
    sel = {**sel, "layout": "tractor"}
    thrust = ap.thrust_table(sel, V=[float(v) for v in range(0, 92, 2)])
    if name == "goshawk":                                                    # GOSHAWK's two motors: momentum theory (HAND), goshawk.thrust_table
        import goshawk as GH
        tt = GH.thrust_table(thrust["table_V"])
        thrust = {**thrust, "table_T": tt, "T0_N": tt[0], "propulsor": {**thrust["propulsor"], "entry": f"2 x {GH.PROP_IN:g} in at {GH.MOTOR_W:g} W, momentum theory (HAND)", "power_w_per_unit": GH.MOTOR_W}}
    if name == "twin":
        # HAWK 2's 13-inch VARIABLE-PITCH propellers are not in the propulsion library. HAND: the library's 11x13 (pitched
        # for the dash) is kept for the fast end, where a propeller of either size turns the same power into thrust; at
        # rest, fine pitch and the larger disc give momentum theory's thrust at a figure of merit of 0.6 (213 N for the
        # two against the 11x13's 125 N: x VP_STATIC), falling in a straight line to the library's by VP_BLEND_M_S
        f = [VP_STATIC + (1.0 - VP_STATIC) * min(1.0, v / VP_BLEND_M_S) for v in thrust["table_V"]]
        thrust = {**thrust, "table_T": [t * k for t, k in zip(thrust["table_T"], f)], "T0_N": thrust["T0_N"] * VP_STATIC,
                  "propulsor": {**thrust["propulsor"], "entry": "13 in variable pitch (HAND) on " + str(thrust["propulsor"].get("entry"))}}
    cda = BODY_CDA + WING_CD * wn["exposed_area_m2"] + (POD_CDA if name == "boost" else 0.0)
    C = am.build(q, pe, L, f"astor_{name}", "tractor_lowcam", mass=m, CD0=cda / (L["S_mm2"] / 1e6), thrust=thrust)
    Se, ce = C["S"], C["c"]
    # the longitudinal set from the lattice on the real panels, brought to this reference (S, c) and to the CG
    kL, kM = VM.S_REF / Se, VM.S_REF * VM.C_REF / (Se * ce)
    xc = V.CG_MM / 1000.0 / VM.C_REF
    tf = 1.0 if sweep < 50.0 else p.get("dash_tail_fan", 0.6)                                        # the dash flies with the tail fan part folded
    r0, r4, re = VM.lattice(sweep, tail_fan=tf, p=p), VM.lattice(sweep, tail_fan=tf, alpha=4.0, p=p), VM.lattice(sweep, tail_fan=tf, tail_inc=1.0, p=p)
    cm = lambda r: r["Cm"] + r["CL"] * xc
    CLa = (r4["CL"] - r0["CL"]) / math.radians(4.0) * kL
    Cma = (cm(r4) - cm(r0)) / math.radians(4.0) * kM
    x_np = -(r4["Cm"] - r0["Cm"]) / (r4["CL"] - r0["CL"]) * VM.C_REF
    CLde = (re["CL"] - r0["CL"]) * kL
    CL_MAX = CL_MAX_SPREAD * max(math.cos(math.radians(sweep - 12.0)), 0.55) * VM.S_REF / Se * (wn["exposed_area_m2"] / 0.198)
    C.update({"aircraft": {"boost": "ASTOR-BOOST", "t": "ASTOR-T", "twin": "ASTOR-TWIN", "twin1": "ASTOR-TWIN (HAWK1)", "goshawk": "GOSHAWK"}.get(name, "ASTOR-V"), "design": name, "sweep_deg": sweep, "tail_fan": tf,
              "CL0": r0["CL"] * kL, "CLa": CLa, "Cm0": cm(r0) * kM, "Cma": Cma, "CLde": CLde, "Cmde": (cm(re) - cm(r0)) * kM,
              "x_cg_m": V.CG_MM / 1000.0, "x_np_m": x_np, "static_margin": -Cma / CLa * 1.0,
              "CL_MAX": CL_MAX, "alpha_stall": (CL_MAX - r0["CL"] * kL) / CLa,
              "Clda_tail": 0.0 if p.get("vtail_dihedral_deg", 0.0) > 0 else CLde * tail_y / C["b"], "Clda_aileron": VM.roll_per_deg_aileron(sweep, p) / (Se * C["b"]),                 # tailerons: +-1 deg on the two tail panels
              "Cnda": 0.0, "Cnb": C["Cnb"] * 0.9,                                        # HAND: the tractor propeller's normal force ahead of the CG (-10 %)
              "mass_budget": mb, "pack_kind": "astor",
              "equipment": {"product": "astor", "launch": "tube" if name == "t" else "rail", "recovery": "parachute", "turning_station": True, "kits": [],
                            "pack": mb["pack"]["name"], "nets": int(p["net_count"])},
              "aero_source": "model (lattice on the swept panels + hand)"})
    if p.get("tail_anhedral_deg", 0.0) > 0:
        # a drooped (inverted V) tail keeps the nose straight as well: its normal-force slope (the lattice's lift per
        # degree of tail incidence, over cos^2 of the droop) x sin^2 of the droop, at the tail's arm (V-tail strip theory)
        td2 = math.tan(math.radians(p["tail_anhedral_deg"])) ** 2
        C["Cnb_tail"] = CLde * 57.3 * td2 * (p["tail_pivot_x"] / 1000.0 + 0.05 - C["x_cg_m"]) / C["b"]
        C["Cnb"] += C["Cnb_tail"]
        C["Cnr"] *= 1.0 + C["Cnb_tail"] / max(abs(C["Cnb"] - C["Cnb_tail"]) + 0.03, 0.03)     # HAND: its yaw damping in proportion
    if p.get("tail_flap_chord", 0.0) > 0.0 and name in TWINS:
        # the tail halves' flaps, geared 1:1 to the all-moving angle: thin-aerofoil effectiveness for their chord x 0.65 (the
        # plain-flap efficiency used for the wing's, HAND +-0.1) more lift per degree of tail command
        th = math.acos(2 * p["tail_flap_chord"] - 1)
        gain = 1.0 + (1 - (th - math.sin(th)) / math.pi) * 0.65 * TAIL_FLAP_GEAR
        for key in ("CLde", "Cmde", "Clda_tail"):
            C[key] *= gain
        C["tail_flaps"] = {"chord": p["tail_flap_chord"], "gear": TAIL_FLAP_GEAR, "control_gain": round(gain, 2), "source": "HAND: thin-aerofoil flap effectiveness x 0.65"}
    C["Clda"] = C["Clda_tail"] + C["Clda_aileron"]                           # one roll command moves both
    C["Cnda"] = -0.1 * C["Clda_aileron"]                                     # HAND: the ailerons' adverse yaw (arigo1_model's rule)
    if name == "boost":
        u = ap._unit("edf-70-p2.20-e0.65-catalogue-s1", FAN_W, None)
        C["boost"] = {"table_V": [float(v) for v in range(0, 92, 2)], "table_T": [float(max(u.full(float(v))[0], 0.0)) for v in range(0, 92, 2)],
                      "power_W": FAN_W, "z_below_axis_m": 0.080}
    cfg = cfd_config(CFD_KEY.get(name, name), sweep)
    if cfg and sum(r.get("status") == "solved" for r in cfg.get("rows", [])) < 4:
        cfg = None
    C["lattice_lon"] = {k: float(C[k]) for k in ("CL0", "CLa", "Cm0", "Cma")}
    if name in ("twin", "goshawk"):
        cm = p["glove_root_chord"] * (1 + (p["glove_taper"] - 1) * p["motor_y"] / (p["glove_span"] / 2)) / 1000.0
        s_in = 2 * 0.5 * p["glove_root_chord"] * (1 + p["glove_taper"]) * (p["glove_span"] / 2 - p["fuselage_diameter"] / 2) / 1e6
        s_out = (p["panel_root_chord"] + p["panel_tip_chord"]) * p["panel_span"] * (p["panel_aileron_to"] - p["panel_aileron_from"]) / 1e6
        C["powered"] = {"prop_m": p["prop_diameter"] / 1000.0, "disc_m2": math.pi * (p["prop_diameter"] / 2000.0) ** 2, "chord_m": cm, "motor_y_m": p["motor_y"] / 1000.0,
                        # the crow brake: flaperons 20 deg up and inboard flaps 40 deg down as split flaps, 0.9 (cf/c)^1.38 (Sf/S) sin^2(d) each
                        "brake_cda_m2": 0.9 * p["panel_aileron_chord"] ** 1.38 * (s_out * math.sin(math.radians(20.0)) ** 2 + s_in * math.sin(math.radians(40.0)) ** 2),
                        "source": "HAND: momentum theory x 0.8 for the slipstream; a textbook flap drag for the brake"}
    if not cfg and name in ("twin", "goshawk"):                              # before its own CFD: HAWK1's, by area
        C1 = coefficients("twin1", sweep)
        if "cfd" in C1 and str(C1.get("aero_source", "")).startswith("cfd"):
            carry_hawk1(C, C1, name, (p["fuselage_diameter"] / 70.0) ** 2 * (0.9 if name == "goshawk" else BODY_LONGER))
            return C
    C_pre = {k: (v.copy() if isinstance(v, (dict, np.ndarray)) else v) for k, v in C.items()}
    if not cfg and name in ("t",) + TWINS:
        from_astor_v(C, p, sweep, glove)
        if name in TWINS:
            import gavilan_flight as GF
            C["CD0"] += (2 * 0.0008 + GF.NET_CDA - 0.0008) / C["S"]           # HAND: two nacelles; the net launcher (gavilan_flight) less its vessel, now inside the body
            C["Cnb"] *= 0.9                                                   # HAND: the two propellers' normal force ahead of the CG
    if cfg:
        ref = cfg.get("reference", {})
        cam_cfd, trav_cfd = ref.get("camber", 0.0), ref.get("carriage_travel_mm", V.RUN1_TRAVEL.get(sweep, 0.0))
        same = abs(cam_cfd - p["panel_camber"]) < 1e-4 and abs(trav_cfd - V.travel(sweep, p)) < 1.0
        if same:
            apply_cfd(C, cfg)
        else:
            # the CFD was flown with another camber or carriage travel: its drag, slopes and stall are kept, and the
            # lattice's difference between that geometry and this one is added to the lift at zero angle and the moment
            p_old = {**p, "panel_camber": cam_cfd, "pivot_slides": False, "pivot_x": p["pivot_x"] - trav_cfd}
            now, was = _lon(p, sweep, tf, kL, kM, xc), _lon(p_old, sweep, tf, kL, kM, xc)
            apply_cfd(C, cfg)
            for key in ("CL0", "Cm0", "Cma"):
                C[key] += now[key] - was[key]
            if p["panel_camber"] > cam_cfd:
                C["CL_MAX"] += 6.0 * (p["panel_camber"] - cam_cfd) * VM.S_REF / Se           # HAND: +0.06 per 1 % of camber (+-0.03)
            C["alpha_stall"] = (C["CL_MAX"] - C["CL0"]) / C["CLa"]
            C["static_margin"] = -C["Cma"] / C["CLa"]
            C["aero_source"] = (f"cfd run 1 (camber {cam_cfd:g}, carriage {trav_cfd:.0f} mm) for drag, slopes and stall + the lattice's "
                                f"difference to this geometry (camber {p['panel_camber']:g}, carriage {V.travel(sweep, p):.0f} mm): rerun the CFD")
    lift_table(C, name, sweep, cfg)
    reflex(C, cfg, cfd_config(CFD_REFLEX_KEY.get(name, "-"), sweep))
    dx = CG_AFT_MM.get(name, 0.0) / 1000.0 if cfg else 0.0

    def cg_aft(Cx):                                                    # the same aircraft balanced further aft: the moment gains the lift x the shift
        if dx:
            Cx["cfd"]["Cm"] = Cx["cfd"]["Cm"] + Cx["cfd"]["CL"] * dx / Cx["c"]
            Cx["Cm0"], Cx["Cma"] = Cx["Cm0"] + Cx["CL0"] * dx / Cx["c"], Cx["Cma"] + Cx["CLa"] * dx / Cx["c"]
            Cx["static_margin"], Cx["x_cg_m"] = -Cx["Cma"] / Cx["CLa"], V.CG_MM / 1000.0 + dx
    cg_aft(C)
    cfgf = cfd_config(CFD_FLAP_KEY.get(name, "-"), sweep)
    if cfg and cfgf and sum(r.get("status") == "solved" for r in cfgf.get("rows", [])) >= 4:
        Cf = C_pre                                                     # the same aircraft, its CFD with every flap down
        apply_cfd(Cf, cfgf)
        lift_table(Cf, name, sweep, cfgf)
        cg_aft(Cf)
        C["flap"] = {"deg": FLAP_DEG, "cfd": Cf["cfd"], "CL_MAX": Cf["CL_MAX"], "alpha_stall": Cf["alpha_stall"], "Cm0": Cf["Cm0"],
                     "alpha_margin_deg": 0.0,     # the CFD's lift with flaps is flat from its peak to 16 deg: no fall to keep away from
                     "source": "cfd (screening RANS) with the flaperons and inboard flaps down"}
    return C


# ASTOR-TWIN's CFD with the 12 % wing and the tube level gave a static margin of 23 % at the 365 mm CG the CFD turned
# about: more than it needs, and it costs elevator at high lift. The pack sits 20 mm further aft (CG 385 mm): ~13 %.
# HAWK 2 with its long tail: the neutral point is at ~456 mm (lattice + HAWK1's CFD correction). The CG is put at 435 mm
# (static margin ~7.5 %): the tail then carries ~9 % of the lift in trim at 8 deg (tail_share); at 385 mm it carried
# none (margin 25 %), at the neutral point 13 %. Less margin than this leans on the pitch damper and on carried-over data.
CG_AFT_MM = {"twin": 70.0, "twin1": 20.0, "goshawk": 300.0 - V.CG_MM}        # goshawk.CG_MM from the nose
CFD_REFLEX_KEY = {"twin": "twin2_reflex", "twin1": "twin_agile_reflex"}
REFLEX_DEG, REFLEX_FULL_BELOW_DEG, REFLEX_NONE_ABOVE_DEG = -6.0, 2.0, 4.0


def reflex(C, cfg, cfgr) -> None:
    """Flaps up, the flight controller reflexes every flap 6 deg UP at low angles of attack (fully below 2 deg, none
    above 4 deg, linear between): the 4 % cambered wing flies fast at about -2 deg, off its design lift, and the CFD of
    2026-11-23 gave it ~25 % more drag there than the thin wing; reflexed, 15 % of that goes (25.9 N against 30.3 N at
    60 m/s). The clean table takes the CFD's differences (reflexed less clean, at the angles both were run) times that
    schedule, so every user of the table flies the scheduled wing."""
    rows = {float(r["alpha_deg"]): r for r in (cfgr or {}).get("rows", []) if r.get("status") == "solved"}
    base = {float(r["alpha_deg"]): r for r in (cfg or {}).get("rows", []) if r.get("status") == "solved"}
    al = sorted(set(rows) & set(base))
    if len(al) < 3 or "cfd" not in C:
        return
    ref = cfg["reference"]
    kL, kM = ref["S"] / C["S"], ref["S"] * ref["c"] / (C["S"] * C["c"])
    A = C["cfd"]["alpha_deg"]
    f = np.clip((REFLEX_NONE_ABOVE_DEG - A) / (REFLEX_NONE_ABOVE_DEG - REFLEX_FULL_BELOW_DEG), 0.0, 1.0)
    d = lambda key, k: np.interp(A, al, [(rows[a][key] - base[a][key]) * k for a in al])
    C["cfd"] = {"alpha_deg": A, "CL": C["cfd"]["CL"] + f * d("Cl", kL), "CD": C["cfd"]["CD"] + f * d("Cd", kL), "Cm": C["cfd"]["Cm"] + f * d("Cm_cg", kM)}
    C["reflex"] = {"deg": REFLEX_DEG, "full_below_deg": REFLEX_FULL_BELOW_DEG, "none_above_deg": REFLEX_NONE_ABOVE_DEG, "source": "cfd (screening RANS) at " + ", ".join(f"{a:g}" for a in al) + " deg"}


# the CFD configurations a design flies (astor_v_cfd.DESIGNS): flaps up, and every flap down. "twin" is HAWK 2 (the
# doubled wing: twin2...); "twin1" the aircraft before it, whose CFD HAWK 2 is carried over from until its own is run.
CFD_KEY = {"twin": "twin2", "twin1": "twin_agile", "goshawk": "goshawk"}
CFD_FLAP_KEY = {"twin": "twin2_flap", "twin1": "twin_agile_flap", "goshawk": "goshawk_flap"}
BODY_LONGER = 1.06                           # HAND: HAWK 2's body is 150 mm longer (its wetted area + 19 %; a third of the body's drag is friction)


def carry_hawk1(C, C1, name="twin", body_scale=BODY_LONGER) -> None:
    """HAWK 2 before its own CFD. The wing is HAWK1's scaled 1.414 x in every length (the same section, flaps, sweep
    and aspect ratio), so at one angle of attack the wing's forces are HAWK1's times the area, and the coefficients
    on each aircraft's own area are the same; the body's drag is the same AREA (x BODY_LONGER) and so half the
    coefficient. The lift slope and the moments are this aircraft's own lattice (its tail, its arm) plus what
    HAWK1's CFD added to HAWK1's lattice. Marked; ``sims/astor.sh --cfd-only`` replaces it."""
    S1, S = C1["S"], C["S"]
    L1, L = C1["lattice_lon"], C["lattice_lon"]
    r = L["CLa"] / L1["CLa"]
    cm0 = L["Cm0"] + (C1["Cm0"] - L1["Cm0"])
    cma = L["Cma"] + (C1["Cma"] - L1["Cma"])
    A_ = C1["cfd"]["alpha_deg"]
    ar = np.radians(A_)
    cd_min = float(np.min(C1["cfd"]["CD"]))
    body1 = cd_min * S1 - WING_CD * 1.3 * S1                           # the body's drag area: the least drag less the wing's profile drag
    dcd = -body1 / S1 + body1 * body_scale / S                         # the body's drag area by its frontal area (and its length: BODY_LONGER)
    asp1, asp = C1["b"] ** 2 / S1, C["b"] ** 2 / S
    k_new = C1["k"] * asp1 / asp                                       # the drag due to lift by the aspect ratio
    dk = k_new - C1["k"]

    def table(t):
        return {"alpha_deg": A_, "CL": t["CL"] * r, "CD": t["CD"] + dcd + dk * (t["CL"] * r) ** 2, "Cm": t["Cm"] + (cm0 - C1["Cm0"]) + (cma - C1["Cma"]) * ar}
    # HAWK 2's CG is further aft than HAWK1's (its long tail moves the neutral point aft: CG_AFT_MM): the moment gains the lift x the shift
    dx = (CG_AFT_MM[name] - CG_AFT_MM["twin1"]) / 1000.0 / C["c"]
    cm0, cma = cm0 + C1["CL0"] * r * dx, cma + C1["CLa"] * r * dx            # ``table`` below takes them: the shift is in its moment once
    C.update({"cfd": table(C1["cfd"]), "CL0": C1["CL0"] * r, "CLa": C1["CLa"] * r, "Cm0": cm0, "Cma": cma, "static_margin": -cma / (C1["CLa"] * r),
              "CL_MAX": C1["CL_MAX"] * r, "alpha_stall": C1["alpha_stall"], "CD0": C1["CD0"] + dcd, "k_model": C["k"], "k": k_new,
              "x_cg_m": V.CG_MM / 1000.0 + CG_AFT_MM[name] / 1000.0, "lift_curve_source": "HAWK1's CFD table, by area",
              "aero_source": ("carried over from HAWK1's CFD by area (HAND): the same section, flaps and aspect ratio at 1.414 x; no CFD of its own yet" if name == "twin" else
                              "carried over from HAWK1's CFD by area, aspect ratio and body size (HAND; at a third of its Reynolds number): no CFD of its own yet")})
    if "flap" in C1:
        F = C1["flap"]
        C["flap"] = {**F, "cfd": table(F["cfd"]), "CL_MAX": F["CL_MAX"] * r, "source": "HAWK1's CFD with every flap down, by area (HAND)"}
    if "reflex" in C1:
        C["reflex"] = {**C1["reflex"], "source": "HAWK1's CFD, by area (HAND)"}
FLAP_DEG, FLAP_TRAVEL_S = 20.0, 1.0


def with_flap(C, f) -> dict:
    """The set with the flaps a share ``f`` (0-1) of the way down: the two CFD tables blended, and the limiter's angle."""
    if f <= 1e-3 or "flap" not in C:
        return C
    f, F = min(f, 1.0), C["flap"]
    out = dict(C)
    out["cfd"] = {k: (C["cfd"][k] if k == "alpha_deg" else (1 - f) * C["cfd"][k] + f * F["cfd"][k]) for k in C["cfd"]}
    out["CL_MAX"] = (1 - f) * C["CL_MAX"] + f * F["CL_MAX"]
    out["alpha_stall"] = (1 - f) * C["alpha_stall"] + f * F["alpha_stall"]
    out["alpha_margin_deg"] = (1 - f) * ALPHA_MARGIN_DEG + f * F["alpha_margin_deg"]
    out["flap_share"] = f
    return out


TABLE_ALPHA = np.arange(-8.0, 25.0, 1.0)     # the common grid [deg] of every state's table (they are blended between sweeps)


def lift_table(C, name, sweep, cfg) -> None:
    """The lift, drag and moment against the angle of attack as a table (``arigo1_model`` flies a table when the set has
    one), in place of the hand stall: linear to a nominal stall angle, then a fall onto the flat plate within 2 deg.
    That fall began 2 deg BEFORE the stall angle, where the angle-of-attack limiter also sits: ASTOR could not fly
    below ~1.3 x its stall speed in the simulator, though its CFD lift keeps rising gently to 16 deg (2026-11-22).

    - a design with its own CFD at this sweep: the CFD's points;
    - ASTOR-T before its CFD: its own linear lift and moment, bent as ASTOR-V's CFD bends from ASTOR-V's linear fit;
    - beyond the CFD's last angle, and for a sweep whose CFD stopped early (60 deg: 6 deg): the linear lift capped
      smoothly at the maximum lift (HAND), then a blend onto the flat plate over 8 deg.
    The stall angle for the limiter and the warnings: where the lift first reaches 95 % of its maximum."""
    A = TABLE_ALPHA
    ar = np.radians(A)
    lin_cl = C["CL0"] + C["CLa"] * ar
    lin_cm = C["Cm0"] + C["Cma"] * ar
    cap = lambda cl: np.where(cl > 0, C["CL_MAX"] * np.tanh(cl / C["CL_MAX"]), -0.8 * C["CL_MAX"] * np.tanh(-cl / (0.8 * C["CL_MAX"])))
    CL, CM, CDx, a_last, src = cap(lin_cl), lin_cm.copy(), np.zeros_like(A), None, "linear lift capped at the maximum lift (HAND)"
    rows = [r for r in (cfg or {}).get("rows", []) if r.get("status") == "solved" and r.get("Cl") is not None]
    if rows and str(C.get("aero_source", "")).startswith("cfd corrections") and max(r["alpha_deg"] for r in rows) >= 10.0:
        ref = cfg["reference"]
        kL, kM = ref["S"] / C["S"], ref["S"] * ref["c"] / (C["S"] * C["c"])
        a = np.array([r["alpha_deg"] for r in rows])
        a_last = float(a.max())
        m = (A >= a.min()) & (A <= a_last)
        CL[m] = np.interp(A[m], a, np.array([r["Cl"] for r in rows]) * kL)
        CM[m] = np.interp(A[m], a, np.array([r["Cm_cg"] for r in rows]) * kM)
        # the CFD's own drag at its angles. (Until 2026-11-24 only its EXCESS over CD0 + k CL^2 was kept, with CD0 the lowest
        # drag sampled: for a cambered wing that minimum is at a lift of ~0.3, and the table then counted that lift's drag
        # twice: ASTOR-TWIN's 4 % wing was flown with ~20 % too much drag at every speed.)
        CDx[m] = np.interp(A[m], a, np.array([r["Cd"] for r in rows]) * kL) - (C["CD0"] + C["k"] * CL[m] ** 2)
        src = "the CFD's points"
    elif name in ("t",) + TWINS:
        Cv = coefficients("v", sweep)
        tv = Cv.get("lift_table_bend")
        if tv is not None:
            a_last = tv["a_last"]
            m = (A >= tv["a_first"]) & (A <= a_last)
            CL[m] = lin_cl[m] * tv["cl_ratio"][m]
            # the same section stalls at the same lift per wing area: the peak brought to the maximum lift carried over
            # (ASTOR-T's higher aspect ratio steepens its lift slope, not its maximum), from 4 deg up
            peak = float(CL[m].max())
            if peak > C["CL_MAX"]:
                a_pk = float(A[m][int(np.argmax(CL[m]))])
                g = 1.0 + (C["CL_MAX"] / peak - 1.0) * np.clip((A[m] - 4.0) / max(a_pk - 4.0, 1.0), 0.0, 1.0)
                CL[m] = CL[m] * g
            CM[m] = lin_cm[m] + tv["cm_excess"][m] * (Cv["S"] * Cv["c"]) / (C["S"] * C["c"]) * C["S"] / Cv["S"]
            CDx[m] = tv["cd_excess"][m]
            src = "its own linear lift and moment, bent as ASTOR-V's CFD bends (HAND carry-over)"
    if a_last is not None:                                             # past the last CFD angle: on toward the flat plate over 8 deg
        k = int(np.argmin(np.abs(A - a_last)))
        for i in range(k + 1, len(A)):
            w = min(1.0, (A[i] - a_last) / 8.0)
            CL[i] = (1 - w) * CL[k] + w * 0.9 * 2 * math.sin(ar[i]) * math.cos(ar[i])
            CM[i] = CM[k] + (lin_cm[i] - lin_cm[k])
            CDx[i] = CDx[k]
    else:
        past = lin_cl > 1.6 * C["CL_MAX"]                              # far past the cap: toward the flat plate
        CL[past] = np.minimum(CL[past], np.maximum(0.9 * 2 * np.sin(ar[past]) * np.cos(ar[past]), 0.6 * C["CL_MAX"]))
    fp_cd = 2 * np.sin(ar) ** 2 * 0.9
    CD = np.maximum(C["CD0"] + C["k"] * np.minimum(CL, C["CL_MAX"]) ** 2 + CDx, np.where(np.abs(A) > 14.0, fp_cd * 0.8, 0.0))
    if src == "the CFD's points":                                      # what ASTOR-T carries over: how the CFD bends from the linear fit
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(np.abs(lin_cl) > 0.15, CL / lin_cl, 1.0)
        C["lift_table_bend"] = {"a_first": float(a.min()), "a_last": a_last, "cl_ratio": np.clip(ratio, 0.3, 1.3), "cm_excess": CM - lin_cm, "cd_excess": CDx}
    C["cfd"] = {"alpha_deg": A, "CL": CL, "CD": CD, "Cm": CM}
    pos = A >= 0
    cl_max = float(CL[pos].max())
    i95 = int(np.argmax(CL[pos] >= 0.95 * cl_max))
    C["CL_MAX"], C["alpha_stall"] = cl_max, float(math.radians(A[pos][i95]))
    C["lift_curve_source"] = src


def from_astor_v(C, p, sweep, extra_wing_m2=0.0) -> None:
    """ASTOR-T before its own CFD: the drag and the maximum lift from ASTOR-V's CFD at this sweep (the same body, tail
    cone, cameras and section), carried over by area — the body's drag area kept, the wing's part by the exposed wing
    area, the drag due to lift by the aspect ratio, the maximum lift per exposed wing area. The lift slope, the balance
    and the controls stay the lattice's on ASTOR-T's own panels. Marked; its own CFD replaces this."""
    Cv = coefficients("v", sweep)
    if not str(Cv.get("aero_source", "")).startswith("cfd"):
        return
    pv, _ = design("v")
    wv, wt = V.wing_numbers(pv, sweep), dict(V.wing_numbers(p, sweep))
    wt["exposed_area_m2"] += extra_wing_m2                                    # ASTOR-TWIN's fixed inner wing
    cda_v = Cv["CD0"] * Cv["S"]
    cda_t = cda_v - WING_CD * (wv["exposed_area_m2"] - wt["exposed_area_m2"]) * 1.3      # x1.3: the CFD's wing drag over the hand value
    ar_v, ar_t = Cv["b"] ** 2 / Cv["S"], C["b"] ** 2 / C["S"]
    C["CD0"] = cda_t / C["S"]
    C["k_model"], C["k"] = C["k"], Cv["k"] * ar_v / ar_t
    C["CL_MAX"] = Cv["CL_MAX"] * (Cv["S"] / wv["exposed_area_m2"]) * (wt["exposed_area_m2"] / C["S"])
    C["alpha_stall"] = (C["CL_MAX"] - C["CL0"]) / C["CLa"]
    C["aero_source"] = "lattice on ASTOR-T's panels; drag and maximum lift carried over from ASTOR-V's CFD by area and aspect ratio (HAND): no CFD of its own yet"


def _lon(p, sweep, tf, kL, kM, xc) -> dict:
    """The lattice's lift at zero angle and moment (value and slope) about the CG for a geometry, in C's reference."""
    r0, r4 = VM.lattice(sweep, tail_fan=tf, p=p), VM.lattice(sweep, tail_fan=tf, alpha=4.0, p=p)
    cm = lambda r: r["Cm"] + r["CL"] * xc
    return {"CL0": r0["CL"] * kL, "Cm0": cm(r0) * kM, "Cma": (cm(r4) - cm(r0)) / math.radians(4.0) * kM}


def apply_cfd(C, cfg) -> None:
    """Correct the model with a CFD sweep (rows of alpha, Cl, Cd, Cm about the CG on the CFD's reference): zero-lift
    drag, maximum lift and its angle, lift slope, neutral point."""
    rows = [r for r in cfg.get("rows", []) if r.get("status") == "solved" and r.get("Cl") is not None]
    if len(rows) < 4:
        return
    ref = cfg["reference"]
    kL, kM = ref["S"] / C["S"], ref["S"] * ref["c"] / (C["S"] * C["c"])
    al = np.radians([r["alpha_deg"] for r in rows])
    cl, cd, cmv = (np.array([r[k] for r in rows]) for k in ("Cl", "Cd", "Cm_cg"))
    lin = np.abs(np.degrees(al)) <= 6.0
    if lin.sum() >= 2:
        a1, a0 = np.polyfit(al[lin], cl[lin] * kL, 1)
        m1, m0 = np.polyfit(al[lin], cmv[lin] * kM, 1)
        C.update({"CLa": float(a1), "CL0": float(a0), "Cma": float(m1), "Cm0": float(m0), "static_margin": float(-m1 / a1)})
    C["CD0"] = float(cd.min() * kL)
    pol = np.abs(np.degrees(al)) <= 9.0                               # the drag due to lift: CD = CD0 + k CL^2, fitted below the stall
    if pol.sum() >= 3:
        x, y = (cl[pol] * kL) ** 2, cd[pol] * kL - C["CD0"]
        kfit = float((x * y).sum() / max((x * x).sum(), 1e-12))
        if 0.02 < kfit < 0.6:
            C["k_model"], C["k"] = C["k"], kfit
    # the maximum lift: the CFD's only if its sweep reached it (the lift peaked, or its slope at the last angle has
    # fallen below 40 % of the linear one); a sweep that stops short (the dash state's ends at 6 deg) leaves the
    # model's value, never below what the CFD did reach
    k = int(np.argmax(cl))
    end_slope = (cl[-1] - cl[-2]) / max(al[-1] - al[-2], 1e-6) * kL if len(cl) >= 2 else 0.0
    if k < len(cl) - 1 or end_slope < 0.4 * C["CLa"]:
        C["CL_MAX"] = float(cl.max() * kL)
        C["alpha_stall"] = float(al[k])
        C["CL_MAX_source"] = "cfd"
    else:
        C["CL_MAX"] = max(C["CL_MAX"], float(cl.max() * kL))
        C["alpha_stall"] = (C["CL_MAX"] - C["CL0"]) / C["CLa"]
        C["CL_MAX_source"] = "model (the CFD sweep stopped before the stall)"
    C["aero_source"] = f"cfd corrections (screening RANS, {len(rows)} angles) on the lattice + hand model"


ALPHA_MARGIN_DEG = 2.0                      # the limiter holds the wing this far below its stall
LIMIT_K, LIMIT_KQ = 6.0, 0.10               # deg of elevator per deg over the limit, and per deg/s of pitch rate (HAND, tuned in the stall test)


# STABILITY AUGMENTATION (ASTOR-TWIN, 2026-11-23; in the flight controller, under every command). The stability run on
# its CFD gave a Dutch roll damped 0.05-0.06, a short period damped 0.20 at 30 m/s and 7.0 g in a 60 m/s doublet.
SAS = {"twin": {"yaw_k": 0.25, "yaw_washout_s": 1.0,     # rudder [deg] per deg/s of yaw rate, the steady turn's rate washed out
                "pitch_k": 0.06, "pitch_v_ref": 30.0,    # elevator [deg] per deg/s of pitch rate at 30 m/s, less with speed (x 30 / V)
                "g_max": 8.0},
       "goshawk": {"yaw_k": 0.25, "yaw_washout_s": 1.0, "pitch_k": 0.06, "pitch_v_ref": 12.0, "g_max": 6.0,
                   "alpha_k": 0.0, "alpha_washout_s": 3.0}}        # the angle-of-attack augmentation OFF: balanced stable (5 %) it is not needed, and in
                                                                   # the forest flights it held the nose down for seconds when a turn asked for load factor      # PROTOHAWK's gains at GOSHAWK's speeds: NOT tuned                           # the client's choice (2026-11-23); the airframe must be built to it (astor_twin's budget: +100 g, HAND)


def alpha_for_g(C, V, n) -> float:
    """The angle of attack [rad] that gives ``n`` g at this airspeed (the table's lift, or the linear one)."""
    cl = n * C["mass"] * am.G / (0.5 * am.RHO * max(V, 1.0) ** 2 * C["S"])
    if "cfd" in C:
        A, CLt = np.asarray(C["cfd"]["alpha_deg"]), np.asarray(C["cfd"]["CL"])
        k = int(np.argmax(CLt))
        return math.radians(float(np.interp(cl, CLt[:k + 1], A[:k + 1]))) if cl < CLt[k] else math.radians(float(A[k]))
    return (cl - C["CL0"]) / C["CLa"]


def augment(sim, C, s, ctl) -> list:
    """The dampers and the g limit on the controls [elevator, aileron, rudder, throttle] (deg): for the designs in SAS."""
    k = SAS.get(C.get("design"))
    if not k or not getattr(sim, "sas", True):
        return ctl
    p, q, r = (math.degrees(x) for x in s["omega_frd"])
    V = max(float(np.linalg.norm(s["v_air_frd"])), 10.0)
    st = sim.__dict__.setdefault("_sas", {"r_lp": r})
    st["r_lp"] += (r - st["r_lp"]) * amj.DT / k["yaw_washout_s"]
    ctl[2] = max(-am.DR_MAX, min(am.DR_MAX, ctl[2] + k["yaw_k"] * (r - st["r_lp"]) * (1 if C["Cndr"] < 0 else -1)))
    ctl[0] = max(-am.DE_MAX, min(am.DE_MAX, ctl[0] + k["pitch_k"] * k["pitch_v_ref"] / V * q))
    if k.get("alpha_k"):
        # relaxed static stability (GOSHAWK): the airframe is balanced near neutral so that its tail lifts; the elevator
        # answers the angle of attack's departure from its own slow average, which is the stiffness a forward CG would give
        al = math.degrees(math.atan2(s["v_air_frd"][2], max(s["v_air_frd"][0], 1.0)))
        st.setdefault("a_lp", al)
        st["a_lp"] += (al - st["a_lp"]) * amj.DT / k["alpha_washout_s"]
        ctl[0] = max(-am.DE_MAX, min(am.DE_MAX, ctl[0] + k["alpha_k"] * (al - st["a_lp"])))
    return ctl


# WHAT THE TWO WING PROPELLERS ADD (HAWK 2, 2026-11-24; all HAND until measured, and marked in ``C["powered"]``):
# - SLIPSTREAM LIFT: the wing behind each propeller sees the slipstream's dynamic pressure, not the flight's. Momentum
#   theory gives the jet's speed and its contracted width; the wing strip in it makes the table's lift coefficient at
#   that pressure, x SLIP_EFF (the jet is finite and turns with the wing: 0.8 +-0.15).
# - DIFFERENTIAL THRUST for yaw: the rudder command also moves the two throttles apart by up to DIFF_THRUST of the
#   thrust available (the fin's rudder fades with airspeed; this does not).
# - the CROW BRAKE (``sim.brake`` 0-1: flaperons up, inboard flaps down): a drag area, a textbook split-flap value until
#   the CFD's twin2_crow state.
SLIP_EFF, DIFF_THRUST = 0.8, 0.15


def powered(sim, C, s, ctl, F, M) -> None:
    """Add the slipstream's lift, the differential thrust's yawing moment and the crow brake's drag to the body forces
    F and moments M (x forward, y right, z down), for a design with ``C["powered"]``."""
    pw = C.get("powered")
    if not pw:
        return
    u, v, w = s["v_air_frd"]
    V = max(math.sqrt(u * u + v * v + w * w), 1.0)
    al = math.atan2(w, max(u, 1.0))
    T = am.max_thrust(C, max(u, 0.0)) * max(min(ctl[3], 1.0), 0.0)
    if T > 0.0 and "cfd" in C:
        vj = math.sqrt(V * V + 2 * (T / 2) / (am.RHO * pw["disc_m2"]))
        width = pw["prop_m"] * math.sqrt((V + vj) / (2 * vj))
        cl = float(np.interp(math.degrees(al), C["cfd"]["alpha_deg"], C["cfd"]["CL"]))
        dL = SLIP_EFF * 0.5 * am.RHO * (vj * vj - V * V) * 2 * width * pw["chord_m"] * cl
        # no more than the jets' own momentum turned through the wing's angle and its flaps' (a 20 deg flap turns the flow ~12 deg)
        turn = min(max(al + math.radians(12.0) * float(C.get("flap_share", 0.0)), 0.0), math.radians(40.0))
        dL = min(dL, T / max(vj - V, 0.5) * vj * math.sin(turn))
        F[2] -= dL * math.cos(al)
        F[0] += dL * math.sin(al)
    M[2] += (-1.0 if C["Cndr"] < 0 else 1.0) * (ctl[2] / am.DR_MAX) * 2 * DIFF_THRUST * am.max_thrust(C, max(u, 0.0)) * pw["motor_y_m"]
    b = float(getattr(sim, "brake", 0.0))
    if b > 0.0:
        D = 0.5 * am.RHO * V * V * pw["brake_cda_m2"] * b
        F[0] -= D * u / V
        F[1] -= D * v / V
        F[2] -= D * w / V


def limit_elevator(C, v_air_frd, q, de) -> float:
    """The angle-of-attack limiter (in the flight controller, under every elevator command, the pilot's included):
    the elevator may not go further nose-up than what holds the limit angle, less a term for the angle beyond it and
    the pitch rate toward it. Positive elevator is nose down."""
    al = math.atan2(v_air_frd[2], max(v_air_frd[0], 1.0))
    a_lim = C["alpha_stall"] - math.radians(C.get("alpha_margin_deg", ALPHA_MARGIN_DEG))
    g_max = SAS.get(C.get("design"), {}).get("g_max")
    if g_max:                                                          # the g limit: the same floor, at the angle that gives g_max
        a_lim = min(a_lim, alpha_for_g(C, float(np.linalg.norm(v_air_frd)), g_max))
    cm_lim = am._cfd_static(C["cfd"], a_lim)[2] if "cfd" in C else C["Cm0"] + C["Cma"] * a_lim
    de_hold = -cm_lim / C["Cmde"]
    floor = de_hold + LIMIT_K * math.degrees(al - a_lim) + LIMIT_KQ * math.degrees(q)
    return max(de, min(floor, am.DE_MAX))


class Fixed(amj.Sim):
    """The aircraft with the wing held at one sweep (the stability tests), the limiter under the elevator."""

    def __init__(self, name, sweep, mesh=None, bandit=False, alpha_limit=True, flap=0.0):
        import mujoco
        self.C = with_flap(get(name, sweep), flap)                       # ``flap`` 0-1: held there for the whole test
        self.m = mujoco.MjModel.from_xml_string(amj.mjcf(self.C, mesh, bandit))
        self.d = mujoco.MjData(self.m)
        self.bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "arigo")
        self.controls = np.array([0.0, 0.0, 0.0, 0.5])
        self.wind = np.zeros(3)
        self.info = {}
        self.alpha_limit = alpha_limit

    def step(self, n=1):
        import mujoco
        for _ in range(n):
            s = self.state()
            ctl = augment(self, self.C, s, list(self.controls))
            if self.alpha_limit:
                ctl[0] = limit_elevator(self.C, s["v_air_frd"], s["omega_frd"][1], ctl[0])
            F, M, info = am.aero(self.C, s["v_air_frd"], s["omega_frd"], *ctl)
            powered(self, self.C, s, ctl, F, M)
            R = self.R()
            self.d.xfrc_applied[self.bid, :3] = R @ (F * amj.FLIP)
            self.d.xfrc_applied[self.bid, 3:] = R @ (M * amj.FLIP)
            self.info = {**info, "load_factor": float(-F[2] / (self.C["mass"] * am.G)), "limited": bool(ctl[0] != self.controls[0])}
            mujoco.mj_step(self.m, self.d)


def get(name="twin", sweep=12.0) -> dict:
    return dict(coefficients(name, float(sweep)))


def blended(name: str, sweep: float) -> dict:
    """The coefficients at any sweep: the built states, linearly between."""
    s = max(SWEEPS[0], min(SWEEPS[-1], sweep))
    k = next(i for i in range(len(SWEEPS) - 1) if s <= SWEEPS[i + 1])
    lo, hi = get(name, SWEEPS[k]), get(name, SWEEPS[k + 1])
    f = (s - SWEEPS[k]) / (SWEEPS[k + 1] - SWEEPS[k])
    C = dict(lo)
    for key, v in lo.items():
        if isinstance(v, float) and isinstance(hi.get(key), float):
            C[key] = v + f * (hi[key] - v)
    C["inertia"] = tuple(a + f * (b - a) for a, b in zip(lo["inertia"], hi["inertia"]))
    if "cfd" in lo and "cfd" in hi:
        C["cfd"] = {k: (lo["cfd"][k] if k == "alpha_deg" else lo["cfd"][k] + f * (hi["cfd"][k] - lo["cfd"][k])) for k in lo["cfd"]}
    C["sweep_deg"] = s
    return C


def tail_share(name="twin", alpha_deg=8.0) -> dict:
    """How much of the lift the tail carries, in trim at this angle of attack with the CG where the flight model has it
    (the lattice: the lift with the tail set for zero moment, less the wing's alone)."""
    p, _ = design(name)
    xc = (V.CG_MM + CG_AFT_MM.get(name, 0.0)) / 1000.0 / VM.C_REF
    cm = lambda r: r["Cm"] + r["CL"] * xc
    a, b = VM.lattice(12.0, alpha=alpha_deg, p=p), VM.lattice(12.0, alpha=alpha_deg, tail_inc=1.0, p=p)
    inc = -cm(a) / (cm(b) - cm(a))                                     # the tail angle for zero moment [deg]
    total = a["CL"] + inc * (b["CL"] - a["CL"])
    wing = VM.lattice(12.0, alpha=alpha_deg, p={**p, "tail_panel_span": 1.0, "tail_panel_root": 2.0, "tail_panel_tip": 1.0})["CL"]
    return {"alpha_deg": alpha_deg, "tail_angle_for_trim_deg": round(inc, 2), "tail_share_of_lift": round((total - wing) / total, 3), "note": "lattice: no flaps, no CFD moment correction"}


def stall(C) -> float:
    return math.sqrt(C["mass"] * am.G / (0.5 * am.RHO * C["S"] * C["CL_MAX"]))


def sweeps(name) -> tuple:
    """The wing states this design flies: one for a fixed wing."""
    return SWEEPS[:1] if name in FIXED_WING else SWEEPS


def schedule(V_air: float) -> float:
    """The sweep the wing takes by airspeed when nothing else commands it: spread below 28 m/s, 60 deg above 50."""
    return SWEEPS[0] + (SWEEPS[-1] - SWEEPS[0]) * min(1.0, max(0.0, (V_air - 28.0) / 22.0))


class SimV(amj.Sim):
    """ASTOR-V / ASTOR-BOOST in MuJoCo. ``sweep_cmd`` (deg, or None: by airspeed) is followed at SWEEP_RATE;
    ``prop_on`` False stops and folds the propeller; ``boost`` 0-1 is ASTOR-BOOST's fan; dropping nets takes their
    mass off (``drop_net``)."""

    SWEEP_RATE = 60.0                        # deg/s (HAND: the carriage and the pivots' actuators)

    def __init__(self, name="twin", mesh=None, bandit=False, sweep=None):
        import mujoco
        self.name = name
        self.sweep = SWEEPS[0] if sweep is None or name in FIXED_WING else sweep
        self.sweep_cmd = sweep
        self.C = blended(name, self.sweep)
        self.m = mujoco.MjModel.from_xml_string(amj.mjcf(self.C, mesh, bandit))
        self.d = mujoco.MjData(self.m)
        self.bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "arigo")
        self.controls = np.array([0.0, 0.0, 0.0, 0.5])
        self.wind = np.zeros(3)
        self.info = {}
        self.prop_on, self.boost, self.nets = True, 0.0, int(self.C["equipment"]["nets"])
        self.alpha_limit = True
        self.flap, self.flap_cmd = 0.0, 0.0                              # the flaps' position and command, 0 (up) to 1 (20 deg down)
        self.brake = 0.0                                                 # the crow brake, 0-1
        self.energy_J = 0.0

    def drop_net(self) -> bool:
        if self.nets <= 0:
            return False
        self.nets -= 1
        self.m.body_mass[self.bid] -= V.NET_G / 1000.0
        return True

    def step(self, n=1):
        import mujoco
        for _ in range(n):
            s = self.state()
            want = SWEEPS[0] if self.name in FIXED_WING else schedule(s["V"]) if self.sweep_cmd is None else self.sweep_cmd
            d = self.SWEEP_RATE * amj.DT
            self.sweep += max(-d, min(d, want - self.sweep))
            trim = getattr(self, "trim", None)
            self.flap += max(-amj.DT / FLAP_TRAVEL_S, min(amj.DT / FLAP_TRAVEL_S, self.flap_cmd - self.flap))
            self.C = with_flap(blended(self.name, self.sweep), self.flap)
            self.C["mass"] = float(self.m.body_mass[self.bid])
            ctl = augment(self, self.C, s, list(self.controls))
            if self.alpha_limit:
                ctl[0] = limit_elevator(self.C, s["v_air_frd"], s["omega_frd"][1], ctl[0])
            if not self.prop_on:
                ctl[3] = 0.0
                self.C["CD0"] += FOLDED_PROP_CDA / self.C["S"]
            F, Mo, info = am.aero(self.C, s["v_air_frd"], s["omega_frd"], *ctl)
            powered(self, self.C, s, ctl, F, Mo)
            P = ctl[3] * self.C["thrust"]["propulsor"]["power_w_per_unit"]
            if self.boost > 0.0 and "boost" in self.C:
                bt = self.C["boost"]
                T = self.boost * float(np.interp(max(s["v_air_frd"][0], 0.0), bt["table_V"], bt["table_T"]))
                F[0] += T
                Mo[1] += T * bt["z_below_axis_m"]                       # the thrust line below the CG: nose up
                P += self.boost * bt["power_W"]
            self.energy_J += (P + 15.0) * amj.DT
            R = self.R()
            self.d.xfrc_applied[self.bid, :3] = R @ (F * amj.FLIP)
            self.d.xfrc_applied[self.bid, 3:] = R @ (Mo * amj.FLIP)
            self.info = {**info, "load_factor": float(-F[2] / (self.C["mass"] * am.G)), "sweep_deg": self.sweep}
            mujoco.mj_step(self.m, self.d)
            if trim is not None:
                self.trim = trim
