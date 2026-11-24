"""ASTOR-X as an aeroplane: the coefficient set ``astor_v_flight`` flies (name ``"x"``), for the stability runs.

ASTOR-X (``designs/astor_x``) is ASTOR-TWIN's body and tail with a one-piece swept wing of 1.2 m span and four pusher
motors on a short X. This builds its set the way ``astor_v_flight.coefficients`` builds the others:

- lateral derivatives, damping and inertias: ``arigo1_model.build`` on the equivalent conventional aircraft (the real
  wing IS a plain swept one here: span, chords and sweep as drawn);
- lift, moment, elevator and aileron power: the vortex lattice on the drawn wing and tail (``lattice`` below, with
  ``astor_v_model``'s strips, camber incidence and aileron effectiveness);
- drag and maximum lift BEFORE the CFD (HAND): ASTOR-TWIN's zero-lift drag area plus the struts'; the induced factor
  from the aspect ratio; ASTOR-TWIN's maximum lift, flaps up;
- its own CFD when installed (``astor_v_cfd``: "x" flaps up, "x_flap" every flap down): corrections and tables by
  ``astor_v_flight.apply_cfd`` / ``lift_table``, the moments brought from the CFD's centre to this CG;
- thrust: ASTOR-TWIN's table (the same power on the same total disc area: HAND);
- the CENTRE OF GRAVITY is put ``MARGIN`` of the mean chord ahead of the lattice's neutral point (``CG``): the drawing
  was laid out on ASTOR-TWIN's 435 mm, which this corrects; the CFD's neutral point replaces the lattice's when there.

WHAT IT IS NOT: the hover and the flight beyond the stall on four rotors. This is the aeroplane, wing-borne, with the
controls an aeroplane has (all-moving tail, ailerons, rudder, plus the four rotors' differential thrust in yaw).
Nothing models the rotors holding the attitude at low speed; that needs its own model.
"""
from __future__ import annotations

import math

import numpy as np

import arigo1_model as am
import astor as A
import astor_v as V
import astor_v_model as VM
import astor_x as AX

MASS_KG = 8.55                     # ASTOR-TWIN's 8.25 with three nets + 0.30 for the struts, pods and two more controllers (astor_big: HAND)
MARGIN = 0.08                      # the static margin the CG is set for, on the mean chord
STRUT_CDA = 0.0019                 # four struts and pods [m2] (astor_big.astor_x_short: HAND)
TAU_FLAP = 0.60                    # a 30 % chord plain flap's effectiveness (thin aerofoil 0.66 x 0.9; HAND +-0.1)
SWEEP = 12.0                       # the "sweep" key astor_v_flight and the CFD file use for a fixed wing's one state


def params() -> dict:
    return AX.resolve()


def lattice(p=None, alpha=0.0, tail_inc=0.0, aileron=0.0, flap=0.0) -> dict:
    """The vortex lattice on the drawn wing (one swept panel a side, to the centre plane) and the tail, on
    ``astor_v_model``'s reference (S_REF, C_REF, moments about the nose). ``aileron`` [deg]: right trailing edge up;
    ``flap`` [deg]: the inboard flaps down."""
    from ogma.aero import vlm
    p = p or params()
    b, cam = p["sw_half_span"], p.get("panel_camber", 0.0)
    quads, inc = [], []
    for side in (1, -1):
        P = VM._outline(p["sw_root_chord"], p["sw_tip_chord"], b, p["sw_sweep_deg"], 0.0, 0.0, p["sw_le_x"], 0.0, side)
        q = VM._strips(P, side, p["wing_plane_z"], n=14)
        quads += q
        for qq in q:
            eta = abs(float(qq.mean(axis=0)[1])) * 1000.0 / b
            a_on = AX.WING["aileron"][0] <= eta <= AX.WING["aileron"][1]
            f_on = max(AX.WING["flap"][0], p["fuselage_diameter"] / 2 / b) <= eta <= AX.WING["flap"][1]
            inc.append(VM.ALPHA0_DEG_PER_CAMBER * cam - side * aileron * VM.TAU_AILERON * a_on + flap * TAU_FLAP * f_on)
    for side in (1, -1):
        P = VM._outline(p["tail_panel_root"], p["tail_panel_tip"], p["tail_panel_span"], 35.0, 0.25, 0.0, p["tail_pivot_x"], p["tail_pivot_y"], side)
        q = VM._strips(P, side, p.get("tail_z", 6.0), n=5)
        quads += q
        inc += [tail_inc] * len(q)
    r = dict(vlm(quads, alpha_deg=alpha, incidence_deg=inc, S_ref=VM.S_REF, c_ref=VM.C_REF, x_ref=0.0, n_span=2))
    if cam > 0:                                                        # the camber's couple (the lattice has only its lift)
        w = AX.planform(p)
        r["Cm"] = r["Cm"] + VM.CM_AC_PER_CAMBER * cam * w["area_m2"] * (w["mean_chord_mm"] / 1e3) / (VM.S_REF * VM.C_REF)
    return r


def balance(p=None) -> dict:
    """The lattice's neutral point and the CG put MARGIN of the mean chord ahead of it."""
    p = p or params()
    r0, r4 = lattice(p), lattice(p, alpha=4.0)
    x_np = -(r4["Cm"] - r0["Cm"]) / (r4["CL"] - r0["CL"]) * VM.C_REF * 1000.0
    mac = AX.planform(p)["mean_chord_mm"]
    return {"neutral_point_mm": round(x_np, 1), "mean_chord_mm": mac, "margin": MARGIN, "cg_mm": round(x_np - MARGIN * mac, 1),
            "wing_mean_quarter_chord_mm": AX.planform(p)["mean_quarter_chord_from_the_nose_mm"], "source": "vortex lattice on the drawn wing and tail (no body)"}


def _roll_per_deg(p) -> float:
    r0, r1 = lattice(p, alpha=2.0), lattice(p, alpha=2.0, aileron=1.0)
    m = lambda r: float(np.sum(np.asarray(r["F"])[:, 2] * np.asarray(r["mid"])[:, 1]))
    return abs(m(r1) - m(r0))


_CACHE: dict = {}


def coefficients() -> dict:
    if "C" in _CACHE:
        return _CACHE["C"]
    import astor_v_flight as VF
    p = params()
    w = AX.planform(p)
    bal = balance(p)
    cg = bal["cg_mm"]
    twin = VF.get("twin")
    lam = p["sw_tip_chord"] / p["sw_root_chord"]
    q = A.variant(span=2 * p["sw_half_span"], root_chord=p["sw_root_chord"], taper=lam, le_sweep_deg=p["sw_sweep_deg"], dihedral_deg=0.0,
                  wing_x=p["sw_le_x"], cg_x=cg, propulsion="tractor_lowcam",
                  fuselage_length=p["fuselage_length"], tail_cone=p["tail_cone"], tail_end_diameter=p["tail_end_diameter"],
                  htail_span=2 * (p["tail_pivot_y"] + p["tail_panel_span"] * math.cos(math.radians(35.0))), htail_root_chord=p["tail_panel_root"],
                  htail_taper=p["tail_panel_tip"] / p["tail_panel_root"], htail_sweep_deg=35.0, htail_z=p.get("tail_z", 6.0),
                  fin_height=p["fin_height"], fin_root_chord=p["fin_root_chord"], fin_taper=p["fin_taper"], fin_sweep_deg=p["fin_sweep_deg"],
                  lowcam_drop=p["lowcam_drop"])
    q["tail_le_x"] = p["tail_pivot_x"] - 0.25 * p["tail_panel_root"]
    pe = A.resolve(q)
    L = A.Astor.layout(pe)
    S0 = L["S_mm2"] / 1e6
    cda = twin["CD0"] * twin["S"] + STRUT_CDA
    thrust = {**twin["thrust"], "propulsor": {**twin["thrust"]["propulsor"], "units": 4, "power_w_per_unit": 2000.0, "layout": "pusher",
                                              "entry": "four 9.2 in pushers: ASTOR-TWIN's table, the same power on the same total disc area (HAND)"}}
    C = am.build(q, pe, L, "astor_x", "tractor_lowcam", mass=MASS_KG, CD0=cda / S0, thrust=thrust)
    Se, ce = C["S"], C["c"]
    kL, kM = VM.S_REF / Se, VM.S_REF * VM.C_REF / (Se * ce)
    xc = cg / 1000.0 / VM.C_REF
    r0, r4, re = lattice(p), lattice(p, alpha=4.0), lattice(p, tail_inc=1.0)
    cm = lambda r: r["Cm"] + r["CL"] * xc
    CLa = (r4["CL"] - r0["CL"]) / math.radians(4.0) * kL
    Cma = (cm(r4) - cm(r0)) / math.radians(4.0) * kM
    CLde = (re["CL"] - r0["CL"]) * kL
    lt = p["tail_panel_tip"] / p["tail_panel_root"]
    tail_y = (p["tail_pivot_y"] + p["tail_panel_span"] * (1 + 2 * lt) / (3 * (1 + lt))) / 1000.0
    ar = w["aspect_ratio"]
    k_ind = twin["k"] + (1 / ar - twin["S"] / twin["b"] ** 2) / (math.pi * 0.85)
    CL_MAX = twin["CL_MAX"] * twin["S"] / Se
    g, wh, amps, src = A.PACKS["12S 5000 mAh LiPo"]
    mb = {"total_g": MASS_KG * 1000.0, "empty_g": MASS_KG * 1000.0 - g, "without_nets_g": MASS_KG * 1000.0 - 450.0,
          "pack": {"name": "12S 5000 mAh LiPo", "g": g, "Wh_nominal": wh, "max_A": amps, "source": src},
          "items": [{"item": "ASTOR-X with one launcher and three nets, less the pack", "g": round(MASS_KG * 1000.0 - g), "source": "ASTOR-TWIN's + 0.30 kg (astor_big: HAND)"}]}
    C.update({"aircraft": "ASTOR-X", "design": "x", "sweep_deg": SWEEP, "tail_fan": 1.0,
              "CL0": r0["CL"] * kL, "CLa": CLa, "Cm0": cm(r0) * kM, "Cma": Cma, "CLde": CLde, "Cmde": (cm(re) - cm(r0)) * kM,
              "x_cg_m": cg / 1000.0, "x_np_m": bal["neutral_point_mm"] / 1000.0, "static_margin": -Cma / CLa, "k_model": C["k"], "k": k_ind,
              "CL_MAX": CL_MAX, "alpha_stall": (CL_MAX - r0["CL"] * kL) / CLa,
              "Clda_tail": CLde * tail_y / C["b"], "Clda_aileron": _roll_per_deg(p) / (Se * C["b"]),
              "mass_budget": mb, "pack_kind": "astor", "balance": bal,
              "equipment": {"product": "astor", "launch": "rail", "recovery": "parachute", "turning_station": True, "kits": [], "pack": mb["pack"]["name"], "nets": 0},
              "aero_source": "model: lattice on the swept wing and tail; ASTOR-TWIN's drag area + struts, induced factor by aspect ratio, ASTOR-TWIN's maximum lift (HAND): no CFD of its own yet"})
    if p.get("tail_flap_chord", 0.0) > 0.0:                              # the tail halves' flaps, geared 1:1 (astor_v_flight's rule)
        th = math.acos(2 * p["tail_flap_chord"] - 1)
        gain = 1.0 + (1 - (th - math.sin(th)) / math.pi) * 0.65 * VF.TAIL_FLAP_GEAR
        for key in ("CLde", "Cmde", "Clda_tail"):
            C[key] *= gain
        C["tail_flaps"] = {"chord": p["tail_flap_chord"], "gear": VF.TAIL_FLAP_GEAR, "control_gain": round(gain, 2), "source": "HAND: thin-aerofoil flap effectiveness x 0.65"}
    C["Clda"] = C["Clda_tail"] + C["Clda_aileron"]
    C["Cnda"] = -0.1 * C["Clda_aileron"]
    C["lattice_lon"] = {k: float(C[k]) for k in ("CL0", "CLa", "Cm0", "Cma")}
    # the four rotors' differential thrust in yaw (astor_v_flight.powered); no wing in their wash: chord 0, no brake
    C["powered"] = {"prop_m": p["x_prop_diameter"] / 1000.0, "disc_m2": 2 * math.pi * (p["x_prop_diameter"] / 2000.0) ** 2, "chord_m": 0.0,
                    "motor_y_m": p["x_radius"] * math.cos(math.radians(45.0)) / 1000.0, "brake_cda_m2": 0.0,
                    "source": "differential thrust in yaw only (HAND); the wing is not blown; no crow brake modelled"}
    cfg, cfgf = VF.cfd_config("x", SWEEP), VF.cfd_config("x_flap", SWEEP)
    enough = lambda c: bool(c) and sum(r.get("status") == "solved" for r in c.get("rows", [])) >= 4
    C_pre = {k: (v.copy() if isinstance(v, (dict, np.ndarray)) else v) for k, v in C.items()}
    dx = (cg - V.CG_MM) / 1000.0                                         # the CFD turned about astor_v.CG_MM

    def to_cg(Cx):
        Cx["cfd"]["Cm"] = Cx["cfd"]["Cm"] + Cx["cfd"]["CL"] * dx / Cx["c"]
        Cx["Cm0"], Cx["Cma"] = Cx["Cm0"] + Cx["CL0"] * dx / Cx["c"], Cx["Cma"] + Cx["CLa"] * dx / Cx["c"]
        Cx["static_margin"] = -Cx["Cma"] / Cx["CLa"]
    if enough(cfg):
        VF.apply_cfd(C, cfg)
        VF.lift_table(C, "x", SWEEP, cfg)
        to_cg(C)
        C["balance"] = {**bal, "static_margin_with_the_cfd": round(C["static_margin"], 3),
                        "note": "the CG is still the lattice's choice: if this margin is not near %.2f, move astor_x_flight's CG rule to the CFD's neutral point" % MARGIN}
        if enough(cfgf):
            Cf = C_pre
            VF.apply_cfd(Cf, cfgf)
            VF.lift_table(Cf, "x", SWEEP, cfgf)
            to_cg(Cf)
            C["flap"] = {"deg": VF.FLAP_DEG, "cfd": Cf["cfd"], "CL_MAX": Cf["CL_MAX"], "alpha_stall": Cf["alpha_stall"], "Cm0": Cf["Cm0"], "alpha_margin_deg": 0.0,
                         "source": "cfd (screening RANS) with the ailerons and inboard flaps down"}
    else:
        VF.lift_table(C, "x", SWEEP, None)
    _CACHE["C"] = C
    return C


def summary() -> dict:
    """What the model says before any CFD: for reading, and for ``hawk``-style documents."""
    import astor_model as AM
    import astor_v_flight as VF
    C = coefficients()
    f = lambda v: round(float(v), 4)
    return {"aircraft": "ASTOR-X", "mass_kg": C["mass"], "wing": AX.planform(), "balance": C["balance"], "static_margin": f(C["static_margin"]),
            "CLa_per_rad": f(C["CLa"]), "CL0": f(C["CL0"]), "CL_MAX": f(C["CL_MAX"]), "alpha_stall_deg": round(math.degrees(C["alpha_stall"]), 1),
            "CD0": f(C["CD0"]), "k": f(C["k"]), "stall_m_s": round(VF.stall(C), 1), "top_speed_m_s": round(AM.top_speed(C), 1),
            "Cnb": f(C["Cnb"]), "Clb": f(C["Clb"]), "Cmq": f(C["Cmq"]), "Clp": f(C["Clp"]), "Cnr": f(C["Cnr"]),
            "Cmde_per_deg": f(C["Cmde"]), "Clda_per_deg": {"ailerons": f(C["Clda_aileron"]), "tail": f(C["Clda_tail"])},
            "linear_modes_bare_airframe_at_30_m_s": {k: {a: (round(b, 3) if isinstance(b, float) else b) for a, b in r.items()} for k, r in am.name_modes(am.modes(C, 30.0)).items()},
            "aero_source": C["aero_source"]}


if __name__ == "__main__":
    import yaml
    out = summary()
    (am.DATA / "astor_x").mkdir(parents=True, exist_ok=True)
    (am.DATA / "astor_x" / "astor_x_model.yaml").write_text(yaml.safe_dump(out, sort_keys=False, allow_unicode=True, width=150))
    print(yaml.safe_dump(out, sort_keys=False, allow_unicode=True, width=150))
