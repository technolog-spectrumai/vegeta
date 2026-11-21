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
def coefficients(name: str = "v", sweep: float = 12.0) -> dict:
    import arigo_propulsion as ap
    p, mb = design(name)
    m = mb["total_g"] / 1000.0
    wn = V.wing_numbers(p, sweep)
    b = max(wn["span_mm"], 520.0) / 1000.0
    S = wn["exposed_area_m2"] + 0.07 * 0.21                                  # the exposed panels + the carry-over across the body
    # the equivalent conventional wing for the hand formulas (lateral, damping, inertias)
    lam = p["panel_tip_chord"] / p["panel_root_chord"]
    cr = 2 * S / (b * (1 + lam)) * 1000.0
    q = A.variant(span=b * 1000.0, root_chord=cr, taper=lam, le_sweep_deg=min(sweep, 35.0), dihedral_deg=0.0,
                  wing_x=V.pivot_x(p, sweep) - 0.3 * cr, cg_x=V.CG_MM, propulsion="tractor_lowcam",
                  fuselage_length=p["fuselage_length"], tail_cone=p["tail_cone"], tail_end_diameter=p["tail_end_diameter"],
                  htail_span=2 * (p["tail_pivot_y"] + p["tail_panel_span"] * math.cos(math.radians(35.0))), htail_root_chord=p["tail_panel_root"],
                  htail_taper=p["tail_panel_tip"] / p["tail_panel_root"], htail_sweep_deg=35.0, htail_z=6.0,
                  fin_height=p["fin_height"], fin_root_chord=p["fin_root_chord"], fin_taper=p["fin_taper"], fin_sweep_deg=p["fin_sweep_deg"],
                  lowcam_drop=p["lowcam_drop"])
    q["tail_le_x"] = p["tail_pivot_x"] - 0.25 * p["tail_panel_root"]
    pe = A.resolve(q)
    L = A.Astor.layout(pe)
    sel = AM.propulsor(m)
    sel = {**sel, "layout": "tractor"}
    thrust = ap.thrust_table(sel, V=[float(v) for v in range(0, 92, 2)])
    cda = BODY_CDA + WING_CD * wn["exposed_area_m2"] + (POD_CDA if name == "boost" else 0.0)
    C = am.build(q, pe, L, f"astor_{name}", "tractor_lowcam", mass=m, CD0=cda / (L["S_mm2"] / 1e6), thrust=thrust)
    Se, ce = C["S"], C["c"]
    # the longitudinal set from the lattice on the real panels, brought to this reference (S, c) and to the CG
    kL, kM = VM.S_REF / Se, VM.S_REF * VM.C_REF / (Se * ce)
    xc = V.CG_MM / 1000.0 / VM.C_REF
    tf = 1.0 if sweep < 50.0 else 0.6                                        # the dash flies with the tail fan part folded
    r0, r4, re = VM.lattice(sweep, tail_fan=tf, p=p), VM.lattice(sweep, tail_fan=tf, alpha=4.0, p=p), VM.lattice(sweep, tail_fan=tf, tail_inc=1.0, p=p)
    cm = lambda r: r["Cm"] + r["CL"] * xc
    CLa = (r4["CL"] - r0["CL"]) / math.radians(4.0) * kL
    Cma = (cm(r4) - cm(r0)) / math.radians(4.0) * kM
    x_np = -(r4["Cm"] - r0["Cm"]) / (r4["CL"] - r0["CL"]) * VM.C_REF
    CLde = (re["CL"] - r0["CL"]) * kL
    CL_MAX = CL_MAX_SPREAD * max(math.cos(math.radians(sweep - 12.0)), 0.55) * VM.S_REF / Se * (wn["exposed_area_m2"] / 0.198)
    C.update({"aircraft": "ASTOR-BOOST" if name == "boost" else "ASTOR-V", "design": name, "sweep_deg": sweep, "tail_fan": tf,
              "CL0": r0["CL"] * kL, "CLa": CLa, "Cm0": cm(r0) * kM, "Cma": Cma, "CLde": CLde, "Cmde": (cm(re) - cm(r0)) * kM,
              "x_cg_m": V.CG_MM / 1000.0, "x_np_m": x_np, "static_margin": -Cma / CLa * 1.0,
              "CL_MAX": CL_MAX, "alpha_stall": (CL_MAX - r0["CL"] * kL) / CLa,
              "Clda_tail": CLde * TAIL_Y_M / C["b"], "Clda_aileron": VM.roll_per_deg_aileron(sweep, p) / (Se * C["b"]),                 # tailerons: +-1 deg on the two tail panels
              "Cnda": 0.0, "Cnb": C["Cnb"] * 0.9,                                        # HAND: the tractor propeller's normal force ahead of the CG (-10 %)
              "mass_budget": mb, "pack_kind": "astor",
              "equipment": {"product": "astor", "launch": "rail", "recovery": "parachute", "turning_station": True, "kits": [],
                            "pack": mb["pack"]["name"], "nets": int(p["net_count"])},
              "aero_source": "model (lattice on the swept panels + hand)"})
    C["Clda"] = C["Clda_tail"] + C["Clda_aileron"]                           # one roll command moves both
    C["Cnda"] = -0.1 * C["Clda_aileron"]                                     # HAND: the ailerons' adverse yaw (arigo1_model's rule)
    if name == "boost":
        u = ap._unit("edf-70-p2.20-e0.65-catalogue-s1", FAN_W, None)
        C["boost"] = {"table_V": [float(v) for v in range(0, 92, 2)], "table_T": [float(max(u.full(float(v))[0], 0.0)) for v in range(0, 92, 2)],
                      "power_W": FAN_W, "z_below_axis_m": 0.080}
    cfg = cfd_config(name, sweep)
    if cfg:
        ref = cfg.get("reference", {})
        cam_cfd, trav_cfd = ref.get("camber", 0.0), ref.get("carriage_travel_mm", V.RUN1_TRAVEL.get(sweep, 0.0))
        same = abs(cam_cfd - p["panel_camber"]) < 1e-4 and abs(trav_cfd - V.travel(sweep)) < 1.0
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
                                f"difference to this geometry (camber {p['panel_camber']:g}, carriage {V.travel(sweep):.0f} mm): rerun the CFD")
    return C


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


def limit_elevator(C, v_air_frd, q, de) -> float:
    """The angle-of-attack limiter (in the flight controller, under every elevator command, the pilot's included):
    the elevator may not go further nose-up than what holds the limit angle, less a term for the angle beyond it and
    the pitch rate toward it. Positive elevator is nose down."""
    al = math.atan2(v_air_frd[2], max(v_air_frd[0], 1.0))
    a_lim = C["alpha_stall"] - math.radians(ALPHA_MARGIN_DEG)
    de_hold = -(C["Cm0"] + C["Cma"] * a_lim) / C["Cmde"]
    floor = de_hold + LIMIT_K * math.degrees(al - a_lim) + LIMIT_KQ * math.degrees(q)
    return max(de, min(floor, am.DE_MAX))


class Fixed(amj.Sim):
    """The aircraft with the wing held at one sweep (the stability tests), the limiter under the elevator."""

    def __init__(self, name, sweep, mesh=None, bandit=False, alpha_limit=True):
        import mujoco
        self.C = get(name, sweep)
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
            ctl = list(self.controls)
            if self.alpha_limit:
                ctl[0] = limit_elevator(self.C, s["v_air_frd"], s["omega_frd"][1], ctl[0])
            F, M, info = am.aero(self.C, s["v_air_frd"], s["omega_frd"], *ctl)
            R = self.R()
            self.d.xfrc_applied[self.bid, :3] = R @ (F * amj.FLIP)
            self.d.xfrc_applied[self.bid, 3:] = R @ (M * amj.FLIP)
            self.info = {**info, "load_factor": float(-F[2] / (self.C["mass"] * am.G)), "limited": bool(ctl[0] != self.controls[0])}
            mujoco.mj_step(self.m, self.d)


def get(name="v", sweep=12.0) -> dict:
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
    C["sweep_deg"] = s
    return C


def stall(C) -> float:
    return math.sqrt(C["mass"] * am.G / (0.5 * am.RHO * C["S"] * C["CL_MAX"]))


def schedule(V_air: float) -> float:
    """The sweep the wing takes by airspeed when nothing else commands it: spread below 28 m/s, 60 deg above 50."""
    return SWEEPS[0] + (SWEEPS[-1] - SWEEPS[0]) * min(1.0, max(0.0, (V_air - 28.0) / 22.0))


class SimV(amj.Sim):
    """ASTOR-V / ASTOR-BOOST in MuJoCo. ``sweep_cmd`` (deg, or None: by airspeed) is followed at SWEEP_RATE;
    ``prop_on`` False stops and folds the propeller; ``boost`` 0-1 is ASTOR-BOOST's fan; dropping nets takes their
    mass off (``drop_net``)."""

    SWEEP_RATE = 60.0                        # deg/s (HAND: the carriage and the pivots' actuators)

    def __init__(self, name="v", mesh=None, bandit=False, sweep=None):
        import mujoco
        self.name = name
        self.sweep = SWEEPS[0] if sweep is None else sweep
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
            want = schedule(s["V"]) if self.sweep_cmd is None else self.sweep_cmd
            d = self.SWEEP_RATE * amj.DT
            self.sweep += max(-d, min(d, want - self.sweep))
            trim = getattr(self, "trim", None)
            self.C = blended(self.name, self.sweep)
            self.C["mass"] = float(self.m.body_mass[self.bid])
            ctl = list(self.controls)
            if self.alpha_limit:
                ctl[0] = limit_elevator(self.C, s["v_air_frd"], s["omega_frd"][1], ctl[0])
            if not self.prop_on:
                ctl[3] = 0.0
                self.C["CD0"] += FOLDED_PROP_CDA / self.C["S"]
            F, Mo, info = am.aero(self.C, s["v_air_frd"], s["omega_frd"], *ctl)
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
