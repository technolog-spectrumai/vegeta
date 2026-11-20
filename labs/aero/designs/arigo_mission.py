"""Mission metrics for the ARIGO family, from the client's side (notebook 39, the "which product for what" round):
how fast it gets there, how far from the protected site it meets a target, whether it can wait on a perch and drop
from it, how it comes back (landing), whether one person can throw it, and how far it can go and still come home.

Point mass (the configuration's polar and thrust, as ``arigo_perf``) unless marked MuJoCo; hand values marked HAND.
Motor failure is out of scope (a lost aircraft is accepted).
"""
from __future__ import annotations

import math

import numpy as np

import arigo1_model as am
import arigo1_mujoco as amj

# ---- the scenario (HAND, the client's assumptions; change them here)
DETECT_M = 3000.0                 # HAND: the site's sensor cues a target at 3 km (radar / RF / acoustic, +-1 km)
ALERT_S = 2.0                     # HAND: from the cue to the launch command on a perch or a site launcher
HAND_ALERT_S = 10.0               # HAND: from the cue to the throw when a person has to pick it up
TARGETS_M_S = (15.0, 30.0, 50.0)  # small quadcopter, fast FPV / small fixed wing, fast one-way fixed wing
INTERCEPT_ALT_M = 50.0            # the target's height for the keep-out distance (a low drone)
HIGH_ALT_M = 150.0                # a high bandit (the Gavilan brief: bandits from 3 m to 150 m and higher)
PERCH_HEIGHT_M = 15.0             # HAND: a mast or a roof edge
PERCH_TEST_M = 60.0               # the drop is flown from 60 m: the height it needs is measured, not clipped
PERCH_PUSH_M_S = 4.0              # HAND: the spring/servo push off the perch
PERCH_NOSE_DOWN_DEG = 30.0
GLIDE_SLOPE_DEG = 6.0             # HAND: the approach path (steeper than a manned aircraft's 3 deg)
APPROACH_HEIGHT_M = 25.0
FLARE_HEIGHT_M = 2.5
OBSTACLE_M = 10.0                 # the landing distance is over a 10 m obstacle (a fence, a building)
CHASE_S = 60.0                    # HAND: full power at the target (the terminal chase) for the radius of action
RESERVE = 0.15                    # HAND: pack reserve at the return
HAND_MASS_KG = 2.0                # HAND: what one person throws comfortably and repeatably


def _perf():
    import arigo_perf
    return arigo_perf


# ------------------------------------------------------------------------------------------------ reach
def reach(C, V0, h0, t0, alt=INTERCEPT_ALT_M, t_end=400.0, dt=0.05) -> dict:
    """Distance flown over time from a launch that ends at (t0, speed V0, height h0): accelerate level to the speed of
    best climb, climb at it to ``alt``, then accelerate level to the top speed (point mass, full throttle)."""
    P = _perf()
    cd_of, _ = P.polar(C)
    W, S, m = C["mass"] * am.G, C["S"], C["mass"]
    q = lambda V: 0.5 * am.RHO * V * V
    D = lambda V: q(V) * S * cd_of(W / (q(V) * S))
    Vg = np.linspace(V0, 60.0, 200)
    roc = np.array([(P.thrust(C, v) - D(v)) * v / W for v in Vg])
    Vc = float(Vg[int(np.argmax(roc))])
    t, x, h, V = t0, 0.0, h0, V0
    T, X, H, Vs = [t], [x], [h], [V]
    while t < t_end:
        ex = P.thrust(C, V) - D(V)
        if h < alt and V >= Vc - 0.3:
            gam = math.asin(max(0.0, min(1.0, ex / W)))
            h += V * math.sin(gam) * dt
            x += V * math.cos(gam) * dt
        else:
            V += max(ex, 0.0) / m * dt
            x += V * dt
        t += dt
        T.append(t), X.append(x), H.append(h), Vs.append(V)
    return {"t": np.array(T), "x": np.array(X), "h": np.array(H), "V": np.array(Vs), "V_climb": Vc}


def time_to(r, dist_m):
    k = np.where(r["x"] >= dist_m)[0]
    return float(r["t"][k[0]]) if len(k) else None


def keep_out(r, Vt, alert_s, detect_m=DETECT_M):
    """A target flies straight at the site at Vt from ``detect_m``; the aircraft launches ``alert_s`` after the cue
    and flies ``r``. The distance from the site at which they meet [m] (head on); <= 0: the target arrives first."""
    for t, x in zip(r["t"] + alert_s, r["x"]):
        if x >= detect_m - Vt * t:
            return round(detect_m - Vt * t, 0)
    return 0.0


# ------------------------------------------------------------------------------------------------ energy
def radius_of_action(C, pm) -> dict:
    """The furthest a target can be met and the aircraft still come home with the reserve: out at the speed of best
    range ("economy") or at 80 % of the top speed ("dash"), CHASE_S at full power, home at the speed of best range."""
    P = _perf()
    cd_of, _ = P.polar(C)
    W, S = C["mass"] * am.G, C["S"]
    q = lambda V: 0.5 * am.RHO * V * V
    D = lambda V: q(V) * S * cd_of(W / (q(V) * S))
    E, _, _ = P.pack(C)
    E_J = E * 3600.0 * (1.0 - RESERVE)
    Vr = pm["energy"]["range_speed_m_s"]
    Vd = 0.8 * pm["max_level_speed_m_s"]
    sel = C["thrust"].get("propulsor") or {}
    rated = sel.get("power_w_per_unit", 400.0) * sel.get("units", 1) / P.ETA_DRIVE + P.P_AVIONICS_W   # the motors' rating
    fin = lambda w: w if math.isfinite(w) else rated                 # beyond the propulsor map: the rating
    Pr, Pd = fin(P.electrical_power(C, D(Vr), Vr)), min(fin(P.electrical_power(C, D(Vd), Vd)), rated)
    Pf = min(fin(P.electrical_power(C, P.thrust(C, Vd), Vd)), rated)
    left = E_J - Pf * CHASE_S
    per_m_eco, per_m_dash = Pr / Vr + Pr / Vr, Pd / Vd + Pr / Vr
    return {"full_power_min": round(E * 3600.0 / Pf / 60.0, 1),
            "economy_km": round(max(left, 0) / per_m_eco / 1000, 1), "dash_out_km": round(max(left, 0) / per_m_dash / 1000, 1),
            "dash_speed_m_s": round(Vd, 1), "range_speed_m_s": Vr, "full_power_W": round(Pf, 0), "dash_power_W": round(Pd, 0),
            "rule": f"out, {CHASE_S:.0f} s at full power, home at the best-range speed, {RESERVE:.0%} of the usable pack left"}


# ------------------------------------------------------------------------------------------------ landing
def landing_point_mass(C, pm) -> dict:
    """The approach (1.3 x stall), the touchdown (1.1 x stall), the energy in it, the glide and the space to land."""
    P = _perf()
    cd_of, cl_max = P.polar(C)
    W, S, m = C["mass"] * am.G, C["S"], C["mass"]
    Vs = pm["min_speed_stall_m_s"]
    Va, Vtd = 1.3 * Vs, 1.1 * Vs
    CLa = W / (0.5 * am.RHO * Va * Va * S)
    gam = math.atan(cd_of(CLa) / CLa)                                # power off at the approach speed
    cls = np.linspace(0.15, cl_max * 0.95, 80)
    ld = cls / np.array([cd_of(c) for c in cls])
    Vmin_sink = [math.sqrt(W / (0.5 * am.RHO * S * c)) for c in cls]
    sink = [v * cd_of(c) / c for v, c in zip(Vmin_sink, cls)]
    floats = (Va * Va - Vtd * Vtd) / (2 * am.G * cd_of(CLa) / CLa)   # the flare: bleeding speed against the drag
    slide = Vtd * Vtd / (2 * am.G * 0.5)                             # HAND: belly on grass, friction 0.5
    return {"approach_speed_m_s": round(Va, 1), "touchdown_speed_m_s": round(Vtd, 1),
            "touchdown_energy_J": round(0.5 * m * Vtd * Vtd, 0), "best_glide_ratio": round(float(ld.max()), 1),
            "min_sink_m_s": round(float(min(sink)), 2), "idle_glide_angle_deg": round(math.degrees(gam), 1),
            "landing_distance_over_10m_m": round(OBSTACLE_M / math.tan(gam) + floats + slide, 0),
            "slide_m": round(slide, 1), "note": "power off; the slide on a belly on grass (friction 0.5, HAND)"}


def landing_sim(sim) -> dict:
    """MuJoCo: a powered approach on a GLIDE_SLOPE_DEG path to an aim point, speed held at 1.3 x stall on the
    throttle; at FLARE_HEIGHT_M the throttle is cut and the nose raised to hold a small sink until the belly touches.
    The touchdown speeds, the miss and the slide."""
    C = sim.C
    Vs = math.sqrt(C["mass"] * am.G / (0.5 * am.RHO * C["S"] * C["CL_MAX"]))
    Va = 1.3 * Vs
    aim = APPROACH_HEIGHT_M / math.tan(math.radians(GLIDE_SLOPE_DEG))
    sim.start(Va, APPROACH_HEIGHT_M)
    ap = amj.Autopilot(sim)
    st = {"td": None, "flare": False}
    gs = math.radians(GLIDE_SLOPE_DEG)

    def fn(t, s):
        sd = s.state()
        x, z = sd["pos"][0], sd["pos"][2]
        if st["td"] is None and z < 0.12:
            st["td"] = {"t": t, "x": x, "V": float(np.linalg.norm(sd["v_world"])), "vz": float(sd["v_world"][2]),
                        "theta_deg": math.degrees(sd["theta"])}
        if st["td"] is not None:
            s.controls[:] = (0.0, 0.0, 0.0, 0.0)
            return
        h_path = max(0.0, (aim - x) * math.tan(gs))
        if z > FLARE_HEIGHT_M and not st["flare"]:
            gam_c = -gs + 0.08 * (h_path - z)                          # the path, with a correction on the height error
            ap(phi_c=0.0, theta_c=gam_c + s.trim["alpha_deg"] * math.pi / 180, V_c=Va)
        else:
            st["flare"] = True
            vz = sd["v_world"][2]
            ap(phi_c=0.0, theta_c=math.radians(4.0) + 0.15 * (-0.5 - vz), throttle=0.0)
    t0, x_stop = sim.d.time, None                                    # own loop: amj.run stops below 0.3 m
    while sim.d.time - t0 < 45.0:
        fn(sim.d.time - t0, sim)
        sim.step(5)
        sd = sim.state()
        if st["td"] is not None and np.linalg.norm(sd["v_world"]) < 0.3:
            x_stop = float(sd["pos"][0])
            break
    td = st["td"]
    if td is None:
        return {"touched_down": False}
    x_stop = x_stop if x_stop is not None else float(sim.state()["pos"][0])
    return {"touched_down": True, "touchdown_speed_m_s": round(td["V"], 1), "sink_at_touchdown_m_s": round(-td["vz"], 2),
            "pitch_at_touchdown_deg": round(td["theta_deg"], 1), "miss_m": round(td["x"] - aim, 1), "slide_m": round(x_stop - td["x"], 1),
            "rule": f"{GLIDE_SLOPE_DEG:.0f} deg path at 1.3 x stall, flare at {FLARE_HEIGHT_M} m, belly on the ground (friction 0.5)"}


# ------------------------------------------------------------------------------------------------ perch
def perch_drop(sim) -> dict:
    """MuJoCo: sitting on a PERCH_HEIGHT_M perch, motor off; on the alert it is pushed off at PERCH_PUSH_M_S nose down
    PERCH_NOSE_DOWN_DEG at full throttle, dives to 1.2 x stall, pulls out and climbs at 12 deg. The height it needs,
    and the time to safe flight (back at the perch's height at 1.3 x stall). Below 0.3 m: it hit the ground."""
    C = sim.C
    Vs = math.sqrt(C["mass"] * am.G / (0.5 * am.RHO * C["S"] * C["CL_MAX"]))
    H = PERCH_TEST_M
    sim.start(1.15 * Vs, H)
    th = math.radians(PERCH_NOSE_DOWN_DEG)
    import mujoco
    sim.d.qpos[3:7] = (math.cos(th / 2), 0.0, math.sin(th / 2), 0.0)
    sim.d.qvel[:3] = (PERCH_PUSH_M_S * math.cos(th), 0.0, -PERCH_PUSH_M_S * math.sin(th))
    sim.d.qvel[3:] = 0.0
    mujoco.mj_forward(sim.m, sim.d)
    ap = amj.Autopilot(sim)
    st = {"pulled": False, "t_safe": None}

    def fn(t, s):
        sd = s.state()
        if not st["pulled"] and sd["V"] >= 1.2 * Vs:
            st["pulled"] = True
        ap(phi_c=0.0, theta_c=math.radians(12.0) if st["pulled"] else -th, throttle=1.0)
        if st["t_safe"] is None and st["pulled"] and sd["pos"][2] >= H and sd["V"] >= 1.3 * Vs:
            st["t_safe"] = t
    h = amj.run(sim, 15.0, fn)
    need = float(H - min(h["z"]))
    ok = need + 1.0 <= PERCH_HEIGHT_M and st["t_safe"] is not None          # 1 m to spare
    return {"ok": ok, "height_needed_m": round(need, 1), "time_to_safe_s": round(st["t_safe"], 2) if st["t_safe"] else None,
            "rule": f"flown from {PERCH_TEST_M:.0f} m; ok on a {PERCH_HEIGHT_M:.0f} m perch (HAND) with 1 m to spare; pushed off at {PERCH_PUSH_M_S} m/s, {PERCH_NOSE_DOWN_DEG:.0f} deg nose down, full throttle"}


def perch_wind(C) -> dict:
    """The wind that lifts a level, parked aircraft off its perch: lift at zero incidence = weight (no tie-down)."""
    P = _perf()
    W, S = C["mass"] * am.G, C["S"]
    Cq = {**C, "thrust": {"T0_N": 0.0, "V0_m_s": 1.0}}
    F, _, _ = am.aero(Cq, (20.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    CL0 = -F[2] / (0.5 * am.RHO * 400.0 * S)
    del P
    return {"lift_off_wind_m_s": round(math.sqrt(W / (0.5 * am.RHO * S * CL0)), 1) if CL0 > 0.01 else None, "CL_at_zero_alpha": round(CL0, 3)}


# ------------------------------------------------------------------------------------------------ the belly
def belly(cls, q, kind) -> dict:
    """What touches the ground first in a level belly landing (CAD): the propulsor's lowest point against the rest's.
    Negative clearance: the propeller (or the duct) touches first."""
    p = cls().resolve(**{**q, "show_props": True})
    parts = cls().parts(p)
    rot = {k: v for k, v in parts.items() if k in ("props", "blades")}
    if not rot and kind == "edf":
        rot = {"fans": parts["motors"]}
    rest = [v for k, v in parts.items() if k not in rot and k not in ("props", "blades")]
    z_rest = min(s.val().BoundingBox().zmin if hasattr(s, "val") else s.BoundingBox().zmin for s in rest)
    z_rot = min(s.val().BoundingBox().zmin if hasattr(s, "val") else s.BoundingBox().zmin for s in rot.values()) if rot else None
    clear = None if z_rot is None else round(z_rot - z_rest, 1)
    folds = kind in ("pusher", "twin")
    if kind == "edf":
        risk = "fans in ducts: no blade strike; the intakes can take in grass and grit"
    elif clear is not None and clear >= 0:
        risk = "the propeller clears the ground in a level belly landing" + ("; folding blades" if folds else "")
    else:
        risk = "the propeller reaches below the belly: stop it before touchdown" + ("; folding blades take the strike" if folds else "; a fixed propeller breaks")
    return {"propulsor_clearance_mm": clear, "folding_blades": folds, "risk": risk}


# ------------------------------------------------------------------------------------------------ the card
def mission(C, pm, sim_factory, launch, kind, cad=None) -> dict:
    """Every mission metric for one configuration. ``launch``: the card's hand/catapult results; ``cad``: (class,
    parameters) for the belly check."""
    Vs = pm["min_speed_stall_m_s"]
    perch = perch_drop(sim_factory())
    # how it leaves the site: the perch if it can drop from it; else a launcher at the site (no set-up time)
    if perch["ok"]:
        site = {"how": "perch", "t0": perch["time_to_safe_s"], "V0": 1.3 * Vs, "h0": PERCH_HEIGHT_M}
    elif launch["catapult"]["ok"]:
        site = {"how": "site launcher", "t0": launch["catapult"]["time_to_safe_s"], "V0": 1.3 * Vs, "h0": amj.SAFE_HEIGHT_M}
    else:
        site = {"how": "none", "t0": 30.0, "V0": 1.3 * Vs, "h0": amj.SAFE_HEIGHT_M}
    r = reach(C, site["V0"], site["h0"], site["t0"])
    rh = reach(C, site["V0"], site["h0"], site["t0"], alt=HIGH_ALT_M)
    hand = launch["hand"]
    r_hand = reach(C, 1.3 * Vs, amj.SAFE_HEIGHT_M, hand["time_to_safe_s"]) if hand["ok"] else None
    k = np.where(rh["h"] >= HIGH_ALT_M - 0.5)[0]
    out = {
        "site_launch": site["how"], "site_launch_time_s": round(site["t0"], 2),
        "perch": {**perch, **perch_wind(C)},
        "reach": {f"time_to_{d // 1000}km_s": (round(time_to(r, d), 1) if time_to(r, d) else None) for d in (1000, 3000, 5000, 10000)},
        "time_to_150m_alt_s": round(float(rh["t"][k[0]]), 1) if len(k) else None,
        "keep_out_m": {f"target_{v:.0f}_m_s": keep_out(r, v, ALERT_S) for v in TARGETS_M_S},
        "keep_out_hand_m": {f"target_{v:.0f}_m_s": keep_out(r_hand, v, HAND_ALERT_S) for v in TARGETS_M_S} if r_hand else None,
        "tail_chase_ok": {f"target_{v:.0f}_m_s": bool(pm["max_level_speed_m_s"] > 1.15 * v) for v in TARGETS_M_S},
        "radius": radius_of_action(C, pm),
        "landing": landing_point_mass(C, pm),
        "landing_sim": landing_sim(sim_factory()),
        "hand": {"mass_kg": round(C["mass"], 3), "throwable": bool(hand["ok"] and C["mass"] <= HAND_MASS_KG),
                 "throw_margin": round(amj.HAND_RELEASE_M_S / Vs, 2)},
        "assumptions": {"detect_m": DETECT_M, "alert_s": ALERT_S, "hand_alert_s": HAND_ALERT_S, "target_alt_m": INTERCEPT_ALT_M,
                        "perch_m": PERCH_HEIGHT_M, "glide_slope_deg": GLIDE_SLOPE_DEG, "hand_mass_kg": HAND_MASS_KG,
                        "note": "HAND values: the client's scenario; see arigo_mission.py"},
    }
    if cad:
        try:
            out["belly"] = belly(*cad, kind)
        except Exception as e:                                       # noqa: BLE001  CAD trouble must not stop the card
            out["belly"] = {"error": str(e)[:200]}
    return out


def summary(ms) -> dict:
    """The mission metrics that go into the card's summary (and the KPIs)."""
    ls, lp = ms["landing_sim"], ms["landing"]
    return {"perch_launch_ok": ms["perch"]["ok"], "perch_height_needed_m": ms["perch"]["height_needed_m"],
            "site_launch_time_s": ms["site_launch_time_s"],
            "time_to_1km_s": ms["reach"]["time_to_1km_s"], "time_to_3km_s": ms["reach"]["time_to_3km_s"],
            "time_to_5km_s": ms["reach"]["time_to_5km_s"], "time_to_10km_s": ms["reach"]["time_to_10km_s"],
            "time_to_150m_alt_s": ms["time_to_150m_alt_s"],
            "keep_out_15_m_s_m": ms["keep_out_m"]["target_15_m_s"], "keep_out_30_m_s_m": ms["keep_out_m"]["target_30_m_s"],
            "keep_out_50_m_s_m": ms["keep_out_m"]["target_50_m_s"],
            "tail_chase_30_m_s": ms["tail_chase_ok"]["target_30_m_s"], "tail_chase_50_m_s": ms["tail_chase_ok"]["target_50_m_s"],
            "radius_economy_km": ms["radius"]["economy_km"], "radius_dash_km": ms["radius"]["dash_out_km"],
            "approach_speed_m_s": lp["approach_speed_m_s"], "touchdown_energy_J": lp["touchdown_energy_J"],
            "best_glide_ratio": lp["best_glide_ratio"], "landing_distance_m": lp["landing_distance_over_10m_m"],
            "landing_sink_m_s": ls.get("sink_at_touchdown_m_s"), "landing_miss_m": abs(ls["miss_m"]) if ls.get("touched_down") else None,
            "landing_touchdown_m_s": ls.get("touchdown_speed_m_s"),
            "propulsor_clearance_mm": (ms.get("belly") or {}).get("propulsor_clearance_mm"),
            "belly_safety": belly_score(ms.get("belly") or {}), "full_power_min": ms["radius"]["full_power_min"],
            "throwable": ms["hand"]["throwable"], "throw_margin": ms["hand"]["throw_margin"],
            "perch_lift_off_wind_m_s": ms["perch"]["lift_off_wind_m_s"]}


def belly_score(b) -> float | None:
    """0-100: clears the ground 100; ducted fans 75 (no blades, but debris in the intakes); below the belly with
    folding blades 50; below the belly, fixed blades 0. HAND scale."""
    c = b.get("propulsor_clearance_mm")
    if c is None:
        return None
    if "fans" in b.get("risk", ""):
        return 75.0
    if c >= 0:
        return 100.0
    return 50.0 if b.get("folding_blades") else 0.0


def setup_of(name):
    """(CAD class, parameters, propulsion kind) from a card's display name, e.g. "ARIGO-0 strut hybrid"."""
    import arigo1 as a1
    w = next((x for x in ("goshawk", "zero", "hybrid") if x in name), "hybrid")
    if name.startswith("ARIGO-20"):
        import arigo20 as a20
        return a20.Arigo20, a20.variant(), "edf"
    if name.startswith("ARIGO-2"):
        import arigo2 as a2
        return a2.Arigo2, a2.variant(w), "twin"
    prop = "pusher" if name.startswith("ARIGO-1") else ("tractor" if "strut" in name else "tractor_lowcam")
    return a1.Arigo1, a1.variant(w, prop), prop
