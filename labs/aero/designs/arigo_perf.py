"""The ARIGO family's flight performance card, one per configuration (notebook 39): speeds, turning, climbing,
roll / pitch / yaw response and chasing. Run by ``sims/arigo_perf.py`` (and by default in ``arigo_all.sh``).

Two sources, marked in the output:

- **point-mass** (the configuration's coefficients — CFD tables where there are, else lattice + hand — and its hand
  thrust curve T = T0 (1 - V/V0)): stall speed, top level speed, sustained and instantaneous turns over speed, corner
  speed, climb rate and angle, the time to run down a fleeing bandit;
- **MuJoCo 6-DOF** (``arigo1_mujoco``): full-aileron roll, full-elevator pitch, full-rudder yaw at cruise, and three
  chases under proportional navigation (``arigo1_mujoco.CHASES``): zig-zag (14 m/s weaving), urban (10 m/s, hard
  90-degree turns every 2.5 s at 20 m height; below 3 m is a crash), open field (18 m/s, gentle weave, 150 m ahead).

Hand limits: the structural load factor N_MAX = 6 (a 2.5 kg foam/composite airframe, +-1), mass as in the models.
"""
from __future__ import annotations

import math

import numpy as np

import arigo1_model as am
import arigo1_mujoco as amj

N_MAX = 6.0                       # HAND: structural limit
# energy (endurance and range): the pack, the efficiencies, the avionics
# the packs and the masses: arigo_mass (the default pack is in each model's mass)
USABLE = 0.85                     # HAND: usable share of the nominal energy (Li-ion to 3.0 V/cell under load, ageing) +-0.05
ETA_PROFILE = 0.80                # HAND: propeller profile + swirl losses on top of the ideal (actuator-disc) efficiency +-0.05
ETA_FAN = 0.80                    # HAND: ducted fan rotor+stator efficiency on top of the ideal (jet) efficiency +-0.05
ETA_DRIVE = 0.85                  # HAND: motor x ESC at part load +-0.05
P_AVIONICS_W = 15.0               # HAND: flight computer, Jetson-class vision, radio, servos (+-5 W)
BANDIT_SPEEDS = (10.0, 14.0, 18.0)


def _lift_drag(C, V, al):
    """Lift and drag coefficients at alpha [rad] (no thrust, no rates, elevator neutral — the rig is in the table)."""
    Cq = {**C, "thrust": {"T0_N": 0.0, "V0_m_s": 1.0}}
    F, _, _ = am.aero(Cq, (V * math.cos(al), 0.0, V * math.sin(al)), (0.0, 0.0, 0.0))
    qS = 0.5 * am.RHO * V * V * C["S"]
    return (-F[2] * math.cos(al) + F[0] * math.sin(al)) / qS, (-F[0] * math.cos(al) - F[2] * math.sin(al)) / qS


def polar(C):
    """CL -> CD along the unstalled branch (alpha from -4 deg to the stall), as a callable."""
    als = np.radians(np.linspace(-4.0, math.degrees(C["alpha_stall"]), 60))
    cl, cd = zip(*(_lift_drag(C, 20.0, a) for a in als))
    cl, cd = np.maximum.accumulate(np.array(cl)), np.array(cd)
    return lambda CL: float(np.interp(CL, cl, cd)), float(cl.max())


def thrust(C, V, throttle=1.0):
    return am.max_thrust(C, V) * throttle


def point_mass(C) -> dict:
    cd_of, cl_max = polar(C)
    W, S = C["mass"] * am.G, C["S"]
    q = lambda V: 0.5 * am.RHO * V * V
    Vs = math.sqrt(W / (q(1.0) * S * cl_max))
    excess = lambda V, n=1.0: thrust(C, V) - q(V) * S * cd_of(n * W / (q(V) * S))
    Vgrid = np.linspace(Vs * 1.02, 68.0, 400)
    ex = np.array([excess(V) for V in Vgrid])
    Vmax = float(Vgrid[np.where(ex > 0)[0].max()]) if (ex > 0).any() else float("nan")
    roc = ex * Vgrid / W
    k = int(np.argmax(roc))
    ang = np.degrees(np.arcsin(np.clip(ex / W, -1, 1)))
    # turns: sustained (thrust = drag at n, capped by CL_max and N_MAX) and instantaneous (CL_max or N_MAX)
    turns = []
    for V in [v for v in (12, 14, 16, 18, 20, 22, 25, 28, 30, 35, 40, 45, 50) if Vs * 1.05 < v < Vmax + 0.1]:
        n_inst = min(q(V) * S * cl_max / W, N_MAX)
        lo, hi = 1.0, n_inst
        if excess(V, 1.0) < 0:
            n_sus = float("nan")
        else:
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if excess(V, mid) > 0 else (lo, mid)
            n_sus = lo
        rate = lambda n: math.degrees(am.G * math.sqrt(max(n * n - 1, 0)) / V)
        rad = lambda n: V * V / (am.G * math.sqrt(n * n - 1)) if n > 1.0001 else float("inf")
        turns.append({"V_m_s": V, "sustained_n": round(n_sus, 2), "sustained_rate_deg_s": round(rate(n_sus), 1) if n_sus == n_sus else None,
                      "sustained_radius_m": round(rad(n_sus), 1) if n_sus == n_sus and n_sus > 1.0001 else None,
                      "instant_n": round(n_inst, 2), "instant_rate_deg_s": round(rate(n_inst), 1), "instant_radius_m": round(rad(n_inst), 1)})
    sus = [t for t in turns if t["sustained_rate_deg_s"]]
    best_s = max(sus, key=lambda t: t["sustained_rate_deg_s"]) if sus else None
    best_i = max(turns, key=lambda t: t["instant_rate_deg_s"]) if turns else None
    sel = C["thrust"].get("propulsor")
    return {"source": f"point-mass on {C.get('aero_source', 'model')}; thrust: " + (f"propulsor map {sel['entry']} x {sel['units']}, {sel['power_w_per_unit']:.0f} W each" if sel else "hand line"),
            "propulsion": sel,
            "mass_kg": C["mass"], "wing_loading_kg_m2": round(C["mass"] / S, 1), "CL_max": round(cl_max, 3),
            "min_speed_stall_m_s": round(Vs, 1), "max_level_speed_m_s": round(Vmax, 1),
            "climb": {"max_rate_m_s": round(float(roc[k]), 2), "at_speed_m_s": round(float(Vgrid[k]), 1),
                      "max_angle_deg": round(float(ang.max()), 1), "at_speed_m_s_for_angle": round(float(Vgrid[int(np.argmax(ang))]), 1)},
            "climb_curve": [{"V_m_s": round(float(V), 1), "rate_m_s": round(float(r), 2)} for V, r in zip(Vgrid[::20], roc[::20]) if r >= 0],
            "corner_speed_m_s": round(math.sqrt(N_MAX * W / (q(1.0) * S * cl_max)), 1),
            "best_sustained_turn": best_s, "best_instant_turn": best_i, "turns": turns,
            "run_down": run_down(C, cd_of, Vmax),
            "energy": energy(C, cd_of, Vs, Vmax),
            "mass_budget": C.get("mass_budget"),
            "pack_trade": pack_trade(C, cd_of)}


def pack(C, name=None):
    """(usable energy [Wh], description, pack mass [kg]) of the configuration's pack (``arigo_mass``), or of ``name``."""
    import arigo_mass
    mb = C["mass_budget"]
    k = arigo_mass.kind_of(C.get("propulsion", ""))
    nm = name or mb["pack"]["name"]
    g, wh, _, src = arigo_mass.packs(k)[nm]
    return wh * USABLE, f"{nm}, {wh:.0f} Wh nominal, {USABLE:.0%} usable, {g:.0f} g ({src})", g / 1000.0


def disc_area(C):
    """The propulsors' total actuator area [m2]: the propeller disc(s), or the fans' exit areas."""
    if C.get("propulsion") == "edf":
        import arigo20 as a20
        p = a20.resolve()
        return 2 * math.pi * (p["nozzle_ratio"] * p["edf_diameter"] / 2000.0) ** 2
    if C.get("propulsion") == "twin":
        import arigo2 as a2
        return 2 * math.pi * (a2.resolve(a2.variant("hybrid"))["twin_prop_diameter"] / 2000.0) ** 2
    import arigo1 as a1
    return math.pi * (a1.resolve()["prop_diameter"] / 2000.0) ** 2


def point_top_speed(C, cd_of):
    W, S = C["mass"] * am.G, C["S"]
    Vg = np.linspace(8.0, 68.0, 300)
    ex = [thrust(C, V) - 0.5 * am.RHO * V * V * S * cd_of(W / (0.5 * am.RHO * V * V * S)) for V in Vg]
    ok = [V for V, e in zip(Vg, ex) if e > 0]
    return float(max(ok)) if ok else 20.0


def electrical_power(C, T, V):
    """Battery power [W] for thrust T at V: the propulsor map's (``arigo_propulsion``) when the model has one, plus
    the avionics; else the momentum-theory estimate below."""
    sel = C["thrust"].get("propulsor") if isinstance(C.get("thrust"), dict) else None
    if sel:
        import arigo_propulsion as ap
        return ap.electrical_power(sel, T, V) + P_AVIONICS_W
    return electrical_power_momentum(C, T, V)


def electrical_power_momentum(C, T, V):
    """Battery power [W] for thrust T at speed V: ideal efficiency from momentum theory over the propulsors' area
    (actuator disc for propellers; for the fans the jet: T = rho A Vj (Vj - V), eta = 2 V / (V + Vj)), times the
    profile/fan and drive efficiencies (HAND), plus the avionics."""
    A = disc_area(C)
    if T <= 0:
        return P_AVIONICS_W
    if C.get("propulsion") == "edf":
        Vj = 0.5 * (V + math.sqrt(V * V + 4 * T / (am.RHO * A)))
        eta = 2 * V / (V + Vj) * ETA_FAN
    else:
        vi = 0.5 * (-V + math.sqrt(V * V + 2 * T / (am.RHO * A)))
        eta = V / (V + vi) * ETA_PROFILE
    return T * V / max(eta, 1e-3) / ETA_DRIVE + P_AVIONICS_W


def pack_trade(C, cd_of) -> list:
    """Every pack option on this airframe: the total mass, the stall speed, endurance and range it gives."""
    import arigo_mass
    mb = C["mass_budget"]
    out = []
    for nm in arigo_mass.packs(arigo_mass.kind_of(C.get("propulsion", ""))):
        m = (mb["empty_g"] + arigo_mass.packs(arigo_mass.kind_of(C.get("propulsion", "")))[nm][0]) / 1000.0
        C2 = {**C, "mass": m}
        cl_max = polar(C)[1]
        Vs = math.sqrt(m * am.G / (0.5 * am.RHO * C["S"] * cl_max))
        e = energy(C2, cd_of, Vs, point_top_speed(C2, cd_of), pack_name=nm)
        out.append({"pack": nm, "total_kg": round(m, 3), "stall_m_s": round(Vs, 1), "endurance_min": e["endurance_min"],
                    "range_km": e["range_km"], "wing_loading_kg_m2": round(m / C["S"], 1), "default": nm == mb["pack"]["name"]})
    return out


def energy(C, cd_of, Vs, Vmax, pack_name=None) -> dict:
    """Endurance and still-air range on the pack: power required over speed (level, 1 g), the speed of least power
    (endurance) and of least energy per metre (range), and at cruise."""
    E, what, _ = pack(C, pack_name)
    W, S = C["mass"] * am.G, C["S"]
    Vs_ = np.linspace(max(Vs * 1.15, 8.0), max(Vmax, Vs * 1.2), 120)            # 15 % above the stall: a usable margin
    P = np.array([electrical_power(C, 0.5 * am.RHO * V * V * S * cd_of(W / (0.5 * am.RHO * V * V * S)), V) for V in Vs_])
    ke, kr = int(np.argmin(P)), int(np.argmin(P / Vs_))
    Vc = 30.0 if C.get("propulsion") == "edf" else 18.0
    Pc = float(np.interp(Vc, Vs_, P))
    return {"pack": what, "usable_Wh": round(E, 1), "avionics_W": P_AVIONICS_W,
            "endurance_min": round(float(E / P[ke] * 60), 1), "endurance_speed_m_s": round(float(Vs_[ke]), 1), "power_at_endurance_W": round(float(P[ke]), 1),
            "range_km": round(float(E / P[kr] * 3600 * Vs_[kr] / 1000), 1), "range_speed_m_s": round(float(Vs_[kr]), 1),
            "cruise": {"speed_m_s": Vc, "power_W": round(Pc, 1), "endurance_min": round(E / Pc * 60, 1), "range_km": round(E / Pc * 3.6 * Vc, 1)},
            "power_curve": [{"V_m_s": round(float(v), 1), "power_W": round(float(p), 1)} for v, p in zip(Vs_[::10], P[::10])],
            "source": "point mass + momentum theory; efficiencies, pack usable share and avionics power are hand values (see arigo_perf.py)"}


def run_down(C, cd_of, Vmax, start_m=60.0, catch_m=5.0, V0=None) -> dict:
    """A bandit flying straight away at constant speed, from ``start_m`` ahead: full throttle level from cruise; the
    time to close to ``catch_m`` (point mass, no manoeuvring)."""
    W, S, m = C["mass"] * am.G, C["S"], C["mass"]
    out = {}
    for Vb in BANDIT_SPEEDS:
        V = V0 or max(18.0, math.sqrt(W / (0.5 * am.RHO * S * 0.6)))
        gap, t, dt = start_m, 0.0, 0.02
        while gap > catch_m and t < 120:
            q = 0.5 * am.RHO * V * V
            D = q * S * cd_of(W / (q * S))
            V += (thrust(C, V) - D) / m * dt
            gap -= (V - Vb) * dt
            t += dt
        out[f"bandit_{Vb:.0f}_m_s"] = {"time_to_5m_s": round(t, 1) if gap <= catch_m else None, "speed_at_catch_m_s": round(V, 1),
                                        "note": None if gap <= catch_m else f"not caught in 120 s (top speed {Vmax:.1f} m/s)"}
    return out


def responses(sim, V) -> dict:
    """Roll, pitch, yaw at speed V (MuJoCo): full aileron 2 s; full up elevator 1 s from trim; full rudder 1.5 s."""
    out = {}
    sim.start(V, 100.0)
    h = amj.run(sim, 2.0, lambda t, s: s.controls.__setitem__(1, am.DA_MAX))
    phi = np.cumsum(np.radians(h["p"])) * amj.DT * 5
    out["roll"] = {"steady_rate_deg_s": round(float(np.median(h["p"][h["t"] > 0.8])), 1),
                   "time_to_90deg_s": round(float(np.interp(math.pi / 2, phi, h["t"])), 3) if phi.max() > math.pi / 2 else None,
                   "time_to_180deg_s": round(float(np.interp(math.pi, phi, h["t"])), 3) if phi.max() > math.pi else None}
    tr = sim.start(V, 100.0)
    h = amj.run(sim, 1.0, lambda t, s: s.controls.__setitem__(0, -am.DE_MAX))
    k3 = h["t"] <= 0.3                                            # the first 0.3 s: the response, before any tumble
    out["pitch"] = {"peak_rate_deg_s": round(float(h["q"][k3].max()), 1), "peak_rate_1s_deg_s": round(float(h["q"].max()), 1),
                    "departed_in_1s": bool(h["alpha"].max() > 45.0), "pitch_gain_1s_deg": round(float(h["theta"][-1] - h["theta"][0]), 1),
                    "peak_load_factor": round(float(h["n"].max()), 2), "peak_alpha_deg": round(float(h["alpha"].max()), 1)}
    sim.start(V, 100.0)
    h = amj.run(sim, 1.5, lambda t, s: s.controls.__setitem__(2, am.DR_MAX))
    out["yaw"] = {"peak_rate_deg_s": round(float(np.abs(h["r"]).max()), 1), "peak_sideslip_deg": round(float(np.abs(h["beta"]).max()), 1),
                  "heading_change_1_5s_deg": round(float(np.degrees(np.unwrap(np.radians(h["psi"])))[-1] - h["psi"][0]), 1),
                  "bank_induced_deg": round(float(h["phi"][-1]), 1)}
    out["speed_m_s"] = V
    out["trim"] = {k: (round(float(v), 3) if not isinstance(v, (bool, str)) else v) for k, v in tr.items()}
    return out


def chases(sim_factory, Vs) -> dict:
    """The chase scenarios (``arigo1_mujoco.CHASES``) under proportional navigation, each entered at its speed or
    1.25 x this aircraft's stall speed, whichever is higher; and the zig-zag under the old lead pursuit, for
    comparison."""
    out = {}
    for name in amj.CHASES:
        V = max(amj.CHASES[name]["V"], 1.25 * Vs)
        _, out[name] = amj.chase(sim_factory(bandit=True), name, V=V)
    _, out["zigzag_lead_pursuit"] = amj.chase(sim_factory(bandit=True), "zigzag", V=max(20.0, 1.25 * Vs), guidance="lead")
    return out


def card(name, C, sim_factory, manoeuvre, V_cruise) -> dict:
    out = {"aircraft": name, "point_mass": point_mass(C)}
    pm = out["point_mass"]
    ch = chases(sim_factory, pm["min_speed_stall_m_s"])
    hand = amj.takeoff(sim_factory(), "hand")
    cat = amj.takeoff(sim_factory(), "catapult")
    launch = {"hand": hand, "catapult": cat,
              "takeoff_time_s": hand["time_to_safe_s"] if hand["ok"] else (cat["time_to_safe_s"] + 5.0 if cat["ok"] else None),
              "rule": "takeoff_time_s: the hand launch's time to safe flight; if a hand launch fails, the catapult's + 5 s (setting up)"}
    out["mujoco"] = {"responses_at_cruise": responses(sim_factory(), V_cruise), "chases": ch, "launch": launch,
                     "source": "MuJoCo 6-DOF on the same coefficients; autopilot and guidance gains by hand"}
    out["summary"] = {"min_speed_m_s": pm["min_speed_stall_m_s"], "max_speed_m_s": pm["max_level_speed_m_s"],
                      "max_climb_m_s": pm["climb"]["max_rate_m_s"],
                      "best_sustained_turn_deg_s": pm["best_sustained_turn"] and pm["best_sustained_turn"]["sustained_rate_deg_s"],
                      "tightest_sustained_radius_m": min((t["sustained_radius_m"] for t in pm["turns"] if t["sustained_radius_m"]), default=None),
                      "best_instant_turn_deg_s": pm["best_instant_turn"] and pm["best_instant_turn"]["instant_rate_deg_s"],
                      "roll_rate_deg_s": out["mujoco"]["responses_at_cruise"]["roll"]["steady_rate_deg_s"],
                      "pitch_rate_deg_s": out["mujoco"]["responses_at_cruise"]["pitch"]["peak_rate_deg_s"],
                      "yaw_rate_deg_s": out["mujoco"]["responses_at_cruise"]["yaw"]["peak_rate_deg_s"],
                      "run_down_14_m_s_bandit_s": pm["run_down"]["bandit_14_m_s"]["time_to_5m_s"],
                      "run_down_18_m_s_bandit_s": pm["run_down"]["bandit_18_m_s"]["time_to_5m_s"],
                      "zigzag_closest_m": ch["zigzag"]["closest_m"], "zigzag_time_to_5m_s": ch["zigzag"]["time_to_5m_s"],
                      "zigzag_share_within_10m": ch["zigzag"]["share_within_10m"],
                      "urban_time_to_5m_s": ch["urban"]["time_to_5m_s"], "urban_share_within_10m": ch["urban"]["share_within_10m"],
                      "urban_crashed": ch["urban"]["crashed"],
                      "open_field_time_to_5m_s": ch["open_field"]["time_to_5m_s"],
                      "open_field_share_within_10m": ch["open_field"]["share_within_10m"],
                      "endurance_min": pm["energy"]["endurance_min"], "range_km": pm["energy"]["range_km"],
                      "takeoff_time_s": launch["takeoff_time_s"], "hand_launch_ok": hand["ok"],
                      "mass_kg": round(C["mass"], 3), "pack": C["mass_budget"]["pack"]["name"],
                      "propulsor": (C["thrust"].get("propulsor") or {}).get("entry"),
                      "best_endurance_pack": max(pm["pack_trade"], key=lambda r: r["endurance_min"])["pack"],
                      "power_at_endurance_W": pm["energy"]["power_at_endurance_W"]}
    import arigo_mission                                         # the client's missions: reach, keep-out, perch, landing
    cls, q, kind = arigo_mission.setup_of(name)
    out["mission"] = arigo_mission.mission(C, pm, sim_factory, launch, kind, (cls, q))
    out["summary"].update(arigo_mission.summary(out["mission"]))
    return out


def configurations():
    """(key, display name, coefficients, sim factory, manoeuvre fn, cruise speed) for every configuration."""
    import arigo2_mujoco as m2j
    import arigo2_model as m2
    import arigo20_mujoco as m20j
    import arigo20_model as m20
    for prop, nm in (("tractor_lowcam", "ARIGO-0"), ("tractor", "ARIGO-0 strut"), ("pusher", "ARIGO-1")):
        for w in ("goshawk", "zero", "hybrid"):
            yield (f"{nm.split()[0].lower()}{'_strut' if prop == 'tractor' else ''}_{w}", f"{nm} {w}", lambda w=w, p=prop: am.coefficients(w, p),
                   lambda bandit=False, w=w, p=prop: amj.Sim(w, p, bandit=bandit), amj.manoeuvre, 18.0)
    for w in ("goshawk", "zero", "hybrid"):
        yield (f"arigo-2_{w}", f"ARIGO-2 {w}", lambda w=w: m2.coefficients(w), lambda bandit=False, w=w: m2j.Sim2(w, bandit=bandit), m2j.manoeuvre, 18.0)
    yield ("arigo-20", "ARIGO-20", m20.get, lambda bandit=False: m20j.Sim20(bandit=bandit), m20j.manoeuvre, m20.CRUISE)
