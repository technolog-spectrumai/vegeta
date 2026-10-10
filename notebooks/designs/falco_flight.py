"""FALCO's aerodynamics and flight performance.

NISUS+'s module does the work through its ``design`` seams (``nisus_plus_flight``: the lattice, the derivative table
with the crow rows, the CG range, the envelope against altitude, the descent and regeneration tables, the downdraft
escape, the launch); this one binds them to FALCO's geometry (one dorsal fin, the tail on the tube, the tractor's
long nose in the pod term) and adds what the tractor brings:

* ``installation``: the propeller in the nose ahead of the fuselage and the wing — the wake fraction and the thrust
  deduction of MERLIN's open-propeller installation model (``air_propeller`` + Boreas' wake) at a flight point; small
  for a tractor (the body is behind the disc), reported and added to the airframe's Cd0 at cruise as ``cd0_installation``
  (the slipstream's scrubbing of the fuselage and the wing root is the other part: NOT in the build-up, the CFD
  rotor-disk case is its check);
* ``compare_with_nisus_plus``: the side-by-side table the study asked for — static thrust, top speed, acceleration at
  the launch and at cruise, 15 → 25 m/s, best climb, L/D, cruise power, mission energy and the feasible survey time —
  both aircraft at their design masses on their own drive maps.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import falco
import falco_systems as fsy
import nisus_plus_flight as ff
import nisus_plus_systems as fs
import peregrine_flight as pf
from merlin_flight import Airframe
from nisus_plus_flight import (FLAP_TAU, N_STRUCTURAL, V_NE_EAS, V_FE_EAS, V_CRUISE_EAS, CROW, flap_k, flap_profile_drag, cl_max,  # noqa: F401
                               trim_crow, coefficients, trim, airframe_dict, DriveUnit, envelope, envelope_vs_altitude, ceiling, climb_strategy,
                               glide_state, descent_table, best_descent, regen_table, downdraft_escape, operating_limits,
                               stall_table, gust_response, pull_up, AIRFOIL, TAIL_SECTION)

try:
    from vegeta.boreas import RHO0
except Exception:                                                        # pragma: no cover
    RHO0 = 1.225
G = 9.81
NAME = "Falco-Zero"


def _d():
    return falco.Falco()


# ================================================================================================= aerodynamics (the seams)
def aero(p=None, **kw) -> dict:
    return ff.aero(falco.resolve(p), design=_d(), **kw)


def crow_increments(p=None, flap_deg=CROW["flap_deg"], aileron_deg=CROW["aileron_deg"], **kw) -> dict:
    return ff.crow_increments(falco.resolve(p), flap_deg, aileron_deg, design=_d(), **kw)


def crow_table(p=None, **kw) -> pd.DataFrame:
    return ff.crow_table(falco.resolve(p), design=_d(), **kw)


def derivatives(p=None, *, x_cg_m: float, **kw) -> pd.DataFrame:
    """NISUS+'s derivative table on FALCO: one fin on the centre line (``n_fins`` 1 through the layout: its slope at the
    plate's own aspect ratio, its side force, the roll coupling from its height above the tube), the Cd0 from FALCO's
    build-up (the tractor fuselage as a body of revolution, the single tube)."""
    return ff.derivatives(falco.resolve(p), x_cg_m=x_cg_m, design=_d(), **kw)


def cg_range(p=None, mass_kg: float | None = None, **kw) -> dict:
    p = falco.resolve(p)
    m = mass_kg if mass_kg is not None else fsy.cg_inertia(fsy.mass_table(p=p), p)["mass_kg"]
    return ff.cg_range(p, m, design=_d(), **kw)


def launch_check(af, dr, h_m: float = 3000.0, **kw) -> dict:
    kw.setdefault("a", aero(kw.get("p")))
    return ff.launch_check(af, dr, h_m, design=_d(), **kw)


def launch_table(af, dr, heights=(1200.0, 2000.0, 3000.0), releases=(9.0, 11.0, 18.0), **kw) -> pd.DataFrame:
    kw.setdefault("a", aero(kw.get("p")))
    return ff.launch_table(af, dr, heights, releases, design=_d(), **kw)


# ================================================================================================= the tractor's installation
def _pod_for_installation(p) -> dict:
    """``air_propeller.PropPod``'s parameters from FALCO's layout (what ``merlin_flight.installation_pod`` builds from
    MERLIN's): the round fuselage behind the disc, its cowl as the front length, the wing as the pylon ``nose_length``
    behind the disc."""
    L = falco.Falco.layout(p)
    prof = falco.Falco.pod_profile(p)
    return dict(layout="tractor", pod_diameter=p["pod_width"], pod_length=float(prof[-1, 0] - prof[0, 0]), front_length=p["fairing_length"],
                rear_length=p["tail_cone"], motor_end_diameter=p["motor_diameter"], hub_diameter=p["hub_diameter"], hub_height=p["hub_height"],
                gap=p["prop_gap"], pylon_chord=p["root_chord"], pylon_thickness=p["thickness"], pylon_height=p["span"] / 2,
                pylon_gap=-L["prop_x"] - p["hub_height"] / 2)


def installation(V_tas: float, thrust_n: float, h_m: float = 3000.0, p=None) -> dict:
    """The effective wake fraction w and the thrust deduction t of the nose propeller at one flight point: MERLIN's
    recipe (``air_propeller.installation_wake`` with Boreas' ``effective_inflow``; ``installation_drag`` with the
    blade-element loading). Returns ``{'w', 't', 'source'}``; when the installation model cannot be imported, the
    tractor's textbook values (w 0.02, t 0.04: ASSUMED) with the reason."""
    p = falco.resolve(p)
    try:
        import air_propeller as ap
        from vegeta import boreas
        from vegeta.boreas import wake
    except Exception as e:                                               # pragma: no cover
        return {"w": 0.02, "t": 0.04, "source": f"ASSUMED (tractor textbook values): the installation model is not importable ({e})"}
    rho = fs.atmosphere(h_m)["rho"]
    prop, af = fsy.propeller(), fsy.blade_section()
    pod = _pod_for_installation(p)
    R_mm = prop.radius * 1000
    W = ap.installation_wake(pod, R_mm, V_tas)
    op0 = boreas.rpm_for_thrust(prop, af, thrust_n, V_tas, rho)
    w, op = wake.effective_inflow(prop, af, op0.rpm, V_tas, W, rho)
    t = ap.installation_drag(pod, R_mm, V_tas, thrust_n, rho=rho, loading=(op.r, op.dT_dr))["t"]
    return {"w": float(w), "t": float(t), "source": "calculated: air_propeller + boreas.wake on FALCO's pod (the slipstream over the wing root is not in it)"}


def installation_table(af: Airframe, dr, p=None, points=None) -> pd.DataFrame:
    """w and t at the flight points (cruise at 3000 m, the climb, the dash) with the thrust each needs."""
    p = falco.resolve(p)
    pts = points or [("cruise 18 m/s EAS, 3000 m", V_CRUISE_EAS, 3000.0, 0.0), ("climb 16 m/s EAS at 6 m/s, 1200 m", 16.0, 1200.0, 6.0),
                     ("dash 30 m/s EAS, 3000 m", 30.0, 3000.0, 0.0)]
    rows = {}
    for name, V_eas, h, roc in pts:
        atm = fs.atmosphere(h)
        V = V_eas / math.sqrt(atm["sigma"])
        T = af.drag(V, atm["rho"]) + af.mass_kg * G * roc / V
        r = installation(V, T, h, p)
        q = 0.5 * atm["rho"] * V ** 2
        rows[name] = {"TAS [m/s]": V, "thrust [N]": T, "wake fraction w": r["w"], "thrust deduction t": r["t"],
                      "installation drag t·T [N]": r["t"] * T, "as ΔCd0": r["t"] * T / (q * af.wing_area_m2), "source": r["source"]}
    return pd.DataFrame(rows).T


# ================================================================================================= the airframe
def airframe(battery_key: str = fsy.DEFAULT_PACK, p=None, *, h_m: float = 0.0, cd0_correction: float = 0.0, a=None, installation_effects: bool = True) -> Airframe:
    """Merlin's parabolic-polar ``Airframe`` for FALCO: mass from FALCO's mass table, Cd0 from FALCO's build-up at
    ``h_m``'s viscosity plus the nose propeller's thrust deduction at cruise (``installation``; off with
    ``installation_effects=False``) and a CFD correction, Oswald e = 0.9 x the lattice's span efficiency."""
    p = falco.resolve(p)
    a = a or aero(p)
    m = fsy.cg_inertia(fsy.mass_table(battery_key, p=p), p)["mass_kg"]
    atm = fs.atmosphere(h_m)
    V = V_CRUISE_EAS / math.sqrt(atm["sigma"])
    b = falco.drag_buildup(p, V, atm["nu"])
    cd0 = b["cd0"] + cd0_correction
    af0 = Airframe(NAME, m, a["S_ref"], a["AR"], cd0, oswald=pf.OSWALD_VISCOUS * a["e"], cl_max=cl_max(p))
    inst = {"w": 0.0, "t": 0.0, "source": "off"}
    if installation_effects:
        inst = installation(V, af0.drag(V, atm["rho"]), max(h_m, 1.0), p)
        cd0 += inst["t"] * af0.drag(V, atm["rho"]) / (0.5 * atm["rho"] * V ** 2 * a["S_ref"])
    af = Airframe(NAME, m, a["S_ref"], a["AR"], cd0, oswald=pf.OSWALD_VISCOUS * a["e"], cl_max=cl_max(p))
    af.cd0_buildup = b["cd0"]
    af.cd0_installation = cd0 - b["cd0"] - cd0_correction
    af.installation = inst
    return af


# ================================================================================================= the propeller trade
def propeller_trade(props=((20.0, 13.0), (20.0, 15.0), (21.0, 14.0), (22.0, 12.0), (22.0, 14.0), (19.0, 14.0), (18.0, 12.0)), *, p=None, h_m: float = 3000.0,
                    af=None, plan: fs.MissionPlan | None = None) -> pd.DataFrame:
    """The propeller's pitch and diameter on the fitted AT5220 at 8S (a map per propeller: ~2 s each): the static point
    (thrust, current, rpm), the cruise point at ``h_m`` (power, rpm, the propeller's efficiency), the best climb, the top
    speed, the launch acceleration, the mission's energy and feasible survey time. The 20x13 of the frame study cruises
    near its zero-thrust advance ratio; the 20x15 is the pick."""
    p = falco.resolve(p)
    plan = plan or fs.MissionPlan()
    af = af or airframe(fsy.DEFAULT_PACK, p, h_m=h_m, installation_effects=False)
    motor, fit = fsy.motor_model()
    atm = fs.atmosphere(h_m)
    V = plan.V_cruise_eas / math.sqrt(atm["sigma"])
    T = af.drag(V, atm["rho"])
    rows = {}
    for d, pi in props:
        import frame_study as fst
        dr = fs.build_drive(battery_v=fsy.BATTERY_V, prop=fst.propeller(d, pi), motor=motor, fit=fit, rpm=fsy.RPM_GRID)
        st = dr.at(0.0, 1.0)
        th = dr.throttle_for_thrust(V, T, atm["rho"])
        pc = dr.at(V, th, atm["rho"]) if np.isfinite(th) else {"electrical": np.nan, "rpm": np.nan, "shaft_power": np.nan}
        perf = _performance(af, dr, h_m, v_launch=17.0, v_cruise_eas=plan.V_cruise_eas)
        me = fs.mission_energy(fsy.pack(), dr, airframe_dict(af), plan)
        rows[f"APC {d:g}x{pi:g}E"] = {"static thrust [N]": st["thrust"], "static current [A]": st["current"], "static rpm": st["rpm"],
                                     "pitch speed at static rpm [m/s]": st["rpm"] / 60 * pi * 0.0254,
                                     f"cruise {plan.V_cruise_eas:g} m/s EAS, {h_m:.0f} m: battery side [W]": pc["electrical"], "cruise rpm": pc["rpm"],
                                     "cruise propeller efficiency": T * V / pc["shaft_power"] if pc["shaft_power"] else np.nan,
                                     f"best climb at {h_m:.0f} m [m/s]": perf[f"best climb at {h_m:.0f} m [m/s]"], f"top speed at {h_m:.0f} m [m/s TAS]": perf[f"top speed, level, {h_m:.0f} m [m/s TAS]"],
                                     f"acceleration at 17 m/s, {h_m:.0f} m [m/s²]": perf[f"acceleration at 17 m/s, {h_m:.0f} m [m/s²]"],
                                     "mission energy [Wh]": me["E_used_wh"], "survey power [W]": me["P_survey_w"], "feasible survey [min]": me["t_survey_feasible_s"] / 60,
                                     "within the motor's 70 A": st["current"] <= fsy.MOTORS["AT5220-A KV220"]["max_a"]}
    df = pd.DataFrame(rows).T
    df.attrs["selected"] = f"APC {fsy.PROP[0]:g}x{fsy.PROP[1]:g}E"
    return df


# ================================================================================================= NISUS+ vs FALCO
def _performance(af: Airframe, dr, h: float, *, v_launch: float, v_cruise_eas: float) -> dict:
    atm = fs.atmosphere(h)
    rho, sig = atm["rho"], atm["sigma"]
    m, S = af.mass_kg, af.wing_area_m2
    drag = lambda V: fs._drag(m, af.cd0, af.aspect_ratio, af.oswald, S, V, rho)
    excess = lambda V: dr.max_thrust(V, rho) - drag(V)
    lo, hi = 12.0, 60.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if excess(mid) > 0 else (lo, mid)
    v_max = lo
    v_ne = V_NE_EAS / math.sqrt(sig)
    Vc = v_cruise_eas / math.sqrt(sig)
    t, V = 0.0, 15.0
    while V < 25.0 and t < 60.0:
        a_ = excess(V) / m
        if a_ <= 0:
            t = math.inf
            break
        V += a_ * 0.05
        t += 0.05
    roc = max(fs.max_roc(dr, m, af.cd0, af.aspect_ratio, af.oswald, S, v, h) for v in np.linspace(13.0, 26.0, 14))
    V_ld = np.linspace(10.0, 30.0, 81)
    ld = float(np.max(m * G / np.array([drag(v) for v in V_ld])))
    P_cruise, _ = fs.level_power(dr, m, af.cd0, af.aspect_ratio, af.oswald, S, Vc, h)
    return {f"top speed, level, {h:.0f} m [m/s TAS]": min(v_max, v_ne), f"V_NE caps it at {h:.0f} m": v_max > v_ne,
            f"acceleration at {v_launch:g} m/s, {h:.0f} m [m/s²]": excess(v_launch) / m, f"acceleration at cruise {v_cruise_eas:g} m/s EAS, {h:.0f} m [m/s²]": excess(Vc) / m,
            f"15 → 25 m/s TAS at {h:.0f} m [s]": t, f"best climb at {h:.0f} m [m/s]": roc, f"L/D max at {h:.0f} m": ld,
            f"cruise power {v_cruise_eas:g} m/s EAS, {h:.0f} m, battery side [W]": P_cruise}


def compare_with_nisus_plus(heights=(1200.0, 3000.0, 4500.0), *, plan: fs.MissionPlan | None = None, v_launch: float = 17.0, p=None, af_n=None, af_f=None,
                            dr_n=None, dr_f=None) -> pd.DataFrame:
    """NISUS+ against FALCO, each at its design mass on its own drive map: the static thrust at sea level, the level top
    speed, the acceleration at the launch speed and at cruise (full throttle, level), the time from 15 to 25 m/s, the
    best climb, L/D max and the cruise power at the heights, the mission's energy and feasible survey time
    (``mission_energy`` with each aircraft's pack), masses and drag areas. The frame study's ``performance_table``
    arithmetic (the climb from ``max_roc``)."""
    plan = plan or fs.MissionPlan()
    af_n = af_n or ff.airframe(fs.DEFAULT_PACK, h_m=heights[0])
    af_f = af_f or airframe(fsy.DEFAULT_PACK, p, h_m=heights[0])
    dr_n = dr_n or fs.drive()
    dr_f = dr_f or fsy.drive()
    cols = {}
    for name, af, dr, pk in (("Nisus+ Zero", af_n, dr_n, fs.pack(fs.DEFAULT_PACK)), ("Falco-Zero", af_f, dr_f, fsy.pack(fsy.DEFAULT_PACK))):
        row = {"mass [kg]": af.mass_kg, "wing area [m²]": af.wing_area_m2, "Cd0 (build-up + installation)": af.cd0, "drag area Cd0·S [m²]": af.cd0 * af.wing_area_m2,
               "Oswald e": af.oswald, "propeller": dr.motor_name if hasattr(dr, "motor_name") else "", "pack": pk.name, "pack energy [Wh]": pk.energy_wh,
               "static thrust, sea level [N]": dr.max_thrust(0.0, RHO0), "static T/W": dr.max_thrust(0.0, RHO0) / (af.mass_kg * G)}
        for h in heights:
            row.update(_performance(af, dr, h, v_launch=v_launch, v_cruise_eas=plan.V_cruise_eas))
        me = fs.mission_energy(pk, dr, airframe_dict(af), plan)
        row.update({"mission energy used [Wh]": me["E_used_wh"], "mission: climb energy [Wh]": me["E_climb_wh"], "mission: climb time [s]": me["t_climb_s"],
                    "mission: survey power [W]": me["P_survey_w"], "mission: energy left [% avail.]": me["E_left_pct_available"],
                    "mission: feasible survey [min]": me["t_survey_feasible_s"] / 60, "mission: reserve kept": me["reserve_kept"]})
        cols[name] = row
    df = pd.DataFrame(cols)
    num = df.apply(pd.to_numeric, errors="coerce")
    ok = num.notna().all(axis=1) & (num["Nisus+ Zero"].abs() > 1e-12)
    df["FALCO vs NISUS+ [%]"] = np.where(ok, 100 * (num["Falco-Zero"] - num["Nisus+ Zero"]) / num["Nisus+ Zero"].abs(), np.nan)
    return df


__all__ = ["NAME", "aero", "crow_increments", "crow_table", "derivatives", "cg_range", "launch_check", "installation", "installation_table", "airframe",
           "compare_with_nisus_plus", "V_NE_EAS", "V_FE_EAS", "V_CRUISE_EAS", "N_STRUCTURAL", "CROW", "cl_max", "trim_crow", "coefficients", "trim",
           "airframe_dict", "DriveUnit", "envelope", "envelope_vs_altitude", "ceiling", "climb_strategy", "glide_state", "descent_table", "best_descent",
           "regen_table", "downdraft_escape", "launch_table", "operating_limits", "stall_table", "gust_response", "pull_up"]
