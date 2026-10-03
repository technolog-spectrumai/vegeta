"""MERLIN — the race to the fire, the smoke plume, the sampling mission and its movie (notebook 26).

Reduced models, honest about what they are, built on the tools the other notebooks already validated:

- ``Airframe``: a parabolic drag polar ``Cd = Cd0 + Cl^2 / (pi AR e)`` (Cd0 from ``merlin.drag_buildup`` or the CFD).
- ``Unit``: a propulsor as two tables over (airspeed, rpm) — the **net** thrust it gives the aircraft and the shaft
  power it takes — built from the existing models: ``open_propeller`` (Boreas BEMT in the installation's effective
  inflow, ``boreas.wake.effective_inflow`` over ``air_propeller.installation_wake``, less the thrust deduction from
  ``air_propeller.installation_drag``: tractor and pusher alike) and ``ducted_fan_unit`` (``boreas.ducted.solve`` less
  ``ducted.nacelle_drag`` and the scrubbing of the annular jet on the fuselage). All units share one battery and one
  electrical power limit (the same ESC and pack), so only the propulsor differs.
- ``race``: launch from the stand → full-power climb to the sampling height → full-power acceleration → the dash at the
  fastest speed that still leaves the energy for the sampling, the flight home at the best-range speed and the reserve;
  the time to the fire and what limits it (thrust or energy). Still air.
- ``Plume``: the smoke column as a Gaussian plume (Briggs buoyant rise, Pasquill–Gifford dispersion, ground
  reflection) carrying CO from the fire's heat release; ``source_estimate`` is the inverse — the CO flux and the fire's
  heat release from the crosswind-integrated concentrations of the passes (is it real, and how big).
- ``fly``: a point-mass fixed-wing flying the mission with wind (heading and bank limits, climb from the excess power,
  the race's speeds), the sensor reading the plume → ``Episode``; ``render_movie`` draws it with the smoke as particles.

Units SI; heights above ground (AGL) unless said otherwise; x east, y north.

Promoted from ``notebooks/designs/merlin_flight.py`` as it was proven there; the notebook copy may move on.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

G = 9.81
RHO = 1.225


# ------------------------------------------------------------------------------------------------- the airframe
@dataclass
class Airframe:
    name: str
    mass_kg: float
    wing_area_m2: float
    aspect_ratio: float
    cd0: float
    oswald: float = 0.8
    cl_max: float = 1.2

    @property
    def k(self):
        return 1.0 / (math.pi * self.aspect_ratio * self.oswald)

    def drag(self, V, rho=RHO, n=1.0):
        """Drag [N] at airspeed V in flight at load factor n (lift = n W)."""
        V = np.maximum(np.asarray(V, float), 1e-3)
        q = 0.5 * rho * V ** 2
        cl = n * self.mass_kg * G / (q * self.wing_area_m2)
        return q * self.wing_area_m2 * (self.cd0 + self.k * cl ** 2)

    def stall_speed(self, rho=RHO, n=1.0):
        return math.sqrt(2 * n * self.mass_kg * G / (rho * self.wing_area_m2 * self.cl_max))


# ------------------------------------------------------------------------------------------------- propulsors
@dataclass
class Unit:
    """A propulsor tabulated over airspeed ``V`` (m/s) and ``rpm``: ``thrust[i, j]`` net thrust [N] on the aircraft,
    ``power[i, j]`` shaft power [W]. Full throttle is the rpm at which the battery's electrical limit
    ``max_electrical_w`` times ``drive_efficiency`` is reached, or ``rpm_max``, whichever comes first."""
    name: str
    V: np.ndarray
    rpm: np.ndarray
    thrust: np.ndarray
    power: np.ndarray
    mass_kg: float
    max_electrical_w: float
    drive_efficiency: float = 0.85
    notes: str = ""

    def _row(self, V):
        V = float(np.clip(V, self.V[0], self.V[-1]))
        i = min(int(np.searchsorted(self.V, V, side="right")) - 1, len(self.V) - 2)
        t = (V - self.V[i]) / (self.V[i + 1] - self.V[i])
        return (1 - t) * self.thrust[i] + t * self.thrust[i + 1], (1 - t) * self.power[i] + t * self.power[i + 1]

    def full(self, V):
        """(net thrust [N], electrical power [W], rpm) at full throttle and airspeed V."""
        T, P = self._row(V)
        p_max = self.max_electrical_w * self.drive_efficiency
        if P[-1] <= p_max:
            return float(T[-1]), float(P[-1] / self.drive_efficiency), float(self.rpm[-1])
        n = float(np.interp(p_max, P, self.rpm))
        return float(np.interp(n, self.rpm, T)), self.max_electrical_w, n

    def electrical_power(self, V, T_net):
        """Electrical power [W] for a net thrust ``T_net`` at V; ``inf`` if more than full throttle gives."""
        T, P = self._row(V)
        if T_net > self.full(V)[0] + 1e-9:
            return math.inf
        n = float(np.interp(T_net, T, self.rpm))           # T rises with rpm on the working branch
        return float(np.interp(n, self.rpm, P)) / self.drive_efficiency


def _tabulate(point, V, rpm):
    T = np.zeros((len(V), len(rpm))); P = np.zeros_like(T)
    for i, v in enumerate(V):
        for j, n in enumerate(rpm):
            T[i, j], P[i, j] = point(n, v)
    # keep thrust monotone in rpm (the working branch) so it can be inverted
    T = np.maximum.accumulate(T, axis=1) + np.arange(len(rpm)) * 1e-9
    return T, P


def open_propeller(name, prop, airfoil, *, wake_fraction=0.0, thrust_deduction=0.0, mass_kg, max_electrical_w,
                   rpm_max, drive_efficiency=0.85, rho=RHO, V=np.linspace(0.0, 70.0, 36), n_rpm=24) -> Unit:
    """An open propeller as a ``Unit``: Boreas BEMT at the effective inflow ``V (1 - w)`` (``wake_fraction`` w, from
    ``boreas.wake.effective_inflow``), net thrust ``T (1 - t)`` (``thrust_deduction`` t, from
    ``air_propeller.installation_drag``)."""
    from vegeta import boreas

    def point(n, v):
        op = boreas.solve(prop, airfoil, n, max(v * (1 - wake_fraction), 0.0), rho)
        return op.thrust * (1 - thrust_deduction), op.power

    rpm = np.linspace(0.2 * rpm_max, rpm_max, n_rpm)
    T, P = _tabulate(point, V, rpm)
    return Unit(name, np.asarray(V, float), rpm, T, P, mass_kg, max_electrical_w, drive_efficiency,
                f"BEMT, w = {wake_fraction:.3f}, t = {thrust_deduction:.3f}")


def ducted_fan_unit(name, fan, airfoil, *, mass_kg, max_electrical_w, rpm_max, scrub_area_m2=0.0, scrub_length_m=0.3,
                    drive_efficiency=0.85, rho=RHO, nu=1.5e-5, V=np.linspace(0.0, 70.0, 36), n_rpm=16) -> Unit:
    """The EDF as a ``Unit``: ``boreas.ducted.solve`` (unit thrust) less the nacelle's outside friction
    (``ducted.nacelle_drag``) and the jet scrubbing the fuselage behind the nozzle: turbulent friction on
    ``scrub_area_m2`` at the jet's extra dynamic pressure ``1/2 rho (V_exit^2 - V^2)`` (no mixing: an upper bound)."""
    from vegeta.boreas import ducted

    def point(n, v):
        e = ducted.solve(fan, airfoil, n, v, rho)
        vj = max(e.exit_velocity, v)
        cf = 0.455 / math.log10(max(vj * scrub_length_m / nu, 1e4)) ** 2.58
        scrub = cf * scrub_area_m2 * 0.5 * rho * (vj ** 2 - v ** 2)
        return e.thrust - ducted.nacelle_drag(fan, v, rho, nu) - scrub, e.power

    rpm = np.linspace(0.3 * rpm_max, rpm_max, n_rpm)
    T, P = _tabulate(point, V, rpm)
    return Unit(name, np.asarray(V, float), rpm, T, P, mass_kg, max_electrical_w, drive_efficiency,
                f"boreas.ducted, duct_loss {fan.duct_loss:.3f}, scrub area {scrub_area_m2:.3f} m^2")


# ------------------------------------------------------------------------------------------------- the three MERLINs
SECTION_KW = dict(name="thin cambered section", cl_alpha=2 * math.pi * 0.9, alpha0_deg=-3.0, cl_max=1.1, cd0=0.018, k=0.04)
# masses [kg] shared by all three (notebook 26's mass budget overrides them) and each propulsion unit's own
COMMON_MASS_KG = {"airframe (printed shells)": 0.60, "battery 6S 8000 mAh": 1.05, "flight controller + GPS": 0.035,
                  "servos (4x)": 0.048, "receiver + telemetry": 0.045, "gas sensor (CO/CO2/PM) + pump": 0.12}
UNIT_MASS_KG = {"edf": 0.405, "tractor": 0.273, "pusher": 0.273}      # EDF 320 g + ESC 85 g; motor 190 + prop 18 + ESC 65 g
EXTRA_CD_AREA_M2 = 0.0015          # landing skid, sensor intake and exhaust, antennas, gaps: about 0.005 on Cd0


def installation_pod(kind, p):
    """``air_propeller.PropPod`` parameters standing in for MERLIN's fuselage (the pod) and the surface next to the
    propeller (the pylon): the wing for the tractor (its leading edge ``nose_length`` behind the disc), the fin for the
    pusher (``pusher_gap`` ahead of it; the two tail-plane halves are added by ``installation``)."""
    from . import merlin as m
    p = m.Merlin().resolve(**dict(p, propulsion=kind))
    pod = dict(layout=kind, pod_diameter=p["fuselage_diameter"], pod_length=m.wetted_areas(p)["fuselage_length"] * 1000,
               front_length=p["fairing_length"], rear_length=p["tail_cone"], motor_end_diameter=p["tail_end_diameter"] if kind == "pusher" else p["motor_diameter"],
               hub_diameter=p["hub_diameter"], hub_height=p["hub_height"], gap=p["prop_gap"])
    if kind == "tractor":
        pod.update(pylon_chord=p["root_chord"], pylon_thickness=p["thickness"], pylon_height=p["span"] / 2,
                   pylon_gap=p["nose_length"] - p["hub_height"] / 2)
    else:
        pod.update(pylon_chord=p["tail_chord"], pylon_thickness=p["tail_thickness"] / p["tail_chord"], pylon_height=p["fin_height"],
                   pylon_gap=p["pusher_gap"])
    return pod, p


def installation(kind, p, prop, airfoil, speed, thrust, rho=RHO):
    """Effective wake fraction w and thrust deduction t of an open propeller on MERLIN at one flight point
    (``air_propeller.installation_wake`` + ``boreas.wake.effective_inflow``; ``air_propeller.installation_drag`` with the
    blade-element loading). The pusher's inflow has the fin's and both tail-plane halves' wakes."""
    from . import air_propeller as ap
    from vegeta import boreas
    from vegeta.boreas import wake
    pod, p = installation_pod(kind, p)
    R_mm = prop.radius * 1000
    W = ap.installation_wake(pod, R_mm, speed)
    if kind == "pusher":
        pw = ap.pylon_wake(dict(pod, pylon_height=p["tail_span"] / 2), R_mm, r_frac=tuple(W.r_frac), n_phi=len(W.phi_deg)).w
        q = len(W.phi_deg) // 4
        W = wake.WakeField(W.r_frac, W.phi_deg, W.w + np.roll(pw, q, axis=1) + np.roll(pw, -q, axis=1), W.source + " + both tail-plane halves")
    op0 = boreas.rpm_for_thrust(prop, airfoil, thrust, speed, rho)
    w, op = wake.effective_inflow(prop, airfoil, op0.rpm, speed, W, rho)
    t = ap.installation_drag(pod, R_mm, speed, thrust, rho=rho, loading=(op.r, op.dT_dr))["t"]
    return {"w": float(w), "t": float(t), "wake": W, "pod": pod}


DESIGN_JSON = Path(__file__).resolve().parents[1] / "data" / "merlin_design.json"


def load_design(path=None):
    """Notebook 26's export of the aircraft (``data/merlin_design.json``): its parameters, masses, polar corrections and the
    mission settings; None when it has not been written."""
    import json
    path = Path(path or DESIGN_JSON)
    return json.loads(path.read_text()) if path.exists() else None


def build_merlins(p=None, *, max_electrical_w=1900.0, drive_efficiency=0.85, pitches_in=(6.0, 8.0, 10.0), prop_diameter_in=10.0,
                  blades=2, edf=None, distance_m=20000.0, mission=None, common_mass_kg=None, unit_mass_kg=None, library=None,
                  design=None, **_ignored):
    """The three MERLINs, each an ``Airframe`` with its ``Unit``, on one battery and one electrical power limit — the airframe
    from notebook 26's design export (``design``, or the design file's defaults), the propulsors from the propulsor
    libraries (``propulsor_maps``), nothing solved here:

    - ``edf``: the fan ``edf`` picks in the library (default notebook 25b's 90 mm, 12-blade, 7-stator fan with the catalogue
      loss, one stage), its map less the annular jet scrubbing MERLIN's fairing (``propulsor_maps.unit``);
    - ``tractor`` / ``pusher``: the ``prop_diameter_in`` propeller of ``blades`` blades with the pitch among ``pitches_in``
      that reaches a fire ``distance_m`` away first, its free-stream map at MERLIN's effective wake and thrust deduction.

    Returns ``{kind: {"airframe", "unit", "info"}}``; ``info`` carries the fan and propeller objects for reference."""
    from . import propulsor_maps as pm
    lib = pm.load() if library is None else (pm.load(library) if not isinstance(library, dict) else library)
    design = dict(design or {})
    if common_mass_kg is not None:
        design["common_mass_kg"] = dict(common_mass_kg)
    p = dict(design.get("merlin_params", {}), **(p or {}))
    mission = Mission() if mission is None else mission
    edf = dict(diameter_mm=90.0, pitch_ratio=1.78, exit_area_ratio=0.9, quality="catalogue", stages=1, **(edf or {}))

    def airframe(kind, entry):
        um = unit_mass_kg[kind] if unit_mass_kg else None
        return pm.merlin_airframe(kind, max_electrical_w, entry, merlin=p, design=design, unit_mass=um)

    out = {}
    hits = pm.find(lib, kind="edf", **edf)
    if hits:
        e = hits[0]
        af = airframe("edf", e)
        u = pm.unit(e, max_electrical_w, drive_efficiency=drive_efficiency, merlin=p)
        u.mass_kg = af.mass_kg - sum(design.get("common_mass_kg", COMMON_MASS_KG).values())
        fan, hp, hg = pm.fan_object(e["diameter_mm"], e["pitch_ratio"], e["exit_area_ratio"], e["duct_loss"], e["stages"])
        out["edf"] = {"airframe": af, "unit": u, "info": {"entry": e, "duct_loss": e["duct_loss"], "fan": fan, "housing": hg,
                                                          "scrub_area_m2": pm.edf_scrub_area(e, p)}}
    for kind in ("tractor", "pusher"):
        best, rows = None, []
        for pin in pitches_in:
            hits = pm.find(lib, kind="propeller", diameter_in=float(prop_diameter_in), pitch_ratio=round(pin / prop_diameter_in, 6), blades=blades)
            if not hits:
                continue
            e = hits[0]
            af = airframe(kind, e)
            u = pm.unit(e, max_electrical_w, layout=kind, drive_efficiency=drive_efficiency)
            u.mass_kg = af.mass_kg - sum(design.get("common_mass_kg", COMMON_MASS_KG).values())
            r = race(af, u, distance_m, mission)
            t = r["time_to_fire_s"] if r["reachable"] else math.inf
            inst = e["installation"][kind]
            rows.append({"pitch [in]": pin, "w": inst["w"], "t": inst["t"], "top speed [m/s]": performance(af, u)["v_top"],
                         f"time to the fire at {distance_m / 1000:.0f} km [s]": t})
            if best is None or t < best[0]:
                best = (t, af, u, e)
        if best is None:
            continue
        _, af, u, e = best
        out[kind] = {"airframe": af, "unit": u, "info": {"entry": e, "prop": pm.prop_object(e["diameter_in"], e["pitch_ratio"], e["blades"]),
                                                          "w": e["installation"][kind]["w"], "t": e["installation"][kind]["t"],
                                                          "pitch_study": rows, "rpm_max": e["rpm_max"]}}
    if not out:
        raise ValueError("no propulsor in the libraries: run notebooks 25 / 25b (their maps sections) or scenarios/propulsor_maps.py")
    return out


def mf_airframe(kind, mass, buildup, cd0):
    return Airframe(f"MERLIN ({kind})", mass, buildup["planform"], buildup["aspect_ratio"], cd0)


# ------------------------------------------------------------------------------------------------- the race
DISTANCES_KM = (5.0, 8.0, 10.0, 20.0, 30.0)        # the fires the race is run to (notebook, scenario, tests)


@dataclass
class Mission:
    sampling_agl_m: float = 150.0
    sampling_s: float = 180.0
    sampling_load_factor: float = 1.25      # turns between the passes
    launch_speed: float = 12.0              # off the stand
    battery_wh: float = 177.6               # 6S 8000 mAh
    usable_fraction: float = 0.85
    reserve_fraction: float = 0.15          # of the usable energy, kept for the landing


def performance(af: Airframe, unit: Unit, rho=RHO, V=None) -> dict:
    """Speed polar of one aircraft with one unit: full thrust, drag, rate of climb, the top speed, the best-climb and
    best-range speeds and the electrical power in level flight."""
    V = np.linspace(max(af.stall_speed(rho) * 1.1, 6.0), unit.V[-1], 120) if V is None else np.asarray(V, float)
    full = np.array([unit.full(v) for v in V])
    T, Pf = full[:, 0], full[:, 1]
    D = af.drag(V, rho)
    W = af.mass_kg * G
    roc = (T - D) * V / W
    P_lvl = np.array([unit.electrical_power(v, d) for v, d in zip(V, D)])
    ok = T >= D
    v_top = float(V[ok].max()) if ok.any() else math.nan
    if ok.any() and ok[-1]:                                   # still accelerating at the table's end
        v_top = float(V[-1])
    per_m = np.where(np.isfinite(P_lvl), P_lvl / V, np.inf)
    return {"V": V, "T_full": T, "P_full": Pf, "D": D, "roc": roc, "P_level": P_lvl, "v_top": v_top,
            "v_climb": float(V[np.argmax(roc)]), "roc_max": float(roc.max()), "v_range": float(V[np.argmin(per_m)]),
            "wh_per_km_best": float(per_m.min() / 3.6), "v_stall": af.stall_speed(rho)}


def race(af: Airframe, unit: Unit, distance_m: float, mission: Mission = Mission(), rho=RHO, *,
         fire_elevation_m: float = 0.0, approach_sink: float = 2.0) -> dict:
    """The fastest legal flight to a fire ``distance_m`` away, and back.

    Out: climb at the best-climb speed and full power to the sampling height above the launch site, accelerate at full
    power to the dash speed, dash. A fire ``fire_elevation_m`` above the launch site (on a mountain) is reached by
    climbing gently all along the dash, at the constant angle that gains that height over the dash's distance: the
    thrust then also carries ``W sin(gamma)``, so the top speed falls and the energy rises by ``m g dh``. The dash speed
    is the highest (up to the top speed on that slope) that keeps the energy for the sampling, the flight home and the
    reserve.

    Over the fire: ``mission.sampling_s`` of passes (the data go out by radio: the **reach time** is when the first
    reading comes in). Home: as fast as the energy left allows (between the best-range speed and the top speed, keeping
    the reserve), descending the same height (the thrust only needs ``D - W sin(gamma)``, a glide when that is
    negative), then an approach from the sampling height at ``approach_sink`` m/s and 1.3 stall speed to a belly
    landing at the launch site.

    Returns the reach time (``time_to_fire_s``), the return time (leaving the fire to touchdown, ``return_time_s``),
    the landing time (touchdown after launch, ``landing_time_s``), the dash speed and its slope, what limited it and
    the energy per phase. Air density is ``rho`` throughout (a 600 m mountain thins the air by ~6 %: not modelled)."""
    pf = performance(af, unit, rho)
    W = af.mass_kg * G
    v_c = pf["v_climb"]
    T_c, P_c, _ = unit.full(v_c)
    roc = (T_c - af.drag(v_c, rho)) * v_c / W
    if roc <= 0:
        return {"reachable": False, "why": "cannot climb"}
    t_climb = mission.sampling_agl_m / roc
    x_climb = v_c * t_climb
    e_climb = P_c * t_climb / 3600
    t_l, x_l, e_l = _accelerate(af, unit, mission.launch_speed, v_c, rho)
    x_out = distance_m - x_l - x_climb                     # the acceleration and the dash
    if x_out <= 0:
        return {"reachable": False, "why": "the fire is closer than the climb-out"}
    sin_out = fire_elevation_m / math.hypot(x_out, fire_elevation_m)
    # the sampling
    v_s = 1.3 * af.stall_speed(rho, mission.sampling_load_factor)
    P_s = unit.electrical_power(v_s, float(af.drag(v_s, rho, mission.sampling_load_factor)))
    e_sample = P_s * mission.sampling_s / 3600
    # home: cruise at the best-range speed, descending fire_elevation_m, then the approach
    v_r, v_app = pf["v_range"], 1.3 * af.stall_speed(rho)
    t_app = mission.sampling_agl_m / approach_sink
    x_app = v_app * t_app
    x_home = max(distance_m - x_app, 0.0)
    sin_home = fire_elevation_m / math.hypot(x_home, fire_elevation_m) if x_home > 0 else 0.0

    def power(v, T):
        return 0.0 if T <= 0 else unit.electrical_power(v, T)

    P_home = power(v_r, float(af.drag(v_r, rho)) - W * sin_home)
    P_app = power(v_app, float(af.drag(v_app, rho)) - W * approach_sink / v_app)
    if not (math.isfinite(P_home) and math.isfinite(P_app)):
        return {"reachable": False, "why": "cannot fly home"}
    t_home = x_home / v_r
    e_home = (P_home * t_home + P_app * t_app) / 3600
    budget = mission.battery_wh * mission.usable_fraction * (1 - mission.reserve_fraction)
    # the top speed on the slope: full thrust = drag + the weight's share along the path
    V = pf["V"]
    ok = pf["T_full"] >= pf["D"] + W * sin_out
    if not ok.any():
        return {"reachable": False, "why": f"cannot hold a {math.degrees(math.asin(sin_out)):.1f} deg climb at any speed"}
    v_top = float(V[ok].max())

    def out_leg(v_d):
        t_a, x_a, e_a = _accelerate(af, unit, v_c, v_d, rho, max_distance=x_out, climb_sin=sin_out)
        x_dash = max(x_out - x_a, 0.0)
        P_d = unit.electrical_power(v_d, float(af.drag(v_d, rho)) + W * sin_out)
        t = t_l + t_climb + t_a + x_dash / v_d
        e = e_l + e_climb + e_a + P_d * x_dash / v_d / 3600
        return t, e, {"accelerate": e_a, "dash": P_d * x_dash / v_d / 3600}

    v_hi = 0.995 * v_top
    v_lo = min(v_r, v_hi)
    if out_leg(v_lo)[1] + e_sample + e_home > budget:
        return {"reachable": False, "why": "not enough energy even at the best-range speed", "distance_km": distance_m / 1000}
    if out_leg(v_hi)[1] + e_sample + e_home <= budget:
        v_d, limit = v_hi, "thrust (top speed)"
    else:
        lo, hi = v_lo, v_hi
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if out_leg(mid)[1] + e_sample + e_home <= budget else (lo, mid)
        v_d, limit = lo, "energy (the way home and the reserve)"
    t, e_out, parts = out_leg(v_d)
    # home as fast as the energy left allows (the reach comes first; the return takes what is left, keeping the reserve)
    e_left = budget - e_out - e_sample - P_app * t_app / 3600
    ok_h = pf["T_full"] >= pf["D"] - W * sin_home
    v_hmax = 0.995 * float(V[ok_h].max()) if ok_h.any() else v_r
    e_at = lambda v: power(v, float(af.drag(v, rho)) - W * sin_home) * x_home / v / 3600
    v_ret = v_r
    if v_hmax > v_r and e_at(v_hmax) <= e_left:
        v_ret = v_hmax
    elif v_hmax > v_r:
        lo, hi = v_r, v_hmax
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if e_at(mid) <= e_left else (lo, mid)
        v_ret = lo
    t_home = x_home / v_ret
    e_home = e_at(v_ret) + P_app * t_app / 3600
    t_return = t_home + t_app
    return {"reachable": True, "distance_km": distance_m / 1000, "fire_elevation_m": fire_elevation_m,
            "time_to_fire_s": t, "sampling_s": mission.sampling_s, "return_time_s": t_return,
            "landing_time_s": t + mission.sampling_s + t_return, "dash_speed": v_d, "limit": limit,
            "dash_climb_deg": math.degrees(math.asin(sin_out)), "climb_speed": v_c, "climb_time_s": t_climb,
            "top_speed": v_top, "range_speed": v_r, "return_speed": v_ret, "sampling_speed": v_s, "approach_s": t_app,
            "energy_out_wh": e_out, "energy_sampling_wh": e_sample, "energy_home_wh": e_home, "budget_wh": budget,
            "home_time_s": t_home, "energy_parts_wh": dict(launch=e_l, climb=e_climb, **parts)}


NAMES = {"edf": "ducted fan", "tractor": "tractor propeller", "pusher": "pusher propeller"}


def race_table(merlins, distances_km=DISTANCES_KM, mission: Mission = Mission(), *, fire_elevation_m=0.0):
    """The race for every aircraft of ``build_merlins`` and every distance, as a table (pandas) indexed by
    (distance, propulsor): reach, return and landing times [min], the dash and return speeds, the dash's climb angle,
    what limited the dash and the energy used. The reach is what counts when the data go out by radio."""
    import pandas as pd
    rows = {}
    for d in distances_km:
        for k, v in merlins.items():
            r = race(v["airframe"], v["unit"], d * 1000, mission, fire_elevation_m=fire_elevation_m)
            key = (f"{d:g} km", NAMES.get(k, k))
            if not r["reachable"]:
                rows[key] = {"reach [min]": math.nan, "limited by": r["why"]}
                continue
            rows[key] = {"reach [min]": r["time_to_fire_s"] / 60, "return [min]": r["return_time_s"] / 60,
                         "landing [min]": r["landing_time_s"] / 60, "dash [m/s]": r["dash_speed"],
                         "dash climb [deg]": r["dash_climb_deg"], "return [m/s]": r["return_speed"], "limited by": r["limit"],
                         "energy used [Wh]": r["energy_out_wh"] + r["energy_sampling_wh"] + r["energy_home_wh"],
                         "of [Wh]": r["budget_wh"]}
    return pd.DataFrame(rows).T


def _accelerate(af, unit, v0, v1, rho, n=60, max_distance=math.inf, climb_sin=0.0):
    """Acceleration at full power from v0 to v1 (on a path climbing at ``asin(climb_sin)``): (time, distance, energy
    Wh); zero if v1 <= v0. It stops after ``max_distance`` (the fire reached before the dash speed: a short race)."""
    if v1 <= v0 or max_distance <= 0:
        return 0.0, 0.0, 0.0
    vs = np.linspace(v0, v1, n)
    full = np.array([unit.full(v) for v in vs])
    a = (full[:, 0] - af.drag(vs, rho) - af.mass_kg * G * climb_sin) / af.mass_kg
    a = np.maximum(a, 1e-3)
    dt = np.diff(vs) / (0.5 * (a[1:] + a[:-1]))
    vm = 0.5 * (vs[1:] + vs[:-1])
    Pm = 0.5 * (full[1:, 1] + full[:-1, 1])
    x = np.cumsum(vm * dt)
    if x[-1] > max_distance:                                  # cut the last step at the distance
        k = int(np.searchsorted(x, max_distance))
        f = (max_distance - (x[k - 1] if k else 0.0)) / (vm[k] * dt[k])
        dt = np.r_[dt[:k], f * dt[k]]
        vm, Pm = vm[:k + 1], Pm[:k + 1]
    return float(dt.sum()), float((vm * dt).sum()), float((Pm * dt).sum() / 3600)


# ------------------------------------------------------------------------------------------------- the smoke plume
@dataclass
class Plume:
    """A Gaussian plume from a fire at ``source_xy`` (m) on the ground: heat release ``heat_mw`` sets Briggs's
    buoyancy flux ``F = 8.8 Q[MW]`` m^4/s^3 and the final rise; Pasquill–Gifford dispersion (Briggs's rural fits) for
    the ``stability`` class with an initial spread from the fire's size; CO emitted at ``ef_co`` kg per kg of fuel
    burnt (heat of combustion ``heat_of_combustion`` J/kg); the wind ``wind_speed`` from ``wind_from_deg`` carries it.
    ``concentration_ppm`` is the CO mixing ratio above the ``background_ppm``."""
    source_xy: tuple = (20000.0, 0.0)
    heat_mw: float = 40.0
    wind_speed: float = 5.0
    wind_from_deg: float = 270.0          # meteorological: from the west, so the smoke drifts east (+x)
    fire_size_m: float = 60.0
    stability: str = "C"
    ef_co: float = 0.10                   # kg CO per kg of fuel (forest fires: 0.06-0.13)
    heat_of_combustion: float = 18.0e6    # J/kg dry fuel
    background_ppm: float = 0.12

    _PG = {"B": (0.16, 0.12, 0.0), "C": (0.11, 0.08, 0.0002), "D": (0.08, 0.06, 0.0015)}

    @property
    def downwind(self):
        a = math.radians(self.wind_from_deg)
        return np.array([-math.sin(a), -math.cos(a)])        # unit vector the smoke travels along

    @property
    def wind_vector(self):
        return self.wind_speed * self.downwind

    @property
    def co_kg_s(self):
        return self.ef_co * self.heat_mw * 1e6 / self.heat_of_combustion

    def rise(self, x):
        F = 8.8 * self.heat_mw
        xf = 49 * F ** 0.625 if F < 55 else 119 * F ** 0.4
        xx = np.clip(np.asarray(x, float), 0.0, xf)
        return 1.6 * F ** (1 / 3) * np.maximum(xx, 1.0) ** (2 / 3) / self.wind_speed

    def sigmas(self, x):
        sy0 = self.fire_size_m / 4.3
        ay, az, bz = self._PG[self.stability]
        x = np.maximum(np.asarray(x, float), 1.0)
        sy = np.hypot(ay * x / np.sqrt(1 + 0.0001 * x), sy0)
        sz = np.hypot(az * x / (np.sqrt(1 + bz * x) if bz else 1.0), sy0)
        return sy, sz

    def local(self, x, y):
        """World (x, y) -> (downwind distance, crosswind offset) from the fire."""
        d = self.downwind
        dx, dy = np.asarray(x, float) - self.source_xy[0], np.asarray(y, float) - self.source_xy[1]
        return dx * d[0] + dy * d[1], -dx * d[1] + dy * d[0]

    def concentration_ppm(self, x, y, z_agl):
        """CO above background [ppm] at world (x, y) and height above ground."""
        s, c = self.local(x, y)
        z = np.asarray(z_agl, float)
        sy, sz = self.sigmas(s)
        H = self.rise(s)
        C = (self.co_kg_s / (2 * np.pi * self.wind_speed * sy * sz) * np.exp(-0.5 * (c / sy) ** 2)
             * (np.exp(-0.5 * ((z - H) / sz) ** 2) + np.exp(-0.5 * ((z + H) / sz) ** 2)))
        ppm = C / RHO * (28.97 / 28.01) * 1e6
        return np.where(np.asarray(s) > 0, ppm, 0.0)

    def cwic(self, s, z, Q=None, H=None, sz=None):
        """Crosswind-integrated concentration [kg/m^2] at downwind distance s, height z (the model, or with Q, H, sz)."""
        Q = self.co_kg_s if Q is None else Q
        H = self.rise(s) if H is None else H
        sz = self.sigmas(s)[1] if sz is None else sz
        return Q / (math.sqrt(2 * math.pi) * self.wind_speed * sz) * (np.exp(-0.5 * ((z - H) / sz) ** 2) + np.exp(-0.5 * ((z + H) / sz) ** 2))

    def particles(self, n, t=0.0, seed=0, max_downwind=3000.0):
        """``n`` smoke particles (world x, y, z AGL) for drawing at time ``t`` [s]: released uniformly in time, carried
        downwind at the wind speed, spread by the plume's sigmas; recycled so the column looks steady."""
        rng = np.random.default_rng(seed)
        age = (rng.uniform(0, max_downwind / self.wind_speed, n) + t) % (max_downwind / self.wind_speed)
        s = age * self.wind_speed + 5.0
        sy, sz = self.sigmas(s)
        c = rng.normal(0, 1, n) * sy
        z = np.abs(self.rise(s) * np.minimum(1.0, (s / 60.0)) + rng.normal(0, 1, n) * sz)
        d = self.downwind
        x = self.source_xy[0] + s * d[0] - c * d[1]
        y = self.source_xy[1] + s * d[1] + c * d[0]
        return np.column_stack([x, y, z]), s


def source_estimate(plume: Plume, passes, wind_speed=None) -> dict:
    """The inverse: from the sampling passes (each a dict with world ``x``, ``y``, ``z_agl`` arrays and ``ppm``
    above background) to the CO flux and the fire's heat release. Each pass gives a crosswind-integrated
    concentration at its height; a Gaussian with ground reflection fitted over the heights gives Q, the plume height
    and sigma_z; the heat release is ``Q hc / ef_co``. The wind speed is the measured one (default: the plume's)."""
    from scipy.optimize import curve_fit
    u = plume.wind_speed if wind_speed is None else wind_speed
    zs, cw, ss = [], [], []
    for p in passes:
        s, c = plume.local(p["x"], p["y"])
        o = np.argsort(c)
        kg_m3 = np.asarray(p["ppm"])[o] * 1e-6 * RHO * (28.01 / 28.97)
        cw.append(float(np.trapezoid(kg_m3, c[o])))
        zs.append(float(np.mean(p["z_agl"])))
        ss.append(float(np.mean(s)))
    zs, cw = np.array(zs), np.array(cw)

    def f(z, Q, H, sz):
        return Q / (math.sqrt(2 * math.pi) * u * sz) * (np.exp(-0.5 * ((z - H) / sz) ** 2) + np.exp(-0.5 * ((z + H) / sz) ** 2))

    p0 = (max(cw.max(), 1e-9) * u * 2.5 * 60, zs[np.argmax(cw)], 60.0)
    try:
        (Q, H, sz), _ = curve_fit(f, zs, cw, p0=p0, bounds=([0, 0, 5], [np.inf, 2000, 1000]), maxfev=20000)
    except Exception:                                           # too few passes: flux through the sampled band only
        Q, H, sz = float(np.trapezoid(cw, zs) * u) if len(zs) > 1 else float("nan"), float("nan"), float("nan")
    heat = Q * plume.heat_of_combustion / plume.ef_co / 1e6
    return {"co_kg_s": float(Q), "plume_height_m": float(H), "sigma_z_m": float(sz), "heat_mw": float(heat),
            "pass_heights_m": zs, "cwic_kg_m2": cw, "downwind_m": float(np.mean(ss)),
            "peak_ppm": float(max(np.max(p["ppm"]) for p in passes))}


# ------------------------------------------------------------------------------------------------- the forest
@dataclass(frozen=True)
class Forest:
    """Gentle wooded hills: height [m ASL] = base + a few long waves. The launch site is at (0, 0)."""
    base: float = 300.0
    waves: tuple = ((35.0, 4200.0, 0.0, 0.3), (22.0, 2600.0, 1.1, 1.4), (14.0, 1500.0, 2.0, 0.7))

    def height(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        h = np.full(np.broadcast(x, y).shape, self.base)
        for a, lam, ph, ang in self.waves:
            h = h + a * np.sin(2 * np.pi * (x * math.cos(ang) + y * math.sin(ang)) / lam + ph)
        return h


# ------------------------------------------------------------------------------------------------- the mission
@dataclass
class Episode:
    t: np.ndarray
    pos: np.ndarray            # world x, y, altitude ASL
    vel: np.ndarray            # ground velocity
    airspeed: np.ndarray
    heading: np.ndarray        # rad, of the body (air-relative)
    bank: np.ndarray
    thrust: np.ndarray
    power: np.ndarray
    energy_wh: np.ndarray
    agl: np.ndarray
    ppm: np.ndarray
    phase: list
    passes: list = field(default_factory=list)
    events: list = field(default_factory=list)
    unit: str = ""

    def phase_table(self):
        import pandas as pd
        rows, start = [], 0
        for i in range(1, len(self.t) + 1):
            if i == len(self.t) or self.phase[i] != self.phase[start]:
                sl = slice(start, i)
                d = np.sum(np.linalg.norm(np.diff(self.pos[sl, :2], axis=0), axis=1)) if i - start > 1 else 0.0
                rows.append({"phase": self.phase[start], "start [s]": self.t[start], "duration [s]": self.t[i - 1] - self.t[start],
                             "ground distance [km]": d / 1000, "mean airspeed [m/s]": float(np.mean(self.airspeed[sl])),
                             "energy [Wh]": self.energy_wh[i - 1] - self.energy_wh[start], "peak CO [ppm]": float(np.max(self.ppm[sl]))})
                start = i
        return pd.DataFrame(rows).set_index("phase")

    def summary(self):
        i_fire = next((i for i, ph in enumerate(self.phase) if ph.startswith("sampling")), len(self.t) - 1)
        return {"unit": self.unit, "flight_time_s": float(self.t[-1]), "time_to_first_pass_s": float(self.t[i_fire]),
                "energy_wh": float(self.energy_wh[-1]), "max_airspeed": float(self.airspeed.max()),
                "peak_ppm": float(self.ppm.max()), "passes": len(self.passes), "events": list(self.events)}


def fly(af: Airframe, unit: Unit, forest: Forest, plume: Plume, *, race_result: dict, mission: Mission = Mission(),
        sample_downwind_m=600.0, pass_heights_agl=(110.0, 150.0, 190.0, 230.0), max_bank_deg=40.0, dt=0.5,
        sensor_tau_s=1.0, seed=0, max_t=7200.0) -> Episode:
    """The mission flown with wind: launch from the stand at the origin towards the fire, climb at the race's climb
    speed, dash at its dash speed (airspeed; the wind adds to the ground speed), then crosswind passes through the
    smoke ``sample_downwind_m`` downwind of the fire at ``pass_heights_agl`` (as many as fit into the sampling time),
    home at the best-range speed, a glide approach and a belly landing at the origin. Point mass: heading follows the
    track with a bank limit, speed changes with the excess thrust, height with the excess power (or a glide)."""
    rng = np.random.default_rng(seed)
    wind = np.r_[plume.wind_vector, 0.0]
    d, n_ = plume.downwind, np.array([-plume.downwind[1], plume.downwind[0]])
    centre = np.array(plume.source_xy) + sample_downwind_m * d
    half = 2.6 * plume.sigmas(sample_downwind_m)[0] + 60.0
    v_s = race_result["sampling_speed"]
    # waypoints: (phase, x, y, agl target, airspeed)
    wps = [("climb", *(np.array(plume.source_xy) * 0.05), mission.sampling_agl_m, race_result["climb_speed"]),
           ("dash to the fire", *(centre - half * n_ - 500 * d), mission.sampling_agl_m, race_result["dash_speed"]),
           ("into the smoke", *(centre - half * n_), pass_heights_agl[0], v_s)]
    t_pass = 2 * half / v_s + math.pi * v_s / (G * math.tan(math.radians(max_bank_deg))) * 1.2
    k_pass = max(1, min(len(pass_heights_agl), int(mission.sampling_s // t_pass)))
    heights = list(pass_heights_agl)[:k_pass]
    side = 1.0
    for k, h in enumerate(heights):                       # each pass ends on the other side of the column
        wps.append((f"sampling pass {k + 1}", *(centre + side * half * n_), h, v_s))
        side = -side
    home = np.array(plume.source_xy) / np.linalg.norm(plume.source_xy)
    wps += [("home", *(1500.0 * home), 120.0, race_result["range_speed"]),
            ("approach", *(500.0 * home), 25.0, 1.3 * af.stall_speed()),
            ("landing", *(-300.0 * home), 0.0, 1.15 * af.stall_speed())]      # aimed past the site: touches down near it
    # state
    pos = np.array([0.0, 0.0, float(forest.height(0, 0)) + 1.5])
    to_fire = np.array(plume.source_xy) / np.linalg.norm(plume.source_xy)
    psi = math.atan2(to_fire[1], to_fire[0])
    V, e, ppm_s, t = mission.launch_speed, 0.0, 0.0, 0.0
    rec = {k: [] for k in ("t", "pos", "vel", "V", "psi", "bank", "T", "P", "E", "agl", "ppm", "phase")}
    passes, events, cur, k = [], [], None, 0
    W = af.mass_kg * G
    bank = 0.0
    while k < len(wps) and t < max_t:
        ph, wx, wy, h_t, v_t = wps[k]
        ground = float(forest.height(pos[0], pos[1]))
        agl = pos[2] - ground
        dxy = np.array([wx, wy]) - pos[:2]
        dist = float(np.linalg.norm(dxy))
        reach = 40.0
        if dist < reach and ph != "landing":
            if ph.startswith("sampling"):
                passes.append(cur); cur = None
            k += 1
            continue
        if ph == "landing" and agl < 0.3:
            events.append(f"t = {t:.0f} s: touchdown {np.linalg.norm(pos[:2]):.0f} m from the launch site")
            break
        # heading: air-relative direction that makes the ground track point at the waypoint
        u_des = dxy / max(dist, 1e-6)
        chi = math.atan2(u_des[1], u_des[0])                 # the wanted ground track; crab into the crosswind
        w_cross = -wind[0] * u_des[1] + wind[1] * u_des[0]
        psi_des = chi - math.asin(float(np.clip(w_cross / max(V, 1.0), -0.9, 0.9)))
        dpsi = (psi_des - psi + math.pi) % (2 * math.pi) - math.pi
        rate_max = G * math.tan(math.radians(max_bank_deg)) / max(V, 1.0)
        rate = float(np.clip(dpsi / 2.0, -rate_max, rate_max))
        psi += rate * dt
        bank = math.atan(abs(rate) * V / G)
        nload = 1.0 / math.cos(bank)
        # speed and height
        Tmax, Pmax, _ = unit.full(V)
        D = float(af.drag(V, RHO, nload))
        h_err = h_t - agl if ph != "landing" else -agl
        if ph == "landing":                                  # a steady glide slope, then a flare over the last 2 m
            vz_cmd = -2.0 if agl > 2.0 else -0.4
        elif ph == "approach":
            vz_cmd = float(np.clip(0.15 * h_err, -2.5, 0.5))
        else:
            vz_cmd = float(np.clip(0.2 * h_err, -3.0, 6.0))
        a_cmd = float(np.clip((v_t - V) / 3.0, -2.0, 4.0))
        T_need = D + W * vz_cmd / V + af.mass_kg * a_cmd
        if T_need > Tmax:                                    # the climb takes what the speed leaves
            T = Tmax
            spare = T - D - af.mass_kg * max(a_cmd, 0.0) * (a_cmd > 0)
            vz = min(vz_cmd, max(spare, 0.0) * V / W) if vz_cmd > 0 else vz_cmd
            a = (T - D - W * vz / V) / af.mass_kg
        else:
            T, vz, a = max(T_need, 0.0), vz_cmd, a_cmd
            if T_need < 0:                                   # glide: drag decelerates
                a = (-D - W * vz / V) / af.mass_kg
        P = 0.0 if T <= 0 else unit.electrical_power(V, T)
        P = min(P, unit.max_electrical_w) if np.isfinite(P) else unit.max_electrical_w
        V = max(V + a * dt, 0.9 * af.stall_speed())
        air = np.array([V * math.cos(psi), V * math.sin(psi), vz])
        vel = air + wind
        pos = pos + vel * dt
        e += P * dt / 3600
        t += dt
        agl = pos[2] - float(forest.height(pos[0], pos[1]))
        c = float(plume.concentration_ppm(pos[0], pos[1], agl))
        ppm_s += (c - ppm_s) * dt / (sensor_tau_s + dt)
        reading = max(ppm_s + rng.normal(0, 0.02), 0.0)
        if ph.startswith("sampling") and abs(agl - h_t) < 8.0:     # the pass proper: at its height
            if cur is None:
                cur = {"x": [], "y": [], "z_agl": [], "ppm": [], "name": ph}
            cur["x"].append(pos[0]); cur["y"].append(pos[1]); cur["z_agl"].append(agl); cur["ppm"].append(reading)
        for key, val in (("t", t), ("pos", pos.copy()), ("vel", vel), ("V", V), ("psi", psi), ("bank", bank), ("T", T), ("P", P),
                         ("E", e), ("agl", agl), ("ppm", reading), ("phase", ph)):
            rec[key].append(val)
        if e > mission.battery_wh * mission.usable_fraction and "battery" not in " ".join(events):
            events.append(f"t = {t:.0f} s: usable battery exhausted")
    if t >= max_t:
        events.append("time limit")
    passes = [{k_: np.array(v_) if k_ != "name" else v_ for k_, v_ in p_.items()} for p_ in passes if p_ is not None and len(p_["x"]) > 10]
    return Episode(np.array(rec["t"]), np.array(rec["pos"]), np.array(rec["vel"]), np.array(rec["V"]), np.array(rec["psi"]),
                   np.array(rec["bank"]), np.array(rec["T"]), np.array(rec["P"]), np.array(rec["E"]), np.array(rec["agl"]),
                   np.array(rec["ppm"]), rec["phase"], passes, events, unit.name)


def profile_figure(ep: Episode, plume: Plume):
    """Height, airspeed, power and energy, the CO reading against time, and the plan view with the smoke."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(14, 7.5))
    t = ep.t
    ax[0, 0].plot(t, ep.agl, color="#1565c0"); ax[0, 0].set(xlabel="time [s]", ylabel="m AGL", title="height above the forest")
    ax[0, 1].plot(t, ep.airspeed, label="airspeed"); ax[0, 1].plot(t, np.linalg.norm(ep.vel[:, :2], axis=1), label="ground speed", alpha=0.7)
    ax[0, 1].set(xlabel="time [s]", ylabel="m/s", title="speed"); ax[0, 1].legend(fontsize=8)
    ax[1, 0].plot(t, ep.power, color="#ef6c00", label="electrical power [W]")
    a2 = ax[1, 0].twinx(); a2.plot(t, ep.energy_wh, color="#2e7d32"); a2.set_ylabel("energy used [Wh]", color="#2e7d32")
    ax[1, 0].set(xlabel="time [s]", ylabel="W", title="propulsion")
    a3 = ax[1, 1]
    xs = np.linspace(min(ep.pos[:, 0].min(), 0) - 500, ep.pos[:, 0].max() + 1500, 160)
    ys = np.linspace(ep.pos[:, 1].min() - 1500, ep.pos[:, 1].max() + 1500, 120)
    X, Y = np.meshgrid(xs, ys)
    a3.contourf(X, Y, plume.concentration_ppm(X, Y, 150.0), levels=12, cmap="Greys")
    a3.plot(ep.pos[:, 0], ep.pos[:, 1], color="#1565c0", lw=1.2); a3.plot(0, 0, "o", color="#1565c0")
    a3.plot(*plume.source_xy, "*", color="#d84315", ms=14)
    a3.set(xlabel="x [m]", ylabel="y [m]", title="plan view, CO at 150 m AGL"); a3.set_aspect("equal", adjustable="datalim")
    for a in (ax[0, 0], ax[0, 1], ax[1, 0]):
        a.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------------------------------------- the movie
def _glyph(scale):
    """A fixed-wing seen as lines in body axes (x forward, y left, z up): fuselage, wing, tail plane, fin."""
    s = scale
    return [np.array([[0.55, 0, 0], [-0.45, 0, 0]]) * s, np.array([[0.1, 0.7, 0], [0.1, -0.7, 0]]) * s,
            np.array([[-0.42, 0.22, 0], [-0.42, -0.22, 0]]) * s, np.array([[-0.42, 0, 0], [-0.45, 0, 0.18]]) * s]


def render_movie(ep: Episode, forest: Forest, plume: Plume, path, *, fps=20, seconds=20.0, size=(1280, 640),
                 sampling_share=0.55, n_smoke=1400, title="MERLIN — rapid smoke sampling", progress=False):
    """Frames with matplotlib (3D, the forest as a shaded surface around the aircraft, the smoke column as grey
    particles from the plume model, the aircraft as a glyph with its trail) plus a map and the CO reading; the Vegeta
    watermark; MP4 through OpenCV. Time-lapse: ``sampling_share`` of the frames go to the passes through the smoke, the
    rest to the transit. Returns ``(path, frames)``."""
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    from vegeta.aeromant._watermark import watermark

    n_frames = max(int(seconds * fps), 2)
    samp = np.array([ph.startswith("sampling") for ph in ep.phase])
    w = np.where(samp, sampling_share / max(samp.sum(), 1), (1 - sampling_share) / max((~samp).sum(), 1))
    cdf = np.cumsum(w); cdf /= cdf[-1]
    idx = np.searchsorted(cdf, np.linspace(0, 1, n_frames, endpoint=False) + 0.5 / n_frames).clip(0, len(ep.t) - 1)
    ls = LightSource(azdeg=315, altdeg=40)
    w_px, h_px = size
    fig = plt.figure(figsize=(w_px / 100, h_px / 100), dpi=100)
    fig.patch.set_facecolor("#a9c7e3")
    ax = fig.add_axes([0.0, 0.0, 0.66, 0.93], projection="3d")
    axm = fig.add_axes([0.70, 0.50, 0.28, 0.36])
    axc = fig.add_axes([0.70, 0.09, 0.28, 0.33])
    fig.text(0.01, 0.975, title, fontsize=13, weight="bold", color="#0d2a4a", va="top")
    txt = fig.text(0.01, 0.925, "", fontsize=9.5, family="monospace", color="#0d2a4a", va="top")
    # the map: route, fire, smoke footprint
    xs = np.linspace(min(ep.pos[:, 0].min(), 0) - 800, max(ep.pos[:, 0].max(), plume.source_xy[0]) + 2500, 200)
    ys = np.linspace(ep.pos[:, 1].min() - 2500, ep.pos[:, 1].max() + 2500, 120)
    X, Y = np.meshgrid(xs, ys)
    axm.imshow(forest.height(X, Y), extent=(xs[0], xs[-1], ys[0], ys[-1]), origin="lower", cmap="Greens_r", alpha=0.8, aspect="auto")
    cmap = plume.concentration_ppm(X, Y, 150.0)
    axm.contourf(X, Y, np.ma.masked_less(cmap, 0.05), levels=8, cmap="Greys", alpha=0.85)
    axm.plot(ep.pos[:, 0], ep.pos[:, 1], color="#ffffff", lw=0.6, alpha=0.6)
    axm.plot(*plume.source_xy, "*", color="#ff5722", ms=12); axm.plot(0, 0, "s", color="#1565c0", ms=6)
    axm.set_xticks([]); axm.set_yticks([]); axm.set_title("launch ■   fire ★   smoke at 150 m AGL", fontsize=9)
    me, = axm.plot([], [], "o", color="#ffeb3b", ms=6, mec="k")
    axc.plot(ep.t, ep.ppm, color="#bbbbbb", lw=0.8)
    cur, = axc.plot([], [], color="#d84315", lw=1.5)
    axc.set(xlabel="time [s]", ylabel="CO above background [ppm]", xlim=(ep.t[0], ep.t[-1]), ylim=(0, max(ep.ppm.max() * 1.1, 1.0)))
    axc.set_title("forward sensor", fontsize=9); axc.grid(alpha=0.3)
    glyph = _glyph(170.0)
    frames = []
    it = range(n_frames)
    if progress:
        from tqdm.auto import tqdm
        it = tqdm(it, desc="movie frames")
    H = 1800.0
    for f in it:
        i = int(idx[f])
        p = ep.pos[i]
        ax.cla()
        ax.computed_zorder = False                      # draw in zorder: forest, smoke, trail, aircraft
        ax.set_facecolor("#a9c7e3")
        gx = np.linspace(p[0] - H, p[0] + H, 34); gy = np.linspace(p[1] - H, p[1] + H, 34)
        GX, GY = np.meshgrid(gx, gy)
        GZ = forest.height(GX, GY)
        rgb = np.zeros(GZ.shape + (3,)); rgb[..., 0], rgb[..., 1], rgb[..., 2] = 0.16, 0.36, 0.17
        ax.plot_surface(GX, GY, GZ, facecolors=ls.shade_rgb(rgb, GZ, vert_exag=4, blend_mode="soft"), linewidth=0,
                        antialiased=False, shade=False, zorder=1)
        pts, s_ = plume.particles(n_smoke, t=ep.t[i], seed=3)
        m = (np.abs(pts[:, 0] - p[0]) < H) & (np.abs(pts[:, 1] - p[1]) < H)
        if m.any():
            zg = forest.height(pts[m, 0], pts[m, 1]) + pts[m, 2]
            age = np.clip(s_[m] / 2500.0, 0, 1)
            col = np.column_stack([0.30 + 0.45 * age, 0.30 + 0.45 * age, 0.32 + 0.45 * age, 0.75 - 0.45 * age])
            ax.scatter(pts[m, 0], pts[m, 1], zg, c=col, s=18 + 50 * age, depthshade=False, zorder=5, linewidths=0)
        fx, fy = plume.source_xy
        if abs(fx - p[0]) < H and abs(fy - p[1]) < H:
            ax.scatter([fx], [fy], [forest.height(fx, fy) + 5], color="#ff5722", s=90, marker="^", depthshade=False, zorder=6)
        c, s = math.cos(ep.heading[i]), math.sin(ep.heading[i])
        Rm = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        for seg in glyph:
            q = (Rm @ seg.T).T * [1, 1, 1] + p
            ax.plot(q[:, 0], q[:, 1], q[:, 2], color="#111111", lw=2.2, zorder=10)
        tr = np.arange(max(0, i - 600), i + 1, 2)
        tr = tr[(np.abs(ep.pos[tr, 0] - p[0]) < H) & (np.abs(ep.pos[tr, 1] - p[1]) < H)]
        ax.plot(ep.pos[tr, 0], ep.pos[tr, 1], ep.pos[tr, 2], color="#ffeb3b", lw=1.2, zorder=9)
        ax.set_xlim(gx[0], gx[-1]); ax.set_ylim(gy[0], gy[-1]); ax.set_zlim(GZ.min() - 20, GZ.min() + 900)
        ax.set_box_aspect((1, 1, 0.42), zoom=1.3)
        for a_ in (ax.xaxis, ax.yaxis, ax.zaxis):
            a_.set_pane_color((0.66, 0.78, 0.89, 1.0)); a_.line.set_color((1, 1, 1, 0)); a_.set_ticks([])
        ax.grid(False)
        ax.view_init(elev=24, azim=math.degrees(ep.heading[i]) - 150 + 25 * math.sin(f / n_frames * math.pi))
        me.set_data([p[0]], [p[1]])
        cur.set_data(ep.t[:i + 1], ep.ppm[:i + 1])
        txt.set_text(f"{ep.unit}   t = {ep.t[i]:5.0f} s   {ep.phase[i]}\n"
                     f"airspeed {ep.airspeed[i]:4.1f} m/s   {ep.agl[i]:4.0f} m AGL   {ep.power[i]:5.0f} W   {ep.energy_wh[i]:5.1f} Wh   CO {ep.ppm[i]:5.2f} ppm")
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        frames.append(watermark(cv2.cvtColor(img, cv2.COLOR_RGB2BGR)))
    plt.close(fig)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    hh, ww = frames[0].shape[:2]
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (ww, hh))
    for fr in frames:
        out.write(fr)
    out.release()
    return path, frames


__all__ = ["DESIGN_JSON", "load_design", "DISTANCES_KM", "NAMES", "race_table", "build_merlins", "installation", "installation_pod", "COMMON_MASS_KG", "UNIT_MASS_KG", "Airframe", "Unit", "open_propeller", "ducted_fan_unit", "Mission", "performance", "race", "Plume",
           "source_estimate", "Forest", "Episode", "fly", "profile_figure", "render_movie"]
