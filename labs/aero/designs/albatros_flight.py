"""ALBATROS sized and flown (notebook 50): the ways to get a turbojet transport into the air and back, each closed on
its own take-off mass, and the fuel each burns to carry the cargo multirotor ``radius_km`` out to a vessel and come
back.

The engines are ``vegeta.boreas.microjet`` classes tabulated as AGUYA's ``JetUnit`` (``aguya_flight.jet_unit``), at sea
level for the launch and at the cruise altitude for the cruise; the drag polar is ``albatros.drag_buildup`` on the
geometry ``albatros.for_design`` draws around the bay (``albatros_bay``), in AGUYA's ``JetAirframe``.

**The layouts** (``CONCEPTS``):

- ``catapult`` — the baseline: a wing-borne jet. It leaves a ship's catapult (or a runway ashore) and lands on a
  runway ashore (skids, a net or a wire). Two turbojets with two-axis vectoring nozzles. Over the vessel it does not
  stop: it slows to 1.3 x its stall speed and releases the multirotor in a pass. Nothing is carried for a hover.
- ``vectored`` — a vertical take-off and landing on jets: the two cruise engines swivel their nozzles down behind the
  centre of gravity, and two larger lift jets in the fuselage ahead of it carry the rest. The lift jets are shut
  down behind doors in the cruise and restarted over the vessel, where it hovers for the release.
- ``tailsitter`` — it stands on its tail fins, four engines at the tail with two-axis nozzles; in the cruise it flies
  on as few of them as hold the drag.

``SET_ASIDE`` keeps a fourth, ``lift_cruise`` (four electric lift rotors on booms and a battery for the deck, the
jets for the cruise): sized and compared in the first pass of the study and dropped — no propellers on Albatros.

**The mission** (``fly``; a point mass, phases with their own time step): start and warm-up; the launch (the
catapult stroke and the first seconds at full thrust; or, thrust-borne, ``takeoff_hover_s`` and a transition); climb
to ``cruise_altitude_m`` at the speed and on the engines that cost the least fuel; cruise out at the speed of best
range (or ``min_cruise_speed``, if that is faster) against ``headwind_out``; the glide down on idling engines; over
the vessel ``release_hover_s`` in the slow pass (or the hover), where the multirotor leaves the bay and the mass
falls by it; the same back with the wind behind; the approach and landing (``landing_hover_s``). Thrust-borne
flight costs the jets fuel at the thrust that holds the weight. ``size`` finds the fuel load that leaves
``reserve`` of it at the end.

**The sizing** (``size``): take-off mass = structure (a fraction of it) + engines + systems + the bay's load + fuel,
the engines scaled from the catalogue for the mass, the wing from the wing loading (the layout's own, or
``Requirements.wing_loading``), the fuselage around the bay and the tanks — the smallest mass at which the aircraft
built for it and fuelled for the mission weighs no more than it. ``compare`` sizes the layouts; ``choose`` picks one
by the rule given there, with the gust numbers of ``albatros_gust``; ``launch`` gives the catapult and the runways.

Limits: point-mass phases (no trim, no transition dynamics: the transition is a time at a thrust), a parabolic polar
with one ``cd0``, the engines' tables between two altitudes, no wind shear, no icing, ISA. The structure fractions
are assumptions listed in ``MASS``: change them and the ranking can change. The sizing's own noise (discrete steps,
the choice of running engines) is about +-4 % of the take-off mass.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from ogma.rotors import ideal_hover_power
from ogma.units import G
from ogma.units import RHO0 as RHO

import aguya_flight as AF
from aguya_flight import JetAirframe, JetUnit

# ------------------------------------------------------------------------------------------------ assumptions
MASS = {
    "structure": 0.215,            # wing, fuselage, tail, bay frames / take-off mass (composite, sealed hull) at aspect
                                   # ratio 7; the wing is 30 % of it and grows as (aspect ratio / 7)^0.6 (Raymer's exponent)
    "lift booms": 0.022,           # lift_cruise: the two booms and their pylons / take-off mass
    "lift-jet bay": 0.020,         # vectored: intake and exhaust doors, the hot bay / take-off mass
    "tail-sitter gear": 0.035,     # tailsitter: leg struts in the fins, the stiffened tail / take-off mass
    "launch and landing": 0.030,   # catapult: the launch hooks and their frames, the skids, the arresting hook / take-off mass
    "actuators": 0.015,            # servos, flaperons, doors / take-off mass
    "systems": 5.5,                # kg: two autopilots, GNSS/INS, satcom and line-of-sight links, deck-relative
                                   # navigation, lights, harness, avionics battery
    "flotation": 1.0,              # kg: seals, bilge sensor, sea anchor
    "fuel system": (1.0, 0.07),    # kg + fraction of the fuel: bladders, pumps, plumbing, unusable fuel
    "vectoring nozzle": 0.15,      # two-axis nozzle and its actuators / engine mass
    "swivel nozzle": 0.28,         # 0-95 deg swivelling nozzle / engine mass
    "rotor drive": 0.0055,         # kg per N of maximum rotor thrust: motor, controller, propeller
    "pack_wh_kg": 150.0, "pack_usable": 0.8, "pack_w_kg": 3000.0,     # the lift battery (high-rate lithium polymer)
}
FIGURE_OF_MERIT, DRIVE_EFFICIENCY = 0.70, 0.88
POD_OFF_DRAG = 0.3                 # Cd on the intake area of a shut-down engine in a pod (spillage, the windmilling rotor)
CLASSES = ("100 N class", "140 N class", "200 N class", "300 N class", "400 N class", "550 N class", "800 N class")


@dataclass(frozen=True)
class Requirements:
    cargo_kg: float = 10.0
    radius_km: float = 300.0
    headwind_out: float = 0.0          # m/s against the outbound leg (it helps on the way home)
    cruise_altitude_m: float = 6000.0  # high: a turbojet's range grows with the true airspeed it holds at a given drag
    min_cruise_speed: float | None = None     # m/s; None: the speed of best range
    start_s: float = 45.0              # engine start and warm-up at idle on the deck
    takeoff_hover_s: float = 25.0      # lift off, clear the deck and the ship's wake
    transition_s: float = 20.0         # each transition, either way
    release_hover_s: float = 60.0      # the hover over the vessel until the multirotor is away
    landing_hover_s: float = 60.0      # the approach to the moving deck, the wait for a quiet period, the landing
    restart_s: float = 40.0            # an in-flight start of a shut-down engine, to idle
    reserve: float = 0.15              # of the fuel load, left at the end
    hover_margin: float = 1.25         # installed thrust / weight for thrust-borne flight (control, a hot day)
    wing_loading: float = 250.0        # kg/m² at take-off
    aspect_ratio: float = 10.0
    sweep_deg: float = 20.0
    cl_max: float = 1.3
    oswald: float = 0.8
    descent_sink: float = 5.0          # m/s, the least sink rate in the descent


@dataclass(frozen=True)
class Concept:
    key: str
    name: str
    kind: str                          # lift_cruise | vectored | tailsitter
    engines: int                       # turbojets in pods (they push the cruise)
    lift_engines: int = 0              # lift jets in the fuselage (``vectored``): shut down behind doors in the cruise
    disk_loading: float = 450.0        # lift rotors [N/m²]
    rotor_margin: float = 1.4          # maximum rotor thrust / weight
    cruise_factor: float = 0.30        # static thrust of the podded engines / weight (not the tail-sitter's)
    wing_loading: float | None = None  # kg/m² at take-off; None: ``Requirements.wing_loading``
    cl_max: float | None = None        # None: ``Requirements.cl_max``


CONCEPTS = {
    # a wing that must fly off a catapult: 120 kg/m² with the flaperons drooped (cl_max 1.7) is a 34 m/s stall at sea level
    "catapult": Concept("catapult", "catapult or runway launch, lands ashore", "catapult", 2, wing_loading=120.0, cl_max=1.7),
    "vectored": Concept("vectored", "pure jet, vectored thrust", "vectored", 2, 2),
    "tailsitter": Concept("tailsitter", "jet tail-sitter", "tailsitter", 4),
}
SET_ASIDE = {"lift_cruise": Concept("lift_cruise", "electric lift + jet cruise", "lift_cruise", 2)}


# ------------------------------------------------------------------------------------------------ engines
_UNITS: dict = {}


def units(name: str, altitude_m: float) -> tuple:
    """(engine, its table at sea level, its table at the altitude), cached."""
    from vegeta.boreas import microjet as mj

    key = (name, round(altitude_m))
    if key not in _UNITS:
        e = mj.from_catalogue(name)
        _UNITS[key] = (e, AF.jet_unit(e), AF.jet_unit(e, atmosphere=mj.isa(altitude_m)))
    return _UNITS[key]


def density(altitude_m: float) -> float:
    from vegeta.boreas import microjet as mj
    return mj.isa(altitude_m).density


def pick_engines(concept: Concept, weight_n: float, req: Requirements, cruise_factor: float = 0.30) -> tuple:
    """((class, scale) of each podded engine, (class, scale) of each lift jet or None). The lift-plus-cruise's two
    engines give ``cruise_factor`` of the weight in static thrust (the climb, the push through the transition); the
    vectored layout's cruise engines the same and its lift jets the rest of ``hover_margin`` x the weight; the
    tail-sitter's engines share ``hover_margin`` x the weight. Each is the catalogue class nearest in thrust, scaled
    to the thrust needed (``scaled``): between and beyond the classes the cycle is the same, the size is not."""
    from vegeta.boreas import microjet as mj

    def nearest(each):
        name = min(CLASSES, key=lambda c: abs(math.log(mj.CATALOGUE[c]["thrust_N"] / each)))
        return name, each / mj.CATALOGUE[name]["thrust_N"]

    if concept.kind == "tailsitter":
        return nearest(req.hover_margin * weight_n / concept.engines), None
    pods = nearest(cruise_factor * weight_n / concept.engines)
    if concept.kind in ("lift_cruise", "catapult"):
        return pods, None
    return pods, nearest(max(req.hover_margin - cruise_factor, 0.1) * weight_n / concept.lift_engines)


def scaled(name: str, scale: float, altitude_m: float) -> tuple:
    """(engine, sea-level table, altitude table) of a catalogue class scaled in air flow by ``scale``: thrust, fuel
    flow and mass grow with the flow, the diameters with its square root, the shaft speed falls with it (the same tip
    speed and so the same cycle). A scale near 1 is a neighbouring engine of the class; far from 1 it is an engine
    nobody sells — the catalogue's classes are steps of about 1.4."""
    e, u0, uh = units(name, altitude_m)
    if abs(scale - 1.0) < 1e-9:
        return e, u0, uh
    r = math.sqrt(scale)
    es = replace(e, design_mass_flow=e.design_mass_flow * scale, impeller_diameter=e.impeller_diameter * r, rpm_max=e.rpm_max / r,
                 rpm_idle=e.rpm_idle / r, mass_kg=e.mass_kg * scale, system_mass_kg=e.system_mass_kg * r, outer_diameter=e.outer_diameter * r,
                 length=e.length * r, spool_time=e.spool_time * r, name=f"{e.name} x {scale:.2f}", a4=e.a4 * scale, a8=e.a8 * scale)
    return es, *(replace(u, name=es.name, rpm=u.rpm / r, thrust=u.thrust * scale, fuel=u.fuel * scale,
                         mass_kg=es.mass_kg + es.system_mass_kg, spool_time=es.spool_time) for u in (u0, uh))


# ------------------------------------------------------------------------------------------------ the flight
@dataclass
class Design:
    """One layout, sized: what it is made of and how it flew."""
    concept: Concept
    mtow_kg: float
    budget: dict
    engine_name: str                   # the catalogue class the engine is scaled from
    engine_scale: float
    engines: int
    engine: object
    lift_engine: tuple | None          # (class, scale) of the lift jets, if any
    wing_area_m2: float
    params: dict
    airframe: JetAirframe
    fuel_kg: float
    battery_kg: float = 0.0
    rotor_diameter_m: float = 0.0
    hover_power_w: float = 0.0
    flight: "Flight | None" = None
    feasible: bool = True
    notes: list = field(default_factory=list)

    @property
    def dry_kg(self) -> float:
        return self.mtow_kg - self.fuel_kg


@dataclass
class Flight:
    fuel_kg: float
    fuel_left_kg: float
    battery_wh: float
    takeoff_mass_kg: float
    landing_mass_kg: float
    cruise_speed_out: float
    cruise_speed_back: float
    time_to_vessel_s: float
    total_time_s: float
    cruise_engines: int
    phases: dict = field(default_factory=dict)       # name -> {"time", "fuel", "dist", "wh"}
    track: dict = field(default_factory=dict)
    feasible: bool = True
    notes: list = field(default_factory=list)

    def table(self):
        import pandas as pd
        rows = {k: {"time [s]": round(v["time"]), "distance [km]": round(v["dist"] / 1000, 1), "fuel [kg]": round(v["fuel"], 2),
                    "battery [Wh]": round(v["wh"])} for k, v in self.phases.items()}
        t = pd.DataFrame(rows).T
        t.loc["total"] = t.sum()
        return t


def best_range_speed(af: JetAirframe, unit: JetUnit, mass_kg: float, rho: float, engines: int, v_min: float = 0.0) -> tuple:
    """(speed, fuel flow) of the most distance per kg of fuel on ``engines`` engines; a speed above ``v_min``."""
    lo = max(1.25 * af.stall_speed(mass_kg, rho), v_min, float(unit.V[1]))
    best = None
    for V in np.linspace(lo, float(unit.V[-1]) - 1.0, 40):
        ff, _, ok = unit.for_thrust(V, af.drag(V, mass_kg, rho) / engines)
        if not ok:
            break
        sr = V / (engines * ff)
        if best is None or sr > best[0]:
            best = (sr, float(V), engines * ff)
    if best is None:
        return float("nan"), float("nan")
    return best[1], best[2]


def cruise_engines(af, unit, mass_kg, rho, total: int, v_min: float = 0.0) -> int:
    """The number of running engines that gives the best range (each extra engine idles more than it saves), with
    20 % of thrust in hand at that speed."""
    best = None
    for k in range(1, total + 1):
        V, ff = best_range_speed(af, unit, mass_kg, rho, k, v_min)
        if not math.isfinite(V) or k * unit.full(V)[0] < 1.2 * af.drag(V, mass_kg, rho):
            continue
        if best is None or V / ff > best[0]:
            best = (V / ff, k)
    return best[1] if best else total


def fly(d: Design, req: Requirements, fuel_kg: float, dropped_kg: float) -> Flight:
    """Fly the mission with ``fuel_kg`` on board (see the module text). ``d`` carries the airframe at its sized
    take-off mass less fuel (``dry_kg``), the engines and the lift system."""
    c, af, n = d.concept, d.airframe, d.engines
    eng, u0, uh = scaled(d.engine_name, d.engine_scale, req.cruise_altitude_m)
    n_l = c.lift_engines if d.lift_engine else 0
    ul = scaled(*d.lift_engine, req.cruise_altitude_m)[1] if n_l else None
    static = n * u0.full(0.0)[0] + (n_l * ul.full(0.0)[0] if n_l else 0.0)
    rho_h = density(req.cruise_altitude_m)
    mass, fuel, wh, t, x = d.dry_kg + fuel_kg, fuel_kg, 0.0, 0.0, 0.0
    m0 = mass
    phases, notes = {}, []
    log = {k: [] for k in ("t", "x", "h", "V", "mass", "fuel_flow")}
    h = 0.0
    thrust_borne = c.kind in ("vectored", "tailsitter")
    winged = c.kind == "catapult"                                  # never thrust-borne: launched and landed on its wing
    pods = n
    a_off = POD_OFF_DRAG * math.pi * (0.375 * eng.outer_diameter) ** 2

    def drag(V, m, rho, k_run):
        return af.drag(V, m, rho) + 0.5 * rho * V * V * a_off * max(pods - k_run, 0)
    disk_area = (d.mtow_kg * G / c.disk_loading) if c.kind == "lift_cruise" else 0.0

    def step(name, dt, ff, V=0.0, dx=0.0, dh=0.0, power_w=0.0):
        nonlocal mass, fuel, wh, t, x, h
        burn = ff * dt
        mass -= burn
        fuel -= burn
        wh += power_w * dt / 3600
        t += dt
        x += dx
        h += dh
        ph = phases.setdefault(name, {"time": 0.0, "fuel": 0.0, "dist": 0.0, "wh": 0.0})
        ph["time"] += dt
        ph["fuel"] += burn
        ph["dist"] += abs(dx)
        ph["wh"] += power_w * dt / 3600
        for k, v in (("t", t), ("x", x), ("h", h), ("V", V), ("mass", mass), ("fuel_flow", ff)):
            log[k].append(v)

    def hover(name, seconds, climb=0.0):
        """Thrust-borne flight: the jets hold the weight (plus 3 % for the nozzles' splay and control); or the rotors
        do and the jets idle."""
        for _ in range(int(round(seconds))):
            W = mass * G
            if thrust_borne:                                        # every engine at the same fraction of its thrust
                frac = 1.03 * W / static
                ff, _, ok = u0.for_thrust(0.0, frac * u0.full(0.0)[0])
                ff = n * ff + (n_l * ul.for_thrust(0.0, frac * ul.full(0.0)[0])[0] if n_l else 0.0)
                if not ok and "the engines cannot hold the weight in the hover" not in notes:
                    notes.append("the engines cannot hold the weight in the hover")
                step(name, 1.0, ff, dh=climb)
            else:
                p_el = ideal_hover_power(W, disk_area, RHO) / (FIGURE_OF_MERIT * DRIVE_EFFICIENCY)
                step(name, 1.0, n * u0.idle(0.0)[1], dh=climb, power_w=p_el)

    def slow(name, seconds, thrust=1.0):
        """Wing-borne at 1.3 x the stall speed at sea level, the engines at ``thrust`` x the drag (1: level; less: the
        approach). The ground covered is not counted towards the trip."""
        for _ in range(int(round(seconds))):
            V = 1.3 * af.stall_speed(mass, RHO)
            ff, _, _ = u0.for_thrust(V, thrust * af.drag(V, mass, RHO) / n)
            step(name, 1.0, n * ff, V)

    def transition(name, v_end, sign=1.0):
        """Half the weight on thrust on average, the cruise engines at full thrust; the speed changes linearly."""
        for i in range(int(round(req.transition_s))):
            V = v_end * (i + 0.5) / req.transition_s if sign > 0 else v_end * (1 - (i + 0.5) / req.transition_s)
            W = mass * G
            if thrust_borne:
                step(name, 1.0, 0.8 * (n * u0.full(V)[1] + (n_l * ul.full(V)[1] if n_l else 0.0)), V, dx=sign_x * V)
            else:
                p_el = 0.5 * ideal_hover_power(W, disk_area, RHO) / (FIGURE_OF_MERIT * DRIVE_EFFICIENCY)
                step(name, 1.0, n * u0.full(V)[1], V, dx=sign_x * V, power_w=p_el)

    def mix(V, s):
        """Full thrust and fuel flow per engine at height fraction s between the two tables."""
        (T0, f0), (T1, f1) = u0.full(V), uh.full(V)
        return (1 - s) * T0 + s * T1, (1 - s) * f0 + s * f1

    def climb(name, k_run):
        nonlocal h
        while h < req.cruise_altitude_m:
            s = h / req.cruise_altitude_m
            rho = RHO + s * (rho_h - RHO)
            # the climb speed that buys the most height (and distance, at a glide ratio of 12) per kg of fuel:
            # a turbojet's thrust power grows with the speed, so it climbs best well above the stall
            # (and on as many of its engines as pay: the tail-sitter's four would climb at 15 m/s and burn for it)
            best = None
            for kk in range(1, k_run + 1):
                for V in np.linspace(1.35 * af.stall_speed(mass, rho), 0.92 * float(u0.V[-1]), 5):
                    T, ff = mix(V, s)
                    r = min((kk * T - drag(V, mass, rho, kk)) * V / (mass * G), 15.0)
                    if r >= 1.5 and (best is None or (r + V / 12.0) / (kk * ff) > best[0]):
                        best = ((r + V / 12.0) / (kk * ff), V, kk * ff, r)
            if best is None:
                roc = 0.0
            else:
                _, V, ff, roc = best
            if roc < 1.0:
                notes.append("cannot climb to the cruise altitude")
                return False
            dt = 15.0
            step(name, dt, ff, V, dx=sign_x * math.sqrt(max(V * V - roc * roc, 0.0)) * dt, dh=roc * dt)
        h = req.cruise_altitude_m
        return True

    def cruise(name, x_end, wind, k_run):
        """Level at the speed of best range until the glide down from here reaches ``x_end``."""
        V, _ = best_range_speed(af, uh, mass, rho_h, k_run, req.min_cruise_speed or 0.0)
        if not math.isfinite(V):
            notes.append(f"{k_run} engine(s) cannot hold the cruise")
            return float("nan")
        gs = V - wind
        glide = descent_dist(mass, k_run, V)                        # from the mass now: it changes little, and the
        while True:                                                 # descent stretches to the end point
            left = abs(x_end - x) - glide
            if left <= 0:
                return V
            ff, _, ok = uh.for_thrust(V, drag(V, mass, rho_h, k_run) / k_run)
            if not ok:
                notes.append(f"{k_run} engine(s) at full thrust do not hold {V:.0f} m/s in the cruise")
                return float("nan")
            dt = min(40.0, left / gs + 1e-6)
            step(name, dt, k_run * ff, V, dx=sign_x * gs * dt)

    def descent_plan(m, k_run, v):
        """The glide down from the cruise altitude at 0.8 v on idling engines: steps (dt, fuel flow, speed, ground
        distance, height change). The sink rate is what the drag less the idle thrust takes, at least
        ``descent_sink`` (air brakes: the flaperons split)."""
        hh, plan = req.cruise_altitude_m, []
        V = 0.8 * v
        while hh > 1e-3:
            sfrac = hh / req.cruise_altitude_m
            rho = RHO + sfrac * (rho_h - RHO)
            (T0, f0), (T1, f1) = u0.idle(V), uh.idle(V)
            T, ff = (1 - sfrac) * T0 + sfrac * T1, (1 - sfrac) * f0 + sfrac * f1
            Vd = max(V, 1.3 * af.stall_speed(m, rho))
            sink = max(Vd * (drag(Vd, m, rho, k_run) - k_run * T) / (m * G), req.descent_sink)
            dt = min(20.0, hh / sink)
            plan.append((dt, k_run * ff, Vd, math.sqrt(max(Vd * Vd - sink * sink, 0.0)) * dt, -sink * dt))
            hh -= sink * dt
        return plan

    def descent(name, k_run, v, x_end):
        """The planned glide, its ground distance stretched or shortened to end at ``x_end`` (a shallower or steeper
        glide: a few per cent, unless the climb alone overshot a short trip)."""
        nonlocal h, x
        plan = descent_plan(mass, k_run, v)
        total = sum(q[3] for q in plan)
        scale = abs(x_end - x) / total if total > 0 else 1.0
        for dt, ff, V, dx, dh in plan:
            step(name, dt, ff, V, dx=sign_x * dx * scale, dh=dh)
        h, x = 0.0, x_end

    def descent_dist(m, k_run, v):
        return sum(q[3] for q in descent_plan(m, k_run, v if math.isfinite(v) else 100.0))

    k = cruise_engines(af, uh, d.dry_kg + 0.6 * fuel_kg, rho_h, pods, req.min_cruise_speed or 0.0)
    k_climb = pods
    radius = req.radius_km * 1000.0
    v_tr = 1.35 * af.stall_speed(mass, RHO)

    # ---- out
    sign_x = 1.0
    for _ in range(int(round(req.start_s))):
        step("start", 1.0, n * u0.idle(0.0)[1] + (n_l * ul.idle(0.0)[1] if n_l else 0.0))
    for _ in range(int(round(eng.spool_time))):
        step("start", 1.0, (n * 0.5 * (u0.idle(0.0)[1] + u0.full(0.0)[1]) + (n_l * 0.5 * (ul.idle(0.0)[1] + ul.full(0.0)[1]) if n_l else 0.0))
             * (0.3 if c.kind == "lift_cruise" else 1.0))
    if winged:
        for _ in range(5):                                          # the catapult's stroke and the first seconds away
            step("launch", 1.0, n * u0.full(v_tr)[1], v_tr, dx=v_tr)
    else:
        hover("take-off", req.takeoff_hover_s, climb=1.2)
        h = 0.0                                                     # the 30 m of the take-off are not counted
        transition("transition out", v_tr)
    ok = climb("climb out", k_climb)
    v_out = v_back = float("nan")
    if ok:
        v_out = cruise("cruise out", radius, req.headwind_out, k)
        ok = math.isfinite(v_out)
    if ok:
        descent("descent to the vessel", k, v_out, radius)
        if thrust_borne and n + n_l > k:                            # restart the engines that slept through the cruise
            for _ in range(int(round(req.restart_s))):
                step("restart", 1.0, n * u0.idle(60.0)[1] + (n_l * ul.idle(60.0)[1] if n_l else 0.0), 60.0)
        sign_x = 0.0
        if winged:
            slow("slow pass and release", req.release_hover_s)
        else:
            transition("transition at the vessel", 1.35 * af.stall_speed(mass, RHO), sign=-1.0)
            hover("hover and release", req.release_hover_s)
        t_vessel = t
        mass -= dropped_kg
        # ---- back
        sign_x = -1.0
        if not winged:
            transition("transition home", 1.35 * af.stall_speed(mass, RHO))
        ok = climb("climb home", k_climb)
    if ok:
        k_b = cruise_engines(af, uh, mass, rho_h, pods, req.min_cruise_speed or 0.0)
        v_back = cruise("cruise home", 0.0, -req.headwind_out, k_b)
        ok = math.isfinite(v_back)
    if ok:
        descent("descent home", k_b, v_back, 0.0)
        if thrust_borne and n + n_l > k_b:
            for _ in range(int(round(req.restart_s))):
                step("restart", 1.0, n * u0.idle(60.0)[1] + (n_l * ul.idle(60.0)[1] if n_l else 0.0), 60.0)
        sign_x = 0.0
        if winged:
            slow("approach and landing", req.landing_hover_s, thrust=0.6)
        else:
            transition("transition to land", 1.35 * af.stall_speed(mass, RHO), sign=-1.0)
            hover("deck approach and landing", req.landing_hover_s)
    feasible = ok and not notes and fuel >= req.reserve * fuel_kg - 1e-9
    if ok and fuel < req.reserve * fuel_kg - 1e-9:
        notes.append(f"fuel left {fuel:.2f} kg, below the {req.reserve:.0%} reserve")
    return Flight(fuel_kg, fuel if ok else float("nan"), wh, m0, mass, v_out, v_back, t_vessel if ok else float("nan"), t, k,
                  phases, {kk: np.array(v) for kk, v in log.items()}, feasible, notes)


def fuel_for(d: Design, req: Requirements, dropped_kg: float, hi: float | None = None) -> Flight:
    """The flight with the smallest fuel load that keeps the reserve (bisection); infeasible when ``hi`` kg (default:
    the take-off mass again) is not enough."""
    lo, hi = 0.5, hi or d.mtow_kg

    def margin(fk):
        f = fly(d, req, fk, dropped_kg)
        return f, (f.fuel_left_kg - req.reserve * fk) if math.isfinite(f.fuel_left_kg) else -1.0

    top, m_hi = margin(hi)
    if m_hi < 0:
        top.feasible = False
        return top
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if margin(mid)[1] >= 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < 0.01:
            break
    return fly(d, req, hi, dropped_kg)


# ------------------------------------------------------------------------------------------------ the sizing
def _build(concept: Concept, req: Requirements, option, mtow: float, fuel: float, battery: float, cruise_factor: float = 0.30) -> Design:
    """The design at a take-off mass, a fuel load and a lift battery: engines, geometry, polar, mass budget."""
    import albatros as al

    W = mtow * G
    (name, scale), lift = pick_engines(concept, W, req, cruise_factor)
    n = concept.engines
    eng, _, _ = scaled(name, scale, req.cruise_altitude_m)
    lift_eng = scaled(*lift, req.cruise_altitude_m)[0] if lift else None
    S = mtow / (concept.wing_loading or req.wing_loading)
    rotors = 4 if concept.kind == "lift_cruise" else 0
    rotor_d = math.sqrt(4 * (W / concept.disk_loading) / (rotors * math.pi)) if rotors else 0.0
    buried = concept.lift_engines if lift else 0
    bay = option.bay()
    p = al.for_design(wing_area_m2=S, aspect_ratio=req.aspect_ratio, bay=bay, fuel_kg=fuel, engine=eng, engines=n, layout=concept.kind, lift_rotors=rotors, rotor_diameter_m=rotor_d or 0.9, sweep_deg=req.sweep_deg,
                      multirotor=option.vehicles[0], buried_engines=buried,
                      buried_diameter_m=lift_eng.outer_diameter if lift_eng else 0.0)
    v_ref = 110.0
    db = al.drag_buildup(p, v_ref, mach=v_ref / 328.6)
    e_mass = eng.mass_kg + eng.system_mass_kg
    k1, k2 = MASS["fuel system"]
    budget = {"structure": MASS["structure"] * (0.7 + 0.3 * (req.aspect_ratio / 7.0) ** 0.6) * mtow, "actuators": MASS["actuators"] * mtow, "systems": MASS["systems"],
              "flotation": MASS["flotation"], f"engines ({n} x {_static_of(name, scale):.0f} N)": n * e_mass, "fuel system": k1 + k2 * fuel}
    hover_power = 0.0
    if concept.kind == "lift_cruise":
        budget["vectoring nozzles"] = MASS["vectoring nozzle"] * n * eng.mass_kg
        budget["lift booms"] = MASS["lift booms"] * mtow
        budget["lift rotors (motors, controllers, propellers)"] = MASS["rotor drive"] * concept.rotor_margin * W
        budget["lift battery"] = battery
        hover_power = ideal_hover_power(W, W / concept.disk_loading, RHO) / (FIGURE_OF_MERIT * DRIVE_EFFICIENCY)
    elif concept.kind == "catapult":
        budget["vectoring nozzles"] = MASS["vectoring nozzle"] * n * eng.mass_kg
        budget["launch hooks, skids and arresting hook"] = MASS["launch and landing"] * mtow
    elif concept.kind == "vectored":
        budget[f"lift jets ({buried} x {_static_of(*lift):.0f} N)"] = buried * (lift_eng.mass_kg + lift_eng.system_mass_kg)
        budget["swivel nozzles (2)"] = MASS["swivel nozzle"] * 2 * eng.mass_kg
        budget["lift-jet bay and doors"] = MASS["lift-jet bay"] * mtow
    else:
        budget["vectoring nozzles"] = MASS["vectoring nozzle"] * n * eng.mass_kg
        budget["tail-sitter legs and tail"] = MASS["tail-sitter gear"] * mtow
    budget["bay doors and release"] = bay["fittings_kg"]
    budget[f"bay load ({option.name})"] = option.dropped_kg
    budget["fuel"] = fuel
    dry = sum(v for kk, v in budget.items() if kk != "fuel")
    af = JetAirframe("ALBATROS " + concept.key, dry, S, db["aspect_ratio"], db["cd0"], req.oswald, concept.cl_max or req.cl_max, fuel)
    return Design(concept, mtow, budget, name, scale, n, eng, lift, S, p, af, fuel, battery, rotor_d, hover_power)


def _static_of(name: str, scale: float) -> float:
    from vegeta.boreas import microjet as mj
    return mj.CATALOGUE[name]["thrust_N"] * scale


def size(concept: Concept | str, req: Requirements | None = None, option=None, *, limit_kg: float = 900.0,
         tolerance_kg: float = 0.2) -> Design:
    """Close the take-off mass of a layout on the mission (see the module text): the smallest mass M at which the
    aircraft built for M (engines, wing, fuselage, lift system) and fuelled for the mission weighs no more than M.
    Found by a scan up from the bay's load and a bisection; ``feasible`` is False when none exists below ``limit_kg``."""
    import albatros_bay as bay_mod

    concept = {**SET_ASIDE, **CONCEPTS}[concept] if isinstance(concept, str) else concept
    req = req or Requirements()
    option = option or bay_mod.options(req.cargo_kg)["hexacopter"]

    def at(mtow):
        """(mass the design needs - mtow, the design flown) at a take-off mass; (+inf, None) when it cannot fly."""
        factor = concept.cruise_factor
        while True:
            fuel, fl = 0.3 * mtow, None
            for _ in range(3):                                        # the fuselage is as long as its tanks: settle the fuel
                d = _build(concept, req, option, mtow, fuel, 0.0, factor)
                fl = _fuel_at(d, req, option.dropped_kg, mtow)
                if fl is None or abs(fl.fuel_kg - fuel) < 0.1:
                    break
                fuel = fl.fuel_kg
            if fl is None and concept.kind != "tailsitter" and factor < 0.9:
                factor += 0.05                                        # the cruise engines are too small for this altitude
                continue
            break
        if fl is None:
            return math.inf, None
        batt = 0.0
        if concept.kind == "lift_cruise":
            batt = max(fl.battery_wh / (MASS["pack_wh_kg"] * MASS["pack_usable"]), 1.15 * d.hover_power_w / MASS["pack_w_kg"])
        d = _build(concept, req, option, mtow, fuel, batt, factor)
        need = sum(d.budget.values())
        d = replace(d, airframe=replace(d.airframe, dry_mass_kg=mtow - fuel), flight=fl)
        return need - mtow, d

    grid = np.geomspace(2.0 * option.dropped_kg + 10.0, limit_kg, 16)
    lo = hi = best = None
    for m in grid:
        g, d = at(float(m))
        if g <= 0:
            hi, best = float(m), d
            break
        lo = float(m)
    if hi is None:
        g, d = at(float(grid[-1]))
        if d is None:
            d = _build(concept, req, option, float(grid[-1]), 0.3 * float(grid[-1]), 0.0)
            d.flight = fly(replace(d, airframe=replace(d.airframe, dry_mass_kg=0.7 * d.mtow_kg)), req, d.fuel_kg, option.dropped_kg)
        d.feasible = False
        d.notes.append(f"no take-off mass below {limit_kg:.0f} kg closes the mission")
        return d
    while lo is not None and hi - lo > tolerance_kg:
        mid = 0.5 * (lo + hi)
        g, d = at(mid)
        if g <= 0:
            hi, best = mid, d
        else:
            lo = mid
    best.feasible = best.flight.feasible
    best.notes += best.flight.notes
    return best


def _fuel_at(d: Design, req: Requirements, dropped_kg: float, mtow: float):
    """The flight at a fixed take-off mass ``mtow`` with the smallest fuel load that keeps the reserve (the fuel
    replaces dry mass); None when no load below 75 % of the mass does."""
    def run(fk):
        try:
            f = fly(replace(d, fuel_kg=fk, airframe=replace(d.airframe, dry_mass_kg=mtow - fk), mtow_kg=mtow), req, fk, dropped_kg)
        except ValueError:                                            # it burned more than it weighs
            return None, -1.0
        return f, (f.fuel_left_kg - req.reserve * fk) if math.isfinite(f.fuel_left_kg) else -1.0

    lo, hi = 0.0, 0.6 * mtow
    if run(hi)[1] < 0:
        return None
    while hi - lo > 0.03:
        mid = 0.5 * (lo + hi)
        if run(mid)[1] >= 0:
            hi = mid
        else:
            lo = mid
    return run(hi)[0]


def summary(d: Design) -> dict:
    f = d.flight
    e = d.engine
    W = d.mtow_kg * G
    hover_fuel = 0.0 if d.concept.kind == "catapult" else sum(
        v["fuel"] for k, v in f.phases.items() if any(s in k for s in ("take-off", "hover", "landing", "transition", "restart")))
    return {"take-off mass [kg]": round(d.mtow_kg, 1), "engines": f"{d.engines} x {_static(d):.0f} N" + (f" + {d.concept.lift_engines} x {_static_of(*d.lift_engine):.0f} N lift" if d.lift_engine else ""),
            "nearest catalogue class": f"{d.engine_name} x {d.engine_scale:.2f}",
            "installed thrust / weight": round((d.engines * _static(d) + (d.concept.lift_engines * _static_of(*d.lift_engine) if d.lift_engine else 0.0)) / W, 2),
            "engines running in cruise": f.cruise_engines, "wing area [m²]": round(d.wing_area_m2, 2),
            "wing loading [kg/m²]": round(d.mtow_kg / d.wing_area_m2), "stall speed at take-off [m/s]": round(d.airframe.stall_speed(d.mtow_kg), 1),
            "span [m]": round(d.params["span"] / 1000, 2), "fuselage length [m]": round(d.params["fuselage_length"] / 1000, 2),
            "cd0": round(d.airframe.cd0, 4), "lift rotor Ø [m]": round(d.rotor_diameter_m, 2) if d.rotor_diameter_m else "-",
            "lift battery [kg]": round(d.battery_kg, 1) if d.battery_kg else "-",
            "hover power [kW]": round(d.hover_power_w / 1000, 1) if d.hover_power_w else "-",
            "fuel [kg]": round(f.fuel_kg, 1), "fuel [L]": round(f.fuel_kg / 0.8, 1), "fuel fraction": round(f.fuel_kg / d.mtow_kg, 3),
            "of it thrust-borne [kg]": round(hover_fuel, 1), "reserve left [kg]": round(f.fuel_left_kg, 1),
            "cruise out [m/s]": round(f.cruise_speed_out, 0), "cruise home [m/s]": round(f.cruise_speed_back, 0),
            "time to the vessel [min]": round(f.time_to_vessel_s / 60, 1), "sortie [min]": round(f.total_time_s / 60, 1),
            "spool-up [s]": e.spool_time, "feasible": d.feasible}


def _static(d: Design) -> float:
    return _static_of(d.engine_name, d.engine_scale)


def compare(req: Requirements | None = None, option=None, concepts=None) -> dict:
    """Size every layout for the same requirements and bay: ``{key: Design}``."""
    return {k: size(c, req, option) for k, c in (concepts or CONCEPTS).items()}


def table(designs: dict):
    import pandas as pd
    rows = {}
    for k, d in designs.items():
        rows[d.concept.name] = summary(d)
    return pd.DataFrame(rows)


GATES = {"gust_excursion_m": 1.0, "hover_wind_limit": 20.0, "deck_height_error_m": 0.25, "tip_wind": 20.0,
         "catapult_stroke_m": 12.0}


def launch(d: Design, *, margin: float = 1.2, catapult_g: float = 8.0, friction: float = 0.05, approach: float = 1.3,
           braking_g: float = 0.35) -> dict:
    """How a wing-borne Albatros leaves and returns: the stall speed at the take-off mass, the catapult's end speed
    (``margin`` x stall, no wind over the deck), its energy and its stroke at ``catapult_g``; the ground roll from a
    runway on its own thrust (to the same speed, the mean acceleration at 0.7 of it); the approach speed at the
    landing mass and the stop from it at ``braking_g`` (skids on grass or a wire: 1 g and more)."""
    vs = d.airframe.stall_speed(d.mtow_kg)
    v = margin * vs
    T = d.engines * scaled(d.engine_name, d.engine_scale, 0.0)[1].full(0.7 * v)[0]
    a = (T - 0.5 * d.airframe.drag(v, d.mtow_kg)) / d.mtow_kg - friction * G
    m_land = d.flight.landing_mass_kg if d.flight else d.dry_kg
    va = approach * d.airframe.stall_speed(m_land)
    return {"stall_speed": vs, "launch_speed": v, "catapult_energy_kj": 0.5 * d.mtow_kg * v * v / 1000,
            "catapult_stroke_m": v * v / (2 * catapult_g * G), "catapult_g": catapult_g,
            "stroke_with_10ms_wind_m": max(v - 10.0, 0.0) ** 2 / (2 * catapult_g * G),
            "runway_roll_m": v * v / (2 * a) if a > 0 else float("inf"), "landing_mass_kg": m_land, "approach_speed": va,
            "release_speed": 1.3 * d.airframe.stall_speed(d.mtow_kg - 0.45 * d.fuel_kg),
            "landing_roll_m": va * va / (2 * braking_g * G)}


def choose(designs: dict, **conditions) -> dict:
    """The layout for Albatros: the one that burns the least fuel among those that pass their gates.

    A wing-borne layout must leave a catapult of at most ``catapult_stroke_m`` at 8 g with no wind over the deck. A thrust-borne one must hold
    its place in the hover within ``gust_excursion_m`` in the design side gust with a hover wind limit of at least
    ``hover_wind_limit`` m/s, follow a heaving deck within ``deck_height_error_m`` (rms), and stand on a rolling deck
    in ``tip_wind`` m/s before it is lashed. Jets pointing down are not a gate (Albatros may land ashore) but stay a
    note: no settling on water, a hot deck.

    ``conditions`` go to ``albatros_gust`` (wind, gust, deck_amplitude, deck_period, roll_deg). Returns the key, the
    ranking by fuel, each layout's failed gates and its notes."""
    import albatros_gust as gu

    failed, fuel, notes = {}, {}, {}
    for k, d in designs.items():
        f, nt = [], []
        if not d.feasible:
            f.append("the mission does not close")
        if d.concept.kind == "catapult":
            la = launch(d)
            if la["catapult_stroke_m"] > GATES["catapult_stroke_m"]:
                f.append(f"needs a {la['catapult_stroke_m']:.1f} m catapult stroke at {la['catapult_g']:g} g")
            nt.append(f"lands ashore at {la['approach_speed']:.0f} m/s; releases the multirotor at {la['release_speed']:.0f} m/s, not from a hover")
        else:
            m = d.flight.landing_mass_kg
            hg = gu.hover_gust(d, wind=conditions.get("wind", 12.0), gust=conditions.get("gust", 8.0), mass_kg=m)
            dh = gu.deck_heave(d, amplitude=conditions.get("deck_amplitude", 1.0), period=conditions.get("deck_period", 8.0), mass_kg=m)
            ds = gu.deck_stand(d, roll_deg=conditions.get("roll_deg", 8.0), mass_kg=m)
            if hg["peak_excursion_m"] > GATES["gust_excursion_m"]:
                f.append(f"drifts {hg['peak_excursion_m']:.1f} m in the gust")
            if hg["wind_limit"] < GATES["hover_wind_limit"]:
                f.append(f"hover wind limit {hg['wind_limit']:.0f} m/s")
            if dh["rms_error_m"] > GATES["deck_height_error_m"]:
                f.append(f"{dh['rms_error_m']:.2f} m rms off the heaving deck")
            if ds["tip_wind_at_roll"] < GATES["tip_wind"]:
                f.append(f"tips on the deck in {ds['tip_wind_at_roll']:.0f} m/s of wind")
            if d.concept.kind != "lift_cruise":
                nt.append("hot jets straight down: no settling on water, a cleared deck")
        failed[k], fuel[k], notes[k] = f, d.flight.fuel_kg, nt
    ranking = sorted(designs, key=lambda k: fuel[k])
    passing = [k for k in ranking if not failed[k]]
    return {"choice": passing[0] if passing else None, "by_fuel": ranking, "failed": failed, "notes": notes, "fuel_kg": fuel}


def top_speed(d: Design, req: Requirements, mass_kg: float | None = None) -> float:
    """Full throttle on every podded engine at the cruise altitude."""
    _, _, uh = scaled(d.engine_name, d.engine_scale, req.cruise_altitude_m)
    rho = density(req.cruise_altitude_m)
    m = mass_kg or d.mtow_kg
    n = d.engines
    lo, hi = 1.1 * d.airframe.stall_speed(m, rho), float(uh.V[-1])
    f = lambda V: n * uh.full(V)[0] - d.airframe.drag(V, m, rho)        # noqa: E731
    if f(hi) > 0:
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return 0.5 * (lo + hi)


__all__ = ["MASS", "CLASSES", "Requirements", "Concept", "CONCEPTS", "Design", "Flight", "units", "density", "pick_engines", "scaled",
           "best_range_speed", "cruise_engines", "fly", "fuel_for", "size", "summary", "compare", "table", "top_speed", "choose", "launch", "GATES", "SET_ASIDE"]
