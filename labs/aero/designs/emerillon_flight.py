"""ÉMERILLON — how fast, on which parts, for how much (notebook 37).

Reduced models, honest about what they are, on the tools the other notebooks validated:

- ``Config`` — one build: motor, propeller, ESC, the 6S pack and how many of it in series and in parallel
  (``emerillon_market`` keys).
- ``geometry`` — the airframe that configuration needs: the body's diameter from the pack's cross-section, its
  length from the packs in line, the arm radius from the propeller. ``emerillon.Emerillon`` builds the same numbers
  in CAD (``cad_parameters``).
- ``mass_budget`` — ``og.mass.Budget`` of the bought parts and the printed ones (``printed_masses``).
- ``drag_buildup`` — flat-plate friction with form factors (Hoerner / Raymer) on the body, the arms, the motor pods
  and the fins, the pods and arms in the propellers' slipstream, an interference and excrescence factor, a
  compressibility correction. A hand estimate: Kestrel's lesson is that such a build-up runs low against RANS, so
  ``DRAG_MARGIN`` scales it and the CFD chapter is what replaces it.
- ``propeller_map`` — Boreas's BEMT over advance ratio and rotational tip Mach, with a section that knows the Mach
  number it works at (``MachAirfoil``: Prandtl–Glauert on the lift slope, Korn's drag divergence, Lock's drag rise).
  Boreas's own section has no compressibility; at 500 km/h the tips are transonic and that is most of the story.
- ``drive`` — four motors (``og.motors``-style first-order model) on the pack with its sag, under the motor's, the
  ESC's and the pack's current limits.
- ``trim`` — level flight: the body at a few degrees of incidence carries the weight (a slender body's normal force
  and the arms as small wings), the thrust pays the drag.
- ``top_speed``, ``speed_run``, ``evaluate`` — where thrust meets drag, the run from a standstill, the energy for two
  passes in opposite directions, the motor's temperature rise, the propeller against its maker's speed limit.
- ``candidates``, ``trade``, ``pareto``, ``picks`` — every build the catalogue allows, its speed and its price, the
  front between them, and three picks: the cheapest that flies the run, the knee, the fastest.

Units SI. The aircraft flies nose-first along its own axis; ``x`` runs aft from the nose tip.
"""
from __future__ import annotations

import bisect
import functools
import itertools
import math
from dataclasses import asdict, dataclass

import numpy as np
from ogma import atmosphere as og_atm
from ogma import mass as og_mass
from ogma.aero import cf_turbulent
from ogma.units import G, NU_AIR as NU, RHO0 as RHO
from vegeta import boreas

import emerillon_market as mk

A_SOUND = 340.3                    # ISA sea level [m/s]

DRAG_MARGIN = 1.6                  # the hand build-up runs low against RANS (Kestrel: 20-47 % on clean airframes; here
                                   # separated arm roots, cooling flow and four pods in propeller wash) — assumed
PRINT_FACTOR = 1.8                 # printed parts against their computed skin mass: ribs, bosses, joints, infill — Kestrel's
                                   # mini raptors weighed about twice their budget (exp/49); assumed
INTERFERENCE = 1.25                # junctions: arm on body, pod on arm (Hoerner-level)
CELL_OCV = 4.0                     # cell voltage through the run, a little below a full charge [V]
CELL_V_MIN = 3.3                   # the pack is not pulled below this under load [V] (assumed limit)
R_CELL_MOHM_AH = 7.5               # cell resistance x capacity for a 100C race cell [mOhm Ah] (assumed)
R_LEAD = 0.006                     # ESC, bullet connectors and wire per motor [Ohm] (assumed)
USABLE = 0.80                      # share of the pack's energy a run may take
STATIC_TW_MIN = 2.0                # thrust to weight standing on the tail: it has to hover and tip over (requirement)
MOTOR_DT_MAX = 90.0                # winding temperature rise allowed over one pass [K] (from 40 C to 130 C; assumed)
PASS_LENGTH = 100.0                # the timed distance of a record pass [m]
RUN_IN = 150.0                     # held at full speed before and after the gates [m]
HOVER_S = 50.0                     # take-off, turn and landing per pass, at hover power [s]


@dataclass(frozen=True)
class Config:
    """One build. The defaults are ÉMERILLON as designed (``DESIGN``): the knee of the trade on shelf parts."""
    motor: str = "eco2807-1300"
    propeller: str = "apc7x10"
    esc: str = "speedybee-f405v4-55"
    battery: str = "tattu-r6-1700-160c"
    n_series: int = 1
    n_parallel: int = 1

    @property
    def cells(self) -> int:
        return 6 * self.n_series

    @property
    def packs(self) -> int:
        return self.n_series * self.n_parallel

    def label(self) -> str:
        return f"{self.motor} / {self.propeller} / {self.cells}S{self.n_parallel}P {self.battery} / {self.esc}"


# ================================================================================================= geometry and mass
WALL = 0.0014                      # printed skin [m]
NOSE_FINENESS, TAIL_FINENESS = 2.6, 3.2
STACK_BAY = 0.050                  # flight controller, receiver, ESC board [m]
ARM_TC = 0.22                      # arm section thickness / chord
FIN_SPAN, FIN_CHORD, FIN_T = 0.055, 0.060, 0.003
PROP_GAP = 0.010                   # propeller plane ahead of the arm's leading edge [m]


def geometry(c: Config) -> dict:
    """The airframe a configuration needs [m]: body diameter and length, arm radius and section, pod and fin sizes,
    wetted areas."""
    bat, prop, motor = mk.part(c.battery), mk.part(c.propeller), mk.part(c.motor)
    L, W, H = (s / 1000.0 for s in bat.size_mm)
    d = math.hypot(W, H) + 0.003 + 2 * WALL                          # the pack's diagonal, a little air, two skins
    cyl = c.packs * (L + 0.006) + STACK_BAY
    nose, tail = NOSE_FINENESS * d, TAIL_FINENESS * d
    D = prop.d_in * 0.0254
    r_arm = max((D + 0.006) / math.sqrt(2), d / 2 + D / 2 + 0.004)   # neighbours clear each other; tips clear the body
    pod_d = motor.d_mm / 1000.0 + 0.004
    pod_l = motor.h_mm / 1000.0 + 2.8 * pod_d                        # the motor, a spinner ahead of it, a cone behind
    arm_t = max(0.008, 0.30 * pod_d)
    arm_c = arm_t / ARM_TC
    arm_l = r_arm - d / 2 - pod_d / 2
    # wetted areas: an ogive nose ~ 0.70 and a cone-ish tail ~ 0.60 of the enclosing cylinder's side
    s_body = math.pi * d * (0.70 * nose + cyl + 0.60 * tail)
    return {"d": d, "nose": nose, "cyl": cyl, "tail": tail, "length": nose + cyl + tail, "prop_d": D, "r_arm": r_arm,
            "pod_d": pod_d, "pod_l": pod_l, "arm_t": arm_t, "arm_c": arm_c, "arm_l": arm_l,
            "x_arm": nose + 0.5 * cyl, "x_prop": nose + 0.5 * cyl - 0.5 * arm_c - PROP_GAP,
            "s_body": s_body, "s_arms": 4 * 2.04 * arm_c * arm_l, "s_pods": 4 * math.pi * pod_d * 0.75 * pod_l,
            "s_fins": 4 * 2.02 * FIN_SPAN * FIN_CHORD, "frontal": math.pi * d * d / 4}


def cad_parameters(c: Config) -> dict:
    """``emerillon.Emerillon``'s parameters [mm] for a configuration."""
    g, motor = geometry(c), mk.part(c.motor)
    return {"body_diameter": round(g["d"] * 1e3, 1), "nose_length": round(g["nose"] * 1e3, 1), "bay_length": round(g["cyl"] * 1e3, 1),
            "tail_length": round(g["tail"] * 1e3, 1), "arm_radius": round(g["r_arm"] * 1e3, 1), "arm_chord": round(g["arm_c"] * 1e3, 1),
            "arm_thickness": round(g["arm_t"] * 1e3, 1), "pod_diameter": round(g["pod_d"] * 1e3, 1),
            "pod_length": round(g["pod_l"] * 1e3, 1), "motor_diameter": motor.d_mm, "motor_height": motor.h_mm,
            "prop_diameter": round(g["prop_d"] * 1e3, 1)}


def printed_masses(c: Config) -> dict:
    """Flying grams of each filament: PETG skins (nose, bay, tail, fins), PA6-CF arms with their pods and the two
    body rings they bolt to, TPU bumpers."""
    g = geometry(c)
    petg, pa6 = mk.part("petg-hf"), mk.part("pa6-cf")
    skin = (g["s_body"] * WALL * PRINT_FACTOR + 4 * FIN_SPAN * FIN_CHORD * FIN_T) * petg.density
    arm = 4 * 0.68 * g["arm_c"] * g["arm_t"] * g["arm_l"] * 0.55 * pa6.density * PRINT_FACTOR / 1.3   # a hollow arm, 55 % solid
    pods = 4 * math.pi * g["pod_d"] * 0.75 * g["pod_l"] * 0.0016 * pa6.density * PRINT_FACTOR / 1.3
    rings = 2 * math.pi * g["d"] * 0.012 * 0.004 * pa6.density
    return {"petg-hf": round(skin * 1e3, 1), "pa6-cf": round((arm + pods + rings) * 1e3, 1), "tpu": 8.0}


def mass_budget(c: Config) -> og_mass.Budget:
    g = geometry(c)
    motor, prop, esc, bat = (mk.part(k) for k in (c.motor, c.propeller, c.esc, c.battery))
    b = og_mass.Budget()
    I = og_mass.Item
    for i, a in enumerate((45, 135, 225, 315)):
        y, z = g["r_arm"] * math.cos(math.radians(a)), g["r_arm"] * math.sin(math.radians(a))
        b.add(I(f"motor {i + 1}", motor.mass_g / 1e3, (g["x_arm"], y, z), group="propulsion"),
              I(f"propeller {i + 1}", prop.mass_g / 1e3, (g["x_prop"], y, z), group="propulsion"))
        if esc.count == 4:
            b.add(I(f"ESC {i + 1}", esc.mass_g / 1e3, (g["x_arm"], y / 2, z / 2), group="propulsion"))
    x_bay = g["nose"]
    if esc.count == 1:
        b.add(I("ESC board", esc.mass_g / 1e3, (x_bay + 0.02, 0, 0), group="propulsion"))
    L = mk.part(c.battery).size_mm[0] / 1e3 + 0.006
    for k in range(c.packs):
        b.add(I(f"pack {k + 1}", bat.mass_g / 1e3, (x_bay + STACK_BAY + (k + 0.5) * L, 0, 0), group="battery"))
    for p in mk.FIXED:
        if p.group == "fc" and esc.with_fc:
            continue
        b.add(I(p.name, p.mass_g / 1e3, (0.6 * g["nose"] if p.group in ("video", "gps") else x_bay + 0.02, 0, 0), group="avionics"))
    pm = printed_masses(c)
    b.add(I("skins (PETG)", pm["petg-hf"] / 1e3, (0.52 * g["length"], 0, 0), group="airframe"),
          I("arms, pods, rings (PA6-CF)", pm["pa6-cf"] / 1e3, (g["x_arm"], 0, 0), group="airframe"),
          I("bumpers (TPU)", pm["tpu"] / 1e3, (g["length"], 0, 0), group="airframe"),
          I("power wiring", 0.012 * c.packs + 0.020, (x_bay + 0.03, 0, 0), group="propulsion"))
    return b


# =============================================================================================================== air
def air_at(altitude: float = 0.0) -> dict:
    """The standard atmosphere at ``altitude`` [m] as the flight model reads it: rho, a (speed of sound), nu, T_C."""
    if not altitude:
        return SEA_LEVEL
    a = og_atm.atmosphere(float(altitude))
    return {"rho": a["rho"], "a": a["a"], "nu": a["nu"], "T_C": a["T_C"], "h": float(altitude)}


SEA_LEVEL = {"rho": RHO, "a": A_SOUND, "nu": NU, "T_C": 15.0, "h": 0.0}


# ============================================================================================================== drag
def drag_buildup(c: Config, V: float, slip: float = 1.0, margin: float | None = None, air: dict | None = None) -> dict:
    """Drag area CdA [m2] at ``V``: friction x form factor per component, the arms and pods at the slipstream's dynamic
    pressure (``slip`` = q_slipstream / q), interference, then ``margin``. Returns the components and the total."""
    air = air or SEA_LEVEL
    g, M = geometry(c), V / air["a"]
    margin = DRAG_MARGIN if margin is None else margin

    def cf(length):
        return cf_turbulent(max(V * length / air["nu"], 1e4), M)

    f = g["length"] / g["d"]
    body = cf(g["length"]) * (1 + 60 / f ** 3 + f / 400) * g["s_body"]                       # Raymer, body of revolution
    arms = cf(g["arm_c"]) * (1 + 2 * ARM_TC + 60 * ARM_TC ** 4) * g["s_arms"] * slip         # Hoerner, a strut section
    fp = g["pod_l"] / g["pod_d"]
    pods = cf(g["pod_l"]) * (1 + 60 / fp ** 3 + fp / 400) * g["s_pods"] * slip
    fins = cf(FIN_CHORD) * (1 + 2 * FIN_T / FIN_CHORD) * g["s_fins"]
    base = 0.12 * math.pi * (0.012 ** 2) / 4 * 4                                             # the tail cone's blunt end (wires, antenna)
    clean = body + (arms + pods) * INTERFERENCE + fins + base
    return {"body": body, "arms": arms * INTERFERENCE, "pods": pods * INTERFERENCE, "fins": fins, "base": base,
            "clean": clean, "CdA": clean * margin, "Cd_frontal": clean * margin / g["frontal"], "margin": margin}


# ====================================================================================================== the propeller
@dataclass(frozen=True)
class MachAirfoil(boreas.Airfoil):
    """Boreas's section at the Mach number of each blade station: Prandtl–Glauert on the lift slope (held from M 0.75),
    Korn's drag-divergence Mach number for the station's thickness and lift, Lock's fourth-power drag rise above the
    critical Mach number, capped where a thin section's wave drag levels off past Mach 1. ``mach`` and ``tc`` are per
    station of the BEMT (the same count as its ``n_stations``)."""
    mach: tuple = ()
    tc: tuple = ()
    korn: float = 0.87
    wave_cap: float = 0.10

    def coefficients(self, alpha_rad):
        a = np.asarray(alpha_rad, dtype=float)
        M, tc = np.asarray(self.mach, dtype=float), np.asarray(self.tc, dtype=float)
        if len(M) != a.size:                                         # a solver on another station count (the mocks coarsen it)
            u, v = np.linspace(0.0, 1.0, a.size), np.linspace(0.0, 1.0, len(M))
            M, tc = np.interp(u, v, M), np.interp(u, v, tc)
        pg = 1.0 / np.sqrt(1.0 - np.minimum(M, 0.75) ** 2)
        lin = self.cl_alpha * pg * (a - math.radians(self.alpha0_deg))
        cl_max = self.cl_max * np.clip(1.0 - 0.6 * (M - 0.6), 0.6, 1.0)                      # shock stall trims the peak
        cl = np.clip(lin, -cl_max, cl_max)
        cd = self.cd0 + self.k * cl ** 2
        stalled = np.abs(lin) > cl_max
        cd = np.where(stalled, cd + 0.5 * (np.abs(a - math.radians(self.alpha0_deg)) - cl_max / (self.cl_alpha * pg)), cd)
        m_crit = self.korn - tc - np.abs(cl) / 10.0 - 0.108
        cd = cd + np.minimum(20.0 * np.maximum(M - m_crit, 0.0) ** 4, self.wave_cap)
        return cl, cd


SECTION = dict(cl_alpha=2 * math.pi * 0.9, alpha0_deg=-2.0, cl_max=1.1, cd0=0.014, k=0.035)   # thin cambered, Re ~ 3e5
N_STATIONS = 30
MT_GRID = (0.25, 0.45, 0.60, 0.72, 0.82, 0.90, 0.98, 1.06, 1.15)


def blade(prop_key: str, hub_d: float = 0.0) -> boreas.Propeller:
    """The catalogue propeller as Boreas geometry: constant geometric pitch, a generic planform from the row's chord
    fractions (root, maximum, tip, of the diameter). The makers publish no blade tables for these."""
    p = mk.part(prop_key)
    D, P = p.d_in * 0.0254, p.p_in * 0.0254
    cr, cm, ct = (x * D for x in p.chord)
    return boreas.Propeller.from_pitch(p.name, D, P, p.blades, chord_root_m=cr, chord_max_m=cm, chord_tip_m=ct,
                                       hub_radius_m=max(0.15 * D / 2, hub_d / 2), mass_kg=p.mass_g / 1e3)


def solve_point(prop_key: str, rpm: float, V: float, rho: float = RHO, compressible: bool = True):
    """One BEMT point of a catalogue propeller, with the Mach-aware section (or Boreas's plain one)."""
    pr, p = blade(prop_key), mk.part(prop_key)
    if not compressible:
        return boreas.solve(pr, boreas.Airfoil(**SECTION), rpm, V, rho, n_stations=N_STATIONS, speed_of_sound=A_SOUND)
    r, _, _, _ = pr.stations(N_STATIONS)
    mach = np.hypot(V, rpm * 2 * math.pi / 60 * r) / A_SOUND
    tc = np.interp(r / pr.radius, [0.15, 1.0], [0.12, p.tc_tip])
    return boreas.solve(pr, MachAirfoil(**SECTION, mach=tuple(mach), tc=tuple(tc)), rpm, V, rho, n_stations=N_STATIONS,
                        speed_of_sound=A_SOUND)


@functools.lru_cache(maxsize=None)
def propeller_map(prop_key: str, n_j: int = 34) -> dict:
    """CT and CP of a propeller over advance ratio J and rotational tip Mach number Mt = pi n D / a (the two numbers
    the BEMT depends on once Reynolds effects are left out). Lists, for the fast bilinear lookup of ``coefficients``."""
    p = mk.part(prop_key)
    D = p.d_in * 0.0254
    js = [round(x, 4) for x in np.linspace(0.0, 1.25 * p.p_in / p.d_in + 0.25, n_j)]
    ct, cp = [], []
    for mt in MT_GRID:
        n = mt * A_SOUND / (math.pi * D)
        row = [solve_point(prop_key, n * 60, j * n * D) for j in js]
        ct.append([o.ct for o in row])
        cp.append([o.cp for o in row])
    return {"J": js, "Mt": list(MT_GRID), "CT": ct, "CP": cp, "D": D}


def coefficients(pm: dict, J: float, Mt: float) -> tuple:
    """(CT, CP) from a map, bilinear, held at the edges."""
    js, ms = pm["J"], pm["Mt"]
    J, Mt = min(max(J, js[0]), js[-1]), min(max(Mt, ms[0]), ms[-1])
    i = min(max(bisect.bisect_right(js, J) - 1, 0), len(js) - 2)
    k = min(max(bisect.bisect_right(ms, Mt) - 1, 0), len(ms) - 2)
    u, w = (J - js[i]) / (js[i + 1] - js[i]), (Mt - ms[k]) / (ms[k + 1] - ms[k])
    out = []
    for t in (pm["CT"], pm["CP"]):
        out.append((1 - w) * ((1 - u) * t[k][i] + u * t[k][i + 1]) + w * ((1 - u) * t[k + 1][i] + u * t[k + 1][i + 1]))
    return out[0], out[1]


# ========================================================================================================== the drive
def pack_resistance(c: Config) -> float:
    """The whole battery's resistance [Ohm]: cells in series, packs in parallel; a higher C rating is a lower cell
    resistance (to the power one half: the ratings flatter)."""
    bat = mk.part(c.battery)
    r_cell = R_CELL_MOHM_AH / 1000.0 / bat.ah * math.sqrt(100.0 / bat.c_rating)
    return r_cell * c.cells / c.n_parallel


def drive(c: Config, V: float, rho: float | None = None, *, air: dict | None = None, throttle: float | None = None) -> dict:
    """Four motors at full throttle at airspeed ``V`` — or at the throttle the tightest current limit allows (motor
    burst, ESC rating, the pack at ``CELL_V_MIN``): rpm, per-motor current and thrust, the bus voltage with its sag,
    the electrical and shaft powers, which limit holds. ``air`` (``air_at``) moves it to altitude; ``throttle`` caps
    the throttle (a part-throttle point)."""
    air = air or SEA_LEVEL
    rho, a_snd = (air["rho"] if rho is None else rho), air["a"]
    motor, esc, pm = mk.part(c.motor), mk.part(c.esc), propeller_map(c.propeller)
    D = pm["D"]
    kt = 60.0 / (2 * math.pi * motor.kv)
    R = motor.r_ohm + R_LEAD
    r_pack = pack_resistance(c)
    i_pack_max = c.cells * (CELL_OCV - CELL_V_MIN) / r_pack
    i_lim = min(motor.max_a, esc.max_a)

    def torque(rpm):
        n = rpm / 60.0
        _, cp = coefficients(pm, V / (n * D), math.pi * n * D / a_snd)
        return cp * rho * n ** 3 * D ** 5 / (2 * math.pi * n)

    def at(volts):                                                    # the rpm where the motor meets the propeller
        lo, hi = 200.0, motor.kv * volts
        for _ in range(34):
            n = 0.5 * (lo + hi)
            lo, hi = (n, hi) if kt * ((volts - n / motor.kv) / R - motor.i0_a) > torque(n) else (lo, n)
        return n, (volts - n / motor.kv) / R

    def state(u):                                                     # throttle u: the bus sags under 4 motors' input
        v_bus = c.cells * CELL_OCV
        for _ in range(12):
            rpm, i_m = at(u * v_bus)
            i_pack = 4 * u * max(i_m, 0.0)
            v_bus = 0.5 * v_bus + 0.5 * (c.cells * CELL_OCV - i_pack * r_pack)
        return rpm, i_m, i_pack, v_bus

    u, limit = (1.0, "voltage") if throttle is None else (float(throttle), "throttle")
    u_top = u
    rpm, i_m, i_pack, v_bus = state(u)
    if i_m > i_lim or i_pack > i_pack_max:
        limit = "motor current" if motor.max_a <= esc.max_a else "ESC current"
        if i_pack / i_pack_max > i_m / i_lim:
            limit = "battery"
        lo, hi = 0.05, u_top
        for _ in range(18):
            u = 0.5 * (lo + hi)
            rpm, i_m, i_pack, v_bus = state(u)
            lo, hi = (lo, u) if (i_m > i_lim or i_pack > i_pack_max) else (u, hi)
        u = lo
        rpm, i_m, i_pack, v_bus = state(u)
    n = rpm / 60.0
    J, Mt = V / (n * D), math.pi * n * D / a_snd
    ct, cp = coefficients(pm, J, Mt)
    T, P = ct * rho * n ** 2 * D ** 4, cp * rho * n ** 3 * D ** 5
    return {"V": V, "throttle": u, "limit": limit, "rpm": rpm, "J": J, "tip_mach": math.hypot(Mt, V / a_snd), "rho": rho,
            "current": i_m, "pack_current": i_pack, "bus_v": v_bus, "cell_v": v_bus / c.cells, "thrust": 4 * T, "shaft_w": 4 * P,
            "electric_w": i_pack * v_bus, "prop_eff": (T * V / P) if P > 0 else 0.0,
            "copper_w": i_m ** 2 * motor.r_ohm, "disk_area": math.pi * D * D / 4}


def _slip(d: dict) -> float:
    """Dynamic pressure in the slipstream over the free stream's, from one propeller's thrust (momentum theory)."""
    V, T, rho = max(d["V"], 1.0), max(d["thrust"] / 4, 0.0), d["rho"]
    vi = 0.5 * (-V + math.sqrt(V * V + 2 * T / (rho * d["disk_area"])))
    return ((V + 2 * vi) / V) ** 2


def normal_force_area(c: Config, alpha: float) -> float:
    """The airframe's normal force over the dynamic pressure [m2] at an angle of attack ``alpha`` [rad] of the body
    axis: the slender body's potential term (2 x the frontal area x sin a cos a), the crossflow on its planform
    (Cd 1.2 x sin2 a), and the four arms as two small low-aspect-ratio wings (a lift slope of 2.5 per radian on
    their horizontal projection). The propellers' own normal force is left out."""
    g = geometry(c)
    plan = g["d"] * (0.67 * g["nose"] + g["cyl"] + 0.55 * g["tail"])
    arms = 2 * g["arm_c"] * g["arm_l"]
    sa, ca = math.sin(alpha), math.cos(alpha)
    return 2 * g["frontal"] * sa * ca + 1.2 * plan * sa * sa + 2.5 * arms * sa * ca


def trim(c: Config, V: float, mass: float, CdA0: float, rho: float = RHO) -> tuple:
    """Level flight at ``V``: the body's angle of attack [rad] at which the thrust (along the body) and the body's own
    normal force carry the weight while the thrust's forward part pays the drag, and the thrust that takes [N]."""
    q, W = 0.5 * rho * V * V, mass * G

    def thrust(a):
        return q * (CdA0 + normal_force_area(c, a) * math.sin(a)) / math.cos(a)

    lo, hi = 0.0, math.radians(89.5)
    for _ in range(40):
        a = 0.5 * (lo + hi)
        lo, hi = (a, hi) if thrust(a) * math.sin(a) + q * normal_force_area(c, a) * math.cos(a) < W else (lo, a)
    return hi, thrust(hi)


def balance(c: Config, V: float, mass: float | None = None, altitude: float = 0.0) -> dict:
    """Thrust against what level flight needs at ``V`` (and ``altitude``): the drag (pods and arms in the slipstream)
    and the weight, carried by the body flying at a small angle of attack (``trim``)."""
    m = mass_budget(c).mass if mass is None else mass
    air = air_at(altitude)
    d = drive(c, V, air=air)
    q = 0.5 * air["rho"] * V * V
    db = drag_buildup(c, V, slip=min(_slip(d), 4.0), air=air)
    alpha, need = trim(c, V, m, db["CdA"], air["rho"])
    return {**d, "drag": q * db["CdA"], "CdA": db["CdA"], "need": need, "excess": d["thrust"] - need,
            "alpha_deg": math.degrees(alpha), "mass": m}


def top_speed(c: Config, mass: float | None = None, altitude: float = 0.0) -> dict:
    """The fastest level speed (at ``altitude``): the last airspeed where the thrust still covers drag and weight."""
    m = mass_budget(c).mass if mass is None else mass
    lo, hi = 15.0, 290.0
    if balance(c, lo, m, altitude)["excess"] < 0:
        return {**balance(c, lo, m, altitude), "V": 0.0, "kmh": 0.0}
    for _ in range(22):
        V = 0.5 * (lo + hi)
        lo, hi = (V, hi) if balance(c, V, m, altitude)["excess"] > 0 else (lo, V)
    b = balance(c, lo, m, altitude)
    return {**b, "kmh": lo * 3.6}


def top_speed_with(c: Config, drag_margin: float | None = None, print_factor: float | None = None) -> dict:
    """``top_speed`` with another drag margin and/or printed-mass factor (the two assumptions the hand model leans on)."""
    global DRAG_MARGIN, PRINT_FACTOR
    keep = DRAG_MARGIN, PRINT_FACTOR
    DRAG_MARGIN = keep[0] if drag_margin is None else drag_margin
    PRINT_FACTOR = keep[1] if print_factor is None else print_factor
    try:
        return top_speed(c)
    finally:
        DRAG_MARGIN, PRINT_FACTOR = keep


def speed_run(c: Config, v_top: float | None = None, mass: float | None = None, n: int = 12) -> dict:
    """The run from a hover to 98 % of the top speed in a straight line: time, distance, the energy, and the heat the
    windings take over the acceleration and the timed stretch (no cooling counted: seconds, not minutes)."""
    m = mass_budget(c).mass if mass is None else mass
    v_top = top_speed(c, m)["V"] if v_top is None else v_top
    motor = mk.part(c.motor)
    vs = np.linspace(3.0, max(0.98 * v_top, 4.0), n)
    pts = [balance(c, float(v), m) for v in vs]
    acc = np.array([max(p["thrust"] * math.cos(math.radians(p["alpha_deg"])) - 0.5 * RHO * v * v * p["CdA"], 1e-3) / m
                    for p, v in zip(pts, vs)])
    pe, cu = np.array([p["electric_w"] for p in pts]), np.array([p["copper_w"] for p in pts])
    dt = np.diff(vs) / (0.5 * (acc[1:] + acc[:-1]))
    t = float(dt.sum())
    dist = float(np.sum(dt * 0.5 * (vs[1:] + vs[:-1])))
    e_acc = float(np.sum(dt * 0.5 * (pe[1:] + pe[:-1])))
    q_acc = float(np.sum(dt * 0.5 * (cu[1:] + cu[:-1])))
    t_hold = (PASS_LENGTH + 2 * RUN_IN) / max(v_top, 1.0)
    static = drive(c, 0.5)
    hover_w = static["electric_w"] * (m * G / max(static["thrust"], 1e-6)) ** 1.5       # P ~ T^1.5 from the static point
    e_pass = e_acc + pe[-1] * t_hold + hover_w * HOVER_S
    heat_capacity = 0.55 * motor.mass_g / 1e3 * 450.0                                    # stator iron and copper [J/K]
    return {"t_accel": t, "d_accel": dist, "t_hold": t_hold, "wh_pass": e_pass / 3600.0, "hover_w": hover_w,
            "motor_dT": (q_acc + cu[-1] * t_hold) / heat_capacity, "static_thrust": static["thrust"],
            "static_tw": static["thrust"] / (m * G), "field_length": 2 * dist + PASS_LENGTH + 2 * RUN_IN}


def evaluate(c: Config, passes: int = 2) -> dict:
    """Everything the trade needs of one configuration: speed, price, mass, the margins and what fails."""
    motor, prop, esc, bat = (mk.part(k) for k in (c.motor, c.propeller, c.esc, c.battery))
    budget = mass_budget(c)
    m = budget.mass
    top = top_speed(c, m)
    run = speed_run(c, top["V"], m)
    wh = bat.cells * 3.7 * bat.ah * c.packs
    rpm_limit = prop.rpm_limit_k / prop.d_in
    fails = []
    if c.cells > motor.max_cells:
        fails.append("motor voltage")
    if c.cells > esc.max_cells:
        fails.append("ESC voltage")
    if run["static_tw"] < STATIC_TW_MIN:
        fails.append("static thrust")
    if passes * run["wh_pass"] > USABLE * wh:
        fails.append("energy")
    if run["motor_dT"] > MOTOR_DT_MAX:
        fails.append("motor heat")
    if top["kmh"] <= 0:
        fails.append("does not fly")
    return {"config": c, "label": c.label(), "kmh": top["kmh"], "V": top["V"], "price": mk.price(c), "mass_g": m * 1e3,
            "rpm": top["rpm"], "rpm_limit": rpm_limit, "overspeed": top["rpm"] / rpm_limit, "tip_mach": top["tip_mach"],
            "J": top["J"], "prop_eff": top["prop_eff"], "limit": top["limit"], "current": top["current"],
            "pack_current": top["pack_current"], "cell_v": top["cell_v"], "electric_kw": top["electric_w"] / 1e3,
            "thrust": top["thrust"], "drag": top["drag"], "CdA": top["CdA"], "alpha_deg": top["alpha_deg"],
            "static_tw": run["static_tw"], "t_accel": run["t_accel"], "d_accel": run["d_accel"],
            "field_length": run["field_length"], "wh_pass": run["wh_pass"], "wh_pack": wh, "passes": USABLE * wh / run["wh_pass"],
            "motor_dT": run["motor_dT"], "fails": fails, "feasible": not fails, "within_rpm_limit": top["rpm"] <= rpm_limit,
            "assumed_parts": [p.name for p in (motor, prop, esc, bat) if p.kind == "assumed"]}


# ========================================================================================================== the trade
def candidates(motors=None, propellers=None, batteries=None, layouts=((1, 1), (1, 2), (2, 1), (2, 2))) -> list[Config]:
    """Every build the catalogue allows: motor x propeller x pack x (series, parallel), each with the cheapest ESC set
    that takes its voltage and its motor's burst current (a smaller ESC would only move the limit to itself)."""
    out = []
    for m, p, b, (ns, np_) in itertools.product(motors or mk.MOTORS, propellers or mk.PROPELLERS, batteries or mk.BATTERIES, layouts):
        if 6 * ns > m.max_cells:
            continue
        esc = mk.esc_for(6 * ns, m.max_a) or mk.esc_for(6 * ns, 0.0)
        if esc is None:
            continue
        out.append(Config(m.key, p.key, esc.key, b.key, ns, np_))
    return out


def trade(configs=None, passes: int = 2, progress=None) -> list[dict]:
    rows = []
    for i, c in enumerate(configs if configs is not None else candidates()):
        rows.append(evaluate(c, passes))
        if progress:
            progress(i)
    return rows


def pareto(rows, key_cost="price", key_gain="kmh") -> list[dict]:
    """The rows no other row beats on both: cheaper and at least as fast. Sorted by price."""
    front, best = [], -1.0
    for r in sorted(rows, key=lambda r: (r[key_cost], -r[key_gain])):
        if r[key_gain] > best + 1e-9:
            front.append(r)
            best = r[key_gain]
    return front


def marginal(front) -> list[dict]:
    """Along a front: what the next step costs per km/h gained."""
    out = []
    for a, b in zip(front, front[1:]):
        out.append({"from": a["label"], "to": b["label"], "d_price": b["price"] - a["price"], "d_kmh": b["kmh"] - a["kmh"],
                    "usd_per_kmh": (b["price"] - a["price"]) / max(b["kmh"] - a["kmh"], 1e-9)})
    return out


def knee(front) -> dict:
    """The point of a front farthest above the straight line between its ends (both axes scaled to the front's span):
    past it, each further km/h costs more than the front's average."""
    if len(front) < 3:
        return front[-1]
    p0, p1, s0, s1 = front[0]["price"], front[-1]["price"], front[0]["kmh"], front[-1]["kmh"]
    return max(front, key=lambda r: (r["kmh"] - s0) / max(s1 - s0, 1e-9) - (r["price"] - p0) / max(p1 - p0, 1e-9))


def picks(rows, within_rpm_limit: bool = False) -> dict:
    """Three builds out of a trade: ``budget`` (the cheapest feasible one), ``knee`` and ``fastest``; with
    ``within_rpm_limit`` only builds whose propeller stays inside its maker's speed limit at top speed."""
    ok = [r for r in rows if r["feasible"] and (r["within_rpm_limit"] or not within_rpm_limit)]
    if not ok:
        return {}
    front = pareto(ok)
    return {"budget": front[0], "knee": knee(front), "fastest": front[-1], "front": front}


def export(rows) -> list[dict]:
    """The trade as plain rows (JSON-ready): the configuration as its keys."""
    out = []
    for r in rows:
        d = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "config"}
        out.append({**asdict(r["config"]), **d})
    return out


def load_trade(path=None) -> list[dict]:
    """The trade ``labs/aero/sims/emerillon_trade.py`` wrote, as ``trade`` returns it (the configuration rebuilt)."""
    import json
    from pathlib import Path
    path = Path(path) if path else Path(__file__).resolve().parent / "data" / "emerillon_trade.json"
    names = list(Config.__dataclass_fields__)
    return [{**{k: v for k, v in r.items() if k not in names}, "config": Config(**{k: r[k] for k in names})}
            for r in json.loads(path.read_text())["rows"]]


# the published record this design is measured against (Guinness World Records, as the press reports it)
RECORD = {"name": "Peregreen V4 (Luke and Mike Bell)", "kmh": 657.59, "date": "2025-12-11", "place": "Cape Town",
          "what": "fastest ground speed by a battery-powered remote-controlled quadcopter, the mean of two opposite runs",
          "motors": "four T-Motor 3120 at 900 KV", "airframe": "3D printed (PETG, PA6-CF, TPU), propellers under 40,000 rpm",
          "source": "https://newatlas.com/drones/luke-mike-bell-peregreen-v4-guinness-speed/"}
ER2 = Config()                                                                      # the knee: shelf parts, about 400 USD
ER1 = Config("eco2807-1300", "apc6x6e", "speedybee-f405v4-55", "cnhl-1300-100c", 1, 1)   # the cheapest build that flies the run
VARIANTS = {"ER1": ER1, "ER2": ER2}
VARIANT_NAME = {"ER1": "Émerillon ER1 — the budget build", "ER2": "Émerillon ER2 — the knee build"}
TARGET = {"ER1": {"kmh": 261.0, "ceiling_m": 4000.0}, "ER2": {"kmh": 383.0, "ceiling_m": 7000.0}}   # what each has to do
# the packs that carry each variant to its ceiling target and back in the flown climb, with the reserve (the slice
# analysis of emerillon_mission.ceiling_options passes smaller ones that the simulated catch and landing do not)
HIGH = {"ER1": Config("eco2807-1300", "apc6x6e", "speedybee-f405v4-55", "tattu-r6-1700-160c", 1, 1),
        "ER2": Config("eco2807-1300", "apc7x10", "speedybee-f405v4-55", "tattu-r6-1700-160c", 1, 2)}   # lands a little under the reserve
# the long-range fit: two Tattu 1700 mAh in parallel — as far as the heavier 2 Ah pair, lighter and cheaper (emerillon_mission.range_options)
LONG = {"ER1": Config("eco2807-1300", "apc6x6e", "speedybee-f405v4-55", "tattu-r6-1700-160c", 1, 2),
        "ER2": Config("eco2807-1300", "apc7x10", "speedybee-f405v4-55", "tattu-r6-1700-160c", 1, 2)}
DESIGN = ER2
RECORD_BUILD = Config("v3120-900", "own7x13", "apd-120f3", "tattu-r6-1700-160c", 2, 2)   # the fast end of the front
