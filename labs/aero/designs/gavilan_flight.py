"""GAVILAN — the reduced flight models, and the fleet it is measured against (notebook 38).

One model for every aircraft, honest about what it is: ``Machine`` is a point mass with a **body axis**. The thrust
acts along the axis; the air gives a lift and a drag that depend on the angle ``theta`` between the axis and the
flight path — a wing's attached polar blended into a flat plate beyond the stall (``ogma.aero.Polar``), a body's
crossflow, or ÉMERILLON's own normal force (``emerillon_flight.normal_force_area``). There is no attitude dynamics
beyond a slew rate, no propeller at an angle of attack, no slipstream on the wing (a tailsitter's blown wing stalls
later than this says), no vortex-ring state. What comes out is **load factors, rates, times and distances**, not
handling.

- ``EMITTER`` — the 8 mm recoilless emitter cut down for an aircraft: sized with the emitter's own model (branch
  ``dev_no_recoil``), its mass a hand estimate with a margin.
- ``mass_budget``, ``wing_aero``, ``drag_buildup`` — GAVILAN's mass (``og.mass.Budget`` on ÉMERILLON's catalogue),
  the wing on the vortex lattice (``ogma.aero.vlm``), a friction and form-factor drag build-up with ÉMERILLON's
  margin.
- ``gavilan(variant)`` — GAVILAN as a ``Machine``: ``A`` the tailsitter (four fixed motors, a broad wing), ``B`` the
  side-panel arrangement, ``F`` the fixed wing with two tractors and the emitter in the nose.
- ``fleet()`` — the aircraft the lab already has, as ``Machine``s from their own modules: MERLIN, PEREGRINE, NISUS,
  NISUS+, FALCO (fixed wings), the X-wing (``kestrel_xwing``) and ÉMERILLON.
- ``turn``, ``brake``, ``accelerate``, ``reversal``, ``climb``, ``level``, ``top_speed``, ``gun_pass`` — the
  manoeuvres; ``scorecard`` — one row per aircraft.

Units SI. ``theta`` is the angle from the flight path to the body axis (0: nose on the flight path; 90 deg: the
axis across it, as a tailsitter hangs on its propellers).
"""
from __future__ import annotations

import dataclasses
import functools
import math
from dataclasses import dataclass, field

import numpy as np
from ogma import mass as og_mass
from ogma.aero import Polar, cf_turbulent, viterna_cd90, vlm
from ogma.units import G, NU_AIR as NU, RHO0 as RHO

import emerillon_flight as ef
import emerillon_market as mk
import gavilan as gv

# ---------------------------------------------------------------------------------------------------------- assumptions
CL_MAX_2D = 1.0                    # NACA 0010 at Re 2-4e5 (hand estimate; symmetric sections stall near 1.0 there)
CL_MAX_3D = 0.9 * CL_MAX_2D        # a finite wing reaches about 0.9 of its section (kestrel_xwing's factor)
OSWALD_VISCOUS = 0.90              # e = 0.90 x the lattice's span efficiency (peregrine_flight's factor)
AREAL_DENSITY = 0.70               # kg/m2 of wetted skin for printed foam surfaces: wing, fins, panels (MERLIN's, PEREGRINE's)
SLEW = 6.0                         # how fast the body axis turns [rad/s] (emerillon_mission.SLEW: assumed)
REVERSE_THRUST = 0.5               # share of the thrust kept when the propellers meet the air backwards (emerillon_mission: assumed)
N_DESIGN = 8.0                     # the load factor GAVILAN's wing and pylons are built for (proposed; not yet checked by FEA)
THETA_MAX = math.radians(150.0)    # the furthest the axis is tipped back in a stop
USABLE = ef.USABLE                 # share of the pack a flight may take
TILT_SERVO_G = 9.0                 # the camera's tilt servo (a sub-micro servo, PEREGRINE's parts list)
CRADLE_G = 45.0                    # the emitter's tilting cradle on a tailsitter: servo, cradle, bearings (hand estimate)

CONFIG_A = ef.Config("eco2807-1300", "hq7x4x3", "speedybee-f405v4-55", "tattu-r6-1700-160c", 1, 2)   # four motors, 6S2P
CONFIG_F = ef.Config("eco2807-1300", "apc7x7e", "speedybee-f405v4-55", "tattu-r6-1700-160c", 1, 1)   # two motors, 6S1P


# ============================================================================================================ the emitter
# The 8 mm recoilless emitter (labs/aero/exp/bearing_recoilless.py on dev_no_recoil @ 27df5be) is a ground piece: a 1.0 m
# tube, a 0.5 L steel plenum, a 0.8 L bottle, 280 m/s. For 30 m an aircraft needs far less. Its own model
# (golf_recoilless.shot with the 8 mm ball, sized for zero recoil) gives, for a 0.30 m tube and a 0.05 L plenum:
# 33.7 bar gauge, a 6.8 mm throat, 130 m/s, 18.0 J at the muzzle, 0.91 g of air per shot, net impulse -0.0001 N s.
# bearing_target's 30 m test at 132 m/s: 14 hits of 20 on its target from a rigid mount, 0.25 s of flight.
# The masses below are HAND ESTIMATES (no part chosen yet); MARGIN is on their sum.
EMITTER = {
    "source": "sized with golf_recoilless.shot (dev_no_recoil @ 27df5be); masses are hand estimates",
    "tube_m": 0.30, "length_m": 0.38, "bore_mm": 8.0, "muzzle_m_s": 130.0, "muzzle_J": 18.0, "pressure_bar_g": 33.7,
    "air_g_per_shot": 0.91, "ball_g": 2.08, "balls": 15, "flight_s_30m": 0.25, "range_m": 30.0,
    "parts_g": {"tube 0.30 m, 8 mm bore, 10 mm stainless": 67.0, "chamber and rear venturi (aluminium)": 35.0,
                "valve with its pilot": 60.0, "plenum 0.05 L, 100 bar": 30.0, "air bottle 0.1 L, 300 bar": 150.0,
                "micro regulator": 80.0, "magazine and 15 balls": 41.0, "clamps": 20.0},
    "margin": 1.25,
}


def emitter_kg(n: int = 1) -> float:
    """``n`` emitters aboard [kg]: the hand-estimated parts times the margin."""
    return n * sum(EMITTER["parts_g"].values()) * EMITTER["margin"] / 1e3


# ============================================================================================================ the machine
@dataclass
class Machine:
    """One aircraft as the manoeuvre models read it."""
    name: str
    kind: str                       # 'tailsitter' (hovers on its tail), 'wing' (cannot hover)
    mass: float                     # [kg]
    span: float                     # the widest dimension in flight [m]
    length: float
    S: float                        # reference area of ``polar`` [m2]
    polar: object                   # ogma.aero.Polar on S (None: ``nfa`` gives the normal force)
    cda0: float                     # drag area nose-on, margin included [m2]
    t_max: object                   # full thrust at an axial airspeed [N]
    power: object                   # electrical power for a thrust at an axial airspeed [W]
    n_struct: float
    battery_wh: float
    plan_body: float = 0.0          # the body's side area for the crossflow [m2]
    nfa: object = None              # normal-force area against theta [m2] (ÉMERILLON)
    slew: float = SLEW
    roll_rate: float = math.radians(360.0)     # [rad/s]
    fixed_w: float = 0.0            # avionics, drawn whatever the thrust
    payload_kg: float = 0.0         # emitters aboard, in ``mass``
    gun: str = "none"               # 'fixed' along the body axis, 'cradle' (tilts 0-90 deg with the camera), 'none'
    theta_limit: float | None = None   # a wing that is controlled beyond its stall (a blown tail): how far its axis may go [rad]
    stops: bool | None = None       # None: what hovers stops; True: a wing that can flare to a standstill on its thrust
    source: str = ""
    notes: dict = field(default_factory=dict)

    @property
    def weight(self) -> float:
        return self.mass * G

    @property
    def theta_max(self) -> float:
        """How far the axis may leave the flight path: a tailsitter anywhere (its motors hold the attitude), a
        conventional wing up to its stall (nothing aboard controls it beyond)."""
        if self.theta_limit is not None:
            return self.theta_limit
        return THETA_MAX if self.kind == "tailsitter" else self.polar.alpha_stall

    def thetas(self, upto: float | None = None) -> np.ndarray:
        return THETAS[THETAS <= min(self.theta_max, THETA_MAX if upto is None else upto) + 1e-9]

    @property
    def hovers(self) -> bool:
        return (self.kind == "tailsitter" or bool(self.stops)) and self.t_max(0.0) > 1.15 * self.weight

    def areas(self, theta: float) -> tuple:
        """(lift area, drag area) [m2] at the axis angle ``theta`` [rad]: the force over the dynamic pressure, the
        lift normal to the flight path on the axis's side."""
        s, c = math.sin(theta), math.cos(theta)
        if self.nfa is not None:
            n = self.nfa(min(theta, math.pi - theta)) * (1.0 if theta <= math.pi / 2 else -1.0)
            return n * c * (1 if theta <= math.pi / 2 else -1), abs(n) * s + self.cda0 * c * c * (1.0 if c >= 0 else 1.5)
        cl, cd, _ = self.polar.coefficients(theta)
        body_n = 1.2 * self.plan_body * s * s
        return self.S * float(cl) + body_n * c, self.S * float(cd) + body_n * s + self.cda0 * c * c * (1.0 if c >= 0 else 1.5)

    def thrust(self, V: float, theta: float) -> float:
        """Full thrust with the axis at ``theta`` to the flight path: the propellers see ``V cos(theta)``; flying
        backwards onto them they keep ``REVERSE_THRUST`` of their static thrust."""
        v_ax = V * math.cos(theta)
        return self.t_max(v_ax) if v_ax >= -2.0 else REVERSE_THRUST * self.t_max(0.0)


# ------------------------------------------------------------------------------------------------- GAVILAN: mass, aero
def geometry(p=None) -> dict:
    """The numbers the models read from the CAD's parameters [m, m2]."""
    p = gv.resolve(p)
    L = gv.Gavilan.layout(p)
    d = p["body_diameter"] / 1e3
    nose, bay, cone = p["nose_length"] / 1e3, p["bay_length"] / 1e3, p["cone_length"] / 1e3
    S = L["S_mm2"] / 1e6
    S_exposed = S - d * L["c_root"] / 1e3
    panel = p["pylon_shape"] == "panel"
    S_pylons = L["S_panels_mm2"] / 1e6 / (1 if panel else 2)       # plan area of the two pylons
    fin = 0.5 * (p["fin_root_chord"] + p["fin_tip_chord"]) * (p["fin_span"] - p["boom_diameter"] / 2) / 1e6
    boom_l = (p["length"] - L["x_boom"]) / 1e3
    return {"d": d, "nose": nose, "bay": bay, "cone": cone, "length": p["length"] / 1e3, "span": p["span"] / 1e3,
            "width": L["width"] / 1e3, "S": S, "AR": L["AR"], "mac": L["mac"] / 1e3, "S_exposed": S_exposed, "t_c": p["thickness"],
            "S_pylons": S_pylons, "pylon_c": p["pylon_chord"] / 1e3, "pylon_t": p["pylon_thickness"] / 1e3, "panel": panel,
            "S_fins": 4 * fin, "fin_c": 0.5 * (p["fin_root_chord"] + p["fin_tip_chord"]) / 1e3, "fin_t": p["fin_thickness"] / 1e3,
            "boom_d": p["boom_diameter"] / 1e3, "boom_l": boom_l, "pod_d": p["pod_diameter"] / 1e3, "pod_l": p["pod_length"] / 1e3,
            "s_body": math.pi * d * (0.70 * nose + bay + 0.60 * cone), "frontal": math.pi * d * d / 4,
            "plan_body": d * (0.67 * nose + bay + 0.55 * cone) + p["boom_diameter"] / 1e3 * boom_l,
            "motor_y": p["motor_y"] / 1e3, "motor_z": p["motor_z"] / 1e3, "x_ac": L["x_ac"] / 1e3, "x_prop": L["x_prop"] / 1e3,
            "x_bay": L["x_bay"] / 1e3, "x_boom": L["x_boom"] / 1e3, "x_fin": L["x_fin"] / 1e3, "foot_half_width": L["foot_half_width"] / 1e3}


def mass_budget(p=None, c: ef.Config = CONFIG_A, emitters: int = 1) -> og_mass.Budget:
    """GAVILAN-A or -B: the bought parts from ÉMERILLON's catalogue, the printed parts (the body's skin as
    ÉMERILLON's, the foam surfaces at ``AREAL_DENSITY``), the camera's servo, ``emitters`` emitters under the bay."""
    g = geometry(p)
    motor, prop, esc, bat = (mk.part(k) for k in (c.motor, c.propeller, c.esc, c.battery))
    petg, pa6 = mk.part("petg-hf"), mk.part("pa6-cf")
    b, I = og_mass.Budget(), og_mass.Item
    for i, (sy, sz) in enumerate(gv.MOTORS):
        y, z = sy * g["motor_y"], sz * g["motor_z"]
        b.add(I(f"motor {i + 1}", motor.mass_g / 1e3, (g["x_prop"] + 0.02, y, z), group="propulsion", source=c.motor),
              I(f"propeller {i + 1}", prop.mass_g / 1e3, (g["x_prop"], y, z), group="propulsion", source=c.propeller))
    b.add(I("ESC board with the flight controller", esc.mass_g / 1e3, (g["x_bay"] + 0.02, 0, 0), group="propulsion", source=c.esc))
    Lp = bat.size_mm[0] / 1e3 + 0.006
    for k in range(c.packs):
        b.add(I(f"pack {k + 1}", bat.mass_g / 1e3, (g["x_bay"] + ef.STACK_BAY + (k + 0.5) * Lp, 0, 0), group="battery", source=c.battery))
    for part in mk.FIXED:
        if part.group == "fc" and esc.with_fc:
            continue
        b.add(I(part.name, part.mass_g / 1e3, (0.6 * g["nose"] if part.group in ("video", "gps") else g["x_bay"] + 0.02, 0, 0),
                group="avionics", source=part.key))
    b.add(I("camera tilt servo", TILT_SERVO_G / 1e3, (0.03, 0, 0), group="avionics", source="hand estimate: a 9 g sub-micro servo"),
          *([I("emitter cradle and its servo", CRADLE_G / 1e3, (g["x_ac"] - 0.02, 0, -g["d"] / 2 - 0.02), group="payload",
               source="hand estimate: a 20 g metal-gear servo, a printed cradle, two bearings")] if emitters else []),
          I("power wiring", 0.012 * c.packs + 0.030, (g["x_bay"] + 0.03, 0, 0), group="propulsion", source="emerillon_flight's allowance, longer leads"))
    skin = g["s_body"] * ef.WALL * ef.PRINT_FACTOR * petg.density
    pods = 4 * math.pi * g["pod_d"] * 0.75 * g["pod_l"] * 0.0016 * pa6.density * ef.PRINT_FACTOR / 1.3
    pylons = (2.04 * g["S_pylons"] * AREAL_DENSITY if g["panel"]
              else 2 * 0.68 * g["pylon_c"] * g["pylon_t"] * 2 * g["motor_z"] * 0.55 * pa6.density * ef.PRINT_FACTOR / 1.3)
    b.add(I("body skin (PETG)", skin, (0.45 * (g["x_boom"]), 0, 0), group="airframe", source="emerillon_flight.printed_masses' rule"),
          I("wing (printed foam)", 2.04 * g["S_exposed"] * AREAL_DENSITY, (g["x_ac"] + 0.15 * g["mac"], 0, 0), group="airframe", source="AREAL_DENSITY"),
          I("pylons" + (" (panels, printed foam)" if g["panel"] else " (PA6-CF)"), pylons, (g["x_ac"], 0, 0), group="airframe", source="rule"),
          I("motor pods (PA6-CF)", pods, (g["x_prop"] + 0.03, 0, 0), group="airframe", source="emerillon_flight.printed_masses' rule"),
          I("fins (printed foam)", 2.04 * g["S_fins"] * AREAL_DENSITY, (g["x_fin"] + 0.09, 0, 0), group="airframe", source="AREAL_DENSITY"),
          I("tail boom (bought carbon tube)", 0.035 * g["boom_l"] + 0.004, (g["x_boom"] + 0.5 * g["boom_l"], 0, 0), group="airframe",
            source="hand estimate: 35 g/m"),
          I("bumpers (TPU)", 0.008, (g["length"], 0, 0), group="airframe", source="emerillon_flight"))
    for k in range(emitters):
        b.add(I(f"emitter {k + 1} with air and 15 balls", emitter_kg(1), (g["x_ac"] - 0.02, 0, -g["d"] / 2 - 0.02), group="payload",
                source="EMITTER (hand estimate x 1.25)"))
    return b


@functools.lru_cache(maxsize=None)
def _wing_aero(key: tuple) -> dict:
    p = gv.resolve(dict(key))
    L = gv.Gavilan.layout(p)
    S, c = L["S_mm2"] / 1e6, L["mac"] / 1e3
    quads = gv.planform(p)["wing"] + gv.planform(p)["pylons"]
    r = vlm(quads, alpha_deg=4.0, S_ref=S, c_ref=c, n_span=10, n_chord=4)
    a = math.radians(4.0)
    cla, k = r["CL"] / a, r["CDi"] / r["CL"] ** 2
    return {"S": S, "AR": L["AR"], "CLa": cla, "k": k, "e": 1.0 / (math.pi * L["AR"] * k)}


def wing_aero(p=None) -> dict:
    """The wing with its pylons on the vortex lattice: lift slope, induced factor ``k``, span efficiency ``e``."""
    return dict(_wing_aero(tuple(sorted(dict(p or {}).items()))))


def drag_buildup(p=None, V: float = 25.0, emitters: int = 1, margin: float | None = None) -> dict:
    """Drag area nose-on [m2] at ``V``: flat-plate friction x form factor per part (Raymer / Hoerner), the junctions'
    interference, the emitter under the belly, then ÉMERILLON's margin (the hand build-up runs low against RANS)."""
    g = geometry(p)
    margin = ef.DRAG_MARGIN if margin is None else margin

    def cf(length):
        return cf_turbulent(max(V * length / NU, 1e4))

    f = (g["nose"] + g["bay"] + g["cone"]) / g["d"]
    body = cf(g["nose"] + g["bay"] + g["cone"]) * (1 + 60 / f ** 3 + f / 400) * g["s_body"]
    wing = cf(g["mac"]) * (1 + 2 * g["t_c"] + 100 * g["t_c"] ** 4) * 2.04 * g["S_exposed"]
    tp = g["pylon_t"] / g["pylon_c"]
    pylons = cf(g["pylon_c"]) * (1 + 2 * tp + 60 * tp ** 4) * 2.04 * g["S_pylons"]
    fp = g["pod_l"] / g["pod_d"]
    pods = 4 * cf(g["pod_l"]) * (1 + 60 / fp ** 3 + fp / 400) * math.pi * g["pod_d"] * 0.75 * g["pod_l"]
    fins = cf(g["fin_c"]) * (1 + 2 * g["fin_t"] / g["fin_c"]) * 2.04 * g["S_fins"]
    boom = cf(g["boom_l"]) * math.pi * g["boom_d"] * g["boom_l"]
    camera = 0.25 * math.pi * 0.016 ** 2                                       # a ball on the nose, Cd 0.25 on its face (hand)
    emitter = emitters * (cf(EMITTER["length_m"]) * math.pi * 0.012 * EMITTER["length_m"] * 1.3 + 0.35 * math.pi * 0.022 ** 2)   # tube; bottle's face (hand)
    clean = body + (wing + pylons + pods) * ef.INTERFERENCE + fins + boom + camera + emitter
    return {"body": body, "wing": wing * ef.INTERFERENCE, "pylons": pylons * ef.INTERFERENCE, "pods": pods * ef.INTERFERENCE,
            "fins": fins, "boom": boom, "camera": camera, "emitter": emitter, "clean": clean, "CdA": clean * margin, "margin": margin}


# ------------------------------------------------------------------------------------------------------------- drives
U_GRID = (0.12, 0.25, 0.40, 0.55, 0.70, 0.85, 1.0)


@functools.lru_cache(maxsize=None)
def drive_map(c: ef.Config, n_motors: int = 4, v_max: float = 80.0, n_v: int = 17) -> dict:
    """``emerillon_flight.drive`` (four motors on the pack: BEMT propeller, motor, sag, current limits) tabulated over
    the axial airspeed and the throttle; ``n_motors`` of the four counted (two motors draw half the pack current, so
    the sag is overstated a little for a twin: on the safe side)."""
    V = np.linspace(0.0, v_max, n_v)
    T, P = np.zeros((n_v, len(U_GRID))), np.zeros((n_v, len(U_GRID)))
    for i, v in enumerate(V):
        for j, u in enumerate(U_GRID):
            d = ef.drive(c, max(float(v), 0.5), throttle=u)
            T[i, j], P[i, j] = max(d["thrust"], 0.0) * n_motors / 4, d["electric_w"] * n_motors / 4
    T = np.maximum.accumulate(T, axis=1) + np.arange(len(U_GRID)) * 1e-9
    return {"V": V, "T": T, "P": P, "n": n_motors}


def _drive_fns(dm: dict, fixed_w: float = 0.0):
    V = dm["V"]

    def row(v):
        v = float(np.clip(v, V[0], V[-1]))
        i = min(int(np.searchsorted(V, v, side="right")) - 1, len(V) - 2)
        t = (v - V[i]) / (V[i + 1] - V[i])
        return (1 - t) * dm["T"][i] + t * dm["T"][i + 1], (1 - t) * dm["P"][i] + t * dm["P"][i + 1]

    def t_max(v):
        return float(row(max(v, 0.0))[0][-1])

    def power(T, v):
        Tr, Pr = row(max(v, 0.0))
        if T <= 0:
            return fixed_w
        if T <= Tr[0]:
            return float(Pr[0] * T / Tr[0]) + fixed_w
        return float(np.interp(min(T, Tr[-1]), Tr, Pr)) + fixed_w

    return t_max, power


# --------------------------------------------------------------------------------------------------------- the GAVILANs
AVIONICS_W = 12.0                  # flight controller, receiver, GPS, camera, video link (hand estimate)

# GAVILAN-F: the fixed wing. notebook 09's twin (fixed_wing.FixedWing) with a small wing; the two tractors leave the
# nose free for the emitter's muzzle and the tail free for its rear jet.
F_PARAMS = {"span": 800.0, "root_chord": 230.0, "taper": 0.65, "dihedral_deg": 2.0, "camber": 0.02, "camber_pos": 0.4, "thickness": 0.10,
            "fuselage_length": 720.0, "fuselage_diameter": 64.0, "nose_length": 210.0, "nacelle_y": 190.0, "nacelle_diameter": 37.5,
            "nacelle_length": 110.0, "nacelle_forward": 60.0, "tail_span": 300.0, "tail_chord": 100.0, "fin_height": 120.0, "tail_thickness": 4.0}
F_BROAD = {"span": 1000.0, "root_chord": 280.0}       # the same aircraft on a broad wing: slower passes, tighter turns, less speed
F_CL_MAX = 1.08                    # PEREGRINE's CL_max for the same class of cambered wing (peregrine_design.json)
F_N_SERVOS = 4                     # two ailerons, elevator, rudder


def f_geometry(q=None) -> dict:
    q = {**F_PARAMS, **(q or {})}
    b, c0, c1 = q["span"] / 1e3, q["root_chord"] / 1e3, q["root_chord"] * q["taper"] / 1e3
    S = 0.5 * (c0 + c1) * b
    d, Lf = q["fuselage_diameter"] / 1e3, q["fuselage_length"] / 1e3
    S_tail = (q["tail_span"] * q["tail_chord"] + q["fin_height"] * q["tail_chord"]) / 1e6
    return {"span": b, "S": S, "AR": b * b / S, "mac": 2 / 3 * (c0 * c0 + c0 * c1 + c1 * c1) / (c0 + c1), "d": d, "length": Lf,
            "S_exposed": S - d * c0, "S_tail": S_tail, "s_body": math.pi * d * 0.80 * Lf, "t_c": q["thickness"],
            "nac_d": q["nacelle_diameter"] / 1e3, "nac_l": q["nacelle_length"] / 1e3, "nose": q["nose_length"] / 1e3,
            "plan_body": 0.8 * d * Lf}


def f_mass_budget(q=None, c: ef.Config = CONFIG_F, emitters: int = 1) -> og_mass.Budget:
    g = f_geometry(q)
    motor, prop, esc, bat = (mk.part(k) for k in (c.motor, c.propeller, c.esc, c.battery))
    petg = mk.part("petg-hf")
    b, I = og_mass.Budget(), og_mass.Item
    x_w = g["nose"]                                                    # wing leading edge from the nose
    for s in (1, -1):
        b.add(I("motor", motor.mass_g / 1e3, (x_w - 0.05, s * 0.19, 0), group="propulsion", source=c.motor),
              I("propeller", prop.mass_g / 1e3, (x_w - 0.07, s * 0.19, 0), group="propulsion", source=c.propeller))
    b.add(I("ESC board with the flight controller", esc.mass_g / 1e3, (x_w, 0, 0), group="propulsion", source=c.esc))
    for k in range(c.packs):
        b.add(I(f"pack {k + 1}", bat.mass_g / 1e3, (x_w + 0.03, 0, 0), group="battery", source=c.battery))
    for part in mk.FIXED:
        if part.group == "fc" and esc.with_fc:
            continue
        b.add(I(part.name, part.mass_g / 1e3, (0.08, 0, 0), group="avionics", source=part.key))
    b.add(I("servos", F_N_SERVOS * TILT_SERVO_G / 1e3, (x_w + 0.2, 0, 0), group="avionics", source="hand estimate: four 9 g servos"),
          I("power wiring", 0.012 * c.packs + 0.025, (x_w, 0, 0), group="propulsion", source="emerillon_flight's allowance"),
          I("fuselage skin (PETG)", g["s_body"] * ef.WALL * ef.PRINT_FACTOR * petg.density, (0.45 * g["length"], 0, 0), group="airframe",
            source="emerillon_flight.printed_masses' rule"),
          I("wing (printed foam)", 2.04 * g["S_exposed"] * AREAL_DENSITY, (x_w + 0.08, 0, 0), group="airframe", source="AREAL_DENSITY"),
          I("tail (printed foam)", 2.04 * g["S_tail"] * AREAL_DENSITY, (g["length"] - 0.05, 0, 0), group="airframe", source="AREAL_DENSITY"),
          I("nacelles (PA6-CF)", 2 * math.pi * g["nac_d"] * g["nac_l"] * 0.0016 * mk.part("pa6-cf").density * ef.PRINT_FACTOR / 1.3,
            (x_w - 0.03, 0, 0), group="airframe", source="rule"))
    for k in range(emitters):
        b.add(I(f"emitter {k + 1} with air and 15 balls", emitter_kg(1), (x_w - 0.02, 0, 0), group="payload", source="EMITTER (hand estimate x 1.25)"))
    return b


def f_drag_buildup(q=None, V: float = 25.0, margin: float | None = None) -> dict:
    g = f_geometry(q)
    margin = ef.DRAG_MARGIN if margin is None else margin

    def cf(length):
        return cf_turbulent(max(V * length / NU, 1e4))

    f = g["length"] / g["d"]
    body = cf(g["length"]) * (1 + 60 / f ** 3 + f / 400) * g["s_body"]
    wing = cf(g["mac"]) * (1 + 2 * g["t_c"] + 100 * g["t_c"] ** 4) * 2.04 * g["S_exposed"]
    tail = cf(0.10) * 1.1 * 2.04 * g["S_tail"]
    fn = g["nac_l"] / g["nac_d"]
    nac = 2 * cf(g["nac_l"]) * (1 + 60 / fn ** 3 + fn / 400) * math.pi * g["nac_d"] * g["nac_l"]
    clean = body + (wing + nac) * ef.INTERFERENCE + tail
    return {"body": body, "wing": wing * ef.INTERFERENCE, "nacelles": nac * ef.INTERFERENCE, "tail": tail, "clean": clean,
            "CdA": clean * margin, "margin": margin}


def gavilan(variant: str = "A", *, emitters: int = 1, config: ef.Config | None = None, p: dict | None = None) -> Machine:
    """GAVILAN as a ``Machine``: ``A`` and ``B`` the tailsitters (``gavilan.VARIANTS``), ``F`` the fixed wing."""
    if variant == "F":
        c = config or CONFIG_F
        g = f_geometry(p)
        m = f_mass_budget(p, c, emitters).mass
        ar = g["AR"]
        e = 0.80                                                           # MERLIN's Oswald factor for a plain tapered wing
        pol = Polar(cl_alpha=2 * math.pi * ar / (ar + 2), cd0=0.0, k=1 / (math.pi * ar * e), cl_max=F_CL_MAX, cd90=viterna_cd90(ar),
                    source="Helmbold slope, e 0.80")
        t_max, power = _drive_fns(drive_map(c, 2), AVIONICS_W)
        return Machine("Gavilan-F (fixed wing, twin tractor)" + (" broad wing" if p else ""), "wing", m, g["span"], g["length"], g["S"], pol, f_drag_buildup(p)["CdA"],
                       t_max, power, N_DESIGN, mk.part(c.battery).cells * 3.7 * mk.part(c.battery).ah * c.packs, plan_body=g["plan_body"],
                       roll_rate=math.radians(250.0), fixed_w=AVIONICS_W, payload_kg=emitter_kg(emitters), gun="fixed" if emitters else "none",
                       source="gavilan_flight (fixed_wing.FixedWing's layout, ÉMERILLON's drive and catalogue)",
                       notes={"config": c.label(), "motors": 2, "roll rate": "hand estimate: p b / 2V = 0.09 at 20 m/s (PEREGRINE's aileron figure)"})
    c = config or CONFIG_A
    q = gv.variant(variant, **(p or {}))
    g, a = geometry(q), wing_aero(q)
    m = mass_budget(q, c, emitters).mass
    pol = Polar(cl_alpha=a["CLa"], cd0=0.0, k=a["k"] / OSWALD_VISCOUS, cl_max=CL_MAX_3D, cd90=viterna_cd90(a["AR"]), source="ogma.aero.vlm")
    t_max, power = _drive_fns(drive_map(c, 4), AVIONICS_W)
    return Machine(gv.VARIANT_NAME[variant].split(" — ")[0] + (" (tailsitter)" if variant == "A" else " (side panels)"), "tailsitter", m,
                   g["width"], g["length"], g["S"], pol, drag_buildup(q, emitters=emitters)["CdA"], t_max, power, N_DESIGN,
                   mk.part(c.battery).cells * 3.7 * mk.part(c.battery).ah * c.packs, plan_body=g["plan_body"],
                   roll_rate=math.radians(600.0), fixed_w=AVIONICS_W, payload_kg=emitter_kg(emitters) + (CRADLE_G / 1e3 if emitters else 0.0),
                   gun="cradle" if emitters else "none", source="gavilan_flight",
                   notes={"config": c.label(), "motors": 4, "roll rate": "hand estimate: a rate limit of the flight controller, not an authority limit"})


# ============================================================================================================== the fleet
def _from_airframe(name, af, full, electrical, n_struct, span, length, battery_wh, source, roll_rate, fixed_w=0.0, notes=None) -> Machine:
    """A fixed wing of the lab as a ``Machine``: its own ``ogma.aero.Airframe`` polar (the whole drag in the polar's
    cd0, Helmbold's lift slope) and its own propulsion unit."""
    ar = af.aspect_ratio
    pol = Polar(cl_alpha=2 * math.pi * ar / (ar + 2), cd0=af.cd0, k=af.k, cl_max=af.cl_max, cd90=viterna_cd90(ar), source="the aircraft's Airframe")

    def power(T, v):                                                   # never better than a 75 % propeller on an 85 % drive
        P = float(electrical(max(v, 0.0), T))
        return max(P if math.isfinite(P) else 0.0, max(T, 0.0) * max(v, 0.0) / (0.75 * 0.85)) + (0.0 if fixed_w == 0.0 else 0.0)

    return Machine(name, "wing", af.mass_kg, span, length, af.wing_area_m2, pol, 0.0, lambda v: float(full(max(v, 0.0))[0]),
                   power, n_struct, battery_wh, roll_rate=roll_rate, fixed_w=fixed_w, source=source,
                   notes=notes or {})


def merlin() -> Machine:
    import merlin_flight as mf
    d = mf.load_design()
    m = mf.build_merlins(design=d)["tractor"]
    p = d["merlin_params"]
    return _from_airframe("Merlin (tractor)", m["airframe"], m["unit"].full, m["unit"].electrical_power, 5.0, p["span"] / 1e3,
                          (p["nose_length"] + p["aft_length"]) / 1e3, d["mission"]["battery_wh"], "merlin_flight.build_merlins, merlin_design.json",
                          math.radians(120.0), notes={"n_struct": "assumed 5 (PEREGRINE's; MERLIN states a gust case only)",
                                                      "roll rate": "hand estimate, p b / 2V = 0.09"})


def peregrine(key: str = "B") -> Machine:
    import json
    import peregrine_flight as pf
    d = json.loads((pf.Path(pf.__file__).resolve().parent / "data" / "peregrine_design.json").read_text())
    ac = pf.make_aircraft(key, {"variant": key}, d["mass_g"][key] / 1e3, n_struct=d["mission"]["n_struct"])
    u = pf.drive()
    return _from_airframe(f"Peregrine {key}", ac.af(0.0), u.full, u.electrical_power, ac.n_struct, ac.span(0.0), 0.67,
                          d["mission"]["battery_wh"], "peregrine_flight.make_aircraft, peregrine_design.json", ac.roll_rate(15.0))


def nisus() -> Machine:
    import nisus
    import nisus_flight as nf
    import nisus_systems as ns
    af = nf.airframe("A")
    unit = nf.MapUnit(ns.propulsion_map()) if hasattr(ns, "propulsion_map") else None
    p = nisus.resolve(None)
    return _from_airframe("Nisus-A", af, unit.full, unit.electrical_power, nf.N_STRUCTURAL, p["span"] / 1e3, 0.9, 24.42,
                          "nisus_flight.airframe, MapUnit", math.radians(120.0))


def xwing() -> Machine:
    """The lab's X-wing tailsitter (``kestrel_xwing``, X-A): four panels at 45 degrees, a motor on each."""
    import kestrel_systems as ks
    import kestrel_xwing as kx
    veh = kx.make_vehicle()
    L, d = veh.L, veh.drive
    pol = Polar(cl_alpha=veh.lift_slope_vertical, cd0=veh.pol.cd0, k=veh.pol.k, cl_max=veh.pol.cl_max * math.cos(L["phi"]),
                cd90=veh.pol.cd90 * math.cos(L["phi"]), source="kestrel_xwing.make_vehicle (vertical lift on S_ref)")
    bat = ks.COMPONENTS["battery"]
    return Machine("X-wing X-A (kestrel_xwing)", "tailsitter", veh.mass, L["span_projected"], 0.55, L["S_ref"], pol, 0.0,
                   lambda v: 4 * float(d.t_max(max(v, 0.0))), lambda T, v: 4 * float(d.power(max(T, 0.0) / 4, max(v, 0.0))) + 13.0,
                   4.0, bat["cells"] * 3.7 * bat["capacity_ah"], plan_body=0.02, roll_rate=math.radians(350.0), fixed_w=13.0,
                   source="kestrel_xwing.make_vehicle, kestrel_systems.drive_table, xwing_design.json",
                   notes={"n_struct": "its spars' design case (xwing_design.json)", "roll rate": "xwing_design.json: 352 deg/s at 16 m/s"})


def emerillon(variant: str = "ER2", emitters: int = 0) -> Machine:
    import emerillon_mission as emn
    c = ef.VARIANTS[variant]
    g = ef.geometry(c)
    m = ef.mass_budget(c).mass + emitter_kg(emitters)
    t_max, power = _drive_fns(drive_map(c, 4, v_max=120.0, n_v=21))
    cda = ef.drag_buildup(c, 30.0)["CdA"]
    return Machine(f"Émerillon {variant}" + (f" + {emitters} emitter" if emitters else ""), "tailsitter", m, 2 * (g["r_arm"] * math.sin(math.radians(45)) + g["prop_d"] / 2),
                   g["length"], g["frontal"], None, cda, t_max, power, emn.ACC_MAX / G, mk.part(c.battery).cells * 3.7 * mk.part(c.battery).ah * c.packs,
                   nfa=lambda th: ef.normal_force_area(c, th), slew=emn.SLEW, roll_rate=math.radians(600.0), payload_kg=emitter_kg(emitters),
                   gun="fixed" if emitters else "none",
                   source="emerillon_flight (drive, drag_buildup, normal_force_area), emerillon_mission (SLEW, ACC_MAX)", notes={"config": c.label()})


def fleet(emitters: int = 1) -> dict:
    """Every aircraft of the comparison by a short key. The lab's own aircraft fly as designed (no emitter: none was
    sized to carry one); GAVILAN carries ``emitters``; ÉMERILLON is listed both ways."""
    out = {"merlin": merlin(), "peregrine": peregrine("B")}
    try:
        out["nisus"] = nisus()
    except Exception:                                                  # its propulsion map is optional data
        pass
    out.update({"xwing": xwing(), "emerillon": emerillon("ER2"), "emerillon_armed": emerillon("ER2", emitters),
                "gavilan_a_unarmed": gavilan("A", emitters=0), "gavilan_a": gavilan("A", emitters=emitters),
                "gavilan_b": gavilan("B", emitters=emitters), "gavilan_f": gavilan("F", emitters=emitters),
                "gavilan_f_broad": gavilan("F", emitters=emitters, p=F_BROAD)})
    return out


# ============================================================================================================= manoeuvres
THETAS = np.radians(np.arange(0.0, 150.25, 0.5))


def turn(m: Machine, V: float, rho: float = RHO) -> dict:
    """At airspeed ``V``: the largest force normal to the flight path over the weight, over every axis angle —
    **instantaneous** (full thrust, whatever it does to the speed) and **sustained** (the thrust's part along the
    flight path pays the drag: the speed holds) — capped at the structure's load factor; the level turn's rate and
    radius at each (rate = g sqrt(n^2 - 1) / V)."""
    q, W = 0.5 * rho * V * V, m.weight
    best = {"inst": (0.0, 0.0), "sus": (0.0, 0.0)}
    for th in m.thetas():
        la, da = m.areas(float(th))
        T = m.thrust(V, float(th))
        s, c = math.sin(th), math.cos(th)
        n_i = (q * la + T * s) / W
        if n_i > best["inst"][0]:
            best["inst"] = (n_i, float(th))
        if c > 1e-3 and T * c >= q * da:
            n_s = (q * la + min(T, q * da / c) * s) / W
            if n_s > best["sus"][0]:
                best["sus"] = (n_s, float(th))
    out = {"V": V}
    for tag, (n, th) in best.items():
        n = min(n, m.n_struct)
        a = G * math.sqrt(max(n * n - 1.0, 0.0))
        out.update({f"n_{tag}": n, f"theta_{tag}_deg": math.degrees(th), f"rate_{tag}_deg_s": math.degrees(a / V) if a > 0 else 0.0,
                    f"radius_{tag}_m": V * V / a if a > 1e-6 else math.inf})
    return out


def level(m: Machine, V: float, rho: float = RHO) -> dict | None:
    """Steady level flight at ``V``: the axis angle at which the lift and the thrust's upward part carry the weight
    while its forward part pays the drag, the thrust and the electrical power; None when nothing balances."""
    q, W = 0.5 * rho * V * V, m.weight

    def residual(th):                                              # upward force left over when the thrust pays the drag
        la, da = m.areas(th)
        c = math.cos(th)
        T = q * da / c if c > 1e-6 else math.inf
        return q * la + T * math.sin(th) - W, T

    top = min(math.radians(89.0), m.theta_max)
    prev = (0.0, *residual(0.0))
    for th in np.arange(math.radians(0.5), top + 1e-9, math.radians(0.5)):
        r, T = residual(float(th))
        if prev[1] < 0 <= r:
            lo, hi = prev[0], float(th)
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if residual(mid)[0] < 0 else (lo, mid)
            r, T = residual(hi)
            if T > m.thrust(V, hi) + 1e-9:
                return None
            return {"V": V, "theta_deg": math.degrees(hi), "thrust": T, "power_w": m.power(T, V * math.cos(hi)),
                    "wing_share": q * m.areas(hi)[0] / W}
        prev = (float(th), r, T)
    return None


def stall_speed(m: Machine, rho: float = RHO) -> float:
    """The slowest speed the **wing** alone carries the weight (the lift area's maximum)."""
    la = max(m.areas(float(th))[0] for th in np.radians(np.arange(0.0, 25.0, 0.5)))
    return math.sqrt(2 * m.weight / (rho * la)) if la > 0 else math.inf


def min_speed(m: Machine) -> float:
    """The slowest steady level flight: 0 for what hovers, otherwise the wing's stall speed."""
    if m.hovers:
        return 0.0
    return stall_speed(m)


def top_speed(m: Machine) -> float:
    lo = None
    for V in np.arange(10.0, 121.0, 2.5):
        if level(m, float(V)) is not None:
            lo = float(V)
    return lo if lo is not None else math.nan


def _slew(theta, target, rate, dt):
    return theta + max(-rate * dt, min(rate * dt, target - theta))


def brake(m: Machine, V0: float = 25.0, dt: float = 0.005, rho: float = RHO) -> dict:
    """From level flight at ``V0`` to a standstill at constant height (what hovers), or to the minimum speed at idle
    (what does not). The axis tips back at ``slew``; at every instant the angle and the thrust give the largest
    deceleration that still carries the weight."""
    if not m.hovers:                                                   # a wing: propellers at idle, the lift held at the weight
        V, x, t, v_end, peak = V0, 0.0, 0.0, 1.2 * stall_speed(m), 0.0
        while V > v_end and t < 120.0:
            q = 0.5 * rho * V * V
            th = next((float(a) for a in np.radians(np.arange(0.0, 25.0, 0.25)) if q * m.areas(float(a))[0] >= m.weight), None)
            if th is None:
                break
            dec = q * m.areas(th)[1] / m.mass
            peak = max(peak, dec)
            V -= dec * dt * 4
            x += V * dt * 4
            t += dt * 4
        return {"V0": V0, "to": v_end, "time_s": t, "distance_m": x, "peak_g": peak / G, "stops": False}
    V, x, t = V0, 0.0, 0.0
    lv = level(m, V0)
    theta = math.radians(lv["theta_deg"]) if lv else 0.0
    v_end = 0.3 if m.hovers else 1.2 * stall_speed(m)
    peak = 0.0
    while V > v_end and t < 60.0:
        q, W = 0.5 * rho * V * V, m.weight
        best = None
        for th in m.thetas():
            la, da = m.areas(float(th))
            s, c = math.sin(th), math.cos(th)
            T = (W - q * la) / s if s > 1e-3 else (0.0 if q * la >= W else math.inf)
            if T < -1e-6 or T > m.thrust(V, float(th)):                # it would balloon, or the thrust cannot hold it up
                continue
            if not m.hovers:
                T = 0.0                                                # a wing brakes with its propellers at idle: a glide at 1 g
                if q * la < W:
                    continue
            dec = (q * da - T * c) / m.mass
            if best is None or dec > best[0]:
                best = (dec, float(th), T)
        if best is None:
            break
        theta = _slew(theta, best[1], m.slew, dt)
        la, da = m.areas(theta)
        s, c = math.sin(theta), math.cos(theta)
        T = min(max((W - q * la) / s if s > 1e-3 else 0.0, 0.0), m.thrust(V, theta)) if m.hovers else 0.0
        dec = (q * da - T * c) / m.mass
        peak = max(peak, dec)
        V -= dec * dt
        x += V * dt
        t += dt
    return {"V0": V0, "to": 0.0 if m.hovers else v_end, "time_s": t, "distance_m": x, "peak_g": peak / G, "stops": m.hovers}


def accelerate(m: Machine, V1: float = 25.0, V0: float | None = None, dt: float = 0.005, rho: float = RHO) -> dict:
    """From a hover (or the minimum speed) to ``V1`` at constant height and full thrust: the axis tips forward at
    ``slew``, held where the thrust's upward part and the lift carry the weight."""
    V = (0.0 if m.hovers else 1.2 * stall_speed(m)) if V0 is None else V0
    start = V
    theta, x, t = (math.pi / 2 if m.hovers else 0.1), 0.0, 0.0
    while V < V1 and t < 60.0:
        q, W = 0.5 * rho * V * V, m.weight
        best = None
        for th in m.thetas(math.pi / 2):
            la, da = m.areas(float(th))
            T = m.thrust(V, float(th))
            if q * la + T * math.sin(th) < W:
                continue
            acc = (T * math.cos(th) - q * da) / m.mass
            if best is None or acc > best[0]:
                best = (acc, float(th))
        if best is None:
            break
        theta = _slew(theta, best[1], m.slew, dt)
        la, da = m.areas(theta)
        acc = (m.thrust(V, theta) * math.cos(theta) - q * da) / m.mass
        V += acc * dt
        x += V * dt
        t += dt
    return {"V0": start, "V1": V1, "time_s": t, "distance_m": x, "mean_g": (V1 - start) / max(t, 1e-9) / G}


def reversal(m: Machine, V: float = 15.0) -> dict:
    """Flying at ``V``, to flying back the way it came at ``V``: the quicker of **stop and go** (what hovers) and a
    180 degree turn at the instantaneous rate (the speed it costs is not counted: optimistic for the wings)."""
    tr = turn(m, V)
    out = {"turn_time_s": 180.0 / tr["rate_inst_deg_s"] if tr["rate_inst_deg_s"] > 0 else math.inf, "turn_width_m": 2 * tr["radius_inst_m"]}
    if m.hovers:
        b, a = brake(m, V), accelerate(m, V)
        out.update({"stop_go_time_s": b["time_s"] + a["time_s"], "stop_go_reach_m": b["distance_m"]})
    how = "stop and go" if m.hovers and out["stop_go_time_s"] < out["turn_time_s"] else "turn"
    out.update({"how": how, "time_s": out["stop_go_time_s"] if how == "stop and go" else out["turn_time_s"],
                "space_m": out["stop_go_reach_m"] if how == "stop and go" else out["turn_width_m"]})
    return out


def climb(m: Machine, h: float = 150.0, dt: float = 0.01, rho: float = RHO) -> dict:
    """From the ground to ``h``: straight up from a standstill at full thrust (what hovers), or at the best rate of
    climb on the wing from its climb speed (what does not: no take-off run counted)."""
    if m.hovers:
        v = z = t = wh = 0.0
        while z < h and t < 120.0:
            T = m.t_max(v)
            v += (T - m.weight - 0.5 * rho * v * v * m.cda0) / m.mass * dt
            z += v * dt
            t += dt
            wh += m.power(T, v) * dt / 3600.0
        return {"time_s": t, "rate_m_s": v, "energy_wh": wh, "how": "vertical"}
    best = (0.0, 0.0)
    for V in np.arange(8.0, 60.0, 1.0):
        q = 0.5 * rho * V * V
        la, da = None, None
        for th in np.radians(np.arange(0.0, 14.0, 0.25)):
            la_, da_ = m.areas(float(th))
            if q * la_ >= m.weight:
                la, da = la_, da_
                break
        if la is None:
            continue
        roc = (m.t_max(float(V)) - q * da) * V / m.weight
        best = max(best, (roc, float(V)))
    roc, V = best
    return {"time_s": h / roc if roc > 0 else math.inf, "rate_m_s": roc, "energy_wh": m.power(m.t_max(V), V) * (h / roc) / 3600.0 if roc > 0 else math.nan,
            "how": f"on the wing at {V:.0f} m/s"}


def endurance(m: Machine) -> dict:
    """Minutes on the usable pack: hovering, at the best cruise speed, and where that speed is."""
    usable = USABLE * m.battery_wh
    hover = usable / m.power(m.weight, 0.0) * 60.0 if m.hovers else math.nan
    best = None
    for V in np.arange(10.0, 45.0, 1.0):
        lv = level(m, float(V))
        if lv and lv["theta_deg"] < 25.0 and (best is None or lv["power_w"] < best[1]):
            best = (float(V), lv["power_w"])
    return {"hover_min": hover, "hover_w": m.power(m.weight, 0.0) if m.hovers else math.nan, "cruise_min": usable / best[1] * 60.0 if best else math.nan,
            "cruise_w": best[1] if best else math.nan, "cruise_v": best[0] if best else math.nan, "usable_wh": usable}


def gun_pass(m: Machine, v_bandit: float = 0.0, r_open: float = 40.0, r_break: float = 15.0) -> dict:
    """A firing pass with a gun fixed along the body axis, on a bandit flying straight at ``v_bandit``: the aircraft
    comes up its tail with the axis on the flight path, so it flies on its wing at the slowest speed that leaves it
    a margin (1.2 x the wing's stall), or faster if the bandit is. The window is the time between ``r_open`` and
    ``r_break`` (the emitter's 30 m sits inside); matching the bandit's speed holds the window open."""
    if m.gun == "cradle" and m.hovers:                                 # it flies at the bandit's speed, the cradle on the bandit
        return {"v_pass": v_bandit, "closing_m_s": 0.0, "window_s": math.inf, "holds": True, "shots_in_window": math.inf, "reattack_s": 0.0}
    v_pass = max(1.2 * stall_speed(m), v_bandit)
    closing = v_pass - v_bandit
    window = (r_open - r_break) / closing if closing > 0.05 else math.inf
    rv = reversal(m, v_pass)
    return {"v_pass": v_pass, "closing_m_s": closing, "window_s": window, "holds": closing <= 0.05,
            "shots_in_window": math.inf if closing <= 0.05 else max(int(window / 1.0), 0),          # one shot a second: the plenum's refill (hand estimate)
            "reattack_s": rv["time_s"] + (2 * r_open / v_pass if closing > 0.05 else 0.0)}


def scorecard(machines: dict | None = None, speeds=(10.0, 15.0, 25.0)) -> list:
    """One row per aircraft: the numbers the job asks for."""
    rows = []
    for key, m in (machines or fleet()).items():
        cl, en, br, ac, rv, gp = climb(m), endurance(m), brake(m), accelerate(m), reversal(m), gun_pass(m)
        row = {"key": key, "aircraft": m.name, "mass [kg]": m.mass, "payload [kg]": m.payload_kg, "span [m]": m.span, "hovers": m.hovers, "gun": m.gun,
               "static thrust / weight": m.t_max(0.0) / m.weight, "wing stall [m/s]": stall_speed(m), "min speed [m/s]": min_speed(m),
               "top speed [m/s]": top_speed(m), "150 m climb [s]": cl["time_s"], "climb [m/s]": cl["rate_m_s"]}
        for V in speeds:
            t = turn(m, V)
            row.update({f"n inst @{V:.0f}": t["n_inst"], f"rate inst @{V:.0f} [deg/s]": t["rate_inst_deg_s"], f"radius inst @{V:.0f} [m]": t["radius_inst_m"],
                        f"n sus @{V:.0f}": t["n_sus"], f"rate sus @{V:.0f} [deg/s]": t["rate_sus_deg_s"], f"radius sus @{V:.0f} [m]": t["radius_sus_m"]})
        row.update({"stops": br["stops"], "brake 25 -> end [s]": br["time_s"], "brake distance [m]": br["distance_m"], "brake ends at [m/s]": br["to"],
                    "to 25 m/s [s]": ac["time_s"], "to 25 m/s [m]": ac["distance_m"], "reverse at 15 m/s [s]": rv["time_s"],
                    "reverse space [m]": rv["space_m"], "reverse how": rv["how"], "hover [min]": en["hover_min"], "hover [W]": en["hover_w"],
                    "cruise [min]": en["cruise_min"], "cruise [W]": en["cruise_w"], "cruise at [m/s]": en["cruise_v"],
                    "pass speed, hovering bandit [m/s]": gp["v_pass"], "firing window [s]": gp["window_s"], "re-attack [s]": gp["reattack_s"],
                    "source": m.source})
        rows.append(row)
    return rows


# =================================================================================================================== KPIs
def fleet_unarmed() -> dict:
    """Every aircraft without an emitter: the airframes against each other."""
    return {"merlin": merlin(), "peregrine": peregrine("B"), "xwing": xwing(), "emerillon": emerillon("ER2"),
            "gavilan_a": gavilan("A", emitters=0), "gavilan_b": gavilan("B", emitters=0), "gavilan_f": gavilan("F", emitters=0),
            "gavilan_f_broad": gavilan("F", emitters=0, p=F_BROAD)}


BANDIT_BAND = (0.0, 20.0)          # the speeds a bandit flies at [m/s]: it hovers, and the job says up to 20


def level_acceleration(m: Machine, V: float = 15.0, rho: float = RHO) -> float:
    """The largest acceleration along a level flight path at ``V`` [g], the weight still carried; nan below the stall."""
    q, W = 0.5 * rho * V * V, m.weight
    best = math.nan
    for th in m.thetas(math.pi / 2):
        la, da = m.areas(float(th))
        T = m.thrust(V, float(th))
        if q * la + T * math.sin(th) >= W:
            a = (T * math.cos(th) - q * da) / W
            best = a if not math.isfinite(best) else max(best, a)
    return best


def speed_band_share(m: Machine, band=BANDIT_BAND) -> float:
    """The share of the bandit's speed band in which the aircraft can fly level beside it."""
    lo, hi = band
    return max(0.0, min(top_speed(m), hi) - max(min_speed(m), lo)) / (hi - lo)


# Each KPI: the column, what it measures and why it is there, which way is better, and what "cannot" counts as.
# ``bandit``: what decides a chase of a quadcopter that stops, hovers, reverses and climbs (the job's requirements 1-6).
# ``general``: what any aircraft of the lab is judged on, chase or not.
KPIS = [
    dict(key="rate10", group="bandit", label="turn rate at 10 m/s [deg/s]", better="high", why="staying inside a slow bandit's turn"),
    dict(key="rate15", group="bandit", label="turn rate at 15 m/s [deg/s]", better="high", why="the bandit's middle speed"),
    dict(key="rate25", group="bandit", label="turn rate at 25 m/s [deg/s]", better="high", why="turning while catching up"),
    dict(key="reverse", group="bandit", label="reverse direction at 15 m/s [s]", better="low", why="the bandit reverses"),
    dict(key="stop", group="bandit", label="stop from 25 m/s [m]", better="low", why="the bandit stops; 'cannot' = it overshoots"),
    dict(key="accel", group="bandit", label="level acceleration at 15 m/s [g]", better="high", why="closing a gap that opened"),
    dict(key="climb", group="bandit", label="ground to 150 m [s]", better="low", why="the bandit climbs away"),
    dict(key="band", group="bandit", label="bandit speeds it can match, 0-20 m/s [%]", better="high", why="flying beside it, hovering included"),
    dict(key="repass", group="bandit", label="back on a hovering bandit after a pass [s]", better="low", why="0 = it never has to leave"),
    dict(key="top", group="general", label="top speed [m/s]", better="high", why="reaching the place"),
    dict(key="cruise", group="general", label="cruise endurance [min]", better="high", why="time on station on the wing"),
    dict(key="range", group="general", label="still-air range [km]", better="high", why="cruise endurance x its speed"),
    dict(key="hover", group="general", label="hover endurance [min]", better="high", why="'cannot' = it does not hover"),
    dict(key="whkm", group="general", label="energy per kilometre [Wh/km]", better="low", why="what a trip costs"),
    dict(key="span", group="general", label="width in flight [m]", better="low", why="gaps, transport, the 4 m alley"),
]


def kpi_table(machines: dict | None = None) -> list:
    """One row per aircraft with every KPI's value (None where the aircraft cannot do it), unarmed by default."""
    rows = []
    for key, m in (machines or fleet_unarmed()).items():
        en, br, rv, cl = endurance(m), brake(m), reversal(m), climb(m)
        v_pass = 1.2 * stall_speed(m)
        acc = level_acceleration(m)
        row = {"key": key, "aircraft": m.name, "mass_kg": m.mass, "hovers": m.hovers, "source": m.source,
               "rate10": turn(m, 10.0)["rate_inst_deg_s"] or None, "rate15": turn(m, 15.0)["rate_inst_deg_s"] or None,
               "rate25": turn(m, 25.0)["rate_inst_deg_s"] or None, "reverse": rv["time_s"] if math.isfinite(rv["time_s"]) else None,
               "stop": br["distance_m"] if br["stops"] else None, "accel": acc if math.isfinite(acc) else None, "climb": cl["time_s"],
               "band": 100.0 * speed_band_share(m), "repass": 0.0 if m.hovers else reversal(m, v_pass)["time_s"] + 2 * 40.0 / v_pass,
               "top": top_speed(m), "cruise": en["cruise_min"], "range": en["cruise_min"] * 60.0 * en["cruise_v"] / 1e3,
               "hover": en["hover_min"] if m.hovers else None, "whkm": en["cruise_w"] / en["cruise_v"] / 3.6, "span": m.span}
        rows.append(row)
    return rows


def kpi_scores(rows: list) -> dict:
    """Each KPI scaled to the best aircraft (1 = the best of the fleet, 0 = cannot do it), and their plain mean per
    group — equal weights, stated: a summary to read beside the bars, not a verdict."""
    out = {r["key"]: {} for r in rows}
    for k in KPIS:
        vals = [r[k["key"]] for r in rows if r[k["key"]] is not None]
        for r in rows:
            v = r[k["key"]]
            if v is None:
                s = 0.0
            elif k["better"] == "high":
                s = v / max(vals) if max(vals) > 0 else 0.0
            else:
                best = min(vals)
                s = 1.0 if v <= best + 1e-12 else (best / v if best > 0 else 1.0 / (1.0 + v / max(max(vals), 1e-9) * 4.0))
            out[r["key"]][k["key"]] = s
    for key, s in out.items():
        for g in ("bandit", "general"):
            ks = [k["key"] for k in KPIS if k["group"] == g]
            s[g] = sum(s[k] for k in ks) / len(ks)
    return out


# ============================================================================================================ the net
# The net launcher (labs/aero/exp/net_drone.py, net_launcher_cad.py on dev_net_launcher @ bb041dd) is the recoilless golf
# cannon firing a 150 g cartridge: a 2.4 m net of 1 mm Dyneema, four 20 g corner weights splayed 12 degrees, 40 m/s.
# Its MuJoCo test against a hovering 1.2 kg quadcopter, from a fixed launcher: captured 5/6 at 4 m, 6/6 at 6 m, 5/6 at
# 8 m, 6/6 at 10 m, 0/6 at 12 m, 1/6 at 15 m — so the net works between 4 and 10 m and not beyond.
# The ground piece is a 600 mm tube on a 3 L tank at 8.5 bar. For an aircraft, the same gas model
# (golf_recoilless.shot with the cartridge, sized for zero recoil) gives, for a 0.30 m tube on a 1.0 L tank and a 25 mm
# valve: 29.8 bar gauge, a 36.9 mm throat, 40.0 m/s, 19.1 g of air per shot, net impulse -0.015 N s. One tank is one shot.
# The masses below are HAND ESTIMATES (no part chosen); MARGIN is on their sum.
NET = {
    "source": "net_drone.py results and golf_recoilless.shot (dev_net_launcher @ bb041dd); masses are hand estimates",
    "range_m": (4.0, 10.0), "captures": {4: "5/6", 6: "6/6", 8: "5/6", 10: "6/6", 12: "0/6", 15: "1/6"},
    "muzzle_m_s": 40.0, "net_side_m": 2.4, "open_span_m": 2.95, "flight_s_10m": 0.51, "cartridge_g": 150.0,
    "tube_m": 0.30, "bore_mm": 42.7, "pressure_bar_g": 29.8, "tank_l": 1.0, "air_g_per_shot": 19.1,
    "parts_g": {"tube 0.30 m, 42.7 mm bore, 1 mm aluminium": 112.0, "chamber and 37 mm rear venturi": 90.0, "25 mm fast valve with its pilot": 150.0,
                "tank 1.0 L, 30 bar": 250.0, "net cartridge": 150.0, "mounts": 40.0},
    "second_shot_g": {"second tube, chamber and venturi": 202.0, "second valve": 150.0, "second tank": 250.0, "second cartridge": 150.0},
    "margin": 1.25,
}
NET_CDA = 0.35 * math.pi * 0.024 ** 2 + 0.0008      # a 48 mm tube's face and a tank under the belly, faired (hand estimate) [m2]


def net_kg(shots: int = 1) -> float:
    """A net launcher with ``shots`` cartridges ready [kg] (one tube, valve and tank per shot): hand estimate x margin."""
    g = sum(NET["parts_g"].values()) + max(shots - 1, 0) * sum(NET["second_shot_g"].values())
    return g * NET["margin"] / 1e3


def with_payload(m: Machine, kg: float, cda: float = 0.0, label: str = "", gun: str = "fixed") -> Machine:
    """The same aircraft carrying ``kg`` more with ``cda`` more drag area. The airframe is not strengthened or
    re-balanced for it: an optimistic loading of an aircraft that was not designed for the load."""
    return dataclasses.replace(m, name=f"{m.name} + {label}" if label else m.name, mass=m.mass + kg, cda0=m.cda0 + cda,
                               payload_kg=m.payload_kg + kg, gun=gun)


# GAVILAN-N: a fixed wing sized for the net. The wing is sized so that it still flies at 11 m/s with the launcher
# aboard (the pass over the bandit is flown slowly); two tractors on the wing leave the belly free for the tube,
# which points forward and down so the aircraft passes over the bandit, not through its own net.
N_PARAMS = {"span": 1300.0, "root_chord": 290.0, "taper": 0.60, "fuselage_length": 880.0, "fuselage_diameter": 70.0, "nose_length": 230.0,
            "nacelle_y": 230.0, "tail_span": 420.0, "tail_chord": 120.0, "fin_height": 150.0}
CONFIG_N = ef.Config("eco2807-1300", "hq7x4x3", "speedybee-f405v4-55", "tattu-r6-1700-160c", 1, 1)   # two motors, the low-pitch propeller
NET_DEPRESSION_DEG = 40.0          # the tube below the flight path on a fixed wing (proposed)


def gavilan_n(shots: int = 1) -> Machine:
    m = gavilan("F", emitters=0, p=N_PARAMS, config=CONFIG_N)
    m = dataclasses.replace(m, name="Gavilan-N (fixed wing for the net)", gun="none")
    return with_payload(m, net_kg(shots), NET_CDA, "", gun="fixed") if shots else m


def fleet_net(shots: int = 1) -> dict:
    """The aircraft that could lift the launcher, each carrying it (``shots`` cartridges). MERLIN flies it in place
    of its 0.12 kg gas sensor; PEREGRINE (0.66 kg all-up) cannot lift 1 kg and is left out."""
    kg = net_kg(shots)
    a = gavilan("A", emitters=0)
    return {"merlin": with_payload(merlin(), kg - 0.12, NET_CDA, "net"), "xwing": with_payload(xwing(), kg, NET_CDA, "net"),
            "emerillon": with_payload(emerillon("ER2"), kg, NET_CDA, "net"),
            "gavilan_a": with_payload(a, kg + CRADLE_G / 1e3, NET_CDA, "net", gun="cradle"),
            "gavilan_f": with_payload(gavilan("F", emitters=0), kg, NET_CDA, "net"),
            "gavilan_f_broad": with_payload(gavilan("F", emitters=0, p=F_BROAD), kg, NET_CDA, "net"),
            "gavilan_n": gavilan_n(shots)}


def net_pass(m: Machine, v_bandit: float = 0.0) -> dict:
    """One net attack on a bandit flying straight at ``v_bandit``. What hovers flies beside the bandit inside the
    net's range and fires when it likes (closing speed 0). A wing overflies it at the slowest speed that leaves a
    margin (1.2 x its stall) with the tube pointing down and ahead: the shot has to be released within the time the
    net's half span takes to pass at the closing speed (``trigger_s``), and the aircraft is inside the net's range
    for ``in_range_s``. ``lineup_radius_m`` is the turn it lines up with at that speed."""
    lo, hi = NET["range_m"]
    if m.hovers:
        return {"v_pass": v_bandit, "closing_m_s": 0.0, "trigger_s": math.inf, "in_range_s": math.inf, "holds": True,
                "lineup_radius_m": turn(m, max(v_bandit, 5.0))["radius_inst_m"], "next_pass_s": 0.0}
    v = max(1.2 * stall_speed(m), v_bandit)
    closing = v - v_bandit
    h = 0.5 * (lo + hi) * math.sin(math.radians(NET_DEPRESSION_DEG))                # the height of the pass over the bandit
    reach = math.sqrt(max(hi * hi - h * h, 0.0))                                   # ground distance at which the range opens
    if closing < 0.05:
        return {"v_pass": v, "closing_m_s": 0.0, "trigger_s": math.inf, "in_range_s": math.inf, "holds": True,
                "lineup_radius_m": turn(m, v)["radius_inst_m"], "next_pass_s": 0.0}
    return {"v_pass": v, "closing_m_s": closing, "trigger_s": 0.5 * NET["open_span_m"] / closing, "in_range_s": reach / closing, "holds": False,
            "lineup_radius_m": turn(m, v)["radius_inst_m"], "next_pass_s": reversal(m, v)["time_s"] + 2 * 40.0 / v, "height_m": h}


NET_KPIS = [
    dict(key="tw", label="static thrust / weight with the net", better="high", why="below 1.15 it no longer hovers"),
    dict(key="vmin", label="slowest level flight [m/s]", better="low", why="0 = it hovers beside the bandit"),
    dict(key="trigger0", label="trigger tolerance, hovering bandit [s]", better="high", why="'holds' = no deadline: it sits in range"),
    dict(key="trigger10", label="trigger tolerance, bandit at 10 m/s [s]", better="high", why="time the net's half span takes to pass"),
    dict(key="lineup", label="line-up turn radius at the pass speed [m]", better="low", why="correcting onto a bandit that sidesteps"),
    dict(key="rate15", label="turn rate at 15 m/s [deg/s]", better="high", why="'cannot' = below its stall with the net"),
    dict(key="repass", label="next pass on a hovering bandit [s]", better="low", why="0 = it never leaves; one tank is one shot"),
    dict(key="top", label="top speed [m/s]", better="high", why="it must outrun a 20 m/s bandit with margin"),
    dict(key="cruise", label="cruise endurance with the net [min]", better="high", why="waiting for the bandit"),
]


def net_table(machines: dict | None = None) -> list:
    rows = []
    for key, m in (machines or fleet_net()).items():
        p0, p10, en = net_pass(m, 0.0), net_pass(m, 10.0), endurance(m)
        r15 = turn(m, 15.0)["rate_inst_deg_s"]
        rows.append({"key": key, "aircraft": m.name, "mass_kg": m.mass, "payload_kg": m.payload_kg, "hovers": m.hovers, "source": m.source,
                     "tw": m.t_max(0.0) / m.weight, "vmin": min_speed(m), "v_pass": p0["v_pass"],
                     "trigger0": None if p0["holds"] else p0["trigger_s"], "trigger10": None if p10["holds"] else p10["trigger_s"],
                     "holds0": p0["holds"], "holds10": p10["holds"], "lineup": p0["lineup_radius_m"] if math.isfinite(p0["lineup_radius_m"]) else None,
                     "rate15": r15 or None, "repass": p0["next_pass_s"], "top": top_speed(m), "cruise": en["cruise_min"],
                     "hover_min": en["hover_min"] if m.hovers else None})
    return rows


# ============================================================================================================== the hawk
# GAVILAN-H: the fixed wing flown like a hawk. A hawk does not hover for a living, but it does not obey its stall
# either: it flares its wings and tail far past it to brake and to turn, on muscle. The aircraft's version: F's layout
# on the broad wing (a hawk's wing loading), the low-pitch propellers for static thrust, and — the one new thing — a
# large all-moving tail in the propellers' wash with differential thrust for yaw, so the attitude is held beyond the
# stall. It is thrown and belly-lands like F; it has no feet. In the model that is F broad allowed to tip its axis to
# 90 degrees (PROPOSED: the blown tail's authority is not designed or checked; the CFD sweep is what would say).
H_PARAMS = {"span": 1000.0, "root_chord": 280.0, "tail_span": 420.0, "tail_chord": 130.0}
CONFIG_H = ef.Config("eco2807-1300", "hq7x4x3", "speedybee-f405v4-55", "tattu-r6-1700-160c", 1, 1)


def gavilan_h(emitters: int = 0) -> Machine:
    m = gavilan("F", emitters=emitters, p=H_PARAMS, config=CONFIG_H)
    return dataclasses.replace(m, name="Gavilan-H (hawk: broad wing, blown tail, flies past its stall)", theta_limit=math.radians(90.0), stops=True,
                               roll_rate=math.radians(300.0),
                               notes={**m.notes, "post-stall": "PROPOSED: attitude held beyond the stall by a blown all-moving tail and differential thrust"})
