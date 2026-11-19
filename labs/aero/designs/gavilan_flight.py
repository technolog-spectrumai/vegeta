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


def f_mass_budget(q=None, c: ef.Config = CONFIG_F, emitters: int = 1, *, pod_and_boom: bool = False, boom: bool = True,
                  packs: bool = True) -> og_mass.Budget:
    """The fixed wings' mass. ``pod_and_boom`` (GAVILAN-H): the printed skin stops at the wing's trailing edge and a
    carbon tube under a thin fairing carries the tail (``boom`` False when something else is that tube: the net's jet
    pipe); ``packs`` False leaves the battery out, to be placed for the balance."""
    g = f_geometry(q)
    motor, prop, esc, bat = (mk.part(k) for k in (c.motor, c.propeller, c.esc, c.battery))
    petg = mk.part("petg-hf")
    b, I = og_mass.Budget(), og_mass.Item
    x_w = g["nose"]                                                    # wing leading edge from the nose
    for s in (1, -1):
        b.add(I("motor", motor.mass_g / 1e3, (x_w - 0.05, s * 0.19, 0), group="propulsion", source=c.motor),
              I("propeller", prop.mass_g / 1e3, (x_w - 0.07, s * 0.19, 0), group="propulsion", source=c.propeller))
    b.add(I("ESC board with the flight controller", esc.mass_g / 1e3, (x_w, 0, 0), group="propulsion", source=c.esc))
    for k in range(c.packs if packs else 0):
        b.add(I(f"pack {k + 1}", bat.mass_g / 1e3, (x_w + 0.03, 0, 0), group="battery", source=c.battery))
    for part in mk.FIXED:
        if part.group == "fc" and esc.with_fc:
            continue
        b.add(I(part.name, part.mass_g / 1e3, (0.08, 0, 0), group="avionics", source=part.key))
    q_ = {**F_PARAMS, **(q or {})}
    x_pod = x_w + q_["root_chord"] / 1e3                                # the pod ends at the wing's trailing edge
    skin_len = (x_pod if pod_and_boom else g["length"])
    skin = math.pi * g["d"] * 0.80 * skin_len * ef.WALL * ef.PRINT_FACTOR * petg.density
    if pod_and_boom and boom:
        Lb = g["length"] - x_pod
        b.add(I("tail boom (bought carbon tube)", 0.035 * Lb + 0.004, (x_pod + 0.5 * Lb, 0, 0), group="airframe", source="hand estimate: 35 g/m"),
              I("boom fairing (printed foam)", math.pi * 0.045 * Lb * AREAL_DENSITY, (x_pod + 0.5 * Lb, 0, 0), group="airframe", source="AREAL_DENSITY"))
    b.add(I("servos", F_N_SERVOS * TILT_SERVO_G / 1e3, (x_w + 0.2, 0, 0), group="avionics", source="hand estimate: four 9 g servos"),
          I("power wiring", 0.012 * c.packs + 0.025, (x_w, 0, 0), group="propulsion", source="emerillon_flight's allowance"),
          I("fuselage skin (PETG)", skin, (0.45 * skin_len, 0, 0), group="airframe", source="emerillon_flight.printed_masses' rule"),
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


def brake(m: Machine, V0: float = 25.0, dt: float = 0.005, rho: float = RHO, v_end: float | None = None) -> dict:
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
    v_end = (0.3 if v_end is None else v_end) if m.hovers else 1.2 * stall_speed(m)
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
    return {"V0": V0, "to": v_end if m.hovers else v_end, "time_s": t, "distance_m": x, "peak_g": peak / G, "stops": m.hovers}


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
            "gavilan_f_broad": gavilan("F", emitters=0, p=F_BROAD), "gavilan_h": gavilan_h()}


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
H_PARAMS = {"span": 1000.0, "root_chord": 280.0, "nacelle_y": 200.0, "fuselage_length": 860.0, "tail_span": 580.0, "tail_chord": 150.0, "fin_height": 130.0}   # gavilan.GavilanH's defaults
TAILERON_MAX_DEG = 30.0            # the tail halves' travel each way (proposed)
TAIL_CL_PER_RAD = 2.2              # a low-aspect-ratio plate's lift slope in a jet (hand estimate: about half the free value of its aspect ratio)
WING_CP_STALLED = 0.42             # the stalled wing's centre of pressure along its chord (flat plate: 0.25 + 0.25 sin(alpha), near 45-60 deg)
CONFIG_H = ef.Config("eco2807-1300", "hq7x4x3", "speedybee-f405v4-55", "tattu-r6-1700-160c", 1, 1)


# The one-shot light net interceptor (labs/aero/exp/net_light_design.py, net_oneshot_cad.py, gavilan_net_cad.py on
# dev_gavilan_net @ d9f6cb8): a 70 g cartridge (a 1.5 m net of 1 mm Dyneema, four 8 g weights) from a 32 mm bore at
# 45 m/s, a rear venturi of 27 mm, one charge vessel opened by a burst diaphragm - no bottle, regulator, plenum or valve.
# Its MuJoCo tests from a fixed launcher: a 1.2 kg quadcopter crossing at 5 m/s captured 5/5 at each of 3, 4, 5, 6 and
# 8 m; a racing drone crossing at 15 m/s 0/5 at every range. So it works to 8 m on a target it is not moving against.
# The vessel was provisional there (80 mL); the same gas model (golf_recoilless.shot, the 0.25 m tube GAVILAN carries, a
# 25 mm diaphragm open in 0.5 ms) gives for 80 mL: 84.2 bar gauge, a 26.9 mm throat, 45.0 m/s, net impulse +0.001 N s,
# 7.7 g of air. The masses are HAND ESTIMATES (the chamber's is net_light_design's); MARGIN is on their sum.
LIGHT_NET = {
    "source": "net_light_design.py, net_light_*.log, gavilan_net_cad.py (dev_gavilan_net @ d9f6cb8); vessel sized with golf_recoilless.shot; masses by hand",
    "range_m": (3.0, 8.0), "captures_quad_5ms": {3: "5/5", 4: "5/5", 5: "5/5", 6: "5/5", 8: "5/5"}, "captures_fpv_15ms": "0/5 at 3, 4, 5, 6 and 8 m",
    "muzzle_m_s": 45.0, "net_side_m": 1.5, "open_span_m": 1.96, "span_at_5m": 1.58, "cartridge_g": 70.0, "tube_m": 0.25, "bore_mm": 32.0,
    "vessel_ml": 80.0, "pressure_bar_g": 84.2, "throat_mm": 26.9, "air_g": 7.7, "backblast_m": 6.4, "shots": 1,
    "parts_g": {"carbon tube 0.25 m, 32 mm bore, 1.5 mm wall": 63.0, "chamber and rear venturi (6061; net_light_design)": 140.0,
                "burst-diaphragm holder and puncture solenoid": 90.0, "charge vessel 80 mL, carbon": 80.0,
                "jet pipe to the tail, 0.57 m carbon, 41 mm, 1 mm wall": 115.0, "pipe fairing and belly blister (printed foam)": 73.0,
                "second tail servo and stub shafts": 15.0, "trigger electronics and wiring": 30.0, "net cartridge": 70.0},
    "margin": 1.25,
}
LIGHT_NET_CDA = 0.0011             # the blister's face (faired) and the fairing's skin (hand estimate) [m2]


def light_net_kg() -> float:
    """The one-shot interceptor as GAVILAN-H carries it [kg]: hand-estimated parts x margin."""
    return sum(LIGHT_NET["parts_g"].values()) * LIGHT_NET["margin"] / 1e3


CG_TARGET_MAC = 0.30               # where GAVILAN-H's centre of gravity is put, along the mean chord (a stable tailed aeroplane's usual place)


def h_mass_budget(net: bool = False) -> og_mass.Budget:
    """GAVILAN-H's mass, balanced: a printed pod to the wing's trailing edge and a carbon boom to the tail (with the
    net aboard the boom **is** the interceptor's jet pipe), the interceptor's parts at their stations in the CAD
    (``GavilanH.net_layout``), and the pack put where the centre of gravity comes to ``CG_TARGET_MAC`` — or as far
    forward as the nose allows (``notes`` on the item says which)."""
    q = gv.h_resolve(gv.H_NET if net else {})
    g = f_geometry(H_PARAMS)
    b = f_mass_budget(H_PARAMS, CONFIG_H, 0, pod_and_boom=True, boom=not net, packs=False)
    nose = q["nose_length"]
    if net:
        N = gv.GavilanH.net_layout(q)
        x = {"carbon tube": 0.5 * (N["x_muzzle"] + N["x_bore"]), "chamber": N["x_bore"] + 25.0, "burst": N["x_bore"] + 20.0, "charge vessel": N["x_vessel"] + 20.0,
             "jet pipe": 0.5 * (N["x_chamber_end"] + N["x_end"]), "pipe fairing": 0.5 * (N["x_bore"] + N["x_end"]), "second tail servo": N["x_end"] - 150.0,
             "trigger": N["x_bore"], "net cartridge": N["x_bore"] - 45.0}
        for name, grams in LIGHT_NET["parts_g"].items():
            key = next(k for k in x if name.startswith(k) or k in name)
            b.add(og_mass.Item(name, grams * LIGHT_NET["margin"] / 1e3, ((x[key] + nose) / 1e3, 0, 0), group="payload", source="LIGHT_NET (hand estimate x 1.25)"))
    bat = mk.part(CONFIG_H.battery)
    target = (nose + CG_TARGET_MAC * g["mac"] * 1e3) / 1e3
    x_bat, clipped = og_mass.place_for_cg(b, bat.mass_g / 1e3, target, (0.075, nose / 1e3 + 0.20))
    b.add(og_mass.Item("pack 1", bat.mass_g / 1e3, (x_bat, 0, 0), group="battery",
                       source=CONFIG_H.battery + (" — as far forward as the nose allows: the balance is NOT reached" if clipped else " — placed for the balance")))
    return b


def gavilan_h(emitters: int = 0, net: bool = False) -> Machine:
    m = gavilan("F", emitters=0, p=H_PARAMS, config=CONFIG_H)
    m = dataclasses.replace(m, name="Gavilan-H (hawk: broad wing, blown tail, flies past its stall)", theta_limit=math.radians(90.0), stops=True,
                            roll_rate=math.radians(300.0), mass=h_mass_budget(False).mass, payload_kg=0.0, gun="none",
                            notes={**m.notes, "post-stall": "PROPOSED: attitude held beyond the stall by a blown all-moving tail and differential thrust"})
    if emitters:
        m = with_payload(m, emitter_kg(emitters), 0.0006, "emitter")
    if net:
        m = dataclasses.replace(m, name=m.name + " + net interceptor", mass=h_mass_budget(True).mass, cda0=m.cda0 + LIGHT_NET_CDA,
                                payload_kg=light_net_kg(), gun="fixed")
    return m


def h_balance(net: bool = True) -> dict:
    """Where GAVILAN-H's centre of gravity sits along the mean chord, where the pack had to go for it, and the pitch
    inertia. The yardstick is the chord position, not a computed neutral point."""
    q = gv.h_resolve()
    g = f_geometry(H_PARAMS)
    b = h_mass_budget(net)
    pr = b.properties()
    pack = next(i for i in b.items if i.name == "pack 1")
    nose = q["nose_length"] / 1e3
    return {"mass": pr.mass, "x_cg": float(pr.com[0]), "cg_mac": float((pr.com[0] - nose) / g["mac"]), "target_mac": CG_TARGET_MAC,
            "pack_x": float(pack.pos[0]), "pack_from_wing_leading_edge": float(pack.pos[0] - nose), "balanced": "NOT reached" not in pack.source,
            "Iyy": float(pr.inertia[1, 1])}


def h_tail(V: float = 0.0, thrust: float | None = None, rho: float = RHO, machine: Machine | None = None) -> dict:
    """GAVILAN-H's blown tail, a HAND ESTIMATE the CFD sweep is meant to replace. Each propeller's wash at the tail
    (momentum theory, fully developed: v = sqrt(V^2 + 2 T / (rho A)) per propeller) over the part of each tail half
    inside it, at full deflection: the pitching moment both halves give together, the rolling moment they give
    against each other, and the yawing moment of the two thrusts' full difference. ``thrust``: the two motors'
    total (default: the weight, as in a hang)."""
    p = gv.h_resolve()
    L = gv.GavilanH.layout(p)
    m = machine or gavilan_h()
    T = m.weight if thrust is None else thrust
    A = math.pi * (p["prop_diameter"] / 2e3) ** 2
    v_wash = math.sqrt(V * V + 2 * (T / 2) / (rho * A))
    q = 0.5 * rho * v_wash ** 2
    cl = TAIL_CL_PER_RAD * math.radians(TAILERON_MAX_DEG)
    force = q * (L["S_blown_mm2"] / 1e6) * cl                           # both halves together
    return {"v_wash": v_wash, "q_wash": q, "force_N": force, "pitch_Nm": force * L["tail_arm"] / 1e3, "roll_Nm": force * p["nacelle_y"] / 1e3,
            "yaw_Nm": m.t_max(V) * 0.5 * p["nacelle_y"] / 1e3, "blown_share": L["blown_share"]}


def h_stall_moment(V: float, alpha_deg: float = 50.0, rho: float = RHO) -> float:
    """The nose-down moment of GAVILAN-H's stalled wing about its quarter chord [N m]: the plate's normal force at its
    centre of pressure (``WING_CP_STALLED``). A hand estimate; the tail has to hold it."""
    g = f_geometry(H_PARAMS)
    a = math.radians(alpha_deg)
    cn = viterna_cd90(g["AR"]) * math.sin(a)
    return 0.5 * rho * V * V * g["S"] * cn * (WING_CP_STALLED - 0.25) * g["mac"]


def h_authority(speeds=(0.0, 10.0, 15.0, 25.0), machine: Machine | None = None) -> list:
    """Can the tail hold the stalled wing? Per speed: the wing's nose-down moment at 50 degrees, the tail's moment in
    the wash at the thrust level flight needs there (``level``; the weight in a hang), and their ratio (> 1 holds)."""
    m, rows = machine or gavilan_h(), []
    for V in speeds:
        lv = level(m, V) if V > 0 else None
        T = lv["thrust"] if lv else m.weight
        need, have = (h_stall_moment(V) if V > 0 else 0.0), h_tail(V, T, machine=m)["pitch_Nm"]
        rows.append({"V": V, "thrust_N": T, "wing_moment_Nm": need, "tail_moment_Nm": have,
                     "tail_at_full_thrust_Nm": h_tail(V, m.t_max(V), machine=m)["pitch_Nm"],
                     "margin": have / need if need > 0 else math.inf})
    return rows


def h_net_table() -> list:
    """GAVILAN-H clean and with the one-shot interceptor, and GAVILAN-A carrying the same interceptor: what the load
    costs, row by row."""
    a = with_payload(gavilan("A", emitters=0), light_net_kg(), LIGHT_NET_CDA, "net interceptor")
    rows = []
    for key, m in (("gavilan_h", gavilan_h()), ("gavilan_h_net", gavilan_h(net=True)), ("gavilan_a_net", a)):
        en, br = endurance(m), brake(m)
        row = {"key": key, "aircraft": m.name, "mass_kg": m.mass, "payload_kg": m.payload_kg, "static thrust / weight": m.t_max(0.0) / m.weight,
               "wing loading [N/m2]": m.weight / m.S, "wing stall [m/s]": stall_speed(m), "hangs on its propellers": m.hovers,
               "hover [min]": en["hover_min"], "hover [W]": en["hover_w"], "cruise [min]": en["cruise_min"], "cruise at [m/s]": en["cruise_v"],
               "top speed [m/s]": top_speed(m), "stop from 25 m/s [m]": br["distance_m"], "reverse at 15 m/s [s]": reversal(m)["time_s"],
               "150 m climb [s]": climb(m)["time_s"]}
        for V in (10.0, 15.0, 25.0):
            row[f"turn rate @{V:.0f} [deg/s]"] = turn(m, V)["rate_inst_deg_s"]
        if key.startswith("gavilan_h"):
            au = h_authority(machine=m)
            row["tail margin at 15 m/s"] = au[2]["margin"]
            row["tail margin at 25 m/s"] = au[3]["margin"]
        rows.append(row)
    return rows


def h_static_margin(net: bool = False) -> dict:
    """GAVILAN-H's stick-fixed static margin, a HAND ESTIMATE (the textbook tail-volume formula, no fuselage term, no
    propeller wash): neutral point = 0.25 + eta V_h (a_t / a)(1 - d eps / d alpha) of the mean chord."""
    q = gv.h_resolve()
    L, g = gv.GavilanH.layout(q), f_geometry(H_PARAMS)
    S_t = L["S_tail_mm2"] / 1e6
    ar_t = (q["tail_span"] / 1e3) ** 2 / (q["tail_span"] * q["tail_chord"] / 1e6)
    a, a_t = 2 * math.pi * g["AR"] / (g["AR"] + 2), 2 * math.pi * ar_t / (ar_t + 2)
    v_h = S_t * (L["tail_arm"] / 1e3) / (g["S"] * g["mac"])
    downwash = 2 * a / (math.pi * g["AR"])
    x_np = 0.25 + 0.9 * v_h * (a_t / a) * (1 - downwash)
    cg = h_balance(net)["cg_mac"]
    return {"tail_volume": v_h, "neutral_point_mac": x_np, "cg_mac": cg, "static_margin": x_np - cg}


# =============================================================================================================== GAVILAN-J
# The hawk's way of hunting, as an aircraft: it sits on a perch, sprints once, flares at the prey, takes it in its
# talons and carries it down. GAVILAN-J is GAVILAN-A cut down for that: one pack (the flight is short), the body one
# pack shorter, guards round the propellers (it touches things), and the talons — four sprung legs on the belly with a
# loose net between them and a line to the aircraft. In the flare the belly leads, so the talons (and the camera,
# tilted 90 degrees) face the bandit, as a hawk's feet do.
CONFIG_J = ef.Config("eco2807-1300", "hq7x4x3", "speedybee-f405v4-55", "tattu-r6-1700-160c", 1, 1)     # four motors, one pack
BANDIT_KG = 1.2                    # the quadcopter the net launcher's tests catch (net_drone.DRONE_M)
CONTACT_SPEED = 3.0                # the closing speed the talons take [m/s] (proposed: a 1.2 kg bandit at 3 m/s is 5.4 J on sprung legs)
SPINUP_S = 0.3                     # from the trigger to full thrust: motors from stopped (hand estimate)
TALONS = {
    "source": "hand estimate: nothing chosen or tested", "reach_m": 0.356, "open_width_m": 0.445,
    "parts_g": {"four sprung carbon legs, 4 mm x 0.42 m, with hooks": 38.0, "loose Dyneema net between the legs": 15.0,
                "hub, hinge and deploy servo": 28.0, "line 3 m of 1.5 mm Dyneema with an elastic section, and its latch": 30.0},
    "margin": 1.25, "cda_m2": 0.0018,
}
GUARDS = {"source": "hand estimate", "parts_g": {"four guard rings, 190 mm, 3 mm carbon rod, with two struts each": 44.0}, "margin": 1.25, "cda_m2": 0.0020}


def talons_kg() -> float:
    return sum(TALONS["parts_g"].values()) * TALONS["margin"] / 1e3


def guards_kg() -> float:
    return sum(GUARDS["parts_g"].values()) * GUARDS["margin"] / 1e3


def j_mass_budget() -> og_mass.Budget:
    b = mass_budget(gv.variant("J"), CONFIG_J, emitters=0)
    g = geometry(gv.variant("J"))
    b.add(og_mass.Item("talons with their line", talons_kg(), (g["x_ac"], 0, -g["d"] / 2 - 0.03), group="payload", source="TALONS (hand estimate x 1.25)"),
          og_mass.Item("propeller guards", guards_kg(), (g["x_prop"], 0, 0), group="airframe", source="GUARDS (hand estimate x 1.25)"))
    return b


def gavilan_j() -> Machine:
    m = gavilan("J", emitters=0, config=CONFIG_J)
    return dataclasses.replace(m, name="Gavilan-J (the hawk's hunt: perch, sprint, talons)", mass=j_mass_budget().mass,
                               cda0=m.cda0 + TALONS["cda_m2"] + GUARDS["cda_m2"], payload_kg=talons_kg(), gun="talons")


def _sprint_trace(m: Machine, v_stop: float = 80.0, dt: float = 0.005, rho: float = RHO) -> dict:
    """Full-thrust level acceleration from a standstill (what hovers) or from the launch speed (a wing, off its
    catapult at 1.2 x its stall): time, distance, speed and electrical energy along the way."""
    V = 0.0 if m.hovers else 1.2 * stall_speed(m)
    theta, x, t, wh = (math.pi / 2 if m.hovers else 0.1), 0.0, 0.0, 0.0
    T_, X_, V_, E_ = [0.0], [0.0], [V], [0.0]
    while V < v_stop and t < 30.0:
        q, W = 0.5 * rho * V * V, m.weight
        best = None
        for th in m.thetas(math.pi / 2)[::2]:
            la, da = m.areas(float(th))
            T = m.thrust(V, float(th))
            if q * la + T * math.sin(th) < W:
                continue
            acc = (T * math.cos(th) - q * da) / m.mass
            if best is None or acc > best[0]:
                best = (acc, float(th))
        if best is None or best[0] < 0.05:
            break
        theta = _slew(theta, best[1], m.slew, dt)
        la, da = m.areas(theta)
        T = m.thrust(V, theta)
        V += (T * math.cos(theta) - q * da) / m.mass * dt
        x += V * dt
        t += dt
        wh += m.power(T, V * math.cos(theta)) * dt / 3600.0
        T_.append(t), X_.append(x), V_.append(V), E_.append(wh)
    return {"t": np.array(T_), "x": np.array(X_), "V": np.array(V_), "wh": np.array(E_)}


@functools.lru_cache(maxsize=None)
def _sprint_tables(key: str) -> tuple:
    m = _SPRINTERS[key]()
    tr = _sprint_trace(m)
    vs = [v for v in (6.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0) if v < tr["V"][-1]]
    br = [brake(m, v, dt=0.01, v_end=CONTACT_SPEED) for v in vs] if m.hovers else []
    return m, tr, np.array([CONTACT_SPEED] + vs), np.array([0.0] + [b["distance_m"] for b in br]), np.array([0.0] + [b["time_s"] for b in br])


def intercept(key: str, distance: float) -> dict:
    """From sitting on the perch to the bandit ``distance`` away, hovering: spin up, sprint at full thrust, then flare
    so as to arrive at ``CONTACT_SPEED``. What cannot stop (a wing) arrives at whatever speed it has: it passes."""
    m, tr, vb, db, tb = _sprint_tables(key)
    if not m.hovers:
        i = int(np.searchsorted(tr["x"], distance))
        i = min(i, len(tr["x"]) - 1)
        return {"time_s": SPINUP_S + float(tr["t"][i]), "arrival_m_s": float(tr["V"][i]), "peak_m_s": float(tr["V"][i]), "catches": False,
                "wh": float(tr["wh"][i]), "launch": "catapult at 1.2 x stall"}
    need = tr["x"] + np.interp(tr["V"], vb, db)                     # distance covered if the flare starts now
    i = min(int(np.searchsorted(need, distance)), len(need) - 1)
    v_peak = float(tr["V"][i])
    t_brake = float(np.interp(v_peak, vb, tb))
    return {"time_s": SPINUP_S + float(tr["t"][i]) + t_brake, "arrival_m_s": CONTACT_SPEED, "peak_m_s": v_peak, "catches": True,
            "wh": float(tr["wh"][i]) + m.power(m.weight, 0.0) * t_brake / 3600.0, "launch": "from its perch"}


_SPRINTERS = {"gavilan_j": lambda: gavilan_j(), "gavilan_a": lambda: gavilan("A", emitters=0), "gavilan_h": lambda: gavilan_h(),
              "emerillon": lambda: emerillon("ER2"), "xwing": lambda: xwing(), "gavilan_f_broad": lambda: gavilan("F", emitters=0, p=F_BROAD),
              "peregrine": lambda: peregrine("B")}


def hawk_table(distances=(50.0, 100.0, 200.0)) -> list:
    """The hawk's hunt in numbers, one row per aircraft: the time from the perch to a hovering bandit at each distance
    (arriving slowly enough to take it), the thrust left with the bandit hanging below, how many such sprints a charge
    gives."""
    rows = []
    for key in _SPRINTERS:
        m = _sprint_tables(key)[0]
        row = {"key": key, "aircraft": m.name, "mass_kg": m.mass, "perches": m.kind == "tailsitter", "stops at the bandit": m.hovers,
               "static thrust / weight": m.t_max(0.0) / m.weight,
               "thrust / weight carrying the bandit": m.t_max(0.0) / ((m.mass + BANDIT_KG) * G)}
        for d in distances:
            r = intercept(key, d)
            row[f"to a bandit at {d:.0f} m [s]"] = r["time_s"]
            row[f"arrival speed at {d:.0f} m [m/s]"] = r["arrival_m_s"]
        r = intercept(key, 100.0)
        back = m.power(m.weight + BANDIT_KG * G, 0.0) * 20.0 / 3600.0 if m.hovers else 0.0          # 20 s of carrying it down (hand)
        row["peak speed in a 100 m sprint [m/s]"] = r["peak_m_s"]
        row["energy per 100 m hunt [Wh]"] = r["wh"] + back
        row["hunts per charge"] = int(USABLE * m.battery_wh / (r["wh"] + back)) if (r["wh"] + back) > 0 else 0
        row["standby"] = "sits on its perch, motors off" if m.kind == "tailsitter" else "on a catapult, or already flying"
        rows.append(row)
    return rows


def j_trade(propellers=("hq7x4x3", "apc7x7e", "apc7x10"), packs=(1, 2), distances=(50.0, 100.0, 200.0)) -> list:
    """GAVILAN-J's sprint against its propeller and its number of packs: the mass, the static thrust over the weight
    alone and with the bandit, and the time from the perch to a bandit at each distance."""
    rows = []
    for prop in propellers:
        for n in packs:
            c = dataclasses.replace(CONFIG_J, propeller=prop, n_parallel=n)
            key = f"_j_{prop}_{n}"

            def make(c=c):
                m = gavilan("J", emitters=0, config=c)
                b = mass_budget(gv.variant("J"), c, emitters=0).mass + talons_kg() + guards_kg()
                return dataclasses.replace(m, name=f"Gavilan-J, {prop}, {n} pack", mass=b, cda0=m.cda0 + TALONS["cda_m2"] + GUARDS["cda_m2"])

            _SPRINTERS[key] = make
            m = _sprint_tables(key)[0]
            row = {"propeller": prop, "packs": n, "mass_kg": m.mass, "static thrust / weight": m.t_max(0.0) / m.weight,
                   "thrust / weight carrying the bandit": m.t_max(0.0) / ((m.mass + BANDIT_KG) * G)}
            for d in distances:
                row[f"to {d:.0f} m [s]"] = intercept(key, d)["time_s"]
            row["hover [min]"] = endurance(m)["hover_min"]
            rows.append(row)
            del _SPRINTERS[key]
    return rows


# =============================================================================================================== GAVILAN-X
# GAVILAN-J's hunt on an X-wing: the same motors, propellers, pack, talons and guards; the wing, pylons, boom and fins
# replaced by four identical panels that are also the feet (gavilan.GavilanX).
def x_geometry(q=None) -> dict:
    p = gv.x_resolve(q)
    L = gv.GavilanX.layout(p)
    d = p["body_diameter"] / 1e3
    nose, bay, cone = p["nose_length"] / 1e3, p["bay_length"] / 1e3, (p["cone_length"] + 30.0) / 1e3
    return {"d": d, "nose": nose, "bay": bay, "cone": cone, "length": p["length"] / 1e3, "width": L["width"] / 1e3, "S": L["S_mm2"] / 1e6,
            "mac": L["mac"] / 1e3, "t_c": p["thickness"], "pod_d": p["pod_diameter"] / 1e3, "pod_l": p["pod_length"] / 1e3,
            "s_body": math.pi * d * (0.70 * nose + bay + 0.60 * cone), "plan_body": d * (0.67 * nose + bay + 0.55 * cone),
            "motor_y": p["motor_y"] / 1e3, "motor_z": p["motor_z"] / 1e3, "x_ac": L["x_ac"] / 1e3, "x_prop": L["x_prop"] / 1e3, "x_bay": L["x_bay"] / 1e3,
            "foot_half_width": L["foot_half_width"] / 1e3, "projected_span": L["projected_span"] / 1e3, "panel_radius": p["panel_radius"] / 1e3}


def x_mass_budget(c: ef.Config = CONFIG_J) -> og_mass.Budget:
    g = x_geometry()
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
        b.add(I(part.name, part.mass_g / 1e3, (0.6 * g["nose"] if part.group in ("video", "gps") else g["x_bay"] + 0.02, 0, 0), group="avionics", source=part.key))
    b.add(I("camera tilt servo", TILT_SERVO_G / 1e3, (0.03, 0, 0), group="avionics", source="hand estimate: a 9 g sub-micro servo"),
          I("power wiring", 0.012 * c.packs + 0.036, (g["x_bay"] + 0.03, 0, 0), group="propulsion", source="emerillon_flight's allowance, leads along the panels"),
          I("body skin (PETG)", g["s_body"] * ef.WALL * ef.PRINT_FACTOR * petg.density, (0.5 * (g["nose"] + g["bay"] + g["cone"]), 0, 0), group="airframe",
            source="emerillon_flight.printed_masses' rule"),
          I("four panels (printed foam)", 2.04 * g["S"] * AREAL_DENSITY, (g["x_ac"] + 0.15 * g["mac"], 0, 0), group="airframe", source="AREAL_DENSITY"),
          I("panel spars (bought carbon tube, four)", 4 * 0.022 * g["panel_radius"], (g["x_ac"], 0, 0), group="airframe",
            source="hand estimate: 22 g/m, a tip motor on each (kestrel_xwing sizes 19 g a panel for a longer one)"),
          I("motor pods (PA6-CF)", 4 * math.pi * g["pod_d"] * 0.75 * g["pod_l"] * 0.0016 * pa6.density * ef.PRINT_FACTOR / 1.3, (g["x_prop"] + 0.03, 0, 0),
            group="airframe", source="emerillon_flight.printed_masses' rule"),
          I("foot bumpers (TPU)", 0.008, (g["length"], 0, 0), group="airframe", source="emerillon_flight"),
          I("talons with their line", talons_kg(), (g["x_ac"], 0, -g["d"] / 2 - 0.03), group="payload", source="TALONS (hand estimate x 1.25)"),
          I("propeller guards", guards_kg(), (g["x_prop"], 0, 0), group="airframe", source="GUARDS (hand estimate x 1.25)"))
    return b


def x_drag_buildup(V: float = 25.0, margin: float | None = None) -> dict:
    g = x_geometry()
    margin = ef.DRAG_MARGIN if margin is None else margin

    def cf(length):
        return cf_turbulent(max(V * length / NU, 1e4))

    f = (g["nose"] + g["bay"] + g["cone"]) / g["d"]
    body = cf(g["nose"] + g["bay"] + g["cone"]) * (1 + 60 / f ** 3 + f / 400) * g["s_body"]
    panels = cf(g["mac"]) * (1 + 2 * g["t_c"] + 100 * g["t_c"] ** 4) * 2.04 * g["S"]
    fp = g["pod_l"] / g["pod_d"]
    pods = 4 * cf(g["pod_l"]) * (1 + 60 / fp ** 3 + fp / 400) * math.pi * g["pod_d"] * 0.75 * g["pod_l"]
    camera = 0.25 * math.pi * 0.016 ** 2
    clean = body + (panels + pods) * ef.INTERFERENCE + camera
    return {"body": body, "panels": panels * ef.INTERFERENCE, "pods": pods * ef.INTERFERENCE, "camera": camera, "clean": clean,
            "CdA": clean * margin + TALONS["cda_m2"] + GUARDS["cda_m2"], "margin": margin}


@functools.lru_cache(maxsize=None)
def x_wing_aero() -> dict:
    """The four panels at 45 degrees on the vortex lattice: the vertical lift slope and the induced factor on their
    total area (an X gives about half the slope of the same area laid flat: kestrel_xwing's finding)."""
    g = x_geometry()
    r = vlm(gv.x_planform(), alpha_deg=4.0, S_ref=g["S"], c_ref=g["mac"], n_span=8, n_chord=4)
    a = math.radians(4.0)
    return {"S": g["S"], "CLa": r["CL"] / a, "k": r["CDi"] / r["CL"] ** 2}


def gavilan_x() -> Machine:
    g, a = x_geometry(), x_wing_aero()
    c45 = math.cos(math.radians(45.0))
    pol = Polar(cl_alpha=a["CLa"], cd0=0.0, k=a["k"] / OSWALD_VISCOUS, cl_max=CL_MAX_3D * c45, cd90=viterna_cd90(2.0) * c45,
                source="ogma.aero.vlm on the four panels; maximum lift and broadside drag x cos 45 (kestrel_xwing's treatment)")
    t_max, power = _drive_fns(drive_map(CONFIG_J, 4), AVIONICS_W)
    return Machine("Gavilan-X (the hunt on an X-wing: four identical panels)", "tailsitter", x_mass_budget().mass, g["width"], g["length"], g["S"], pol,
                   x_drag_buildup()["CdA"], t_max, power, N_DESIGN, mk.part(CONFIG_J.battery).cells * 3.7 * mk.part(CONFIG_J.battery).ah,
                   plan_body=g["plan_body"], roll_rate=math.radians(600.0), fixed_w=AVIONICS_W, payload_kg=talons_kg(), gun="talons",
                   source="gavilan_flight (gavilan.GavilanX, ÉMERILLON's drive and catalogue)", notes={"config": CONFIG_J.label(), "motors": 4})


_SPRINTERS["gavilan_x"] = lambda: gavilan_x()

PRINTED = {   # printed part types and pieces, counted from the CAD's ``parts`` (bought: motors, electronics, pack, carbon tubes)
    "gavilan_j": {"types": ["nose", "bay with its cone", "wing half", "pylon", "motor pod", "fin"], "pieces": 1 + 1 + 2 + 2 + 4 + 4, "bought tubes": 1},
    "gavilan_x": {"types": ["nose", "bay with its cone", "panel", "motor pod"], "pieces": 1 + 1 + 4 + 4, "bought tubes": 4},
}


def j_x_table() -> list:
    """GAVILAN-J and GAVILAN-X side by side: the build, the hunt, the flying."""
    rows = []
    for key, m, geo in (("gavilan_j", gavilan_j(), geometry(gv.variant("J"))), ("gavilan_x", gavilan_x(), x_geometry())):
        en, br = endurance(m), brake(m)
        row = {"key": key, "aircraft": m.name, "mass_kg": m.mass, "width [m]": m.span, "length [m]": m.length, "feet half-width [m]": geo["foot_half_width"],
               "printed part types": len(PRINTED[key]["types"]), "printed pieces": PRINTED[key]["pieces"], "lifting area [m2]": m.S,
               "static thrust / weight": m.t_max(0.0) / m.weight, "thrust / weight carrying the bandit": m.t_max(0.0) / ((m.mass + BANDIT_KG) * G)}
        for d in (50.0, 100.0, 200.0):
            row[f"to a bandit at {d:.0f} m [s]"] = intercept(key, d)["time_s"]
        for V in (10.0, 15.0, 25.0):
            row[f"turn rate @{V:.0f} [deg/s]"] = turn(m, V)["rate_inst_deg_s"]
        row.update({"stop from 25 m/s [m]": br["distance_m"], "wing stall [m/s]": stall_speed(m), "top speed [m/s]": top_speed(m),
                    "hover [min]": en["hover_min"], "cruise [min]": en["cruise_min"], "cruise at [m/s]": en["cruise_v"], "cruise [W]": en["cruise_w"]})
        rows.append(row)
    return rows


# ============================================================================================================== GAVILAN-XP
# GAVILAN-X with a hawk's feet: two thin legs that swing out of the belly, each with a four-toed foot that snaps shut
# (gavilan.GavilanXP). What follows is the catch as a **sequence in time** — kinematics and timing, each number from the
# sprint model or from a stated hand estimate. It is not a contact simulation: nothing here knows whether the toes
# actually close on the bandit.
FEET = {
    "source": "hand estimate: nothing chosen or tested",
    "parts_g": {"two leg tubes, 6 mm carbon, 0.24 m": 14.0, "two feet: four printed toes each with steel talon tips, springs and a latch": 44.0,
                "two hip servos (metal gear, 20 g class)": 40.0, "hip brackets and bearings": 14.0, "line with an elastic section, and its latch": 20.0},
    "margin": 1.25, "cda_m2": 0.0016,
}
HIP_RATE_DEG_S = 300.0             # the legs' swing under load: a 0.10 s / 60 deg servo at half its free speed (hand estimate)
LEG_FOLDED_DEG, LEG_OUT_DEG = 180.0, 100.0
TOE_OPEN_DEG, TOE_SHUT_DEG = 80.0, 20.0
TOE_SPRING_NM = 0.15               # each toe's closing spring, mean torque over the stroke (hand estimate)
TOE_G = 6.0                        # one toe [g] (hand estimate)
TRIGGER_S = 0.010                  # from the sensor seeing the bandit to the latch letting go (hand estimate)
XP_CONTACT_SPEED = 1.5             # the closing speed at the grab [m/s]: slow enough for the toes to shut round the bandit (see catch_sequence)
LEGS_READY_S = 0.20                # the legs are out and still this long before the contact
BANDIT_GRIP_MM = 50.0              # what the foot closes on: the bandit's body across its thickness (net_drone's hull: 2 x 25 mm)
BIND_S = 0.05                      # the time the sprung legs take the shock over (hand estimate)
BANDIT_THRUST_TW = 2.0             # a bandit still fighting pulls with this many times its weight (hand estimate)


def feet_kg() -> float:
    return sum(FEET["parts_g"].values()) * FEET["margin"] / 1e3


def gavilan_xp() -> Machine:
    x = gavilan_x()
    return dataclasses.replace(x, name="Gavilan-XP (the X-wing with a hawk's feet)", mass=x.mass - talons_kg() + feet_kg(),
                               cda0=x.cda0 - TALONS["cda_m2"] + FEET["cda_m2"], payload_kg=feet_kg(), gun="feet")


_SPRINTERS["gavilan_xp"] = lambda: gavilan_xp()


def toe_snap_s(p=None) -> float:
    """The time a toe takes from spread to shut on its spring, a rod about its root: sqrt(2 theta I / torque), after
    the trigger's delay."""
    p = gv.xp_resolve(p)
    L = (p["toe_first"] + p["toe_second"]) / 1e3
    inertia = TOE_G / 1e3 * L * L / 3.0
    return TRIGGER_S + math.sqrt(2 * math.radians(TOE_OPEN_DEG - TOE_SHUT_DEG) * inertia / TOE_SPRING_NM)


def _flare_trace(m: Machine, V0: float, v_end: float, dt: float = 0.01, rho: float = RHO) -> dict:
    """``brake`` with its history: the hardest level deceleration from ``V0`` to ``v_end`` that still carries the weight."""
    lv = level(m, V0)
    theta, V, x, t = (math.radians(lv["theta_deg"]) if lv else 0.0), V0, 0.0, 0.0
    T_, X_, V_, TH = [0.0], [0.0], [V0], [theta]
    while V > v_end and t < 30.0:
        q, W = 0.5 * rho * V * V, m.weight
        best = None
        for th in m.thetas()[::2]:
            la, da = m.areas(float(th))
            s, c = math.sin(th), math.cos(th)
            T = (W - q * la) / s if s > 1e-3 else (0.0 if q * la >= W else math.inf)
            if T < -1e-6 or T > m.thrust(V, float(th)):
                continue
            dec = (q * da - T * c) / m.mass
            if best is None or dec > best[0]:
                best = (dec, float(th))
        if best is None:
            break
        theta = _slew(theta, best[1], m.slew, dt)
        la, da = m.areas(theta)
        s, c = math.sin(theta), math.cos(theta)
        T = min(max((W - q * la) / s if s > 1e-3 else 0.0, 0.0), m.thrust(V, theta))
        V -= (q * da - T * c) / m.mass * dt
        x += V * dt
        t += dt
        T_.append(t), X_.append(x), V_.append(V), TH.append(theta)
    return {"t": np.array(T_), "x": np.array(X_), "V": np.array(V_), "theta_deg": np.degrees(np.array(TH))}


def catch_sequence(distance: float = 100.0, height: float = 30.0, p=None) -> dict:
    """GAVILAN-XP's catch of a bandit hovering ``distance`` away, as a timeline from the perch: spin-up, sprint, flare
    to ``XP_CONTACT_SPEED``, the legs swung out (finished ``LEGS_READY_S`` before the contact), the toes spread, the
    snap, the bind, the descent from ``height`` with the bandit aboard. With it: the windows the catch has to hit
    (how long the bandit is inside the open foot, how far off-line it may be) and the loads. Kinematics and hand
    estimates; no contact is simulated."""
    q = gv.xp_resolve(p)
    F = gv.GavilanXP.foot_layout({**q, "leg_deg": LEG_OUT_DEG, "toe_open_deg": TOE_OPEN_DEG})
    m = gavilan_xp()
    tr = _sprint_trace(m, v_stop=60.0, dt=0.01)
    fl = _flare_trace(m, float(tr["V"][-1]), XP_CONTACT_SPEED)
    to_go = fl["x"][-1] - np.interp(tr["V"], fl["V"][::-1], fl["x"][::-1])        # the flare's length from each sprint speed
    i = min(int(np.searchsorted(tr["x"] + to_go, distance)), len(tr["x"]) - 1)
    v_peak, t_switch = float(tr["V"][i]), float(tr["t"][i])
    j = int(np.searchsorted(-fl["V"], -v_peak))                                   # where the flare's history is at that speed
    t_fl, x_fl, v_fl, th_fl = fl["t"][j:] - fl["t"][j], fl["x"][j:] - fl["x"][j], fl["V"][j:], fl["theta_deg"][j:]
    t_contact = SPINUP_S + t_switch + float(t_fl[-1])
    swing = (LEG_FOLDED_DEG - LEG_OUT_DEG) / HIP_RATE_DEG_S
    snap = toe_snap_s(q)
    depth, aperture = F["tip_depth"] / 1e3, F["aperture"] / 1e3
    window = depth / XP_CONTACT_SPEED                                              # the bandit's body between the toe tips and the ankle
    v_after = m.mass * XP_CONTACT_SPEED / (m.mass + BANDIT_KG)                    # the two together, just after the grab
    force = BANDIT_KG * v_after / BIND_S
    spare = m.t_max(0.0) - (m.mass + BANDIT_KG) * G
    events = [{"t": 0.0, "event": "trigger on the perch: motors start"},
              {"t": SPINUP_S, "event": "lift-off and tip-over: the sprint begins"},
              {"t": SPINUP_S + t_switch, "event": f"peak speed {v_peak:.0f} m/s: the flare begins, nose up, belly first"},
              {"t": t_contact - LEGS_READY_S - swing, "event": f"the legs swing out of the belly ({swing:.2f} s), the toes spread to {aperture * 1e3:.0f} mm"},
              {"t": t_contact - LEGS_READY_S, "event": "legs out and still, the camera on the bandit"},
              {"t": t_contact - snap, "event": f"the trigger: the toes are let go {snap * 1e3:.0f} ms before the contact"},
              {"t": t_contact, "event": f"contact at {XP_CONTACT_SPEED:g} m/s: the toes shut round the bandit's body"},
              {"t": t_contact + BIND_S, "event": f"the bind: the two move together at {v_after:.1f} m/s, the legs took {force:.0f} N"},
              {"t": t_contact + BIND_S + 1.0, "event": "steadied in a hang, the bandit held under the belly"},
              {"t": t_contact + BIND_S + 1.0 + height / 2.0, "event": f"set down after a descent from {height:.0f} m at 2 m/s"}]
    t_all = np.concatenate([[0.0], SPINUP_S + tr["t"][:i + 1], SPINUP_S + t_switch + t_fl[1:]])
    x_all = np.concatenate([[0.0], tr["x"][:i + 1], tr["x"][i] + x_fl[1:]])
    v_all = np.concatenate([[0.0], tr["V"][:i + 1], v_fl[1:]])
    rng = np.maximum(x_all[-1] - x_all, 0.0)
    leg = np.clip(LEG_FOLDED_DEG - (t_all - (t_contact - LEGS_READY_S - swing)) * HIP_RATE_DEG_S, LEG_OUT_DEG, LEG_FOLDED_DEG)
    toe = np.where(t_all >= t_contact - LEGS_READY_S - swing, TOE_OPEN_DEG, TOE_SHUT_DEG).astype(float)
    return {"aircraft": m.name, "mass_kg": m.mass, "distance_m": distance, "events": events, "time_to_contact_s": t_contact, "peak_m_s": v_peak,
            "flare_s": float(t_fl[-1]), "flare_m": float(x_fl[-1]), "nose_up_at_contact_deg": float(th_fl[-1]),
            "leg_swing_s": swing, "toe_snap_s": snap, "in_foot_window_s": window, "snap_fits_window": snap <= window + TRIGGER_S,
            "trigger_tolerance_s": 0.5 * window, "aperture_m": aperture, "sideways_tolerance_m": 0.5 * (aperture - BANDIT_GRIP_MM / 1e3),
            "between_feet_m": F["between_feet"] / 1e3, "speed_after_bind_m_s": v_after, "bind_force_N": force,
            "thrust_over_weight_with_bandit": m.t_max(0.0) / ((m.mass + BANDIT_KG) * G), "spare_thrust_N": spare,
            "spare_thrust_if_bandit_fights_N": spare - BANDIT_THRUST_TW * BANDIT_KG * G, "set_down_s": events[-1]["t"],
            "trace": {"t": t_all, "range_m": rng, "speed_m_s": v_all, "leg_deg": leg, "toe_deg": toe}}


# ======================================================================================================= GAVILAN-X + net
# The lightest net there is: the one-shot interceptor's 70 g cartridge (a 1.5 m net, 0.15 m mesh of 1 mm Dyneema, four
# 8 g weights) at 45 m/s, here in a pod under GAVILAN-X's belly (gavilan.X_NET): the tube forward along the body axis,
# the chamber, the burst diaphragm, the 80 mL charge vessel and a short jet pipe inside a fairing, the jet out at the
# body's end. A fouled bandit falls (gavilan_net_fall: from 20 m it hits the ground at ~16.6 m/s); what slows it is
# left for later. Masses are HAND ESTIMATES (the chamber's is net_light_design's); MARGIN on their sum.
NET_POD = {
    "source": "LIGHT_NET's parts without GAVILAN-H's long jet pipe; the pod fairing by hand",
    "parts_g": {"carbon tube 0.25 m, 32 mm bore": 63.0, "chamber and rear venturi (6061)": 140.0, "burst-diaphragm holder and puncture solenoid": 90.0,
                "charge vessel 80 mL, carbon": 80.0, "jet pipe 0.15 m, carbon": 30.0, "pod fairing and its strut (printed foam)": 45.0,
                "trigger electronics and wiring": 30.0, "net cartridge": 70.0},
    "margin": 1.25, "cda_m2": 0.0016,
}
NET_FLIGHT_S = {3.0: 0.09, 4.0: 0.12, 5.0: 0.18, 6.0: 0.24, 8.0: 0.39}   # the cartridge to the bandit (net_light_quad.log, fixed launcher)
NET_RANGE = (3.0, 8.0)                                                    # every shot captured in that test (5/5 at each range)
STOP_MARGIN = 1.5                                                         # the hunter stops this short of the bandit: half the opened net and a margin


def net_pod_kg() -> float:
    return sum(NET_POD["parts_g"].values()) * NET_POD["margin"] / 1e3


def gavilan_x_net() -> Machine:
    x = gavilan_x()
    return dataclasses.replace(x, name="Gavilan-X with the one-shot net", mass=x.mass - talons_kg() + net_pod_kg(),
                               cda0=x.cda0 - TALONS["cda_m2"] + NET_POD["cda_m2"], payload_kg=net_pod_kg(), gun="fixed")


_SPRINTERS["gavilan_x_net"] = lambda: gavilan_x_net()


def net_shot(v: float, rng: float, m: Machine | None = None) -> dict:
    """GAVILAN-X with the net fires from ``rng`` while closing at ``v``, its nose on the bandit, then flares: does it
    stop short of the bandit (and its net)? The net's flight time from the launcher test (the aircraft's own speed
    added to the net's shortens it a little: not counted, the safe side)."""
    m = m or gavilan_x_net()
    t_net = float(np.interp(rng, list(NET_FLIGHT_S), list(NET_FLIGHT_S.values())))
    react = 0.05                                                            # from the shot to the flare starting (hand estimate)
    b = brake(m, max(v, 0.31), dt=0.01) if v > 0.3 else {"distance_m": 0.0, "time_s": 0.0}
    travel = v * react + b["distance_m"]
    return {"v": v, "range": rng, "net_flight_s": t_net, "stop_m": travel, "room_m": rng - STOP_MARGIN - travel, "safe": travel <= rng - STOP_MARGIN}


def fastest_safe_shot(rng: float, m: Machine | None = None) -> float:
    """The highest closing speed at which the shot from ``rng`` still leaves the hunter stopped short of the bandit."""
    m = m or gavilan_x_net()
    lo, hi = 0.0, 25.0
    for _ in range(16):
        v = 0.5 * (lo + hi)
        lo, hi = (v, hi) if net_shot(v, rng, m)["safe"] else (lo, v)
    return lo


def x_net_attack(distance: float, rng: float = 6.0) -> dict:
    """From the perch to the shot at a bandit ``distance`` away: sprint, flare to the fastest safe shooting speed at
    ``rng``, fire; the net's flight; when the bandit is fouled."""
    m = gavilan_x_net()
    v = fastest_safe_shot(rng, m)
    key = f"_xnet_{rng:g}"
    _SPRINTERS[key] = gavilan_x_net
    global CONTACT_SPEED
    keep = CONTACT_SPEED
    CONTACT_SPEED = max(v, 0.5)
    try:
        r = intercept(key, max(distance - rng, 1.0))
    finally:
        CONTACT_SPEED = keep
    t_net = float(np.interp(rng, list(NET_FLIGHT_S), list(NET_FLIGHT_S.values())))
    return {"distance_m": distance, "fire_range_m": rng, "fire_speed_m_s": v, "time_to_shot_s": r["time_s"], "time_to_foul_s": r["time_s"] + t_net,
            "peak_m_s": r["peak_m_s"]}


def x_variants_table() -> list:
    """GAVILAN-X with its three catchers: the four-legged net (talons), the hawk's feet (XP), the one-shot net."""
    rows = []
    for key, m, catcher in (("x_net_shot", gavilan_x_net(), "net"), ("x_talons", gavilan_x(), "talons"), ("x_feet", gavilan_xp(), "feet")):
        en, br = endurance(m), brake(m)
        row = {"key": key, "aircraft": m.name, "mass_kg": m.mass, "catcher_kg": m.payload_kg, "static thrust / weight": m.t_max(0.0) / m.weight,
               "turn rate @15 [deg/s]": turn(m, 15.0)["rate_inst_deg_s"], "stop from 25 m/s [m]": br["distance_m"],
               "top speed [m/s]": top_speed(m), "hover [min]": en["hover_min"], "cruise [min]": en["cruise_min"]}
        if catcher == "net":
            for d in (50.0, 100.0, 200.0):
                row[f"perch to the bandit fouled, {d:.0f} m [s]"] = x_net_attack(d)["time_to_foul_s"]
            row.update({"reach to the bandit [m]": "3 to 8 (shot)", "aiming": "the net opens to 1.6-2.0 m; 5/5 at every range from 3 to 8 m in the launcher test",
                        "shots": 1, "the bandit afterwards": "falls (about 17 m/s from 20 m): not caught alive yet"})
        else:
            key2 = "gavilan_x" if catcher == "talons" else "_xp_grab"
            _SPRINTERS["_xp_grab"] = gavilan_xp
            global CONTACT_SPEED
            keep = CONTACT_SPEED
            CONTACT_SPEED = CONTACT_SPEED if catcher == "talons" else XP_CONTACT_SPEED          # the feet grab at 1.5 m/s
            try:
                for d in (50.0, 100.0, 200.0):
                    row[f"perch to the bandit fouled, {d:.0f} m [s]"] = intercept(key2, d)["time_s"]
            finally:
                CONTACT_SPEED = keep
            row.update({"reach to the bandit [m]": "0 (contact)", "aiming": "±0.22 m (net on legs)" if catcher == "talons" else "±0.06 m (feet)",
                        "shots": "repeatable", "the bandit afterwards": "held, carried down"})
        rows.append(row)
    return rows


# ============================================================================================ GAVILAN: the chosen design
# The sparrowhawk: GAVILAN-X (the four-panel X-wing tailsitter on one pack) waiting on its charging perch as a camera,
# launching at a bandit that comes near, a short pursuit, and a net thrown at 30 m/s by the opposed latex slingshot
# (gavilan_slingshot) from 3 to 5 m. The capture figures are labs/aero/exp/gavilan_slow_net_results.json (the light
# net at 30 m/s from a fixed launcher: 4/4 at 3, 4 and 5 m against a hovering quadcopter, 4/4, 4/4 and 3/4 crossing at
# 5 m/s); the flight times are the same net's free flight at 30 m/s (net_drone.calibrate).
SLING_SPEED = 30.0
SLING_NET_FLIGHT_S = {2.0: 0.08, 3.0: 0.13, 4.0: 0.18, 6.0: 0.35}
SLING_RANGE = (3.0, 5.0)
SLING_CDA = 0.0018                  # the tube, its rings and bands under the belly, broadside-free along the flow (hand estimate) [m2]


def sling_kg() -> float:
    import gavilan_slingshot as gs
    return gs.launcher(SLING_SPEED, "latex bands", True)["launcher_kg"]


def gavilan_mass_budget() -> og_mass.Budget:
    """GAVILAN as chosen: GAVILAN-X's parts without the talons, with the slingshot under the belly and the charging
    parts aboard (gavilan_perch.ONBOARD_G), each at its station."""
    import gavilan_perch as gpe
    g = x_geometry()
    b = og_mass.Budget([i for i in x_mass_budget().items if not i.name.startswith("talons")])
    b.add(og_mass.Item("net slingshot (opposed, latex, 30 m/s; two nets)", sling_kg(), (0.24, 0, -0.064), group="payload",
                       source="gavilan_slingshot.launcher (hand estimate x 1.25)"),
          og_mass.Item("charging: foot soles, wiring, diode bridge, balancer", gpe.onboard_kg(), (g["x_ac"], 0, 0), group="avionics",
                       source="gavilan_perch.ONBOARD_G (hand estimate x 1.25)"))
    return b


def gavilan() -> Machine:
    """The chosen GAVILAN as a ``Machine``."""
    x = gavilan_x()
    return dataclasses.replace(x, name="Gavilan (X-wing, net slingshot, charging perch)", mass=gavilan_mass_budget().mass,
                               cda0=x.cda0 - TALONS["cda_m2"] + SLING_CDA, payload_kg=sling_kg(), gun="slingshot")


_SPRINTERS["gavilan"] = lambda: gavilan()


def sling_attack(distance: float, rng: float = 4.0) -> dict:
    """From the perch to the bandit fouled, ``distance`` away and hovering: sprint, flare to the fastest closing speed
    that still stops ``STOP_MARGIN`` short of the bandit after the shot from ``rng``, throw, the net's flight. The
    net leaves at the slingshot's speed plus the closing speed."""
    m = gavilan()
    v = fastest_safe_shot(rng, m)
    key = f"_sling_{rng:g}"
    _SPRINTERS[key] = gavilan
    global CONTACT_SPEED
    keep = CONTACT_SPEED
    CONTACT_SPEED = max(v, 0.5)
    try:
        r = intercept(key, max(distance - rng, 1.0))
    finally:
        CONTACT_SPEED = keep
    t_net = float(np.interp(rng, list(SLING_NET_FLIGHT_S), list(SLING_NET_FLIGHT_S.values()))) * SLING_SPEED / (SLING_SPEED + v)
    return {"distance_m": distance, "fire_range_m": rng, "closing_m_s": v, "net_relative_m_s": SLING_SPEED + v,
            "time_to_shot_s": r["time_s"], "time_to_foul_s": r["time_s"] + t_net, "peak_m_s": r["peak_m_s"]}


def flight_card() -> dict:
    """The plain numbers a game reads (Armageddon): GAVILAN as chosen, from the models of this module — each with
    where it comes from. No Vegeta needed to read it."""
    import gavilan_perch as gpe
    m = gavilan()
    p = gv.x_resolve(gv.X_SLING)
    en, br, cl = endurance(m), brake(m), climb(m)
    acc = accelerate(m, 25.0)
    V_wing = 1.2 * stall_speed(m)
    to_wing = accelerate(m, V_wing)
    to_hover = brake(m, V_wing)
    usable = USABLE * m.battery_wh
    p_hard = m.power(m.t_max(0.0), 0.0)
    pb = gpe.budget()
    turns = []
    for V in (5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0):
        t = turn(m, V)
        turns.append({"speed_m_s": V, "turn_rate_deg_s": round(t["rate_inst_deg_s"], 1), "load_factor": round(t["n_inst"], 2),
                      "turn_radius_m": round(t["radius_inst_m"], 2), "sustained_turn_rate_deg_s": round(t["rate_sus_deg_s"], 1)})
    r = lambda x, k=2: round(float(x), k)
    return {
        "unit": "Gavilan", "kind": "sparrowhawk interceptor: X-wing tailsitter, waits on a charging perch as a camera, short pursuit, throws a net",
        "units": "SI: kg, m, s, m/s, deg/s, W, Wh, min",
        "source": "vegeta labs/aero/designs/gavilan_flight.py (reduced point-mass models; hand estimates where stated); notebook 38",
        "mass_kg": r(m.mass, 3), "span_m": r(m.span, 3), "length_m": r(p["length"] / 1e3, 3), "hover": True,
        "min_speed_m_s": 0.0, "max_speed_m_s": r(top_speed(m), 1), "wing_borne_above_m_s": r(V_wing, 1),
        "max_acceleration_g": r(acc["mean_g"], 2), "hover_to_25_m_s": {"time_s": r(acc["time_s"]), "distance_m": r(acc["distance_m"], 1)},
        "max_braking_g": r(br["peak_g"], 2), "25_m_s_to_hover": {"time_s": r(br["time_s"]), "distance_m": r(br["distance_m"], 1)},
        "climb_rate_m_s": r(cl["rate_m_s"], 1), "ground_to_150_m_s": r(cl["time_s"]),
        "turns": turns, "structural_load_factor": N_DESIGN,
        "roll_rate_deg_s": r(math.degrees(m.roll_rate), 0), "pitch_rate_deg_s": r(math.degrees(m.slew), 0),
        "transition_hover_to_wing_s": r(to_wing["time_s"]), "transition_wing_to_hover_s": r(to_hover["time_s"]),
        "endurance_min": {"hover": r(en["hover_min"], 1), "cruise": r(en["cruise_min"], 1), "cruise_speed_m_s": r(en["cruise_v"], 1),
                          "manoeuvring_at_2_g": r(usable / m.power(2.0 * m.weight, 0.0) * 60.0, 1),
                          "full_power": r(usable / p_hard * 60.0, 1), "watch_on_its_pack_h": r(pb["watch_off_the_perch_h"], 2),
                          "watch_on_the_perch": "unlimited (charged through its feet)"},
        "perch": {"recharge_after_a_hunt_min": r(pb["recharge_after_a_100_m_hunt_min"], 0), "full_charge_min": r(pb["full_charge_min"], 0),
                  "landing_tolerance_m": r(gpe.Perch.layout(gpe.resolve())["landing_tolerance"] / 1e3, 3)},
        "net": {"launcher": "opposed latex slingshot: one net forward, one aft (no recoil)", "nets": 2, "throw_speed_m_s": SLING_SPEED,
                "range_m": list(SLING_RANGE), "opening_m": 1.8, "reload": "by hand, on the ground",
                "capture_tests": "4/4 at 3, 4, 5 m hovering; 4/4, 4/4, 3/4 crossing at 5 m/s (fixed launcher, MuJoCo)",
                "after_the_catch": "the bandit falls (about 17 m/s from 20 m): a parachute is left for later"},
        "perch_to_fouled_s": {f"{d:.0f}_m": r(sling_attack(d)["time_to_foul_s"]) for d in (50.0, 100.0, 200.0)},
    }


def design_record() -> dict:
    """``data/gavilan_design.json``: the CAD's parameters, the mass table, the flight card."""
    pr = gavilan_mass_budget()
    props = pr.properties()
    return {"source": "gavilan.GavilanX (X_SLING), gavilan_flight.gavilan_mass_budget(), flight_card(); notebook 38",
            "parameters": {k: v for k, v in gv.x_resolve(gv.X_SLING).items()},
            "mass_table": [{"item": i.name, "kg": round(i.kg, 4), "group": i.group, "x_m": round(i.pos[0], 3), "source": i.source} for i in pr.items],
            "mass_kg": round(pr.mass, 4), "by_group_kg": {k: round(v, 4) for k, v in pr.by_group().items()},
            "centre_of_gravity_m": [round(float(c), 4) for c in props.com], "flight_card": flight_card()}
