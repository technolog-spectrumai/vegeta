"""DRAFT, not yet running (the Chiron robot definition fails validation; the autopilot is untested). Mini raptor races in MuJoCo (notebook 50 ``50_mini_races``): Apus Mini, Alcotan, Gavilan and the 5-inch racing
quad flown round three courses — a three-floor hangar, an industrial zone, a farm — by the same autopilot, through
Chiron's lab (``vegeta.chiron``: MuJoCo with scene hooks).

Every aircraft is a rigid body whose thrust axis is its body x (the tailsitters' nose, the quad's "up"), standing on
its tail at the start. One scene hook, ``Airframe``, puts on it every control step: the motors' thrusts at their
stations (first-order lag, the drive table's limit at the axial inflow, the power from the same table), the wing's
lift and drag from the design's polar (linear to the stall, flat-plate beyond — ``mini_raptor``'s numbers), the
pod's and motors' drag, the control surfaces' moments within the authority the slipstream and the airspeed give
them (``mini_raptor.elevon_pitch_authority``'s model), and aerodynamic damping. The quad has no wing and no
surfaces; its body is the frame seen from below and edge-on.

One autopilot, ``Racer``, flies them all: a reference point runs ahead along the course's polyline at a speed
schedule the aircraft's own limits allow (its thrust-vectoring lateral acceleration at every corner's radius, its
braking, its top speed), a position loop asks for an acceleration, the thrust vector is what is left once the wing's
estimated force is counted, a geometric attitude loop turns the thrust axis onto it and keeps the wing's lift plane
towards the turn, and a mixer shares the moments between differential thrust and the surfaces. Touching anything
— a wall, a beam, a tree, the ground — ends the race as a crash. Nothing here is a validated flight controller: the
races rank the airframes under one honest pilot, not the pilots.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import mini_raptor as mr
from vegeta import chiron as ch

G, RHO = 9.81, 1.225
LAB_OPTIONS = dict(timestep=0.002, control_dt=0.004, log_dt=0.05, cone="pyramidal")
UP_QUAT = (math.cos(-math.pi / 4), 0.0, math.sin(-math.pi / 4), 0.0)        # body x → world z (standing on the tail)


# ================================================================================================= the aircraft
@dataclass
class Craft:
    """One aircraft for the race: its masses, its motor stations in the body frame (x the thrust axis, z the wing's
    lift), its drive table, its polar and drag, its surfaces' authority."""
    name: str
    kind: str                       # tailsitter | quad
    m: float
    stations: np.ndarray            # (n, 3) [m] from the CG
    spin: np.ndarray                # +1 / -1 per motor
    drive: object                   # mini_raptor DriveTable
    kappa: float                    # torque per newton of thrust [m]
    S: float                        # the lifting area on which the polar is written [m²]
    cl_alpha: float                 # per rad, the vertical slope on S
    cl_max: float
    alpha0: float
    cd0: float                      # on S (0 lift area: use CdA_axial / CdA_side instead)
    k: float
    cd90: float
    CdA_axial: float                # the body end-on (along x) [m²]
    CdA_side: float                 # the body across [m²]
    span: float
    chord: float
    M_pitch_coeff: float            # surface pitch moment per Pa of effective dynamic pressure [N·m/Pa]
    M_roll_coeff: float
    omega_att: float                # the attitude loop's bandwidth [rad/s]
    colour: str
    geometry: dict = field(default_factory=dict)
    spec: dict = field(default_factory=dict)

    @property
    def n_motors(self) -> int:
        return len(self.stations)

    @property
    def T_max_static(self) -> float:
        return self.n_motors * float(self.drive.T_max[0])

    def a_lateral(self) -> float:
        """The lateral acceleration the thrust vector can hold in a corner while carrying the weight (70 % of the
        thrust, the tilt limit of a pilot)."""
        tw = 0.7 * self.T_max_static / (self.m * G)
        return G * math.sqrt(max(tw * tw - 1.0, 0.0))

    def v_top(self) -> float:
        return float(self.spec.get("sustained top speed [m/s]", 30.0))

    def hover_pitch_time(self) -> float:
        return float(self.spec.get("hover pitch-over 90° [s]", 0.2))


def craft(variant: str) -> Craft:
    """A mini raptor from ``mini_raptor.performance``."""
    perf = mr.performance(variant)
    L, s, ci, tab = perf["L"], perf["summary"], perf["cg"], perf["drive"]
    m = s["mass [kg]"]
    x_cg = ci["x_cg"]
    if L["layout"] == "biplane":
        st = [(L["prop_x"] - x_cg, sy * L["motor_y"], sz * L["motor_z"]) for sy, sz in ((-1, 1), (1, 1), (-1, -1), (1, -1))]
    elif L["layout"] == "cruciform":
        st = [(L["prop_x"] - x_cg, -L["arm"], 0.0), (L["prop_x"] - x_cg, L["arm"], 0.0), (L["prop_x"] - x_cg, 0.0, L["arm"]), (L["prop_x"] - x_cg, 0.0, -L["arm"])]
    elif L["layout"] == "xwing":
        st = [(L["prop_x"] - x_cg, sy * L["motor_y"], sz * L["motor_z"]) for sy, sz in ((-1, 1), (1, 1), (-1, -1), (1, -1))]
    else:
        st = [(L["prop_x"] - x_cg, y, 0.0) for y in L["stations_y"]]
    stations = np.array(st)
    # the body frame here has x forward (the nose), the layout's x runs aft from the nose: flip it
    stations[:, 0] = -stations[:, 0]
    spin = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(len(st))])
    AR = L["AR"]
    phi = L.get("phi", 0.0)
    cl_alpha = 2 * math.pi * AR / (AR + 2) * math.cos(phi) ** 2
    d = mr.drag_buildup(L)
    k = mr.induced_factor(L)["k"]
    W = m * G
    v_i = math.sqrt(W / len(st) / (2 * RHO * math.pi * mr.R_PROP ** 2))
    e = mr.elevon_pitch_authority(L, W, ci["Iyy"])
    q_hover = 0.5 * RHO * (1.4 * v_i) ** 2
    M_pitch = e["elevon pitch moment in the hover [N·m]"] / q_hover               # per Pa of effective q
    M_roll = M_pitch * 0.5 * (L["span"] / L["chord"]) * 0.3                       # the ailerons' arm against the elevons'
    strong = L["layout"] in ("biplane", "cruciform", "xwing")
    return Craft(variant, "tailsitter", m, stations, spin, tab, float(tab.kap(0.0)), L["S_ref"], cl_alpha, L["cl_max"], -math.radians(2.0) * L["camber"] / 0.02 if L["camber"] else 0.0,
                 d["Cd0"], k, mr.kf.viterna_cd90(AR) if hasattr(mr, "kf") else 1.2 + 0.2 / (1 + AR), d["CdA_m2"], 1.1 * L["pod_width"] * L["pod_length"] + 0.3 * L["S_ref"] * 0.1,
                 L["span"], L["chord"], M_pitch, M_roll, 12.0 if strong else 7.0, L["colour"],
                 geometry={"layout": L["layout"], "x_feet": L["x_feet"] - x_cg, "pod_length": L["pod_length"], "pod_width": L["pod_width"], "x_le": L["x_le"] - x_cg, "gap": L.get("gap", 0.0),
                           "arm": L.get("arm", 0.0), "fin_chord": L.get("fin_chord", 0.0), "x_nose": -x_cg, "motor_y": L["motor_y"], "motor_z": L["motor_z"]},
                 spec=s)


def quad_craft() -> Craft:
    """The racing quad in the same frame: thrust along body x, the four motors on the 155 mm square in the y–z plane,
    no wing; its body seen end-on (from below) and across."""
    base = mr.quad()
    s, tab, ci = base["summary"], base["drive"], base["cg"]
    m = s["mass [kg]"]
    h = 0.0775
    stations = np.array([(0.0, sy * h, sz * h) for sy, sz in ((1, 1), (-1, 1), (-1, -1), (1, -1))])
    spin = np.array([1.0, -1.0, 1.0, -1.0])
    return Craft("racing quad", "quad", m, stations, spin, tab, float(tab.kap(0.0)), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.2, 2.2 * base["CdA_m2"], base["CdA_m2"], 0.28, 0.0, 0.0, 0.0, 16.0, "#555555",
                 geometry={"layout": "quad", "x_feet": 0.04, "pod_length": 0.0, "pod_width": 0.0, "x_le": 0.0, "gap": 0.0, "arm": 0.0, "fin_chord": 0.0, "x_nose": 0.0, "motor_y": h, "motor_z": h}, spec=s)


def robot(c: Craft) -> ch.Robot:
    """The craft as a Chiron robot: one free body (role 'body': touching anything is a crash) with the pod, the
    surfaces, the motors and the propeller discs as colliding shapes, four tail feet."""
    g = c.geometry
    dark, skin, disc = (0.15, 0.17, 0.2, 1.0), tuple(int(c.colour.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)) + (1.0,), (0.6, 0.7, 0.8, 0.35)
    geoms, masses = [], []
    m_motor = (mr.STACK["motor"]["mass_kg"] + 0.004 + 0.006)
    m_struct_sys = c.m - c.n_motors * m_motor
    if c.kind == "quad":
        geoms.append(ch.Geom("frame", "box", (0.012, 0.11, 0.11), pos=(0.0, 0.0, 0.0), mass=m_struct_sys, role="body", rgba=dark))
    else:
        Lp, w = g["pod_length"], g["pod_width"]
        x_mid = g["x_nose"] * -1 - Lp / 2          # the pod from the nose (x = -x_cg ... wait: x_nose is the nose position from the CG, forward positive)
        x_mid = g["x_nose"] + 0.0 - Lp / 2 if False else (-g["x_nose"]) * 0 + (g["x_nose"] * -1) * 0  # placeholder, set below
        x_nose = -g["x_nose"]                       # layout x runs aft from the nose; the CG sits x_cg behind the nose, so the nose is +x_cg ahead
        geoms.append(ch.Geom("pod", "capsule", (w / 2,), fromto=(x_nose - Lp + w / 2, 0, 0, x_nose - w / 2, 0, 0), mass=0.55 * m_struct_sys, role="body", rgba=dark))
        x_le = x_nose - (g["x_le"] + g["x_nose"]) if False else x_nose - (g["x_le"] - (-g["x_nose"]) ) * 0  # set below
        x_le_world = x_nose - ( (g["x_le"]) + (-x_nose) ) if False else None
        # the wing: its leading edge x_le (from the nose) → body x = x_nose − x_le_from_nose; g["x_le"] holds x_le − x_cg, and x_nose = x_cg
        xle = -g["x_le"]
        c_half = c.chord / 2
        xc = xle - c_half
        t_half = max(0.05 * c.chord, 0.004)
        m_wing = 0.45 * m_struct_sys
        if g["layout"] == "biplane":
            for i, z in enumerate((g["gap"] / 2, -g["gap"] / 2)):
                geoms.append(ch.Geom(f"wing{i}", "box", (c_half, c.span / 2, t_half), pos=(xc, 0.0, z), mass=m_wing / 2, role="body", rgba=skin))
            for i, y in enumerate((g["motor_y"], -g["motor_y"])):
                geoms.append(ch.Geom(f"post{i}", "capsule", (0.004,), fromto=(xc, y, -g["gap"] / 2, xc, y, g["gap"] / 2), mass=0.005, role="body", rgba=dark))
        elif g["layout"] == "cruciform":
            geoms.append(ch.Geom("wing", "box", (c_half, c.span / 2, t_half), pos=(xc, 0.0, 0.0), mass=m_wing * 0.7, role="body", rgba=skin))
            geoms.append(ch.Geom("fins", "box", (g["fin_chord"] / 2, t_half, g["arm"] + 0.01), pos=(xc, 0.0, 0.0), mass=m_wing * 0.3, role="body", rgba=skin))
        elif g["layout"] == "xwing":
            for i, (sy, sz) in enumerate(((-1, 1), (1, 1), (-1, -1), (1, -1))):
                ang = math.atan2(sz * g["motor_z"], sy * g["motor_y"])
                r1 = math.hypot(g["motor_y"], g["motor_z"]) + 0.08
                geoms.append(ch.Geom(f"panel{i}", "box", (c_half, r1 / 2, t_half), pos=(xc, 0.5 * r1 * math.cos(ang), 0.5 * r1 * math.sin(ang)),
                                     quat=(math.cos(ang / 2), math.sin(ang / 2), 0.0, 0.0), mass=m_wing / 4, role="body", rgba=skin))
        else:
            geoms.append(ch.Geom("wing", "box", (c_half, c.span / 2, t_half), pos=(xc, 0.0, 0.0), mass=m_wing, role="body", rgba=skin))
    for i, (x, y, z) in enumerate(c.stations):
        geoms.append(ch.Geom(f"motor{i}", "cylinder", (0.014, 0.016), pos=(x - 0.02, y, z), quat=(math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0), mass=m_motor, role="body", rgba=dark))
        geoms.append(ch.Geom(f"disc{i}", "cylinder", (mr.R_PROP, 0.001), pos=(x, y, z), quat=(math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0), mass=None, role="body", rgba=disc))
    feet = []
    xf = -g["x_feet"] if c.kind != "quad" else -0.04
    for i, (y, z) in enumerate(((0.08, 0.08), (-0.08, 0.08), (-0.08, -0.08), (0.08, -0.08)) if c.kind == "quad" else [(st[1], st[2]) for st in c.stations]):
        geoms.append(ch.Geom(f"foot{i}", "sphere", (0.006,), pos=(xf, y, z), mass=0.002, role="foot", friction=(0.8, 0.005, 0.0001), rgba=dark))
        feet.append(ch.FootSpec(f"foot{i}", f"foot{i}", [], "tail"))
    root = ch.Link("craft", geoms=geoms, masses=masses, log=True)
    rb = ch.Robot(c.name, root=root, feet=feet, notes="a mini raptor for the races: one body, the forces from the Airframe hook")
    rb.validate()
    return rb


# ================================================================================================= the forces
class Airframe:
    """Scene hook: the motors' thrusts (commanded by the Racer, lagged, limited at the inflow), the wing's lift and
    drag, the body's drag, the surfaces' moments, damping; the power and the energy; the history for the plots."""

    def __init__(self, c: Craft, *, spin_up_s: float = 0.04, log_every: int = 5):
        self.c, self.spin_up_s, self.log_every = c, spin_up_s, log_every

    def reset(self, lab):
        self.root = lab._body_id("craft")
        self.qv = int(lab.model.jnt_dofadr[lab._root_jnt])
        n = self.c.n_motors
        self.command = np.full(n, 0.0)
        self.surface = np.zeros(2)                      # pitch (about y), roll (about x) moment commands [N·m]
        self.thrust = np.zeros(n)
        self.energy_J, self.n_calls, self.history = 0.0, 0, []
        self.alpha_lag = 1.0 - math.exp(-lab.control_dt / self.spin_up_s)
        self.state = {}

    def aero(self, vb: np.ndarray, T_total: float) -> tuple:
        """Body-frame aerodynamic force and the effective dynamic pressure at the surfaces, for an airspeed ``vb``."""
        c = self.c
        V = float(np.linalg.norm(vb))
        q = 0.5 * RHO * V * V
        F = np.zeros(3)
        if V > 0.3:
            alpha = math.atan2(-vb[2], vb[0])
            cosa = vb[0] / V
            a_tot = math.acos(max(-1.0, min(1.0, cosa)))
            if c.S > 0:
                a_s = c.alpha0 + c.cl_max / c.cl_alpha
                if abs(alpha - c.alpha0) <= a_s - c.alpha0:
                    CL = c.cl_alpha * (alpha - c.alpha0)
                    CD = c.cd0 + c.k * CL * CL
                else:
                    CL = c.cd90 * math.sin(alpha) * math.cos(alpha) * 1.0
                    CD = c.cd0 + c.cd90 * math.sin(alpha) ** 2
                lift_dir = np.array([math.sin(alpha), 0.0, math.cos(alpha)])
                F += q * c.S * CL * lift_dir - q * c.S * CD * vb / V
                # the body across the flow (sideslip): the pod and the fins
                F += -0.5 * RHO * c.CdA_side * abs(vb[1]) * np.array([0.0, vb[1], 0.0])
            else:
                CdA = c.CdA_axial * cosa * cosa + c.CdA_side * (1 - cosa * cosa)
                F += -0.5 * RHO * CdA * V * vb
        v_i = math.sqrt(max(T_total, 0.0) / c.n_motors / (2 * RHO * math.pi * mr.R_PROP ** 2))
        q_eff = max(q, 0.5 * RHO * (1.4 * v_i) ** 2)
        return F, q_eff

    def __call__(self, lab):
        d, c = lab.data, self.c
        dt = lab.control_dt
        R = d.xmat[self.root].reshape(3, 3)
        v = np.array(d.qvel[self.qv:self.qv + 3])
        omega_b = np.array(d.qvel[self.qv + 3:self.qv + 6])            # MuJoCo's free-joint angular velocity is in the body frame
        vb = R.T @ v
        v_ax = max(float(vb[0]), 0.0)
        T_lim = float(c.drive.t_max(v_ax))
        cmd = np.clip(self.command, 0.0, T_lim)
        self.thrust += (cmd - self.thrust) * self.alpha_lag
        T = self.thrust
        xb = R[:, 0]
        torque_b = np.zeros(3)
        for i, st in enumerate(c.stations):
            torque_b += np.cross(st, np.array([T[i], 0.0, 0.0])) + np.array([c.spin[i] * c.kappa * T[i], 0.0, 0.0])
        F_aero_b, q_eff = self.aero(vb, float(T.sum()))
        M_lim = np.array([c.M_roll_coeff * q_eff, c.M_pitch_coeff * q_eff])
        surf = np.clip(self.surface, -M_lim, M_lim) if c.M_pitch_coeff > 0 else np.zeros(2)
        torque_b += np.array([surf[0], surf[1], 0.0])
        V = float(np.linalg.norm(vb))
        damp = 0.5 * RHO * max(V, 1.0) * np.array([c.S * c.span ** 2 * 0.05 + 0.002, c.S * c.chord ** 2 * 1.0 + c.CdA_axial * 0.3 ** 2 * 0.5, c.S * c.span ** 2 * 0.02 + c.CdA_axial * 0.3 ** 2 * 0.5])
        torque_b += -damp * omega_b
        force = R @ (np.array([T.sum(), 0.0, 0.0]) + F_aero_b)
        lab.body_force("craft", force, R @ torque_b)
        P = sum(c.drive.power(float(t), v_ax) for t in T) + mr.FPV_W
        self.energy_J += P * dt
        self.state = {"V": V, "T": float(T.sum()), "P": P, "q_eff": q_eff, "F_aero_b": F_aero_b, "surf": surf, "M_lim": M_lim}
        if self.n_calls % self.log_every == 0:
            p = np.array(d.xpos[self.root])
            self.history.append([lab.time, *p, *v, float(T.sum()), P, V, math.degrees(math.acos(max(-1, min(1, xb[2]))))])
        self.n_calls += 1

    @property
    def energy_Wh(self) -> float:
        return self.energy_J / 3600.0


# ================================================================================================= the course
@dataclass
class Gate:
    p: tuple                        # the point to fly through [m]
    r: float = 2.0                  # the corner radius the opening allows [m]
    name: str = ""


@dataclass
class Course:
    name: str
    props: list
    gates: list                     # Gate, in order; the first is the start point above the ground
    extent: tuple                   # x0, x1, y0, y1 of the scene [m]
    description: str = ""
    boxes: list = field(default_factory=list)     # (centre, half) for the drawings

    @property
    def points(self) -> np.ndarray:
        return np.array([g.p for g in self.gates], dtype=float)

    def length(self) -> float:
        P = self.points
        return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())


def _box(name, centre, half, rgba=(0.6, 0.6, 0.62, 1.0)):
    return ch.Prop(ch.Link(name, pos=tuple(centre), geoms=[ch.Geom(f"{name}_g", "box", tuple(half), rgba=rgba)]))


def _cyl(name, centre, r, half_h, rgba=(0.4, 0.3, 0.2, 1.0)):
    return ch.Prop(ch.Link(name, pos=tuple(centre), geoms=[ch.Geom(f"{name}_g", "cylinder", (r, half_h), rgba=rgba)]))


def _sphere(name, centre, r, rgba=(0.2, 0.5, 0.2, 1.0)):
    return ch.Prop(ch.Link(name, pos=tuple(centre), geoms=[ch.Geom(f"{name}_g", "sphere", (r,), rgba=rgba)]))


def hangar() -> Course:
    """A hangar 40 × 18 m, three floors 4 m apart: ground floor, two mezzanine floors with a stair opening and a
    window-sized hole each, a straight staircase, open windows in the long walls, roof beams under the roof. The
    course: in by a ground-floor door, up the stair opening to the first floor, through the hole to the second,
    along under the beams, down the far opening to the ground, out by a window."""
    props, boxes = [], []
    Lx, Ly, H = 40.0, 18.0, 12.0
    wall = 0.15
    grey, floorc, beam = (0.62, 0.62, 0.66, 1.0), (0.5, 0.5, 0.55, 1.0), (0.45, 0.3, 0.2, 1.0)

    def add(name, centre, half, rgba=grey):
        props.append(_box(name, centre, half, rgba)); boxes.append((tuple(centre), tuple(half)))
    # long walls with windows (1.2 × 1.2 m at 2 m height, every 8 m) and a door in the near end wall
    for side, y in (("s", -Ly / 2), ("n", Ly / 2)):
        add(f"wall_{side}_low", (0.0, y, 0.7), (Lx / 2, wall / 2, 0.7))
        add(f"wall_{side}_high", (0.0, y, (H + 2.6) / 2), (Lx / 2, wall / 2, (H - 2.6) / 2))
        xs = np.arange(-Lx / 2, Lx / 2 + 1e-6, 8.0)
        for i, x0 in enumerate(xs[:-1]):
            x1 = xs[i + 1]
            add(f"wall_{side}_pier{i}", ((x0 + x1 - 1.2) / 2 - 0.0, y, 2.0), ((x1 - x0 - 1.2) / 2, wall / 2, 0.6))       # the window is the last 1.2 m of each bay
    add("wall_w_a", (-Lx / 2, -Ly / 4 - 1.0, H / 2), (wall / 2, Ly / 4 - 1.0, H / 2))                                   # a 4 m wide, 3 m tall door in the middle of the west wall
    add("wall_w_b", (-Lx / 2, Ly / 4 + 1.0, H / 2), (wall / 2, Ly / 4 - 1.0, H / 2))
    add("wall_w_top", (-Lx / 2, 0.0, (H + 3.0) / 2), (wall / 2, 2.0, (H - 3.0) / 2))
    add("wall_e", (Lx / 2, 0.0, H / 2), (wall / 2, Ly / 2, H / 2))
    # floors at 4 and 8 m: slabs with a stair opening (6 × 3 m at the west end) and a hole (3 × 3 m at the east end)
    for k, z in enumerate((4.0, 8.0)):
        t = 0.15
        add(f"floor{k}_main", (0.0, -1.5 - 0.0, z), (Lx / 2 - 6.0, Ly / 2 - 3.0 - 1.5 + 0.0, t / 2), floorc)         # the middle, leaving 3 m strips at the north edge
        add(f"floor{k}_north", (0.0, Ly / 2 - 1.5, z), (Lx / 2, 1.5, t / 2), floorc)
        add(f"floor{k}_west", (-Lx / 2 + 3.0, -Ly / 2 + 4.5 + 1.5, z), (3.0, 3.0 + 1.5, t / 2), floorc)            # west end: the stair opening is the south 6 × 3 m
        add(f"floor{k}_east_a", (Lx / 2 - 3.0, -Ly / 2 + 1.5, z), (3.0, 1.5, t / 2), floorc)                      # east end: the hole is the middle 3 × 3 m (y −3 … 0)
        add(f"floor{k}_east_b", (Lx / 2 - 3.0, Ly / 2 - 6.0 - 1.5, z), (3.0, 4.5 - 3.0 + 1.5 - 0.0, t / 2), floorc)
    # the staircase: a straight flight of boxes rising from the ground to the first floor in the west stair opening
    for i in range(8):
        add(f"stair{i}", (-Lx / 2 + 0.6 + 0.75 * i, -Ly / 2 + 1.5, 0.25 + 0.5 * i), (0.4, 1.4, 0.25), (0.55, 0.55, 0.6, 1.0))
    # roof beams: every 5 m across the hall, 0.5 m deep under the roof
    for i, x in enumerate(np.arange(-Lx / 2 + 5.0, Lx / 2, 5.0)):
        add(f"beam{i}", (x, 0.0, H - 0.25), (0.15, Ly / 2, 0.25), beam)
    add("roof", (0.0, 0.0, H + 0.1), (Lx / 2 + 0.5, Ly / 2 + 0.5, 0.1), (0.5, 0.5, 0.5, 0.3))
    gates = [Gate((-Lx / 2 - 8.0, 0.0, 1.5), 3.0, "start"), Gate((-Lx / 2 - 1.0, 0.0, 1.5), 1.5, "door"), Gate((-Lx / 2 + 6.0, 0.0, 1.8), 2.0, "hall"),
             Gate((-Lx / 2 + 3.0, -Ly / 2 + 1.5, 2.0), 1.3, "to the stair"), Gate((-Lx / 2 + 3.0, -Ly / 2 + 1.5, 5.8), 1.3, "stair opening, 1st floor"),
             Gate((0.0, -4.0, 5.8), 2.5, "1st floor"), Gate((Lx / 2 - 3.0, -1.5, 5.8), 1.3, "under the hole"), Gate((Lx / 2 - 3.0, -1.5, 9.8), 1.3, "2nd floor"),
             Gate((5.0, 0.0, 10.3), 2.5, "under the beams"), Gate((-10.0, 2.0, 10.3), 2.5, "beams"), Gate((-Lx / 2 + 3.0, -Ly / 2 + 1.5, 9.8), 1.3, "stair opening, top"),
             Gate((-Lx / 2 + 3.0, -Ly / 2 + 1.5, 1.8), 1.3, "down to the ground"), Gate((-Lx / 2 + 12.0, -Ly / 2 + 3.0, 2.0), 1.5, "to the window"),
             Gate((-Lx / 2 + 8.0 - 0.6, -Ly / 2 - 0.0, 2.0), 0.55, "window"), Gate((-Lx / 2 + 8.0 - 0.6, -Ly / 2 - 6.0, 2.0), 2.0, "finish")]
    return Course("hangar, three floors", props, gates, (-Lx / 2 - 12.0, Lx / 2 + 4.0, -Ly / 2 - 10.0, Ly / 2 + 4.0),
                  "a 40 × 18 m hangar: in by the door, up the stair opening, through the hole to the second floor, under the roof beams, down, out by a 1.2 m window", boxes)


def industrial() -> Course:
    """Outside the hangar: six large buildings (20–35 m, 10–16 m tall) with 10–15 m gaps between them; the course
    weaves through the gaps, round the far end and back, at 6 m."""
    props, boxes = [], []
    bl = [((0.0, -25.0, 7.0), (15.0, 8.0, 7.0)), ((45.0, -28.0, 6.0), (12.0, 10.0, 6.0)), ((90.0, -22.0, 8.0), (16.0, 7.0, 8.0)),
          ((10.0, 20.0, 5.5), (17.0, 9.0, 5.5)), ((55.0, 18.0, 7.5), (14.0, 8.0, 7.5)), ((100.0, 20.0, 6.5), (12.0, 10.0, 6.5)), ((140.0, 0.0, 8.0), (10.0, 20.0, 8.0))]
    for i, (c, h) in enumerate(bl):
        props.append(_box(f"building{i}", c, h, (0.55, 0.5, 0.48, 1.0))); boxes.append((c, h))
    for i, (c, h) in enumerate([((30.0, -3.0, 2.5), (1.0, 1.0, 2.5)), ((70.0, 4.0, 3.0), (1.2, 1.2, 3.0)), ((120.0, -6.0, 6.0), (0.6, 0.6, 6.0))]):
        props.append(_box(f"stack{i}", c, h, (0.4, 0.4, 0.42, 1.0))); boxes.append((c, h))
    gates = [Gate((-30.0, 0.0, 1.5), 4.0, "start"), Gate((-5.0, -3.0, 6.0), 5.0, "gap 1"), Gate((30.0, -10.0, 6.0), 4.0, "between"), Gate((60.0, -6.0, 6.0), 5.0, "gap 2"),
             Gate((95.0, 0.0, 6.0), 5.0, "gap 3"), Gate((125.0, 12.0, 6.0), 4.0, "the turn"), Gate((150.0, 25.0, 8.0), 6.0, "far end"), Gate((165.0, 0.0, 8.0), 6.0, "round the last building"),
             Gate((140.0, -30.0, 8.0), 6.0, "back"), Gate((105.0, -8.0, 10.0), 5.0, "over the stack"), Gate((70.0, -2.0, 6.0), 4.0, "gap 2 again"), Gate((30.0, 6.0, 6.0), 4.0, "gap 1 again"),
             Gate((-10.0, 2.0, 4.0), 5.0, "home straight"), Gate((-30.0, 0.0, 3.0), 5.0, "finish")]
    return Course("industrial zone", props, gates, (-40.0, 180.0, -45.0, 45.0), "six buildings with 10–15 m gaps, three stacks: a 400 m lap through the gaps and round the far end", boxes)


def farm() -> Course:
    """A farm: a cottage with a pitched roof, a barn open at both ends (fly through), trees in a line and a copse;
    the course goes round the cottage, through the barn, slaloms the tree line and finishes over the yard."""
    props, boxes = [], []
    grey, wood, trunk, crown = (0.8, 0.75, 0.65, 1.0), (0.55, 0.35, 0.2, 1.0), (0.35, 0.25, 0.15, 1.0), (0.2, 0.45, 0.2, 1.0)

    def add(name, centre, half, rgba):
        props.append(_box(name, centre, half, rgba)); boxes.append((tuple(centre), tuple(half)))
    add("cottage", (0.0, 15.0, 2.0), (5.0, 4.0, 2.0), grey)
    add("cottage_roof", (0.0, 15.0, 4.8), (5.5, 4.5, 0.8), (0.6, 0.3, 0.25, 1.0))
    add("chimney", (3.0, 15.0, 6.3), (0.4, 0.4, 0.7), grey)
    # the barn: 20 m long along x, 10 m wide, 6 m tall, open at both ends: two long walls and a roof
    add("barn_wall_s", (40.0, -5.0, 3.0), (10.0, 0.15, 3.0), wood)
    add("barn_wall_n", (40.0, 5.0, 3.0), (10.0, 0.15, 3.0), wood)
    add("barn_roof", (40.0, 0.0, 6.3), (10.5, 5.5, 0.3), (0.5, 0.25, 0.2, 1.0))
    for i, x in enumerate(np.arange(32.0, 49.0, 4.0)):
        add(f"barn_post{i}", (x, 0.0, 3.0), (0.15, 0.15, 3.0), wood)                    # a row of posts down the barn's middle: pick a side
    # a line of trees along y = -20 every 8 m, a copse near the finish
    k = 0
    for x in np.arange(-10.0, 70.0, 8.0):
        props.append(_cyl(f"trunk{k}", (x, -20.0, 2.5), 0.25, 2.5, trunk)); props.append(_sphere(f"crown{k}", (x, -20.0, 6.5), 3.0, crown)); boxes.append(((x, -20.0, 6.5), (3.0, 3.0, 3.0))); k += 1
    for (x, y) in ((80.0, 5.0), (86.0, 12.0), (78.0, 14.0), (90.0, 2.0)):
        props.append(_cyl(f"trunk{k}", (x, y, 3.0), 0.3, 3.0, trunk)); props.append(_sphere(f"crown{k}", (x, y, 8.0), 4.0, crown)); boxes.append(((x, y, 8.0), (4.0, 4.0, 4.0))); k += 1
    gates = [Gate((-20.0, 0.0, 1.5), 4.0, "start"), Gate((-8.0, 15.0, 3.0), 3.0, "round the cottage"), Gate((0.0, 23.0, 3.0), 3.0, "behind it"), Gate((8.0, 15.0, 3.0), 3.0, "and back"),
             Gate((20.0, 2.0, 3.0), 3.0, "to the barn"), Gate((29.0, 2.5, 3.0), 1.5, "barn door"), Gate((51.0, 2.5, 3.0), 1.5, "barn far door"), Gate((62.0, -8.0, 4.0), 3.0, "towards the trees"),
             Gate((58.0, -20.0, 4.0), 2.0, "tree 1"), Gate((50.0, -16.0, 4.0), 2.0, "tree 2"), Gate((42.0, -24.0, 4.0), 2.0, "tree 3"), Gate((34.0, -16.0, 4.0), 2.0, "tree 4"),
             Gate((26.0, -24.0, 4.0), 2.0, "tree 5"), Gate((18.0, -16.0, 4.0), 2.0, "tree 6"), Gate((10.0, -24.0, 4.0), 2.0, "tree 7"), Gate((0.0, -10.0, 4.0), 4.0, "the yard"),
             Gate((-20.0, 0.0, 3.0), 4.0, "finish")]
    return Course("farm", props, gates, (-30.0, 100.0, -35.0, 30.0), "round the cottage, through the barn beside its posts, a slalom down the tree line, home over the yard", boxes)


COURSES = {"hangar": hangar, "industrial": industrial, "farm": farm}


# ================================================================================================= the path and its speed schedule
class Path:
    """The course's polyline with arc length, a tangent, and a speed schedule for one craft: the corner speed each
    gate's radius allows at the craft's lateral acceleration, braking and acceleration between them, the top speed."""

    def __init__(self, course: Course, c: Craft, *, safety: float = 0.8, v_cap: float | None = None):
        P = course.points
        self.P = P
        seg = np.diff(P, axis=0)
        self.L = np.linalg.norm(seg, axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(self.L)])
        self.tan = seg / self.L[:, None]
        a_lat = c.a_lateral()
        a_brake = min(0.6 * a_lat, 2.5 * G)
        a_acc = min(0.5 * (c.T_max_static / c.m - G), 2.0 * G)
        v_top = min(c.v_top() * 0.9, v_cap or 1e9)
        v = np.full(len(P), v_top)
        for i, g in enumerate(course.gates):
            if 0 < i < len(P) - 1:
                t0, t1 = self.tan[i - 1], self.tan[i]
                turn = math.acos(max(-1.0, min(1.0, float(np.dot(t0, t1)))))
                r = g.r / max(math.tan(turn / 2), 0.05)             # the radius a turn through 'turn' fits inside the gate's clearance
                r = min(r, 60.0)
                v[i] = min(v_top, safety * math.sqrt(a_lat * r), safety * math.sqrt(a_lat * g.r * 3.0))
            else:
                v[i] = min(v_top, 3.0)
        v[-1] = 2.0
        for i in range(len(P) - 2, -1, -1):                         # braking backwards
            v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2 * a_brake * self.L[i]))
        for i in range(1, len(P)):                                  # accelerating forwards
            v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2 * a_acc * self.L[i - 1]))
        self.v = v
        self.a_lat, self.a_brake, self.a_acc, self.v_top = a_lat, a_brake, a_acc, v_top

    def at(self, s: float) -> tuple:
        s = float(np.clip(s, 0.0, self.s[-1]))
        i = min(int(np.searchsorted(self.s, s, side="right")) - 1, len(self.L) - 1)
        f = (s - self.s[i]) / self.L[i]
        p = self.P[i] + f * (self.P[i + 1] - self.P[i])
        v = self.v[i] + f * (self.v[i + 1] - self.v[i])
        return p, self.tan[i], v, i

    def project(self, p: np.ndarray, s_hint: float) -> float:
        """Arc length of the point on the polyline nearest ``p``, searched near ``s_hint`` (no going back a leg)."""
        best, s_best = 1e9, s_hint
        i0 = max(int(np.searchsorted(self.s, s_hint, side="right")) - 2, 0)
        for i in range(i0, min(i0 + 3, len(self.L))):
            d = p - self.P[i]
            f = float(np.clip(np.dot(d, self.tan[i]) / self.L[i], 0.0, 1.0))
            q = self.P[i] + f * (self.P[i + 1] - self.P[i])
            dist = float(np.linalg.norm(p - q))
            if dist < best:
                best, s_best = dist, self.s[i] + f * self.L[i]
        return max(s_best, s_hint - 0.5)

    def time_estimate(self) -> float:
        """The schedule's own lap time (the reference's, not the aircraft's)."""
        t = 0.0
        for i in range(len(self.L)):
            t += 2 * self.L[i] / max(self.v[i] + self.v[i + 1], 0.5)
        return t


# ================================================================================================= the autopilot
class Racer:
    """One autopilot for every craft (see the module). Phases: ``takeoff`` (straight up to the first gate's height),
    ``course`` (the reference point ahead on the path), ``done``."""

    def __init__(self, c: Craft, course: Course, airframe: Airframe, *, omega_pos: float = 2.0, zeta: float = 1.0, lookahead_s: float = 0.5, v_cap: float | None = None):
        self.c, self.course, self.af = c, course, airframe
        self.omega_pos, self.zeta, self.lookahead_s = omega_pos, zeta, lookahead_s
        self.path = Path(course, c, v_cap=v_cap)
        self.name = f"Racer: {c.name} on {course.name}"

    def reset(self, lab, seed=None):
        import mujoco
        self.m = float(lab.total_mass)
        d = mujoco.MjData(lab.model)
        d.qpos[:] = lab.data.qpos
        mujoco.mj_forward(lab.model, d)
        M = np.zeros((lab.model.nv, lab.model.nv))
        try:
            mujoco.mj_fullM(lab.model, d, M)
        except TypeError:
            mujoco.mj_fullM(lab.model, M, d.qM)
        a = int(lab.model.jnt_dofadr[lab._root_jnt])
        self.J = np.diag(M[a + 3:a + 6, a + 3:a + 6]).copy()
        w = self.c.omega_att
        self.kR = self.J * w * w
        self.kW = 2 * 0.8 * self.J * w
        n = self.c.n_motors
        st = self.c.stations
        self.A = np.vstack([np.ones(n), self.c.spin * self.c.kappa, st[:, 2], -st[:, 1]])      # T, Mx, My, Mz from the thrusts
        self.phase = "takeoff"
        self.s = 0.0
        self.t_start = None
        self.gate_times = {}
        self.next_gate = 1
        self.events = []
        self.b1_prev = np.array([0.0, 0.0, 1.0])
        self.T_min = 0.03 * self.c.T_max_static / n

    # ---- the loops
    def mix(self, T: float, M: np.ndarray) -> tuple:
        """Motor thrusts for the collective ``T`` and body moments ``M``; the moments the motors cannot give go to the surfaces."""
        n = self.c.n_motors
        hi = self.c.T_max_static / n
        lo = self.T_min
        T = float(np.clip(T, n * lo, n * hi))
        if n == 4:
            u, *_ = np.linalg.lstsq(self.A, np.array([T, M[0], M[1], M[2]]), rcond=None)
            if u.min() < lo or u.max() > hi:
                for k in (0.6, 0.35, 0.15, 0.0):
                    u, *_ = np.linalg.lstsq(self.A, np.array([T, k * M[0], k * M[1], k * M[2]]), rcond=None)
                    if lo <= u.min() and u.max() <= hi:
                        break
            u = np.clip(u, lo, hi)
            got = self.A @ u
            return u, M - got[1:]
        # two motors side by side: the collective and the yaw moment (about z, from the y stations)
        A2 = np.vstack([np.ones(2), -self.c.stations[:, 1]])
        u = np.linalg.solve(A2, np.array([T, M[2]]))
        if u.min() < lo or u.max() > hi:
            u = np.linalg.solve(A2, np.array([T, 0.3 * M[2]]))
        u = np.clip(u, lo, hi)
        got = self.A @ u
        return u, M - got[1:]

    def attitude(self, R, omega_b, b1, b3_hint):
        b2 = np.cross(b3_hint, b1)
        if np.linalg.norm(b2) < 1e-6:
            b2 = np.cross(np.array([0.0, 0.0, 1.0]), b1)
            if np.linalg.norm(b2) < 1e-6:
                b2 = np.array([0.0, 1.0, 0.0])
        b2 /= np.linalg.norm(b2)
        b3 = np.cross(b1, b2)
        Rd = np.column_stack([b1, b2, b3])
        E = 0.5 * (Rd.T @ R - R.T @ Rd)
        eR = np.array([E[2, 1], E[0, 2], E[1, 0]])
        return -self.kR * eR - self.kW * omega_b

    def __call__(self, obs):
        c, af = self.c, self.af
        p = np.asarray(obs.base_pos, dtype=float)
        v = np.asarray(obs.body_linvel[0], dtype=float)
        R = _quat_to_R(obs.base_quat)
        omega_b = np.asarray(obs.body_angvel[0], dtype=float)
        t = float(obs.t)
        m = self.m
        P = self.path
        start = P.P[0]
        if self.phase == "takeoff":
            p_ref, v_ref, a_ref = start.copy(), np.zeros(3), np.zeros(3)
            if abs(p[2] - start[2]) < 0.4 and np.linalg.norm(v) < 1.5:
                self.phase, self.t_start = "course", t
                self.events.append((t, "off the start"))
        if self.phase == "course":
            self.s = P.project(p, self.s)
            _, _, v_here, _ = P.at(self.s)
            la = max(1.5, self.lookahead_s * max(np.linalg.norm(v), v_here))
            p_ref, tan, v_sched, i = P.at(self.s + la)
            v_ref = tan * v_sched
            p_next, tan_next, _, _ = P.at(self.s + la + 1.0)
            a_ref = (tan_next - tan) * v_sched * v_sched / 1.0 * 0.5
            while self.next_gate < len(P.P) and self.s >= P.s[self.next_gate] - 0.3:
                self.gate_times[self.course.gates[self.next_gate].name or str(self.next_gate)] = t - self.t_start
                self.next_gate += 1
            if self.s >= P.s[-1] - 0.5:
                self.phase = "done"
                self.events.append((t, "finish"))
        if self.phase == "done":
            p_ref, v_ref, a_ref = P.P[-1].copy(), np.zeros(3), np.zeros(3)
        wp = self.omega_pos
        kx, kv = m * wp * wp, 2 * self.zeta * m * wp
        a_des = a_ref - kx / m * (p - p_ref) - kv / m * (v - v_ref)
        a_lim = max(0.75 * c.T_max_static / m - G, 1.5 * G)
        na = np.linalg.norm(a_des)
        if na > a_lim:
            a_des *= a_lim / na
        S = a_des + np.array([0.0, 0.0, G])                                     # the specific force wanted
        vb = R.T @ v
        F_aero_b, _ = af.aero(vb, m * float(np.linalg.norm(S)))
        F = m * S - R @ F_aero_b                                               # what the thrust must give
        nF = np.linalg.norm(F)
        if nF < 0.15 * m * G:
            b1 = self.b1_prev
            T = 0.15 * m * G
        else:
            b1 = F / nF
            T = float(nF)
        # a weak-pitch craft turns its thrust axis slowly: rate-limit the demanded direction
        max_turn = 6.0 * obs_dt(obs) if c.M_pitch_coeff > 0 and c.n_motors == 2 else 20.0 * obs_dt(obs)
        ang = math.acos(max(-1.0, min(1.0, float(np.dot(b1, self.b1_prev)))))
        if ang > max_turn:
            axis = np.cross(self.b1_prev, b1); axis /= np.linalg.norm(axis)
            b1 = _rotate(self.b1_prev, axis, max_turn)
        self.b1_prev = b1
        # the wing's lift plane towards the specific force (the bank in a turn), else up
        b3 = S - np.dot(S, b1) * b1
        if np.linalg.norm(b3) < 0.5 or c.S == 0:
            b3 = np.array([0.0, 0.0, 1.0]) - b1[2] * b1
            if np.linalg.norm(b3) < 0.2:
                b3 = np.array([-1.0, 0.0, 0.0]) - b1[0] * b1 * 0 + np.array([0.0, 0.0, 0.0]) if False else np.cross(b1, np.array([0.0, 1.0, 0.0]))
        b3 /= np.linalg.norm(b3)
        M = self.attitude(R, omega_b, b1, b3)
        T_cmd = float(np.clip(np.dot(F, R[:, 0]), 0.15 * m * G, c.T_max_static)) if nF >= 0.15 * m * G else T
        u, residual = self.mix(T_cmd, M)
        af.command = u
        af.surface = np.array([residual[0], residual[1]])
        return ch.Command(q_target={})


def obs_dt(obs) -> float:
    return 0.004


def _rotate(v, axis, ang):
    return v * math.cos(ang) + np.cross(axis, v) * math.sin(ang) + axis * np.dot(axis, v) * (1 - math.cos(ang))


def _quat_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


# ================================================================================================= crashes and the run
class Watch:
    """Scene hook: a contact between the craft's body (not its feet before it has left the ground) and anything is a
    crash; records where and when."""

    def __init__(self, racer: Racer):
        self.racer = racer

    def reset(self, lab):
        m = lab.model
        self.root = lab._body_id("craft")
        self.craft_geoms = {g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == self.root}
        names = [m.geom(g).name for g in self.craft_geoms]
        self.feet = {g for g in self.craft_geoms if m.geom(g).name.startswith("foot")}
        self.crashed = None

    def __call__(self, lab):
        if self.crashed is not None:
            return
        d, m = lab.data, lab.model
        for k in range(d.ncon):
            con = d.contact[k]
            g1, g2 = int(con.geom1), int(con.geom2)
            mine = g1 if g1 in self.craft_geoms else g2 if g2 in self.craft_geoms else None
            if mine is None:
                continue
            if mine in self.feet and self.racer.phase in ("takeoff", "done"):
                continue
            other = g2 if mine == g1 else g1
            what = m.geom(other).name or "the ground"
            self.crashed = {"t": float(lab.time), "what": what.replace("_g", ""), "pos": np.array(d.xpos[self.root]).tolist(), "phase": self.racer.phase, "V": float(np.linalg.norm(d.qvel[lab.model.jnt_dofadr[lab._root_jnt]:lab.model.jnt_dofadr[lab._root_jnt] + 3]))}
            lab.log_event("crash", f"{self.racer.c.name} hit {what} at {lab.time:.1f} s")
            return


def race(c: Craft, course: Course, *, duration: float | None = None, seed: int = 0, v_cap: float | None = None, racer_kw: dict | None = None) -> dict:
    """One craft round one course: the lap time, the energy, the outcome, the trace."""
    rb = robot(c)
    opts = dict(LAB_OPTIONS)
    opts["course_extent"] = course.extent
    opts["heightfield_cell"] = 0.5
    lab = ch.ChironLab(rb, ch.Flat(), props=course.props, **opts)
    af = Airframe(c)
    lab.add_hook(af)
    racer = Racer(c, course, af, v_cap=v_cap, **(racer_kw or {}))
    watch = Watch(racer)
    lab.add_hook(watch)
    start = course.gates[0].p
    z0 = c.geometry["x_feet"] + 0.006 if c.kind != "quad" else 0.05
    T_est = racer.path.time_estimate()
    duration = duration or min(1.6 * T_est + 15.0, 240.0)
    ep = lab.run(racer, duration=duration, rules=None, settle=0.2, seed=seed, base_pos=(start[0], start[1], z0), base_quat=UP_QUAT,
                 stop_when=lambda lab: watch.crashed is not None or (racer.phase == "done" and lab.time > (racer.events[-1][0] + 1.0)))
    H = np.asarray(af.history, dtype=float).reshape(-1, 11)
    fin = racer.phase == "done"
    t_lap = (racer.events[-1][0] - racer.t_start) if fin and racer.t_start is not None else np.nan
    out = {"craft": c.name, "course": course.name, "finished": fin, "crashed": watch.crashed, "lap time [s]": t_lap, "schedule time [s]": T_est,
           "energy [Wh]": af.energy_Wh, "Wh per km": af.energy_Wh / (course.length() / 1e3), "max speed [m/s]": float(H[:, 9].max()) if len(H) else np.nan,
           "mean speed on course [m/s]": course.length() / t_lap if fin else np.nan, "peak power [W]": float(H[:, 8].max()) if len(H) else np.nan,
           "course length [m]": course.length(), "gate times": dict(racer.gate_times), "history": H, "path": racer.path, "events": racer.events + list(map(tuple, ep.log.get("events", []))) if isinstance(ep.log.get("events"), list) else racer.events}
    return out


def tournament(crafts: dict, courses: dict, **kw) -> tuple:
    rows, runs = [], {}
    for cn, course in courses.items():
        for name, c in crafts.items():
            r = race(c, course, **kw)
            runs[(cn, name)] = r
            rows.append({"course": cn, "craft": name, "finished": r["finished"], "lap time [s]": r["lap time [s]"], "schedule [s]": r["schedule time [s]"], "energy [Wh]": r["energy [Wh]"],
                         "Wh per km": r["Wh per km"], "max speed [m/s]": r["max speed [m/s]"], "mean speed [m/s]": r["mean speed on course [m/s]"], "peak power [W]": r["peak power [W]"],
                         "crash": (f"{r['crashed']['what']} at {r['crashed']['t']:.0f} s, {r['crashed']['V']:.0f} m/s" if r["crashed"] else "")})
    return pd.DataFrame(rows), runs


# ================================================================================================= drawings
def draw_course(course: Course, runs: dict | None = None, ax=None, *, view: str = "top", title: str | None = None):
    """The course's boxes and gates (top or side view) with the flown traces."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))
    a, b = (0, 1) if view == "top" else (0, 2)
    for centre, half in course.boxes:
        ax.add_patch(Rectangle((centre[a] - half[a], centre[b] - half[b]), 2 * half[a], 2 * half[b], facecolor="#c9c4bb", edgecolor="#8a8378", lw=0.6, alpha=0.8))
    P = course.points
    ax.plot(P[:, a], P[:, b], "k:", lw=0.8)
    for g in course.gates:
        ax.plot(g.p[a], g.p[b], "k+", ms=6)
    if runs:
        for (cn, name), r in runs.items():
            if cn != course.name and cn not in course.name:
                continue
            H = r["history"]
            if len(H):
                ax.plot(H[:, 1 + a], H[:, 1 + b], lw=1.2, label=f"{name}: {'%.0f s' % r['lap time [s]'] if r['finished'] else 'crashed'}")
                if r["crashed"]:
                    ax.plot(r["crashed"]["pos"][a], r["crashed"]["pos"][b], "rx", ms=9)
        ax.legend(fontsize=8, loc="best")
    ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]" if view == "top" else "z [m]"); ax.grid(alpha=0.2)
    ax.set_title(title or f"{course.name}: {view} view")
    return ax
