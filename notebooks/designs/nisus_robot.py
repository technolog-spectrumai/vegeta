"""NISUS in ChironLab (MuJoCo): the airframe as a Chiron robot, the aerodynamic and propulsion forces as a scene
hook, the wind, the battery (notebook 31, ``nisus_scenario``).

* **the airframe** — one rigid body (``Link('nisus')``): the pod as an ellipsoid shell, the wing, booms, stabiliser
  and fins as thin boxes and capsules (they collide with the ground: a wing or nose strike is a belly contact), the
  belly skid and the two fin stubs as the **feet** (what the aircraft lands on), the motor and the propeller disc
  drawn. Every row of ``nisus_systems.mass_table`` is on it: the structure as the shapes' masses, the components as
  point masses where the table puts them — so MuJoCo's mass, centre of gravity and inertia are the mass table's
  (``inertia_check`` compares them). No joints: the control surfaces are states of the aerodynamic model, not
  MuJoCo hinges (their inertia is negligible and their loads come from the derivatives).
* **the forces** (``Aero``, a scene hook): every control step the relative wind in body axes gives α, β, V and the
  rates; the coefficient table of ``nisus_flight.derivatives`` (calculated / analytic / assumed, each tagged) gives
  CL, CD, CY, Cl, Cm, Cn with a smooth post-stall blend (an assumption: a flat-plate normal force beyond the stall
  angle), the control deflections follow the servo commands with the servo's rate limit and a lag, the thrust comes
  from the propulsion map at the axial airspeed and the (lagged) throttle, tilted by the down-thrust and acting at
  the motor's position; the forces and moments go on the body with ``lab.body_force``. Gravity is MuJoCo's; the
  built-in fluid model is **off** (``density = viscosity = 0``, asserted) so nothing is counted twice. The electrical
  power (map + the electronics of the phase) is integrated into the battery's energy.
* **the wind** (``Wind``): a steady vector, continuous turbulence as a first-order (Ornstein–Uhlenbeck) process per
  axis with a seed, and an optional discrete 1−cos gust.

Units SI; the Chiron body frame is x forward, y left, z up (the aerodynamic body axes x forward, y right, z down
are converted inside ``Aero``); positions from ``nisus``'s frame (mm, x aft) through ``to_body``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from vegeta.chiron import FootSpec, Geom, Link, PointMass, Robot

import nisus
import nisus_flight as nf
import nisus_systems as ns

G = 9.81
RHO = 1.225

__all__ = ["G", "RHO", "to_body", "nisus_robot", "inertia_check", "Wind", "Actuators", "Aero", "LAB_OPTIONS", "quat_to_R", "euler_from_R"]


def to_body(x_mm, y_mm, z_mm) -> tuple:
    """A point of ``nisus``'s frame (mm; x aft, y right, z up, origin at the wing root leading edge) in the Chiron body
    frame (m; x forward, y left, z up, same origin)."""
    return (-x_mm / 1000.0, -y_mm / 1000.0, z_mm / 1000.0)


def quat_to_R(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def euler_from_R(R) -> tuple:
    """(roll right-wing-down, pitch nose-up, heading from +x towards +y) [rad] of a Chiron body frame (x fwd, y left, z up)."""
    pitch = math.asin(max(-1.0, min(1.0, R[2, 0])))
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


# ================================================================================================= the robot
def _wing_x(table) -> float:
    """The mass-weighted x [mm] of the mass table's wing rows: the wing box carries their mass at their centroid."""
    w = table[table.index.str.startswith("wing:")]
    return float((w["mass [g]"] * w["x [mm]"]).sum() / w["mass [g]"].sum())


def _pod_x(table, default) -> float:
    w = table[table.index.str.startswith("pod:") & table.index.str.contains("shell")]
    return float((w["mass [g]"] * w["x [mm]"]).sum() / w["mass [g]"].sum()) if len(w) else default

def nisus_robot(variant: str = "Zero", battery_key: str = "gens-ace-3s-2200", p=None, *, table=None) -> Robot:
    """The aircraft as a Chiron ``Robot`` (see the module). ``table``: a mass table to use instead of the default."""
    p = nisus.resolve(p)
    L = nisus.Nisus.layout(p)
    t = ns.mass_table(variant, battery_key, p=p) if table is None else table
    m = t["mass [g]"] / 1000.0
    pod_rgba, foam, carbon, dark = (0.86, 0.87, 0.9, 1.0), (0.93, 0.93, 0.95, 1.0), (0.12, 0.12, 0.14, 1.0), (0.1, 0.1, 0.1, 1.0)
    groups = {"wing": 0.0, "pod": 0.0, "booms": 0.0, "tail": 0.0}
    masses = []
    for item, row in t.iterrows():
        mi = float(row["mass [g]"]) / 1000.0
        if item.startswith("wing:"):
            groups["wing"] += mi
        elif item.startswith("pod:") and "shell" in item:
            groups["pod"] += mi
        elif item.startswith("booms:"):
            groups["booms"] += mi
        elif item.startswith("tail:"):
            groups["tail"] += mi
        else:
            masses.append(PointMass(item[:60].replace(" ", "_").replace("/", "-"), mi, to_body(row["x [mm]"], row["y [mm]"], row["z [mm]"])))
    c0, b = p["root_chord"], p["span"]
    x_pod_c = L["x_nose"] + p["pod_length"] / 2
    geoms = [
        Geom("pod", "ellipsoid", (p["pod_length"] / 2000, p["pod_width"] / 2000, p["pod_height"] / 2000), pos=to_body(_pod_x(t, x_pod_c), 0, L["pod_axis_z"]),
             mass=groups["pod"], role="body", rgba=pod_rgba),
        Geom("wing", "box", (0.5 * (c0 + p["tip_chord"]) / 2000, b / 2000, 0.012), pos=to_body(_wing_x(t), 0, 8.0),
             mass=groups["wing"], role="body", rgba=foam),
        Geom("stab", "box", (p["tail_chord"] / 2000, p["tail_span"] / 2000, 0.0035), pos=to_body(L["tail_le"] + p["tail_chord"] / 2, 0, p["boom_z"]),
             mass=0.4 * groups["tail"], role="body", rgba=foam),
        Geom("motor", "cylinder", (p["motor_diameter"] / 2000, p["motor_length"] / 2000), pos=to_body(0.5 * (L["x_motor0"] + L["x_motor1"]), 0, p["motor_z"]),
             quat=(math.cos(math.pi / 4), 0, math.sin(math.pi / 4), 0), mass=None, role="visual", rgba=dark),
        Geom("prop_disc", "cylinder", (L["prop_R"] / 1000, 0.0006), pos=to_body(L["prop_x"], 0, p["motor_z"]),
             quat=(math.cos(math.pi / 4), 0, math.sin(math.pi / 4), 0), mass=None, role="visual", rgba=(0.6, 0.75, 0.95, 0.30)),
        Geom("skid", "box", ((p["skid_x1"] - p["skid_x0"]) / 2000, 0.003, p["skid_depth"] / 2000),
             pos=to_body(0.5 * (p["skid_x0"] + p["skid_x1"]), 0, -p["pod_height"] - p["skid_depth"] / 2), mass=None, role="foot",
             friction=(0.5, 0.005, 0.0001), rgba=(0.9, 0.4, 0.1, 1.0)),
        Geom("nose_tip", "sphere", (0.02,), pos=to_body(L["x_nose"] + 20, 0, L["pod_axis_z"]), mass=None, role="body", rgba=pod_rgba),
    ]
    for side, s in (("L", -1), ("R", 1)):          # nisus y right positive: side R is +y
        geoms.append(Geom(f"boom_{side}", "capsule", (p["boom_od"] / 2000,),
                          fromto=(*to_body(p["boom_x0"], s * p["boom_y"], p["boom_z"]), *to_body(L["boom_x1"], s * p["boom_y"], p["boom_z"])),
                          mass=groups["booms"] / 2, role="body", rgba=carbon))
        geoms.append(Geom(f"fin_{side}", "box", (p["fin_chord"] / 2000, 0.003, (p["fin_height"] + p["fin_ventral"]) / 2000),
                          pos=to_body(L["fin_le"] + p["fin_chord"] / 2, s * p["boom_y"], p["boom_z"] + 0.5 * (p["fin_height"] - p["fin_ventral"])),
                          mass=0.3 * groups["tail"], role="body", rgba=foam))
        geoms.append(Geom(f"stub_{side}", "capsule", (0.004,), fromto=(*to_body(L["fin_le"] + 10, s * p["boom_y"], p["boom_z"] - p["fin_ventral"]),
                                                                       *to_body(L["boom_x1"] - 10, s * p["boom_y"], p["boom_z"] - p["fin_ventral"])),
                          mass=None, role="foot", friction=(0.5, 0.005, 0.0001), rgba=dark))
    feet = [FootSpec("skid", "skid", [], "nisus"), FootSpec("stub_L", "stub_L", [], "nisus"), FootSpec("stub_R", "stub_R", [], "nisus")]
    root = Link("nisus", geoms=geoms, masses=masses, log=True)
    robot = Robot(f"Nisus-{variant}", root=root, feet=feet,
                  notes=f"Nisus-{variant} on the {battery_key} pack: nisus_systems.mass_table as geom and point masses; no joints "
                        "(the control surfaces are states of the Aero hook)",
                  sources={"geometry": "designs/nisus.py", "masses": "designs/nisus_systems.py: mass_table", "aerodynamics": "designs/nisus_flight.py: derivatives",
                           "propulsion": "designs/nisus_systems.py: propulsion_map (Boreas BEMT + fitted X2216)"})
    robot.validate()
    robot.params, robot.variant, robot.battery_key, robot.mass_table = p, variant, battery_key, t
    robot.layout = L
    return robot


def inertia_check(lab, table) -> dict:
    """MuJoCo's total mass, COM and inertia of the root body (about its COM, body axes) against ``nisus_systems.cg_inertia``
    of the same table (point masses + simple shapes): the two must agree to a few percent."""
    import mujoco
    m, d = lab.model, lab.data
    b = lab._body_id("nisus")
    mass = float(m.body_subtreemass[b])
    hand = ns.cg_inertia(table)
    # MuJoCo keeps the body's inertia about its own COM in the inertial frame (body_iquat): rotate to the body frame
    R = quat_to_R(m.body_iquat[b])
    I = R @ np.diag(m.body_inertia[b]) @ R.T
    com_body = m.body_ipos[b]                      # body frame (x fwd, y left, z up): convert to nisus frame mm
    return {"mujoco_mass_kg": mass, "table_mass_kg": hand["mass_kg"], "mass_error": mass / hand["mass_kg"] - 1,
            "mujoco_cg_x_mm": -com_body[0] * 1000, "table_cg_x_mm": hand["x_cg_m"] * 1000, "mujoco_cg_z_mm": com_body[2] * 1000, "table_cg_z_mm": hand["z_cg_m"] * 1000,
            "mujoco_Ixx": float(I[0, 0]), "mujoco_Iyy": float(I[1, 1]), "mujoco_Izz": float(I[2, 2]),
            "table_Ixx": hand["Ixx"], "table_Iyy": hand["Iyy"], "table_Izz": hand["Izz"],
            "Ixx_ratio": float(I[0, 0]) / hand["Ixx"], "Iyy_ratio": float(I[1, 1]) / hand["Iyy"], "Izz_ratio": float(I[2, 2]) / hand["Izz"]}


# ================================================================================================= wind
@dataclass
class Wind:
    """The air's velocity [m/s, world]: ``steady`` (vx, vy, vz) plus turbulence — an Ornstein–Uhlenbeck process per
    axis with standard deviation ``sigma`` (horizontal; half of it vertical) and time constant ``tau`` (a 200 m
    length scale at cruise speed: an assumption standing in for a Dryden spectrum) — plus an optional discrete 1−cos
    gust (``gust``: amplitude [m/s], start [s], duration [s], direction unit vector). Seeded: a trial repeats exactly."""
    steady: tuple = (0.0, 0.0, 0.0)
    sigma: float = 0.0
    tau: float = 12.5
    gust: tuple | None = None
    seed: int = 0

    def reset(self):
        self.rng = np.random.default_rng(self.seed)
        self.turb = np.zeros(3)

    def step(self, dt):
        if self.sigma > 0:
            s = np.array([self.sigma, self.sigma, 0.5 * self.sigma])
            a = math.exp(-dt / self.tau)
            self.turb = a * self.turb + math.sqrt(1 - a * a) * s * self.rng.standard_normal(3)

    def at(self, t) -> np.ndarray:
        w = np.asarray(self.steady, float) + self.turb
        if self.gust is not None:
            amp, t0, dur, direction = self.gust
            if t0 <= t <= t0 + dur:
                w = w + amp * 0.5 * (1 - math.cos(2 * math.pi * (t - t0) / dur)) * np.asarray(direction, float)
        return w


# ================================================================================================= actuators
@dataclass
class Actuators:
    """The servos as a rate limit and a first-order lag on each deflection (travel ``travel_deg``, ``rate_deg_s``
    from the servo's 0.10 s per 60°, lag ``tau`` for the linkage and the servo loop), the throttle as a first-order
    lag (ESC response and the propeller's spin-up, ``tau_throttle``)."""
    travel_deg: float = 25.0
    rate_deg_s: float = 600.0
    tau: float = 0.03
    tau_throttle: float = 0.15

    def reset(self):
        self.delta = np.zeros(3)       # aileron, elevator, rudder [rad]
        self.throttle = 0.0

    def step(self, cmd_delta, cmd_throttle, dt):
        lim = math.radians(self.travel_deg)
        target = np.clip(np.asarray(cmd_delta, float), -lim, lim)
        a = 1 - math.exp(-dt / self.tau)
        want = self.delta + a * (target - self.delta)
        step = math.radians(self.rate_deg_s) * dt
        self.delta = self.delta + np.clip(want - self.delta, -step, step)
        at = 1 - math.exp(-dt / self.tau_throttle)
        self.throttle += at * (float(np.clip(cmd_throttle, 0.0, 1.0)) - self.throttle)
        return self.delta, self.throttle


# ================================================================================================= the forces
LAB_OPTIONS = dict(timestep=0.002, control_dt=0.01, log_dt=0.1, log_geoms=True, flat_as_plane=True)


class Aero:
    """Scene hook: the aerodynamic forces and moments, the thrust, the actuators and the battery (see the module).
    The controller writes ``command`` (aileron, elevator, rudder [rad], throttle 0..1) and may change
    ``electronics_w`` (the phase's electronics power, battery side); ``history`` holds one row per ``log_every``
    control steps (``COLUMNS``)."""

    COLUMNS = ["t", "V", "alpha_deg", "beta_deg", "h", "roll_deg", "pitch_deg", "yaw_deg", "p", "q", "r", "n_z", "throttle", "thrust_N",
               "P_el_W", "E_used_Wh", "aileron_deg", "elevator_deg", "rudder_deg", "wind_x", "wind_y", "wind_z", "stalled", "P_electronics_W",
               "x", "y", "vx", "vy", "vz", "CL", "CD", "margin_wh"]

    def __init__(self, robot: Robot, coeff: dict, pm: ns.PropulsionMap, battery: ns.Battery, *, wind: Wind | None = None,
                 actuators: Actuators | None = None, electronics_w: float = 0.0, derating: float = 0.90, rho: float = RHO, log_every: int = 2,
                 post_stall: bool = True):
        self.robot, self.c, self.pm, self.battery = robot, dict(coeff), pm, battery
        self.wind = wind or Wind()
        self.act = actuators or Actuators()
        self.electronics_w, self.derating, self.rho, self.log_every, self.post_stall = float(electronics_w), derating, rho, log_every, post_stall
        p, L = robot.params, robot.layout
        self.S, self.b, self.cbar = self.c["S"], self.c["b"], self.c["c"]
        self.downthrust = math.radians(p["motor_downthrust_deg"])
        self.motor_pos = np.array(to_body(L["prop_x"], 0.0, p["motor_z"]))
        self.E_available_Wh = battery.energy_wh * derating
        self.E_prelaunch_Wh = 0.0

    # ---- lifecycle
    def reset(self, lab):
        import mujoco
        self.lab = lab
        assert lab.model.opt.density == 0.0 and lab.model.opt.viscosity == 0.0, "MuJoCo's fluid model must be off: Aero is the only aerodynamics"
        self.root = lab._body_id(self.robot.root.name)
        self.qv = int(lab.model.jnt_dofadr[lab._root_jnt])
        self.com_offset = lab.model.body_ipos[self.root].copy()          # the body's COM in its frame
        self.mass = float(lab.model.body_subtreemass[self.root])
        self.command = np.zeros(4)
        self.act.reset()
        self.wind.reset()
        self.energy_J = 3600.0 * self.E_prelaunch_Wh
        self.n, self.history, self.last = 0, [], {}
        self.hold = True                                                  # the thrower's hand holds the aircraft still until the launch
        self.margin_wh = float("nan")
        self.qadr = int(lab._root_qadr)
        self.hold_pose = lab.data.qpos[self.qadr:self.qadr + 7].copy()

    def add_energy_wh(self, wh: float):
        """Energy used outside the simulated time (the ground operation before launch)."""
        self.energy_J += 3600.0 * wh

    @property
    def E_used_Wh(self):
        return self.energy_J / 3600.0

    @property
    def E_remaining_Wh(self):
        return self.E_available_Wh - self.E_used_Wh

    # ---- the aerodynamic model
    def coefficients(self, alpha, beta, p_, q_, r_, delta, V):
        c = self.c
        da, de, dr = delta
        ph, qh, rh = p_ * self.b / (2 * V), q_ * self.cbar / (2 * V), r_ * self.b / (2 * V)
        CL_lin = c["CL0"] + c["CLa"] * alpha + c["CLq"] * qh + c["CLde"] * de
        CD_lin = c["CD0"] + c["k"] * (c["CL0"] + c["CLa"] * alpha) ** 2 + c["CD_de"] * (de * de + dr * dr + 0.5 * da * da)
        Cm_lin = c["Cm0"] + c["Cma"] * alpha + c["Cmq"] * qh + c["Cmde"] * de
        a_s = math.radians(c["alpha_stall_deg"])
        stalled = False
        if self.post_stall:
            # a smooth blend to a flat plate beyond the stall angle (assumption: CN = 2 sin α, a nose-down Cm of -0.15)
            s = 1 / (1 + math.exp(-(abs(alpha) - a_s) / math.radians(1.5)))
            stalled = s > 0.5
            CN = 2.0 * math.sin(alpha) * (1 if alpha >= 0 else 1)
            CL_fp = CN * math.cos(alpha) * 0.9 + c["CLde"] * de * 0.5
            CD_fp = c["CD0"] + abs(CN * math.sin(alpha)) + 0.1
            Cm_fp = -0.15 * math.copysign(1.0, alpha) + c["Cmde"] * de * 0.5 + c["Cmq"] * qh
            CL = (1 - s) * CL_lin + s * CL_fp
            CD = (1 - s) * CD_lin + s * CD_fp
            Cm = (1 - s) * Cm_lin + s * Cm_fp
        else:
            CL, CD, Cm = CL_lin, CD_lin, Cm_lin
        CLn = max(min(CL, 1.5), -1.0)                                  # the lift-dependent lateral terms follow the actual lift
        CY = c["Cyb"] * beta + c["Cyp"] * ph + c["Cyr"] * rh + c["Cydr"] * dr
        Cl = c["Clb"] * beta + c["Clp"] * ph + (CLn / 4 + (c["Clr"] - 0.1)) * rh + c["Clda"] * da + c["Cldr"] * dr
        Cn = c["Cnb"] * beta + (-CLn / 8) * ph + c["Cnr"] * rh + (-0.2 * CLn * c["Clda"]) * da + c["Cndr"] * dr
        return CL, CD, CY, Cl, Cm, Cn, stalled

    def __call__(self, lab):
        d = lab.data
        dt = lab.control_dt
        t = lab.time
        self.wind.step(dt)
        if self.hold:                                                     # held in the hand: no motion, no aerodynamic force
            d.qpos[self.qadr:self.qadr + 7] = self.hold_pose
            d.qvel[self.qv:self.qv + 6] = 0.0
            self.act.step(self.command[:3], self.command[3], dt)
        R = d.xmat[self.root].reshape(3, 3)                           # body → world
        pos = d.xpos[self.root]
        v_w = d.qvel[self.qv:self.qv + 3].copy()                      # COM velocity, world
        w_b = d.qvel[self.qv + 3:self.qv + 6].copy()                  # angular velocity, body frame (x fwd, y left, z up)
        wind = self.wind.at(t)
        v_air_b = R.T @ (v_w - wind)
        u, v_right, w_down = v_air_b[0], -v_air_b[1], -v_air_b[2]      # aero body axes
        V = float(np.linalg.norm(v_air_b))
        delta, thr = self.act.step(self.command[:3], self.command[3], dt)
        F_w = np.zeros(3); M_w = np.zeros(3)
        stalled = False
        CL = CD = 0.0
        alpha = beta = 0.0
        p_, q_, r_ = w_b[0], -w_b[1], -w_b[2]
        if V > 1.0:
            alpha = math.atan2(w_down, u)
            beta = math.asin(max(-1.0, min(1.0, v_right / V)))
            CL, CD, CY, Cl, Cm, Cn, stalled = self.coefficients(alpha, beta, p_, q_, r_, delta, V)
            qS = 0.5 * self.rho * V * V * self.S
            lift_dir = np.array([math.sin(alpha), 0.0, -math.cos(alpha)])          # aero body axes, perpendicular to the wind in the symmetry plane
            drag_dir = -np.array([u, v_right, w_down]) / V
            F_a = qS * (CL * lift_dir + CD * drag_dir + CY * np.array([0.0, 1.0, 0.0]))
            M_a = qS * np.array([Cl * self.b, Cm * self.cbar, Cn * self.b])
            F_b = np.array([F_a[0], -F_a[1], -F_a[2]])                            # aero → Chiron body
            M_b = np.array([M_a[0], -M_a[1], -M_a[2]])
            F_w, M_w = R @ F_b, R @ M_b
        # thrust along the tilted motor axis at the motor's position
        prop = self.pm.at(max(u, 0.0), thr) if thr > 1e-3 else {"thrust": 0.0, "electrical": 0.0, "rpm": 0.0}
        T = prop["thrust"]
        t_b = np.array([math.cos(self.downthrust), 0.0, -math.sin(self.downthrust)])   # forward and a little down (Chiron z up)
        T_b = T * t_b
        r_b = self.motor_pos - self.com_offset
        F_w = F_w + R @ T_b
        M_w = M_w + R @ np.cross(r_b, T_b)
        lab.body_force(self.robot.root.name, F_w, M_w)
        P = prop["electrical"] + self.electronics_w
        self.energy_J += P * dt
        roll, pitch, yaw = euler_from_R(R)
        n_z = float((R.T @ F_w)[2] / (self.mass * G))                   # the load factor: aerodynamic + thrust force along the body's z, / weight
        self.last = {"V": V, "alpha": alpha, "beta": beta, "h": float(pos[2]), "roll": roll, "pitch": pitch, "yaw": yaw, "n_z": n_z, "thrust": T,
                     "P": P, "pos": pos.copy(), "v": v_w, "wind": wind, "stalled": stalled, "throttle": thr, "delta": delta.copy(), "u": u, "R": R,
                     "omega_b": w_b, "CL": CL, "CD": CD}
        if self.n % self.log_every == 0:
            self.history.append([t, V, math.degrees(alpha), math.degrees(beta), pos[2], math.degrees(roll), math.degrees(pitch), math.degrees(yaw),
                                 p_, q_, r_, n_z, thr, T, P, self.E_used_Wh, *np.degrees(delta), *wind, float(stalled), self.electronics_w,
                                 pos[0], pos[1], *v_w, CL, CD, self.margin_wh])
        self.n += 1

    def frame(self):
        import pandas as pd
        return pd.DataFrame(np.asarray(self.history, float).reshape(-1, len(self.COLUMNS)), columns=self.COLUMNS)
