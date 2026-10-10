"""NISUS+ in ChironLab (MuJoCo) over the mountains: the airframe as a Chiron robot, the terrain, the mountain wind and the
aerodynamic, propulsion and battery forces as a scene hook that knows the altitude (notebook 33, ``nisus_plus_scenario``).

* **the airframe** — NISUS's recipe (``nisus_robot``): one rigid body (``Link('nisus_plus')``) carrying every row of
  ``nisus_plus_systems.mass_table`` (the structure as the shapes' masses, the components as point masses), the skid and the
  two tail bumpers as its feet; ``inertia_check`` compares MuJoCo's mass, CG and inertia with the table.
* **the mountains** (``Massif``): an analytic valley (the meadow at 1200 m, the floor rising gently east) between a
  north ridge (crest ~3900-4200 m, its south face the survey slope) and a lower south ridge, with some texture. The
  same function is the analyses' terrain, the controller's terrain database (the onboard DEM: perfect here — a real
  one is a 30 m grid with its errors) and MuJoCo's height field (``as_chiron``: 25 m cells over 10 x 9 km). World frame:
  x east, y north, z the altitude above sea level [m] — the ISA reads it directly.
* **the wind** (``MountainWind``): NISUS's Ornstein–Uhlenbeck turbulence and discrete gust (``nisus_robot.Wind``) plus
  what the mountains do to a synoptic wind ``steady`` — a logarithmic shear over the height above the ground, the
  **slope wind** ``w = (U · ∇h) e^(−agl/H)`` (the air follows the terrain: lift on the windward face, sink on the
  lee), the **lee** amplified (``lee_factor``, the separated flow behind a crest sinks harder) with stronger
  turbulence (``lee_sigma``). A reduced model, honest about it: no rotor dynamics, no thermals, no valley winds — it
  makes the controller meet lift and sink where a pilot would expect them.
* **the forces** (``NisusPlusAero``, a subclass of NISUS's ``Aero``): every control step the air's density from the ISA at
  the aircraft's altitude (and the scenario's temperature offset), the coefficients of ``nisus_plus_flight.derivatives``
  with the **crow** increments (ΔCL, ΔCD, ΔCm, k, ΔCL_max: the flap channel 0..1), the drive from
  ``nisus_plus_systems.NisusPlusDrive`` at that density and at the pack's **loaded voltage** (``LiIonPack``: the state of charge
  and the temperature) — throttle, freewheel or the **propeller brake** (the brake channel 0..1) with the
  **regeneration** charging the pack within its limits. Energy: the battery-side power (the drive's, negative when it
  charges, plus the electronics) integrated.

Units SI; the Chiron body frame is x forward, y left, z up (the aerodynamic axes are converted inside the hook).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from vegeta import chiron as ch
from vegeta.chiron import FootSpec, Geom, Link, PointMass, Robot

import nisus_plus
import nisus_plus_systems as fs
import nisus_robot as nr
from nisus_robot import to_body, quat_to_R, euler_from_R

G = 9.81
BODY = "nisus_plus"

__all__ = ["G", "BODY", "Massif", "nisus_plus_robot", "inertia_check", "MountainWind", "NisusPlusActuators", "NisusPlusAero", "lab_options", "to_body",
           "quat_to_R", "euler_from_R"]


# ================================================================================================= the mountains
def _smooth(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


@dataclass(frozen=True)
class Massif:
    """The analytic mountain valley (see the module). Heights [m ASL] at (x east, y north) [m]."""
    meadow: float = 1200.0             # the valley floor at home (0, 0)
    floor_rise: float = 0.03           # the floor rises east (m per m)
    north_y0: float = 600.0            # the north face's foot
    north_crest_y: float = 4000.0
    north_crest: float = 3900.0
    north_peak: float = 400.0          # a summit on the crest (Gaussian, at north_peak_x)
    north_peak_x: float = 3000.0
    south_y0: float = -600.0
    south_crest_y: float = -3200.0
    south_crest: float = 2900.0
    texture: float = 25.0              # ripples [m] on the slopes
    extent: tuple = (-3000.0, 7000.0, -4000.0, 5500.0)
    cell: float = 25.0

    def floor(self, x):
        return self.meadow + self.floor_rise * np.maximum(np.asarray(x, float), 0.0)

    def height(self, x, y):
        x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
        fl = self.floor(x)
        crest_n = self.north_crest + self.north_peak * np.exp(-((x - self.north_peak_x) / 2200.0) ** 2)
        un = (y - self.north_y0) / (self.north_crest_y - self.north_y0)
        north = (crest_n - fl) * _smooth(un)
        beyond = y > self.north_crest_y                                   # the far side of the crest falls away
        north = np.where(beyond, (crest_n - fl) * (1 - 0.6 * _smooth((y - self.north_crest_y) / 1500.0)), north)
        us = (y - self.south_y0) / (self.south_crest_y - self.south_y0)
        south = (self.south_crest - fl) * _smooth(us)
        slope = np.maximum(_smooth(un), _smooth(us))
        tex = self.texture * slope * np.sin(x / 310.0 + 0.7 * np.sin(y / 450.0)) * np.cos(y / 270.0)
        h = fl + np.maximum(north, south) + tex
        return float(h) if h.ndim == 0 else h

    def grad(self, x, y, d=10.0):
        return np.array([(self.height(x + d, y) - self.height(x - d, y)) / (2 * d), (self.height(x, y + d) - self.height(x, y - d)) / (2 * d)])

    def max_along(self, xy, direction, length, n=12, width=60.0) -> float:
        """The highest terrain within ``length`` ahead of ``xy`` along ``direction`` (and ±``width`` to the sides)."""
        u = np.asarray(direction, float)
        nu = np.linalg.norm(u)
        u = u / nu if nu > 1e-6 else np.array([1.0, 0.0])
        nrm = np.array([-u[1], u[0]])
        s = np.linspace(0.0, length, n)
        pts = np.asarray(xy, float)[None] + s[:, None] * u[None]
        hs = [self.height(pts[:, 0] + k * width * nrm[0], pts[:, 1] + k * width * nrm[1]) for k in (-1, 0, 1)]
        return float(np.max(hs))

    def as_chiron(self):
        """The terrain as a Chiron ``Custom`` height field (with ``extent`` for the lab's ``course_extent``)."""
        t = ch.Custom(lambda x, y: self.height(x, y), name="nisus_plus massif")
        t.extent = self.extent
        return t


def lab_options(massif: Massif = Massif(), **kw) -> dict:
    opts = dict(timestep=0.002, control_dt=0.01, log_dt=0.1, log_geoms=True, flat_as_plane=False, course_extent=massif.extent,
                heightfield_cell=massif.cell)
    opts.update(kw)
    return opts


# ================================================================================================= the robot
def _wing_x(table) -> float:
    w = table[table.index.str.startswith("wing:")]
    return float((w["mass [g]"] * w["x [mm]"]).sum() / w["mass [g]"].sum())


def nisus_plus_robot(battery_key: str = fs.DEFAULT_PACK, p=None, *, table=None) -> Robot:
    """The aircraft as a Chiron ``Robot`` (NISUS's ``nisus_robot`` on NISUS+; root body ``nisus_plus``)."""
    p = nisus_plus.resolve(p)
    L = nisus_plus.NisusPlus.layout(p)
    t = fs.mass_table(battery_key, p=p) if table is None else table
    pod_rgba, foam, carbon, dark = (0.86, 0.87, 0.9, 1.0), (0.95, 0.66, 0.2, 1.0), (0.12, 0.12, 0.14, 1.0), (0.1, 0.1, 0.1, 1.0)
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
    shell = t[t.index.str.startswith("pod:") & t.index.str.contains("shell")]
    x_pod = float(shell["x [mm]"].iloc[0]) if len(shell) else L["x_nose"] + p["pod_length"] / 2
    geoms = [
        Geom("pod", "ellipsoid", (p["pod_length"] / 2000, p["pod_width"] / 2000, p["pod_height"] / 2000), pos=to_body(x_pod, 0, L["pod_axis_z"]),
             mass=groups["pod"], role="body", rgba=pod_rgba),
        Geom("wing", "box", (0.5 * (c0 + p["tip_chord"]) / 2000, b / 2000, 0.016), pos=to_body(_wing_x(t), 0, 10.0), mass=groups["wing"], role="body", rgba=foam),
        Geom("stab", "box", (p["tail_chord"] / 2000, p["tail_span"] / 2000, 0.005), pos=to_body(L["tail_le"] + p["tail_chord"] / 2, 0, p["boom_z"]),
             mass=0.4 * groups["tail"], role="body", rgba=foam),
        Geom("motor", "cylinder", (p["motor_diameter"] / 2000, p["motor_length"] / 2000), pos=to_body(0.5 * (L["x_motor0"] + L["x_motor1"]), 0, p["motor_z"]),
             quat=(math.cos(math.pi / 4), 0, math.sin(math.pi / 4), 0), mass=None, role="visual", rgba=dark),
        Geom("prop_disc", "cylinder", (L["prop_R"] / 1000, 0.0008), pos=to_body(L["prop_x"], 0, p["motor_z"]),
             quat=(math.cos(math.pi / 4), 0, math.sin(math.pi / 4), 0), mass=None, role="visual", rgba=(0.6, 0.75, 0.95, 0.30)),
        Geom("skid", "box", ((p["skid_x1"] - p["skid_x0"]) / 2000, p["skid_width"] / 2000, p["skid_depth"] / 2000),
             pos=to_body(0.5 * (p["skid_x0"] + p["skid_x1"]), 0, -p["pod_height"] - p["skid_depth"] / 2), mass=None, role="foot",
             friction=(0.5, 0.005, 0.0001), rgba=(0.9, 0.4, 0.1, 1.0)),
        Geom("nose_tip", "sphere", (0.03,), pos=to_body(L["x_nose"] + 30, 0, L["pod_axis_z"]), mass=None, role="body", rgba=pod_rgba),
    ]
    for side, s in (("L", -1), ("R", 1)):
        geoms.append(Geom(f"boom_{side}", "capsule", (p["boom_od"] / 2000,),
                          fromto=(*to_body(p["boom_x0"], s * p["boom_y"], p["boom_z"]), *to_body(L["boom_x1"], s * p["boom_y"], p["boom_z"])),
                          mass=groups["booms"] / 2, role="body", rgba=carbon))
        geoms.append(Geom(f"fin_{side}", "box", (p["fin_chord"] / 2000, 0.005, (p["fin_height"] + p["fin_ventral"]) / 2000),
                          pos=to_body(L["fin_le"] + p["fin_chord"] / 2, s * p["boom_y"], p["boom_z"] + 0.5 * (p["fin_height"] - p["fin_ventral"])),
                          mass=0.3 * groups["tail"], role="body", rgba=foam))
        geoms.append(Geom(f"stub_{side}", "capsule", (0.006,), fromto=(*to_body(L["fin_le"] + 15, s * p["boom_y"], p["boom_z"] - p["fin_ventral"]),
                                                                       *to_body(L["boom_x1"] - 15, s * p["boom_y"], p["boom_z"] - p["fin_ventral"])),
                          mass=None, role="foot", friction=(0.5, 0.005, 0.0001), rgba=dark))
    feet = [FootSpec("skid", "skid", [], BODY), FootSpec("stub_L", "stub_L", [], BODY), FootSpec("stub_R", "stub_R", [], BODY)]
    root = Link(BODY, geoms=geoms, masses=masses, log=True)
    robot = Robot("Nisus+ Zero", root=root, feet=feet,
                  notes=f"Nisus+ Zero on the {battery_key} pack: nisus_plus_systems.mass_table as geom and point masses; no joints (the control surfaces "
                        "and the crow flaps are states of the NisusPlusAero hook)",
                  sources={"geometry": "designs/nisus_plus.py", "masses": "designs/nisus_plus_systems.py: mass_table", "aerodynamics": "designs/nisus_plus_flight.py: derivatives",
                           "propulsion": "designs/nisus_plus_systems.py: NisusPlusDrive (Boreas BEMT, the fitted AT4125 KV540)"})
    robot.validate()
    robot.params, robot.battery_key, robot.mass_table, robot.layout = p, battery_key, t, L
    return robot


def inertia_check(lab, table) -> dict:
    """MuJoCo's mass, COM and inertia against ``nisus_plus_systems.cg_inertia`` of the same table."""
    m, d = lab.model, lab.data
    b = lab._body_id(BODY)
    mass = float(m.body_subtreemass[b])
    hand = fs.cg_inertia(table)
    R = quat_to_R(m.body_iquat[b])
    I = R @ np.diag(m.body_inertia[b]) @ R.T
    com_body = m.body_ipos[b]
    return {"mujoco_mass_kg": mass, "table_mass_kg": hand["mass_kg"], "mass_error": mass / hand["mass_kg"] - 1,
            "mujoco_cg_x_mm": -com_body[0] * 1000, "table_cg_x_mm": hand["x_cg_m"] * 1000, "mujoco_cg_z_mm": com_body[2] * 1000, "table_cg_z_mm": hand["z_cg_m"] * 1000,
            "Ixx_ratio": float(I[0, 0]) / hand["Ixx"], "Iyy_ratio": float(I[1, 1]) / hand["Iyy"], "Izz_ratio": float(I[2, 2]) / hand["Izz"]}


# ================================================================================================= the mountain wind
@dataclass
class MountainWind(nr.Wind):
    """NISUS's wind (``steady``, OU turbulence ``sigma``, a discrete gust) over the ``massif`` (see the module):
    ``at(t, pos)`` adds the shear over the height above the ground, the slope wind and the lee's sink and turbulence.
    ``steady`` is the wind well above the ground (the synoptic wind)."""
    massif: Massif = field(default_factory=Massif)
    z0: float = 0.1                     # roughness length [m] (grass, rock): the log-law shear
    slope_depth: float = 300.0          # H: the slope wind decays over this height above the ground
    lee_factor: float = 1.6             # the lee's sink stronger than the windward lift (separation)
    lee_sigma: float = 2.5              # turbulence multiplier in the lee, near the ground
    max_vertical: float = 8.0           # the slope wind is capped [m/s]

    def at(self, t, pos=None) -> np.ndarray:
        base = super().at(t)
        if pos is None:
            return base
        U = np.asarray(self.steady, float)
        if np.linalg.norm(U[:2]) < 1e-6 and self.sigma <= 0:
            return base
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        agl = max(z - float(self.massif.height(x, y)), 0.5)
        shear = min(max(math.log(agl / self.z0) / math.log(300.0 / self.z0), 0.2), 1.0)
        g = self.massif.grad(x, y)
        decay = math.exp(-agl / self.slope_depth)
        w = float(U[:2] @ g) * decay
        lee = w < 0
        if lee:
            w *= self.lee_factor
        w = max(-self.max_vertical, min(self.max_vertical, w))
        turb = self.turb * (1.0 + (self.lee_sigma - 1.0) * decay if lee else 1.0)
        out = np.array([U[0] * shear, U[1] * shear, U[2] + w]) + turb
        if self.gust is not None:
            out = out + (base - U - self.turb)
        return out

    def vertical(self, x, y, agl) -> float:
        """The slope wind's vertical component at (x, y) and a height above the ground (no turbulence): for maps."""
        U = np.asarray(self.steady, float)
        g = self.massif.grad(x, y)
        w = float(U[:2] @ g) * math.exp(-max(agl, 0.5) / self.slope_depth)
        w = w * self.lee_factor if w < 0 else w
        return max(-self.max_vertical, min(self.max_vertical, w))


# ================================================================================================= actuators
@dataclass
class NisusPlusActuators(nr.Actuators):
    """NISUS's servos (aileron, elevator, rudder), the throttle's lag, plus the crow channel (0..1 of the crow setting:
    the flap servos under load, ``crow_time`` for the full travel) and the brake channel (0..1, the ESC's response)."""
    crow_time: float = 1.2
    tau_brake: float = 0.25

    def reset(self):
        super().reset()
        self.crow = 0.0
        self.brake = 0.0

    def step_extra(self, cmd_crow, cmd_brake, dt):
        rate = dt / self.crow_time
        self.crow += float(np.clip(np.clip(cmd_crow, 0.0, 1.0) - self.crow, -rate, rate))
        self.brake += (1 - math.exp(-dt / self.tau_brake)) * (float(np.clip(cmd_brake, 0.0, 1.0)) - self.brake)
        return self.crow, self.brake


# ================================================================================================= the forces
class NisusPlusAero(nr.Aero):
    """NISUS's scene hook over the mountains (see the module). The controller writes ``command`` (aileron, elevator,
    rudder [rad], throttle 0..1, crow 0..1, brake 0..1; a throttle ≤ ``idle`` with no brake freewheels the propeller)."""

    COLUMNS = nr.Aero.COLUMNS + ["z_asl", "agl", "rho", "sigma", "EAS", "crow", "brake", "rpm", "P_prop_W", "V_batt", "soc", "regen_Wh", "pack_T_C", "w_air"]

    def __init__(self, robot: Robot, coeff: dict, drive: fs.NisusPlusDrive, pack: fs.LiIonPack, massif: Massif, *, wind: MountainWind | None = None,
                 actuators: NisusPlusActuators | None = None, electronics_w: float = 0.0, derating: float = 0.90, dT: float = 0.0,
                 pack_T_C: float = 15.0, log_every: int = 10, post_stall: bool = True, idle: float = 0.02):
        super().__init__(robot, coeff, None, pack, wind=wind or MountainWind(massif=massif), actuators=actuators or NisusPlusActuators(),
                         electronics_w=electronics_w, derating=derating, rho=fs.RHO0, log_every=log_every, post_stall=post_stall)
        self.drive, self.pack, self.massif, self.dT, self.pack_T_C, self.idle = drive, pack, massif, dT, pack_T_C, idle
        self.E_nominal_Wh = pack.energy_wh
        self.E_available_Wh = min(pack.usable_wh(pack_T_C, 20.0), pack.energy_wh) * derating

    def reset(self, lab):
        super().reset(lab)
        self.command = np.zeros(6)
        self.regen_J = 0.0
        self.v_batt = float(self.pack.ocv(1.0))
        self.charge_As = 0.0

    @property
    def soc(self):
        return max(0.0, 1.0 - self.charge_As / 3600.0 / (self.pack.capacity_ah * self.pack.cell.capacity_factor(self.pack_T_C)))

    def air(self, z):
        a = fs.atmosphere(float(z), self.dT)
        return a["rho"], a["sigma"]

    def coefficients(self, alpha, beta, p_, q_, r_, delta, V, crow=0.0):
        c = self.c
        da, de, dr = delta
        ph, qh, rh = p_ * self.b / (2 * V), q_ * self.cbar / (2 * V), r_ * self.b / (2 * V)
        CL_lin = c["CL0"] + c["CLa"] * alpha + c["CLq"] * qh + c["CLde"] * de + crow * c["dCL_crow"]
        k = c["k"] + crow * (c["k_crow"] - c["k"])
        CD_lin = c["CD0"] + crow * c["dCD_crow"] + k * (c["CL0"] + c["CLa"] * alpha + crow * c["dCL_crow"]) ** 2 + c["CD_de"] * (de * de + dr * dr + 0.5 * da * da)
        Cm_lin = c["Cm0"] + c["Cma"] * alpha + c["Cmq"] * qh + c["Cmde"] * de + crow * c["dCm_crow"]
        a_s = math.radians(c["alpha_stall_deg"]) + crow * c["dCLmax_crow"] / c["CLa"]
        stalled = False
        if self.post_stall:
            s = 1 / (1 + math.exp(-(abs(alpha) - a_s) / math.radians(1.5)))
            stalled = s > 0.5
            CN = 2.0 * math.sin(alpha)
            CL_fp = CN * math.cos(alpha) * 0.9 + c["CLde"] * de * 0.5
            CD_fp = c["CD0"] + crow * c["dCD_crow"] + abs(CN * math.sin(alpha)) + 0.1
            Cm_fp = -0.15 * math.copysign(1.0, alpha) + c["Cmde"] * de * 0.5 + c["Cmq"] * qh
            CL, CD, Cm = (1 - s) * CL_lin + s * CL_fp, (1 - s) * CD_lin + s * CD_fp, (1 - s) * Cm_lin + s * Cm_fp
        else:
            CL, CD, Cm = CL_lin, CD_lin, Cm_lin
        CLn = max(min(CL, 1.5), -1.0)
        CY = c["Cyb"] * beta + c["Cyp"] * ph + c["Cyr"] * rh + c["Cydr"] * dr
        Cl = c["Clb"] * beta + c["Clp"] * ph + (CLn / 4 + (c["Clr"] - 0.1)) * rh + c["Clda"] * da + c["Cldr"] * dr
        Cn = c["Cnb"] * beta + (-CLn / 8) * ph + c["Cnr"] * rh + (-0.2 * CLn * c["Clda"]) * da + c["Cndr"] * dr
        return CL, CD, CY, Cl, Cm, Cn, stalled

    def _propulsor(self, u, thr, brake, rho):
        i_chg = self.pack.charge_limit_a(self.pack_T_C, self.soc)
        dr = self.drive
        V = max(u, 0.0)
        if brake > 0.01:
            return dr.brake(V, brake, rho, self.v_batt, i_charge_max=i_chg)
        if thr <= self.idle:
            return dr.freewheel(V, rho)
        return dr.at(V, thr, rho, self.v_batt, i_charge_max=i_chg)

    def __call__(self, lab):
        d = lab.data
        dt = lab.control_dt
        t = lab.time
        self.wind.step(dt)
        if self.hold:
            d.qpos[self.qadr:self.qadr + 7] = self.hold_pose
            d.qvel[self.qv:self.qv + 6] = 0.0
        R = d.xmat[self.root].reshape(3, 3)
        pos = d.xpos[self.root].copy()
        rho, sigma = self.air(pos[2])
        v_w = d.qvel[self.qv:self.qv + 3].copy()
        w_b = d.qvel[self.qv + 3:self.qv + 6].copy()
        wind = self.wind.at(t, pos)
        v_air_b = R.T @ (v_w - wind)
        u, v_right, w_down = v_air_b[0], -v_air_b[1], -v_air_b[2]
        V = float(np.linalg.norm(v_air_b))
        cmd = self.command
        delta, thr = self.act.step(cmd[:3], cmd[3], dt)
        crow, brake = self.act.step_extra(cmd[4], cmd[5], dt)
        F_w = np.zeros(3); M_w = np.zeros(3)
        stalled = False
        CL = CD = alpha = beta = 0.0
        p_, q_, r_ = w_b[0], -w_b[1], -w_b[2]
        if V > 1.0:
            alpha = math.atan2(w_down, u)
            beta = math.asin(max(-1.0, min(1.0, v_right / V)))
            CL, CD, CY, Cl, Cm, Cn, stalled = self.coefficients(alpha, beta, p_, q_, r_, delta, V, crow)
            qS = 0.5 * rho * V * V * self.S
            lift_dir = np.array([math.sin(alpha), 0.0, -math.cos(alpha)])
            drag_dir = -np.array([u, v_right, w_down]) / V
            F_a = qS * (CL * lift_dir + CD * drag_dir + CY * np.array([0.0, 1.0, 0.0]))
            M_a = qS * np.array([Cl * self.b, Cm * self.cbar, Cn * self.b])
            F_w, M_w = R @ np.array([F_a[0], -F_a[1], -F_a[2]]), R @ np.array([M_a[0], -M_a[1], -M_a[2]])
        if self.hold:
            prop = {"thrust": 0.0, "electrical": 0.0, "rpm": 0.0, "current": 0.0}
        else:
            prop = self._propulsor(u, thr, brake, rho)
        T = prop["thrust"]
        t_b = np.array([math.cos(self.downthrust), 0.0, -math.sin(self.downthrust)])
        T_b = T * t_b
        F_w = F_w + R @ T_b
        M_w = M_w + R @ np.cross(self.motor_pos - self.com_offset, T_b)
        lab.body_force(self.robot.root.name, F_w, M_w)
        P = prop["electrical"] + self.electronics_w
        self.energy_J += P * dt
        if prop["electrical"] < 0:
            self.regen_J += -prop["electrical"] * dt
        I_b = P / max(self.v_batt, 1.0)
        self.charge_As += I_b * dt
        self.v_batt = float(self.pack.terminal_v(I_b, self.soc, self.pack_T_C))
        roll, pitch, yaw = euler_from_R(R)
        n_z = float((R.T @ F_w)[2] / (self.mass * G))
        agl = float(pos[2] - self.massif.height(pos[0], pos[1]))
        self.last = {"V": V, "EAS": V * math.sqrt(sigma), "rho": rho, "sigma": sigma, "alpha": alpha, "beta": beta, "h": float(pos[2]), "agl": agl,
                     "roll": roll, "pitch": pitch, "yaw": yaw, "n_z": n_z, "thrust": T, "P": P, "pos": pos, "v": v_w, "wind": wind, "stalled": stalled,
                     "throttle": thr, "crow": crow, "brake": brake, "delta": delta.copy(), "u": u, "R": R, "omega_b": w_b, "CL": CL, "CD": CD,
                     "v_batt": self.v_batt, "soc": self.soc}
        if self.n % self.log_every == 0:
            self.history.append([t, V, math.degrees(alpha), math.degrees(beta), pos[2], math.degrees(roll), math.degrees(pitch), math.degrees(yaw),
                                 p_, q_, r_, n_z, thr, T, P, self.E_used_Wh, *np.degrees(delta), *wind, float(stalled), self.electronics_w,
                                 pos[0], pos[1], *v_w, CL, CD, self.margin_wh,
                                 pos[2], agl, rho, sigma, V * math.sqrt(sigma), crow, brake, prop.get("rpm", 0.0), prop["electrical"], self.v_batt,
                                 self.soc, self.regen_J / 3600, self.pack_T_C, wind[2]])
        self.n += 1

    @property
    def regen_Wh(self):
        return self.regen_J / 3600.0
