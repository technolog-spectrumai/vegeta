"""Drako's cargo runs in MuJoCo (ChironLab): from the valley helipad up to a mountain hut and back, then into the
village, in calm air or in wind.

**The map** (``Scene``; x east, y north, z up, metres). A valley floor at z = 0 with the **helipad** at the origin; a
mountain (a 900 m Gaussian peak) north-east whose shoulder carries the **hut** on a levelled pad at about 550 m,
2.7 km from the helipad; the **village** 1.7 km south-west on a low hill (about 70 m), its landing square between
houses. Every pad is levelled over 25 m, blended into the slope by 45 m, and carries a 30 x 30 m concrete slab
(``SLAB``: totes and skids stand on box contacts, which MuJoCo resolves without the jitter a box shows on the height field). MuJoCo sees the terrain as a height field
every 10 m (``CELL_M``); the pads, houses, the hut and trees are scenery boxes and cylinders.

**The packages** (``Scene.packages``): totes of 600 x 400 x 350 mm waiting in a row east of each pad's centre (Drako hops over to one, and delivers on the centre) — medical supplies (10 kg) at the
helipad for the hut, samples (4 kg) at the hut for the valley, groceries (8 kg) at the helipad for the village. Each is
a free body. Drako lands over one (it stands between the skids, under the cradle), closes the latches and the
``Cargo`` hook welds it to the frame at the pose it has then (the hooks under the tote's lip; the latch geoms are drawn
only); opening the latches releases the weld and the tote stands on the pad.

**The air** (``Air``, a hook): six rotors with a spin-up lag; each rotor's thrust limit falls with the air density
(ISA, ``rho(z)``); the rotors' electrical power from Drako's Boreas system (``drako_robot.system``: U12 II KV120 +
30x10.5, a thrust → power table at sea level, scaled by sqrt(rho0 / rho)); quadratic drag on the frame (CdA
0.25 m^2) and on a latched tote (0.15 m^2) against the **wind** (``Wind``: a mean wind at 10 m above the ground with a
power-law profile, a speed-up over the mountain, and gusts as first-order filtered noise). ``energy_Wh`` and the
battery's state of charge; a battery swap at the helipad between legs (``Plan.swap_battery_at``).

    import drako_scenario as ds
    scene = ds.Scene(wind=ds.Wind(8.0, from_deg=270, gust=0.25))
    lab = ds.make_lab(scene)
    ep = ds.run(lab, scene)
    ds.legs(ep), ds.deliveries(ep), ds.plot_mission(ep, scene, "mission.png")
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from vegeta import chiron as ch

import drako_controller as dcn
import drako_robot as dk
from ogma import rotors

RHO0, G = 1.225, 9.81
CELL_M = 10.0
SLAB = (15.0, 0.05)                              # m: each pad a concrete slab, half-side and thickness, on the levelled ground
CDA_BODY, CDA_TOTE = 0.25, 0.15                  # m^2: Drako.sizing's cda_body, cda_package (drako_cfd measures them)
TOTE = (0.60, 0.40, 0.35)                        # m: the tote Drako's cradle is drawn for
#: 4 ms physics, the flight controller at 83 Hz; elliptic friction cones with a high impedance ratio and the no-slip
#: solver: a waiting tote stays put (pyramidal cones let it creep ~9 mm a minute, enough to miss the latches).
LAB_OPTIONS = dict(timestep=0.004, control_dt=0.012, log_dt=0.2, cone="elliptic", impratio=10.0, noslip_iterations=4,
                   integrator="implicitfast")


def rho(z):
    """ISA air density [kg/m^3] at altitude z [m] above the valley (taken as sea level)."""
    return RHO0 * (1 - 2.25577e-5 * np.asarray(z, float)) ** 4.2559


# --------------------------------------------------------------------------------------------- the wind
@dataclass
class Wind:
    """Mean wind ``speed`` [m/s] at 10 m above the ground, blowing **from** ``from_deg`` (meteorological: 270 = from
    the west), growing with height above the ground as (h / 10)^``shear`` up to ``shear_top_m`` (the boundary layer); ``speedup`` the extra fraction over the
    mountain's top (a Gaussian about the peak, as hill speed-up); ``gust`` the turbulence intensity (sigma / mean)
    with a ``gust_time_s`` correlation time; ``seed`` fixes the gusts."""

    speed: float = 0.0
    from_deg: float = 270.0
    shear: float = 0.14
    shear_top_m: float = 150.0
    speedup: float = 0.4
    gust: float = 0.2
    gust_time_s: float = 3.0
    seed: int = 0

    def __post_init__(self):
        a = math.radians(self.from_deg)
        self.dir = -np.array([math.sin(a), math.cos(a), 0.0])           # the direction the air moves to
        self.reset()

    def reset(self):
        self.rng = np.random.default_rng(self.seed)
        self.g = np.zeros(3)
        self.t = None

    @property
    def calm(self) -> bool:
        return self.speed <= 0.0

    def at(self, p, t, ground_z, peak=None) -> np.ndarray:
        """The air's velocity [m/s, world] at ``p`` and time ``t`` (``ground_z`` the terrain under ``p``)."""
        if self.calm:
            return np.zeros(3)
        h = min(max(float(p[2]) - float(ground_z), 1.0), self.shear_top_m)
        u = self.speed * (h / 10.0) ** self.shear
        if peak is not None:
            d2 = (p[0] - peak[0]) ** 2 + (p[1] - peak[1]) ** 2
            u *= 1.0 + self.speedup * math.exp(-d2 / (2 * peak[2] ** 2))
        dt = 0.0 if self.t is None else max(float(t) - self.t, 0.0)
        self.t = float(t)
        if dt > 0:                                                         # first-order (Ornstein-Uhlenbeck) gusts
            a = math.exp(-dt / self.gust_time_s)
            sig = self.gust * u * np.array([1.0, 0.8, 0.5])
            self.g = a * self.g + math.sqrt(1 - a * a) * sig * self.rng.standard_normal(3)
        return u * self.dir + self.g


# --------------------------------------------------------------------------------------------- the map
@dataclass
class Scene:
    """The map and the packages (see the module). ``pads``: name → (x, y); their heights are the levelled terrain's.
    ``packages``: name → {mass_kg, at (pad), to (pad), rgba}."""

    wind: Wind = field(default_factory=Wind)
    pads: dict = field(default_factory=lambda: {"helipad": (0.0, 0.0), "hut": (2400.0, 1250.0), "village": (-1500.0, -700.0)})
    peak: tuple = (3000.0, 1650.0, 900.0, 700.0)              # x, y, height, sigma [m]
    hill: tuple = (-1750.0, -850.0, 80.0, 600.0)
    pad_flat_m: float = 25.0
    pad_blend_m: float = 45.0
    extent: tuple = (-2300.0, 3900.0, -1600.0, 2700.0)
    packages: dict = field(default_factory=lambda: {
        "medical": {"mass_kg": 10.0, "at": "helipad", "to": "hut", "rgba": (0.85, 0.15, 0.15, 1.0)},
        "samples": {"mass_kg": 4.0, "at": "hut", "to": "helipad", "rgba": (0.15, 0.45, 0.85, 1.0)},
        "groceries": {"mass_kg": 8.0, "at": "helipad", "to": "village", "rgba": (0.20, 0.65, 0.25, 1.0)}})
    spare_spot_m: float = 6.0                                   # totes wait in a row east of the pad's centre

    def __post_init__(self):
        self._pad_z = {k: self._raw(np.array(x), np.array(y)) for k, (x, y) in self.pads.items()}
        seen = {}
        self.package_xy = {}
        for name, pk in self.packages.items():
            n = seen.get(pk["at"], 0)
            seen[pk["at"]] = n + 1
            x, y = self.pads[pk["at"]]
            self.package_xy[name] = (x + (n + 1) * self.spare_spot_m, y)    # the pad's centre is where Drako delivers

    def _raw(self, x, y):
        px, py, ph, ps = self.peak
        hx, hy, hh, hs = self.hill
        return (ph * np.exp(-((x - px) ** 2 + (y - py) ** 2) / (2 * ps * ps))
                + hh * np.exp(-((x - hx) ** 2 + (y - hy) ** 2) / (2 * hs * hs)))

    def height(self, x, y):
        """The terrain [m]: the mountain and the hill, every pad levelled at its centre's height."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        z = self._raw(x, y)
        for k, (cx, cy) in self.pads.items():
            r = np.hypot(x - cx, y - cy)
            w = np.clip((self.pad_blend_m - r) / (self.pad_blend_m - self.pad_flat_m), 0.0, 1.0)
            w = w * w * (3 - 2 * w)
            z = z * (1 - w) + self._pad_z[k] * w
        return z

    def pad(self, name) -> tuple:
        """The pad's centre on its slab's top."""
        x, y = self.pads[name]
        return (x, y, float(self._pad_z[name]) + SLAB[1])

    def surface(self, x, y) -> float:
        """What stands under (x, y): a pad's slab top, or the terrain."""
        for k, (cx, cy) in self.pads.items():
            if abs(x - cx) <= SLAB[0] and abs(y - cy) <= SLAB[0]:
                return float(self._pad_z[k]) + SLAB[1]
        return float(self.height(x, y))

    def spot(self, package) -> tuple:
        """Where a package waits (and where Drako lands to latch it)."""
        x, y = self.package_xy[package]
        return (x, y, self.surface(x, y))

    def terrain(self) -> ch.Custom:
        return ch.Custom(self.height, name="drako valley, mountain and village")


def _tote(name, x, y, z, mass, rgba) -> ch.Prop:
    lx, ly, lz = TOTE
    w = 0.012
    geoms = [ch.Geom(f"{name}_body", "box", (lx / 2 - 0.01, ly / 2 - 0.01, lz / 2), mass=mass, rgba=rgba),
             ch.Geom(f"{name}_lip", "box", (lx / 2, ly / 2, w / 2), pos=(0, 0, lz / 2 - w / 2), role="visual",
                     rgba=(0.15, 0.15, 0.15, 1.0))]
    return ch.Prop(ch.Link(name, pos=(x, y, z + lz / 2 + 0.002), geoms=geoms), free=True, condim=4)


def scenery(scene: Scene) -> list:
    """Pads (discs with an H), the hut, the village's houses round its square, a few trees; fixed, drawn and colliding."""
    out = []
    for k, name in enumerate(scene.pads):
        x, y, z = scene.pad(name)                                       # z: the slab's top
        a, t = SLAB
        g = [ch.Geom(f"{name}_slab", "box", (a, a, t / 2), pos=(0, 0, -t / 2), rgba=(0.62, 0.62, 0.60, 1.0)),
             ch.Geom(f"{name}_ring", "cylinder", (6.0, 0.002), pos=(0, 0, 0.002), role="visual", rgba=(0.30, 0.30, 0.32, 1.0)),
             ch.Geom(f"{name}_H1", "box", (0.3, 2.0, 0.002), pos=(-1.2, 0, 0.005), role="visual", rgba=(1, 1, 1, 1)),
             ch.Geom(f"{name}_H2", "box", (0.3, 2.0, 0.002), pos=(1.2, 0, 0.005), role="visual", rgba=(1, 1, 1, 1)),
             ch.Geom(f"{name}_H3", "box", (1.2, 0.3, 0.002), pos=(0, 0, 0.005), role="visual", rgba=(1, 1, 1, 1))]
        out.append(ch.Prop(ch.Link(f"{name}_site", pos=(x, y, z), geoms=g)))
    hx, hy, hz = scene.pad("hut")
    out.append(ch.Prop(ch.Link("hut", pos=(hx - 15, hy + 12, hz), geoms=[
        ch.Geom("hut_walls", "box", (5.0, 4.0, 2.5), pos=(0, 0, 2.5), rgba=(0.55, 0.38, 0.22, 1.0)),
        ch.Geom("hut_roof", "box", (5.5, 4.5, 0.4), pos=(0, 0, 5.4), rgba=(0.35, 0.12, 0.10, 1.0))])))
    vx, vy, _ = scene.pad("village")
    for k, (dx, dy) in enumerate([(-18, 14), (0, 20), (18, 14), (-20, -12), (20, -14), (2, -22)]):
        x, y = vx + dx, vy + dy
        z = float(scene.height(x, y))
        out.append(ch.Prop(ch.Link(f"house{k}", pos=(x, y, z), geoms=[
            ch.Geom(f"house{k}_walls", "box", (4.0, 3.5, 3.0), pos=(0, 0, 3.0), rgba=(0.90, 0.86, 0.75, 1.0)),
            ch.Geom(f"house{k}_roof", "box", (4.4, 3.9, 0.5), pos=(0, 0, 6.5), rgba=(0.65, 0.20, 0.12, 1.0))])))
    rng = np.random.default_rng(3)
    for k in range(10):
        x, y = rng.uniform(-60, 60), rng.uniform(15, 60)
        z = float(scene.height(x, y))
        out.append(ch.Prop(ch.Link(f"tree{k}", pos=(x, y, z), geoms=[
            ch.Geom(f"tree{k}_trunk", "cylinder", (0.3, 2.0), pos=(0, 0, 2.0), rgba=(0.35, 0.25, 0.15, 1.0)),
            ch.Geom(f"tree{k}_crown", "sphere", (2.5,), pos=(0, 0, 6.0), rgba=(0.15, 0.40, 0.15, 1.0))])))
    return out


def package_props(scene: Scene) -> list:
    return [_tote(n, *scene.spot(n), pk["mass_kg"], pk["rgba"]) for n, pk in scene.packages.items()]


def welds(scene: Scene) -> list:
    return [ch.Weld(f"latch_{n}", n, "drako", active=False) for n in scene.packages]


# --------------------------------------------------------------------------------------------- the hooks
class Air:
    """Rotors, drag and wind, power and energy (see the module). ``command``: the six thrusts the controller wants [N]."""

    def __init__(self, scene: Scene, *, log_every: int = 10):
        self.scene, self.log_every = scene, log_every
        ft = dk.full_throttle()
        self.T_max, self.kM = ft["thrust_N"], ft["kM"]
        pos, spin = dk.rotor_layout()
        self.pos, self.spin = np.asarray(pos, float), np.asarray(spin, float)
        sys_ = dk.system()
        Ts = np.linspace(0.05, 1.0, 24) * self.T_max
        self.P_T = Ts
        self.P_W = np.array([sys_.for_thrust(T).electrical_power for T in Ts])
        _, _, _, battery = dk.boreas_system()
        self.usable_Wh = battery.voltage * battery.capacity_ah * battery.usable_fraction
        self.tau = dk.SPIN_UP_S

    def power_W(self, T, rho_) -> float:
        T = np.clip(np.asarray(T, float), 0.0, None)
        P = np.interp(T, np.r_[0.0, self.P_T], np.r_[0.0, self.P_W])
        return float(P.sum() * math.sqrt(RHO0 / rho_))

    def swap_battery(self, t):
        self.swaps.append((float(t), self.energy_J / 3600 - self.swap_base_Wh))
        self.swap_base_Wh = self.energy_J / 3600

    def reset(self, lab):
        self.lab = lab
        self.root = lab._body_id("drako")
        self.qv = int(lab.model.jnt_dofadr[lab._root_jnt])
        self.command = np.zeros(6)
        self.thrust = np.zeros(6)
        self.energy_J, self.swap_base_Wh, self.swaps, self.n, self.history = 0.0, 0.0, [], 0, []
        self.min_soc = 1.0
        self.scene.wind.reset()
        self.alpha = 1.0 - math.exp(-lab.control_dt / self.tau)
        p = self.scene.peak
        self.peak = (p[0], p[1], p[3])

    @property
    def energy_Wh(self) -> float:
        return self.energy_J / 3600.0

    @property
    def soc(self) -> float:
        """State of charge of the battery in use (usable energy)."""
        return 1.0 - (self.energy_Wh - self.swap_base_Wh) / self.usable_Wh

    def __call__(self, lab):
        d = lab.data
        p = d.xpos[self.root].copy()
        r = float(rho(p[2]))
        t_lim = self.T_max * r / RHO0
        cmd = np.clip(self.command, 0.0, t_lim)
        self.thrust += (cmd - self.thrust) * self.alpha
        R = d.xmat[self.root].reshape(3, 3)
        F, M = rotors.wrench(self.thrust, R, self.pos, self.spin, self.kM, com_offset=d.xipos[self.root] - d.xpos[self.root])
        gz = float(self.scene.height(p[0], p[1]))
        w = self.scene.wind.at(p, lab.time, gz, self.peak)
        v = d.qvel[self.qv:self.qv + 3]
        F = F + rotors.quadratic_drag(v - w, CDA_BODY, r)
        lab.body_force("drako", F, M)
        for name in lab.cargo.latched_names():                            # welded: the tote moves with the frame
            lab.body_force(name, rotors.quadratic_drag(v - w, CDA_TOTE, r), None)
        P = self.power_W(self.thrust, r)
        self.energy_J += P * lab.control_dt
        self.min_soc = min(self.min_soc, self.soc)
        if self.n % self.log_every == 0:
            self.history.append([lab.time, *p, *v, *w, *self.thrust, P, self.soc, r])
        self.n += 1


class Cargo:
    """The packages: latching (a weld at the current relative pose), releasing, and where each one ends up."""

    def __init__(self, scene: Scene):
        self.scene = scene

    def reset(self, lab):
        self.lab = lab
        self.ids = {n: lab._body_id(n) for n in self.scene.packages}
        self.records = {n: {"latched": None, "released": None, "released_at": None, "delivered": None, "to": pk["to"],
                            "mass_kg": pk["mass_kg"], "how": ""} for n, pk in self.scene.packages.items()}
        self._latched = set()

    def latched_names(self):
        return list(self._latched)

    def _rel(self, n):
        import mujoco
        d = self.lab.data
        b1, b2 = self.ids[n], self.lab._body_id("drako")
        R1 = d.xmat[b1].reshape(3, 3)
        dp = R1.T @ (d.xpos[b2] - d.xpos[b1])
        q1inv = np.zeros(4)
        mujoco.mju_negQuat(q1inv, d.xquat[b1])
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, q1inv, d.xquat[b2])
        return dp, q

    def latch(self, n, t) -> bool:
        """Weld package ``n`` to the frame if it stands under the cradle: within 0.15 m of the frame's axis and its
        top within 0.12 m of the cradle."""
        d = self.lab.data
        pb, pd = d.xpos[self.ids[n]], d.xpos[self.lab._body_id("drako")]
        leg = dk.design_params()["leg_height"] / 1000
        top_gap = (pd[2] - leg + TOTE[2]) - (pb[2] + TOTE[2] / 2)
        if math.hypot(*(pb[:2] - pd[:2])) > 0.15 or abs(top_gap) > 0.12:
            return False
        dp, q = self._rel(n)
        e = self.lab._weld_id[f"latch_{n}"]
        self.lab.model.eq_data[e, 0:3] = 0.0
        self.lab.model.eq_data[e, 3:6] = dp
        self.lab.model.eq_data[e, 6:10] = q
        self.lab.set_weld(f"latch_{n}", True)
        self._latched.add(n)
        self.records[n]["latched"] = float(t)
        return True

    def unlatch(self, n, t):
        if n in self._latched:
            self.lab.set_weld(f"latch_{n}", False)
            self._latched.discard(n)
            self.lab.body_force(n, None)
            r = self.records[n]
            r["released"] = float(t)
            r["released_at"] = self.lab.data.xpos[self.ids[n]].copy().tolist()
            to = self.scene.pad(r["to"])
            dist = math.hypot(r["released_at"][0] - to[0], r["released_at"][1] - to[1])
            if dist < self.scene.pad_flat_m:
                r["delivered"], r["how"] = float(t), f"at the {r['to']}, {dist:.1f} m from the pad's centre"
            else:
                r["how"] = f"released {dist:.0f} m from the {r['to']}"

    def __call__(self, lab):
        pass


def make_lab(scene: Scene, robot=None, **kwargs):
    """Drako on the map with its packages and the scenery; hooks ``lab.air`` and ``lab.cargo``."""
    robot = robot or dk.drako(latch_contact=False)
    opts = dict(LAB_OPTIONS, course_extent=scene.extent, heightfield_cell=CELL_M)
    opts.update(kwargs)
    lab = ch.ChironLab(robot, scene.terrain(), props=package_props(scene) + scenery(scene), welds=welds(scene), **opts)
    lab.air = Air(scene)
    lab.cargo = Cargo(scene)
    lab.add_hook(lab.air)
    lab.add_hook(lab.cargo)
    return lab


def run(lab, scene: Scene, plan: dcn.Plan | None = None, duration: float = 2400.0, seed: int = 0, flight_kw=None):
    """The whole route of ``plan``; stops when the mission is finished."""
    plan = plan or dcn.Plan()
    mission = dcn.Mission(dcn.mission(scene, plan), dcn.Flight(**(flight_kw or {})),
                          name=f"Drako cargo, wind {scene.wind.speed:g} m/s", plan=plan)
    lab.scene = scene
    start = scene.pad(plan.route[0][0])
    base = (start[0], start[1], start[2] + lab.nominal_base_height + 0.005)     # on the slab, not the terrain under it
    ep = lab.run(mission, duration=duration, rules=None, settle=1.0, seed=seed, base_pos=base,
                 info={"controller": mission.name, "treatment": f"wind {scene.wind.speed:g}"},
                 stop_when=lambda lab_: mission.finished)
    ep.log["mission"] = [list(x) for x in mission.log]
    ep.log["mission_finished"] = mission.finished and not mission.aborted
    ep.log["aborted"] = mission.aborted
    ep.log["records"] = {k: dict(v) for k, v in lab.cargo.records.items()}
    ep.log["air_history"] = np.asarray(lab.air.history, float).reshape(-1, 19)
    ep.log["energy_Wh"] = lab.air.energy_Wh
    ep.log["swaps"] = list(lab.air.swaps)
    ep.log["min_soc"] = lab.air.min_soc
    ep.log["wind"] = dict(speed=scene.wind.speed, from_deg=scene.wind.from_deg, gust=scene.wind.gust)
    ep.outcome = outcome(ep)
    return ep


def outcome(ep) -> dict:
    recs = ep.log["records"]
    missing = [k for k, r in recs.items() if r["delivered"] is None]
    ok = ep.log["mission_finished"] and not missing and ep.log["min_soc"] > 0.0
    why = "delivered" if ok else ("battery empty" if ep.log["min_soc"] <= 0 else
                                  ep.log["aborted"] if ep.log["aborted"] else
                                  ("not delivered: " + ", ".join(missing) if missing else "mission not finished"))
    return {"success": bool(ok), "reason": why, "energy_Wh": ep.log["energy_Wh"], "min_soc": ep.log["min_soc"],
            "t_end": float(ep.log["air_history"][-1, 0]) if len(ep.log["air_history"]) else float("nan")}


# --------------------------------------------------------------------------------------------- tables
COLS = ["t", "x", "y", "z", "vx", "vy", "vz", "wx", "wy", "wz", "T1", "T2", "T3", "T4", "T5", "T6", "P_W", "soc", "rho"]


def timeseries(ep) -> pd.DataFrame:
    return pd.DataFrame(ep.log["air_history"][:, :19], columns=COLS)


def legs(ep) -> pd.DataFrame:
    """One row per leg (take off from a pad → spool down on the next): time, distance flown, energy, peak power,
    lowest charge, peak wind, the largest rotor thrust as a share of full throttle."""
    ts = timeseries(ep)
    rows, cur = [], None
    t_max = dk.full_throttle()["thrust_N"]
    for t, n, note in ep.log["mission"]:
        if note == "start" and n.startswith("take off from the "):
            cur = (t, n[len("take off from the "):])
        elif note == "done" and cur and (n.startswith("spool down on the ") or n == "emergency spool down"):
            a, frm = cur
            to = n[len("spool down on the "):] if n.startswith("spool down on the ") else "emergency landing"
            cur = None
            s = ts[(ts.t >= a) & (ts.t <= t)]
            if len(s) < 2:
                continue
            xyz = s[["x", "y", "z"]].to_numpy()
            dt = np.diff(s.t.to_numpy(), prepend=s.t.iloc[0])
            wind = np.linalg.norm(s[["wx", "wy", "wz"]].to_numpy(), axis=1)
            rows.append({"leg": f"{frm} -> {to}", "time [min]": (t - a) / 60,
                         "distance [km]": float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()) / 1000,
                         "climb [m]": float(xyz[:, 2].max() - xyz[0, 2]), "energy [Wh]": float((s.P_W * dt).sum() / 3600),
                         "peak power [kW]": float(s.P_W.max()) / 1000, "lowest charge [%]": 100 * float(s.soc.min()),
                         "peak wind [m/s]": float(wind.max()),
                         "max thrust share": float(s[["T1", "T2", "T3", "T4", "T5", "T6"]].to_numpy().max() / t_max)})
    return pd.DataFrame(rows)


def deliveries(ep) -> pd.DataFrame:
    return pd.DataFrame(ep.log["records"]).T[["mass_kg", "to", "latched", "released", "delivered", "how"]]


def plot_mission(ep, scene: Scene, path, title=None):
    """The route over the map (contours), the height profile and the charge against time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ts = timeseries(ep)
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(1, 2, 1)
    x0, x1, y0, y1 = scene.extent
    X, Y = np.meshgrid(np.linspace(x0, x1, 300), np.linspace(y0, y1, 220))
    cs = ax.contourf(X, Y, scene.height(X, Y), levels=18, cmap="terrain")
    fig.colorbar(cs, ax=ax, label="terrain [m]")
    sc = ax.scatter(ts.x, ts.y, c=ts.t / 60, s=2, cmap="plasma")
    fig.colorbar(sc, ax=ax, label="time [min]")
    for n in scene.pads:
        px, py, _ = scene.pad(n)
        ax.plot(px, py, "ks", ms=7)
        ax.annotate(n, (px, py), xytext=(8, 8), textcoords="offset points", fontsize=11, weight="bold")
    if not scene.wind.calm:
        cx, cy = x0 + 0.12 * (x1 - x0), y1 - 0.12 * (y1 - y0)
        ax.annotate("", xy=(cx + 500 * scene.wind.dir[0], cy + 500 * scene.wind.dir[1]), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle="-|>", lw=3, color="navy"))
        ax.text(cx, cy - 220, f"wind {scene.wind.speed:g} m/s", color="navy", ha="center")
    ax.set_aspect("equal"); ax.set_xlabel("x east [m]"); ax.set_ylabel("y north [m]")
    ax.set_title(title or ep.log["mission"][0][1])
    a2 = fig.add_subplot(4, 2, 2)
    gz = scene.height(ts.x.to_numpy(), ts.y.to_numpy())
    a2.fill_between(ts.t / 60, 0, gz, color="#b59b6d", alpha=0.6, label="terrain under Drako")
    a2.plot(ts.t / 60, ts.z, "k", lw=1, label="Drako")
    a2.set_ylabel("altitude [m]"); a2.legend(loc="upper right")
    a3 = fig.add_subplot(4, 2, 4, sharex=a2)
    a3.plot(ts.t / 60, ts.P_W / 1000, lw=0.6)
    a3.set_ylabel("power [kW]")
    a4 = fig.add_subplot(4, 2, 6, sharex=a2)
    a4.plot(ts.t / 60, np.linalg.norm(ts[["wx", "wy", "wz"]].to_numpy(), axis=1), "b", lw=0.5)
    a4.set_ylabel("wind at Drako [m/s]")
    a5 = fig.add_subplot(4, 2, 8, sharex=a2)
    a5.plot(ts.t / 60, 100 * ts.soc, "g")
    a5.set_ylabel("charge [%]"); a5.set_xlabel("time [min]"); a5.set_ylim(0, 105)
    o = ep.outcome
    fig.suptitle(f"{'OK' if o['success'] else 'FAILED'}: {o['reason']}; energy {o['energy_Wh']:.0f} Wh, "
                 f"lowest charge {100 * o['min_soc']:.0f} %, {o['t_end'] / 60:.1f} min")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path
