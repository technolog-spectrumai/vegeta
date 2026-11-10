"""The world every Pinna launch clip shares: the carrier on dynamic positioning in a simple sea, the wind, the water,
and the simulator that steps it with the drones (the agents: ``gannet_sim``, ``fulmar_sim``, ``walrus_sim``).

* **the sea** (``Sea``): the carrier's heave, roll and pitch as sums of sines (a long swell and a shorter sea; Hs
  about 1.2 m, inside Gannet's 2 m deck limit). The carrier is a MuJoCo body on three joints (heave slide, roll and
  pitch hinges) whose position *and velocity* are prescribed every physics step, so a drone on its deck feels the
  deck's motion through the contacts and the welds (a mocap body would move without a velocity). The water surface is
  a translucent slab that heaves with the swell (the carrier rides the swell; the local wave is not modelled).
* **the wind** (``Wind``): a mean wind from the bow (the carrier heads into it on DP) and gusts as a first-order
  (Ornstein-Uhlenbeck) process per axis, seeded; the drones' aerodynamic forces use the air's velocity.
* **the carrier** (``carrier_xml``): Pinna's CAD parts as visual meshes (``meshes``), the surfaces drones touch as
  simple collision boxes and cylinders (flight deck, pads, main deck, hangar, house, hulls, the mast pole), the
  catapult rail, the recovery net (a panel on a sprung slide joint between two masts) and the four davits (one, the
  aft port davit, has a luffing boom the Walrus clip drives). The directional light casts no shadows (MuJoCo's shadow
  map draws its frustum's edge on a large sea).
* **the simulator** (``Sim``): builds the MJCF from the carrier and the agents, then each physics step prescribes the
  carrier and the agents' robot joints, lets each agent apply its forces (``xfrc_applied``: rotors, wings, thrusters,
  buoyancy) and steps MuJoCo. ``Sim.run(t_end, every, callback)`` calls back on a fixed time grid (the video frames).

Frame: x forward (the bow), y to port, z up from the mean waterline; metres, seconds, newtons.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import meshes

G = 9.81
RHO_AIR, RHO_SW = 1.225, 1025.0


def _pinna_geometry() -> dict:
    """Where things are on Pinna (m), from its design's parameters and CAD layout (``pinna.Pinna``)."""
    from pinna import Pinna
    d, p = Pinna(), Pinna.p()
    z_deck = d._z_deck(p) / 1000                         # main deck top
    z_fd = z_deck + 3.6 + 0.15                           # flight deck top
    return {"length": p["length"] / 1000, "beam": p["beam"] / 1000, "hull_beam": p["hull_beam"] / 1000, "draught": p["draught"] / 1000,
            "z_deck": z_deck, "z_fd": z_fd, "z_pad": z_fd + 0.04,
            "pads": [(x / 1000, y / 1000) for x, y in d._pad_at(p)],
            "net_x": -9.1, "net_masts_y": 3.2, "net_bottom": z_fd + 1.0, "net_size": (6.0, 4.0),
            "rail_x": (-3.7, 3.3), "rail_top": z_fd + 0.5 + 0.125, "ram_x0": -3.6,
            "mast_x": 8.2, "mast_top": z_deck + 7.8 + 0.3,
            "davits": [(x, s * (p["beam"] / 2000 - 0.45)) for x in (-9.5, -11.5) for s in (1, -1)]}


GEO = None


def geometry() -> dict:
    global GEO
    if GEO is None:
        GEO = _pinna_geometry()
    return GEO


#: The catapult's yaw (deg, to port, about the rail's aft end). The CAD's rail lies on the centreline and points the
#: Fulmar at the autonomy mast's pole 8.4 m ahead of the rail end, at the height it leaves the rail: no climb clears
#: it, and at 12-15 deg the starboard wing still meets the radar bar (+/-1.6 m). The sim yaws the rail 18 deg so
#: the climb-out passes outside the radar bar (a finding for the CAD: move the mast off the centreline or yaw the rail).
CATAPULT_YAW_DEG = 18.0


# ------------------------------------------------------------------------------------------------ rotations
def quat_to_R(q) -> np.ndarray:
    R = np.zeros(9)
    import mujoco
    mujoco.mju_quat2Mat(R, np.asarray(q, dtype=float))
    return R.reshape(3, 3)


def R_to_quat(R) -> np.ndarray:
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=float).reshape(9))
    return q


def euler_zyx(R) -> tuple:
    """(roll, pitch, yaw) of a rotation in the x-forward, y-left, z-up frame: R = Rz(yaw) Ry(pitch) Rx(roll); pitch
    positive nose up here (the conventional sense), roll positive starboard wing down."""
    yaw = math.atan2(R[1, 0], R[0, 0])
    pitch = -math.asin(max(-1.0, min(1.0, R[2, 0])))
    roll = math.atan2(R[2, 1], R[2, 2])
    return roll, -pitch, yaw


def min_jerk(t, t0, t1, a, b):
    """Position, velocity, acceleration of a minimum-jerk move from ``a`` at ``t0`` to ``b`` at ``t1`` (arrays)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if t <= t0:
        return a.copy(), np.zeros_like(a), np.zeros_like(a)
    if t >= t1:
        return b.copy(), np.zeros_like(a), np.zeros_like(a)
    T = t1 - t0
    s = (t - t0) / T
    p = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
    v = (30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4) / T
    acc = (60 * s - 180 * s ** 2 + 120 * s ** 3) / T ** 2
    return a + (b - a) * p, (b - a) * v, (b - a) * acc


# ------------------------------------------------------------------------------------------------ sea and wind
@dataclass
class Sea:
    """The carrier's motion on DP: (amplitude, period s) per component; heave in m, roll and pitch in degrees. All
    sines start at zero so the carrier is at its rest pose at t = 0 (where the welds of the locked drones are made)."""

    heave: tuple = ((0.40, 7.6), (0.12, 4.7))
    roll: tuple = ((2.5, 6.4), (0.7, 3.9))
    pitch: tuple = ((1.1, 5.3), (0.4, 3.4))

    @staticmethod
    def _s(comp, t, deg):
        k = math.pi / 180 if deg else 1.0
        x = sum(a * k * math.sin(2 * math.pi * t / T) for a, T in comp)
        v = sum(a * k * 2 * math.pi / T * math.cos(2 * math.pi * t / T) for a, T in comp)
        return x, v

    def state(self, t) -> tuple:
        """(heave, roll, pitch) and their rates at t (m, rad)."""
        (z, zd), (r, rd), (p, pd) = self._s(self.heave, t, False), self._s(self.roll, t, True), self._s(self.pitch, t, True)
        return np.array([z, r, p]), np.array([zd, rd, pd])

    def eta(self, t) -> float:
        """The water surface's elevation (the swell the carrier rides)."""
        return self._s(self.heave, t, False)[0]

    def describe(self) -> str:
        h = sum(a for a, _ in self.heave)
        return f"heave +/-{h:.2f} m, roll +/-{sum(a for a, _ in self.roll):.1f} deg, pitch +/-{sum(a for a, _ in self.pitch):.1f} deg"


@dataclass
class Wind:
    """A mean wind (m/s, the air's velocity: from the bow it is (-U, 0, 0)) and gusts per axis (OU, ``tau`` s)."""

    mean: tuple = (-6.0, 0.0, 0.0)
    sigma: tuple = (1.2, 0.9, 0.3)
    tau: float = 3.0
    seed: int = 7
    _g: np.ndarray = field(default_factory=lambda: np.zeros(3), repr=False)

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    def update(self, dt: float):
        a = math.exp(-dt / self.tau)
        self._g = a * self._g + np.sqrt(1 - a * a) * np.asarray(self.sigma) * self._rng.standard_normal(3)

    def at(self, z: float = 10.0) -> np.ndarray:
        """The air's velocity at height z (a 1/7 power law from its 10 m value, none at the sea)."""
        k = (max(z, 0.5) / 10.0) ** (1 / 7)
        return (np.asarray(self.mean) + self._g) * k

    def describe(self) -> str:
        return f"wind {np.linalg.norm(self.mean):.0f} m/s from the bow, gusts {self.sigma[0]:.1f} m/s"


# ------------------------------------------------------------------------------------------------ MJCF
def _f(v) -> str:
    return " ".join(f"{x:.6g}" for x in np.atleast_1d(v))


def mesh_assets(use_meshes: bool) -> tuple:
    """The <asset> mesh lines and the paths (empty when the meshes are off)."""
    if not use_meshes:
        return "", {}
    paths = meshes.export_all()
    lines = "".join(f'<mesh name="{k}" file="{p}" scale="0.001 0.001 0.001" inertia="shell"/>\n' for k, p in paths.items())
    return lines, paths


def _vis_mesh(name, rgba, pos=(0, 0, 0), quat=(1, 0, 0, 0)) -> str:
    return f'<geom name="v_{name}" type="mesh" mesh="{name}" class="visual" rgba="{_f(rgba)}" pos="{_f(pos)}" quat="{_f(quat)}"/>\n'


def carrier_xml(use_meshes: bool, extra: str = "") -> str:
    """The carrier body (joints, inertia, meshes, collision shapes, net, davits, catapult) with ``extra`` children."""
    g = geometry()
    col = meshes.colours()
    zd, zf = g["z_deck"], g["z_fd"]
    s = ['<body name="carrier" pos="0 0 0">',
         '<joint name="c_heave" type="slide" axis="0 0 1" damping="0"/>',
         '<joint name="c_roll" type="hinge" axis="1 0 0"/>',
         '<joint name="c_pitch" type="hinge" axis="0 1 0"/>',
         '<inertial pos="0 0 1.5" mass="60000" diaginertia="6e5 2.9e6 3.2e6"/>']
    if use_meshes:
        for part in meshes.PINNA_PARTS:
            if part in ("net", "catapult"):                     # drawn below: the net panel moves, the rail is yawed
                continue
            s.append(_vis_mesh(f"pinna_{part}", col[f"pinna_{part}"]))
    else:
        s.append(f'<geom class="visual" type="box" size="11 5 0.2" pos="-0.96 0 {zd - 0.2:.3f}" rgba="0.58 0.65 0.65 1"/>')
        s.append(f'<geom class="visual" type="box" size="6.25 4.8 0.075" pos="-3 0 {zf - 0.075:.3f}" rgba="0.45 0.5 0.52 1"/>')
        for y in (3.6, -3.6):
            s.append(f'<geom class="visual" type="box" size="10.8 1.4 1.6" pos="-1.2 {y} 0.1" rgba="0.2 0.29 0.37 1"/>')
    # what the drones touch (not drawn: group 3)
    s.append(f'<geom name="c_flightdeck" class="hull" type="box" size="6.25 4.8 0.075" pos="-3 0 {zf - 0.075:.4f}"/>')
    for k, (x, y) in enumerate(g["pads"]):
        s.append(f'<geom name="c_pad{k}" class="hull" type="cylinder" size="1.3 0.02" pos="{x} {y} {zf + 0.02:.4f}"/>')
    s.append(f'<geom name="c_deck" class="hull" type="box" size="11.04 5 0.2" pos="-0.96 0 {zd - 0.2:.4f}"/>')
    s.append(f'<geom name="c_hangar" class="hull" type="box" size="6 4.6 1.8" pos="-3 0 {zd + 1.8:.4f}"/>')
    s.append(f'<geom name="c_house" class="hull" type="box" size="3.5 3.8 1.3" pos="3.5 0 {zd + 1.3:.4f}"/>')
    for y in (3.6, -3.6):
        s.append(f'<geom name="c_hull{"p" if y > 0 else "s"}" class="hull" type="box" size="10.8 1.4 1.6" pos="-1.2 {y} 0.1"/>')
    s.append(f'<geom name="c_mastpole" class="hull" type="cylinder" size="0.22 2.6" pos="{g["mast_x"]} 0 {zd + 2.6 + 2.6:.4f}"/>')
    s.append(f'<geom name="c_mastradar" class="hull" type="box" size="0.2 1.6 0.11" pos="{g["mast_x"]} 0 {zd + 6.4:.4f}"/>')
    # the catapult: the rail (yawed, CATAPULT_YAW_DEG), its legs and the ram, drawn and colliding
    yaw = math.radians(CATAPULT_YAW_DEG)
    x0, x1 = g["rail_x"]
    L = x1 - x0
    cx, cy = x0 + L / 2 * math.cos(yaw), L / 2 * math.sin(yaw)
    qy = (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))
    red = "0.75 0.22 0.17 1"
    s.append(f'<body name="catapult" pos="{x0} 0 {zf}" quat="{_f(qy)}">')
    s.append(f'<geom name="rail" type="box" size="{L / 2} 0.15 0.125" pos="{L / 2} 0 0.5" rgba="{red}" class="hullvis"/>')
    for xl in (0.3, 3.7, 6.7):
        s.append(f'<geom type="box" size="0.1 0.1 0.25" pos="{xl} 0 0.25" rgba="0.3 0.3 0.32 1" class="visual"/>')
    s.append(f'<geom type="cylinder" size="0.18 1.4" pos="1.5 0 0.3" quat="0.7071 0 0.7071 0" rgba="0.6 0.6 0.62 1" class="visual"/>')
    s.append('<body name="shuttle" pos="0.4 0 0.625"><joint name="shuttle_x" type="slide" axis="1 0 0"/>'
             '<inertial pos="0 0 0" mass="30" diaginertia="1 1 1"/>'
             '<geom type="box" size="0.35 0.2 0.04" pos="0 0 0.04" rgba="0.15 0.15 0.17 1" class="visual"/>'
             '<geom type="box" size="0.25 0.1 0.06" pos="0.1 0 0.14" rgba="0.25 0.25 0.27 1" class="visual"/>'
             '<geom type="box" size="0.05 0.25 0.12" pos="0.3 0 0.15" rgba="0.15 0.15 0.17 1" class="visual"/></body>')
    s.append('</body>')
    # the recovery net: two masts and a panel on a sprung slide (it gives forward when the Fulmar flies in)
    nx, ny, nb = g["net_x"], g["net_masts_y"], g["net_bottom"]
    w, h = g["net_size"]
    for y in (ny, -ny):
        s.append(f'<geom type="cylinder" size="0.11 2.6" pos="{nx} {y} {zf + 2.6:.4f}" rgba="0.3 0.32 0.35 1" class="hullvis"/>')
    s.append(f'<body name="net" pos="{nx} 0 {nb + h / 2:.4f}"><joint name="net_x" type="slide" axis="1 0 0" stiffness="700" damping="160" range="-0.05 2.4" limited="true"/>'
             f'<inertial pos="0 0 0" mass="4" diaginertia="5 1 5"/>'
             f'<geom name="net_panel" type="box" size="0.05 {w / 2} {h / 2}" rgba="0.1 0.12 0.15 0.25" contype="2" conaffinity="2" group="1"/>')
    for k in range(-6, 7):                                       # the mesh, drawn
        s.append(f'<geom type="box" size="0.012 0.012 {h / 2}" pos="0 {k * w / 12:.3f} 0" rgba="0.15 0.17 0.2 0.9" class="visual"/>')
    for k in range(-4, 5):
        s.append(f'<geom type="box" size="0.012 {w / 2} 0.012" pos="0 0 {k * h / 8:.3f}" rgba="0.15 0.17 0.2 0.9" class="visual"/>')
    s.append('</body>')
    # the davits: a post, a boom (the aft port one luffs: ``davit_boom``), a reel drum
    for k, (x, y) in enumerate(g["davits"]):
        side = 1 if y > 0 else -1
        s.append(f'<geom type="box" size="0.15 0.15 1.3" pos="{x} {y} {zd + 1.3:.4f}" rgba="0.9 0.49 0.13 1" class="hullvis"/>')
        # the tether reel: 0.8 m further inboard than the CAD's (1.8 m) so the Walrus's deck cradle fits under the boom
        s.append(f'<geom type="cylinder" size="0.45 0.4" pos="{x} {y - side * 2.6:.3f} {zd + 0.5:.4f}" quat="0.7071 0.7071 0 0" rgba="0.3 0.3 0.32 1" class="visual"/>')
        if (x, side) == (-11.5, 1):
            s.append(f'<body name="davit_boom" pos="{x} {y} {zd + 2.6:.4f}"><joint name="davit_luff" type="hinge" axis="-1 0 0"/>'
                     f'<inertial pos="0 0 0.9" mass="80" diaginertia="20 20 1"/>'
                     f'<geom type="box" size="0.12 0.12 0.95" pos="0 0 0.95" rgba="0.9 0.49 0.13 1" class="visual"/>'
                     f'<geom type="box" size="0.18 0.18 0.1" pos="0 0 1.9" rgba="0.25 0.25 0.27 1" class="visual"/>'
                     f'<site name="boom_tip" pos="0 0 1.85" size="0.03"/></body>')
        else:
            s.append(f'<geom type="box" size="0.12 0.95 0.12" pos="{x} {y + side * 0.55:.3f} {zd + 2.55:.4f}" rgba="0.9 0.49 0.13 1" class="visual"/>')
    s.append(extra)
    s.append('</body>')
    return "\n".join(s)


def scene_xml(agents, use_meshes: bool = True, timestep: float = 0.001) -> str:
    """The whole MJCF: options, visual settings, assets, the sea and the sky, the carrier and the agents."""
    assets, _ = mesh_assets(use_meshes)
    frag = {k: "" for k in ("asset", "world", "carrier", "equality", "tendon", "contact")}
    for a in agents:
        for k, v in a.xml(use_meshes).items():
            frag[k] += v
    return f"""<mujoco model="pinna_launch">
<compiler angle="radian" autolimits="true"/>
<option timestep="{timestep}" integrator="implicitfast" gravity="0 0 -{G}" density="0" viscosity="0"/>
<size memory="64M"/>
<statistic extent="30" center="0 0 5"/>
<visual><global offwidth="1920" offheight="1080" fovy="50"/><quality shadowsize="4096" offsamples="4"/>
<map znear="0.004" zfar="110" fogstart="0.05" fogend="0.6" haze="0.25" shadowclip="4"/>
<rgba haze="0.75 0.83 0.9 1" fog="0.05 0.25 0.32 1"/><headlight ambient="0.35 0.35 0.38" diffuse="0.55 0.55 0.55" specular="0.1 0.1 0.1"/></visual>
<default>
  <geom solref="0.004 1" solimp="0.95 0.99 0.001" friction="0.9 0.02 0.002"/>
  <default class="visual"><geom contype="0" conaffinity="0" group="1" mass="0"/></default>
  <default class="hull"><geom contype="1" conaffinity="1" group="3" rgba="1 0 0 0.3" mass="0"/></default>
  <default class="hullvis"><geom contype="1" conaffinity="1" group="1" mass="0"/></default>
  <default class="drone"><geom contype="1" conaffinity="3" group="3" rgba="0 1 0 0.3" mass="0"/></default>
</default>
<asset>
<texture name="sky" type="skybox" builtin="gradient" rgb1="0.55 0.72 0.9" rgb2="0.93 0.95 0.98" width="512" height="512"/>
<texture name="seatex" type="2d" builtin="checker" rgb1="0.10 0.31 0.43" rgb2="0.11 0.33 0.45" width="512" height="512" mark="random" random="0.004" markrgb="0.55 0.7 0.8"/>
<material name="sea" texture="seatex" texrepeat="2200 2200" reflectance="0.12" rgba="1 1 1 0.72"/>
<texture name="sandtex" type="2d" builtin="checker" rgb1="0.62 0.56 0.42" rgb2="0.58 0.52 0.39" width="256" height="256"/>
<material name="sand" texture="sandtex" texrepeat="60 60"/>
{assets}{frag['asset']}
</asset>
<worldbody>
<light name="sun" directional="true" pos="0 0 50" dir="-0.35 0.25 -1" diffuse="0.75 0.75 0.72" specular="0.2 0.2 0.2" castshadow="false"/>
<body name="water" mocap="true" pos="0 0 0"><geom name="sea" type="box" size="1500 1500 0.01" pos="0 0 -0.01" material="sea" class="visual"/></body>
<geom name="seabed" type="plane" size="1500 1500 1" pos="0 0 -10" material="sand" contype="1" conaffinity="1"/>
{carrier_xml(use_meshes, frag['carrier'])}
{frag['world']}
</worldbody>
<equality>{frag['equality']}</equality>
<tendon>{frag['tendon']}</tendon>
<contact><exclude body1="carrier" body2="net"/>{frag['contact']}</contact>
</mujoco>"""


# ------------------------------------------------------------------------------------------------ the simulator
class Sim:
    """The carrier, the sea, the wind and the agents in one MuJoCo model, stepped together."""

    def __init__(self, agents, sea: Sea | None = None, wind: Wind | None = None, use_meshes: bool = True,
                 timestep: float = 0.001, control_every: int = 2):
        import mujoco
        self.mj = mujoco
        self.agents = list(agents)
        self.sea, self.wind = sea or Sea(), wind or Wind()
        self.xml = scene_xml(self.agents, use_meshes, timestep)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.control_every = control_every
        self.dt_control = timestep * control_every
        m = self.model
        self.carrier = self.body("carrier")
        self.water = int(m.body_mocapid[self.body("water")])
        self._cj = [(int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])) for j in (self.joint("c_heave"), self.joint("c_roll"), self.joint("c_pitch"))]
        self._prescribed = {}                    # joint name -> function t -> (q, qd)
        self._k = 0
        for a in self.agents:
            a.bind(self)
        self._apply_prescribed(0.0)
        mujoco.mj_forward(m, self.data)

    # ids
    def body(self, name):
        return self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY, name)

    def joint(self, name):
        return self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_JOINT, name)

    def geom(self, name):
        return self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_GEOM, name)

    def site(self, name):
        return self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_SITE, name)

    def eq(self, name):
        return self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_EQUALITY, name)

    def tendon(self, name):
        return self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_TENDON, name)

    def prescribe(self, joint: str, fn):
        """Drive ``joint`` kinematically: fn(t) -> (q, qd), set every physics step (a robot on its trajectory)."""
        j = self.joint(joint)
        self._prescribed[joint] = (int(self.model.jnt_qposadr[j]), int(self.model.jnt_dofadr[j]), fn)

    def _apply_prescribed(self, t):
        q, qd = self.sea.state(t)
        d = self.data
        for (qa, va), x, v in zip(self._cj, q, qd):
            d.qpos[qa], d.qvel[va] = x, v
        for qa, va, fn in self._prescribed.values():
            d.qpos[qa], d.qvel[va] = fn(t)
        d.mocap_pos[self.water] = (0.0, 0.0, self.sea.eta(t))

    # the carrier's kinematics
    def carrier_point(self, p_local) -> tuple:
        """A point fixed on the carrier (its frame): world position and velocity now."""
        d, b = self.data, self.carrier
        R = d.xmat[b].reshape(3, 3)
        r = R @ np.asarray(p_local, dtype=float)
        v6 = np.zeros(6)
        self.mj.mj_objectVelocity(self.model, d, self.mj.mjtObj.mjOBJ_BODY, b, v6, 0)
        return d.xpos[b] + r, v6[3:] + np.cross(v6[:3], r)

    def carrier_R(self) -> np.ndarray:
        return self.data.xmat[self.carrier].reshape(3, 3).copy()

    def relock(self, eq_name: str, active: bool = True):
        """Make a weld hold its two bodies where they are now (a harpoon lock, the net's hook) and switch it on/off."""
        m, d = self.model, self.data
        e = self.eq(eq_name)
        b1, b2 = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
        R1 = d.xmat[b1].reshape(3, 3)
        rp = R1.T @ (d.xpos[b2] - d.xpos[b1])
        qi, rq = np.zeros(4), np.zeros(4)
        self.mj.mju_negQuat(qi, d.xquat[b1])
        self.mj.mju_mulQuat(rq, qi, d.xquat[b2])
        m.eq_data[e, 0:3] = 0.0
        m.eq_data[e, 3:6] = rp
        m.eq_data[e, 6:10] = rq
        d.eq_active[e] = 1 if active else 0

    def contact_between(self, geoms_a: set, geoms_b: set) -> bool:
        d = self.data
        for i in range(d.ncon):
            g1, g2 = int(d.contact[i].geom1), int(d.contact[i].geom2)
            if (g1 in geoms_a and g2 in geoms_b) or (g2 in geoms_a and g1 in geoms_b):
                return True
        return False

    def body_geoms(self, body: int) -> set:
        m = self.model
        return {g for g in range(m.ngeom) if m.geom_bodyid[g] == body and (m.geom_contype[g] or m.geom_conaffinity[g])}

    @property
    def t(self) -> float:
        return float(self.data.time)

    def step(self):
        d = self.data
        t = float(d.time)
        if self._k % self.control_every == 0:
            self.wind.update(self.dt_control)
            d.xfrc_applied[:] = 0.0
            for a in self.agents:
                a.control(self, t, self.dt_control)
        self._apply_prescribed(t)
        self.mj.mj_step(self.model, d)
        self._k += 1

    def run(self, t_end: float, every: float | None = None, callback=None, t_first: float | None = None):
        """Step to ``t_end``; call ``callback(sim)`` at ``t_first`` and every ``every`` seconds after it."""
        nxt = t_first if t_first is not None else (self.t if every else None)
        while self.t < t_end - 1e-9:
            if callback is not None and nxt is not None and self.t >= nxt - 1e-9:
                self.mj.mj_forward(self.model, self.data)
                callback(self)
                nxt += every
            self.step()
        return self

    def describe(self) -> str:
        return f"{self.sea.describe()}; {self.wind.describe()}"
