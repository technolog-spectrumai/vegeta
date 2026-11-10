"""The arena as MJCF: the field, a hoop on a stick at each end, the balloons, and the Émerillons as rigid bodies.

Frame: x along the field (team A's hoop at -x, team B's at +x), y across, z up; SI. An Émerillon's body frame has its
**z axis along the nose** (the thrust axis), so it stands on its tail at the identity orientation and the lab's quad
controller (``races.flight.Flight``: turn the frame's z onto the wanted force) flies it unchanged. Mass, centre of
gravity and inertia come from ``emerillon_flight.mass_budget``; the collision shape is a capsule for the body, four
for the motor pods and a disk per propeller.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

import emerillon_flight as ef
import emerillon_market as mk

TEAM_COLOR = {"A": "0.84 0.15 0.16 1", "B": "0.12 0.47 0.71 1"}
PALETTE = ("#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22")


def rgba(color: str) -> str:
    """A '#rrggbb' colour as MuJoCo's 'r g b a'."""
    return " ".join(f"{int(color[i:i + 2], 16) / 255:.3f}" for i in (1, 3, 5)) + " 1"


@dataclass
class Arena:
    """The field's numbers [m]. ``hoop_stick`` is the stick (the user's 2 m); the ring sits on top of it."""
    length: float = 60.0
    width: float = 30.0
    ceiling: float = 14.0
    hoop_x: float = 24.0                       # the hoops stand at -hoop_x (team A's) and +hoop_x (team B's)
    hoop_stick: float = 2.0
    hoop_radius: float = 0.9
    hoop_tube: float = 0.04
    start_x: float = 16.0                      # the aircraft start this far from the centre, in front of their own hoop
    start_z: float = 3.0
    n_balloons: int = 8
    balloon_radius: float = 0.45
    balloon_mass: float = 0.02
    seed: int = 1
    hoops: dict = field(init=False)

    def __post_init__(self):
        zc = self.hoop_stick + self.hoop_radius
        self.hoops = {"A": np.array([-self.hoop_x, 0.0, zc]), "B": np.array([self.hoop_x, 0.0, zc])}

    def extra_xml(self) -> str:
        """More of the scene (a subclass's walls and posts)."""
        return ""

    def start(self, team: str, slot: int = 0) -> np.ndarray:
        s = -1.0 if team == "A" else 1.0
        return np.array([s * self.start_x, (slot - 0.0) * 3.0, self.start_z])

    def ring_start(self, i: int, n: int, radius: float = 14.0) -> np.ndarray:
        """Start ``i`` of ``n`` on a circle round the centre (the group chases)."""
        ang = 2 * math.pi * i / n
        return np.array([radius * math.cos(ang), 0.8 * radius * math.sin(ang), self.start_z + 0.5 * (i % 2)])

    def balloons(self) -> np.ndarray:
        """Where the balloons start: spread over the middle of the field, clear of the starts and the hoops."""
        rng = np.random.default_rng(self.seed)
        pts = []
        while len(pts) < self.n_balloons:
            p = np.array([rng.uniform(-0.55, 0.55) * self.hoop_x * 2 / 1.1 * 0.5 * 1.1, rng.uniform(-0.4, 0.4) * self.width, rng.uniform(2.0, 0.6 * self.ceiling)])
            if all(np.linalg.norm(p - q) > 4 * self.balloon_radius for q in pts) and abs(abs(p[0]) - self.start_x) > 3.0:
                pts.append(p)
        return np.array(pts)


def body_numbers(variant: str) -> dict:
    """What the scene and the controller need of a variant: mass, the inertia about the CG in the nose-up body frame,
    where the body's capsule and the rotors sit from the CG, the rotor layout."""
    c = ef.VARIANTS[variant]
    g, b = ef.geometry(c), ef.mass_budget(c)
    pr = b.properties()
    x_cg = float(pr.com[0])
    I = np.diag(pr.inertia)                    # flight frame: x aft along the axis, y, z across
    shell = b.by_group().get("airframe", 0.0)
    i_axial = float(I[0]) + shell * (g["d"] / 2) ** 2 + 4 * 0.004 * 0.0       # the skins' own spin inertia; the point masses carry the rest
    i_cross = float(0.5 * (I[1] + I[2])) + shell * g["length"] ** 2 / 12.0
    r = g["r_arm"]
    ang = np.radians([45.0, 135.0, 225.0, 315.0])
    return {"variant": variant, "config": c, "mass": b.mass, "inertia": (i_cross, i_cross, max(i_axial, 1e-4)), "x_cg": x_cg, "length": g["length"],
            "d": g["d"], "r_arm": r, "pod_d": g["pod_d"], "pod_l": g["pod_l"], "prop_r": g["prop_d"] / 2,
            "z_nose": x_cg, "z_tail": x_cg - g["length"], "z_rotor": x_cg - g["x_prop"],
            "rotor_pos": np.column_stack([r * np.cos(ang), r * np.sin(ang), np.full(4, x_cg - g["x_prop"])]), "spin": np.array([1.0, -1.0, 1.0, -1.0]),
            "reach": math.hypot(r + g["prop_d"] / 2, max(x_cg, g["length"] - x_cg))}


def drone_xml(name: str, variant: str, color: str, pos) -> str:
    n = body_numbers(variant)
    col = rgba(color)
    rb = n["d"] / 2
    z0, z1 = n["z_tail"] + rb, n["z_nose"] - rb
    geoms = [f'<geom name="{name}_body" type="capsule" fromto="0 0 {z0:.4f} 0 0 {z1:.4f}" size="{rb:.4f}" rgba="{col}" mass="0"/>']
    for i, (x, y, z) in enumerate(n["rotor_pos"]):
        geoms.append(f'<geom name="{name}_pod{i}" type="capsule" fromto="{x:.4f} {y:.4f} {z - 0.6 * n["pod_l"]:.4f} {x:.4f} {y:.4f} {z:.4f}" size="{n["pod_d"] / 2:.4f}" rgba="{col}" mass="0"/>')
        geoms.append(f'<geom name="{name}_disk{i}" type="cylinder" pos="{x:.4f} {y:.4f} {z + 0.01:.4f}" size="{n["prop_r"]:.4f} 0.004" rgba="0.2 0.2 0.2 0.35" mass="0"/>')
        geoms.append(f'<geom name="{name}_arm{i}" type="capsule" fromto="0 0 {z - 0.03:.4f} {x:.4f} {y:.4f} {z - 0.03:.4f}" size="0.006" rgba="{col}" mass="0"/>')
    ix, iy, iz = n["inertia"]
    return (f'<body name="{name}" pos="{pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}">\n  <joint name="{name}_free" type="free"/>\n'
            f'  <inertial pos="0 0 0" mass="{n["mass"]:.4f}" diaginertia="{ix:.6f} {iy:.6f} {iz:.6f}"/>\n  ' + "\n  ".join(geoms) +
            f'\n  <camera name="{name}_chase" pos="0 -2.5 1.0" xyaxes="1 0 0 0 0.4 1" mode="trackcom"/>\n</body>')


def hoop_xml(team: str, a: Arena, segments: int = 20) -> str:
    c = a.hoops[team]
    col = TEAM_COLOR[team]
    out = [f'<geom name="hoop{team}_stick" type="capsule" fromto="{c[0]:.3f} 0 0 {c[0]:.3f} 0 {a.hoop_stick:.3f}" size="{a.hoop_tube:.3f}" rgba="{col}"/>']
    th = np.linspace(0, 2 * math.pi, segments + 1)
    for i in range(segments):                  # the ring in the y-z plane: its axis along the field
        p0 = (c[0], c[1] + a.hoop_radius * math.sin(th[i]), c[2] - a.hoop_radius * math.cos(th[i]))
        p1 = (c[0], c[1] + a.hoop_radius * math.sin(th[i + 1]), c[2] - a.hoop_radius * math.cos(th[i + 1]))
        out.append(f'<geom name="hoop{team}_ring{i}" type="capsule" fromto="{p0[0]:.3f} {p0[1]:.3f} {p0[2]:.3f} {p1[0]:.3f} {p1[1]:.3f} {p1[2]:.3f}" size="{a.hoop_tube:.3f}" rgba="{col}"/>')
    return "\n".join(out)


def to_mjcf(a: Arena, drones: list) -> str:
    """The scene. ``drones``: (name, variant, colour '#rrggbb', start position). Balloons are free bodies with their weight taken
    off (``gravcomp``) and a little damping: they drift where the match pushes them."""
    bal = []
    for i, p in enumerate(a.balloons()):
        bal.append(f'<body name="balloon{i}" pos="{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}" gravcomp="1">\n  <joint name="balloon{i}_free" type="free" damping="0.02"/>\n'
                   f'  <geom name="balloon{i}" type="sphere" size="{a.balloon_radius:.3f}" mass="{a.balloon_mass:.4f}" rgba="1.0 0.75 0.1 0.85"/>\n</body>')
    L, W = a.length / 2, a.width / 2
    return f'''<mujoco model="air_match">
  <option timestep="0.002" integrator="implicitfast" density="0" viscosity="0"/>
  <visual><global offwidth="1280" offheight="720"/><quality shadowsize="2048"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.30 0.45 0.30" rgb2="0.27 0.41 0.27" width="512" height="512"/>
    <material name="grass" texture="grid" texrepeat="12 6"/>
  </asset>
  <default><geom condim="3" friction="0.8 0.02 0.001"/></default>
  <worldbody>
    <light pos="0 0 40" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
    <geom name="ground" type="plane" size="{L + 10:.1f} {W + 10:.1f} 0.1" material="grass"/>
    <geom name="line_mid" type="box" pos="0 0 0.002" size="0.05 {W:.1f} 0.002" rgba="1 1 1 0.8" contype="0" conaffinity="0"/>
    <camera name="overview" pos="0 0 {1.25 * a.length:.1f}" xyaxes="1 0 0 0 1 0"/>
    <camera name="grandstand" pos="0 {-1.5 * a.width:.1f} 14" xyaxes="1 0 0 0 0.3 1"/>
{chr(10).join(hoop_xml(k, a) for k in a.hoops)}
{a.extra_xml()}
{chr(10).join(bal)}
{chr(10).join(drone_xml(*d) for d in drones)}
  </worldbody>
</mujoco>'''
