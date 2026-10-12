"""Perseus (notebook 37) and Perseus Mini (notebook 38) — the branch-cutting rovers' numbers and their Chiron (MuJoCo)
models. Both are one ``Variant`` each (``PERSEUS`` here, ``perseus_mini_robot.MINI``): the chassis parameters
(``perseus_rover.PERSEUS_CHASSIS``, a Pekari track architecture), the boom parameters, stored CAD numbers, materials,
the parts list, the track drive and the boom drives (``actuators.py``), the chainsaw (``cutting.py``), the stowed
pose and the simulation options. Every function takes the variant (default ``PERSEUS``) and a parameter dict.

The chassis numbers come from ``pekari_rover_robot`` (``prr``) with the variant's keywords: the mass budget, the
battery placement (which also balances the boom in its stowed pose), the roller model of the tracks. The boom is one
Link tree under the hull (``_arm``), in the Onager Manus pattern (``onager_manus_robot``): a pedestal (yaw), the boom
(shoulder pitch), the stick (elbow pitch) and the saw head (wrist pitch) carrying the saw body (its mass from the saw
entry) and the bar — a thin box geom ``saw_bar`` whose contact with a branch is what the scene's ``SawCutter`` reads.

=================  ===========================================  ===============================================
joint              Perseus                                      convention
=================  ===========================================  ===============================================
``arm_yaw``        harmonic 150 Nm module                       about the pedestal's vertical axis, + to the left
``arm_shoulder``   boom cylinder 12 kN on the lift lever        about y; the boom's elevation is ``−q``
``arm_elbow``      stick cylinder 8 kN on the stick lever       about y, relative; the stick's elevation ``−(q_sh + q_el)``
``arm_wrist``      harmonic 150 Nm module                       about y, relative; the bar's elevation ``−(q_sh + q_el + q_wr)``
=================  ===========================================  ===============================================

A cylinder drive is a hinge with the cylinder's force on its lever arm (``perseus_rover.lift_lever`` /
``stick_lever``, which change with the pose): the Chiron servo takes the smallest lever over the joint's range
(conservative) and ``four_quadrant`` (the brake holds); ``joint_limits`` gives the pose-dependent values for the
torque checks. ``arm_ik`` / ``arm_fk`` are the Manus two-link solution with the head as the palm: the tool point is
the **bar root** (the bar's start at the saw body), ``bar_point`` any point along the bar.

    import perseus_robot as pbr
    robot = pbr.perseus()                      # ~180 kg
    lab = pbr.perseus_lab(terrain, props=..., welds=...)
    pbr.arm_torques(pbr.elevations_to_q(30, -20, 0), load=cutting.head_loads(saw, wood))
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from vegeta.chiron import ChironLab, Geom, Joint, Link, PointMass, Robot, Servo

import actuators as act
import cutting
import pekari_rover_robot as prr
import tracks
from perseus_rover import PERSEUS_ARM, PERSEUS_CHASSIS, PerseusRover

__all__ = ["G", "Variant", "PERSEUS", "ARM_JOINTS", "design_params", "chassis_params", "cad_numbers", "arm_geometry",
           "arm_joints", "arm_ik", "arm_fk", "bar_point", "elevations_to_q", "q_to_elevations", "stow_q", "nominal_qpos",
           "carry_q", "arm_masses", "arm_mass_points", "mass_budget", "masses", "cg", "total_mass", "joint_limits", "arm_torques",
           "workspace", "support_point", "tip_over", "track_loads", "drive_check", "perseus", "perseus_lab"]

G = 9.81
ARM_JOINTS = ("arm_yaw", "arm_shoulder", "arm_elbow", "arm_wrist")


@dataclass(frozen=True)
class Variant:
    """One machine: its parameters, stored CAD numbers, materials, parts, drives, saw and poses."""

    name: str
    design_cls: str                       # the Dedalus class in perseus_rover.py
    chassis: dict                         # Pekari parameters (mm)
    arm: dict                             # boom parameters (mm, deg)
    cad: dict                             # stored CAD measurements (mm², mm³, mm)
    materials: dict                       # densities [kg/mm³], the Pekari keys plus the arm's
    shell_t_mm: float
    wheel_fill: float
    pedestal_fill: float                  # the pedestal is a casting / weldment: this fraction of the solid CAD volume
    parts_kg: dict                        # the chassis parts list (Pekari's name prefixes)
    arm_parts_kg: dict                    # the boom's listed parts
    drive: str                            # track drive catalogue key
    arm_drives: dict                      # joint -> ("rotary", key) | ("linear", key)
    arm_gains: dict                       # joint -> (kp, kd, armature)
    joint_ranges: dict                    # joint -> (lo, hi) deg (q, the Manus conventions)
    saw: str                              # cutting.SAWS key
    wood: str                             # the design wood (cutting.WOODS key)
    k_feed: float                         # feed force / chain force while cutting (cutting.feed_force)
    stow: dict                            # elevations deg: boom, stick, head (yaw 0)
    carry_head_deg: float                 # the bar raised by this much over the stow for a swing between branches
    battery_Wh: float
    electronics_W: float
    track_mu: float
    internal_f0: float
    lab_options: dict
    assumptions: dict = field(default_factory=dict)

    @property
    def n_bogies(self) -> int:
        return int(self.chassis["road_wheels"]) // 2

    @property
    def payload_kg(self) -> float:
        return 0.0

    def cut_loads(self, wood=None, cut_from: str = "top", dynamic: float = 1.0) -> dict:
        """``cutting.head_loads`` of this machine's saw on ``wood`` (default the design wood) at its ``k_feed``."""
        return cutting.head_loads(cutting.get_saw(self.saw), wood or self.wood, cut_from=cut_from, k_feed=self.k_feed,
                                  dynamic=dynamic)


#: CAD measurements of the default Perseus (``perseus_rover.PerseusRover``, CadQuery; mm², mm³, mm):
#: ``cad_numbers(recompute=True)`` rebuilds them.
PERSEUS_CAD = {
    "hull_surface_area": 2124846.9, "hull_com": (5.36, 0.0, 302.2), "link_volume": 130553.5, "sprocket_volume": 650991.9,
    "idler_volume": 5213595.5, "road_wheel_volume": 1999871.9, "pinion_volume": 9543.7, "basket_volume": 0.0,
    "basket_com": (0.0, 0.0, 0.0),
    "pedestal_volume": 4098839.0, "boom_volume": 3334865.4, "stick_volume": 3476827.6, "saw_head_volume": 6903973.2,
}

PERSEUS = Variant(
    name="Perseus", design_cls="PerseusRover", chassis=dict(PERSEUS_CHASSIS), arm=dict(PERSEUS_ARM), cad=PERSEUS_CAD,
    materials={**prr.MATERIALS, "Al 6082-T6 arm tubes": 2.70e-6, "Al casting pedestal": 2.70e-6},
    shell_t_mm=3.0, wheel_fill=0.35, pedestal_fill=0.4,
    parts_kg={
        "battery 48 V 40 Ah LiFePO4 (1.9 kWh)": 16.0,
        "computer, motor controllers, radio, IMU, GNSS receiver": 3.0,
        "sensors: stereo camera, 3D lidar, GNSS mast, beacon (sensor box)": 3.5,
        "wiring, connectors, e-stop, switches": 3.0,
        "hull frame, deck, fasteners": 9.0,
        "track frames, bogies, axles, bearings 2x": 14.0,
    },
    arm_parts_kg={"head mount (wrist housing, saw clamp, guard)": 2.0, "cylinder lugs and brackets": 1.0,
                  "arm cabling, saw cable and covers": 1.5},
    drive="perseus track drive 1.5 kW, 48 V",
    arm_drives={"yaw": ("rotary", "harmonic 150 Nm, brake"), "shoulder": ("linear", "boom cylinder 12 kN"),
                "elbow": ("linear", "stick cylinder 8 kN"), "wrist": ("rotary", "harmonic 150 Nm, brake")},
    arm_gains={"yaw": (1500.0, 120.0, 0.3), "shoulder": (4000.0, 300.0, 0.8), "elbow": (2500.0, 150.0, 0.5),
               "wrist": (1500.0, 100.0, 0.3)},
    joint_ranges={"yaw": (-170.0, 170.0), "shoulder": (-110.0, 0.0), "elbow": (15.0, 140.0), "wrist": (-115.0, 115.0)},
    saw="chainsaw 2 kW, 36 V", wood="softwood (spruce, pine), green", k_feed=1.5,
    stow={"boom": 78.0, "stick": -62.0, "head": 0.0}, carry_head_deg=50.0,
    battery_Wh=1900.0, electronics_W=60.0, track_mu=0.6, internal_f0=0.035,
    lab_options={"heightfield_cell": 0.03, "course_extent": (-2.0, 14.0, -4.0, 4.0), "timestep": 0.001,
                 "control_dt": 0.002, "log_dt": 0.02},
    assumptions={
        "class": "a ~180 kg forestry-class tracked machine (the user's choice from the reference image): the Pekari "
                 "track architecture scaled by parameters, one boom with a 2 kW battery chainsaw, branches to ~120 mm "
                 "(the 400 mm bar in one pass, up to 2x with two cuts)",
        "hull": "3 mm Al 6061 sheet over the CAD hull's area; frame, deck and fasteners listed",
        "links": "PA6-GF30 links with 10 mm steel pins as the Pekari's (the pin and link stresses in notebook 37 §2 "
                 "decide whether steel links are needed: a follow-up)",
        "drive": "one 'perseus track drive 1.5 kW, 48 V' per track: sized so the 30 deg climb (0.5 W + the rolling and "
                 "internal losses ~ 1100 N, 66 N m per sprocket at r 0.12 m) is within the continuous 60 N m x 2 "
                 "with the grouser traction limit, and 1.5 m/s on the flat is below the no-load speed",
        "boom_drives": "the lift and stick drives are electric cylinders (ball screws with brakes, the reference image's "
                       "cylinders) acting on lever arms that change with the pose (perseus_rover.lift_lever / "
                       "stick_lever); in MuJoCo each is a hinge servo with the force on the smallest lever of the "
                       "joint's range and four-quadrant braking (the brake holds the boom unpowered). The joint ranges "
                       "(boom 0-110 deg up — past vertical to reach over a close branch — stick 15-140 deg folded on the boom) keep the cylinders off their dead "
                       "centres: outside them a cylinder's line passes through its pivot (no lever)",
        "stow": "boom 78 deg up, the stick folded 140 deg (its cylinder's range), the bar level: it overhangs the nose "
                "by ~0.8 m at the sensor box's height (a forestry boom travels folded up, the saw forward)",
        "battery_clamp": "the boom ahead of the hull centre asks the battery to sit further back than the hull allows: "
                         "it is clamped at the rear wall and the stowed machine's CG stays a few cm ahead of the "
                         "road wheels' centre (cg()['x'])",
        "arm_gains": "kp so the loaded boom sags < 2 deg at full reach without feed-forward; kd for zeta ~ 0.6 with the "
                     "arm's inertia (shoulder ~ 60 kg m2 outstretched); the controller adds qfrc_bias as feed-forward",
        "pedestal": "an Al casting / weldment: 40 % of the solid CAD volume (pedestal_fill)",
        "saw": "the saw's mass is the catalogue entry's (powerhead without battery, bar, chain, mount), not the CAD "
               "saw body (a solid box); it is fed from the rover's 48 V battery through a converter (in the electronics)",
        "saw_head_cad": "the saw_head CAD volume (mount + solid saw body + bar) is not used for mass: the mount is "
                        "listed, the saw is the catalogue entry",
        "cylinders_in_mujoco": "the cylinders are drawn in the ready pose on the boom (visual geoms) and their masses "
                               "ride on the boom at mid-stroke; they do not articulate (the hinge servo is their effect)",
        "bar_contact": "the bar is a box of the plate's thickness; its friction on wood 0.4 (steel on green wood); the "
                       "branch prop's contact settings govern the pair (Chiron: prop priority)",
        "battery_position": "on the hull floor at the x that balances the empty machine, boom stowed, over the road "
                            "wheels' centre (within the hull)",
        "cut_wood": "the design wood is green softwood (spruce, pine); the hardwood and shrub classes are in cutting.WOODS",
    },
)


# ----------------------------------------------------------------------------------------------- parameters and CAD
def design_params(overrides: dict | None = None, v: Variant = PERSEUS) -> dict:
    """The variant's chassis and boom parameters with ``overrides`` (mm, deg)."""
    p = {**v.chassis, **v.arm}
    for k, val in (overrides or {}).items():
        if k not in p:
            raise KeyError(f"unknown {v.name} parameter {k!r}")
        p[k] = val
    return p


def chassis_params(p: dict | None = None, v: Variant = PERSEUS) -> dict:
    """The Pekari subset of the parameters (what ``pekari_rover_robot`` takes)."""
    p = design_params(None, v) if p is None else p
    return {k: p[k] for k in prr.DESIGN}


def _design(v: Variant):
    import perseus_rover

    return getattr(perseus_rover, v.design_cls)


def cad_numbers(p: dict | None = None, v: Variant = PERSEUS, *, recompute: bool = False) -> dict:
    """The CAD areas, volumes and centres the mass budget needs (``Variant.cad``): stored for the default design,
    else rebuilt with the Dedalus class (CadQuery)."""
    p = design_params(None, v) if p is None else dict(p)
    if not recompute and p == design_params(None, v):
        return dict(v.cad)
    design = _design(v)()
    names = {q.name for q in design.parameters}
    over = {k: val for k, val in p.items() if k in names}
    m = {k: design.generate(part=k, **over).measure()
         for k in ("hull", "track_link", "sprocket", "idler", "road_wheel", "pinion", "pedestal", "boom", "stick", "saw_head")}
    return {"hull_surface_area": m["hull"]["surface_area"], "hull_com": tuple(m["hull"]["center_of_mass"]),
            "link_volume": m["track_link"]["volume"], "sprocket_volume": m["sprocket"]["volume"],
            "idler_volume": m["idler"]["volume"], "road_wheel_volume": m["road_wheel"]["volume"],
            "pinion_volume": m["pinion"]["volume"], "basket_volume": 0.0, "basket_com": (0.0, 0.0, 0.0),
            "pedestal_volume": m["pedestal"]["volume"], "boom_volume": m["boom"]["volume"],
            "stick_volume": m["stick"]["volume"], "saw_head_volume": m["saw_head"]["volume"]}


# ----------------------------------------------------------------------------------------------- geometry
def arm_geometry(p: dict | None = None, v: Variant = PERSEUS) -> dict:
    """Boom lengths [m] and the shoulder point: ``x`` in the vehicle frame, ``z`` above the hull centre (the hull
    link's frame), ``z_world`` above the ground; ``Lb`` boom, ``Ls`` stick, ``Lh`` head (wrist to bar root),
    ``Lbar``, ``bar_w``, ``bar_t``, the saw body, the pedestal."""
    p = design_params(None, v) if p is None else p
    mm = 1e-3
    sx, _, sz = PerseusRover.shoulder_point(p)
    z_hull = (p["ground_clearance"] + p["hull_height"] / 2) * mm
    return {"x": sx * mm, "y": 0.0, "z": sz * mm - z_hull, "z_world": sz * mm, "z_hull": z_hull,
            "Lb": p["boom_length"] * mm, "Ls": p["stick_length"] * mm, "Lh": p["head_length"] * mm,
            "Lbar": p["bar_length"] * mm, "bar_w": p["bar_width"] * mm, "bar_t": p["bar_thickness"] * mm,
            "saw_L": p["saw_body_length"] * mm, "saw_w": p["saw_body_width"] * mm, "saw_h": p["saw_body_height"] * mm,
            "ped_h": p["pedestal_height"] * mm, "ped_r": p["pedestal_diameter"] / 2 * mm,
            "boom_w": p["boom_width"] * mm, "boom_d": p["boom_depth"] * mm,
            "stick_w": p["stick_width"] * mm, "stick_d": p["stick_depth"] * mm,
            "reach": PerseusRover.reach(p) * mm}


def arm_joints() -> list:
    """``[arm_yaw, arm_shoulder, arm_elbow, arm_wrist]``."""
    return list(ARM_JOINTS)


def elevations_to_q(boom_deg: float, stick_deg: float, head_deg: float, yaw_deg: float = 0.0) -> tuple:
    """Joint angles (yaw, sh, el, wr) [rad] from the elevations of the boom, the stick and the bar [deg above
    horizontal] (the Manus conventions: q_sh = −boom, q_el = boom − stick, q_wr = stick − head)."""
    b, s, h = (math.radians(a) for a in (boom_deg, stick_deg, head_deg))
    return math.radians(yaw_deg), -b, b - s, s - h


def q_to_elevations(q) -> tuple:
    """Elevations (boom, stick, head) [deg] and the yaw [deg] of joint angles (yaw, sh, el, wr)."""
    yaw, sh, el, wr = q[:4]
    return math.degrees(-sh), math.degrees(-(sh + el)), math.degrees(-(sh + el + wr)), math.degrees(yaw)


def arm_ik(target, elevation: float, p: dict | None = None, v: Variant = PERSEUS):
    """Arm angles (yaw, sh, el, wr) [rad] that put the **bar root** at ``target`` (x, y, z) [m, hull frame] with the
    bar at ``elevation`` [rad] above horizontal; elbow up (the Manus two-link solution). None when out of reach."""
    g = arm_geometry(p, v)
    dx, dy, dz = target[0] - g["x"], target[1] - g["y"], target[2] - g["z"]
    yaw = math.atan2(dy, dx)
    r = math.hypot(dx, dy)
    wr, wz = r - g["Lh"] * math.cos(elevation), dz - g["Lh"] * math.sin(elevation)
    d = math.hypot(wr, wz)
    Lu, Lf = g["Lb"], g["Ls"]
    if d > Lu + Lf - 1e-9 or d < abs(Lu - Lf) + 1e-9:
        return None
    beta = math.acos(max(-1.0, min(1.0, (Lu * Lu + d * d - Lf * Lf) / (2 * Lu * d))))
    phi1 = math.atan2(wz, wr) + beta
    ex, ez = Lu * math.cos(phi1), Lu * math.sin(phi1)
    phi2 = math.atan2(wz - ez, wr - ex)
    q = (yaw, -phi1, phi1 - phi2, phi2 - elevation)
    for name, val in zip(("yaw", "shoulder", "elbow", "wrist"), q):
        lo, hi = v.joint_ranges[name]
        if not math.radians(lo) - 1e-9 <= val <= math.radians(hi) + 1e-9:
            return None
    return q


def arm_fk(q, p: dict | None = None, v: Variant = PERSEUS) -> tuple:
    """Bar root (x, y, z) [m, hull frame] and the bar's elevation [rad] for arm angles (yaw, sh, el, wr)."""
    g = arm_geometry(p, v)
    yaw, sh, el, wr = q[:4]
    phi1, phi2, phi3 = -sh, -(sh + el), -(sh + el + wr)
    r = g["Lb"] * math.cos(phi1) + g["Ls"] * math.cos(phi2) + g["Lh"] * math.cos(phi3)
    z = g["Lb"] * math.sin(phi1) + g["Ls"] * math.sin(phi2) + g["Lh"] * math.sin(phi3)
    return (g["x"] + r * math.cos(yaw), g["y"] + r * math.sin(yaw), g["z"] + z), phi3


def bar_point(q, s: float, p: dict | None = None, v: Variant = PERSEUS) -> tuple:
    """A point ``s`` [m] along the bar from its root (x, y, z) [m, hull frame]; ``s`` = ``Lbar`` is the nose."""
    (x, y, z), phi = arm_fk(q, p, v)
    yaw = q[0]
    return (x + s * math.cos(phi) * math.cos(yaw), y + s * math.cos(phi) * math.sin(yaw), z + s * math.sin(phi))


def _arm_points(q, p, v) -> dict:
    """Shoulder, elbow, wrist, bar root and nose (x, y, z) [m, hull frame] for arm angles."""
    g = arm_geometry(p, v)
    yaw, sh, el, wr = q[:4]
    phi1, phi2, phi3 = -sh, -(sh + el), -(sh + el + wr)
    c, s_ = math.cos(yaw), math.sin(yaw)

    def pt(r, z):
        return (g["x"] + r * c, g["y"] + r * s_, g["z"] + z)
    r1, z1 = g["Lb"] * math.cos(phi1), g["Lb"] * math.sin(phi1)
    r2, z2 = r1 + g["Ls"] * math.cos(phi2), z1 + g["Ls"] * math.sin(phi2)
    r3, z3 = r2 + g["Lh"] * math.cos(phi3), z2 + g["Lh"] * math.sin(phi3)
    r4, z4 = r3 + g["Lbar"] * math.cos(phi3), z3 + g["Lbar"] * math.sin(phi3)
    return {"shoulder": pt(0.0, 0.0), "elbow": pt(r1, z1), "wrist": pt(r2, z2), "bar_root": pt(r3, z3),
            "nose": pt(r4, z4), "phi": (phi1, phi2, phi3)}


def stow_q(v: Variant = PERSEUS) -> tuple:
    """The stowed pose (yaw, sh, el, wr) [rad]: the boom up, the stick folded back, the bar level over the nose."""
    return elevations_to_q(v.stow["boom"], v.stow["stick"], v.stow["head"])


def carry_q(v: Variant = PERSEUS, yaw: float = 0.0) -> tuple:
    """The carry pose for a swing between branches (yaw, sh, el, wr) [rad]: the stow with the bar raised by
    ``carry_head_deg`` so the saw clears the trunk while the pedestal turns."""
    return elevations_to_q(v.stow["boom"], v.stow["stick"], v.stow["head"] + v.carry_head_deg, math.degrees(yaw))


def nominal_qpos(p: dict | None = None, v: Variant = PERSEUS) -> dict:
    """The arm joints in the stowed pose (the rollers and bogies are the chassis builder's)."""
    return dict(zip(ARM_JOINTS, stow_q(v)))


# ----------------------------------------------------------------------------------------------- masses
def _drive_mass_kg(spec) -> float:
    kind, key = spec
    return (act.get(key) if kind == "rotary" else act.get_linear(key)).mass_g / 1000.0


def arm_masses(p: dict | None = None, cad: dict | None = None, v: Variant = PERSEUS) -> dict:
    """The boom assembly's parts [kg]: CAD volumes × densities (boom, stick, the pedestal at its fill), the
    drives (modules or cylinders), the saw (``cutting``), the listed parts."""
    p = design_params(None, v) if p is None else p
    cad = cad_numbers(p, v) if cad is None else cad
    al = v.materials["Al 6082-T6 arm tubes"]
    out = {"boom (Al 6082, CAD)": cad["boom_volume"] * al, "stick (Al 6082, CAD)": cad["stick_volume"] * al,
           f"pedestal (Al casting, {v.pedestal_fill:.0%} of CAD)": cad["pedestal_volume"] * v.materials["Al casting pedestal"] * v.pedestal_fill}
    for j, spec in v.arm_drives.items():
        out[f"{j} drive ({spec[1]})"] = _drive_mass_kg(spec)
    saw = cutting.get_saw(v.saw)
    out[f"saw ({saw.key})"] = saw.mass_kg
    out.update(v.arm_parts_kg)
    return out


def arm_mass_points(q=None, p: dict | None = None, cad: dict | None = None, v: Variant = PERSEUS) -> list:
    """Every boom mass with its position [(name, kg, x, y, z)] in the vehicle frame (z above the ground) for the
    pose ``q`` (default: stowed): the boom and the stick at their middles, the drives at their joints (a cylinder at
    its mid-stroke on the boom), the saw body behind the bar root, the head mount at the wrist."""
    p = design_params(None, v) if p is None else p
    q = stow_q(v) if q is None else q
    m = arm_masses(p, cad, v)
    g = arm_geometry(p, v)
    pts = _arm_points(q, p, v)
    phi1, phi2, phi3 = pts["phi"]
    z0 = g["z_hull"]                                                # hull frame -> above the ground

    def world(pt):
        return (pt[0], pt[1], pt[2] + z0)

    def mid(a, b, f=0.5):
        return tuple(a[i] + (b[i] - a[i]) * f for i in range(3))
    sh, el, wr, root = pts["shoulder"], pts["elbow"], pts["wrist"], pts["bar_root"]
    yaw = q[0]
    saw_c = (root[0] - g["saw_L"] / 2 * math.cos(phi3) * math.cos(yaw), root[1] - g["saw_L"] / 2 * math.cos(phi3) * math.sin(yaw),
             root[2] - g["saw_L"] / 2 * math.sin(phi3))
    ped = (sh[0], sh[1], sh[2] - g["ped_h"] / 2)
    where = {}
    for name in m:
        if name.startswith("boom "):
            where[name] = mid(sh, el)
        elif name.startswith("stick "):
            where[name] = mid(el, wr)
        elif name.startswith("pedestal"):
            where[name] = ped
        elif name.startswith("yaw drive"):
            where[name] = (sh[0], sh[1], sh[2] - g["ped_h"])
        elif name.startswith("shoulder drive"):
            where[name] = mid(sh, el, 0.3) if v.arm_drives["shoulder"][0] == "linear" else sh
        elif name.startswith("elbow drive"):
            where[name] = mid(sh, el, 0.75) if v.arm_drives["elbow"][0] == "linear" else el
        elif name.startswith("wrist drive"):
            where[name] = wr
        elif name.startswith("saw "):
            where[name] = saw_c
        elif name.startswith("head mount"):
            where[name] = mid(wr, root)
        elif name.startswith("cylinder lugs"):
            where[name] = mid(sh, el)
        else:
            where[name] = mid(sh, el)                               # cabling along the boom
    return [(name, m[name], *world(where[name])) for name in m]


def mass_budget(p: dict | None = None, cad: dict | None = None, v: Variant = PERSEUS) -> dict:
    """The whole machine [kg]: the chassis (``prr.mass_budget`` with the variant's parts) and the boom assembly."""
    p = design_params(None, v) if p is None else p
    cad = cad_numbers(p, v) if cad is None else cad
    out = prr.mass_budget(chassis_params(p, v), cad, v.parts_kg, drive=v.drive, shell_t_mm=v.shell_t_mm,
                          materials=v.materials, wheel_fill=v.wheel_fill)
    out["boom assembly (see arm_masses)"] = sum(arm_masses(p, cad, v).values())
    return out


def total_mass(p: dict | None = None, v: Variant = PERSEUS) -> float:
    return sum(mass_budget(p, None, v).values())


def masses(p: dict | None = None, q=None, cad: dict | None = None, v: Variant = PERSEUS) -> list:
    """Every mass with its position [(name, kg, x, y, z)] [m] in the vehicle frame (x from the road wheels' centre,
    z above the ground): the chassis (``prr.masses``, the battery placed with the stowed boom's moment) and the
    boom in pose ``q`` (default stowed)."""
    p = design_params(None, v) if p is None else p
    cad = cad_numbers(p, v) if cad is None else cad
    cp = chassis_params(p, v)
    budget = prr.mass_budget(cp, cad, v.parts_kg, drive=v.drive, shell_t_mm=v.shell_t_mm, materials=v.materials,
                             wheel_fill=v.wheel_fill)
    arm_stow = arm_mass_points(stow_q(v), p, cad, v)
    m_arm = sum(m for _, m, *_ in arm_stow)
    moment = sum(m * x for _, m, x, _, _ in arm_stow)
    chassis = prr.masses(cp, 0.0, cad=cad, budget=budget, extra_mass_kg=m_arm, extra_moment_kgm=moment)
    out = [(name, m, x, 0.0, z) for name, m, x, z in chassis]
    return out + (arm_stow if q is None else arm_mass_points(q, p, cad, v))


def cg(p: dict | None = None, q=None, v: Variant = PERSEUS) -> dict:
    """Mass [kg] and CG (x, y, z) [m] with the boom in pose ``q`` (default stowed), and the battery's x."""
    ms = masses(p, q, None, v)
    m = sum(k for _, k, *_ in ms)
    out = {"mass_kg": m, "x": sum(k * x for _, k, x, _, _ in ms) / m, "y": sum(k * y for _, k, _, y, _ in ms) / m,
           "z": sum(k * z for _, k, _, _, z in ms) / m}
    out["battery_x"] = next(x for n, _, x, _, _ in ms if n.startswith("battery"))
    return out


# ----------------------------------------------------------------------------------------------- drives and torques
def _lever_over_range(joint: str, p: dict, v: Variant, n: int = 61) -> tuple:
    """(min, nominal) lever arm [m] of a cylinder drive over its joint's range (the nominal at the ready pose)."""
    mm = 1e-3
    D = PerseusRover
    lo, hi = v.joint_ranges["shoulder"]
    booms = np.linspace(-hi, -lo, n)                                 # boom elevation = −q_sh
    if joint == "shoulder":
        levers = [D.lift_lever(p, b) for b in booms]
        nominal = D.lift_lever(p)
    else:
        lo_e, hi_e = v.joint_ranges["elbow"]
        levers = [D.stick_lever(p, b, b - e) for b in booms[::6] for e in np.linspace(lo_e, hi_e, 25)]
        nominal = D.stick_lever(p)
    return max(min(levers), 1e-6) * mm, max(nominal, 1e-6) * mm


def joint_limits(joint: str, q=None, p: dict | None = None, v: Variant = PERSEUS) -> dict:
    """Stall and rated torque [N·m] of a joint's drive at pose ``q`` (a cylinder's force on its lever arm there;
    None: the smallest lever of the range), and its no-load speed [rad/s]."""
    p = design_params(None, v) if p is None else p
    kind, key = v.arm_drives[joint]
    if kind == "rotary":
        a = act.get(key)
        return {"stall_Nm": a.stall_Nm, "rated_Nm": a.rated_Nm, "no_load_rad_s": a.no_load_rpm * 2 * math.pi / 60,
                "lever_m": None, "drive": key}
    a = act.get_linear(key)
    mm = 1e-3
    if q is None:
        lever, _ = _lever_over_range(joint, p, v)
    else:
        boom, stick, _, _ = q_to_elevations(q)
        lever = (PerseusRover.lift_lever(p, boom) if joint == "shoulder" else PerseusRover.stick_lever(p, boom, stick)) * mm
    lever = max(lever, 1e-6)
    return {"stall_Nm": a.stall_N * lever, "rated_Nm": a.rated_N * lever, "no_load_rad_s": a.no_load_mm_s * mm / lever,
            "lever_m": lever, "drive": key}


def arm_torques(q, p: dict | None = None, v: Variant = PERSEUS, load: dict | None = None, *, cad: dict | None = None) -> dict:
    """Static torques [N·m] at the yaw, shoulder, elbow and wrist in pose ``q`` from the boom's own masses and an
    external ``load`` (``cutting.head_loads``: forces in the head frame, applied at the middle of the bar), against
    each drive's stall and rated torque in that pose."""
    p = design_params(None, v) if p is None else p
    pts = _arm_points(q, p, v)
    g = arm_geometry(p, v)
    yaw = q[0]
    phi3 = pts["phi"][2]
    z0 = g["z_hull"]
    piv = {"shoulder": pts["shoulder"], "elbow": pts["elbow"], "wrist": pts["wrist"]}
    # forces in the hull frame: gravity on the boom masses, the load at the bar's middle
    forces = []
    for name, m, x, y, z in arm_mass_points(q, p, cad, v):
        if name.startswith("pedestal") or name.startswith("yaw drive"):
            continue
        forces.append(((x, y, z - z0), (0.0, 0.0, -m * G), name))
    if load is not None:
        fx, fy, fz = load["total"]
        c, s = math.cos(phi3), math.sin(phi3)
        fx_h, fz_h = fx * c - fz * s, fx * s + fz * c              # head frame -> hull xz (pitch), then yaw
        F = (fx_h * math.cos(yaw) - fy * math.sin(yaw), fx_h * math.sin(yaw) + fy * math.cos(yaw), fz_h)
        forces.append((bar_point(q, g["Lbar"] / 2, p, v), F, "cutting load"))
    axis_y = (-math.sin(yaw), math.cos(yaw), 0.0)                   # the pitch axes after the yaw
    out = {}
    for j in ("shoulder", "elbow", "wrist"):
        tau = 0.0
        for r, F, name in forces:
            if j == "elbow" and (name.startswith("boom ") or name.startswith("shoulder drive") or name.startswith("elbow drive")
                                 or name.startswith("cylinder lugs") or name.startswith("arm cabling")):
                continue                                             # inboard of the elbow
            if j == "wrist" and not (name.startswith("saw ") or name.startswith("head mount") or name == "cutting load"):
                continue
            d = tuple(r[i] - piv[j][i] for i in range(3))
            mo = np.cross(d, F)
            tau += float(np.dot(mo, axis_y))
        lim = joint_limits(j, q, p, v)
        out[f"arm_{j}"] = {"tau_Nm": tau, **lim, "sf_stall": lim["stall_Nm"] / abs(tau) if tau else math.inf,
                           "sf_rated": lim["rated_Nm"] / abs(tau) if tau else math.inf}
    tau_z = 0.0
    if load is not None:
        r, F, _ = forces[-1]
        d = (r[0] - pts["shoulder"][0], r[1] - pts["shoulder"][1], 0.0)
        tau_z = float(np.cross(d, F)[2])
    lim = joint_limits("yaw", q, p, v)
    out["arm_yaw"] = {"tau_Nm": tau_z, **lim, "sf_stall": lim["stall_Nm"] / abs(tau_z) if tau_z else math.inf,
                      "sf_rated": lim["rated_Nm"] / abs(tau_z) if tau_z else math.inf}
    return out


def workspace(p: dict | None = None, v: Variant = PERSEUS, elevation: float = 0.0, xs=None, zs=None) -> tuple:
    """Reachability of the bar root in the xz plane (y = 0) with the bar at ``elevation``: (X, Z, reachable) with
    X, Z [m] in the vehicle frame (z above the ground)."""
    p = design_params(None, v) if p is None else p
    g = arm_geometry(p, v)
    R = g["reach"]
    xs = np.linspace(g["x"] - R, g["x"] + R, 121) if xs is None else np.asarray(xs)
    zs = np.linspace(0.0, g["z_world"] + R, 101) if zs is None else np.asarray(zs)
    X, Z = np.meshgrid(xs, zs)
    ok = np.zeros(X.shape, dtype=bool)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            ok[i, j] = arm_ik((X[i, j], 0.0, Z[i, j] - g["z_hull"]), elevation, p, v) is not None
    return X, Z, ok


# ----------------------------------------------------------------------------------------------- stability and tracks
def support_point(q=None, p: dict | None = None, v: Variant = PERSEUS, load: dict | None = None) -> dict:
    """Where the ground reaction must act (x, y) [m, vehicle frame] for static equilibrium with the boom in pose
    ``q`` and the cutting ``load`` (``cutting.head_loads``, the forces the wood puts on the saw): the weighted
    centre of the masses shifted by the load's moments; with no load, the CG's footprint."""
    p = design_params(None, v) if p is None else p
    q = stow_q(v) if q is None else q
    ms = masses(p, q, None, v)
    W = sum(m for _, m, *_ in ms) * G
    mx = sum(m * x for _, m, x, _, _ in ms) * G
    my = sum(m * y for _, m, _, y, _ in ms) * G
    Fx = Fy = Fz = 0.0
    xF = yF = zF = 0.0
    if load is not None:
        pts = _arm_points(q, p, v)
        phi3, yaw = pts["phi"][2], q[0]
        fx, fy, fz = load["total"]
        c, s = math.cos(phi3), math.sin(phi3)
        fx_h, fz_h = fx * c - fz * s, fx * s + fz * c
        Fx, Fy, Fz = fx_h * math.cos(yaw) - fy * math.sin(yaw), fx_h * math.sin(yaw) + fy * math.cos(yaw), fz_h
        g = arm_geometry(p, v)
        xF, yF, zF = bar_point(q, g["Lbar"] / 2, p, v)
        zF += g["z_hull"]
    Rz = W - Fz
    x_p = (mx + zF * Fx - xF * Fz) / Rz
    y_p = (my - yF * Fz + zF * Fy) / Rz
    return {"x": x_p, "y": y_p, "R_N": Rz, "W_N": W, "load_N": (Fx, Fy, Fz)}


def tip_over(q=None, p: dict | None = None, v: Variant = PERSEUS, load: dict | None = None) -> dict:
    """Static tip-over margins [m] of the support point inside the tracks' footprint (the contact length fore and
    aft, the outer track edges sideways) with the boom in pose ``q`` and the cutting ``load``; negative = tips."""
    p = design_params(None, v) if p is None else p
    g = prr.geometry(chassis_params(p, v))
    sp = support_point(q, p, v, load)
    half_w = g["B"] / 2 + g["b"] / 2
    return {**sp, "margin_front_m": g["L"] / 2 - sp["x"], "margin_rear_m": sp["x"] + g["L"] / 2,
            "margin_side_m": half_w - abs(sp["y"]),
            "stable": g["L"] / 2 - abs(sp["x"]) > 0 and half_w - abs(sp["y"]) > 0}


def track_loads(q=None, p: dict | None = None, v: Variant = PERSEUS, load: dict | None = None) -> dict:
    """Each track's share of the ground reaction [N] and its road-wheel loads (``tracks.road_wheel_loads`` with the
    bogies) with the boom in pose ``q`` and the cutting ``load``."""
    p = design_params(None, v) if p is None else p
    cp = chassis_params(p, v)
    g = prr.geometry(cp)
    sp = support_point(q, p, v, load)
    B = g["B"]
    wl = sp["R_N"] * (0.5 + sp["y"] / B)                            # the left track carries more when y > 0
    wr = sp["R_N"] - wl
    n = len(g["wheel_x"])
    per = n // v.n_bogies
    bogies = [tuple(range(k * per, (k + 1) * per)) for k in range(v.n_bogies)] if per == 2 else None
    out = {"left_N": wl, "right_N": wr, "x_p": sp["x"]}
    for side, w in (("left", wl), ("right", wr)):
        out[f"{side}_wheels_N"] = tracks.road_wheel_loads(w, g["wheel_x"], sp["x"], bogies=bogies)
    return out


def drive_check(p: dict | None = None, v: Variant = PERSEUS, *, soil="gravel", grade_deg: float = 0.0, v_ms: float = 0.5,
                CdA: float = 0.0) -> dict:
    """The drive against the ground on ``soil`` at ``grade_deg``: the resistance (``tracks.resistance``), the two
    drives' force at the sprocket on their torque line at ``v_ms``, the traction limit, and the margin."""
    p = design_params(None, v) if p is None else p
    cp = chassis_params(p, v)
    g = prr.geometry(cp)
    W = total_mass(p, v) * G
    d = act.get(v.drive)
    r = g["r_sprocket"]
    w0 = d.no_load_rpm * 2 * math.pi / 60
    omega = v_ms / r
    F_drive = 2 * d.stall_Nm / r * max(0.0, 1 - omega / w0)
    F_rated = 2 * d.rated_Nm / r
    res = tracks.resistance(W, b=g["b"], L=g["L"], soil=soil, grade_deg=grade_deg, v=v_ms, CdA=CdA,
                            f0=v.internal_f0)
    F_trac = tracks.tractive_limit(W, b=g["b"], L=g["L"], soil=soil, grade_deg=grade_deg)
    return {"W_N": W, "resistance_N": res["total"], "drive_stall_N": F_drive, "drive_rated_N": F_rated,
            "traction_N": F_trac, "v_top_flat_m_s": w0 * r, "ok_rated": F_rated >= res["total"],
            "ok_traction": F_trac >= res["total"], "resistance": res}


# ----------------------------------------------------------------------------------------------- the robot
def _servo(joint: str, p: dict, v: Variant) -> Servo:
    kp, kd, arm = v.arm_gains[joint]
    kind, key = v.arm_drives[joint]
    if kind == "rotary":
        s = Servo.from_actuator(act.get(key), kp=kp, kd=kd, armature=arm)
        return replace(s, four_quadrant=act.get(key).self_locking)         # a brake holds: braking at any speed
    a = act.get_linear(key)
    lever_min, lever_nom = _lever_over_range(joint, p, v)
    return Servo(stall_torque=a.stall_N * lever_min, rated_torque=a.rated_N * lever_min,
                 no_load_speed=a.no_load_mm_s * 1e-3 / lever_nom, stall_current=a.stall_A, voltage=a.voltage_V,
                 kp=kp, kd=kd, armature=arm, four_quadrant=a.self_locking,
                 source=f"{key} on the {joint}'s smallest lever {lever_min * 1e3:.0f} mm ({a.source})")


def _arm(p: dict, cad: dict, v: Variant, bar_mu: float = 0.4) -> tuple:
    """The boom as a Link tree under the hull, its yaw module as a hull point mass, and (mass, Σ m x stowed)."""
    g = arm_geometry(p, v)
    m = arm_masses(p, cad, v)
    rng = {k: tuple(math.radians(a) for a in r) for k, r in v.joint_ranges.items()}
    arm_rgba, saw_rgba, bar_rgba, cyl_rgba = (0.85, 0.65, 0.12, 1.0), (0.25, 0.26, 0.28, 1.0), (0.7, 0.7, 0.72, 1.0), (0.35, 0.35, 0.38, 1.0)
    fr = (0.5, 0.005, 0.0001)
    saw = cutting.get_saw(v.saw)
    mm = 1e-3
    # the head: mount box from the wrist, the saw body behind the bar root, the bar beyond it (its own geom)
    head_geoms = [
        Geom("head_mount", "box", (g["Lh"] / 2, g["stick_w"] / 2, g["stick_d"] / 2), pos=(g["Lh"] / 2, 0.0, 0.0),
             mass=m["head mount (wrist housing, saw clamp, guard)"], role="link", friction=fr, rgba=arm_rgba),
        Geom("saw_body", "box", (g["saw_L"] / 2, g["saw_w"] / 2, g["saw_h"] / 2), pos=(g["Lh"] - g["saw_L"] / 2, 0.0, 0.0),
             mass=saw.mass_kg, role="link", friction=fr, rgba=saw_rgba),
        Geom("saw_bar", "box", (g["Lbar"] / 2, g["bar_t"] / 2, g["bar_w"] / 2), pos=(g["Lh"] + g["Lbar"] / 2, 0.0, 0.0),
             mass=None, role="link", friction=(bar_mu, 0.005, 0.0001), rgba=bar_rgba),
    ]
    head = Link("arm_head", pos=(g["Ls"], 0.0, 0.0),
                joints=[Joint("arm_wrist", axis=(0, 1, 0), range=rng["wrist"], tag="wrist", servo=_servo("wrist", p, v))],
                geoms=head_geoms)
    stick = Link("arm_stick", pos=(g["Lb"], 0.0, 0.0),
                 joints=[Joint("arm_elbow", axis=(0, 1, 0), range=rng["elbow"], tag="elbow", servo=_servo("elbow", p, v))],
                 geoms=[Geom("arm_stick", "box", (g["Ls"] / 2, g["stick_w"] / 2, g["stick_d"] / 2), pos=(g["Ls"] / 2, 0.0, 0.0),
                             mass=m["stick (Al 6082, CAD)"], role="link", friction=fr, rgba=arm_rgba)],
                 masses=[PointMass("arm_wrist_drive", m[next(k for k in m if k.startswith("wrist drive"))], (g["Ls"], 0.0, 0.0))],
                 children=[head])
    boom_masses = [PointMass("arm_cabling", m["arm cabling, saw cable and covers"] if "arm cabling, saw cable and covers" in m
                             else m[next(k for k in m if k.startswith("arm cabling"))], (g["Lb"] / 2, 0.0, 0.0))]
    boom_geoms = [Geom("arm_boom", "box", (g["Lb"] / 2, g["boom_w"] / 2, g["boom_d"] / 2), pos=(g["Lb"] / 2, 0.0, 0.0),
                       mass=m["boom (Al 6082, CAD)"], role="link", friction=fr, rgba=arm_rgba)]
    sh_key = next(k for k in m if k.startswith("shoulder drive"))
    el_key = next(k for k in m if k.startswith("elbow drive"))
    base_masses = [PointMass("arm_pedestal", m[next(k for k in m if k.startswith("pedestal"))], (0.0, 0.0, -g["ped_h"] / 2))]
    if v.arm_drives["shoulder"][0] == "linear":
        boom_masses.append(PointMass("arm_lift_cylinder", m[sh_key], (0.3 * g["Lb"], 0.0, -g["boom_d"])))
        boom_masses.append(PointMass("arm_cylinder_lugs", m[next(k for k in m if k.startswith("cylinder lugs"))], (g["Lb"] / 2, 0.0, 0.0)))
        d_cyl = p["cylinder_diameter"] * mm
        # the cylinders drawn in the ready pose on the boom (visual)
        D = PerseusRover
        a, b = D._lift_ends(p, p["boom_deg"])
        t = math.radians(p["boom_deg"])
        a_b = ((a[0] * math.cos(t) + a[1] * math.sin(t)) * mm, (-a[0] * math.sin(t) + a[1] * math.cos(t)) * mm)   # into the boom frame
        _, b0 = D._lift_ends(p, 0.0)
        boom_geoms.append(Geom("arm_lift_cylinder", "capsule", (d_cyl / 2,), fromto=(a_b[0], 0.0, a_b[1], b0[0] * mm, 0.0, b0[1] * mm),
                               role="visual", rgba=cyl_rgba))
    else:
        base_masses.append(PointMass("arm_shoulder_drive", m[sh_key], (0.0, 0.0, 0.0)))
    if v.arm_drives["elbow"][0] == "linear":
        boom_masses.append(PointMass("arm_stick_cylinder", m[el_key], (0.75 * g["Lb"], 0.0, g["boom_d"])))
        D = PerseusRover
        a0, _ = D._stick_ends(p, 0.0, 0.0)
        _, b = D._stick_ends(p, p["boom_deg"], p["stick_deg"])
        t = math.radians(p["boom_deg"])
        b_b = ((b[0] * math.cos(t) + b[1] * math.sin(t)) * mm, (-b[0] * math.sin(t) + b[1] * math.cos(t)) * mm)
        boom_geoms.append(Geom("arm_stick_cylinder", "capsule", (p["cylinder_diameter"] * mm / 2,),
                               fromto=(g["Lb"] + a0[0] * mm, 0.0, a0[1] * mm, g["Lb"] + b_b[0], 0.0, b_b[1]),
                               role="visual", rgba=cyl_rgba))
    else:
        boom_masses.append(PointMass("arm_elbow_drive", m[el_key], (g["Lb"], 0.0, 0.0)))
    boom = Link("arm_boom", joints=[Joint("arm_shoulder", axis=(0, 1, 0), range=rng["shoulder"], tag="shoulder",
                                          servo=_servo("shoulder", p, v))],
                geoms=boom_geoms, masses=boom_masses, children=[stick])
    base = Link("arm_base", pos=(g["x"], g["y"], g["z"]),
                joints=[Joint("arm_yaw", axis=(0, 0, 1), range=rng["yaw"], tag="yaw", servo=_servo("yaw", p, v))],
                geoms=[Geom("arm_pedestal", "cylinder", (g["ped_r"], g["ped_h"] / 2), pos=(0.0, 0.0, -g["ped_h"] / 2),
                            role="visual", rgba=arm_rgba)],
                masses=base_masses, children=[boom])
    yaw_mass = PointMass("arm_yaw_drive", m[next(k for k in m if k.startswith("yaw drive"))], (g["x"], g["y"], g["z"] - g["ped_h"]))
    pts = arm_mass_points(stow_q(v), p, cad, v)
    return base, yaw_mass, sum(mm_ for _, mm_, *_ in pts), sum(mm_ * x for _, mm_, x, _, _ in pts)


def perseus(overrides: dict | None = None, *, v: Variant = PERSEUS, cad: dict | None = None, bar_mu: float = 0.4,
            name: str | None = None, **chassis_kw) -> Robot:
    """The machine as a Chiron ``Robot``: ``pekari_rover_robot.pekari`` with the variant's chassis numbers and the
    boom as an extra child of the hull. ``chassis_kw``: the chassis builder's options (``track_mu``, ``bogies`` ...)."""
    p = design_params(overrides, v)
    cad = cad_numbers(p, v) if cad is None else cad
    cp = chassis_params(p, v)
    base, yaw_mass, m_arm, moment = _arm(p, cad, v, bar_mu)
    g = prr.geometry(cp)
    d = act.get(v.drive)
    r_g = g["d_wheel"] / 2 + g["link_thickness"] + p["grouser_height"] * 1e-3
    F_stall = d.stall_Nm / g["r_sprocket"]
    kd = (F_stall / prr.ROLLER_SHARE * r_g) / (0.1 / r_g)            # 0.1 m/s belt error -> the roller's share of stall
    opts = dict(track_mu=v.track_mu, internal_f0=v.internal_f0, roller_kd=kd, roller_armature=prr.ROLLER_ARMATURE * (d.mass_g / 800.0))
    opts.update(chassis_kw)
    robot = prr.pekari(params=cp, cad=cad, parts_kg=v.parts_kg, drive=v.drive, shell_t_mm=v.shell_t_mm,
                       materials=v.materials, wheel_fill=v.wheel_fill, n_bogies=v.n_bogies,
                       name=name or v.name, extra_children=[base], extra_hull_masses=[yaw_mass],
                       extra_mass_kg=m_arm, extra_moment_kgm=moment, payload_kg=0.0,
                       notes=f"{v.name}: the Pekari track architecture ({v.drive} per track) with one chainsaw boom "
                             f"({', '.join(f'{k}: {s[1]}' for k, s in v.arm_drives.items())}; saw {v.saw})",
                       sources={"geometry": "designs/perseus_rover.py", "masses": "perseus_robot.masses",
                                "actuators": "designs/actuators.py", "saw": "designs/cutting.py",
                                "assumptions": "perseus_robot.PERSEUS.assumptions"}, **opts)
    robot.nominal_qpos.update(nominal_qpos(p, v))
    robot.validate()
    robot.params, robot.arm, robot.variant = p, arm_geometry(p, v), v
    robot.saw = cutting.get_saw(v.saw)
    return robot


def perseus_lab(terrain=None, robot: Robot | None = None, v: Variant = PERSEUS, **kwargs) -> ChironLab:
    """``ChironLab(perseus(), terrain, **v.lab_options)``; ``kwargs`` override (``props``, ``welds``, ``log_geoms``...)."""
    opts = dict(v.lab_options)
    opts.update(kwargs)
    return ChironLab(robot or perseus(v=v), terrain, **opts)
