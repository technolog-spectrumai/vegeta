"""Perseus Mini (notebook 38): the ~30 kg garden version of Perseus as a ``perseus_robot.Variant`` — Pekari-class
tracks on 90 mm belts, a 200 W gearmotor per track, joint modules at every boom axis (no cylinders), a one-hand
pruning chainsaw (``cutting``: ``mini chainsaw 0.6 kW, 20 V``, a 150 mm bar) for branches to ~50 mm, hedges and
garden clearing. Every calculation is ``perseus_robot``'s with ``v=MINI``:

    import perseus_mini_robot as pmr
    robot = pmr.mini()                       # perseus_robot.perseus(v=MINI)
    pmr.mass_budget(), pmr.cg(), pmr.arm_torques(q), pmr.tip_over(q, load=...)
    lab = pmr.mini_lab(terrain, props=..., welds=...)
"""
from __future__ import annotations

import functools

import perseus_robot as pbr
import pekari_rover_robot as prr
from perseus_rover import MINI_ARM, MINI_CHASSIS

__all__ = ["MINI", "MINI_CAD", "mini", "mini_lab", "design_params", "cad_numbers", "arm_geometry", "arm_ik", "arm_fk",
           "bar_point", "stow_q", "arm_masses", "mass_budget", "masses", "cg", "total_mass", "joint_limits",
           "arm_torques", "workspace", "support_point", "tip_over", "track_loads", "drive_check"]

#: CAD measurements of the default Mini (``perseus_rover.PerseusMini``; mm², mm³, mm).
MINI_CAD = {
    "hull_surface_area": 668102.9, "hull_com": (4.49, 0.0, 176.83), "link_volume": 35658.2, "sprocket_volume": 349924.8,
    "idler_volume": 1039571.9, "road_wheel_volume": 361459.1, "pinion_volume": 2624.6, "basket_volume": 0.0,
    "basket_com": (0.0, 0.0, 0.0),
    "pedestal_volume": 557938.0, "boom_volume": 643036.4, "stick_volume": 521386.0, "saw_head_volume": 1963313.2,
}

MINI = pbr.Variant(
    name="Perseus Mini", design_cls="PerseusMini", chassis=dict(MINI_CHASSIS), arm=dict(MINI_ARM), cad=MINI_CAD,
    materials={**prr.MATERIALS, "Al 6082-T6 arm tubes": 2.70e-6, "Al casting pedestal": 2.70e-6},
    shell_t_mm=1.5, wheel_fill=0.35, pedestal_fill=0.4,
    parts_kg={
        "battery 24 V 14 Ah LiFePO4 (340 Wh)": 3.2,
        "computer, motor controllers, radio, IMU, GNSS receiver": 0.9,
        "sensors: stereo camera, 2D lidar, GNSS antenna, beacon (sensor box)": 1.0,
        "wiring, connectors, e-stop, switches": 0.6,
        "hull frame, deck, fasteners": 1.6,
        "track frames, bogies, axles, bearings 2x": 2.0,
    },
    arm_parts_kg={"head mount (wrist housing, saw clamp, guard)": 0.3, "arm cabling, saw cable and covers": 0.3},
    drive="perseus mini drive 20:1 planetary",
    arm_drives={"yaw": ("rotary", "smart servo 25 Nm"), "shoulder": ("rotary", "harmonic 150 Nm, brake"),
                "elbow": ("rotary", "qdd 60 Nm"), "wrist": ("rotary", "qdd 24 Nm")},
    arm_gains={"yaw": (200.0, 8.0, 0.02), "shoulder": (600.0, 30.0, 0.3), "elbow": (250.0, 12.0, 0.03),
               "wrist": (100.0, 5.0, 0.02)},
    joint_ranges={"yaw": (-170.0, 170.0), "shoulder": (-120.0, 30.0), "elbow": (-10.0, 160.0), "wrist": (-115.0, 115.0)},
    saw="mini chainsaw 0.6 kW, 20 V", wood="garden shrub (hazel, privet, fruit), green", k_feed=1.0,
    stow={"boom": 80.0, "stick": -80.0, "head": 0.0}, carry_head_deg=35.0,
    battery_Wh=340.0, electronics_W=15.0, track_mu=0.6, internal_f0=0.035,
    lab_options={"heightfield_cell": 0.02, "course_extent": (-1.5, 8.0, -3.0, 3.0), "timestep": 0.001,
                 "control_dt": 0.002, "log_dt": 0.02},
    assumptions={
        "class": "a ~30 kg class garden machine (the user's choice; 33 kg with the boom modules the cut asks for): "
                 "Pekari-class tracks, a 200 W gearmotor per track, a "
                 "one-hand pruning chainsaw on a boom of joint modules; branches to ~50 mm (the 150 mm bar: two "
                 "cuts up to 300 mm for a hedge stem), hedges, saplings, garden clearing on lawns and banks",
        "hull": "1.5 mm Al 6061 sheet over the CAD hull's area (as the Pekari)",
        "links": "PA6-GF30 links with 6 mm steel pins (the Pekari's design at 90 mm width)",
        "drive": "'perseus mini drive 20:1 planetary' (200 W): the 20 deg garden bank at 30 kg (~130 N with the losses) "
                 "within 2 x 10 N m continuous at r 0.07 m, 1.4 m/s on the flat",
        "boom_drives": "the cutting load, not the saw's weight, sizes the Mini's boom: the slow chain of a one-hand saw "
                       "(8 m/s at 390 W) pulls 49 N and is fed with as much (k_feed 1.0: a light-pushed pruning saw), "
                       "~70 N at the bar a metre out. The shoulder is a 'harmonic 150 Nm, brake' module (3.2 kg: SF > 1.5 "
                       "on stall at full reach, and it holds unpowered), the elbow a 'qdd 60 Nm', the wrist a "
                       "'qdd 24 Nm', the yaw a 'smart servo 25 Nm'; the qdd modules hold on power only",
        "arm_gains": "kp so the loaded arm sags < 2 deg at full reach; kd for zeta ~ 0.6; qfrc_bias feed-forward",
        "pedestal": "an Al casting: 40 % of the solid CAD volume",
        "saw": "the mini saw's mass is the catalogue entry's (without its battery; fed from the rover's 24 V)",
        "saw_head_cad": "the saw_head CAD volume is not used for mass (the mount is listed, the saw is the entry)",
        "battery_position": "on the hull floor at the x that balances the stowed machine over the road wheels",
        "cut_wood": "the design wood is green garden shrub; the two timber classes are in cutting.WOODS",
    },
)


def _with_mini(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        kwargs.setdefault("v", MINI)
        return fn(*args, **kwargs)
    return wrapper


design_params = _with_mini(pbr.design_params)
cad_numbers = _with_mini(pbr.cad_numbers)
arm_geometry = _with_mini(pbr.arm_geometry)
arm_ik = _with_mini(pbr.arm_ik)
arm_fk = _with_mini(pbr.arm_fk)
bar_point = _with_mini(pbr.bar_point)
stow_q = _with_mini(pbr.stow_q)
arm_masses = _with_mini(pbr.arm_masses)
mass_budget = _with_mini(pbr.mass_budget)
masses = _with_mini(pbr.masses)
cg = _with_mini(pbr.cg)
total_mass = _with_mini(pbr.total_mass)
joint_limits = _with_mini(pbr.joint_limits)
arm_torques = _with_mini(pbr.arm_torques)
workspace = _with_mini(pbr.workspace)
support_point = _with_mini(pbr.support_point)
tip_over = _with_mini(pbr.tip_over)
track_loads = _with_mini(pbr.track_loads)
drive_check = _with_mini(pbr.drive_check)


def mini(overrides: dict | None = None, **kw):
    """The Perseus Mini as a Chiron ``Robot`` (``perseus_robot.perseus(v=MINI)``)."""
    return pbr.perseus(overrides, v=MINI, **kw)


def mini_lab(terrain=None, robot=None, **kwargs):
    """``ChironLab(mini(), terrain, **MINI.lab_options)``."""
    return pbr.perseus_lab(terrain, robot, v=MINI, **kwargs)
