"""Pekari-2M — the light response platform of the Pekari family with two manipulators (``docs/pekari_pumba.md``).

The Pekari Rover (``pekari_rover.py``, ``pekari_rover_robot.py``: ~20 kg, 5 kg payload, 80 mm tracks) stays as it
is. Pekari-2M is the same track architecture stretched one size — 700 mm hull, 100 mm tracks, six road wheels on
three bogies a side, the Perseus Mini's 20:1 track drives — with two light 5-DOF arms (``pekari_arms.PEKARI_ARM``)
on the front corners of the roof and the payload deck (the Pekari basket) between them at the rear. The arms fold
back along the deck's sides (upper arm pointing back, forearm folded forward on top of it, grippers forward), so
the folded machine is 0.58 m wide and ~0.43 m high: it goes through a 0.8 m door and under a 0.5 m pipe rack.

    import pekari_2m_robot as p2m
    import pekari_family as pf
    pf.summary(p2m.PEKARI_2M)
    robot = p2m.pekari_2m()                    # Chiron Robot, arms folded, 30 kg on the deck
"""
from __future__ import annotations

import pekari_arms as pa
import pekari_family as pf
import pekari_rover_robot as prr

__all__ = ["CHASSIS", "CAD", "PEKARI_2M", "pekari_2m", "pekari_2m_lab"]

#: The chassis: Pekari's parameters one size up (mm).
CHASSIS = {
    "hull_length": 700.0, "hull_width": 340.0, "hull_height": 160.0, "hull_chamfer": 15.0, "ground_clearance": 100.0,
    "sensor_box": 90.0, "basket_length": 300.0, "basket_width": 200.0, "basket_height": 100.0, "basket_wall": 1.5,
    "basket_x": -170.0, "frame_offset": 70.0, "frame_thickness": 6.0, "track_width": 100.0, "track_pitch": 36.0,
    "link_thickness": 8.0, "grouser_height": 8.0, "guide_height": 14.0, "guide_width": 18.0, "pin_diameter": 6.0,
    "sprocket_teeth": 12, "sprocket_x": -370.0, "sprocket_z": 160.0, "idler_diameter": 130.0, "idler_x": 370.0,
    "idler_z": 120.0, "road_wheel_diameter": 80.0, "road_wheels": 6, "road_wheel_spacing": 125.0,
    "return_roller_diameter": 45.0, "return_roller_z": 190.0, "axle_diameter": 12.0, "sprocket_bore": 14.0,
    "pinion_module": 1.0, "pinion_teeth": 18, "pinion_width": 12.0, "pinion_bore": 6.0,
}

#: CAD measurements of the chassis (``pekari_rover.PekariRover`` with ``CHASSIS``; mm², mm³, mm):
#: ``pekari_family.cad_numbers(PEKARI_2M, recompute=True)`` rebuilds them.
CAD = {
    "hull_surface_area": 816200.0, "hull_com": (4.12, 0.0, 181.46), "link_volume": 39184.4, "sprocket_volume": 361470.2,
    "idler_volume": 1171173.2, "road_wheel_volume": 410593.6, "pinion_volume": 2624.6, "basket_volume": 257663.5,
    "basket_com": (0.0, 0.0, 37.18),
}

#: Rescue/industrial colours: safety orange arms, graphite hull (RAL 2009 / RAL 7024 class).
COLORS = {"arm": (0.95, 0.45, 0.05, 1.0), "stabilizer": (0.95, 0.45, 0.05, 1.0), "hull": (0.20, 0.21, 0.22, 1.0)}

PEKARI_2M = pf.Platform(
    name="Pekari-2M", chassis=dict(CHASSIS), cad=CAD,
    mounts=(pf.Mount("L", pa.PEKARI_ARM, x=0.17, y=0.135), pf.Mount("R", pa.PEKARI_ARM, x=0.17, y=-0.135)),
    stabilizers=(),
    parts_kg={
        "battery 48 V 20 Ah LiFePO4 (0.96 kWh)": 9.0,
        "computer, motor and arm controllers, radio mesh node, IMU, 48/24 V converter": 2.2,
        "sensors: 2 RGB-D cameras, 3D lidar (option), thermal camera (option), lights (sensor box)": 1.8,
        "wiring, connectors, e-stop, switches": 1.2,
        "hull frame, deck, fasteners": 3.5,
        "track frames, bogies, axles, bearings 2x": 4.0,
    },
    drive="perseus mini drive 20:1 planetary",
    materials=dict(prr.MATERIALS), shell_t_mm=2.0, wheel_fill=0.35,
    bus_V=48.0, battery_Wh=960.0, electronics_W=55.0,
    payload_kg=30.0, payload_range_kg=(20.0, 40.0), v_design=1.4,
    sensors=("rgbd_front", "rgbd_wrist", "lidar_3d (option)", "thermal (option)", "imu"),
    colors=COLORS,
    bom_extra={
        "chassis: hull shell and frame (sheet Al, machined), deck, track frames, bogies, axles, bearings": (1, 1500.0, 3500.0),
        "tracks: PA6-GF30 links (printed or moulded), pins, wheels, sprockets, 2 sides": (1, 600.0, 1800.0),
        "battery 48 V 20 Ah LiFePO4 with BMS, charger": (1, 500.0, 900.0),
        "48/24 V converter, motor and arm controllers": (1, 600.0, 1200.0),
        "computer (embedded GPU module class), IMU": (1, 900.0, 1800.0),
        "mesh radio node and safety (e-stop) radio receiver": (1, 900.0, 2700.0),
        "RGB-D cameras (front, 2 wrists)": (3, 300.0, 500.0),
        "3D lidar (option)": (1, 1500.0, 4000.0),
        "thermal camera (option)": (1, 600.0, 2500.0),
        "parallel grippers with quick-change flanges": (2, 400.0, 1500.0),
        "wiring, connectors, e-stop, lights, covers": (1, 600.0, 1500.0),
    },
    assumptions={
        "class": "a ~60 kg confined-space response platform: the Pekari track architecture one size up (the 20 kg "
                 "Pekari's 80 mm tracks and 100 W drives do not carry 20-40 kg plus two arms); unchanged Pekari code",
        "drive": "the Perseus Mini's 20:1 track drives (24 V, 10 N m continuous): 24 V from the 48 V bus through a "
                 "converter in the electronics; checked by drive_check on gravel at 20 deg with 30 kg",
        "battery": "48 V 20 Ah LiFePO4 on the hull floor, placed by prr.masses to balance the folded platform",
        "bus": "one 48 V bus for the arm modules (the strain-wave modules of the catalogue are 48 V), shared with Pumba",
        "arms": "two pekari_arms.PEKARI_ARM; masses from tube lengths and catalogue modules (no CAD weighing)",
        "payload": "30 kg nominal on the deck (target 20-40 kg), its centre 0.1 m above the deck floor",
        "shell": "2 mm Al 6061 over the CAD hull area (Pekari 1.5 mm): the arm pedestals' loads go into the frame",
        "sensors": "front RGB-D in the sensor box, an RGB-D at each wrist (in the gripper mass), lidar and thermal "
                   "camera optional (their mass is in the sensor row either way)",
    },
)


def pekari_2m(**kw):
    """Pekari-2M as a Chiron ``Robot`` (``pekari_family.robot`` keywords: ``payload_kg``, ``pose_name``, ...)."""
    return pf.robot(PEKARI_2M, **kw)


def pekari_2m_lab(terrain=None, robot=None, **kwargs):
    return pf.lab(PEKARI_2M, terrain, robot or pekari_2m(), **kwargs)
