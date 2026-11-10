"""Pumba — the heavy derivative of the Pekari family (``docs/pekari_pumba.md``): the same track architecture,
software, 48 V electrical architecture, arm conventions and interfaces as Pekari-2M, at WILK-class size.

Reference: the Stohid Technology WILK UGV as reported by its maker (~180 kg empty, 200 kg payload, 8 km/h,
1700 x 1170 x 700 mm). Pumba takes its envelope class (1.65 x 1.15 x ~0.75 m folded) and principles — a low,
wide tracked chassis, the battery low between the tracks, a flat equipment deck, the sensors high at the front,
service hatches on the roof — not its geometry. Pumba is heavier than WILK because it carries two arms (WILK
carries none).

Two arms by default — the 6-DOF heavy arm ``main`` (``pekari_arms.PUMBA_MAIN_ARM``, 60 kg at 1.2 m) on the front
left and the 5-DOF auxiliary arm ``aux`` (``PUMBA_AUX_ARM``) on the front right — and four swing-out stabilizers
(two ahead of the tracks, two behind), deployed for heavy lifts at reach. The deck (the Pekari basket, 600 x 380
mm) sits between the folded arms at the rear.

    import pumba_robot as pr
    import pekari_family as pf
    pf.summary(pr.PUMBA)
    pf.lift_capacity(pr.PUMBA, "main", 1.2, stabilizers_deployed=True)
    robot = pr.pumba()
"""
from __future__ import annotations

import pekari_arms as pa
import pekari_family as pf
import pekari_rover_robot as prr

__all__ = ["CHASSIS", "CAD", "PUMBA", "WILK_REFERENCE", "pumba", "pumba_lab"]

#: Manufacturer-reported values of the reference (not verified here; see docs/pekari_pumba.md §2).
WILK_REFERENCE = {"empty_kg": 180.0, "payload_kg": 200.0, "v_max_m_s": 8 / 3.6, "length_m": 1.70, "width_m": 1.17,
                  "height_m": 0.70}

#: The chassis: Pekari's parameters at WILK class (mm).
CHASSIS = {
    "hull_length": 1300.0, "hull_width": 700.0, "hull_height": 330.0, "hull_chamfer": 35.0, "ground_clearance": 170.0,
    "sensor_box": 160.0, "basket_length": 600.0, "basket_width": 380.0, "basket_height": 150.0, "basket_wall": 2.0,
    "basket_x": -320.0, "frame_offset": 125.0, "frame_thickness": 10.0, "track_width": 200.0, "track_pitch": 60.0,
    "link_thickness": 16.0, "grouser_height": 14.0, "guide_height": 24.0, "guide_width": 30.0, "pin_diameter": 12.0,
    "sprocket_teeth": 15, "sprocket_x": -660.0, "sprocket_z": 300.0, "idler_diameter": 260.0, "idler_x": 660.0,
    "idler_z": 220.0, "road_wheel_diameter": 160.0, "road_wheels": 8, "road_wheel_spacing": 165.0,
    "return_roller_diameter": 90.0, "return_roller_z": 340.0, "axle_diameter": 25.0, "sprocket_bore": 30.0,
    "pinion_module": 2.0, "pinion_teeth": 18, "pinion_width": 25.0, "pinion_bore": 12.0,
}

#: CAD measurements of the chassis (``pekari_rover.PekariRover`` with ``CHASSIS``; mm², mm³, mm).
CAD = {
    "hull_surface_area": 3141692.9, "hull_com": (5.45, 0.0, 337.09), "link_volume": 236980.8,
    "sprocket_volume": 817688.3, "idler_volume": 9808913.1, "road_wheel_volume": 3516931.3, "pinion_volume": 21871.4,
    "basket_volume": 1073792.0, "basket_com": (0.0, 0.0, 46.87),
}

COLORS = {"arm": (0.98, 0.78, 0.05, 1.0), "stabilizer": (0.98, 0.78, 0.05, 1.0), "hull": (0.20, 0.21, 0.22, 1.0)}

PUMBA = pf.Platform(
    name="Pumba", chassis=dict(CHASSIS), cad=CAD,
    mounts=(pf.Mount("main", pa.PUMBA_MAIN_ARM, x=0.33, y=0.24), pf.Mount("aux", pa.PUMBA_AUX_ARM, x=0.33, y=-0.26)),
    stabilizers=(pf.Stabilizer("FL", x=0.80, y=0.80, pad_radius=0.09, drive="boom cylinder 12 kN", stroke=0.25, mass_kg=14.0),
                 pf.Stabilizer("FR", x=0.80, y=-0.80, pad_radius=0.09, drive="boom cylinder 12 kN", stroke=0.25, mass_kg=14.0),
                 pf.Stabilizer("RL", x=-0.75, y=0.80, pad_radius=0.09, drive="boom cylinder 12 kN", stroke=0.25, mass_kg=14.0),
                 pf.Stabilizer("RR", x=-0.75, y=-0.80, pad_radius=0.09, drive="boom cylinder 12 kN", stroke=0.25, mass_kg=14.0)),
    parts_kg={
        "battery 48 V 100 Ah LiFePO4 (4.8 kWh), two swappable 50 Ah modules": 44.0,
        "computer, motor and arm controllers, radio mesh node, IMU, GNSS": 5.0,
        "sensors: 2 RGB-D cameras, 3D lidar, thermal camera, radar, lights (sensor box)": 5.5,
        "wiring, connectors, e-stop, switches, contactors": 7.0,
        "hull frame (steel subframe under the arm pedestals), deck, fasteners": 38.0,
        "track frames, bogies, axles, bearings 2x": 32.0,
    },
    drive="pumba track drive 4 kW, 48 V",
    materials=dict(prr.MATERIALS), shell_t_mm=3.0, wheel_fill=0.35,
    bus_V=48.0, battery_Wh=4800.0, electronics_W=120.0,
    payload_kg=150.0, payload_range_kg=(100.0, 200.0), v_design=8 / 3.6,
    sensors=("rgbd_front", "rgbd_wrist", "lidar_3d", "thermal", "radar", "gnss", "imu"),
    colors=COLORS,
    bom_extra={
        "chassis: steel subframe, Al shell, deck, service hatches": (1, 5000.0, 11000.0),
        "tracks: 200 mm links or a rubber band track, wheels, sprockets, idlers, tensioners, 2 sides": (1, 3000.0, 7000.0),
        "battery 48 V 100 Ah LiFePO4 (2 swappable modules) with BMS, charger": (1, 2500.0, 4500.0),
        "motor and arm controllers, contactors, DC/DC": (1, 1500.0, 3500.0),
        "computer (embedded GPU module class), IMU, RTK GNSS": (1, 1500.0, 3500.0),
        "mesh radio node and safety (e-stop) radio receiver": (1, 900.0, 2700.0),
        "RGB-D cameras (front, wrist)": (2, 300.0, 500.0),
        "3D lidar": (1, 2000.0, 6000.0),
        "thermal camera": (1, 1500.0, 4000.0),
        "radar (dust and smoke)": (1, 500.0, 2000.0),
        "heavy two-jaw grab with quick coupler": (1, 1500.0, 4000.0),
        "parallel gripper (auxiliary arm)": (1, 600.0, 2000.0),
        "stabilizer legs, hinges and pads (fabrication)": (4, 400.0, 900.0),
        "wiring, connectors, e-stop, lights, covers": (1, 1500.0, 3500.0),
    },
    assumptions={
        "class": "a WILK-class envelope (manufacturer-reported reference values, WILK_REFERENCE) on the Pekari track "
                 "architecture; heavier than WILK because two arms and stabilizers are carried",
        "links": "PA6-GF30 links with 12 mm steel pins as Pekari and Perseus: at Pumba's load a rubber-pad steel or "
                 "rubber-band track is the likely answer — the link and pin stresses are not checked yet (open)",
        "drive": "one 'pumba track drive 4 kW, 48 V' (catalogue class value) per track, sized on the 20 deg grade "
                 "with 150 kg on the deck (drive_check)",
        "battery": "48 V 100 Ah LiFePO4 in two swappable 50 Ah modules on the hull floor, placed by prr.masses to "
                   "balance the folded platform",
        "frame": "a steel subframe under the arm pedestals carries the main arm's ~2.5 kN m (listed mass, no FEA yet)",
        "stabilizers": "four swing-out legs with 12 kN electric cylinders (catalogue), pads at (±0.80 / -0.75, ±0.80) "
                       "m; each leg 14 kg; the pads are on the ground for lifts at reach",
        "arms": "pekari_arms.PUMBA_MAIN_ARM (6 DOF + grab) and PUMBA_AUX_ARM (5 DOF + gripper): tube masses from "
                "the lengths, catalogue modules; deflection and the tube stresses are open",
        "payload": "150 kg nominal on the deck (target 100-200 kg), its centre 0.1 m above the deck floor",
    },
)


def pumba(**kw):
    """Pumba as a Chiron ``Robot`` (``pekari_family.robot`` keywords: ``payload_kg``, ``pose_name``,
    ``stabilizers_deployed``)."""
    return pf.robot(PUMBA, **kw)


def pumba_lab(terrain=None, robot=None, **kwargs):
    return pf.lab(PUMBA, terrain, robot or pumba(), **kwargs)
