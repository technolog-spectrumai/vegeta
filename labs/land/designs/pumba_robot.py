"""Pumba — the heavy derivative of the Pekari family, redesigned light (``docs/pekari_pumba.md``): the same track
architecture, software, 48 V electrical architecture, arm conventions and interfaces as Pekari-2M, at WILK class
in envelope **and** chassis mass. The first, heavier design (444 kg empty) is now Calydon (``calydon_robot``).

Reference: the Stohid Technology WILK UGV as reported by its maker (~180 kg empty, 200 kg payload, 8 km/h,
1700 x 1170 x 700 mm; ``WILK_REFERENCE``). Pumba's chassis — everything but the arms and stabilizers — is ~170 kg,
WILK's class; the two arms and the front stabilizers (~68 kg) are what WILK does not carry.

Where the mass went, against Calydon (see ``docs/pekari_pumba.md`` §4):

* tracks: a moulded rubber band track (``band_track_kg_m``) instead of PA6 links on steel pins, 180 mm wide, six
  road wheels a side instead of eight, the wheels ribbed to 25 % instead of 35 %;
* hull: 1250 x 640 x 280 mm in 2 mm Al (Calydon 1300 x 700 x 330 in 3 mm), a welded Al 6082 frame instead of a
  steel subframe (the lighter main arm puts ~470 N m on the pedestal, Calydon's ~1060);
* drives: two 2.5 kW track drives (11 kg each) instead of 4 kW (18 kg);
* battery: 2.9 kWh (one swappable 60 Ah module) instead of 4.8 kWh;
* arms: the 6-DOF ``PUMBA_MAIN_ARM`` (30 kg at 1.0 m, 1.58 m reach, ~47 kg) and, as the second arm, Pekari-2M's
  own light arm (``PEKARI_ARM``: one arm module for the family) instead of 80 + 14 kg;
* stabilizers: two front legs (8 kN cylinders, 8 kg each) instead of four 14 kg legs.

    import pumba_robot as pr
    import pekari_family as pf
    pf.summary(pr.PUMBA)
    robot = pr.pumba()
"""
from __future__ import annotations

import pekari_arms as pa
import pekari_family as pf
import pekari_rover_robot as prr

__all__ = ["CHASSIS", "CAD", "PUMBA", "PUMBA_SAW", "TOOL_KIT", "WILK_REFERENCE", "BAND_TRACK_KG_M", "pumba", "pumba_lab",
           "chassis_mass"]

#: Manufacturer-reported values of the reference (not verified here; see docs/pekari_pumba.md §2).
WILK_REFERENCE = {"empty_kg": 180.0, "payload_kg": 200.0, "v_max_m_s": 8 / 3.6, "length_m": 1.70, "width_m": 1.17,
                  "height_m": 0.70}

#: The chassis (mm): Pekari's parameters, WILK class, light. The track pitch is a rubber track's (72 mm).
CHASSIS = {
    "hull_length": 1250.0, "hull_width": 640.0, "hull_height": 280.0, "hull_chamfer": 30.0, "ground_clearance": 160.0,
    "sensor_box": 140.0, "basket_length": 560.0, "basket_width": 340.0, "basket_height": 120.0, "basket_wall": 1.5,
    "basket_x": -300.0, "frame_offset": 130.0, "frame_thickness": 8.0, "track_width": 180.0, "track_pitch": 72.0,
    "link_thickness": 14.0, "grouser_height": 12.0, "guide_height": 22.0, "guide_width": 28.0, "pin_diameter": 10.0,
    "sprocket_teeth": 13, "sprocket_x": -640.0, "sprocket_z": 280.0, "idler_diameter": 240.0, "idler_x": 640.0,
    "idler_z": 200.0, "road_wheel_diameter": 150.0, "road_wheels": 6, "road_wheel_spacing": 225.0,
    "return_roller_diameter": 80.0, "return_roller_z": 320.0, "axle_diameter": 22.0, "sprocket_bore": 28.0,
    "pinion_module": 2.0, "pinion_teeth": 18, "pinion_width": 22.0, "pinion_bore": 10.0,
}

#: CAD measurements of the chassis (``pekari_rover.PekariRover`` with ``CHASSIS``; mm², mm³, mm).
CAD = {
    "hull_surface_area": 2658811.6, "hull_com": (4.8, 0.0, 301.6), "link_volume": 232867.7, "sprocket_volume": 835580.6,
    "idler_volume": 7500458.2, "road_wheel_volume": 2778644.5, "pinion_volume": 20007.1, "basket_volume": 641283.5,
    "basket_com": (0.0, 0.0, 37.37),
}

#: A 180 mm moulded rubber track with steel cords and inner lugs, light-UGV class [kg/m] — assumed (no supplier
#: sheet): steel-insert mini-excavator tracks of this width are ~7 kg/m; cord-only tracks are lighter.
BAND_TRACK_KG_M = 5.0

COLORS = {"arm": (0.98, 0.78, 0.05, 1.0), "stabilizer": (0.98, 0.78, 0.05, 1.0), "hull": (0.20, 0.21, 0.22, 1.0)}

PUMBA = pf.Platform(
    name="Pumba", chassis=dict(CHASSIS), cad=CAD,
    mounts=(pf.Mount("main", pa.PUMBA_MAIN_ARM, x=0.30, y=0.21), pf.Mount("aux", pa.PEKARI_ARM, x=0.30, y=-0.22)),
    stabilizers=(pf.Stabilizer("FL", x=0.75, y=0.72, pad_radius=0.08, drive="stick cylinder 8 kN", stroke=0.22, mass_kg=8.0),
                 pf.Stabilizer("FR", x=0.75, y=-0.72, pad_radius=0.08, drive="stick cylinder 8 kN", stroke=0.22, mass_kg=8.0)),
    parts_kg={
        "battery 48 V 60 Ah LiFePO4 (2.9 kWh), one swappable module": 26.0,
        "computer, motor and arm controllers, radio mesh node, IMU, GNSS": 4.0,
        "sensors: 2 RGB-D cameras, 3D lidar, thermal camera, lights (sensor box)": 4.0,
        "wiring, connectors, e-stop, switches, contactors": 4.5,
        "hull frame (welded Al 6082, pedestal plates), deck, fasteners": 20.0,
        "track frames, bogies, axles, bearings 2x": 18.0,
    },
    drive="pumba track drive 2.5 kW, 48 V",
    materials=dict(prr.MATERIALS), shell_t_mm=2.0, wheel_fill=0.25, band_track_kg_m=BAND_TRACK_KG_M,
    bus_V=48.0, battery_Wh=2880.0, electronics_W=100.0,
    payload_kg=150.0, payload_range_kg=(100.0, 200.0), v_design=8 / 3.6,
    sensors=("rgbd_front", "rgbd_wrist", "lidar_3d", "thermal", "gnss", "imu"),
    colors=COLORS,
    bom_extra={
        "chassis: welded Al frame, Al shell, deck, service hatches": (1, 3500.0, 8000.0),
        "tracks: 180 mm rubber band tracks, wheels, sprockets, idlers, tensioners, 2 sides": (1, 2000.0, 5000.0),
        "battery 48 V 60 Ah LiFePO4 (swappable module) with BMS, charger": (1, 1600.0, 3000.0),
        "motor and arm controllers, contactors, DC/DC": (1, 1200.0, 3000.0),
        "computer (embedded GPU module class), IMU, RTK GNSS": (1, 1500.0, 3500.0),
        "mesh radio node and safety (e-stop) radio receiver": (1, 900.0, 2700.0),
        "RGB-D cameras (front, wrist)": (2, 300.0, 500.0),
        "3D lidar": (1, 2000.0, 6000.0),
        "thermal camera": (1, 1500.0, 4000.0),
        "two-jaw grab with quick coupler": (1, 1000.0, 3000.0),
        "parallel gripper with quick-change flange (aux arm)": (1, 400.0, 1500.0),
        "stabilizer legs, hinges and pads (fabrication)": (2, 300.0, 700.0),
        "wiring, connectors, e-stop, lights, covers": (1, 1000.0, 2500.0),
    },
    assumptions={
        "class": "WILK-class chassis mass (~170 kg without arms and stabilizers) and envelope (manufacturer-reported "
                 "reference values, WILK_REFERENCE) on the Pekari track architecture; the arms and stabilizers on top",
        "tracks": "a 180 mm moulded rubber band track at BAND_TRACK_KG_M = 5 kg/m (assumed) replaces the PA6 links and "
                  "pins in the mass budget; the CAD still draws the link belt as the band's envelope; the sprocket "
                  "engages the band's inner lugs at its 72 mm pitch",
        "wheels": "PA6-GF30 wheels ribbed to 25 % of the solid CAD volume (Pekari's 35 %); an Al hub with a rubber "
                  "tyre would be the alternative",
        "drive": "one 'pumba track drive 2.5 kW, 48 V' (catalogue class value) per track: 20 deg with 150 kg at peak "
                 "torque, 8 km/h on the flat (drive_check)",
        "battery": "48 V 60 Ah LiFePO4, one swappable module on the hull floor, placed by prr.masses to balance the "
                   "folded platform",
        "frame": "a welded Al 6082 frame with pedestal plates carries the main arm's ~470 N m (listed mass, no FEA yet)",
        "stabilizers": "two front swing-out legs with 8 kN electric cylinders (catalogue), pads at (0.75, ±0.72) m, "
                       "8 kg each; heavy lifts at reach on soft ground are Calydon's",
        "arms": "pekari_arms.PUMBA_MAIN_ARM (6 DOF + grab) and Pekari-2M's PEKARI_ARM as the auxiliary arm (one module "
                "across the family): tube masses from the lengths, catalogue modules; deflection open",
        "payload": "150 kg nominal on the deck (target 100-200 kg), its centre 0.1 m above the deck floor",
    },
)


#: Interchangeable tools: the main arm's quick coupler takes the grab (default) or Perseus's 2 kW chainsaw; the
#: auxiliary arm's ISO flange takes Pekari-2M's grippers. (Perseus Mini's 0.6 kW pruning saw fits the ISO flange
#: but the light arm cannot hold its cutting loads: ``pekari_arms.cut_check``.)
TOOL_KIT = {"main": ("two-jaw grab", "chainsaw 2 kW"), "aux": ("parallel gripper",)}

#: Pumba with the chainsaw on the main arm (the grab off): a branch-cutting configuration in Perseus's role.
PUMBA_SAW = pf.with_tool(PUMBA, "main", "chainsaw 2 kW", bom_rows={
    "two-jaw grab with quick coupler": None,
    "chainsaw head 2 kW with quick-coupler half and guard (36 V saw on a 48/36 V converter)": (1, 1200.0, 2600.0)})


def pumba(tool: str | None = None, **kw):
    """Pumba as a Chiron ``Robot`` with ``tool`` on the main arm (None: the grab; ``"chainsaw 2 kW"``: the saw)
    (``pekari_family.robot`` keywords: ``payload_kg``, ``pose_name``, ``stabilizers_deployed``)."""
    pl = PUMBA if tool in (None, "two-jaw grab") else pf.with_tool(PUMBA, "main", tool)
    return pf.robot(pl, **kw)


def chassis_mass(pl: pf.Platform = PUMBA) -> float:
    """The mass [kg] without the arms and the stabilizers — the number to hold against WILK's ~180 kg empty."""
    return sum(v for k, v in pf.mass_budget(pl).items() if not k.startswith(("arm '", "stabilizer '")))


def pumba_lab(terrain=None, robot=None, **kwargs):
    return pf.lab(PUMBA, terrain, robot or pumba(), **kwargs)
