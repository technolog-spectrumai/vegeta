"""The fantasy humanoids' numbers, without CadQuery: the ghoul's pincers, the ork (a ``humanoid.Humanoid`` variant
[mm, deg]) with its broom and trash-bin lid, and the materials of what they carry (Yax's materials shape)."""
from __future__ import annotations

__all__ = ["PINCER", "ORK", "ORK_EXTRA", "MATERIALS"]

#: the ghoul's pincers: jaw length, the CAD's opening per jaw, the robot's jaw range [deg]
PINCER = {"pincer_length": 150.0, "pincer_open_deg": 20.0, "jaw_range_deg": (-10.0, 45.0)}

#: the ork: Humanoid parameters (bulkier than Talia) and the extras in ``ORK_EXTRA``
ORK = {"pelvis_width": 340.0, "pelvis_depth": 220.0, "pelvis_height": 140.0, "torso_width": 470.0,
       "torso_depth": 270.0, "torso_height": 470.0, "waist_taper": 0.85, "shell_fillet": 30.0, "hip_spacing": 240.0,
       "thigh_length": 340.0, "shin_length": 330.0, "leg_width": 115.0, "leg_thickness": 95.0,
       "joint_boss_diameter": 125.0, "ankle_height": 80.0, "foot_length": 300.0, "foot_width": 150.0,
       "foot_thickness": 35.0, "knee_bend_deg": 20.0, "upper_arm_length": 340.0, "forearm_length": 320.0,
       "arm_diameter": 100.0, "elbow_bend_deg": 35.0, "neck_length": 30.0, "head_width": 240.0, "head_depth": 250.0,
       "head_height": 220.0, "eye_diameter": 26.0, "battery_bay_depth": 0.0,
       "tusk_length": 70.0, "broom_tilt_deg": 12.0, "pole_diameter": 32.0, "pole_above_hand": 380.0,
       "broom_width": 320.0, "stock_depth": 70.0, "stock_height": 45.0, "bristle_length": 160.0,
       "lid_diameter": 560.0, "lid_thickness": 1.0, "lid_dish": 40.0}
ORK_EXTRA = {"tusk_length": "tusks above the jaw", "broom_tilt_deg": "the pole forward of vertical",
             "pole_diameter": "", "pole_above_hand": "pole length above the fist", "broom_width": "the broom head",
             "stock_depth": "", "stock_height": "", "bristle_length": "", "lid_diameter": "the trash-bin lid",
             "lid_thickness": "galvanised sheet", "lid_dish": "the lid's dome height"}

MATERIALS = {
    "ash": {"density_kg_m3": 670.0, "E_pa": 12.0e9, "nu": 0.37, "yield_pa": None, "ultimate_pa": 100e6,
            "source": "European ash, 12 % moisture (Wood Handbook): broom poles and tool handles"},
    "broom straw": {"density_kg_m3": 150.0, "E_pa": None, "nu": None, "yield_pa": None, "ultimate_pa": None,
                    "source": "assumed: a packed block of broomcorn bristles (bulk density)"},
    "galvanised steel": {"density_kg_m3": 7850.0, "E_pa": 200e9, "nu": 0.29, "yield_pa": 280e6, "ultimate_pa": 380e6,
                         "source": "DX51D+Z galvanised sheet (EN 10346)"},
}
