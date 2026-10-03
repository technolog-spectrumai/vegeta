"""The quadcopter frame's analyses (notebook 08 Part 1): the printed frame under full thrust and a hard landing (Talos)
and the frame with a canopy in forward flight (Aeromant). Lifted from cells 13 and 29 as they were."""
from __future__ import annotations

import math
from pathlib import Path

from vegeta import aeromant, talos

PETG_CF = dict(name="PETG-CF", youngs_modulus=4800.0, poissons_ratio=0.38, density=1.25e-9, yield_strength=45.0,
               source="nominal filament datasheet values, XY orientation")
DENSITY_G_MM3 = 1.25e-3                                  # PETG-CF, nominal (cell 9)
ELEMENT_MM = {"smoke": 8.0, "quick": 6.0, "full": 4.0}   # full: the notebook's
CANOPY_FIDELITY = {"smoke": dict(iterations=60, surface_level=3, near_level=2, wake_level=1, cells_per_length=2.0),
                   "quick": dict(iterations=250, surface_level=3, near_level=2, wake_level=1, cells_per_length=2.0),
                   "full": dict(iterations=250, surface_level=3, near_level=2, wake_level=1, cells_per_length=2.0)}


def frame_regions(p: dict) -> list:
    """Bolt-hole cylinders of each motor (motor0..3) and of the stack (stack0..3), by bounding box."""
    R, m, r = p["wheelbase"] / 2, p["motor_pattern"] / 2, p["motor_hole"] / 2 + 0.2
    regions = []
    for k in range(4):
        a = math.radians(45 + 90 * k)
        cx, cy = R * math.cos(a), R * math.sin(a)
        regions.append(talos.SurfacesInBox(f"motor{k}", (cx - m - r, cy - m - r, -0.1, cx + m + r, cy + m + r, 100.0)))
    s, r = p["stack_pattern"] / 2, p["stack_hole"] / 2 + 0.2
    for k, (sx, sy) in enumerate([(s, s), (-s, s), (s, -s), (-s, -s)]):
        regions.append(talos.SurfacesInBox(f"stack{k}", (sx - r, sy - r, -0.1, sx + r, sy + r, 100.0)))
    return regions


def frame_models(step: Path, p: dict, *, max_thrust_per_motor_N: float, landing_N: float,
                 element_mm: float) -> dict[str, talos.StructuralModel]:
    """``max_thrust`` (every motor at full thrust) and ``hard_landing`` (one pad takes the landing), held at the stack."""
    def model(loads, name):
        return talos.StructuralModel(step, "mm-N-MPa", talos.Material(**PETG_CF), frame_regions(p),
                                     supports=[talos.FixedSupport(f"stack{k}") for k in range(4)], loads=loads,
                                     mesh_settings=talos.MeshSettings(element_size=element_mm), name=name)
    return {"max_thrust": model([talos.Force(f"motor{k}", fz=max_thrust_per_motor_N) for k in range(4)], "max_thrust"),
            "hard_landing": model([talos.Force("motor0", fz=landing_N)], "hard_landing")}


def canopy_params(p: dict, speed_m_s: float, fidelity: str) -> dict:
    """Forward flight of the frame with its canopy (cell 29): reference area the centre plate."""
    return dict(velocity=speed_m_s, kinematic_viscosity=1.5e-5, density=1.2, reference_area=(p["plate_size"] / 1000) ** 2,
                reference_length=0.1, center_of_rotation=(0, 0, 0), **CANOPY_FIDELITY[fidelity])


def canopy_case(stl: Path, params: dict, workdir: Path, env: aeromant.OpenFOAMEnvironment) -> aeromant.CFDCase:
    return aeromant.CFDCase("rans_ksst_external", stl, params, workdir=workdir, geometry_units="mm", environment=env)
