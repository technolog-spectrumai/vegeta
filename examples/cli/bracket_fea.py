"""Talos model for the Dedalus example bracket (run `dedalus generate` first).

Illustrative values: aluminium 6061-T6 nominal properties; verify for real use.
"""
from pathlib import Path

import talos

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent.parent / "runs"

model = talos.StructuralModel(
    geometry=RUNS / "bracket" / "Bracket.step",
    units="mm-N-MPa",
    material=talos.Material("Al 6061-T6", youngs_modulus=68900, poissons_ratio=0.33,
                            density=2.70e-9, yield_strength=276, source="nominal handbook values"),
    regions=[talos.SurfacesOnPlane("clamped", "x", -40.0), talos.SurfacesOnPlane("loaded", "x", 40.0)],
    supports=[talos.FixedSupport("clamped")],
    loads=[talos.Force("loaded", fz=-200.0)],
    mesh_settings=talos.MeshSettings(element_size=3.0, order=2),
)
