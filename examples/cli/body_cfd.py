"""Aeromant case for the Dedalus example streamlined body (run `dedalus generate` first).

Air at 10 m/s; Re = U L / nu ~ 6.7e4 based on the 100 mm length, so the RANS template is used.
The mesh is intentionally coarse for a quick demonstration.
"""
import math
from pathlib import Path

from vegeta import aeromant

RUNS = Path(__file__).resolve().parents[2] / "runs"

case = aeromant.CFDCase(
    "rans_ksst_external_simplefoam",
    RUNS / "body" / "StreamlinedBody.stl",
    dict(velocity=10.0, kinematic_viscosity=1.5e-5, density=1.2,
         reference_area=math.pi * 0.010 ** 2, reference_length=0.100, center_of_rotation=(0.05, 0, 0),
         iterations=300),
    workdir=RUNS / "body_cfd",
    geometry_units="mm",
    environment=aeromant.OpenFOAMEnvironment.detect(),
)
