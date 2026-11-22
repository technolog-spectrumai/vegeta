"""STERNA's CFD cases (Aeromant / OpenFOAM), simplified (notebook 51).

The question the lattice and the hand estimates cannot answer: the drag of the real fuselage with its rails and
tail, and the pitching moment about the CG with the fuselage in it (the neutral point). The cases: the whole airframe
without the propeller at four angles of attack at the design cruise speed: three in the attached range for the
fit, one at 8 degrees to see where the lift curve starts to bend.

Simplified means: the ``screening`` mesh (minutes per case on 4 cores), steady RANS k-omega SST with wall functions,
no transition model, no propeller, no control deflections, no sideslip. At a chord Reynolds number of 2e5 the real
boundary layers are partly laminar with separation bubbles: the drag is an over-estimate of the friction and an
unreliable estimate of separation. Trends, not absolutes; the stall is not in these cases.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

import sterna as st
import sterna_model as mo
from nisus_cfd import NU, QUALITY, RHO

ALPHAS = (-2.0, 1.0, 4.0, 8.0)
FIT_MAX_ALPHA = 4.0            # the model is fitted to the cases up to here: at 8 degrees the screening mesh already shows the lift curve bending over
SPEED = 14.0


def environment():
    """The machine's OpenFOAM; ``FOAM_PREFIX`` names a conda prefix that ``detect`` does not look in."""
    from vegeta import aeromant
    prefix = os.environ.get("FOAM_PREFIX")
    return aeromant.OpenFOAMEnvironment.conda(prefix) if prefix else aeromant.OpenFOAMEnvironment.detect()


def reference(case: str = "daylight_1") -> dict:
    C = mo.coefficients(case)
    return {"S": C["S"], "c": C["c"], "b": C["b"], "cg_mm": (C["x_cg_m"] * 1e3, C["z_cg_m"] * 1e3)}


def l_ref(ref) -> float:
    """The CFD's reference length [m]: the mean chord, unless the body is over 4 of them (the template's limit);
    then body/3.9, and ``row`` scales Cm back to the mean chord."""
    return max(ref["c"], 1.5 / 3.9)


def to_flow(x_mm, z_mm, alpha_deg):
    a = math.radians(alpha_deg)
    x, z = x_mm / 1000, z_mm / 1000
    return np.array([x * math.cos(a) + z * math.sin(a), 0.0, -x * math.sin(a) + z * math.cos(a)])


def export_stl(folder, alpha_deg) -> Path:
    from vegeta import dedalus
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"sterna_a{alpha_deg:+.0f}.stl"
    if not path.exists():
        shape = st.Sterna().generate().shape.rotate((0, 0, 0), (0, 1, 0), alpha_deg)
        dedalus.Geometry.from_cadquery(shape, name="sterna").export_stl(str(path), tolerance=0.2)
    return path


def case(workdir, *, alpha_deg=0.0, V=SPEED, quality="screening", env=None):
    from vegeta import aeromant
    ref = reference()
    stl = export_stl(Path(workdir) / "stl", alpha_deg)
    params = dict(velocity=V, kinematic_viscosity=NU, density=RHO, reference_area=ref["S"], reference_length=l_ref(ref),
                  center_of_rotation=tuple(to_flow(*ref["cg_mm"], alpha_deg).tolist()), **QUALITY[quality])
    return aeromant.CFDCase("rans_ksst_external", str(stl), params, workdir=str(Path(workdir) / f"a{alpha_deg:+.0f}_V{V:.0f}_{quality}"),
                            geometry_units="mm", environment=env or environment())


def row(alpha_deg, result, V=SPEED) -> dict:
    """One solved case as a row: Cl, Cd, Cm about the design CG on the mean chord, L/D, the scatter of the last iterations."""
    ref = reference()
    m = result.metrics
    k = l_ref(ref) / ref["c"]
    return {"alpha_deg": alpha_deg, "V": V, "Cl": m["Cl_mean_last50"], "Cd": m["Cd_mean_last50"], "Cm_cg": m["Cm_mean_last50"] * k,
            "L_D": m["Cl_mean_last50"] / m["Cd_mean_last50"], "x_cg_m": ref["cg_mm"][0] / 1e3, "status": result.status,
            "cells": m["mesh_cells"], "iterations": m["iterations"], "converged": m["converged"],
            "Cl_std_last50": m["Cl_std_last50"], "Cd_std_last50": m["Cd_std_last50"]}
