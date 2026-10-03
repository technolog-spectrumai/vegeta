"""The leg: hand checks of its pins, the peak foot force of a gait, the two-link leg's joint torques and the FEA of
plate legs. Lifted as they were from notebooks 11, 16, 18, 19 and 20 (each had its own copy, the constants as
arguments here).

The legs themselves stay with their machines (copied unchanged): the IK in ``gait`` (``ik_two_link_planar``,
``ik_dog``, ``ik_myropod``), the plate-link leg CAD in ``onager.OnagerSentinel`` (parts ``upper_leg``, ``lower_leg``)
and ``robot_dog.RobotDog``, the bar leg in ``myropod.Myropod`` and ``apheloria.Apheloria``, the Chiron leg builder
``myropod_robot._leg``, the actuators in ``actuators``.
"""
from __future__ import annotations

import math

import numpy as np
from vegeta import talos

from . import gait

AL7075 = dict(name="Al 7075-T6", youngs_modulus=71700.0, poissons_ratio=0.33, density=2.81e-9, yield_strength=503.0,
              source="handbook, plate; weld-free machined part")                                   # notebook 20 cell 19


def pin_bending(F, lever_mm, d_mm):
    """Bending stress [MPa] of a pin as a cantilever (11 cell 16, 16 cell 16, 20 cell 23)."""
    return F * lever_mm / (math.pi * d_mm ** 3 / 32)


def pin_shear(F, d_mm, n=2):
    """Shear stress [MPa] of a pin in ``n``-fold shear (16 cell 16: double shear; 20 cell 23)."""
    return F / (n * math.pi * d_mm ** 2 / 4)


def foot_peak(W, beta, legs=4):
    """Peak foot (or wheel) force of a gait with duty factor ``beta``: a half-sine over the stance (16 cell 8, 18 cell 7,
    19 cell 13, 20 cell 14 ``wheel_peak``)."""
    return (math.pi / 2) * W / (legs * beta)


def two_link_torques(x_axle, fx, fz, *, L1, L2, h_axle, r_wheel):
    """Shoulder and knee torques [N m] holding a ground force (fx forward, fz up) under an axle ``x_axle`` [m] ahead of
    the shoulder, the hull at the standing height (notebook 20 cell 14, the Onager's wheel-leg)."""
    ik = gait.ik_two_link_planar(x_axle * 1000, -h_axle * 1000, L1 * 1000, L2 * 1000)
    if ik is None:
        return float("nan"), float("nan")
    a1, a2 = map(math.radians, ik)
    knee = np.array([-L1 * math.sin(a1), -L1 * math.cos(a1)])
    contact = np.array([x_axle, -h_axle - r_wheel])
    return contact[0] * fz - contact[1] * fx, (contact[0] - knee[0]) * fz - (contact[1] - knee[1]) * fx


def to_lower(fx, fz, a2):
    """A ground force into the lower leg's frame (20 cell 22)."""
    return fx * math.cos(a2) - fz * math.sin(a2), fx * math.sin(a2) + fz * math.cos(a2)


def to_upper(fx, fz, a1):
    """A ground force into the upper leg's frame (20 cell 22)."""
    return fx * math.cos(a1) + fz * math.sin(a1), -fx * math.sin(a1) + fz * math.cos(a1)


def plate_leg_regions(p: dict) -> dict:
    """The joint bores of the Onager's plate legs (20 cell 19): lower leg knee + axle, upper leg shoulder + knee."""
    t_low, t_up = p["lower_leg_thickness"], p["upper_leg_thickness"]
    ra, rp = p["axle_diameter"] / 2 + 0.5, p["pin_diameter"] / 2 + 0.5
    L2mm, L1mm = p["lower_leg_length"], p["upper_leg_length"]
    return {"lower": [talos.SurfacesInBox("knee", (-ra, -t_low, -ra, ra, t_low, ra)),
                      talos.SurfacesInBox("axle", (-ra, -t_low, -L2mm - ra, ra, t_low, -L2mm + ra))],
            "upper": [talos.SurfacesInBox("shoulder", (-rp, -t_up, -rp, rp, t_up, rp)),
                      talos.SurfacesInBox("knee", (-rp, -t_up, -L1mm - rp, rp, t_up, -L1mm + rp))]}


def plate_leg_models(lower_step, upper_step, p: dict, force: dict, *, a1: float, a2: float, name: str,
                     element_mm: float = 10.0, min_element_mm: float = 3.0, material: dict | None = None) -> dict:
    """The lower leg (held at the knee, the force at the axle) and the upper leg (held at the shoulder, the force at
    the knee) under one ground force ``{"fx", "fy", "fz"}`` [N] (20 cells 19 and 22)."""
    mat = talos.Material(**(material or AL7075))
    reg = plate_leg_regions(p)
    lx, lz = to_lower(force["fx"], force["fz"], a2)
    ux, uz = to_upper(force["fx"], force["fz"], a1)
    mesh = talos.MeshSettings(element_size=element_mm, min_element_size=min_element_mm)
    return {f"lower_{name}": talos.StructuralModel(lower_step, "mm-N-MPa", mat, reg["lower"], [talos.FixedSupport("knee")],
                                                   [talos.Force("axle", fx=lx, fy=force["fy"], fz=lz)], mesh, name=f"lower_{name}"),
            f"upper_{name}": talos.StructuralModel(upper_step, "mm-N-MPa", mat, reg["upper"], [talos.FixedSupport("shoulder")],
                                                   [talos.Force("knee", fx=ux, fy=force["fy"], fz=uz)], mesh, name=f"upper_{name}")}
