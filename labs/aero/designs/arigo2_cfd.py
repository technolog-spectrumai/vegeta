"""ARIGO-2's CFD cases (Aeromant / OpenFOAM) — prepared here, RUN BY YOU (``sims/arigo2_all.sh``), not in a session.

Two kinds of case, each an angle-of-attack sweep at 18 m/s:

- **power off** (``aero``): the airframe with its nacelles, no propellers (the folded glide, and the reference) —
  CL_max, the stall, CD0, Cm about the CG; ``rans_ksst_external``;
- **power on** (``blown``): the same with both propellers as counter-rotating rotor disks (blade elements; an
  8x6 generic planform) at a fixed rpm — how much the slipstream adds to the lift of the inner wing and how much it
  delays the stall, and its pitching moment; ``aircraft_rotor_disks``. The forces are the airframe's only (the
  propellers' thrust is the disks', in the solver log). rpm: ``RPM`` (HAND: about full power for an 8x6 on 4S at
  18 m/s; +-15 %).

Steady RANS k-omega SST, wall functions: trends, not absolutes; past the stall a mean at best (scatter written out).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import arigo1_cfd as ac
import arigo2 as a2
import arigo2_model as m2
from nisus_cfd import NU, QUALITY, RHO

ALPHAS = ac.ALPHAS
BLOWN_ALPHAS = (0.0, 8.0, 12.0, 16.0, 20.0)
SPEED = ac.SPEED
RPM = 13000.0


def reference(wing: str) -> dict:
    C = m2.coefficients(wing)
    return {"S": C["S"], "c": C["c"], "b": C["b"], "mass": C["mass"], "cg_mm": (C["x_cg_m"] * 1e3, C["z_cg_m"] * 1e3),
            "model": {k: C[k] for k in ("CL0", "CLa", "CL_MAX", "alpha_stall", "CD0", "k", "Cm0", "Cma", "x_np_m")}}


def export_stl(folder, wing, alpha_deg) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"arigo2_{wing}_a{alpha_deg:+.0f}.stl"
    if not path.exists():
        from vegeta import dedalus
        shape = a2.Arigo2().generate(**a2.variant(wing)).shape.rotate((0, 0, 0), (0, 1, 0), alpha_deg)
        dedalus.Geometry.from_cadquery(shape, name="arigo2").export_stl(str(path), tolerance=0.2)
    return path


def propeller_8x6():
    from vegeta import boreas
    d, pitch = boreas.inches(8, 6)
    return boreas.Propeller.from_pitch("8x6 (generic planform)", d, pitch, blades=2, chord_root_m=0.013, chord_max_m=0.020,
                                       chord_tip_m=0.006, mass_kg=0.010, rotor_mass_kg=0.040,
                                       notes="generic planform; the real blade's chord and twist are not in the model")


def case(workdir, wing="hybrid", *, alpha_deg=0.0, blown=False, V=SPEED, rpm=RPM, quality="screening", env=None):
    from vegeta import aeromant
    import nisus_systems as ns
    ref = reference(wing)
    stl = export_stl(Path(workdir) / "stl", wing, alpha_deg)
    params = dict(velocity=V, kinematic_viscosity=NU, density=RHO, reference_area=ref["S"], reference_length=ref["c"],
                  center_of_rotation=tuple(ac.to_flow(*ref["cg_mm"], alpha_deg).tolist()), **QUALITY[quality])
    template = "rans_ksst_external"
    if blown:
        p = a2.resolve(a2.variant(wing))
        L = a2.Arigo2.layout(p)
        pr = propeller_8x6()
        af = ns.blade_section()
        th = math.radians(alpha_deg + p["down_thrust_deg"])
        (yl, z), (yr, _) = sorted(L["props"])                   # left (y < 0) first
        cen = lambda y: (ac.to_flow(L["x_prop"], z, alpha_deg) + np.array([0.0, y / 1000.0, 0.0])).tolist()
        alphas = np.unique(np.r_[np.arange(-180, -30, 10), np.arange(-30, 31, 1), np.arange(40, 181, 10)]).astype(float)
        cl, cd = af.coefficients(np.radians(alphas))
        params.update(disk1_center=cen(yl), disk2_center=cen(yr), disk_axis=[-math.cos(th), 0.0, -math.sin(th)], diameter=pr.diameter,
                      rpm=float(rpm), blades=pr.blades, blade=[[r, b, c] for r, b, c in zip(pr.r, pr.beta_deg, pr.chord)],
                      polar=[[float(a), float(d), float(l)] for a, l, d in zip(alphas, cl, cd)],
                      rotation1=1.0 if p["inboard_up"] else -1.0, rotation2=-1.0 if p["inboard_up"] else 1.0)
        template = "aircraft_rotor_disks"
    env = env or aeromant.OpenFOAMEnvironment.detect()
    name = f"{wing}_{'blown' if blown else 'aero'}_a{alpha_deg:+.0f}_V{V:.0f}_{quality}"
    return aeromant.CFDCase(template, str(stl), params, workdir=str(Path(workdir) / name), geometry_units="mm", environment=env)


def row(wing, alpha_deg, V, result, blown=False) -> dict:
    r = ac.row(wing, "twin", alpha_deg, V, result)          # the model beside it: ARIGO-1's twin coefficients
    r["blown"] = blown
    return r


summary = ac.summary
