"""Drako — the airframe's drag in Aeromant (OpenFOAM, steady RANS k-omega SST). Prepared here, run where OpenFOAM is.

The sizing assumes a drag area CdA of 0.25 m^2 for the empty airframe and 0.15 m^2 more for the package
(``Drako.sizing``: ``cda_body``, ``cda_package``); these cases measure them. The body is everything but the
propellers (plates, arms, motors, battery, gear, cradle, package), pitched nose down by the tilt that holds the speed
(the sizing's ``tilt_*``), the flow along +x. The rotors are left out: their induced flow over the body is the next
step (the ``aircraft_rotor_disks`` template).

Cases (``cases``): **empty_80** the empty airframe at 80 km/h; **loaded_40** with the package at 40 km/h; **loaded_60**
with the package at 60 km/h (the package's Cd at a higher Re). Read back with ``table``: Cd, drag, CdA against the
sizing's assumption. Under ``vegeta.mock`` the run returns fixed numbers (they mean nothing).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from drako import Drako

RHO, NU = 1.225, 1.5e-5
CASES = {  # name: (speed km/h, with package, sizing's tilt key, assumed CdA m^2)
    "empty_80": (80.0, False, "tilt_80kmh_empty_deg", 0.25),
    "loaded_40": (40.0, True, "tilt_40kmh_max_deg", 0.40),
    "loaded_60": (60.0, True, "tilt_40kmh_max_deg", 0.40),
}


def params(**over):
    p = {q.name: q.default for q in Drako.parameters}
    p.update(over)
    return p


def body_cad(p=None, package=True):
    """The CFD body: Drako without propeller discs (they are disks of the next model, not walls)."""
    import cadquery as cq
    p = p or params()
    d = Drako()
    body = d.build({**p, "part": "frame"}).union(d._gear(p)).union(d._cradle(p))
    R, z = p["wheelbase"] / 2, d._arm_z(p) + p["arm_diameter"] / 2 + p["mount_thickness"]
    top = 2 * p["plate_thickness"] + p["plate_gap"]
    body = body.union(cq.Workplane("XY").box(300, 160, 110).translate((0, 0, top + 55)))
    for a in d._arm_angles(p):
        c, s = math.cos(math.radians(a)), math.sin(math.radians(a))
        body = body.union(cq.Workplane("XY").center(R * c, R * s).circle(50).extrude(45).translate((0, 0, z)))
    if package:
        body = body.union(cq.Workplane("XY").box(p["package_x"], p["package_y"], p["package_z"])
                          .translate((0, 0, -p["leg_height"] + p["package_z"] / 2)))
    return body


def export_stl(shape, path, tilt_deg=0.0):
    """STL in mm, pitched nose down by ``tilt_deg`` about y (the flow is along +x)."""
    import cadquery as cq
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cq.exporters.export(shape, str(path), tolerance=0.5, angularTolerance=0.2)
    return tilt_stl(path, tilt_deg)


def tilt_stl(path, tilt_deg):
    from vegeta import aeromant
    s = aeromant.read_stl(path)
    a = math.radians(tilt_deg)              # nose (+x) down: rotate about +y by +a
    R = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    aeromant.write_stl_ascii(aeromant.Surface(s.triangles @ R.T, s.name), path)
    return Path(path)


def frontal_area_m2(p=None, package=True):
    """Projected area normal to x from the parts' boxes and tubes (the reference area; Cd is read on it)."""
    p = p or params()
    a = p["plate_diameter"] * (2 * p["plate_thickness"] + p["plate_gap"])
    a += 160 * 110                                             # battery seen from the front
    a += 4 * (p["wheelbase"] / 2) * p["arm_diameter"]          # arms not along x (approx.)
    a += 6 * 100 * 45                                          # motors
    a += 4 * p["leg_height"] * p["leg_diameter"] + 2 * p["leg_diameter"] ** 2
    if package:
        a += p["package_y"] * p["package_z"]
    return a * 1e-6


def cases(workdir, p=None, *, stls=None, iterations=600, environment=None):
    """One ``CFDCase`` per entry of ``CASES``. ``stls``: {name: path} of bodies already exported (tests, a CAD-less
    machine); otherwise the bodies are built and exported here."""
    from vegeta import aeromant
    p = p or params()
    s = Drako.sizing(p)
    env = environment or aeromant.OpenFOAMEnvironment.detect()
    out = {}
    for name, (kmh, pkg, tilt_key, _) in CASES.items():
        wd = Path(workdir) / name
        stl = Path(stls[name]) if stls else export_stl(body_cad(p, pkg), wd.parent / "stl" / f"{name}.stl", s[tilt_key])
        A = frontal_area_m2(p, pkg)
        out[name] = aeromant.CFDCase(
            "rans_ksst_external", stl,
            dict(velocity=kmh / 3.6, kinematic_viscosity=NU, density=RHO, reference_area=A,
                 reference_length=p["wheelbase"] / 1000, center_of_rotation=(0.0, 0.0, 0.0), iterations=iterations,
                 cells_per_length=6.0, surface_level=4),
            workdir=wd, geometry_units="mm", environment=env)
    return out


def run(cases_, *, run=True, processors=4):
    """Prepare every case (always) and run it (``run``); a failure is a result in the table."""
    res = {}
    for name, c in cases_.items():
        try:
            c.prepare(overwrite=True)
            res[name] = c.run(processors=processors) if run else None
            if run:
                res[name] = c.results()
        except Exception as exc:
            res[name] = exc
    return res


def table(cases_, results, p=None):
    import pandas as pd
    p = p or params()
    rows = {}
    for name, (kmh, pkg, _, cda_assumed) in CASES.items():
        r = results.get(name)
        A = frontal_area_m2(p, pkg)
        row = {"speed [km/h]": kmh, "package": pkg, "reference area [m2]": round(A, 3), "assumed CdA [m2]": cda_assumed}
        if r is None:
            row["status"] = "NOT RUN"
        elif isinstance(r, Exception):
            row.update(status="FAILED", note=f"{type(r).__name__}: {str(r)[:120]}")
        else:
            cd = r.metrics.get("Cd")
            row.update(status="solved" if r.ok else "FAILED", Cd=cd, **{"drag [N]": r.metrics.get("drag_force_N"),
                       "CdA [m2]": cd * A if cd is not None else np.nan,
                       "CdA / assumed": cd * A / cda_assumed if cd is not None else np.nan})
        rows[name] = row
    return pd.DataFrame(rows).T
