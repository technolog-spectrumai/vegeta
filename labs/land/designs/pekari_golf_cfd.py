"""The golf ball's drag by CFD (Aeromant / OpenFOAM), against the hand calculation (``pekari_golf_aero``).

Cases: the ball in free flight at the launcher's speeds (20, 45, 70 m/s: Re 5.8e4 .. 2.0e5), smooth and dimpled, at
sea level. Template ``rans_ksst_external`` (steady k-omega SST, wall functions, no prism layers) and, at 20 m/s,
``laminar_external`` as the other bound.

What to expect from it, said before it runs
-------------------------------------------
- A **smooth sphere** below the drag crisis is the classic hard case for steady RANS: the real flow separates
  laminar at ~80 degrees with an unsteady wake (Cd 0.47); fully turbulent SST separates late, as if past the
  crisis, and gives a low Cd (typically 0.3..0.4 on such meshes). The CFD is a check on the *setup* and the mesh,
  and its Cd is a lower bound for the smooth ball.
- For the **dimpled ball** the real boundary layer *is* turbulent (the dimples trip it), which is what SST assumes,
  so the steady RANS number is closer to the physics than for the smooth one: the hand value is 0.25..0.26. The
  dimples are 0.25 mm deep: they need the surface level that puts several cells across a 3.8 mm dimple
  (level 6 on a 2-cells-per-diameter background: 0.33 mm cells) — expensive; the screening runs at level 5.

The comparison table (``table``) puts each case's Cd next to ``cd_smooth`` / ``cd_dimpled`` at its Re.
"""
from __future__ import annotations

import math
from pathlib import Path

import pekari_golf_aero as aero

SPEEDS = (20.0, 45.0, 70.0)
QUALITY = {
    "screening": dict(cells_per_length=2.0, surface_level=5, near_level=3, wake_level=3, iterations=400, residual_target=1e-5,
                      upstream=5.0, downstream=10.0, lateral=4.0, wake_length=4.0),
    "fine": dict(cells_per_length=3.0, surface_level=6, near_level=4, wake_level=3, iterations=800, residual_target=1e-5,
                 upstream=6.0, downstream=12.0, lateral=5.0, wake_length=5.0),
}


def export_stl(workdir, ball="smooth"):
    from pekari_golf import PekariGolf
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    path = workdir / f"ball_{ball}.stl"
    if not path.exists():
        g = PekariGolf().generate(part="ball" if ball == "smooth" else "ball_dimpled")
        g.export_stl(str(path), tolerance=0.01 if ball == "smooth" else 0.02, angular_tolerance=0.1) \
            if "angular_tolerance" in g.export_stl.__code__.co_varnames else g.export_stl(str(path), tolerance=0.01)
    return path


def case(workdir, *, v=45.0, ball="smooth", template="rans_ksst_external", quality="screening", env=None):
    from vegeta import aeromant
    stl = export_stl(Path(workdir) / "stl", ball)
    params = dict(velocity=v, kinematic_viscosity=aero.NU, density=aero.RHO, reference_area=aero.A_BALL,
                  reference_length=aero.D_BALL, center_of_rotation=(0.0, 0.0, 0.0), **QUALITY[quality])
    if template == "rans_ksst_external":
        params.update(turbulence_intensity=0.005, viscosity_ratio=10.0)
    env = env or aeromant.OpenFOAMEnvironment.detect()
    name = f"{ball}_{template.split('_')[0]}_V{v:.0f}_{quality}"
    return aeromant.CFDCase(template, str(stl), params, workdir=str(Path(workdir) / name), geometry_units="mm", environment=env)


def specs(quality="screening"):
    out = {}
    for v in SPEEDS:
        out[f"smooth, {v:.0f} m/s, SST"] = dict(v=v, ball="smooth", quality=quality)
    out["smooth, 20 m/s, laminar"] = dict(v=20.0, ball="smooth", template="laminar_external", quality=quality)
    for v in SPEEDS:
        out[f"dimpled, {v:.0f} m/s, SST"] = dict(v=v, ball="dimpled", quality=quality)
    return out


def hand(kw):
    re = aero.reynolds(kw["v"])
    return re, (aero.cd_smooth(re) if kw["ball"] == "smooth" else aero.cd_dimpled(re))


def table(results):
    rows = []
    for name, (kw, r) in results.items():
        re, cd_h = hand(kw)
        m = getattr(r, "metrics", {}) if r is not None and getattr(r, "ok", False) else {}
        rows.append({"case": name, "Re": re, "Cd hand": cd_h, "Cd CFD": m.get("Cd"), "drag N (CFD)": m.get("drag_force_N"),
                     "converged": m.get("converged"), "cells": m.get("mesh_cells"),
                     "status": "solved" if m else ("NOT RUN" if r is None else "FAILED")})
    return rows


def run(workdir="out/golf_cfd", quality="screening", processors=4, only=None):
    results = {}
    for name, kw in specs(quality).items():
        if only and not any(o in name for o in only):
            continue
        c = case(workdir, **kw)
        try:
            r = c.ensure(run=True, processors=processors)
        except Exception as e:      # noqa: BLE001 - a failed case is a row, not a crash
            print(name, "FAILED", e)
            r = None
        results[name] = (kw, r)
        print(name, getattr(r, "metrics", {}).get("Cd") if r is not None else None, flush=True)
    return results


if __name__ == "__main__":
    import json
    import sys
    res = run(only=sys.argv[1:] or None)
    print(json.dumps(table(res), indent=1, default=str))
