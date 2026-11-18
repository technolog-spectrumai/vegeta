"""Drag, speed and spin of a ball in flight, for the game: a few coarse steady CFD runs (Aeromant on OpenFOAM's
simpleFoam) at the ball's speeds, then a flight table built on the CFD drag.

    python ball_aero.py golf|bearing [--out file.yaml] [--processors N] [--dry]

How it is approximated (quick on purpose):
- **golf**: a smooth sphere of 42.67 mm with the fully turbulent k-omega SST model (``rans_ksst_external``). The
  dimples' job is to make the boundary layer turbulent; SST assumes it is, so a smooth sphere in SST stands in for the
  dimpled ball (no dimples to mesh). Expect Cd ~0.25..0.35 (a real ball: ~0.25).
- **bearing**: an 8 mm smooth steel ball, laminar (``laminar_external``): a smooth sphere at Re 1e4..2e5 separates
  laminar, Cd ~0.45..0.5. Steady laminar on a sphere wobbles; the Cd is averaged over the last iterations.
- **speed**: the solver is incompressible: Mach is added afterwards from a sphere's drag-rise table (Hoerner).
- **rotation**: a spinning ball needs a rotating wall, not in these templates. Spin comes from published fits instead:
  Magnus lift Cl = 0.54 S^0.4 for the golf ball (Smits & Smith 1994), ~0.2 S^0.4 for a smooth ball, spin decay from
  a moment coefficient 0.012 S, with S = omega r / V. Marked as such in the YAML.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

import common as C

BALLS = {
    "golf": dict(d=0.04267, m=0.04593, template="rans_ksst_external", speeds=(20.0, 45.0, 70.0),
                 cl_k=0.54, cd_spin=0.18, spin_rpm=3000.0, cd_hand=0.25,
                 note="smooth sphere in k-omega SST standing in for the dimpled ball"),
    "bearing": dict(d=0.008, m=0.00208, template="laminar_external", speeds=(30.0, 70.0, 130.0),
                    cl_k=0.20, cd_spin=0.10, spin_rpm=10000.0, cd_hand=0.47,
                    note="8 mm G100 stainless ball, laminar"),
}
MACH = np.array([0.0, 0.5, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0])
RISE = np.array([0.0, 0.0, 0.02, 0.08, 0.18, 0.33, 0.43, 0.48, 0.53, 0.48])      # Cd added by compressibility


def stl(name, b, workdir):
    path = Path(workdir) / f"{name}.stl"
    if not path.exists():
        import cadquery as cq
        path.parent.mkdir(parents=True, exist_ok=True)
        cq.exporters.export(cq.Workplane("XY").sphere(b["d"] * 500), str(path), tolerance=b["d"] * 1e3 / 400, angularTolerance=0.1)
    return path


def case(name, b, v, workdir):
    from vegeta import aeromant
    area = math.pi / 4 * b["d"] ** 2
    params = dict(velocity=v, kinematic_viscosity=C.NU, density=C.RHO, reference_area=area, reference_length=b["d"],
                  center_of_rotation=(0.0, 0.0, 0.0),
                  # coarse for a game: 2 cells per diameter in the background, level 5 at the ball (~d/64)
                  cells_per_length=2.0, surface_level=5, near_level=3, wake_level=2, iterations=300,
                  residual_target=1e-4, upstream=4.0, downstream=8.0, lateral=3.0, wake_length=3.0)
    if b["template"] == "rans_ksst_external":
        params.update(turbulence_intensity=0.005, viscosity_ratio=10.0)
    return aeromant.CFDCase(b["template"], str(stl(name, b, Path(workdir) / "stl")), params,
                            workdir=str(Path(workdir) / f"{name}_V{v:.0f}"), geometry_units="mm",
                            environment=aeromant.OpenFOAMEnvironment.detect())


def cd_at(v, table, b):
    """Cd at speed v: the CFD points interpolated (held flat outside), plus the compressible rise."""
    vs = sorted(table)
    cd0 = float(np.interp(v, vs, [table[s] for s in vs])) if vs else b["cd_hand"]
    return cd0 + float(np.interp(v / C.A_SOUND, MACH, RISE))


def flight(v0, elev, b, table, spin_rpm=0.0, dt=2e-4):
    """The ball's flight with the CFD drag and the fitted Magnus lift and spin decay; returns its milestones."""
    r = b["d"] / 2
    A = math.pi * r * r
    I = 0.4 * b["m"] * r * r
    a = math.radians(elev)
    x, z, vx, vz, w, t = 0.0, 0.4, v0 * math.cos(a), v0 * math.sin(a), spin_rpm * math.pi / 30, 0.0
    apex = 0.0
    while z >= 0 and t < 30:
        v = math.hypot(vx, vz)
        S = abs(w) * r / v
        cd = cd_at(v, table, b) + b["cd_spin"] * S
        cl = b["cl_k"] * min(S, 0.4) ** 0.4 if w else 0.0
        q = 0.5 * C.RHO * v * A / b["m"]
        ax = -q * (cd * vx + cl * vz)
        az = -q * (cd * vz - cl * vx) - 9.81
        w -= 0.012 * S * 0.5 * C.RHO * v * v * A * 2 * r / I * (1 if w > 0 else -1) * dt if w else 0.0
        vx, vz = vx + ax * dt, vz + az * dt
        x, z, t = x + vx * dt, z + vz * dt, t + dt
        apex = max(apex, z)
    return dict(range=x, apex=apex, time=t, v_land=math.hypot(vx, vz), spin_land_rpm=w * 30 / math.pi)


def main(argv):
    name = argv[0]
    b = BALLS[name]
    a = C.args(argv[1:], C.OUT / f"aero_{name}.yaml")
    work = C.OUT / f"aero_{name}"
    C.banner(f"{name} ball in flight: {b['note']}")
    C.log(f"ball d = {b['d'] * 1e3:.2f} mm, m = {b['m'] * 1e3:.2f} g; template {b['template']}; speeds {b['speeds']} m/s")
    C.log(f"cases in {work}" + ("   (--dry: nothing runs, hand values only)" if a.dry else f", {a.processors} processor(s)"))
    table, rows = {}, []
    for i, v in enumerate(b["speeds"], 1):
        res = None
        label = f"[{i}/{len(b['speeds'])}] {v:g} m/s"
        C.log(f"{label}: Re {v * b['d'] / C.NU:.3g}, Mach {v / C.A_SOUND:.2f}, hand Cd {b['cd_hand']}")
        if not a.dry:
            c = case(name, b, v, work)
            C.log(f"{label}: case {c.workdir}")
            res = c.ensure(run=True, processors=a.processors, progress=C.Progress(label))
            if not res.ok:
                C.log(f"{label}: FAILED: {'; '.join(res.messages)[-300:]}")
        m = res.metrics if res is not None and res.ok else {}
        if m.get("Cd") is not None:
            table[v] = float(m["Cd"])
            C.log(f"{label}: Cd = {m['Cd']:.3f}, drag {m.get('drag_force_N', float('nan')):.3g} N, "
                  f"converged {m.get('converged')}, {m.get('mesh_cells')} cells, {m.get('iterations')} iterations")
        else:
            C.log(f"{label}: NOT RUN" if a.dry else f"{label}: no Cd")
        rows.append((v, m))
    A = math.pi / 4 * b["d"] ** 2
    y = C.Yaml(f"{name} ball in flight: drag, speed and spin (game CFD, approximate)",
               [b["note"] + ".", "Cd: OpenFOAM (coarse, steady) at each speed; null = NOT RUN (the hand value is used).",
                "Magnus lift and spin decay: published fits, not CFD. Mach: a drag-rise table added on top."])
    y.section("ball").kv("diameter_m", b["d"]).kv("mass_kg", b["m"]).kv("area_m2", A, "frontal").end().blank()
    y.section("cfd", f"OpenFOAM template {b['template']}, ~2 cells/diameter background, level 5 on the ball")
    for v, m in rows:
        y.section(f"v{v:.0f}", f"{v:g} m/s, Re {v * b['d'] / C.NU:.3g}, Mach {v / C.A_SOUND:.2f}")
        y.kv("cd", m.get("Cd"), "drag coefficient on the frontal area")
        y.kv("drag_N", m.get("drag_force_N"), "drag force")
        y.kv("decel_mps2", (m["drag_force_N"] / b["m"]) if m.get("drag_force_N") else None, "drag / mass")
        y.kv("converged", m.get("converged"), "residuals reached the target")
        y.kv("cells", m.get("mesh_cells"))
        y.end()
    y.end().blank()
    y.section("game_model", "what the game should use")
    y.kv("cd_subsonic", np.mean(list(table.values())) if table else b["cd_hand"],
         "mean CFD Cd" if table else "hand value (CFD NOT RUN)")
    y.kv("cd_mach_table_M", list(MACH)).kv("cd_mach_rise", list(RISE), "added to cd_subsonic at that Mach")
    y.kv("magnus_cl", f"{b['cl_k']} * S^0.4, S = omega r / V (S <= 0.4)", "fit, not CFD")
    y.kv("cd_spin_extra", f"{b['cd_spin']} * S", "drag added by spin, fit")
    y.kv("spin_decay", "dw/dt = -0.012 S q A d / I", "moment fit (~4 %/s for a golf drive)")
    y.kv("terminal_speed_mps", math.sqrt(2 * b["m"] * 9.81 / (C.RHO * (table and np.mean(list(table.values())) or b["cd_hand"]) * A)), "falling")
    y.end().blank()
    y.section("flights", "from 0.4 m, still air; v0 m/s, elevation deg, spin rpm (backspin +) -> range, apex, time, landing")
    for v0 in b["speeds"]:
        for elev in (10, 30):
            for spin in (0.0, b["spin_rpm"]):
                f = flight(v0, elev, b, table, spin)
                y.kv(f"v{v0:.0f}_e{elev}_s{spin:.0f}", [f["range"], f["apex"], f["time"], f["v_land"], f["spin_land_rpm"]],
                     "[range m, apex m, time s, landing speed m/s, landing spin rpm]", nd=4)
    y.end()
    C.banner(f"{name}: results")
    C.table(["v m/s", "Re", "Cd CFD", "Cd used", "drag N", "decel m/s2"],
            [[v, v * b["d"] / C.NU, m.get("Cd"), cd_at(v, table, b), 0.5 * C.RHO * v * v * A * cd_at(v, table, b),
              0.5 * C.RHO * v * v * A * cd_at(v, table, b) / b["m"]] for v, m in rows])
    print(flush=True)
    C.table(["v0", "elev", "spin rpm", "range m", "apex m", "time s", "land m/s"],
            [[v0, e, s, *[flight(v0, e, b, table, s)[k] for k in ("range", "apex", "time", "v_land")]]
             for v0 in b["speeds"] for e in (10, 30) for s in (0.0, b["spin_rpm"])])
    y.write(a.out)


if __name__ == "__main__":
    main(sys.argv[1:])
