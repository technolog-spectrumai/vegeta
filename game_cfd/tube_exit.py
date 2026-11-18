"""Exit velocity of a ball from an air-cannon tube, for the game: a coarse transient CFD of the shot (OpenFOAM
rhoCentralFoam on a moving mesh, the ball a sliding piston: Vegeta's ``sonic_ball`` study) against the 0-D gun.

    python tube_exit.py thick|thin [--case slow|subsonic_fast|near_sonic ...] [--out file.yaml] [--dry]

- **thick**: the golf ejector (``golf_gun``): 600 mm tube, bore 42.7 mm, the golf ball + its sealing cup.
- **thin**: the same tanks and pressures with the 8 mm tube (``bearing_gun``): the 2.1 g steel ball.

Quick on purpose: a 2-D wedge, coarse cells (bore/8), the valve ideal (open at t = 0, full bore), at the pressure
that gives the case's speed through an ideal valve in the 0-D model. The CFD/0-D ratio then corrects the
real-valve 0-D speed, which is what the game should use. Spin out of a smooth tube: ~0 (the ball slides).
"""
from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

import common as C

CASES = ("slow", "subsonic_fast", "near_sonic")


def setup(kind):
    import golf_gun as gg
    import harpoon as hp
    import sonic_ball as sb
    import sonic_flow as sf
    if kind == "thick":
        bore_mm, mass = gg.BORE * 1e3, gg.M_BALL

        def real(c):
            return gg.gun(c.p_gauge, c.tank, c.valve)["muzzle"]

        def ideal(c, p):
            return gg.gun(p, c.tank, None)["muzzle"]
    else:
        import bearing_gun as bg
        bore_mm, mass = bg.BORE_MM, bg.MASS

        def real(c):
            return bg.shot(c)["muzzle"]

        def ideal(c, p):
            g = replace(bg.device(c), valve_orifice_mm=1e3, valve_open_s=1e-9, valve_cd=1.0)
            return hp.gun(p * bg.LEAK, g, mass, dt=2e-7)["muzzle"]
    return gg, sb, sf, bore_mm, mass, real, ideal


def ideal_pressure(c, target, ideal):
    lo, hi = 0.01, 200.0
    for _ in range(40):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if ideal(c, mid) < target else (lo, mid)
    return hi


def main(argv):
    import argparse
    kind = argv[0]
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", action="append", choices=CASES)
    ap.add_argument("--out", default=str(C.OUT / f"aero_tube_{kind}.yaml"))
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv[1:])
    gg, sb, sf, bore_mm, mass, real, ideal = setup(kind)
    pipe = bore_mm / 8                                    # cells across the radius: 4
    mesh = sf.Mesh(pipe=pipe, container_far=max(10.0, 6 * pipe), ambient_far=max(10.0, 6 * pipe))
    rows = []
    for c in gg.CASES:
        key = c.name.replace(" ", "_")
        if a.case and key not in a.case:
            continue
        c = gg.size(c)                                    # the golf device's pressure for this case
        v_real = real(c)
        p_id = ideal_pressure(c, v_real, ideal)
        cfd_v = None
        if not a.dry:
            rig = sf.Rig(pipe_bore=bore_mm, pipe_length=gg.BARREL * 1e3, pipe_wall=3.0, valve_x=20.0,
                         container_volume_l=c.tank.volume_l, container_bore=c.tank.bore_mm * 2.5, container_wall=4.0,
                         ambient_length=max(60.0, 6 * bore_mm), ambient_radius=max(30.0, 3 * bore_mm))
            vol = math.pi / 6 * (bore_mm * 1e-3) ** 3
            ball = sb.Ball(diameter=bore_mm, density=mass / vol, rear_x=20.5, stop_front_mm=gg.BARREL * 1e3 - 4)
            cond = replace(sb.BALL_CONDITIONS, p0=sf.Conditions().p_amb + p_id * 1e5)
            path = C.OUT / f"tube_{kind}" / key
            sb.solve(path, sf.AIR, rig, cond, mesh, ball, run=True, max_time=0.1,
                     chunk=max(25e-6, 2.5e-3 / max(v_real, 1.0)))
            cfd_v = sb.muzzle(sb.ball_track(path), ball, rig)["speed"]
        v_id = ideal(c, p_id)
        ratio = cfd_v / v_id if cfd_v else None
        rows.append(dict(case=key, tank=c.tank.name, valve=c.valve.name, p=c.p_gauge, p_id=p_id, v_real=v_real,
                         v_id=v_id, cfd=cfd_v, ratio=ratio, game=v_real * ratio if ratio else v_real))
    title = "golf ball ejector (thick tube, bore 42.7 mm)" if kind == "thick" else "bearing ball ejector (thin tube, bore 8 mm)"
    y = C.Yaml(f"Exit velocity: {title} (game CFD, approximate)",
               ["CFD: rhoCentralFoam, 2-D wedge, moving mesh, ball = sliding piston, coarse (bore/8 cells), ideal valve.",
                "0-D: the lumped gun with the real valve. game_exit_speed = real-valve 0-D x (CFD / ideal-valve 0-D).",
                "cfd null = NOT RUN: the game value is then the 0-D one.", f"ball mass {mass * 1e3:.2f} g, tube 600 mm"])
    for r in rows:
        y.section(r["case"], f"{r['tank']}, {r['valve']}")
        y.kv("tank_bar_g", r["p"], "the device's pressure (the golf cannon's sizing)")
        y.kv("exit_speed_0d_real_valve", r["v_real"], "m/s, lumped model with the real valve")
        y.kv("cfd_pressure_bar_g", r["p_id"], "fired through an ideal valve at this pressure in the CFD")
        y.kv("exit_speed_0d_ideal_valve", r["v_id"], "m/s, the 0-D at the CFD's conditions")
        y.kv("exit_speed_cfd", r["cfd"], "m/s, transient CFD")
        y.kv("cfd_over_0d", r["ratio"], "the correction the CFD gives the 0-D")
        y.kv("game_exit_speed", r["game"], "m/s: use this")
        y.kv("game_exit_mach", r["game"] / C.A_SOUND)
        y.kv("exit_spin_rpm", 0.0, "smooth bore, the ball slides: no spin")
        y.end().blank()
    y.write(a.out)


if __name__ == "__main__":
    main(sys.argv[1:])
