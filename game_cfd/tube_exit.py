"""Exit velocity of a ball from an air-cannon tube, for the game: a SHORT, coarse transient CFD of the start of the
shot (OpenFOAM rhoCentralFoam on a moving mesh, the ball a sliding piston: Vegeta's ``sonic_ball``) used to
calibrate the 0-D gun, which then gives the exit speed. Minutes per case, not hours.

    python tube_exit.py thick|thin [--case slow|subsonic_fast|near_sonic ...] [--travel MM] [--full] [--out f.yaml] [--dry]

- **thick**: the golf ejector (``golf_gun``): 600 mm tube, bore 42.7 mm, golf ball + sealing cup (48.9 g).
- **thin**: the same tanks and pressures with the 8 mm tube (``bearing_gun``): the 2.1 g steel ball.

How it is kept fast (it is a game):
- 2 cells across the bore's radius (bore/4), CFL 0.3, a compact tank (as long as it is wide, coarse cells);
- the CFD fires the device's own pressure through an ideal valve (open at t = 0, full bore): a quick shot;
- only the first ``--travel`` mm of the tube (default 200) are simulated; the 0-D gun with the same ideal valve is
  read at the same travel, and their ratio corrects the real-valve 0-D speed at the muzzle:
      game_exit_speed = 0-D(real valve, muzzle) x CFD(travel) / 0-D(ideal valve, travel)
- OpenFOAM is restarted rarely (long chunks), the last ones shortened to stop at the travel.
``--full`` simulates the whole 600 mm (slower). Spin out of a smooth tube: ~0 (the ball slides).
"""
from __future__ import annotations

import math
import sys
import time
from dataclasses import replace

import numpy as np

import common as C

CASES = ("slow", "subsonic_fast", "near_sonic")
REAR_MM = 20.5          # the ball's rear at the start (valve seat at 20 mm)


def setup(kind):
    import golf_gun as gg
    import harpoon as hp
    if kind == "thick":
        bore_mm, mass = gg.BORE * 1e3, gg.M_BALL

        def real(c):
            return gg.gun(c.p_gauge, c.tank, c.valve)["muzzle"]

        def ideal_at(c, travel_m):
            """0-D, ideal valve, the device's pressure: speed when the ball has moved ``travel_m``."""
            r = gg.gun(c.p_gauge, c.tank, None)
            return float(np.interp(travel_m, r["x"], r["u"]))
    else:
        import bearing_gun as bg
        bore_mm, mass = bg.BORE_MM, bg.MASS

        def real(c):
            return bg.shot(c)["muzzle"]

        def ideal_at(c, travel_m):
            g = replace(bg.device(c), valve_orifice_mm=1e3, valve_open_s=1e-9, valve_cd=1.0, barrel_m=0.03 + travel_m)
            return hp.gun(c.p_gauge * bg.LEAK, g, mass, dt=2e-7)["muzzle"]
    return gg, bore_mm, mass, real, ideal_at


def rig_for(sf, c, bore_mm, pipe_len_mm, cell):
    """A compact tank: a cylinder as long as it is wide (V = pi/4 D^3); far cells capped so the grading works."""
    D = (4 * c.tank.volume_l * 1e6 / math.pi) ** (1 / 3)
    rig = sf.Rig(pipe_bore=bore_mm, pipe_length=pipe_len_mm, pipe_wall=3.0, valve_x=20.0,
                 container_volume_l=c.tank.volume_l, container_bore=D, container_wall=4.0,
                 ambient_length=max(40.0, 4 * bore_mm), ambient_radius=max(25.0, 2.5 * bore_mm))
    far_c = max(cell, min(6 * cell, rig.container_length / 3, (D - bore_mm) / 6))
    far_a = max(cell, min(6 * cell, rig.ambient_length / 3, (rig.ambient_radius - bore_mm / 2) / 3))
    return rig, sf.Mesh(pipe=cell, container_far=far_c, ambient_far=far_a)


def main(argv):
    import argparse
    kind = argv[0]
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", action="append", choices=CASES)
    ap.add_argument("--travel", type=float, default=200.0, help="mm of the tube the CFD simulates (default 200)")
    ap.add_argument("--full", action="store_true", help="simulate the whole 600 mm tube (slower)")
    ap.add_argument("--out", default=str(C.OUT / f"aero_tube_{kind}.yaml"))
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv[1:])
    gg, bore_mm, mass, real, ideal_at = setup(kind)
    tube_mm = gg.BARREL * 1e3
    travel_mm = tube_mm - REAR_MM - bore_mm - 4 if a.full else min(a.travel, tube_mm - REAR_MM - bore_mm - 4)
    cell = bore_mm / 4
    C.banner(f"exit velocity, {kind} tube: bore {bore_mm:g} mm, {tube_mm:.0f} mm, ball {mass * 1e3:.2f} g")
    C.log(f"CFD: {cell:.2f} mm cells, CFL 0.3, first {travel_mm:.0f} mm of travel" + ("   (--dry: nothing runs)" if a.dry else ""))
    rows = []
    for c in gg.CASES:
        key = c.name.replace(" ", "_")
        if a.case and key not in a.case:
            continue
        c = gg.size(c)
        v_real = real(c)
        C.log(f"{key}: {c.tank.name}, {c.valve.name}, {c.p_gauge:.2f} bar g; 0-D real valve at the muzzle {v_real:.1f} m/s")
        cfd_v = cfd_x = v_id = None
        if not a.dry:
            import sonic_ball as sb
            import sonic_flow as sf
            # the CFD domain ends just past the travel: a short pipe is a smaller, faster mesh
            pipe_len = REAR_MM + bore_mm + travel_mm + 4 + (0 if a.full else bore_mm)
            if a.full:
                pipe_len = tube_mm
            rig, mesh = rig_for(sf, c, bore_mm, pipe_len, cell)
            vol = math.pi / 6 * (bore_mm * 1e-3) ** 3
            ball = sb.Ball(diameter=bore_mm, density=mass / vol, rear_x=REAR_MM, stop_front_mm=REAR_MM + bore_mm + travel_mm)
            cond = replace(sb.BALL_CONDITIONS, p0=sf.Conditions().p_amb + c.p_gauge * 1e5, cfl=0.3)
            # the 0-D shot time to the travel sets the chunk: ~15 OpenFOAM restarts per case
            a0 = c.p_gauge * 1e5 * math.pi / 4 * (bore_mm * 1e-3) ** 2 / mass
            t_est = math.sqrt(2 * travel_mm * 1e-3 / max(a0, 1.0)) * 1.5
            path = C.OUT / f"tube_{kind}" / key
            C.log(f"{key}: case {path}; ~{t_est * 1e3:.2f} ms of shot, chunks of {t_est / 15 * 1e3:.3f} ms")
            t0, last = time.time(), [0.0]

            def on_chunk(t, front, u, key=key, last=last):
                if time.time() - last[0] >= 3.0:
                    last[0] = time.time()
                    C.log(f"{key}: t = {t * 1e3:7.3f} ms   travel {front - REAR_MM - bore_mm:6.1f} / {travel_mm:.0f} mm   "
                          f"speed {u:6.1f} m/s")
            sb.solve(path, sf.AIR, rig, cond, mesh, ball, run=True, max_time=max(10 * t_est, 2e-3),
                     chunk=t_est / 15, on_chunk=on_chunk)
            tr = sb.ball_track(path)
            cfd_v = float(tr["u"][-1])
            cfd_x = float(tr["x"][-1] - tr["x"][0])
            v_id = ideal_at(c, cfd_x)
            C.log(f"{key}: done in {time.time() - t0:.0f} s: CFD {cfd_v:.1f} m/s after {cfd_x * 1e3:.0f} mm, 0-D ideal {v_id:.1f} m/s")
        ratio = cfd_v / v_id if cfd_v and v_id else None
        game = v_real * ratio if ratio else v_real
        C.log(f"{key}: game exit speed {game:.1f} m/s" + ("" if ratio else " (0-D only)"))
        rows.append(dict(case=key, tank=c.tank.name, valve=c.valve.name, p=c.p_gauge, v_real=v_real, x=cfd_x,
                         v_id=v_id, cfd=cfd_v, ratio=ratio, game=game))
    title = "golf ball ejector (thick tube, bore 42.7 mm)" if kind == "thick" else "bearing ball ejector (thin tube, bore 8 mm)"
    y = C.Yaml(f"Exit velocity: {title} (game CFD, approximate)",
               [f"CFD: rhoCentralFoam, 2-D wedge, moving mesh, ball = sliding piston, {cell:.2f} mm cells, ideal valve,",
                f"the device's pressure, first {travel_mm:.0f} mm of travel. 0-D: the lumped gun.",
                "game_exit_speed = 0-D real valve at the muzzle x CFD / 0-D ideal valve at the same travel.",
                "cfd null = NOT RUN: the game value is then the 0-D one.", f"ball mass {mass * 1e3:.2f} g, tube {tube_mm:.0f} mm"])
    for r in rows:
        y.section(r["case"], f"{r['tank']}, {r['valve']}")
        y.kv("tank_bar_g", r["p"], "the device's pressure")
        y.kv("exit_speed_0d_real_valve", r["v_real"], "m/s at the muzzle, lumped model with the real valve")
        y.kv("cfd_travel_m", r["x"], "how far the CFD followed the ball")
        y.kv("speed_cfd_at_travel", r["cfd"], "m/s, transient CFD (ideal valve)")
        y.kv("speed_0d_at_travel", r["v_id"], "m/s, 0-D with the same ideal valve")
        y.kv("cfd_over_0d", r["ratio"], "the CFD's correction to the 0-D")
        y.kv("game_exit_speed", r["game"], "m/s: use this")
        y.kv("game_exit_mach", r["game"] / C.A_SOUND)
        y.kv("exit_spin_rpm", 0.0, "smooth bore, the ball slides: no spin")
        y.end().blank()
    C.banner(f"{kind} tube: results")
    C.table(["case", "bar g", "0-D real", "CFD mm", "CFD m/s", "0-D ideal", "CFD/0-D", "game m/s", "Mach"],
            [[r["case"], r["p"], r["v_real"], r["x"] * 1e3 if r["x"] else None, r["cfd"], r["v_id"], r["ratio"], r["game"],
              r["game"] / C.A_SOUND] for r in rows])
    y.write(a.out)


if __name__ == "__main__":
    main(sys.argv[1:])
