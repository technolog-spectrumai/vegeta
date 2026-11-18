"""Quick game CFD for the recoilless 8 mm emitter (``labs/aero/exp/bearing_recoilless``): the steel ball's drag at its
new speeds, and the start of the shot in the 1.0 m tube. Coarse on purpose (minutes, not hours).

    python cfd_8mm.py [--only drag|tube] [--processors N] [--travel MM] [--out file.yaml] [--dry]

1. **drag**: an 8 mm smooth steel ball at 100, 200 and 280 m/s (Aeromant ``laminar_external``, as aero_bearing.sh,
   ~85k cells). The solver is incompressible: at 280 m/s (Mach 0.82) the drag rise is added from the Mach table and
   the CFD gives the subsonic part. Checked against the measured 0.47 (rejected if more than 35 % off).
2. **tube**: the shot's first ``--travel`` mm (default 300) in the 8 mm, 1.0 m tube at the plenum's 26.3 bar
   (rhoCentralFoam, moving mesh, ball = piston, 2 mm cells, an ideal valve: the plenum opens into the bore), against
   the 0-D gun with the same ideal valve at the same travel. Their ratio corrects the recoilless design's 0-D muzzle
   speed (280 m/s) -> game_exit_speed. The rear venturi is not in the CFD: the 0-D carries it (it lowers the chamber
   pressure; the CFD/0-D ratio is for the tube's gas dynamics).
"""
from __future__ import annotations

import math
import sys
import time
from dataclasses import replace

import numpy as np

import common as C

SPEEDS = (100.0, 200.0, 280.0)


def drag(a, y):
    import ball_aero as BA
    b = dict(BA.BALLS["bearing"], speeds=SPEEDS)
    work = C.OUT / "cfd_8mm_drag"
    C.banner("8 mm ball drag at the recoilless emitter's speeds")
    table, rows = {}, []
    for i, v in enumerate(SPEEDS, 1):
        label = f"[{i}/{len(SPEEDS)}] {v:g} m/s"
        C.log(f"{label}: Re {v * b['d'] / C.NU:.3g}, Mach {v / C.A_SOUND:.2f}")
        m = {}
        if not a.dry:
            c = BA.case("bearing", b, v, work)
            C.log(f"{label}: case {c.workdir}")
            res = c.ensure(run=True, processors=a.processors, progress=C.Progress(label))
            m = res.metrics if res.ok else {}
            if not res.ok:
                C.log(f"{label}: FAILED: {'; '.join(res.messages)[-300:]}")
        if m.get("Cd") is not None:
            table[v] = float(m["Cd"])
            C.log(f"{label}: Cd {m['Cd']:.3f}, converged {m.get('converged')}, {m.get('mesh_cells')} cells")
        else:
            C.log(f"{label}: NOT RUN" if a.dry else f"{label}: no Cd")
        rows.append((v, m))
    mean = float(np.mean(list(table.values()))) if table else None
    ok = mean is not None and abs(mean / b["cd_hand"] - 1) <= 0.35
    cd0 = mean if ok else b["cd_hand"]
    y.section("drag", "8 mm smooth steel ball, laminar_external, coarse; Mach rise added on top")
    for v, m in rows:
        y.section(f"v{v:.0f}", f"Re {v * b['d'] / C.NU:.3g}, Mach {v / C.A_SOUND:.2f}")
        y.kv("cd_cfd", m.get("Cd"), "incompressible CFD")
        y.kv("cd_game", cd0 + float(np.interp(v / C.A_SOUND, BA.MACH, BA.RISE)), "subsonic Cd + Mach rise: use this")
        y.kv("converged", m.get("converged"))
        y.end()
    y.kv("cd_cfd_mean", mean).kv("cd_measured", b["cd_hand"]).kv("cfd_accepted", ok, "within 35 % of the measurement")
    y.kv("cd_subsonic_game", cd0)
    y.end().blank()
    C.table(["v m/s", "Mach", "Cd CFD", "Cd game"], [[v, v / C.A_SOUND, m.get("Cd"),
             cd0 + float(np.interp(v / C.A_SOUND, BA.MACH, BA.RISE))] for v, m in rows])


def tube(a, y):
    import bearing_gun as bg
    import bearing_recoilless as br
    import golf_recoilless as g
    import harpoon as hp
    import tube_exit as T
    case = replace(br.CASE, p_gauge=26.3)
    bore_mm, mass, tube_mm = 8.0, bg.MASS, br.BARREL_M * 1e3
    cell = bore_mm / 4
    rear = T.rear_mm(cell)
    travel = min(a.travel, tube_mm - rear - bore_mm - 4)
    C.banner(f"8 mm recoilless emitter: the shot's first {travel:.0f} mm (tube {tube_mm:.0f} mm, {case.p_gauge} bar g)")
    th = 6.91
    v_design = g.shot(case, th, br.EPS, 1.0, case.p_gauge, proj=br.PROJ, dt=br.DT, friction=br.FRICTION)["v"]
    gun = hp.Gun(bore_mm=bore_mm, barrel_m=0.03 + travel * 1e-3, tank_l=case.tank.volume_l, tank_rated_bar=100.0,
                 valve_name="ideal", valve_orifice_mm=bore_mm, valve_cd=1.0, valve_open_s=1e-9, friction_n=br.FRICTION)
    v_id = hp.gun(case.p_gauge, gun, mass, dt=2e-7)["muzzle"]
    C.log(f"0-D: recoilless design at the muzzle {v_design:.1f} m/s; ideal valve after {travel:.0f} mm {v_id:.1f} m/s")
    cfd_v = cfd_x = None
    if not a.dry:
        import sonic_ball as sb
        import sonic_flow as sf
        rig, mesh = T.rig_for(sf, case, bore_mm, rear + bore_mm + travel + 4 + bore_mm, cell)
        vol = math.pi / 6 * (bore_mm * 1e-3) ** 3
        ball = sb.Ball(diameter=bore_mm, density=mass / vol, rear_x=rear, stop_front_mm=rear + bore_mm + travel)
        cond = replace(sb.BALL_CONDITIONS, p0=sf.Conditions().p_amb + case.p_gauge * 1e5, cfl=0.3)
        a0 = case.p_gauge * 1e5 * math.pi / 4 * (bore_mm * 1e-3) ** 2 / mass
        t_est = math.sqrt(2 * travel * 1e-3 / a0) * 1.5
        path = C.OUT / "cfd_8mm_tube"
        C.log(f"case {path}; ~{t_est * 1e3:.2f} ms of shot")
        last = [0.0]

        def on_chunk(t, front, u):
            if time.time() - last[0] >= 3.0:
                last[0] = time.time()
                C.log(f"t = {t * 1e3:7.3f} ms   travel {front - rear - bore_mm:6.1f} / {travel:.0f} mm   speed {u:6.1f} m/s")
        t0 = time.time()
        sb.solve(path, sf.AIR, rig, cond, mesh, ball, run=True, max_time=max(10 * t_est, 2e-3), chunk=t_est / 15,
                 on_chunk=on_chunk)
        tr = sb.ball_track(path)
        cfd_v, cfd_x = float(tr["u"][-1]), float(tr["x"][-1] - tr["x"][0])
        gun = replace(gun, barrel_m=0.03 + cfd_x)
        v_id = hp.gun(case.p_gauge, gun, mass, dt=2e-7)["muzzle"]
        C.log(f"done in {time.time() - t0:.0f} s: CFD {cfd_v:.1f} m/s after {cfd_x * 1e3:.0f} mm, 0-D {v_id:.1f} m/s")
    ratio = cfd_v / v_id if cfd_v else None
    game = v_design * ratio if ratio else v_design
    y.section("tube", f"8 mm bore, {tube_mm:.0f} mm, plenum {case.p_gauge} bar g, recoilless (throat {th} mm)")
    y.kv("exit_speed_0d_recoilless", v_design, "m/s, the design (0-D with the valve and the venturi)")
    y.kv("cfd_travel_m", cfd_x).kv("speed_cfd_at_travel", cfd_v, "m/s, CFD, ideal valve, no venturi")
    y.kv("speed_0d_at_travel", v_id, "m/s, 0-D, the same").kv("cfd_over_0d", ratio)
    y.kv("game_exit_speed", game, "m/s: use this").kv("game_exit_mach", game / C.A_SOUND)
    y.kv("exit_spin_rpm", 0.0, "smooth bore")
    y.end().blank()
    C.table(["0-D design", "CFD mm", "CFD m/s", "0-D same", "CFD/0-D", "game m/s"],
            [[v_design, cfd_x * 1e3 if cfd_x else None, cfd_v, v_id, ratio, game]])


def main(argv):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=("drag", "tube"))
    ap.add_argument("--processors", type=int, default=2)
    ap.add_argument("--travel", type=float, default=300.0)
    ap.add_argument("--out", default=str(C.OUT / "cfd_8mm.yaml"))
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv)
    y = C.Yaml("Recoilless 8 mm emitter: ball drag and tube exit speed (game CFD, approximate)",
               ["drag: OpenFOAM laminar_external, coarse, steady; the Mach rise added from a table.",
                "tube: rhoCentralFoam moving mesh, 2 mm cells, ideal valve, the first part of the shot; 0-D carries the",
                "valve and the venturi. null = NOT RUN (the 0-D / measured value is used)."])
    if a.only in (None, "drag"):
        drag(a, y)
    if a.only in (None, "tube"):
        tube(a, y)
    y.write(a.out)


if __name__ == "__main__":
    main(sys.argv[1:])
