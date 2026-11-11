"""The clips: which agents, which stretch of the simulation, the camera and the overlay; ``make(name, out_dir)``
writes ``<name>.mp4`` (1280x720, 30 fps, H.264) and a few PNG stills, and returns a summary (duration, the numbers
the run produced: touch-down speed, release speed, peak g, catch speed, depth, ...).

    python -m pinna_launch all --out DIR          # every clip
    python -m pinna_launch gannet_launch fulmar_net --out DIR --stills 3
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import video, world

FOOT = "Pinna drone carrier (Vegeta, labs/marine) - MuJoCo {mj} - {sea}"


def _lerp(a, b, s):
    s = max(0.0, min(1.0, s))
    return a + (b - a) * s


def _ease(t, t0, t1):
    if t <= t0:
        return 0.0
    if t >= t1:
        return 1.0
    s = (t - t0) / (t1 - t0)
    return s * s * (3 - 2 * s)


@dataclass
class Clip:
    name: str
    title: str
    t0: float
    t1: float
    agents: callable                # () -> [agents]
    camera: callable                # (sim, agents, t, smooth) -> (lookat, distance, azimuth, elevation)
    stills: tuple = ()              # times of the PNG stills
    summary: callable = None        # (sim, agents) -> dict
    tau: float = 0.35               # the camera's look-at lag (s)
    sea: callable = None            # () -> world.Sea (default: world.Sea())
    sim_kw: dict = None             # more for world.Sim (seabed, reels, track)
    hud: callable = None            # (sim, agents) -> lines (default: the agents' own)


# ------------------------------------------------------------------------------------------------ cameras
def _pad(sim, g):
    return sim.carrier_point(g.pad_local)[0]


def cam_gannet_launch(sim, ag, t, sm):
    """Over the deck for the lift-off and the climb, then ahead of the Gannet looking back at it and the carrier."""
    g = ag[0]
    pad, p = _pad(sim, g), g.focus(sim)
    s = _ease(t, 9.0, 13.0)
    deck = pad + 0.6 * (p - pad) + np.array([0, 0, 0.6])
    look = sm(t, deck + s * (p - deck))
    d = _lerp(7.5 + 0.75 * float(np.linalg.norm(p - pad)), 13.0, s)
    az = _lerp(_lerp(125.0, 100.0, t / 9.0), 160.0, s)
    return look, d, az, _lerp(_lerp(-16.0, -9.0, t / 9.0), -13.0, s)


def cam_gannet_landing(sim, ag, t, sm):
    g = ag[0]
    pad, p = _pad(sim, g), g.focus(sim)
    look = sm(t, pad + 0.5 * (p - pad) + np.array([0, 0, 0.6]))
    d = 8.0 + 0.55 * float(np.linalg.norm(p - pad))
    return look, d, _lerp(-165.0, -140.0, (t - 60.0) / 22.0), _lerp(-17.0, -21.0, (t - 60.0) / 22.0)


def cam_fulmar_launch(sim, ag, t, sm):
    """From the starboard side of the rail for the stroke, then behind the Fulmar on its climb-out."""
    f = ag[0]
    p = f.focus(sim)
    psi_r = world.CATAPULT_YAW_DEG
    v = sim.data.qvel[f.va:f.va + 3]
    heading = math.degrees(math.atan2(v[1], v[0])) if np.linalg.norm(v[:2]) > 3 else psi_r
    s = _ease(t, f.t_fire + 0.6, f.t_fire + 3.5)
    look = sm(t, p)
    return look, _lerp(10.0, 14.0, s), _lerp(psi_r + 100.0, heading + 30.0, s), _lerp(-22.0, -10.0, s)


def cam_fulmar_net(sim, ag, t, sm):
    f = ag[0]
    g = world.geometry()
    net = sim.carrier_point((g["net_x"], 0.0, f.z_net))[0]
    p = f.focus(sim)
    dist = float(np.linalg.norm(p - net))
    look = sm(t, net + 0.45 * (p - net))
    return look, 12.0 + 0.6 * dist, _lerp(-70.0, -85.0, t / 16.0), _lerp(-7.0, -12.0, t / 16.0)


def cam_walrus(sim, ag, t, sm):
    """From astern and above (the deck, the davit, the swing to port), down to the water with it, then under."""
    w = ag[0]
    p = w.focus(sim)
    look = sm(t, p)
    a = _ease(t, 3.0, 12.0)
    u = _ease(t, w.t_lower + 6.5, w.t_lower + 10.5)
    az = _lerp(_lerp(15.0, -40.0, a), -70.0, u)
    el = _lerp(_lerp(-30.0, -12.0, a), 3.0, u)
    return look, _lerp(_lerp(7.0, 5.0, _ease(t, 6.0, 13.0)), 4.5, u), az, el


def cam_overview(sim, ag, t, sm):
    look = sm(t, np.array([-4.0, 0.5, 6.0]) + np.array([0, 0, 6.0]) * _ease(t, 3.0, 11.0))
    return look, _lerp(27.0, 36.0, _ease(t, 2.0, 14.0)), _lerp(-50.0, -75.0, t / 20.0), _lerp(-20.0, -14.0, t / 20.0)


# ------------------------------------------------------------------------------------------------ the clips
def _gannet():
    from .gannet_sim import GannetAgent
    return [GannetAgent()]


def _fulmar_launch():
    from .fulmar_sim import FulmarLaunchAgent
    return [FulmarLaunchAgent(t_fire=3.0)]


def _fulmar_net():
    from .fulmar_sim import FulmarNetAgent
    return [FulmarNetAgent()]


def _walrus():
    from .walrus_sim import WalrusAgent
    return [WalrusAgent(compensated=True)]


def _walrus_off():
    from .walrus_sim import WalrusAgent
    return [WalrusAgent(compensated=False)]


def _all():
    from .fulmar_sim import FulmarLaunchAgent
    from .gannet_sim import GannetAgent
    from .walrus_sim import WalrusAgent
    return [GannetAgent(), FulmarLaunchAgent(t_fire=7.0), WalrusAgent()]


def _sum_gannet(sim, ag):
    g = ag[0]
    L = np.array(g.log)
    return {"max_alt_m": float(L[:, 3].max()), "max_speed_m_s": float(np.linalg.norm(L[:, 4:6], axis=1).max()),
            "touchdown": g.touch, "landed": g.landed}


def _sum_fulmar(sim, ag):
    f = ag[0]
    L = np.array(f.log)
    return {"release_m_s": getattr(f, "v_release", None), "peak_g": f.peak_g, "stroke_m": f.c["stroke"], "accel_g": f.a_cat / world.G,
            "alt_end_m": float(L[-1, 3]), "min_air_m_s_after_release": float(L[L[:, 0] > (f.t_release or 1e9), 7].min()) if f.t_release else None}


def _sum_net(sim, ag):
    f = ag[0]
    nx = sim.model.jnt_qposadr[sim.joint("net_x")]
    return {"caught": f.caught, "net_travel_max_m": f.net_max, "peak_g_in_net": f.peak_g_catch, "net_travel_now_m": float(sim.data.qpos[nx])}


def _sum_walrus(sim, ag):
    w = ag[0]
    L = np.array(w.log)
    return {"compensated": w.compensated, "residual": w.residual(), "released_s": w.released, "max_cable_pull_N": float(L[:, 7].max()), "max_cable_pull_unfiltered_N": float(L[:, 9].max()), "depth_end_m": float(w.depth()),
            "standoff_y_m": float(L[-1, 2])}


def _oil():
    from .oil_platform_sim import JacketScenery, SquillaAgent, inspection_gannet
    return [JacketScenery(), inspection_gannet(), SquillaAgent()]


def _oil_kw():
    from .oil_platform_sim import track
    return {"seabed": -110.0, "reels": False, "track": track}


def _oil_hud(sim, ag):
    _, g, sq = ag
    t = sim.t
    from .oil_platform_sim import ARRIVE
    pin = "arriving from the west at 4 m/s" if t < ARRIVE[2] else "on DP station off the jacket's south-west corner"
    p, v, R, w = g.state()
    return [f"Pinna: {pin}   carrying one Squilla AUV and one Gannet quad   {sim.sea.describe()}",
            f"Gannet {g.c['mass']:.1f} kg (empty, inspection)   alt {p[2]:5.1f} m   speed {np.linalg.norm(v):4.1f} m/s   phase: {g.mission.phase(t, g.landed)}",
            sq.hud(sim)[0] + ("   davit: MRU + AHC (uprated 3 kN winch)" if sq.released is None else "")]


def cam_oil(sim, ag, t, sm):
    """Cuts: the arrival (wide), Gannet's lift-off (deck), Gannet at leg A1, the davit lowering Squilla through the
    splash zone, Gannet up the topsides, Squilla diving to the leg (under water)."""
    from .oil_platform_sim import JACKET_AT, leg_point
    _, g, sq = ag
    cuts = (8.5, 13.0, 17.5, 25.0, 31.0)
    k = sum(t >= c for c in cuts)
    if getattr(sm, "cut", None) != k:
        sm.cut, sm.x = k, None
    pg, ps = g.focus(sim), sq.focus(sim)
    if k == 0:
        look = sm(t, np.array([JACKET_AT[0] - 6.0 - 0.6 * (30.0 - sim.track(t)[0]), 18.0, 10.0]))
        return look, 78.0, _lerp(62.0, 70.0, t / 8.5), -14.0
    if k == 1:
        pad = sim.carrier_point(g.pad_local)[0]
        look = sm(t, pad + 0.6 * (pg - pad) + np.array([0, 0, 0.8]))
        return look, 11.0 + 0.7 * float(np.linalg.norm(pg - pad)), _lerp(140.0, 120.0, (t - 8.5) / 4.5), -14.0
    if k in (2, 4):                                  # from outside the jacket, south-west of the quad, looking at it and the leg
        leg = leg_point("A1", float(np.clip(pg[2], 0, 30)))
        look = sm(t, 0.7 * pg + 0.3 * leg)
        if k == 2:
            return look, 18.0, _lerp(30.0, 45.0, (t - 13.0) / 4.5), -8.0
        return look, _lerp(16.0, 24.0, (t - 25.0) / 6.0), _lerp(40.0, 20.0, (t - 25.0) / 6.0), _lerp(-4.0, -10.0, (t - 25.0) / 6.0)
    if k == 3:                                       # astern and outboard of the davit, down with Squilla
        look = sm(t, ps + np.array([0, 0, 0.3]))
        u = _ease(t, 21.5, 24.0)
        return look, _lerp(9.0, 7.0, u), _lerp(-25.0, -40.0, (t - 17.5) / 7.5), _lerp(-12.0, 2.0, u)
    leg = leg_point("A1", -12.0)
    look = sm(t, 0.65 * ps + 0.35 * leg)
    return look, 13.0, _lerp(-170.0, -150.0, (t - 31.0) / 9.0), 6.0


def _sum_oil(sim, ag):
    _, g, sq = ag
    com = sq.state()[0]
    return {"gannet_max_alt_m": float(np.array(g.log)[:, 3].max()), "squilla_released_s": sq.released,
            "squilla_to_goal_m": float(np.linalg.norm(sq.goal - com)), "squilla_depth_m": float(sq.depth()),
            "davit_residual": sq.residual()}


def _sum_all(sim, ag):
    return {"gannet_alt_m": float(ag[0].focus(sim)[2]), "fulmar_released": ag[1].released, "walrus_released_s": ag[2].released}


CLIPS = {
    "gannet_launch": Clip("gannet_launch", "Gannet: vertical take-off from Pinna's pad with a 5 kg crate", 0.0, 18.0, _gannet, cam_gannet_launch,
                          stills=(1.0, 6.0, 15.0), summary=_sum_gannet),
    "gannet_landing": Clip("gannet_landing", "Gannet: back from the circuit, landing on the moving deck", 60.0, 82.0, _gannet, cam_gannet_landing,
                           stills=(66.0, 76.0, 80.5), summary=_sum_gannet),
    "fulmar_launch": Clip("fulmar_launch", "Fulmar: catapult launch (2.8 m stroke) and climb-out", 0.0, 15.0, _fulmar_launch, cam_fulmar_launch,
                          stills=(2.5, 3.2, 6.0), summary=_sum_fulmar, tau=0.12),
    "fulmar_net": Clip("fulmar_net", "Fulmar: net recovery over the stern, into the wind", 0.0, 16.5, _fulmar_net, cam_fulmar_net,
                       stills=(8.0, 12.6, 13.6), summary=_sum_net),
    "walrus_launch": Clip("walrus_launch", "Walrus over the side: motion-compensated davit ON (heavy sea)", 0.0, 25.0, _walrus, cam_walrus,
                          stills=(4.5, 11.0, 13.5, 21.0), summary=_sum_walrus, tau=1.0, sea=world.Sea.heavy),
    "walrus_launch_uncompensated": Clip("walrus_launch_uncompensated", "Walrus over the side: compensation OFF (heavy sea)", 0.0, 25.0, _walrus_off,
                                        cam_walrus, stills=(11.0, 13.5), summary=_sum_walrus, tau=1.0, sea=world.Sea.heavy),
    "oil_platform_deploy": Clip("oil_platform_deploy", "Pinna at an oil platform: a Gannet inspection pass and a Squilla AUV over the compensated davit",
                                0.0, 40.0, _oil, cam_oil, stills=(5.0, 11.0, 16.0, 21.0, 24.0, 28.0, 36.0), summary=_sum_oil, tau=0.5,
                                hud=_oil_hud),
    "overview": Clip("overview", "Pinna: Gannet, Fulmar and Walrus launched together", 0.0, 20.0, _all, cam_overview,
                     stills=(9.0, 12.0), summary=_sum_all),
}


def build(name: str, use_meshes: bool = True):
    c = CLIPS[name]
    ag = c.agents()
    kw = _oil_kw() if name == "oil_platform_deploy" else (c.sim_kw or {})
    return world.Sim(ag, sea=c.sea() if c.sea else None, use_meshes=use_meshes, **kw), ag


def make(name: str, out_dir: Path, *, use_meshes: bool = True, stills: bool = True, size=(video.W, video.H)) -> dict:
    """Simulate and render one clip; returns its summary (also written as ``<name>.json``)."""
    video.gl_backend()
    import mujoco
    c = CLIPS[name]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wall = time.time()
    sim, ag = build(name, use_meshes)
    sim.run(c.t0)                                             # what comes before the clip, not drawn
    fr = video.Frames(sim, size)
    enc = video.Encoder(out_dir / f"{name}.mp4", size=size)
    sm = video.Smooth(c.tau)
    foot = FOOT.format(mj=mujoco.__version__, sea=sim.describe())
    todo = sorted(c.stills) if stills else []
    saved = []

    def frame(s):
        t = s.t
        look, dist, az, el = c.camera(s, ag, t, sm)
        if c.hud:
            lines = c.hud(s, ag)
        elif len(ag) > 1:                                     # the overview: one line each
            lines = [a.hud(s)[0] for a in ag]
        else:
            lines = ag[0].hud(s)
        chart = ag[0].trace(s) if len(ag) == 1 and hasattr(ag[0], "trace") else None
        img = fr.render(look, dist, az, el, title=c.title, lines=lines, footer=foot, clock=f"t = {t:5.1f} s", chart=chart)
        enc.write(img)
        while todo and t >= todo[0] - 1e-6:
            from PIL import Image
            p = out_dir / f"{name}_{todo.pop(0):05.1f}s.png"
            Image.fromarray(img).save(p)
            saved.append(str(p))

    sim.run(c.t1 - 1e-6, every=1.0 / video.FPS, callback=frame, t_first=c.t0)
    mp4 = enc.close()
    fr.r.close()
    summary = {"clip": name, "title": c.title, "mp4": str(mp4), "frames": enc.n, "duration_s": enc.n / video.FPS,
               "sim_window_s": [c.t0, c.t1], "wall_s": round(time.time() - wall, 1), "stills": saved,
               "sea": sim.sea.describe(), "wind": sim.wind.describe(), "catapult_yaw_deg": world.CATAPULT_YAW_DEG}
    if c.summary:
        summary["result"] = c.summary(sim, ag)
    (out_dir / f"{name}.json").write_text(json.dumps(summary, indent=1, default=float))
    return summary


def compare(out_dir: Path, use_meshes: bool = True) -> dict:
    """``walrus_compare.mp4``: the compensated and the plain davit side by side (each 640x720, the same sea, the same
    camera), made from two half-width runs in their own processes and stacked by ffmpeg."""
    import shutil
    import subprocess
    import sys
    out_dir = Path(out_dir)
    tmp = out_dir / "_compare"
    tmp.mkdir(parents=True, exist_ok=True)
    for n in ("walrus_launch", "walrus_launch_uncompensated"):
        subprocess.run([sys.executable, "-m", "pinna_launch", n, "--out", str(tmp), "--size", "640x720", "--no-stills"]
                       + ([] if use_meshes else ["--no-meshes"]), check=True, stdout=subprocess.DEVNULL)
    out = out_dir / "walrus_compare.mp4"
    subprocess.run([shutil.which("ffmpeg"), "-y", "-loglevel", "error", "-i", str(tmp / "walrus_launch.mp4"), "-i", str(tmp / "walrus_launch_uncompensated.mp4"),
                    "-filter_complex", "[0:v][1:v]hstack=inputs=2,format=yuv420p", "-c:v", "libx264", "-crf", "21", "-r", str(video.FPS),
                    "-movflags", "+faststart", str(out)], check=True)
    res = {n: json.loads((tmp / f"{n}.json").read_text())["result"]["residual"] for n in ("walrus_launch", "walrus_launch_uncompensated")}
    shutil.rmtree(tmp, ignore_errors=True)
    s = {"clip": "walrus_compare", "mp4": str(out), "duration_s": 25.0, "residual": res}
    (out_dir / "walrus_compare.json").write_text(json.dumps(s, indent=1, default=float))
    return s


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m pinna_launch", description=__doc__.split("\n")[0])
    ap.add_argument("clips", nargs="*", default=["all"], help=f"clip names ({', '.join(CLIPS)}) or 'all'")
    ap.add_argument("--out", default=str(world.meshes.ROOT / "labs" / "marine" / "sims" / "output" / "pinna_launch" / "videos"))
    ap.add_argument("--no-meshes", action="store_true", help="boxes instead of the CAD meshes")
    ap.add_argument("--no-stills", action="store_true")
    ap.add_argument("--size", default=f"{video.W}x{video.H}", help="WxH of the frames")
    a = ap.parse_args(argv)
    size = tuple(int(v) for v in a.size.split("x"))
    names = list(CLIPS) + ["walrus_compare"] if a.clips in (["all"], []) else a.clips
    if names == ["walrus_compare"]:
        print(json.dumps(compare(Path(a.out), use_meshes=not a.no_meshes), default=float))
        return
    if len(names) > 1:                  # one process per clip: a second offscreen GL context in one process is not reliable
        import subprocess
        import sys
        for n in names:
            subprocess.run([sys.executable, "-m", "pinna_launch", n, "--out", a.out] + (["--no-meshes"] if a.no_meshes else [])
                           + (["--no-stills"] if a.no_stills else []), check=True)
        return
    s = make(names[0], Path(a.out), use_meshes=not a.no_meshes, stills=not a.no_stills, size=size)
    print(json.dumps(s, default=float))


if __name__ == "__main__":
    main()
