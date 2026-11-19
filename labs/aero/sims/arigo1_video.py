"""ARIGO-1 manoeuvre videos in MuJoCo — run by ``arigo1_video.sh`` (notebook 39). Needs an OpenGL context
(``MUJOCO_GL=egl`` on a GPU machine, ``osmesa`` in software, or a display) and ffmpeg.

Manoeuvres under the autopilot: break turn (80 deg bank, 4.5 g), loop (4 g), split-S, aileron roll, power-off stall
and recovery, chase of a zig-zagging bandit. Each is an mp4 with a chase camera, the CAD as the aircraft.

Writes ``labs/aero/sims/output/arigo1_video/<variant>_<propulsion>_<manoeuvre>.mp4`` and
``labs/aero/sims/output/arigo1_video.yaml`` (each manoeuvre's peak g, speeds, radius, altitude, stall) — PASTE THE
YAML BACK. ``--no-render`` flies them and writes only the YAML.
"""
import argparse
import json
import math
import os  # noqa: F401
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

if "--no-render" in sys.argv:                  # no OpenGL context needed (and none may exist)
    import os
    os.environ.pop("MUJOCO_GL", None)

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "labs" / "aero" / "designs"))
import arigo1_mujoco as amj  # noqa: E402

OUT = ROOT / "labs" / "aero" / "sims" / "output"


class Recorder:
    def __init__(self, sim, path, size=(1280, 720), fps=30):
        import mujoco
        self.mj, self.sim, self.fps = mujoco, sim, fps
        self.r = mujoco.Renderer(sim.m, size[1], size[0])
        self.cam = mujoco.MjvCamera()
        self.cam.distance, self.cam.elevation = 3.5, -12.0
        self.every = max(1, round(1 / (fps * amj.DT * 5)))
        self.k = 0
        self.ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}",
                                    "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(path)], stdin=subprocess.PIPE)
        self.psi = None

    def __call__(self, sim):
        self.k += 1
        if self.k % self.every:
            return
        s = sim.state()
        vh = s["v_world"][:2]
        psi = math.degrees(math.atan2(vh[1], vh[0])) if np.linalg.norm(vh) > 1 else (self.psi or 0.0)
        self.psi = psi if self.psi is None else self.psi + 0.15 * ((psi - self.psi + 180) % 360 - 180)   # a smoothed follow
        self.cam.lookat[:] = s["pos"]
        self.cam.azimuth = self.psi + 180.0 - 25.0
        self.r.update_scene(sim.d, camera=self.cam)
        self.ff.stdin.write(self.r.render().tobytes())

    def close(self):
        self.ff.stdin.close()
        self.ff.wait()


def eta(done, total, t0):
    """'k/N, elapsed, about X left' from the mean time per finished item."""
    import time as _t
    el = _t.time() - t0
    left = el / done * (total - done) if done else float("nan")
    f = lambda s: "?" if s != s else (f"{s / 3600:.1f} h" if s >= 3600 else f"{s / 60:.0f} min" if s >= 90 else f"{s:.0f} s")
    return f"[{done}/{total}, {f(el)} elapsed, ~{f(left)} left]"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="*", default=["hybrid"], choices=("goshawk", "zero", "hybrid"))
    ap.add_argument("--propulsion", nargs="*", default=["pusher"], choices=("pusher", "tractor", "twin"))
    ap.add_argument("--manoeuvres", nargs="*", default=list(amj.MANOEUVRES), choices=amj.MANOEUVRES)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--size", type=int, nargs=2, default=(1280, 720))
    a = ap.parse_args(argv)
    vdir = OUT / "arigo1_video"
    vdir.mkdir(parents=True, exist_ok=True)
    doc = {"aircraft": "ARIGO-1", "note": "aerodynamics from arigo1_model (lattice + hand estimates); autopilot gains by hand", "runs": {}}
    t0, n, total = time.time(), 0, len(a.variants) * len(a.propulsion) * len(a.manoeuvres)
    for v in a.variants:
        for p in a.propulsion:
            mesh = None if a.no_render else amj.export_mesh(OUT / "arigo1_video" / "mesh", v, p)
            for name in a.manoeuvres:
                sim = amj.Sim(v, p, mesh=mesh, bandit=(name == "chase"))
                key = f"{v}_{p}_{name}"
                rec = None if a.no_render else Recorder(sim, vdir / f"{key}.mp4", tuple(a.size))
                t1 = time.time()
                try:
                    _, summ = amj.manoeuvre(name, sim, rec)
                except Exception as exc:
                    summ = {"error": f"{type(exc).__name__}: {exc}"[:300]}
                if rec:
                    rec.close()
                    summ["video"] = str((vdir / f"{key}.mp4").relative_to(ROOT))
                doc["runs"][key] = summ
                n += 1
                print(f"  {eta(n, total, t0)} {key}: {time.time() - t1:.1f} s  {summ}")
                (OUT / "arigo1_video.yaml").write_text(yaml.safe_dump(json.loads(json.dumps(doc, default=float)), sort_keys=False, width=140))
    print(f"-> {OUT / 'arigo1_video.yaml'}  videos in {vdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
