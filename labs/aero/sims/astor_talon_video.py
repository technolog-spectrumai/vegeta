"""A MuJoCo video of ASTOR-T taking a quadcopter with its talons and flying on with it (``designs/astor_catch``).

    python labs/aero/sims/astor_talon_video.py                 # seed 3 -> sims/output/astor_talon_video/quad_catch_seed3.mp4
    python labs/aero/sims/astor_talon_video.py --seed 5 --size 1920 1080

One real sortie of ``astor_catch.catch`` against a quadcopter cruising straight at 9 m/s, NOT evading (against the
jinking Armageddon bandit the talons have not made a catch). Everything shown is the simulation: ASTOR-T's 6-DOF
flight model, the tracker with its latency and noise, the guidance, the legs as a hinged, sprung body, the
quadcopter as a body that locks to the foot at the catch, and the recovery. Not every seed makes the catch: the
script says so and still writes the video. The last 15 m before the catch and the first 1.5 s after it are shown
four times slower.

Needs OpenGL off-screen (MUJOCO_GL=egl, set here if unset) and ffmpeg. One sortie: a few seconds of flight, about
a minute of rendering; ``-j`` is accepted for uniformity with the other scripts and not used.
"""
import argparse
import math
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "designs"))
sys.path.append(str(HERE))

import numpy as np  # noqa: E402

def build_sim(folder):
    """``astor_catch``'s model with ASTOR-T's CAD (without legs: they are bodies of the model) as the picture."""
    import astor_catch as AC
    import astor_t as T
    folder.mkdir(parents=True, exist_ok=True)
    stl = folder / "astor_t_no_legs.stl"
    if not stl.exists():
        T.AstorT().generate(**T.variant(wing_sweep_deg=12.0, tail_fan=1.0, prop_fold_deg=0.0, talon_deploy=-1.0)).export_stl(str(stl), tolerance=0.4)
    return AC.build(str(stl))


class Film:
    def __init__(self, sim, path, size, fps=25):
        import mujoco
        self.mj, self.sim, self.fps = mujoco, sim, fps
        self.r = mujoco.Renderer(sim.m, size[1], size[0])
        self.cam = mujoco.MjvCamera()
        self.look, self.dist, self.azi, self.elev = None, 4.0, None, -10.0
        self.k, self.frames = 0, 0
        self.ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}", "-r", str(fps),
                                    "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "21", str(path)], stdin=subprocess.PIPE)

    def shot(self, pos, other, heading_deg, every, near, wide=None):
        """One call per 0.01 s of flight; a frame every ``every`` calls. The camera eases between a chase view and a side view of the pair."""
        self.k += 1
        if self.k % every:
            return
        rng = float(np.linalg.norm(np.asarray(other) - np.asarray(pos)))
        look = np.asarray(pos) if not near else 0.5 * (np.asarray(pos) + np.asarray(other))
        dist = 4.5 if not near else max(2.6, 0.85 * rng)
        if wide is not None:                                              # the descent: the pair and its canopy in the picture
            dist, look = max(wide, 1.9 * rng), 0.5 * (np.asarray(pos) + np.asarray(other))     # the pair on its line in the picture
        azi = heading_deg + (155.0 if not near else 105.0)
        elev = -12.0 if not near else -6.0
        e = 0.12
        self.look = look if self.look is None else self.look + 0.5 * (look - self.look)
        self.dist += e * (dist - self.dist)
        self.azi = azi if self.azi is None else self.azi + e * ((azi - self.azi + 180) % 360 - 180)
        self.elev += e * (elev - self.elev)
        self.cam.lookat[:] = self.look
        self.cam.distance, self.cam.azimuth, self.cam.elevation = self.dist, self.azi, self.elev
        self.r.update_scene(self.sim.d, camera=self.cam)
        self.ff.stdin.write(self.r.render().tobytes())
        self.frames += 1

    def close(self):
        self.ff.stdin.close()
        self.ff.wait()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=3, help="the sortie (the quadcopter's offset and heading, the camera's noise)")
    ap.add_argument("--size", type=int, nargs=2, default=(1280, 720))
    ap.add_argument("-j", "--cpu", type=int, default=1, help="not used (one sortie)")
    ap.add_argument("--out", type=Path, default=HERE / "output" / "astor_talon_video")
    a = ap.parse_args(argv)
    import yaml
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    sim = build_sim(a.out / "mesh")
    path = a.out / f"quad_catch_seed{a.seed}.mp4"
    film = Film(sim, path, tuple(a.size))
    import astor_catch as AC

    def rec(sim_, bandit, t, caught):
        s = sim_.state()
        q = sim_.d.xpos[sim_.ids["quad"]]
        rng = math.dist(q, s["pos"])
        vh = s["v_world"][:2]
        st["t_c"] = t if (caught and st.get("t_c") is None) else st.get("t_c")
        slow = (not caught and rng < 15.0) or (caught and t - st["t_c"] < 1.5)
        film.shot(s["pos"], q, math.degrees(math.atan2(vh[1], vh[0])), 1 if slow else 4, rng < 60.0, wide=(3.6 if caught else None))
    st = {}
    r = AC.catch(seed=a.seed, record=rec, sim=sim)
    out = {"sortie": {k: v for k, v in r.items() if not k.startswith("history")}, "history_t_dz_V_pitch_q_bank_hip_stretch_vz": r.get("history_t_dz_V_pitch_q_bank_hip_stretch_vz")}
    film.close()
    out.update({"video": str(path), "frames": film.frames, "seconds_of_video": round(film.frames / film.fps, 1), "wall_s": round(time.time() - t0)})
    (a.out / f"quad_catch_seed{a.seed}.yaml").write_text(yaml.safe_dump(__import__("json").loads(__import__("json").dumps(out, default=float)), sort_keys=False, width=140))
    print({k: out[k] for k in ("video", "frames", "seconds_of_video", "wall_s")}, out["sortie"])
    if not r["caught"]:
        print("this seed's pass did not take the quadcopter (see 'pass'): try another --seed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
