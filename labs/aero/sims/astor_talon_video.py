"""A MuJoCo video of ASTOR-T taking a quadcopter with its talons (``designs/astor_talon``).

    python labs/aero/sims/astor_talon_video.py                 # seed 0 -> sims/output/astor_talon_video/quad_catch_seed0.mp4
    python labs/aero/sims/astor_talon_video.py --seed 3 --size 1920 1080

One real sortie of ``astor_talon.strike`` against a quadcopter cruising straight at 9 m/s, NOT evading (against
the jinking Armageddon bandit the talons have not made a catch). What is flown: ASTOR-T's 6-DOF flight model, the
tracker with its latency and noise, the guidance; a catch is the feet passing within the legs' reach. What is drawn
only: the legs swinging down (two CAD meshes are swapped 70 m out). After the catch the video goes on with a SIMPLE
descent, not the flight model: the two masses joined (momentum kept, the pitch rate of ``strike_loads``), the motor
cut, the parachute's drag at the hatch and a rotational damping (HAND). The last 15 m before the catch are shown
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

QUAD = ('<geom type="box" size="0.07 0.045 0.03" rgba="0.15 0.15 0.17 1" contype="0" conaffinity="0"/>'
        + "".join(f'<geom type="capsule" fromto="0 0 0 {0.16 * cx:.3f} {0.16 * cy:.3f} 0.01" size="0.008" rgba="0.8 0.15 0.1 1" contype="0" conaffinity="0"/>'
                  f'<geom type="cylinder" pos="{0.16 * cx:.3f} {0.16 * cy:.3f} 0.035" size="0.09 0.003" rgba="0.6 0.6 0.65 0.55" contype="0" conaffinity="0"/>'
                  for cx, cy in ((1, 1), (1, -1), (-1, 1), (-1, -1))))
CHUTE = ('<body name="chute" mocap="true" pos="0 0 -60"><geom type="ellipsoid" size="0.5 0.5 0.28" rgba="0.92 0.35 0.2 0.92" contype="0" conaffinity="0"/></body>')


def build_sim(folder):
    """ASTOR-T with two CAD meshes (legs folded / thrown), a quadcopter for the bandit, a canopy kept out of sight."""
    import arigo1_mujoco as amj
    import astor_t as T
    import astor_v_flight as VF
    folder.mkdir(parents=True, exist_ok=True)
    stl = {}
    for key, dep in (("folded", 0.0), ("thrown", 1.0)):
        stl[key] = folder / f"astor_t_legs_{key}.stl"
        if not stl[key].exists():
            T.AstorT().generate(**T.variant(wing_sweep_deg=12.0, tail_fan=1.0, prop_fold_deg=0.0, talon_deploy=dep)).export_stl(str(stl[key]), tolerance=0.4)
    orig = amj.mjcf

    def mjcf(C, mesh=None, bandit=False):
        x = orig(C, mesh, bandit)
        i = x.index('<geom type="mesh" mesh="arigo"')
        j = x.index("/>", i) + 2
        g = x[i:j]
        x = x[:i] + g.replace('<geom type="mesh"', '<geom name="vis_a" type="mesh"') + g.replace('mesh="arigo"', 'mesh="arigo2"').replace(
            '<geom type="mesh"', '<geom name="vis_b" type="mesh"').replace('rgba="0.75 0.72 0.62 1"', 'rgba="0.75 0.72 0.62 0"') + x[j:]
        x = x.replace('<mesh name="arigo"', f'<mesh name="arigo2" file="{stl["thrown"]}" scale="0.001 0.001 0.001"/><mesh name="arigo"', 1)
        x = x.replace('<geom type="box" size="0.18 0.18 0.05" rgba="0.8 0.15 0.1 1" contype="0" conaffinity="0"/>', QUAD)
        return x.replace("</worldbody>", CHUTE + "</worldbody>")
    amj.mjcf = mjcf
    try:
        sim = VF.SimV("t", mesh=str(stl["folded"]), bandit=True)
    finally:
        amj.mjcf = orig
    return sim


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
        g = lambda n: mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_GEOM, n)
        self.ga, self.gb = g("vis_a"), g("vis_b")

    def legs(self, thrown):
        self.sim.m.geom_rgba[self.ga, 3], self.sim.m.geom_rgba[self.gb, 3] = (0.0, 1.0) if thrown else (1.0, 0.0)

    def shot(self, pos, other, heading_deg, every, near, wide=None):
        """One call per 0.01 s of flight; a frame every ``every`` calls. The camera eases between a chase view and a side view of the pair."""
        self.k += 1
        if self.k % every:
            return
        rng = float(np.linalg.norm(np.asarray(other) - np.asarray(pos)))
        look = np.asarray(pos) if not near else 0.5 * (np.asarray(pos) + np.asarray(other))
        dist = 4.5 if not near else max(2.6, 0.85 * rng)
        if wide is not None:                                              # the descent: the pair and its canopy in the picture
            dist, look = wide, np.asarray(pos) + np.array([0.0, 0.0, 0.5])
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


def descent(sim, bandit, film, seconds=6.0):
    """After the catch (SIMPLE, see the module's notes): the pair as one body, motor cut, the canopy's drag, damping."""
    import mujoco
    import arigo1_model as am
    import astor_t as T
    import astor_talon as TA
    m, d, bid = sim.m, sim.d, sim.bid
    mid = m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "bandit")]
    cid = m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "chute")]
    p = T.resolve()
    mb = TA.BANDITS_KG["quad"]
    ma = float(m.body_mass[bid])
    vb = np.asarray(bandit.v, float)
    v_close = float(np.linalg.norm(d.qvel[:3] - vb))
    loads = TA.strike_loads(v_close, mb, p, ma)
    d.qvel[:3] = (ma * d.qvel[:3] + mb * vb) / (ma + mb)                  # the two stick: momentum kept
    d.qvel[4] += math.radians(loads["pitch_rate_after_deg_s"])           # nose down (+ about the body's y, to the left)
    m.body_mass[bid] = ma + mb
    feet = np.array([sim.C["x_cg_m"] - 0.178, 0.0, -(p["fuselage_diameter"] / 2000.0 + 0.008 + TA.leg_drop(p)) - 0.03])
    hatch = np.array([-0.12, 0.0, 0.04])
    t, k = 0.0, 0
    while t < seconds:
        R = d.xmat[bid].reshape(3, 3)
        pos, v = d.xpos[bid].copy(), d.qvel[:3].copy()
        d.qfrc_applied[:] = 0.0
        open_ = min(1.0, max(0.0, (t - 0.35) / 0.6))                      # the canopy fills between 0.35 and 0.95 s
        F = -0.5 * am.RHO * TA.CHUTE_CDA * open_ * float(np.linalg.norm(v)) * v
        tq = -0.30 * (R @ d.qvel[3:6])                                    # HAND: the tumbling damped (the canopy's lines, the air)
        mujoco.mj_applyFT(m, d, F, tq, pos + R @ hatch, bid, d.qfrc_applied)
        mujoco.mj_step(m, d)
        t += m.opt.timestep
        k += 1
        R, pos = d.xmat[bid].reshape(3, 3), d.xpos[bid]
        bandit.p = list(pos + R @ feet)
        d.mocap_pos[mid], d.mocap_quat[mid] = bandit.p, d.xquat[bid]
        if open_ > 0.0:
            u = -v / max(float(np.linalg.norm(v)), 1e-6)
            d.mocap_pos[cid] = pos + R @ hatch + u * (0.3 + 1.0 * open_)
        if k % 5 == 0:
            film.shot(pos, bandit.p, math.degrees(math.atan2(v[1], v[0])) if np.hypot(v[0], v[1]) > 1.0 else 0.0, 2 if t < 1.2 else 4, True, wide=2.8 + 3.4 * open_)
        if pos[2] < 0.6:
            break
    v = d.qvel[:3]
    return {"closing_m_s": round(v_close, 1), "loads": loads, "sink_at_the_end_m_s": round(float(-v[2]), 1), "height_at_the_end_m": round(float(d.xpos[bid][2]), 1),
            "seconds": round(t, 1)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0, help="the sortie (the quadcopter's offset and heading, the camera's noise)")
    ap.add_argument("--size", type=int, nargs=2, default=(1280, 720))
    ap.add_argument("-j", "--cpu", type=int, default=1, help="not used (one sortie)")
    ap.add_argument("--out", type=Path, default=HERE / "output" / "astor_talon_video")
    a = ap.parse_args(argv)
    import yaml
    import astor_talon as TA
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    sim = build_sim(a.out / "mesh")
    path = a.out / f"quad_catch_seed{a.seed}.mp4"
    film = Film(sim, path, tuple(a.size))
    st = {"bandit": None}

    def rec(sim_, bandit, t):
        st["bandit"] = bandit
        s = sim_.state()
        rng = math.dist(bandit.p, s["pos"])
        film.legs(rng < 70.0)
        vh = s["v_world"][:2]
        film.shot(s["pos"], bandit.p, math.degrees(math.atan2(vh[1], vh[0])), 1 if rng < 15.0 else 4, rng < 60.0)
    r = TA.strike("t", "quad", seed=a.seed, duration_s=40.0, record=rec, sim=sim)
    out = {"sortie": {k: v for k, v in r.items() if k != "pass_list"}, "passes": r["pass_list"]}
    if r["caught"]:
        out["descent_simple_model"] = descent(sim, st["bandit"], film)
    film.close()
    out.update({"video": str(path), "frames": film.frames, "seconds_of_video": round(film.frames / film.fps, 1), "wall_s": round(time.time() - t0)})
    (a.out / f"quad_catch_seed{a.seed}.yaml").write_text(yaml.safe_dump(__import__("json").loads(__import__("json").dumps(out, default=float)), sort_keys=False, width=140))
    print({k: out[k] for k in ("video", "frames", "seconds_of_video", "wall_s")}, out["sortie"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
