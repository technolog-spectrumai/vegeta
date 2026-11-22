"""A MuJoCo video of ASTOR-TWIN netting a quadcopter with its launcher (``designs/astor_netshot``).

    python labs/aero/sims/astor_netshot_video.py                    # seed 0 (a weaving quadcopter)
    python labs/aero/sims/astor_netshot_video.py --seed 2           # a straight flier
    python labs/aero/sims/astor_netshot_video.py --kind jink --seed 4

One real sortie of ``astor_netshot.netshot``: ASTOR-TWIN's flight model, the tracker, the approach along the
quadcopter's mean track, the launcher's pan-tilt mount, the shot. The net is NOT simulated cord by cord: it is the
launcher's test reduced to a point that flies and opens (see ``astor_netshot``); the picture shows it as a square with
its four corner weights, growing as it opens. A quadcopter taken falls with the net. The 25 m before the shot and
the 1.5 s after it are shown four times slower.

Needs OpenGL off-screen (MUJOCO_GL=egl, set here if unset) and ffmpeg. ``-j`` is accepted and not used.
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "designs"))
sys.path.append(str(HERE))

import numpy as np  # noqa: E402

from astor_talon_video import Film  # noqa: E402

NET_XML = ('<body name="net" mocap="true" pos="0 0 -80"><geom name="net_sheet" type="box" size="0.05 0.05 0.004" rgba="0.95 0.95 0.95 0.45" contype="0" conaffinity="0"/>'
           + "".join(f'<geom name="net_w{k}" type="sphere" size="0.03" pos="0 0 0" rgba="0.1 0.1 0.1 1" contype="0" conaffinity="0"/>' for k in range(4)) + "</body>")


def build_sim(folder):
    import mujoco
    import arigo1_mujoco as amj
    import astor_catch as AC
    import astor_twin as W
    import astor_v_flight as VF
    folder.mkdir(parents=True, exist_ok=True)
    stl = folder / "astor_twin_launcher.stl"
    if not stl.exists():
        W.AstorTwin().generate(**W.variant(wing_sweep_deg=12.0, tail_fan=1.0, prop_fold_deg=0.0, talon_deploy=-1.0, net_launchers=1, net_count=0)).export_stl(str(stl), tolerance=0.4)
    orig = amj.mjcf

    def mjcf(C, mesh=None, bandit=False):
        x = orig(C, mesh, True)
        x = x.replace('<geom type="box" size="0.18 0.18 0.05" rgba="0.8 0.15 0.1 1" contype="0" conaffinity="0"/>', AC.QUAD_GEOMS)
        return x.replace("</worldbody>", NET_XML + "</worldbody>")
    amj.mjcf = mjcf
    try:
        sim = VF.SimV("twin", mesh=str(stl), bandit=True)
    finally:
        amj.mjcf = orig
    m = sim.m
    g = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
    b = lambda n: m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)]
    sim.vis = {"sheet": g("net_sheet"), "w": [g(f"net_w{k}") for k in range(4)], "net": b("net"), "bandit": b("bandit")}
    return sim


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kind", choices=("straight", "weave", "jink"), default=None, help="the quadcopter (default: what the seed draws)")
    ap.add_argument("--size", type=int, nargs=2, default=(1280, 720))
    ap.add_argument("-j", "--cpu", type=int, default=1, help="not used (one sortie)")
    ap.add_argument("--out", type=Path, default=HERE / "output" / "astor_netshot_video")
    a = ap.parse_args(argv)
    import mujoco
    import yaml
    import astor_netshot as NS
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    sim = build_sim(a.out / "mesh")
    m, d, vis = sim.m, sim.d, sim.vis
    path = a.out / f"netshot_seed{a.seed}{'_' + a.kind if a.kind else ''}.mp4"
    film = Film(sim, path, tuple(a.size))
    st = {"t_shot": None, "last": None}

    def rec(sim_, bandit, net, t, netted):
        s = sim_.state()
        d.mocap_pos[vis["bandit"]] = bandit.p
        if net is not None:
            st["t_shot"] = st["t_shot"] if st["last"] is net else t
            st["last"] = net
            h = max(NS.half_span(float(np.linalg.norm(net.p - net.p0))), 0.04)
            d.mocap_pos[vis["net"]] = net.p
            q = np.zeros(4)
            mujoco.mju_quatZ2Vec(q, net.v / max(float(np.linalg.norm(net.v)), 1e-6))
            d.mocap_quat[vis["net"]] = q
        elif netted is not None:                                           # taken: the net bunched round the quadcopter
            h = 0.28
            d.mocap_pos[vis["net"]] = bandit.p
        else:
            h = 0.04
            d.mocap_pos[vis["net"]] = (0.0, 0.0, -80.0)
        m.geom_size[vis["sheet"]] = (h, h, 0.004)
        for k, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
            m.geom_pos[vis["w"][k]] = (sx * h, sy * h, 0.0)
        rng = math.dist(bandit.p, s["pos"])
        vh = s["v_world"][:2]
        slow = (netted is None and rng < 25.0) or (st["t_shot"] is not None and t - st["t_shot"] < 1.5)
        after = netted is not None
        film.shot(s["pos"], bandit.p if not after or t - netted < 2.5 else s["pos"], math.degrees(math.atan2(vh[1], vh[0])), 1 if slow else 4, rng < 60.0 or after,
                  wide=(4.0 if after else None))
    r = NS.netshot(seed=a.seed, kind=a.kind, record=rec, sim=sim)
    film.close()
    out = {"sortie": r, "video": str(path), "frames": film.frames, "seconds_of_video": round(film.frames / film.fps, 1), "wall_s": round(time.time() - t0)}
    (a.out / (path.stem + ".yaml")).write_text(yaml.safe_dump(__import__("json").loads(__import__("json").dumps(out, default=float)), sort_keys=False, width=140))
    print({k: out[k] for k in ("video", "frames", "seconds_of_video", "wall_s")}, r["outcome"], r["quadcopter"], r["shots"])
    if not r["win"]:
        print("this sortie did not net the quadcopter: try another --seed or --kind")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
