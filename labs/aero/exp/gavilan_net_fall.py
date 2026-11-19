"""What a net does to a bandit after the catch: does it come down, how fast, and what does a line to the hunter take?

Built on the net launcher's MuJoCo test (``net_drone.py`` on dev_gavilan_net @ d9f6cb8: knots as free bodies, rope
tendons, cord drag per link, a 1.2 kg quadcopter whose rotors stop 0.05 s after a knot touches them). Changes here:

- **the hunter fires level, from the bandit's height** (``HEIGHT`` above the ground, ``RANGE`` away), as GAVILAN would
  from a hover beside it;
- **the bandit keeps falling after it is fouled**: its drag is quadratic with a tumbling drag area ``BANDIT_CDA``
  (hand estimate; ``net_drone`` gives it a weak linear drag, enough for a 4 m test but not for a 20 m fall);
- **the net's density**: the light interceptor's 1.5 m net at its 0.15 m mesh, and denser meshes of the same cord
  (more knots, more cord, the cord's mass in proportion); the cartridge's speed is kept at 45 m/s, so a heavier net
  needs a larger charge (not sized here);
- **a line to the hunter** (``--tether``): a tendon from a point that holds still (the hunter hovering) to the net's
  centre knot, ``LINE`` long, with an elastic section (``LINE_STRETCH`` of give at ``LINE_PEAK_N``): the peak force
  in the line against what the hunter can spare.

    python gavilan_net_fall.py --net-py PATH/TO/net_drone.py [--meshes 0.15,0.075,0.05] [--shots 3] [--tether]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HEIGHT = 20.0            # m above the ground: the bandit's height and the hunter's
RANGE = 5.0              # m: the shot, inside the light net's tested 3-8 m
BANDIT_CDA = 0.04        # m2: a tumbling 0.35 m quadcopter with its rotors stopped (hand estimate)
LINE = 8.0               # m: the line from the hunter to the net's centre
LINE_STRETCH = 1.0       # m of give in the elastic section
LINE_PEAK_N = 60.0       # N at full stretch: the spring the elastic section is chosen to be
CORNER_G = 8.0           # each corner weight, as the light interceptor's
CORD_G_PER_M = 30.0 / (2 * 11 * 1.5)    # the light net's cord: 30 g in 2 x 11 lines of 1.5 m


def load(path):
    spec = importlib.util.spec_from_file_location("net_drone", path)
    nd = importlib.util.module_from_spec(spec)
    sys.modules["net_drone"] = nd
    spec.loader.exec_module(nd)
    return nd


def setup(nd, mesh, tether):
    nd.configure(light=True, drone="quad")
    n = int(round(1.5 / mesh)) + 1
    nd.N, nd.NET = n, 1.5
    nd.MESH = nd.NET / (n - 1)
    nd.NET_MASS = CORD_G_PER_M * 2 * n * 1.5 / 1e3
    nd.CORNER = CORNER_G / 1e3
    nd.MUZZLE = np.array([0.0, 0.0, HEIGHT])
    xml0 = nd.xml

    def xml():
        s = xml0()
        if not tether:
            return s
        c = (n - 1) // 2
        s = s.replace("<worldbody>", f'<worldbody><site name="hunter" pos="0 0 {HEIGHT}"/>', 1)
        line = (f'<spatial name="line" limited="true" range="0 {LINE}" solreflimit="{0.08 * LINE_STRETCH:.4f} 0.7" damping="0.5">'
                f'<site site="hunter"/><site site="s{c}_{c}"/></spatial>')
        return s.replace("</tendon>", line + "</tendon>")

    nd.xml = xml
    return n


class Fall:
    """``net_drone.Sim`` with the bandit's quadratic drag once it is fouled, and the line's force read out."""

    def __init__(self, nd, tether):
        self.nd, self.sim, self.tether = nd, nd.Sim(), tether
        self.line = self.sim.m.tendon("line").id if tether else None
        step0 = self.sim.step
        sim = self.sim

        def step():
            step0()
            if sim.fouled_at is not None and sim.d.time - sim.fouled_at >= 0.05:
                v = sim.d.qvel[sim.ddof:sim.ddof + 3]
                sim.d.xfrc_applied[sim.drone, :3] = -0.5 * nd.RHO * BANDIT_CDA * np.linalg.norm(v) * v
        sim.step = step

    def tendon_force(self):
        import mujoco
        d, m = self.sim.d, self.sim.m
        f = 0.0
        for i in range(d.nefc):
            if d.efc_type[i] == mujoco.mjtConstraint.mjCNSTR_LIMIT_TENDON and d.efc_id[i] == self.line:
                f += abs(d.efc_force[i])
        return f


def one(nd, cal, rng, tether, t_max=8.0):
    target = np.array([RANGE, 0.0, HEIGHT])
    elev, az, t_fly = nd.aim(cal, target, np.zeros(3))
    elev += math.radians(0.5) * rng.standard_normal()
    az += math.radians(0.5) * rng.standard_normal()
    speed = nd.V0 * (1 + 0.03 * rng.standard_normal())
    f = Fall(nd, tether)
    sim = f.sim
    sim.reset(elev, az, speed, target, np.zeros(3), origin=nd.MUZZLE)
    peak_line, z_min, v_down_max, out = 0.0, HEIGHT, 0.0, {}
    while sim.d.time < t_max:
        for _ in range(20):
            sim.step()
            if tether:
                peak_line = max(peak_line, f.tendon_force())
        z = float(sim.d.xpos[sim.drone][2])
        v = sim.d.qvel[sim.ddof:sim.ddof + 3]
        v_down_max = max(v_down_max, -float(v[2]))
        z_min = min(z_min, z)
        if sim.fouled_at is None and sim.d.time > t_fly + 0.8:
            out = {"result": "miss"}
            break
        if z < 0.15:                                                   # on the ground (the hull's half height and the arms)
            out = {"result": "down", "impact_m_s": float(np.linalg.norm(v)), "fall_s": float(sim.d.time - (sim.fouled_at or 0.0))}
            break
    else:
        out = {"result": "held" if tether and sim.fouled_at is not None else ("fouled, still falling" if sim.fouled_at is not None else "miss"),
               "hang_height_m": z_min}
    out.update({"fouled": sim.fouled_at is not None, "t_foul": sim.fouled_at, "max_descent_m_s": v_down_max, "peak_line_N": peak_line,
                "lowest_m": z_min})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net-py", required=True, type=Path)
    ap.add_argument("--meshes", default="0.15,0.075,0.05")
    ap.add_argument("--shots", type=int, default=3)
    ap.add_argument("--tether", action="store_true")
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("gavilan_net_fall_results.json"))
    a = ap.parse_args(argv)
    nd = load(a.net_py)
    results = json.loads(a.out.read_text()) if a.out.exists() else {}
    for mesh in (float(x) for x in a.meshes.split(",")):
        n = setup(nd, mesh, a.tether)
        cal = nd.calibrate(nd.Sim())
        key = f"mesh {mesh:.3f} m{' tethered' if a.tether else ''}"
        rows = []
        rng = np.random.default_rng(11)
        for k in range(a.shots):
            t0 = time.time()
            r = one(nd, cal, rng, a.tether)
            r["seconds"] = round(time.time() - t0, 1)
            rows.append(r)
            print(f"{key}: {n} x {n} knots, net {nd.NET_MASS * 1e3:.0f} g, cartridge ~{nd.NET_MASS * 1e3 + 4 * CORNER_G + 8:.0f} g; shot {k + 1}: "
                  + ", ".join(f"{x} {v:.2f}" if isinstance(v, float) else f"{x} {v}" for x, v in r.items()), flush=True)
        results[key] = {"knots": n, "net_g": nd.NET_MASS * 1e3, "height_m": HEIGHT, "range_m": RANGE, "bandit_cda_m2": BANDIT_CDA,
                        "line": {"length_m": LINE, "stretch_m": LINE_STRETCH} if a.tether else None, "shots": rows}
        a.out.write_text(json.dumps(results, indent=1, default=lambda o: None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
