"""A recoilless net launcher against bandit drones (MuJoCo test).

The launcher is the recoilless golf cannon (``golf_recoilless``: 3 L tank, 1.5" diaphragm valve, 600 mm tube, bore
42.7 mm, rear venturi) firing a net cartridge instead of a ball:

- **cartridge** 150 g: a 2.4 x 2.4 m net of 1 mm braided Dyneema, 0.2 m mesh (13 x 13 knots, ~55 g), four 20 g corner
  weights, a 15 g shell. In the tube the corner weights sit in four splayed pockets of the shell, so at the muzzle
  they leave diverging 12 degrees from the axis and pull the net open (the classic net-gun layout);
- **the shot** (``design``): 40 m/s, zero net recoil at 8.5 bar with a 36 mm throat (a plain gun: 2.35 bar);
  air 22 g per shot, back-blast to ~7 m.

The MuJoCo model:
- **net**: 169 knots (free bodies, small spheres; the corner ones carry the weights) joined by rope tendons (tension
  only: a tendon limit at the mesh length); air drag on every knot (its share of cord, Cd 1.2 across it, skin friction along it, over the link's present length);
- **drone**: a 1.2 kg quadcopter (0.35 m diagonal, four 0.18 m rotors), holding its position (or a straight course)
  with a PD controller acting as its total thrust and torque; when a knot touches a rotor or an arm the props are
  fouled: thrust off 0.05 s later. A **capture**: fouled and down (below 0.5 m) within the run;
- **aim**: at the drone's position, the elevation from a calibration flight of the net's centre (no drone), the lead
  for a moving drone from the flight time; then scattered (0.5 deg, 3 % speed, standard deviations).

    python net_drone.py [--ranges 5,10,15,20,25] [--shots 8] [--height 4] [--drone-speed 0] [--out net_drone_results]
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import mujoco
import numpy as np

RHO, G = 1.204, 9.81
N = 13                       # knots per side
NET = 2.4                    # m, the net's side
MESH = NET / (N - 1)
CORD_D = 0.001
NET_MASS = 0.055
CORNER = 0.020
V0 = 40.0                    # m/s, the cartridge at the muzzle
SPLAY = math.radians(12.0)   # the corner weights' divergence
MUZZLE = np.array([0.0, 0.0, 0.5])
TENDON_DAMPING = 0.02         # N s/m along each link: the cord's own hysteresis, takes the snap out of the net
DRONE_M = 1.2
ARM = 0.125                  # rotor centre from the drone's centre, along x and y
ROTOR_R = 0.09
TITLE = "Recoilless net launcher (golf cannon, 40 m/s, 2.4 m net) vs a 1.2 kg quadcopter"
TAG = ""

# the light interceptor (``--light``): 32 mm bore, 0.35 m tube, 0.25 L plenum at 32.7 bar, 1" hammer valve, rear venturi
# throat 27.2 mm: 45 m/s, zero recoil, 8.8 g of air per shot (golf_recoilless with proj=, see net_light_design)
LIGHT = dict(N=11, NET=1.5, NET_MASS=0.030, CORNER=0.008, V0=45.0)
DRONES = {   # mass kg, arm (rotor centre, along x and y) m, rotor radius m
    "quad": (1.2, 0.125, 0.09),     # a 0.35 m (diagonal) camera quadcopter, 9" props
    "fpv": (0.6, 0.09, 0.064),      # a 5" FPV racer, 0.25 m diagonal
}


def configure(light=False, drone="quad"):
    """Set the net (the golf cannon's 2.4 m net, or the light 1.5 m one) and the drone."""
    global N, NET, MESH, NET_MASS, CORNER, V0, DRONE_M, ARM, ROTOR_R, TITLE, TAG
    if light:
        N, NET, NET_MASS, CORNER, V0 = (LIGHT[k] for k in ("N", "NET", "NET_MASS", "CORNER", "V0"))
    MESH = NET / (N - 1)
    DRONE_M, ARM, ROTOR_R = DRONES[drone]
    TITLE = (f"{'Light recoilless net launcher (45 m/s, 1.5 m net, 70 g cartridge)' if light else 'Recoilless net launcher (golf cannon, 40 m/s, 2.4 m net)'}"
             f" vs a {DRONE_M:g} kg {'FPV racer' if drone == 'fpv' else 'quadcopter'}")
    TAG = f"{'light_' if light else ''}{drone}_"


def xml():
    knot_m = NET_MASS / (N * N)
    bodies, tendons = [], []
    for i in range(N):
        for j in range(N):
            corner = i in (0, N - 1) and j in (0, N - 1)
            m = knot_m + (CORNER if corner else 0.0)
            r = 0.012 if corner else 0.008
            bodies.append(f'<body name="k{i}_{j}" pos="0 {(i - (N - 1) / 2) * 0.004} {0.5 + (j - (N - 1) / 2) * 0.004}"><freejoint/>'
                          f'<geom name="kg{i}_{j}" type="sphere" size="{r}" mass="{m}" contype="2" conaffinity="1" '
                          f'condim="3" friction="1.2"/><site name="s{i}_{j}"/></body>')
    for i in range(N):
        for j in range(N):
            for di, dj in ((1, 0), (0, 1)):
                a, b = i + di, j + dj
                if a < N and b < N:
                    tendons.append(f'<spatial limited="true" range="0 {MESH}" solreflimit="0.004 1" damping="{TENDON_DAMPING}" width="0.002">'
                                   f'<site site="s{i}_{j}"/><site site="s{a}_{b}"/></spatial>')
    rotors = "".join(
        f'<geom name="arm{k}" type="capsule" fromto="0 0 0 {sx * ARM} {sy * ARM} 0" size="0.01" mass="0.05" contype="1" conaffinity="2"/>'
        f'<geom name="rot{k}" type="cylinder" pos="{sx * ARM} {sy * ARM} 0.03" size="{ROTOR_R} 0.006" mass="0.03" contype="1" conaffinity="2" rgba="0.2 0.2 0.2 0.5"/>'
        for k, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))))
    return f"""
<mujoco model="net_vs_drone">
  <option timestep="1e-4" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic"/>
  <worldbody>
    <geom name="ground" type="plane" size="80 40 0.1" contype="1" conaffinity="3"/>
    <body name="drone" pos="20 0 4"><freejoint/>
      <geom name="hull" type="box" size="0.06 0.06 0.025" mass="{DRONE_M - 0.32}" contype="1" conaffinity="2"/>
      {rotors}
    </body>
    {''.join(bodies)}
  </worldbody>
  <tendon>{''.join(tendons)}</tendon>
</mujoco>"""


class Sim:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_string(xml())
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.kb = np.array([m.body(f"k{i}_{j}").id for i in range(N) for j in range(N)])
        self.kdof = np.array([m.jnt_dofadr[m.body_jntadr[b]] for b in self.kb])
        self.kq = np.array([m.jnt_qposadr[m.body_jntadr[b]] for b in self.kb])
        self.kgeom = set(m.geom(f"kg{i}_{j}").id for i in range(N) for j in range(N))
        self.drone = m.body("drone").id
        self.ddof = m.jnt_dofadr[m.body_jntadr[self.drone]]
        self.dq = m.jnt_qposadr[m.body_jntadr[self.drone]]
        self.foul_geoms = set(m.geom(n).id for n in ("arm0", "arm1", "arm2", "arm3", "rot0", "rot1", "rot2", "rot3"))
        self.drone_geoms = self.foul_geoms | {m.geom("hull").id}
        # the links (knot index pairs) for the cord drag; the corner weights' own drag
        la, lb = [], []
        for i in range(N):
            for j in range(N):
                for di, dj in ((1, 0), (0, 1)):
                    if i + di < N and j + dj < N:
                        la.append(i * N + j)
                        lb.append((i + di) * N + j + dj)
        self.la, self.lb = np.array(la), np.array(lb)
        self.cda = np.zeros(N * N)
        for i in (0, N - 1):
            for j in (0, N - 1):
                self.cda[i * N + j] = 0.47 * math.pi * 0.012 ** 2
        self.total_mass = self.m.body_subtreemass[0] - DRONE_M

    def reset(self, elev, az, speed, drone_pos, drone_vel, with_drone=True, origin=None):
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        axis = np.array([math.cos(elev) * math.cos(az), math.cos(elev) * math.sin(az), math.sin(elev)])
        e1 = np.cross(axis, [0, 0, 1.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(e1, axis)
        for i in range(N):
            for j in range(N):
                u, v = (i - (N - 1) / 2) / ((N - 1) / 2), (j - (N - 1) / 2) / ((N - 1) / 2)    # -1..1
                q, dv = self.kq[i * N + j], self.kdof[i * N + j]
                d.qpos[q:q + 3] = (MUZZLE if origin is None else origin) + axis * 0.05 + 0.02 * (u * e1 + v * e2)
                d.qpos[q + 3:q + 7] = (1, 0, 0, 0)
                direc = axis + math.tan(SPLAY) * (u * e1 + v * e2)       # corners splay 12 deg, the rest in proportion
                d.qvel[dv:dv + 3] = speed * direc / np.linalg.norm(direc)
        p = drone_pos if with_drone else np.array([0.0, 30.0, 4.0])
        d.qpos[self.dq:self.dq + 3] = p
        d.qpos[self.dq + 3:self.dq + 7] = (1, 0, 0, 0)
        d.qvel[self.ddof:self.ddof + 3] = drone_vel if with_drone else 0
        self.target0, self.tvel, self.fouled_at, self.with_drone = np.array(p, float), np.array(drone_vel, float), None, with_drone
        mujoco.mj_forward(m, d)

    def step(self):
        m, d = self.m, self.d
        # drag: on every link of cord, across it (Cd 1.2) and along it (skin friction 0.01), over its present length
        # between the knots (a bunched net shields its own cord: the drag grows as it opens), half to each knot;
        # plus the corner weights
        v = d.qvel[self.kdof[:, None] + np.arange(3)]
        x = d.xpos[self.kb]
        lv = x[self.lb] - x[self.la]
        L = np.linalg.norm(lv, axis=1) + 1e-9
        tdir = lv / L[:, None]
        vm = 0.5 * (v[self.la] + v[self.lb])
        vt = np.sum(vm * tdir, axis=1)[:, None] * tdir
        vn = vm - vt
        fl = -0.5 * RHO * L[:, None] * (1.2 * CORD_D * np.linalg.norm(vn, axis=1)[:, None] * vn
                                        + 0.01 * math.pi * CORD_D * np.linalg.norm(vt, axis=1)[:, None] * vt)
        f = np.zeros((N * N, 3))
        np.add.at(f, self.la, 0.5 * fl)
        np.add.at(f, self.lb, 0.5 * fl)
        sp = np.linalg.norm(v, axis=1)
        f -= 0.5 * RHO * (self.cda * sp)[:, None] * v
        d.xfrc_applied[self.kb, :3] = f
        # the drone: thrust + attitude PD, until fouled
        p = d.qpos[self.dq:self.dq + 3]
        quat = d.qpos[self.dq + 3:self.dq + 7]
        vel = d.qvel[self.ddof:self.ddof + 3]
        om = d.qvel[self.ddof + 3:self.ddof + 6]
        f = np.zeros(6)
        if self.with_drone and (self.fouled_at is None or d.time - self.fouled_at < 0.05):
            target = self.target0 + self.tvel * d.time
            acc = 4.0 * (target - p) + 3.0 * (self.tvel - vel) + np.array([0, 0, G])
            thrust = DRONE_M * acc
            thrust = thrust * min(1.0, 2.5 * DRONE_M * G / max(np.linalg.norm(thrust), 1e-9))
            rv = np.zeros(3)
            mujoco.mju_quat2Vel(rv, quat, 1.0)
            f[:3] = thrust
            f[3:] = -0.6 * rv - 0.08 * om
        else:
            f[:3] = -0.3 * vel                                   # a tumbling drone's drag
        d.xfrc_applied[self.drone] = f
        mujoco.mj_step(m, d)
        if self.with_drone and self.fouled_at is None and d.ncon:
            g1, g2 = d.contact.geom1[:d.ncon], d.contact.geom2[:d.ncon]
            for a, b in zip(g1, g2):
                if (a in self.kgeom and b in self.foul_geoms) or (b in self.kgeom and a in self.foul_geoms):
                    self.fouled_at = d.time
                    break

    def centre(self):
        return self.d.xpos[self.kb].mean(axis=0)

    def span(self):
        c = self.d.xpos[self.kb[[0, N - 1, N * (N - 1), N * N - 1]]]
        return max(np.linalg.norm(c[0] - c[3]), np.linalg.norm(c[1] - c[2]))


def calibrate(sim, t_max=2.0):
    """The net centre's flight (no drone) fired level, 50 m up (clear of the ground): (t, x, z change, span)."""
    up = np.array([0.0, 0.0, 50.0])
    sim.reset(0.0, 0.0, V0, None, np.zeros(3), with_drone=False, origin=up)
    rows = []
    while sim.d.time < t_max:
        for _ in range(50):
            sim.step()
        c = sim.centre()
        rows.append((sim.d.time, c[0], c[2] - up[2], sim.span()))
        if c[2] < up[2] - 6:
            break
    return np.array(rows)


def aim(cal, target, drone_vel):
    """Elevation and azimuth to put the net's centre on the drone: the level flight's drop added back and rotated
    (small angles), the lead from the flight time to the drone's (moved) range."""
    p = np.array(target, float)
    for _ in range(3):
        rng = math.hypot(p[0], p[1])
        t = float(np.interp(rng, cal[:, 1], cal[:, 0]))
        drop = float(np.interp(rng, cal[:, 1], cal[:, 2]))
        p = np.array(target, float) + drone_vel * t
    rng = math.hypot(p[0], p[1])
    elev = math.atan2(p[2] - MUZZLE[2] - drop, rng)
    return elev, math.atan2(p[1], p[0]), t


def shot(sim, cal, target, drone_vel, rng_gen, scatter_deg=0.5, scatter_speed=0.03, t_max=3.0, record=False):
    elev, az, t_fly = aim(cal, target, drone_vel)
    elev += math.radians(scatter_deg) * rng_gen.standard_normal()
    az += math.radians(scatter_deg) * rng_gen.standard_normal()
    speed = V0 * (1 + scatter_speed * rng_gen.standard_normal())
    sim.reset(elev, az, speed, np.array(target, float), drone_vel)
    path, snaps, drone_path = [], {}, []
    snap_t = sorted({0.05, 0.15, round(t_fly, 2), round(t_fly + 0.4, 2)})
    while sim.d.time < t_max:
        for _ in range(20):
            sim.step()
        t = sim.d.time
        if record:
            path.append((t, *sim.centre()))
            drone_path.append((t, *sim.d.xpos[sim.drone]))
            for st in snap_t:
                if st not in snaps and t >= st:
                    snaps[st] = sim.d.xpos[sim.kb].copy()
        if sim.fouled_at is not None and sim.d.xpos[sim.drone][2] < 0.5:
            break
        if sim.fouled_at is None and t > t_fly + 0.8:
            break
    captured = sim.fouled_at is not None and sim.d.xpos[sim.drone][2] < 0.5
    return dict(captured=captured, fouled=sim.fouled_at is not None, t_foul=sim.fouled_at, t_fly=t_fly,
                elev=math.degrees(elev), path=np.array(path), drone_path=np.array(drone_path), snaps=snaps)


def run(ranges, shots, height, drone_speed, out, seed=5):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sim = Sim()
    t0 = time.time()
    cal = calibrate(sim)
    print(f"calibration: net open to {cal[:, 3].max():.2f} m, at 5 m: span {np.interp(5, cal[:, 1], cal[:, 3]):.2f} m, "
          f"at 10 m {np.interp(10, cal[:, 1], cal[:, 3]):.2f} m; centre speed falls from {V0} m/s to "
          f"{np.interp(10, cal[:, 1], np.gradient(cal[:, 1], cal[:, 0])):.1f} m/s at 10 m  [{time.time() - t0:.0f} s]", flush=True)
    rng = np.random.default_rng(seed)
    results = {}
    for R in ranges:
        target = np.array([R, 0.0, height])
        vel = np.array([0.0, drone_speed, 0.0])
        rs = []
        for k in range(shots):
            r = shot(sim, cal, target, vel, rng, record=(k == 0))
            rs.append(r)
            print(f"  range {R:4.0f} m, shot {k + 1}/{shots}: {'CAPTURED' if r['captured'] else ('fouled' if r['fouled'] else 'miss')}"
                  f"  (flight {r['t_fly']:.2f} s, elevation {r['elev']:.1f} deg)  [{time.time() - t0:.0f} s]", flush=True)
        results[R] = rs
    return cal, results


def plot(cal, results, height, drone_speed, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(out)
    ranges = sorted(results)
    p = [np.mean([r["captured"] for r in results[R]]) for R in ranges]
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    a = ax[0, 0]
    a.plot(cal[:, 1], cal[:, 3], color="tab:blue")
    dd = 2 * math.sqrt(2) * ARM + 2 * ROTOR_R
    a.axhline(dd, color="tab:red", ls="--", lw=0.8)
    a.annotate(f"drone {dd:.2f} m", (0.5, dd + 0.03), fontsize=8, color="tab:red")
    a.set_xlabel("distance [m]")
    a.set_ylabel("net span (corner to corner) [m]")
    a.set_title("Net opening (calibration, level, no drone)")
    a2 = a.twinx()
    vx = np.gradient(cal[:, 1], cal[:, 0])
    a2.plot(cal[:, 1], vx, color="tab:gray", ls=":")
    a2.set_ylabel("net centre speed [m/s]", color="tab:gray")
    a = ax[0, 1]
    a.bar([str(R) for R in ranges], [100 * x for x in p], color="tab:green")
    for i, R in enumerate(ranges):
        fo = np.mean([r["fouled"] for r in results[R]])
        a.annotate(f"{100 * p[i]:.0f}% ({100 * fo:.0f}% fouled)", (i, 100 * p[i] + 2), ha="center", fontsize=8)
    a.set_ylim(0, 110)
    a.set_xlabel("range [m]")
    a.set_ylabel("captured [%]")
    a.set_title(f"Capture rate, {len(results[ranges[0]])} shots per range, drone at {height} m"
                + (f", crossing at {drone_speed} m/s" if drone_speed else ", hovering"))
    # side view: the first shot at each range, the net centre and snapshots
    a = ax[1, 0]
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(ranges)))
    for c, R in zip(cols, ranges):
        r = results[R][0]
        if len(r["path"]):
            a.plot(r["path"][:, 1], r["path"][:, 3], color=c, lw=1, label=f"{R} m ({'captured' if r['captured'] else 'miss'})")
            a.plot(r["drone_path"][:, 1], r["drone_path"][:, 3], color=c, lw=2, ls="--")
            for st, kn in r["snaps"].items():
                a.plot(kn[:, 0], kn[:, 2], ".", color=c, ms=1.5)
    a.set_xlabel("x [m]")
    a.set_ylabel("z [m]")
    a.set_title("Side view: net centre (line), knots at snapshots (dots), drone (dashed)")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)
    a.set_aspect("equal", adjustable="datalim")
    # the net seen from the launcher at the drone, the 10 m shot (or the nearest)
    a = ax[1, 1]
    R = min(ranges, key=lambda x: abs(x - 10))
    r = results[R][0]
    if r["snaps"]:
        st = max(r["snaps"])
        kn = r["snaps"][st]
        for i in range(N):
            a.plot(kn[i * N:(i + 1) * N, 1], kn[i * N:(i + 1) * N, 2], color="tab:blue", lw=0.6)
            a.plot(kn[i::N, 1], kn[i::N, 2], color="tab:blue", lw=0.6)
        dp = r["drone_path"]
        j = np.searchsorted(dp[:, 0], st)
        j = min(j, len(dp) - 1)
        a.plot(dp[j, 2], dp[j, 3], "s", color="tab:red", ms=10, label="drone")
        a.set_title(f"The net seen from the launcher, {R} m, t = {st:.2f} s")
        a.set_aspect("equal")
        a.legend()
    a.set_xlabel("y [m]")
    a.set_ylabel("z [m]")
    a.grid(alpha=0.3)
    for b in (ax[0, 0], ax[0, 1]):
        b.grid(alpha=0.3)
    fig.suptitle(TITLE, fontsize=12)
    fig.tight_layout()
    path = out / f"net_drone_{TAG}h{height:g}_v{drone_speed:g}.png"
    fig.savefig(path, dpi=120)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranges", default="5,10,15,20,25")
    ap.add_argument("--shots", type=int, default=8)
    ap.add_argument("--height", type=float, default=4.0)
    ap.add_argument("--drone-speed", type=float, default=0.0, help="the drone crosses at this speed (m/s), 0 = hovers")
    ap.add_argument("--out", default="net_drone_results")
    ap.add_argument("--light", action="store_true", help="the light interceptor: 1.5 m net, 70 g cartridge, 45 m/s")
    ap.add_argument("--drone", choices=sorted(DRONES), default="quad")
    a = ap.parse_args()
    configure(a.light, a.drone)
    cal, res = run([float(x) for x in a.ranges.split(",")], a.shots, a.height, a.drone_speed, a.out)
    for R, rs in res.items():
        print(f"range {R:4.0f} m: captured {sum(r['captured'] for r in rs)}/{len(rs)}, fouled {sum(r['fouled'] for r in rs)}")
    print("wrote", plot(cal, res, a.height, a.drone_speed, a.out))
