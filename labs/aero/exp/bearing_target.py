"""Target practice for the bearing-ball ejector (MuJoCo): N shots at a lollipop target, hits counted, the trajectories
plotted (matplotlib, no rendering).

The ground: flat. The launcher: the Pekari's muzzle 0.4 m up. The target: a lollipop, a 300 mm disc (20 mm plywood,
1.5 kg) on a spring hinge (20 N m/rad: a hit knocks it back, it swings up again) at the top of a 1.2 m pole;
rings every 15 mm (10 points in the middle).

Each shot: an 8 mm, 2.08 g steel ball at the game exit speed (``game_cfd``'s thin-tube result, default the
subsonic-fast case: 132 m/s), aimed at the bullseye with the elevation that the same drag model needs, then scattered
like the game's launcher (1 % in speed, 0.25 deg in direction, standard deviations). In flight MuJoCo integrates
gravity and the contacts; the drag is applied each step (Cd 0.45 plus the Mach rise of ``game_cfd.ball_aero``). A hit
is the ball's first contact with the disc; its place on the disc gives the ring.

    python bearing_target.py [--shots 20] [--speed 132] [--distance 30] [--seed 1] [--out bearing_target.png]
"""
from __future__ import annotations

import argparse
import math

import mujoco
import numpy as np

D, MASS = 0.008, 0.00208
RHO, A_SOUND = 1.204, 343.2
AREA = math.pi / 4 * D * D
MACH = np.array([0.0, 0.5, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0])
RISE = np.array([0.0, 0.0, 0.02, 0.08, 0.18, 0.33, 0.43, 0.48, 0.53, 0.48])
CD0 = 0.45
MUZZLE = np.array([0.0, 0.0, 0.40])
DISC_R, DISC_T, POLE_H = 0.150, 0.020, 1.20


def cd(v):
    return CD0 + float(np.interp(v / A_SOUND, MACH, RISE))


def xml(distance, n):
    balls = "\n".join(
        f'<body name="b{i}" pos="0 {-1 - 0.02 * i} 0.004"><freejoint/>'
        f'<geom name="g{i}" type="sphere" size="{D / 2}" mass="{MASS}" contype="1" conaffinity="2" condim="3"/></body>'
        for i in range(n))
    return f"""
<mujoco model="target_practice">
  <option timestep="2e-5" gravity="0 0 -9.81" integrator="implicitfast"/>
  <worldbody>
    <geom name="ground" type="plane" size="60 20 0.1" contype="2" conaffinity="1"/>
    <geom name="pole" type="cylinder" fromto="{distance} 0 0 {distance} 0 {POLE_H - DISC_R - 0.02}" size="0.015"
          contype="2" conaffinity="1"/>
    <body name="lolly" pos="{distance} 0 {POLE_H - DISC_R - 0.02}">
      <joint name="hinge" type="hinge" axis="0 1 0" stiffness="20" damping="0.4" range="-60 60" limited="true"/>
      <geom name="stick" type="cylinder" fromto="0 0 0 0 0 0.02" size="0.012" mass="0.05" contype="2" conaffinity="1"/>
      <geom name="disc" type="cylinder" pos="0 0 {0.02 + DISC_R}" euler="0 90 0" size="{DISC_R} {DISC_T / 2}" mass="1.5"
            contype="2" conaffinity="1"/>
    </body>
    {balls}
  </worldbody>
</mujoco>"""


def flight_aim(speed, distance, target_z):
    """The elevation that puts a ball from the muzzle through (distance, target_z) with this drag (bisection on a
    point-mass flight, the same model MuJoCo integrates)."""
    def z_at(elev):
        a = math.radians(elev)
        p = MUZZLE.copy()
        v = np.array([speed * math.cos(a), 0.0, speed * math.sin(a)])
        dt = 1e-4
        while p[0] < distance:
            s = np.linalg.norm(v)
            v = v + (-0.5 * RHO * cd(s) * AREA * s / MASS * v + np.array([0, 0, -9.81])) * dt
            p = p + v * dt
            if p[2] < -1:
                return -1e9
        return p[2]
    lo, hi = -2.0, 30.0
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if z_at(mid) < target_z else (lo, mid)
    return (lo + hi) / 2


def run(shots=20, speed=132.0, distance=30.0, seed=1, gap=0.25, scatter_speed=0.01, scatter_deg=0.25):
    rng = np.random.default_rng(seed)
    m = mujoco.MjModel.from_xml_string(xml(distance, shots))
    d = mujoco.MjData(m)
    disc = m.geom("disc").id
    gid = [m.geom(f"g{i}").id for i in range(shots)]
    bid = [m.body(f"b{i}").id for i in range(shots)]
    centre_z = POLE_H
    elev = flight_aim(speed, distance, centre_z)
    fired = [False] * shots
    tracks = [[] for _ in range(shots)]
    hit = [None] * shots
    done = [False] * shots
    t_end = gap * shots + 0.6
    step = 0
    rec_every = 10                                   # record every 0.2 ms
    while d.time < t_end:
        i = int(d.time // gap)
        if i < shots and not fired[i]:              # fire ball i
            fired[i] = True
            s = speed * (1 + scatter_speed * rng.standard_normal())
            el = math.radians(elev + scatter_deg * rng.standard_normal())
            az = math.radians(scatter_deg * rng.standard_normal())
            q = m.jnt_qposadr[m.body_jntadr[bid[i]]]
            dv = m.jnt_dofadr[m.body_jntadr[bid[i]]]
            d.qpos[q:q + 3] = MUZZLE
            d.qpos[q + 3:q + 7] = (1, 0, 0, 0)
            d.qvel[dv:dv + 3] = (s * math.cos(el) * math.cos(az), s * math.cos(el) * math.sin(az), s * math.sin(el))
            d.qvel[dv + 3:dv + 6] = 0
        # drag on the flying balls
        for k in range(shots):
            if fired[k] and not done[k]:
                dv = m.jnt_dofadr[m.body_jntadr[bid[k]]]
                v = d.qvel[dv:dv + 3]
                sp = np.linalg.norm(v)
                d.xfrc_applied[bid[k], :3] = -0.5 * RHO * cd(sp) * AREA * sp * v if sp > 0.1 else 0
        mujoco.mj_step(m, d)
        step += 1
        # contacts: the first one with the disc is a hit
        for c in d.contact[:d.ncon]:
            pair = {c.geom1, c.geom2}
            for k in range(shots):
                if fired[k] and hit[k] is None and gid[k] in pair and disc in pair:
                    # where on the disc: in the disc's own frame (its axis is the disc body's x)
                    R = d.xmat[m.body("lolly").id].reshape(3, 3)
                    local = R.T @ (np.array(c.pos) - d.geom_xpos[disc])
                    hit[k] = (local[1], local[2], float(np.linalg.norm(d.qvel[m.jnt_dofadr[m.body_jntadr[bid[k]]]:][:3])))
        if step % rec_every == 0:
            for k in range(shots):
                if fired[k] and not done[k]:
                    p = d.xpos[bid[k]].copy()
                    tracks[k].append((d.time, *p))
                    if p[0] > distance + 3 or (p[2] < 0.01 and d.time - gap * k > 0.05) or d.time - gap * k > 0.6:
                        done[k] = True
                        d.xfrc_applied[bid[k], :3] = 0
    rings = []
    for h in hit:
        if h is None:
            rings.append(0)
        else:
            r = math.hypot(h[0], h[1])
            rings.append(max(0, 10 - int(r / 0.015)) if r <= DISC_R else 0)
    return dict(elev=elev, tracks=[np.array(t) for t in tracks], hit=hit, rings=rings, distance=distance, speed=speed,
                swing_deg=math.degrees(d.qpos[m.jnt_qposadr[m.joint("hinge").id]]))


def plot(res, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    n = len(res["tracks"])
    hits = sum(h is not None for h in res["hit"])
    L = res["distance"]
    fig = plt.figure(figsize=(14, 8.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.2, 1])
    side, top, face = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])
    for k, tr in enumerate(res["tracks"]):
        if not len(tr):
            continue
        col = "tab:red" if res["hit"][k] is not None else "tab:gray"
        side.plot(tr[:, 1], tr[:, 3], color=col, lw=0.8, alpha=0.8)
        top.plot(tr[:, 1], tr[:, 2], color=col, lw=0.8, alpha=0.8)
    for ax in (side, top):
        ax.axvline(L, color="k", lw=0.5, ls=":")
        ax.set_xlim(-0.5, L + 2)
        ax.grid(alpha=0.3)
    # the lollipop in the side view, the ground
    side.plot([L, L], [0, POLE_H - DISC_R], color="saddlebrown", lw=3)
    side.plot([L, L], [POLE_H - DISC_R, POLE_H + DISC_R], color="goldenrod", lw=6)
    side.axhline(0, color="olive", lw=1)
    side.plot(0, MUZZLE[2], "ks", ms=6)
    side.set_ylabel("height z [m]")
    side.set_title(f"Side view: {n} shots, {res['speed']:.0f} m/s, elevation {res['elev']:.2f}°  (red: hit)")
    side.set_ylim(-0.1, max(1.8, max(tr[:, 3].max() for tr in res["tracks"] if len(tr)) + 0.2))
    top.plot([L, L], [-DISC_R, DISC_R], color="goldenrod", lw=6)
    top.set_ylim(-1.0, 1.0)
    top.set_xlabel("distance x [m]")
    top.set_ylabel("sideways y [m]")
    top.set_title("Top view")
    # the target face: rings and hits
    for r in np.arange(DISC_R, 0, -0.015):
        face.add_patch(Circle((0, 0), r * 100, fill=False, lw=0.6, color="k"))
    face.add_patch(Circle((0, 0), DISC_R * 100, color="goldenrod", alpha=0.25))
    face.add_patch(Circle((0, 0), 1.5, color="tab:red", alpha=0.4))
    for k, h in enumerate(res["hit"]):
        if h is not None:
            face.plot(h[0] * 100, h[1] * 100, "o", color="tab:red", ms=6, mec="k")
            face.annotate(str(k + 1), (h[0] * 100, h[1] * 100), xytext=(4, 4), textcoords="offset points", fontsize=7)
    # where the misses passed the target's plane
    for k, tr in enumerate(res["tracks"]):
        if res["hit"][k] is None and len(tr) and tr[:, 1].max() >= L:
            j = np.searchsorted(tr[:, 1], L)
            face.plot(tr[j, 2] * 100, (tr[j, 3] - POLE_H) * 100, "x", color="tab:gray", ms=6)
    face.set_xlim(-40, 40)
    face.set_ylim(-40, 40)
    face.set_aspect("equal")
    face.set_xlabel("cm")
    face.set_title(f"Target face at {L:.0f} m: {hits}/{n} hits, {sum(res['rings'])} points\n(o hit, x miss through the plane)")
    face.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=20)
    ap.add_argument("--speed", type=float, default=132.0)
    ap.add_argument("--distance", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="bearing_target.png")
    a = ap.parse_args()
    r = run(a.shots, a.speed, a.distance, a.seed)
    hits = sum(h is not None for h in r["hit"])
    print(f"{a.shots} shots at {a.speed:.0f} m/s, {a.distance:.0f} m, elevation {r['elev']:.3f} deg: {hits} hits, "
          f"{sum(r['rings'])} points, rings {r['rings']}, the lollipop swung to {r['swing_deg']:.1f} deg")
    for k, h in enumerate(r["hit"]):
        if h is not None:
            print(f"  shot {k + 1:2d}: hit at ({h[0] * 100:+.1f}, {h[1] * 100:+.1f}) cm, {h[2]:.0f} m/s, ring {r['rings'][k]}")
    print("wrote", plot(r, a.out))
