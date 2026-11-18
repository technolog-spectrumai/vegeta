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


def path_to(speed, elev, az, distance, wind=(0.0, 0.0), with_time=False):
    """Point-mass flight (the model MuJoCo integrates) until x = distance: (y, z[, t]) there; drag on the speed
    relative to the air (``wind``: the air's velocity, x along the range, y to the left)."""
    a, b = math.radians(elev), math.radians(az)
    p = MUZZLE.copy()
    v = np.array([speed * math.cos(a) * math.cos(b), speed * math.cos(a) * math.sin(b), speed * math.sin(a)])
    w = np.array([wind[0], wind[1], 0.0])
    dt, t = 1e-4 if distance < 60 else 5e-4, 0.0
    while p[0] < distance:
        u = v - w
        s = np.linalg.norm(u)
        v = v + (-0.5 * RHO * cd(s) * AREA * s / MASS * u + np.array([0, 0, -9.81])) * dt
        p = p + v * dt
        t += dt
        if p[2] < -1 or v[0] <= 0:
            return (0.0, -1e9, t) if with_time else (0.0, -1e9)
    return (p[1], p[2], t) if with_time else (p[1], p[2])


def flight_aim(speed, distance, target_z, wind=(0.0, 0.0)):
    """Elevation and azimuth (deg) that put a ball through (distance, 0, target_z) in this mean wind (bisection on
    the elevation, the azimuth corrected for the drift; calm: azimuth 0). Returns (elevation, azimuth, flight time)."""
    az = 0.0
    for _ in range(4 if wind[1] else 1):
        lo, hi = -2.0, 40.0                           # the low (direct) solution
        if path_to(speed, hi, az, distance, wind)[1] < target_z:
            raise ValueError(f"{speed:g} m/s cannot reach {distance:g} m at {target_z:g} m high")
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if path_to(speed, mid, az, distance, wind)[1] < target_z else (lo, mid)
        e = (lo + hi) / 2
        y = path_to(speed, e, az, distance, wind)[0]
        az -= math.degrees(math.atan2(y, distance))
    y, z, t = path_to(speed, e, az, distance, wind, True)
    return e, az, t


def run(shots=20, speed=132.0, distance=30.0, seed=1, gap=0.25, scatter_speed=0.01, scatter_deg=0.25,
        wind=(0.0, 0.0), gust=0.0, aim_wind=False):
    """``wind``: the mean air velocity (x along the range, + = tail wind; y across, + = blowing to the left) m/s;
    ``gust``: each shot's own wind deviation (standard deviation, both components); ``aim_wind``: the aim corrects
    for the mean wind (else it is aimed as in calm air)."""
    rng = np.random.default_rng(seed)
    m = mujoco.MjModel.from_xml_string(xml(distance, shots))
    d = mujoco.MjData(m)
    disc = m.geom("disc").id
    gid = [m.geom(f"g{i}").id for i in range(shots)]
    bid = [m.body(f"b{i}").id for i in range(shots)]
    centre_z = POLE_H
    elev, az0, t_fly = flight_aim(speed, distance, centre_z, wind if aim_wind else (0.0, 0.0))
    if wind != (0.0, 0.0) or gust:
        t_fly = max(t_fly, flight_aim(speed, distance, centre_z, wind)[2])
    winds = [np.array([wind[0] + gust * rng.standard_normal(), wind[1] + gust * rng.standard_normal(), 0.0])
             for _ in range(shots)]
    t_track = 1.4 * t_fly + 0.2                      # follow each ball this long
    fired = [False] * shots
    tracks = [[] for _ in range(shots)]
    hit = [None] * shots
    done = [False] * shots
    t_end = gap * (shots - 1) + t_track
    step = 0
    rec_every = 10                                   # record every 0.2 ms
    while d.time < t_end:
        i = int(d.time // gap)
        if i < shots and not fired[i]:              # fire ball i
            fired[i] = True
            s = speed * (1 + scatter_speed * rng.standard_normal())
            el = math.radians(elev + scatter_deg * rng.standard_normal())
            az = math.radians(az0 + scatter_deg * rng.standard_normal())
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
                u = d.qvel[dv:dv + 3] - winds[k]            # the speed relative to the air
                sp = np.linalg.norm(u)
                d.xfrc_applied[bid[k], :3] = -0.5 * RHO * cd(sp) * AREA * sp * u if sp > 0.1 else 0
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
                    dv = m.jnt_dofadr[m.body_jntadr[bid[k]]]
                    tracks[k].append((d.time - gap * k, *p, np.linalg.norm(d.qvel[dv:dv + 3])))
                    if p[0] > distance + 3 or (p[2] < 0.01 and d.time - gap * k > 0.05) or d.time - gap * k > t_track:
                        done[k] = True
                        d.xfrc_applied[bid[k], :3] = 0
    rings = []
    for h in hit:
        if h is None:
            rings.append(0)
        else:
            r = math.hypot(h[0], h[1])
            rings.append(max(0, 10 - int(r / 0.015)) if r <= DISC_R else 0)
    return dict(elev=elev, az=az0, wind=wind, gust=gust, aim_wind=aim_wind, t_fly=t_fly, tracks=[np.array(t) for t in tracks], hit=hit, rings=rings, distance=distance, speed=speed,
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
    wtxt = ""
    if res.get("wind", (0, 0)) != (0.0, 0.0) or res.get("gust"):
        wtxt = (f", wind ({res['wind'][0]:+g}, {res['wind'][1]:+g}) m/s ± {res['gust']:g}, "
                + ("aim corrected" if res["aim_wind"] else "aim uncorrected"))
    side.set_title(f"Side view: {n} shots, {res['speed']:.0f} m/s, elevation {res['elev']:.2f}°{wtxt}  (red: hit)")
    side.set_ylim(-0.1, max(1.8, max(tr[:, 3].max() for tr in res["tracks"] if len(tr)) + 0.2))
    top.plot([L, L], [-DISC_R, DISC_R], color="goldenrod", lw=6)
    ymax = max(1.0, 1.2 * max(abs(tr[:, 2]).max() for tr in res["tracks"] if len(tr)))
    top.set_ylim(-ymax, ymax)
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
    lim = 40.0
    for k, tr in enumerate(res["tracks"]):
        if res["hit"][k] is None and len(tr) and tr[:, 1].max() >= L:
            j = np.searchsorted(tr[:, 1], L)
            lim = max(lim, 1.15 * 100 * max(abs(tr[j, 2]), abs(tr[j, 3] - POLE_H)))
    face.set_xlim(-lim, lim)
    face.set_ylim(-lim, lim)
    face.set_aspect("equal")
    face.set_xlabel("cm")
    face.set_title(f"Target face at {L:.0f} m: {hits}/{n} hits, {sum(res['rings'])} points\n(o hit, x miss through the plane)")
    face.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    return out


def plot_energy(res, out):
    """Kinetic energy and speed along the flight for every shot, and the energy each hit delivered."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    L = res["distance"]
    m = MASS
    fig, (ax_e, ax_v, ax_h) = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw={"width_ratios": [1.3, 1.3, 1]})
    for k, tr in enumerate(res["tracks"]):
        if not len(tr):
            continue
        j = tr[:, 1] <= L + 0.5
        col = "tab:red" if res["hit"][k] is not None else "tab:gray"
        ax_e.plot(tr[j, 1], 0.5 * m * tr[j, 4] ** 2, color=col, lw=0.8, alpha=0.8)
        ax_v.plot(tr[j, 1], tr[j, 4], color=col, lw=0.8, alpha=0.8)
    v0 = res["speed"]
    for ax in (ax_e, ax_v):
        ax.axvline(L, color="k", lw=0.6, ls=":")
        ax.set_xlabel("distance x [m]")
        ax.grid(alpha=0.3)
        ax.set_xlim(0, L + 1)
    ax_e.set_ylabel("kinetic energy [J]")
    ax_e.set_title(f"Energy along the flight (muzzle {0.5 * m * v0 ** 2:.1f} J)")
    for lvl, name in ((1.0, "1 J"),):
        ax_e.axhline(lvl, color="tab:blue", lw=0.6, ls="--")
        ax_e.annotate(name, (1, lvl), xytext=(2, 2), textcoords="offset points", fontsize=8, color="tab:blue")
    ax_v.set_ylabel("speed [m/s]")
    ax_v.set_title(f"Speed (terminal ~{math.sqrt(2 * m * 9.81 / (RHO * CD0 * AREA)):.0f} m/s)")
    hits = [(k + 1, 0.5 * m * h[2] ** 2) for k, h in enumerate(res["hit"]) if h is not None]
    if hits:
        ax_h.bar([str(k) for k, _ in hits], [e for _, e in hits], color="tab:red")
        ax_h.set_title(f"Energy delivered by each hit at {L:.0f} m (mean {np.mean([e for _, e in hits]):.2f} J)")
    else:
        ax_h.text(0.5, 0.5, "no hits", ha="center", va="center", transform=ax_h.transAxes, fontsize=14)
        ax_h.set_title(f"Energy delivered at {L:.0f} m")
    ax_h.set_xlabel("shot")
    ax_h.set_ylabel("J")
    ax_h.grid(alpha=0.3, axis="y")
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
    ap.add_argument("--scatter-deg", type=float, default=0.25, help="direction scatter, 1 sigma (deg)")
    ap.add_argument("--scatter-speed", type=float, default=0.01, help="speed scatter, 1 sigma (fraction)")
    ap.add_argument("--wind", default="0,0", help="mean wind x,y m/s (x + = tail wind, y + = to the left)")
    ap.add_argument("--gust", type=float, default=0.0, help="each shot's wind deviation, 1 sigma m/s")
    ap.add_argument("--aim-wind", action="store_true", help="aim corrects for the mean wind")
    ap.add_argument("--gap", type=float, default=0.25, help="s between shots")
    a = ap.parse_args()
    r = run(a.shots, a.speed, a.distance, a.seed, gap=a.gap, scatter_speed=a.scatter_speed, scatter_deg=a.scatter_deg,
            wind=tuple(float(x) for x in a.wind.split(",")), gust=a.gust, aim_wind=a.aim_wind)
    hits = sum(h is not None for h in r["hit"])
    print(f"{a.shots} shots at {a.speed:.0f} m/s, {a.distance:.0f} m, elevation {r['elev']:.3f} deg: {hits} hits, "
          f"{sum(r['rings'])} points, rings {r['rings']}, the lollipop swung to {r['swing_deg']:.1f} deg")
    for k, h in enumerate(r["hit"]):
        if h is not None:
            print(f"  shot {k + 1:2d}: hit at ({h[0] * 100:+.1f}, {h[1] * 100:+.1f}) cm, {h[2]:.0f} m/s, ring {r['rings'][k]}")
    print(f"aim {r['elev']:.2f} deg, flight {r['t_fly']:.2f} s")
    print("wrote", plot(r, a.out))
    print("wrote", plot_energy(r, a.out.replace(".png", "_energy.png")))
