"""Pictures of a match: one summary figure, a replay video drawn with matplotlib (no OpenGL needed), and — where an
OpenGL context exists (``MUJOCO_GL=egl``) — the same match through a MuJoCo camera."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

COLOR = {"A": "#d62728", "B": "#1f77b4"}


def _field(ax, a, side=False):
    L, W = a.length / 2, a.width / 2
    if side:
        ax.axhline(0, color="#3c6e3c", lw=2)
        for team, c in a.hoops.items():
            ax.plot([c[0], c[0]], [0, a.hoop_stick], color=COLOR[team], lw=2)
            ax.plot([c[0], c[0]], [c[2] - a.hoop_radius, c[2] + a.hoop_radius], color=COLOR[team], lw=4)
        ax.set(xlim=(-L - 2, L + 2), ylim=(0, a.ceiling + 1), xlabel="x [m]", ylabel="height [m]")
    else:
        ax.add_patch(__import__("matplotlib.pyplot").pyplot.Rectangle((-L, -W), 2 * L, 2 * W, fc="#e9f2e4", ec="#3c6e3c", lw=1.2))
        ax.plot([0, 0], [-W, W], color="#3c6e3c", lw=0.8)
        for team, c in a.hoops.items():
            ax.plot([c[0], c[0]], [c[1] - a.hoop_radius, c[1] + a.hoop_radius], color=COLOR[team], lw=4)
        ax.set(xlim=(-L - 2, L + 2), ylim=(-W - 2, W + 2), xlabel="x [m]", ylabel="y [m]")
        ax.set_aspect("equal")


def summary(result: dict, path=None):
    """The match on one sheet: the tracks from above (thick where the aircraft carries the token) with the goals,
    tags and faults; the same from the side; who held the token, and the score, against time."""
    import matplotlib.pyplot as plt
    a = result["arena"]
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[2.4, 1.0, 0.8], width_ratios=[2.2, 1])
    top, side, line, txt = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[:, 1])
    _field(top, a)
    _field(side, a, side=True)
    bal = result["balloons"]
    for i in range(bal.shape[1] if bal.ndim == 3 else 0):
        top.plot(bal[:, i, 0], bal[:, i, 1], color="#f0a500", lw=0.8, alpha=0.6)
        top.add_patch(plt.Circle(bal[-1, i, :2], a.balloon_radius, color="#f0a500", alpha=0.7))
        side.plot(bal[:, i, 0], bal[:, i, 2], color="#f0a500", lw=0.8, alpha=0.6)
    for name, p in result["players"].items():
        h, c = p["hist"], COLOR[p["team"]]
        top.plot(h["x"], h["y"], color=c, lw=0.6, alpha=0.7, label=f"{name} — {p['pilot']} ({p['variant']})")
        hold = h["token"] > 0.5
        top.scatter(h["x"][hold], h["y"][hold], s=5, color=c)
        side.plot(h["x"], h["z"], color=c, lw=0.6, alpha=0.8)
        line.fill_between(h["t"], 0, hold.astype(float), step="post", color=c, alpha=0.35)
    mark = {"goal": ("*", 220, "k"), "tag": ("D", 50, "k"), "foul": ("X", 90, "#7f0000"), "crash": ("X", 70, "#7f0000")}
    for e in result["events"]:
        if e["what"] in mark and "who" in e:
            p = result["players"][e["who"]]
            k = int(np.searchsorted(p["hist"]["t"], e["t"])) - 2
            k = min(max(k, 0), len(p["hist"]["t"]) - 1)
            m, sz, col = mark[e["what"]]
            top.scatter([p["hist"]["x"][k]], [p["hist"]["y"][k]], marker=m, s=sz, color=COLOR[p["team"]], edgecolors=col, zorder=5)
            if e["what"] == "goal":
                line.axvline(e["t"], color=COLOR[p["team"]], lw=2)
    top.legend(fontsize=8, loc="upper center", ncol=2)
    top.set_title(f"score  A {result['score']['A']} : {result['score']['B']} B   ({result['duration']:.0f} s)   ★ goal  ◆ tag  ✕ fault   dots: carrying the token")
    line.set(xlabel="s", yticks=[], xlim=(0, result["duration"]), title="who holds the token (bars: goals)")
    txt.axis("off")
    lines = [f"winner: {result['winner'] or 'a draw'}", ""]
    for name, p in result["players"].items():
        lines += [f"{name} {p['pilot']} ({p['variant']}, team {p['team']})", f"   goals {p['goals']}  fouls {p['fouls']}  crashes {p['crashes']}",
                  f"   {p['energy_wh']:.1f} of {p['pack_wh']:.0f} Wh, max {p['max_speed'] * 3.6:.0f} km/h", ""]
    lines.append("events")
    for e in result["events"]:
        if e["what"] == "token":
            lines.append(f" {e['t']:5.1f}  token → {e['to']} ({e['why']})")
        elif e["what"] == "tag":
            lines.append(f" {e['t']:5.1f}  TAG {e['who']} takes it from {e['from']}")
        elif e["what"] == "goal":
            lines.append(f" {e['t']:5.1f}  GOAL {e['who']}  {e['score']['A']}:{e['score']['B']}")
        elif e["what"] in ("foul", "crash"):
            lines.append(f" {e['t']:5.1f}  {e['what']} {e['who']} ({e['on']})")
    txt.text(0, 1, "\n".join(lines[:58]), va="top", family="monospace", fontsize=7.5)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=110)
    return fig


def video(result: dict, path, fps: int = 20, speed: float = 1.0, seconds: float | None = None) -> Path | None:
    """The match as an mp4 from above and from the side (matplotlib frames through OpenCV). ``speed`` 1 is real time."""
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a = result["arena"]
    players = result["players"]
    t = next(iter(players.values()))["hist"]["t"]
    t_end = min(seconds or t[-1], t[-1])
    frames = np.arange(0.0, t_end, speed / fps)
    fig, (top, side) = plt.subplots(2, 1, figsize=(9.6, 7.2), gridspec_kw={"height_ratios": [2.3, 1]}, dpi=100)
    _field(top, a)
    _field(side, a, side=True)
    bal = result["balloons"]
    nb = bal.shape[1] if bal.ndim == 3 else 0
    b_top = [plt.Circle((0, 0), a.balloon_radius, color="#f0a500", alpha=0.85) for _ in range(nb)]
    b_side = [plt.Circle((0, 0), a.balloon_radius, color="#f0a500", alpha=0.85) for _ in range(nb)]
    for c1, c2 in zip(b_top, b_side):
        top.add_patch(c1), side.add_patch(c2)
    side.set_aspect("equal")
    art = {}
    for name, p in players.items():
        c = COLOR[p["team"]]
        art[name] = (top.plot([], [], color=c, lw=3, solid_capstyle="round")[0], top.plot([], [], color=c, lw=0.8, alpha=0.5)[0],
                     side.plot([], [], color=c, lw=3, solid_capstyle="round")[0], top.plot([], [], "o", ms=16, mfc="none", mec="k", mew=1.5)[0])
    tok = top.plot([], [], "o", ms=9, color="#2ca02c", mec="k")[0]
    tok_s = side.plot([], [], "o", ms=9, color="#2ca02c", mec="k")[0]
    title = top.set_title("")
    fig.tight_layout()
    path = Path(path)
    w, h = fig.canvas.get_width_height()
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    goals = [(e["t"], e["score"]) for e in result["events"] if e["what"] == "goal"]
    for tf in frames:
        k = min(int(np.searchsorted(t, tf)), len(t) - 1)
        for i in range(nb):
            b_top[i].center = bal[k, i, :2]
            b_side[i].center = (bal[k, i, 0], bal[k, i, 2])
        for name, p in players.items():
            hh = p["hist"]
            x, y, z = hh["x"][k], hh["y"][k], hh["z"][k]
            bx, by, bz = hh["bx"][k], hh["by"][k], hh["bz"][k]
            body, trail, sbody, ring = art[name]
            body.set_data([x - 0.6 * bx, x + 0.6 * bx], [y - 0.6 * by, y + 0.6 * by])
            j = max(k - 40, 0)
            trail.set_data(hh["x"][j:k + 1], hh["y"][j:k + 1])
            sbody.set_data([x - 0.6 * bx, x + 0.6 * bx], [z - 0.6 * bz, z + 0.6 * bz])
            ring.set_data([x], [y]) if hh["token"][k] > 0.5 else ring.set_data([], [])
        tk, holder = result["token_hist"][k]
        tok.set_data([tk[0]], [tk[1]]) if holder is None else tok.set_data([], [])
        tok_s.set_data([tk[0]], [tk[2]]) if holder is None else tok_s.set_data([], [])
        sc = next((s for tt, s in reversed(goals) if tt <= tf), {"A": 0, "B": 0})
        title.set_text(f"A {sc['A']} : {sc['B']} B     t = {tf:5.1f} s     (ring: carrying the token; green: the free token)")
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        out.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    out.release()
    plt.close(fig)
    return path if path.exists() and path.stat().st_size > 0 else None


def mujoco_video(result: dict, path, camera: str = "grandstand", fps: int = 30, size=(1280, 720)) -> Path | None:
    """The recorded match through a MuJoCo camera (``overview``, ``grandstand``, ``a1_chase``, ``b1_chase``). Needs an
    OpenGL context (``MUJOCO_GL=egl`` or a display); returns None without one. Written against the recorded ``qpos``;
    not run on the machine that wrote it (it had no OpenGL)."""
    try:
        import cv2
        import mujoco
        m = mujoco.MjModel.from_xml_string(result["xml"])
        d = mujoco.MjData(m)
        r = mujoco.Renderer(m, size[1], size[0])
    except Exception as exc:                                      # no renderer here
        print(f"no MuJoCo renderer ({type(exc).__name__}: {exc}); use replay.video")
        return None
    t = next(iter(result["players"].values()))["hist"]["t"]
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for tf in np.arange(0.0, t[-1], 1.0 / fps):
        k = min(int(np.searchsorted(t, tf)), len(t) - 1)
        d.qpos[:] = result["qpos"][k]
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera=camera)
        out.write(cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR))
    out.release()
    return Path(path)
