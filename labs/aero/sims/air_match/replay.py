"""Pictures of a match: one summary figure, a replay video drawn with matplotlib (no OpenGL needed), and — where an
OpenGL context exists (``MUJOCO_GL=egl``) — the same match through a MuJoCo camera."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

COLOR = {"A": "#d62728", "B": "#1f77b4"}


def _score(result) -> str:
    sc = result["score"]
    return "  ".join(f"{k} {v:g}" for k, v in sc.items()) if len(sc) != 2 or set(sc) != {"A", "B"} else f"A {sc['A']} : {sc['B']} B"


def _field(ax, a, side=False):
    L, W = a.length / 2, a.width / 2
    if hasattr(a, "half"):                                          # the circus: the spina and its turning posts
        if side:
            ax.fill([-a.half, a.half, a.half, -a.half], [0, 0, a.spina_h, a.spina_h], color="#bfb399")
            for x in (-a.half, a.half):
                ax.plot([x, x], [0, a.meta_h], color="#d9a633", lw=3)
        else:
            ax.fill([-a.half, a.half, a.half, -a.half], [-a.spina_w / 2, -a.spina_w / 2, a.spina_w / 2, a.spina_w / 2], color="#bfb399", zorder=2)
            for x in (-a.half, a.half):
                ax.plot([x], [0], "o", color="#d9a633", ms=7, zorder=3)
            ax.plot([0, 0], [-W, -a.spina_w / 2], color="k", lw=1.5, ls="--")
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
        h, c = p["hist"], p.get("color") or COLOR[p["team"]]
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
            pc = p.get("color") or COLOR[p["team"]]
            top.scatter([p["hist"]["x"][k]], [p["hist"]["y"][k]], marker=m, s=sz, color=pc, edgecolors=col, zorder=5)
            if e["what"] == "goal":
                line.axvline(e["t"], color=pc, lw=2)
    top.legend(fontsize=8, loc="upper center", ncol=2)
    top.set_title(f"{_score(result)}   ({result['duration']:.0f} s)   ★ goal  ◆ tag  ✕ fault   dots: carrying the token", fontsize=10)
    line.set(xlabel="s", yticks=[], xlim=(0, result["duration"]), title="who holds the token (bars: goals)")
    txt.axis("off")
    lines = [f"winner: {result['winner'] or 'a draw'}", ""]
    for name, p in result["players"].items():
        lines += [f"{name} {p['pilot']} ({p['variant']})  goals {p['goals']}  held {p.get('hold_s', 0):.0f} s  hoops {p.get('hoops', 0)}",
                  f"   fouls {p['fouls']}  crashes {p['crashes']}  {p['energy_wh']:.1f} Wh  max {p['max_speed'] * 3.6:.0f} km/h"]
    lines.append("events")
    for e in result["events"]:
        if e["what"] == "token":
            lines.append(f" {e['t']:5.1f}  token → {e['to']} ({e['why']})")
        elif e["what"] == "tag":
            lines.append(f" {e['t']:5.1f}  TAG {e['who']} takes it from {e['from']}")
        elif e["what"] == "goal":
            lines.append(f" {e['t']:5.1f}  GOAL {e['who']}  " + ":".join(str(v) for v in e["score"].values()))
        elif e["what"] == "hoop":
            lines.append(f" {e['t']:5.1f}  HOOP {e['who']} +{e['points']:g}")
        elif e["what"] == "formation":
            lines.append(f" {e['t']:5.1f}  → {e['to']} in {e['seconds']} s")
        elif e["what"] in ("foul", "crash"):
            lines.append(f" {e['t']:5.1f}  {e['what']} {e['who']} ({e['on']})")
    txt.text(0, 1, "\n".join(lines[:70]), va="top", family="monospace", fontsize=6.8)
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
        c = p.get("color") or COLOR[p["team"]]
        art[name] = (top.plot([], [], color=c, lw=3, solid_capstyle="round")[0], top.plot([], [], color=c, lw=0.8, alpha=0.5)[0],
                     side.plot([], [], color=c, lw=3, solid_capstyle="round")[0], top.plot([], [], "o", ms=16, mfc="none", mec="k", mew=1.5)[0])
    wind_arrow = top.quiver([-a.length / 2 + 14], [a.width / 2 - 8], [0.0], [0.0], color="#1f77b4", scale=1.0, scale_units="xy", angles="xy", width=0.004) if "race" in result else None
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
        if "race" in result:
            lapn = {n: int(sum(1 for e in result["events"] if e["what"] == "lap" and e["who"] == n and e["t"] <= tf)) for n in players}
            lead = sorted(players, key=lambda n: (-lapn[n], n))
            wh = result["race"].get("wind_hist")
            wtxt = ""
            if wh is not None and len(wh):
                kw_ = min(int(np.searchsorted(wh[:, 0], tf)), len(wh) - 1)
                wtxt = f"   wind {wh[kw_, 1]:.0f} m/s from {(wh[kw_, 2] + 180) % 360:.0f}°"
                wind_arrow.set_UVC(wh[kw_, 3], wh[kw_, 4])
            title.set_text(f"circus race   t = {tf:5.1f} s{wtxt}   laps: " + "  ".join(f"{n} {lapn[n]}" for n in lead))
        elif "formation" in result:
            f = result["formation"]
            kk = min(int(np.searchsorted(f["t"], tf)), len(f["t"]) - 1)
            shape = ("changing" if f["morphing"][kk] > 0.5 else "") + (" wedge" if f["wedge"][kk] > 0.5 else " column")
            title.set_text(f"formation chase   t = {tf:5.1f} s   {shape.strip()}   slot error {f['err_mean'][kk]:.2f} m   closest pair {f['closest'][kk]:.2f} m")
        elif goals or set(result["score"]) == {"A", "B"}:
            sc = next((s for tt, s in reversed(goals) if tt <= tf), {"A": 0, "B": 0})
            title.set_text(f"A {sc['A']} : {sc['B']} B     t = {tf:5.1f} s     (ring: carrying the token; green: the free token)")
        else:
            held = "  ".join(f"{n} {float(np.sum(p['hist']['token'][:k + 1]) * (t[1] - t[0])):.0f}s" for n, p in players.items())
            title.set_text(f"token chase   t = {tf:5.1f} s   held: {held}")
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


def formation_summary(result: dict, path=None):
    """The formation chase on one sheet: the tracks over the course, the slot error and the closest pair against time
    with the shape flown (wedge shaded), and the numbers."""
    import matplotlib.pyplot as plt
    a, f = result["arena"], result["formation"]
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[2.2, 1, 1], width_ratios=[2.4, 1])
    top, e1, e2, txt = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[:, 1])
    _field(top, a)
    c = f["course"]
    top.plot(np.append(c[:, 0], c[0, 0]), np.append(c[:, 1], c[0, 1]), ":", color="k", lw=0.8, label="the course")
    bal = result["balloons"]
    for i in range(bal.shape[1] if bal.ndim == 3 else 0):
        top.plot(bal[:, i, 0], bal[:, i, 1], color="#f0a500", lw=0.6, alpha=0.5)
        top.add_patch(plt.Circle(bal[-1, i, :2], a.balloon_radius, color="#f0a500", alpha=0.7))
    t_end = f["t"][-1]
    for name, p in result["players"].items():
        h = p["hist"]
        last = h["t"] > t_end - 6.0
        top.plot(h["x"], h["y"], color=p["color"], lw=0.4, alpha=0.35)
        top.plot(h["x"][last], h["y"][last], color=p["color"], lw=1.6, label=name)
        top.plot(h["x"][-1], h["y"][-1], "o", color=p["color"], ms=5, mec="k")
    top.legend(fontsize=7, ncol=5, loc="upper center")
    top.set_title("formation chase: seven behind the hare (thick: the last six seconds)")
    for ax_, key, lab in ((e1, "err_mean", "mean slot error [m]"), (e2, "closest", "closest pair, centre to centre [m]")):
        ax_.fill_between(f["t"], 0, 1, where=f["wedge"] > 0.5, transform=ax_.get_xaxis_transform(), color="#1f77b4", alpha=0.10, label="wedge")
        ax_.fill_between(f["t"], 0, 1, where=f["morphing"] > 0.5, transform=ax_.get_xaxis_transform(), color="#ff7f0e", alpha=0.18, label="changing")
        ax_.plot(f["t"], f[key], color="k", lw=0.9)
        ax_.set(ylabel=lab, xlim=(0, t_end))
        ax_.grid(alpha=0.3)
    e1.plot(f["t"], f["err_max"], color="0.6", lw=0.6)
    e1.legend(fontsize=7, loc="upper right")
    e2.axhline(f["summary"]["touching_at_m"], color="#7f0000", ls=":", lw=0.8)
    e2.axhline(f["spacing"], color="0.5", ls="--", lw=0.8)
    e2.set_xlabel("s")
    txt.axis("off")
    sm = f["summary"]
    lines = [f"{k.replace('_', ' ')}: {v:.2f}" if isinstance(v, float) else f"{k.replace('_', ' ')}: {v}" for k, v in sm.items() if k != "hoops_flown"]
    lines += ["", "hoops flown through:"] + [f"   {k}: {v}" for k, v in sm["hoops_flown"].items()] + ["", "changes of formation (Hungarian assignment):"]
    lines += [f" {ch['t']:5.1f} s → {ch['to']:6s} {ch['seconds']:.1f} s  crossings {ch['crossings']} (numbers kept: {ch['kept_numbers_crossings']})" for ch in f["changes"]]
    lines += ["", "faults:"] + [f" {e['t']:5.1f}  {e['what']} {e['who']} ({e.get('on', e.get('hoop', ''))})" for e in result["events"] if e["what"] in ("foul", "crash", "missed")]
    txt.text(0, 1, "\n".join(lines[:64]), va="top", family="monospace", fontsize=7.2)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=110)
    return fig


def race_summary(result: dict, path=None):
    """The circus race on one sheet: the lines flown, each racer's progress against time, the standings and the faults."""
    import matplotlib.pyplot as plt
    a, race = result["arena"], result["race"]
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], width_ratios=[2.2, 1])
    top, prog, txt = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])
    _field(top, a)
    bal = result["balloons"]
    for i in range(bal.shape[1] if bal.ndim == 3 else 0):
        top.plot(bal[:, i, 0], bal[:, i, 1], color="#f0a500", lw=0.5, alpha=0.5)
        top.add_patch(plt.Circle(bal[-1, i, :2], a.balloon_radius, color="#f0a500", alpha=0.7))
    place = {s["strategy"]: s for s in race["standings"]}
    for name, p in result["players"].items():
        h = p["hist"]
        fly = h["role"] == 0
        top.plot(np.where(fly, h["x"], np.nan), np.where(fly, h["y"], np.nan), color=p["color"], lw=0.8, alpha=0.8, label=f"{place[name]['place']}. {name}")
        laps_t = [0.0] + [e["t"] for e in result["events"] if e["what"] == "lap" and e["who"] == name]
        prog.step(laps_t, range(len(laps_t)), where="post", color=p["color"], lw=1.6)
    for e in race["faults"]:
        p = result["players"][e["who"]]
        k = min(max(int(np.searchsorted(p["hist"]["t"], e["t"])) - 2, 0), len(p["hist"]["t"]) - 1)
        top.scatter([p["hist"]["x"][k]], [p["hist"]["y"][k]], marker="X", s=70, color=p["color"], edgecolors="k", zorder=5)
        prog.axvline(e["t"], color=p["color"], lw=0.6, ls=":")
    hand, lab = top.get_legend_handles_labels()
    order = np.argsort([int(x.split(".")[0]) for x in lab])
    top.legend([hand[i] for i in order], [lab[i] for i in order], fontsize=8, ncol=7, loc="upper center")
    w = race["wind"]
    top.annotate("", xy=(0.12, 0.08), xytext=(0.03, 0.08), xycoords="axes fraction", arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=2))
    top.text(0.03, 0.11, f"wind {w['speed']:g} m/s" + (" on average, changing" if w.get("changing") else ""), transform=top.transAxes, color="#1f77b4", fontsize=9)
    wh = race.get("wind_hist")
    if w.get("changing") and wh is not None and len(wh):
        tw = prog.twinx()
        tw.plot(wh[:, 0], wh[:, 1], color="#1f77b4", lw=0.9, alpha=0.8)
        tw.plot(wh[:, 0], wh[:, 2] / 10.0, color="#17becf", lw=0.7, alpha=0.8, ls="--")
        tw.set_ylabel("wind at 10 m: speed [m/s] (solid), direction / 10 [deg] (dashed)", color="#1f77b4", fontsize=8)
    top.set_title(f"circus race, {race['laps']} laps, anticlockwise — winner: {race['winner'] or 'nobody finished'}   (✕ a fault: back to the last gate)")
    prog.set(xlabel="s", ylabel="laps completed", ylim=(0, race["laps"] + 0.3), title="laps against time (dotted: a fault)")
    prog.grid(alpha=0.3)
    txt.axis("off")
    lines = ["place  strategy   time     laps  best lap  faults  Wh"]
    for s in race["standings"]:
        lines.append(f"{s['place']:>3}    {s['strategy']:<9s} {(str(s['time_s']) + ' s') if s['finished'] else 'DNF':>8}  {s['laps']:>3}   {s['best_lap_s'] or 0:>6.2f}   {s['faults']:>3}   {s['energy_wh']:>5.1f}")
    lines += ["", "the strategies:"] + [f" {s['strategy']}: {s['what']}" for s in race["standings"]] + ["", "faults:"]
    lines += [f" {e['t']:5.1f}  {e['who']}: {e['what']} ({e['on']})" for e in race["faults"]] or [" none"]
    lines += ["", f"balloons blown out and back in: {race['balloon_respawns']}"]
    import textwrap
    wrapped = []
    for ln in lines:
        wrapped += textwrap.wrap(ln, 62, subsequent_indent="     ") or [""]
    txt.text(0, 1, "\n".join(wrapped[:60]), va="top", family="monospace", fontsize=7.6)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=110)
    return fig

