"""Overview maps, traces and short demonstration replays of a run (mock or MuJoCo): the course's obstacles and zones
in plan and elevation with the flown path, the speed and power traces, the gate split table, and an animated replay
(a GIF from the trajectory: a chase-style plan view that follows the aircraft) that needs no renderer. Rendered
camera frames from MuJoCo are ``races.sim.render`` (manual, needs OpenGL)."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

COL = {"start": "#2a9d2a", "gate": "#3377dd", "checkpoint": "#f29d1c", "finish": "#d62828"}


def _patches(course, ax, a, b):
    from matplotlib.patches import Circle, Rectangle
    for o in course.obstacles:
        c, h = np.asarray(o.pos), o.half_extents()
        face = tuple(o.rgba[:3])
        if o.kind == "sphere" or (o.kind == "cylinder" and (a, b) == (0, 1)):
            ax.add_patch(Circle((c[a], c[b]), o.size[0], facecolor=face, edgecolor="#555", lw=0.4, alpha=0.85))
        else:
            ax.add_patch(Rectangle((c[a] - h[a], c[b] - h[b]), 2 * h[a], 2 * h[b], facecolor=face, edgecolor="#555", lw=0.4, alpha=0.85))
        if o.landmark:
            ax.annotate(o.name.replace("_", " "), (c[a], c[b]), fontsize=7, color="#333", ha="center", va="bottom", xytext=(0, 4), textcoords="offset points")
    for i, z in enumerate(course.zones):
        c, h = np.asarray(z.center), np.asarray(z.half)
        ax.add_patch(Rectangle((c[a] - h[a], c[b] - h[b]), 2 * h[a], 2 * h[b], facecolor=COL[z.kind], edgecolor=COL[z.kind], alpha=0.35, lw=1.0))
        ax.annotate(str(i), (c[a], c[b]), fontsize=7, ha="center", va="center", color="k")
    for g in course.start_grid:
        ax.plot(g[a], g[b], "s", color="#444", ms=4)


def overview(course, result=None, *, ax=None, view="plan", title=None):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(13, 6))
    a, b = (0, 1) if view == "plan" else (0, 2)
    _patches(course, ax, a, b)
    P = course.points
    ax.plot(P[:, a], P[:, b], "k:", lw=0.8, label="racing line")
    if result is not None:
        H = result["history"]
        ax.plot(H[:, 1 + a], H[:, 1 + b], color="#111", lw=1.4, label=f"flown ({result['stats']['mode']}): {result['stats']['status']}" + (f", {result['stats']['lap time [s]']:.1f} s" if result["stats"]["status"] == "finished" else ""))
        for t, pos in result["referee"].collisions:
            ax.plot(pos[a], pos[b], "rx", ms=9, mew=2)
    ax.set_aspect("equal"); ax.grid(alpha=0.2); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]" if view == "plan" else "z [m]")
    x0, x1, y0, y1 = course.extent
    if view == "plan":
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    else:
        ax.set_xlim(x0, x1); ax.set_ylim(-1, max(o.pos[2] + o.half_extents()[2] for o in course.obstacles) + 4)
    ax.set_title(title or f"{course.name}: {view}")
    ax.legend(fontsize=8, loc="upper right")
    return ax


def traces(result, ax=None):
    import matplotlib.pyplot as plt
    H = result["history"]
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(H[:, 0], H[:, 9], color="#111", label="speed [m/s]")
    ax2 = ax.twinx()
    ax2.plot(H[:, 0], H[:, 8], color="#d62828", lw=0.8, alpha=0.8, label="electrical [W]")
    for t, text in result["events"]:
        if text.startswith("zone") or text.startswith("collision") or text.startswith("reset") or text.startswith("finish"):
            ax.axvline(t, color="#3377dd" if text.startswith("zone") else "#d62828", lw=0.5, alpha=0.6)
    ax.set_xlabel("t [s]"); ax.set_ylabel("speed [m/s]"); ax2.set_ylabel("W"); ax.grid(alpha=0.2)
    ax.set_title(f"{result['stats']['course']}: speed and power ({result['stats']['mode']})")
    return ax


def splits(result):
    import pandas as pd
    gt = result["gate times"]
    rows = []
    prev = 0.0
    for name, tt in gt.items():
        rows.append({"zone": name, "time [s]": tt, "split [s]": tt - prev}); prev = tt
    return pd.DataFrame(rows).set_index("zone")


def animate(course, result, path, *, fps: int = 20, every: int = 3, window: float = 25.0, max_frames: int = 300):
    """A short replay as a GIF: a plan view that follows the aircraft (a chase camera in plan), with the clock,
    the speed and the next zone."""
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    H = result["history"][::every][:max_frames]
    fig, ax = plt.subplots(figsize=(5, 5), dpi=72)
    _patches(course, ax, 0, 1)
    P = course.points
    ax.plot(P[:, 0], P[:, 1], "k:", lw=0.6)
    trail, = ax.plot([], [], color="#111", lw=1.2)
    dot, = ax.plot([], [], "o", color="#d62828", ms=7)
    txt = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", fontsize=9, bbox=dict(facecolor="white", alpha=0.7, lw=0))
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    zones_t = {t: text for t, text in result["events"] if text.startswith("zone")}

    def frame(i):
        x, y = H[i, 1], H[i, 2]
        trail.set_data(H[:i + 1, 1], H[:i + 1, 2]); dot.set_data([x], [y])
        ax.set_xlim(x - window / 2, x + window / 2); ax.set_ylim(y - window / 2, y + window / 2)
        passed = sum(1 for t in zones_t if t <= H[i, 0])
        txt.set_text(f"{course.name}\nt = {H[i, 0]:5.1f} s   v = {H[i, 9]:4.1f} m/s   alt {H[i, 3]:4.1f} m\nzones {passed} / {len(course.zones) - 1}")
        return trail, dot, txt
    anim = FuncAnimation(fig, frame, frames=len(H), blit=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(path), writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path
