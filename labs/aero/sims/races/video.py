"""Videos of a run: from MuJoCo camera frames (``sim.render``, needs OpenGL) through ffmpeg, or — without a renderer
— the matplotlib replay (a plan view that follows the aircraft, a crude 'onboard' view looking along the velocity)
written straight to mp4 with ffmpeg. The chapter script makes them; the notebook shows them."""
from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

import numpy as np

from . import replay

QUALITY = {"screening": {"fps": 15, "size": (480, 270), "every": 8, "max_frames": 150, "seconds": 12.0},
           "fine": {"fps": 30, "size": (960, 540), "every": 2, "max_frames": 2000, "seconds": 1e9},
           "lap": {"fps": 30, "size": (960, 540), "every": None, "max_frames": None, "seconds": 1e9}}     # the whole lap in real time (frames by time)


def ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def frames_to_mp4(frames: list, out: Path, fps: int) -> Path | None:
    """PNG frames (paths, in order) → mp4 with ffmpeg; None when ffmpeg is missing."""
    exe = ffmpeg()
    if exe is None or not frames:
        return None
    out = Path(out)
    lst = out.with_suffix(".txt")
    lst.write_text("".join(f"file '{Path(f).resolve()}'\nduration {1 / fps:.5f}\n" for f in frames))
    subprocess.run([exe, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-vf", "format=yuv420p", "-r", str(fps), str(out)], check=True)
    lst.unlink(missing_ok=True)
    return out


def mujoco_video(course, result: dict, out: Path, *, camera: str, quality: str = "screening", work_dir: Path | None = None,
                 speed: float = 1.0, hud: bool = False, label: str = "", logo=None, settings: dict | None = None,
                 panel: bool = False) -> Path | None:
    """Render the run's trajectory from a scene camera (needs OpenGL: set MUJOCO_GL=egl or osmesa) and encode it;
    ``speed`` (presets without ``every``), ``hud``, ``label`` and ``logo`` go to ``sim.render``; ``panel`` adds the
    autopilot's side panel (``races.video_panel``) with the ``settings`` flown."""
    from . import sim
    q = QUALITY[quality]
    work = Path(work_dir or Path(out).parent / f"frames_{camera}")
    side = None
    if panel:
        from .video_panel import SidePanel
        side = SidePanel(course, result, settings, (int(0.8 * q["size"][1]), q["size"][1]), label=label)
    try:
        frames = sim.render(course, result, work, camera=camera, size=q["size"], every=q["every"], max_frames=q["max_frames"],
                            fps=q["fps"], speed=speed, hud=hud, label=label, panel=side, logo=logo)
    finally:
        if side is not None:
            side.close()
    mp4 = frames_to_mp4(frames, out, q["fps"])
    if mp4 is not None:
        shutil.rmtree(work, ignore_errors=True)
    return mp4


def matplotlib_video(course, result: dict, out: Path, *, camera: str = "chase", quality: str = "screening") -> Path:
    """The replay without a renderer: ``chase`` (a plan view following the aircraft), ``overview`` (the whole course),
    ``onboard`` (a plan view turned to look along the velocity, with the horizon of the next zones)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
    q = QUALITY[quality]
    camera = "onboard" if camera == "fpv" else camera
    H = result["history"]
    H = H[::max(q["every"] // 2, 1)]
    if len(H) > q["max_frames"]:
        H = H[np.linspace(0, len(H) - 1, q["max_frames"]).astype(int)]
    fig, ax = plt.subplots(figsize=(q["size"][0] / 96, q["size"][1] / 96), dpi=96)
    replay._patches(course, ax, 0, 1)
    P = course.points
    ax.plot(P[:, 0], P[:, 1], "k:", lw=0.6)
    trail, = ax.plot([], [], color="#111", lw=1.2)
    dot, = ax.plot([], [], "o", color="#d62828", ms=7)
    txt = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", fontsize=8, bbox=dict(facecolor="white", alpha=0.7, lw=0))
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    x0, x1, y0, y1 = course.extent
    zones_t = {t: text for t, text in result["events"] if text.startswith("zone")}
    win = 25.0 if camera == "chase" else 40.0

    def frame(i):
        x, y = H[i, 1], H[i, 2]
        trail.set_data(H[:i + 1, 1], H[:i + 1, 2]); dot.set_data([x], [y])
        if camera == "overview":
            ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        elif camera == "onboard":
            vx, vy = H[i, 4], H[i, 5]
            ang = math.atan2(vy, vx) if math.hypot(vx, vy) > 0.5 else 0.0
            cx, cy = x + 0.35 * win * math.cos(ang), y + 0.35 * win * math.sin(ang)
            ax.set_xlim(cx - win / 2, cx + win / 2); ax.set_ylim(cy - win / 2, cy + win / 2)
        else:
            ax.set_xlim(x - win / 2, x + win / 2); ax.set_ylim(y - win / 2, y + win / 2)
        passed = sum(1 for t in zones_t if t <= H[i, 0])
        txt.set_text(f"{course.name} — {camera}\nt = {H[i, 0]:5.1f} s   v = {H[i, 9]:4.1f} m/s   alt {H[i, 3]:4.1f} m   zones {passed} / {len(course.zones) - 1}")
        return trail, dot, txt
    anim = FuncAnimation(fig, frame, frames=len(H), blit=False)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if ffmpeg() and out.suffix == ".mp4":
        anim.save(str(out), writer=FFMpegWriter(fps=q["fps"], bitrate=1200))
    else:
        out = out.with_suffix(".gif")
        anim.save(str(out), writer=PillowWriter(fps=q["fps"]))
    plt.close(fig)
    return out
