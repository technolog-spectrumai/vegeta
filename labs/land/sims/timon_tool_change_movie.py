#!/usr/bin/env python3
"""Timon changes Pumba's tool — the movie (``designs/timon_tool_change.py``): a kinematic animation of both
machines' Chiron models (joint trajectories between IK solutions, MuJoCo forward kinematics; not a dynamics run).

    xvfb-run -a python3 labs/land/sims/timon_tool_change_movie.py
    xvfb-run -a python3 labs/land/sims/timon_tool_change_movie.py --stills 4,15,22,30,38    # PNGs at those times

Writes ``sims/output/timon_tool_change.mp4`` (H.264).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "labs" / "land" / "designs"))

import timon_tool_change as ttc  # noqa: E402

OUT = ROOT / "labs" / "land" / "sims" / "output"


def render(frames, size=(1280, 720), stills=None):
    import pyvista as pv
    from vegeta.chiron import viz

    pv.OFF_SCREEN = True
    pl = pv.Plotter(off_screen=True, window_size=list(size))
    pl.set_background("#dfe9ef")
    pl.add_mesh(pv.Plane(center=(0.5, 1.3, 0.0), direction=(0, 0, 1), i_size=9, j_size=9), color="#8d8a78")
    actors = []
    for kind, s, rgba, _, _ in frames[0][2]:
        mesh = viz._geom_mesh(pv, kind, s)
        col = rgba[:3] if not np.allclose(rgba, [0.5, 0.5, 0.5, 1.0]) else (0.6, 0.6, 0.6)
        actors.append(pl.add_mesh(mesh, color=col, smooth_shading=True, opacity=float(rgba[3]) or 1.0))
    pl.camera_position = [(3.6, -1.6, 2.4), (0.35, 1.35, 0.55), (0, 0, 1)]
    images, still_imgs = [], {}
    for t, label, geoms in frames:
        for a, (_, _, _, pos, mat) in zip(actors, geoms):
            M = np.eye(4)
            M[:3, :3], M[:3, 3] = mat, pos
            a.user_matrix = M
        pl.add_text(f"{label}", position="upper_left", font_size=12, color="black", name="label")
        pl.add_text(f"t = {t:4.1f} s   (kinematic animation: Timon services Pumba)", position="lower_left",
                    font_size=8, color="#333333", name="time")
        pl.render()
        img = np.asarray(pl.screenshot(return_img=True))[..., :3].copy()
        images.append(img)
        if stills:
            for s_t in stills:
                if abs(t - s_t) < 1e-6 or (s_t not in still_imgs and t >= s_t):
                    still_imgs.setdefault(s_t, img)
    pl.close()
    return images, still_imgs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--stills", default="", help="comma-separated times [s]: write those frames as PNGs only")
    a = ap.parse_args(argv)
    scene = ttc.Scene()
    OUT.mkdir(parents=True, exist_ok=True)
    import cv2
    if a.stills:
        ts = [float(x) for x in a.stills.split(",")]
        fr = [f for f in scene.frames(fps=a.fps) if any(abs(f[0] - x) < 0.5 / a.fps for x in ts)]
        _, st = render(fr, stills=ts)
        for t, img in st.items():
            cv2.imwrite(str(OUT / f"timon_tool_change_{t:04.1f}.png"), img[..., ::-1])
        print("stills:", sorted(st))
        return 0
    images, _ = render(scene.frames(fps=a.fps))
    raw = OUT / "timon_tool_change_raw.mp4"
    from vegeta.chiron import viz
    viz.to_video(images, raw, fps=a.fps)
    out = OUT / "timon_tool_change.mp4"
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22",
                        "-movflags", "+faststart", str(out)], check=True)
        raw.unlink()
    else:
        raw.rename(out)
    print(f"wrote {out} ({len(images)} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
