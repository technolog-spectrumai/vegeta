"""Rendering: MuJoCo's offscreen renderer (``MUJOCO_GL=egl``, else ``osmesa``), a free camera the clip steers, a
text overlay (PIL) and H.264 through ffmpeg (``yuv420p``, so every player opens it)."""
from __future__ import annotations

import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

W, H, FPS = 1280, 720, 30


def gl_backend() -> str:
    """Choose the OpenGL backend before mujoco is imported: EGL if it works here, else OSMesa."""
    if "MUJOCO_GL" in os.environ:
        os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
        return os.environ["MUJOCO_GL"]
    import sys
    for gl in ("egl", "osmesa"):
        r = subprocess.run([sys.executable, "-c", "import mujoco; m = mujoco.MjModel.from_xml_string('<mujoco/>'); mujoco.Renderer(m, 8, 8)"],
                           env={**os.environ, "MUJOCO_GL": gl, "PYOPENGL_PLATFORM": gl}, capture_output=True)
        if r.returncode == 0:
            os.environ["MUJOCO_GL"] = os.environ["PYOPENGL_PLATFORM"] = gl
            return gl
    raise RuntimeError("no OpenGL backend for MuJoCo's offscreen renderer (install libegl1 or libosmesa6)")


def _font(size):
    from PIL import ImageFont
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


class Encoder:
    """Raw RGB frames into ffmpeg's stdin -> H.264 mp4."""

    def __init__(self, out: Path, fps: int = FPS, size=(W, H), crf: int = 21):
        exe = shutil.which("ffmpeg")
        if exe is None:
            raise RuntimeError("ffmpeg not found")
        self.out = Path(out)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.n = 0
        self.p = subprocess.Popen([exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}",
                                   "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
                                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.out)], stdin=subprocess.PIPE)

    def write(self, img: np.ndarray):
        self.p.stdin.write(np.ascontiguousarray(img, dtype=np.uint8).tobytes())
        self.n += 1

    def close(self):
        self.p.stdin.close()
        if self.p.wait() != 0:
            raise RuntimeError(f"ffmpeg failed on {self.out}")
        return self.out


class Smooth:
    """A first-order lag on a vector (the camera's look-at), ``tau`` seconds."""

    def __init__(self, tau: float):
        self.tau, self.x, self.t = tau, None, None

    def __call__(self, t, x):
        x = np.asarray(x, dtype=float)
        if self.x is None or self.tau <= 0:
            self.x = x.copy()
        else:
            self.x += (x - self.x) * (1 - math.exp(-max(t - self.t, 0.0) / self.tau))
        self.t = t
        return self.x.copy()


class Frames:
    """Render the simulation's current state from a free camera with an overlay."""

    def __init__(self, sim, size=(W, H)):
        import mujoco
        self.mj, self.sim = mujoco, sim
        self.r = mujoco.Renderer(sim.model, size[1], size[0])
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = 0
        self.opt.geomgroup[0] = self.opt.geomgroup[1] = 1     # the drawn geoms; group 3 (the collision shapes) hidden
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = 1
        self.title_font, self.font, self.small = _font(26), _font(17), _font(14)

    def camera_pos(self):
        c = self.cam
        az, el = math.radians(c.azimuth), math.radians(c.elevation)
        fwd = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        return np.array(c.lookat) - c.distance * fwd

    def render(self, lookat, distance, azimuth, elevation, title="", lines=(), footer="", clock=""):
        from PIL import Image, ImageDraw
        sim = self.sim
        c = self.cam
        c.lookat[:] = lookat
        c.distance, c.azimuth, c.elevation = distance, azimuth, elevation
        under = self.camera_pos()[2] < sim.sea.eta(sim.t) - 0.05
        con = self.r._mjr_context                                 # under water: a short blue fog (metres)
        con.fogStart, con.fogEnd = (0.5, 16.0) if under else (1e4, 2e4)
        con.fogRGBA[:] = (0.04, 0.22, 0.28, 1.0)
        self.r.update_scene(sim.data, camera=c, scene_option=self.opt)
        self.r.scene.flags[self.mj.mjtRndFlag.mjRND_FOG] = 1 if under else 0
        img = self.r.render()
        im = Image.fromarray(img)
        if under:
            tint = Image.new("RGB", im.size, (10, 60, 80))
            im = Image.blend(im, tint, 0.25)
        d = ImageDraw.Draw(im, "RGBA")
        y = 10
        hh = 40 + 24 * len(lines)
        d.rectangle([0, 0, im.width, hh], fill=(0, 0, 0, 120))
        d.text((14, y), title, font=self.title_font, fill=(255, 255, 255, 255))
        if clock:
            w = d.textlength(clock, font=self.title_font)
            d.text((im.width - w - 14, y), clock, font=self.title_font, fill=(255, 230, 120, 255))
        y += 36
        for s in lines:
            d.text((14, y), s, font=self.font, fill=(235, 240, 245, 255))
            y += 24
        if footer:
            d.rectangle([0, im.height - 26, im.width, im.height], fill=(0, 0, 0, 110))
            d.text((14, im.height - 22), footer, font=self.small, fill=(220, 225, 230, 255))
        if under:
            d.text((im.width - 150, im.height - 52), "under water", font=self.font, fill=(170, 220, 240, 255))
        return np.asarray(im)
