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
        k = max(0.62, size[0] / W)
        self.title_font, self.font, self.small = _font(round(26 * k)), _font(round(17 * k)), _font(round(14 * k))

    def camera_pos(self):
        c = self.cam
        az, el = math.radians(c.azimuth), math.radians(c.elevation)
        fwd = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        return np.array(c.lookat) - c.distance * fwd

    def _wrap(self, d, text, font, width):
        """Split ``text`` at spaces so each piece fits ``width`` pixels."""
        out, cur = [], ""
        for w in text.split("  "):                               # the HUD's fields are separated by double spaces
            cand = (cur + "  " + w) if cur else w
            if d.textlength(cand, font=font) <= width or not cur:
                cur = cand
            else:
                out.append(cur)
                cur = w.strip()
        return out + ([cur] if cur else [])

    def _chart(self, d, chart, box):
        """A strip chart of the series in ``chart`` (``t`` and named series, m) in ``box`` (x0, y0, x1, y1)."""
        x0, y0, x1, y1 = box
        d.rectangle(box, fill=(0, 0, 0, 130))
        t = chart["t"]
        if len(t) < 2:
            return
        names = [k for k in chart if k != "t"]
        span = max(1.0, max(float(np.max(np.abs(chart[k]))) for k in names))
        ym = (y0 + y1) / 2
        sc = (y1 - y0 - 24) / 2 / span
        d.line([x0 + 4, ym, x1 - 4, ym], fill=(120, 120, 120, 255))
        cols = [(170, 180, 190, 255), (255, 215, 60, 255), (120, 220, 255, 255)]
        tt = t[-1] - 10.0
        for k, col in zip(names, cols):
            pts = [(x0 + 4 + (x1 - x0 - 8) * (ti - tt) / 10.0, ym - sc * float(v)) for ti, v in zip(t, chart[k])]
            d.line(pts, fill=col, width=2)
        y = y0 + 3
        for k, col in zip(names, cols):
            d.text((x0 + 6, y), k, font=self.small, fill=col)
            y += 15
        d.text((x1 - 70, y0 + 3), f"+/-{span:.1f} m", font=self.small, fill=(200, 200, 200, 255))
        d.text((x1 - 50, y1 - 17), "10 s", font=self.small, fill=(200, 200, 200, 255))

    def render(self, lookat, distance, azimuth, elevation, title="", lines=(), footer="", clock="", chart=None):
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
        cw = d.textlength(clock, font=self.title_font) if clock else 0
        tl = self._wrap(d, title.replace(": ", ":  "), self.title_font, im.width - cw - 40)
        ll = [p for s in lines for p in self._wrap(d, s, self.font, im.width - 28)]
        lh = self.font.size + 7
        hh = 14 + (self.title_font.size + 8) * len(tl) + lh * len(ll)
        d.rectangle([0, 0, im.width, hh], fill=(0, 0, 0, 120))
        if clock:
            d.text((im.width - cw - 14, y), clock, font=self.title_font, fill=(255, 230, 120, 255))
        for s in tl:
            d.text((14, y), s, font=self.title_font, fill=(255, 255, 255, 255))
            y += self.title_font.size + 8
        for s in ll:
            d.text((14, y), s, font=self.font, fill=(235, 240, 245, 255))
            y += lh
        if chart is not None:
            cw2, ch2 = min(380, im.width - 28), 140
            self._chart(d, chart, (im.width - cw2 - 14, im.height - ch2 - 40, im.width - 14, im.height - 40))
        if footer:
            fl = self._wrap(d, footer.replace(" - ", "  "), self.small, im.width - 28)
            fh = (self.small.size + 6) * len(fl) + 6
            d.rectangle([0, im.height - fh, im.width, im.height], fill=(0, 0, 0, 110))
            for k2, s in enumerate(fl):
                d.text((14, im.height - fh + 4 + k2 * (self.small.size + 6)), s, font=self.small, fill=(220, 225, 230, 255))
        if under:
            d.text((im.width - 150, im.height - 52), "under water", font=self.font, fill=(170, 220, 240, 255))
        return np.asarray(im)
