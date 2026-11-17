"""A software renderer for the gorilla sims: Belisario's CAD (``designs/belisario_mesh``: the bones, the modules,
the bays, the plates, the struts between their anchors) on the MuJoCo bodies, the scene's props from their
geoms, the ground - a perspective camera, flat shading, the painter's order, drawn with PIL. MuJoCo's own renderer
needs a GL this machine does not have; this one needs nothing but numpy and PIL and draws a frame in well under a
second.

    r = Renderer(model, mesh)                      # mesh = belisario_mesh.load("plastic")
    img = r.render(data, Camera(eye, target), title="...")   # a PIL image
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

__all__ = ["Camera", "Renderer", "orbit"]

GROUND = (0.60, 0.58, 0.53)
SKY = (0.93, 0.94, 0.96)
NEAR = 0.15


@dataclass
class Camera:
    eye: tuple
    target: tuple
    up: tuple = (0.0, 0.0, 1.0)
    fov_deg: float = 32.0


def orbit(target, elev_deg, azim_deg, distance):
    """A camera ``distance`` from ``target`` at matplotlib's elevation and azimuth."""
    e, a = math.radians(elev_deg), math.radians(azim_deg)
    d = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    return Camera(tuple(np.asarray(target) + distance * d), tuple(target))


# ---- primitives as triangles (unit shapes in the geom's frame)
def _box(half):
    hx, hy, hz = half
    v = np.array([[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return np.array([[v[a], v[b], v[c]] for a, b, c, d in faces for a, b, c in ((a, b, c), (a, c, d))])


def _cylinder(r, hl, n=12, caps=True):
    ang = np.linspace(0, 2 * math.pi, n, endpoint=False)
    ring = np.stack([r * np.cos(ang), r * np.sin(ang)], axis=1)
    top, bot = np.c_[ring, np.full(n, hl)], np.c_[ring, np.full(n, -hl)]
    tris = []
    for k in range(n):
        j = (k + 1) % n
        tris += [[bot[k], bot[j], top[j]], [bot[k], top[j], top[k]]]
        if caps:
            tris += [[top[k], top[j], [0, 0, hl]], [bot[j], bot[k], [0, 0, -hl]]]
    return np.array(tris)


def _sphere(r, n=10, m=6):
    tris = []
    for i in range(m):
        t0, t1 = math.pi * i / m, math.pi * (i + 1) / m
        for k in range(n):
            p0, p1 = 2 * math.pi * k / n, 2 * math.pi * (k + 1) / n
            a = [r * math.sin(t0) * math.cos(p0), r * math.sin(t0) * math.sin(p0), r * math.cos(t0)]
            b = [r * math.sin(t0) * math.cos(p1), r * math.sin(t0) * math.sin(p1), r * math.cos(t0)]
            c = [r * math.sin(t1) * math.cos(p1), r * math.sin(t1) * math.sin(p1), r * math.cos(t1)]
            d = [r * math.sin(t1) * math.cos(p0), r * math.sin(t1) * math.sin(p0), r * math.cos(t1)]
            if i > 0:
                tris.append([a, b, c])
            if i < m - 1:
                tris.append([a, c, d])
    return np.array(tris)


def _capsule(r, hl):
    cyl = _cylinder(r, hl, caps=False)
    s = _sphere(r)
    top = s[s[:, :, 2].mean(axis=1) >= 0] + [0, 0, hl]
    bot = s[s[:, :, 2].mean(axis=1) < 0] - [0, 0, hl]
    return np.concatenate([cyl, top, bot])


def _between(a, b, r, n=10):
    """A cylinder of radius ``r`` from ``a`` to ``b`` (world triangles)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d = b - a
    L = np.linalg.norm(d)
    if L < 1e-6:
        return np.zeros((0, 3, 3))
    z = d / L
    x = np.cross(z, [0, 0, 1.0]) if abs(z[2]) < 0.9 else np.cross(z, [1.0, 0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    return _cylinder(r, L / 2, n) @ R.T + (a + b) / 2


class Renderer:
    def __init__(self, model, mesh, width=960, height=540, light=(-0.35, 0.5, 0.8), ground=GROUND, sky=SKY):
        self.m, self.mesh, self.w, self.h = model, mesh, width, height
        self.light = np.asarray(light, dtype=float) / np.linalg.norm(light)
        self.ground, self.sky = ground, sky
        self.body = {}
        for name in mesh.bodies():
            self.body[name] = model.body(name).id
        root = model.body_rootid[model.body("pelvis").id]
        self.props = []                                                   # (geom id, unit triangles, rgba): everything that is not the robot or the terrain
        for g in range(model.ngeom):
            if model.body_rootid[model.geom_bodyid[g]] == root or model.geom(g).name == "chiron_terrain":
                continue
            t, s = int(model.geom_type[g]), model.geom_size[g]
            if t == mujoco.mjtGeom.mjGEOM_BOX:
                tris = _box(s[:3])
            elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
                tris = _capsule(s[0], s[1])
            elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
                tris = _cylinder(s[0], s[1])
            elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
                tris = _sphere(s[0])
            else:
                continue
            rgba = tuple(model.geom_rgba[g])
            self.props.append((g, tris, rgba if rgba[3] > 0 else (0.5, 0.5, 0.5, 1.0)))
        try:
            self.font = ImageFont.load_default(size=17)
        except TypeError:
            self.font = ImageFont.load_default()

    # ---- the world's triangles this frame
    def triangles(self, data):
        tris, cols = [], []
        xpos, xmat = data.xpos, data.xmat
        for body, local, rgba in self.mesh.pieces:
            b = self.body[body]
            tris.append(local @ xmat[b].reshape(3, 3).T + xpos[b])
            cols.append(np.repeat([rgba[:3]], len(local), axis=0))
        from belisario_mesh import STRUT
        for ba, la, bb, lb, rb, rr in self.mesh.struts:
            a = xpos[self.body[ba]] + xmat[self.body[ba]].reshape(3, 3) @ la
            b = xpos[self.body[bb]] + xmat[self.body[bb]].reshape(3, 3) @ lb
            barrel = _between(a, a + 0.55 * (b - a), rb)
            rod = _between(a + 0.5 * (b - a), b, rr)
            tris += [barrel, rod]
            cols += [np.repeat([STRUT[:3]], len(barrel), axis=0), np.repeat([(0.6, 0.6, 0.62)], len(rod), axis=0)]
        for g, unit, rgba in self.props:
            tris.append(unit @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])
            cols.append(np.repeat([rgba[:3]], len(unit), axis=0))
        return np.concatenate(tris), np.concatenate(cols)

    # ---- projection
    def _basis(self, cam):
        eye, target = np.asarray(cam.eye, dtype=float), np.asarray(cam.target, dtype=float)
        f = target - eye
        f /= np.linalg.norm(f)
        r = np.cross(f, np.asarray(cam.up, dtype=float))
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        F = (self.h / 2) / math.tan(math.radians(cam.fov_deg) / 2)
        return eye, np.stack([r, u, f], axis=1), F

    def _project(self, pts, eye, B, F):
        c = (pts - eye) @ B                                               # camera coordinates: x right, y up, z forward
        z = np.maximum(c[..., 2], 1e-6)
        return np.stack([self.w / 2 + F * c[..., 0] / z, self.h / 2 - F * c[..., 1] / z], axis=-1), c[..., 2]

    def _draw(self, draw, tris, cols, eye, B, F, sort=True):
        if len(tris) == 0:
            return
        xy, z = self._project(tris, eye, B, F)
        ok = np.all(z > NEAR, axis=1) & np.any(np.all(np.abs(xy - [self.w / 2, self.h / 2]) < [self.w, self.h], axis=2), axis=1)
        tris, cols, xy, z = tris[ok], cols[ok], xy[ok], z[ok]
        if len(tris) == 0:
            return
        n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        shade = 0.42 + 0.58 * np.abs(n @ self.light)
        rgb = np.clip(cols * shade[:, None] * 255, 0, 255).astype(int)
        order = np.argsort(-z.mean(axis=1)) if sort else np.arange(len(tris))
        for k in order:
            draw.polygon([tuple(q) for q in xy[k]], fill=tuple(rgb[k]))

    def render(self, data, cam: Camera, title: str | None = None, ground_extent=14.0):
        img = Image.new("RGB", (self.w, self.h), tuple(int(c * 255) for c in self.sky))
        draw = ImageDraw.Draw(img)
        eye, B, F = self._basis(cam)
        # the ground: 1 m tiles round the target, drawn first (everything stands on it)
        t = np.asarray(cam.target, dtype=float)
        xs = np.arange(math.floor(t[0] - ground_extent), math.ceil(t[0] + ground_extent) + 1, 1.0)
        ys = np.arange(math.floor(t[1] - ground_extent), math.ceil(t[1] + ground_extent) + 1, 1.0)
        tiles = []
        for x in xs[:-1]:
            for y in ys[:-1]:
                tiles += [[[x, y, 0], [x + 1, y, 0], [x + 1, y + 1, 0]], [[x, y, 0], [x + 1, y + 1, 0], [x, y + 1, 0]]]
        tiles = np.array(tiles, dtype=float)
        checker = ((np.floor(tiles[:, 0, 0] + 0.5) + np.floor(tiles[:, 0, 1] + 0.5)) % 2).astype(bool)
        g = np.array(self.ground)
        cols = np.where(checker[:, None], g * 0.96, g)
        self._draw(draw, tiles, cols, eye, B, F)
        tris, cols = self.triangles(data)
        self._draw(draw, tris, cols, eye, B, F)
        if title:
            draw.text((12, 10), title, fill=(35, 35, 40), font=self.font)
        return img
