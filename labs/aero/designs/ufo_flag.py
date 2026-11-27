"""The alien Jolly Roger: the owner's flag for the UFOs (``ufo.py``) - a white alien head with black almond eyes over
crossed bones, on black, 3:2. Redrawn as simple shapes from the owner's image (1536 x 1024 px) so the texture is
reproducible from the repository; the shapes are in the image's pixels (x right, y down). The same shapes give the
flag's outline for anything that needs geometry, not a texture.

    python ufo_flag.py [OUT.png]        # writes data/alien_jolly_roger.png by default
"""
from __future__ import annotations

import math
import os

W, H = 1536, 1024
HEAD_CX, HEAD_TOP, HEAD_WIDE_Y, HEAD_HALF_W, CHIN_Y = 768.0, 68.0, 300.0, 250.0, 668.0


def _half_width(y: float) -> float:
    """The head's half-width below its widest line: (1 - s^1.3)^0.6 of the widest, s from the widest line (0) to
    the chin (1) - an exponent under 1 makes the chin round, not a point."""
    s = min(max((y - HEAD_WIDE_Y) / (CHIN_Y - HEAD_WIDE_Y), 0.0), 1.0)
    return HEAD_HALF_W * (1 - s ** 1.3) ** 0.6


def head(n: int = 120) -> list:
    """The head's outline: an ellipse above its widest line, tapering to the chin below it."""
    pts = []
    ry = HEAD_WIDE_Y - HEAD_TOP
    for i in range(n + 1):                                    # the crown, right to left over the top
        a = math.pi * i / n
        pts.append((HEAD_CX + HEAD_HALF_W * math.cos(a), HEAD_WIDE_Y - ry * math.sin(a)))
    for i in range(1, n + 1):                                 # down the left side to the chin, then up the right
        s = i / n
        y = HEAD_WIDE_Y + s * (CHIN_Y - HEAD_WIDE_Y)
        w = _half_width(y)
        pts.append((HEAD_CX - w, y))
    for i in range(n, 0, -1):
        s = i / n
        y = HEAD_WIDE_Y + s * (CHIN_Y - HEAD_WIDE_Y)
        w = _half_width(y)
        pts.append((HEAD_CX + w, y))
    return pts


def eye(side: int, n: int = 80) -> list:
    """An almond eye (side -1 left, +1 right): an ellipse slanting down toward the nose."""
    cx, cy, a, b, tilt = HEAD_CX + side * 120.0, 402.0, 122.0, 72.0, side * -math.radians(42)
    out = []
    for i in range(n):
        t = 2 * math.pi * i / n
        x, y = a * math.cos(t), b * math.sin(t) * (1.0 + 0.25 * math.cos(t) * side * -1)   # fuller at the outer end
        out.append((cx + x * math.cos(tilt) - y * math.sin(tilt), cy + x * math.sin(tilt) + y * math.cos(tilt)))
    return out


def bones() -> list:
    """The two crossed bones: each a shaft (a polygon) and a pair of knuckles (circles) at each end.
    Returns [(shaft_polygon, [(cx, cy, r), ...]), ...]."""
    out = []
    for (x0, y0), (x1, y1) in (((490, 640), (1050, 900)), ((1050, 640), (490, 900))):
        dx, dy = x1 - x0, y1 - y0
        L = math.hypot(dx, dy)
        ux, uy = dx / L, dy / L
        nx, ny = -uy, ux
        hw = 19.0
        shaft = [(x0 + nx * hw, y0 + ny * hw), (x1 + nx * hw, y1 + ny * hw), (x1 - nx * hw, y1 - ny * hw), (x0 - nx * hw, y0 - ny * hw)]
        knobs = []
        for (x, y), sgn in (((x0, y0), -1), ((x1, y1), 1)):
            for k in (-1, 1):
                knobs.append((x + sgn * ux * 8 + nx * k * 30, y + sgn * uy * 8 + ny * k * 30, 36.0))
        out.append((shaft, knobs))
    return out


def render(path: str, scale: int = 2) -> str:
    """Draw the flag (supersampled ``scale`` times, then reduced: smooth edges) to a PNG."""
    from PIL import Image, ImageDraw
    S = scale
    im = Image.new("RGB", (W * S, H * S), (0, 0, 0))
    d = ImageDraw.Draw(im)
    sc = lambda pts: [(x * S, y * S) for x, y in pts]
    for shaft, knobs in bones():
        d.polygon(sc(shaft), fill=(255, 255, 255))
        for x, y, r in knobs:
            d.ellipse([(x - r) * S, (y - r) * S, (x + r) * S, (y + r) * S], fill=(255, 255, 255))
    d.polygon(sc(head()), fill=(255, 255, 255))
    for s in (-1, 1):
        d.polygon(sc(eye(s)), fill=(0, 0, 0))
    im = im.resize((W, H), Image.LANCZOS)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    im.save(path, optimize=True)
    return path


DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "alien_jolly_roger.png")

if __name__ == "__main__":
    import sys
    print(render(sys.argv[1] if len(sys.argv) > 1 else DEFAULT))
