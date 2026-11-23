"""SVG in millimetres, one group per operation (``cut`` red, ``score`` blue, ``engrave`` black unless the operation
says otherwise), for xTool Creative Space, LightBurn and the like. The sheet's front-left corner is the SVG's
bottom-left corner (SVG y runs down, the sheet's y runs away from the operator)."""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path as FsPath

import numpy as np

from .geometry import Part, Path
from .settings import Operation

DEFAULT_STROKES = {"cut": "#ff0000", "score": "#0000ff", "engrave": "#000000"}
ORDER = {"engrave": 0, "score": 1, "cut": 2}


def _d(path: Path, height: float) -> str:
    pts = path.points
    s = " ".join(f"{'M' if i == 0 else 'L'}{x:.3f},{height - y:.3f}" for i, (x, y) in enumerate(pts))
    return s + (" Z" if path.closed else "")


def write_svg(path: str | FsPath, parts: list[Part], width_mm: float, height_mm: float,
              operations: dict[str, Operation] | None = None, labels: bool = False, title: str = "") -> FsPath:
    """Write ``parts`` (already positioned, sheet coordinates) on a ``width_mm`` x ``height_mm`` page."""
    path = FsPath(path)
    ops = operations or {}
    present = sorted({p.op for part in parts for p in part.paths()} | ({"engrave"} if labels else set()),
                     key=lambda o: (ops[o].order if o in ops else ORDER.get(o, 9), o))
    lines = [f'<?xml version="1.0" encoding="UTF-8"?>',
             f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_mm:g}mm" height="{height_mm:g}mm" '
             f'viewBox="0 0 {width_mm:g} {height_mm:g}">',
             f'  <title>{_esc(title or path.stem)}</title>']
    for op in present:
        stroke = ops[op].stroke if op in ops else DEFAULT_STROKES.get(op, "#808080")
        lines.append(f'  <g id="{op}" fill="none" stroke="{stroke}" stroke-width="0.1">')
        for part in parts:
            for p in part.paths():
                if p.op == op:
                    lines.append(f'    <path d="{_d(p, height_mm)}" data-part="{_esc(part.name)}"/>')
            if labels and op == "engrave":
                cx, cy = part.outer.centroid()
                lines.append(f'    <text x="{cx:.3f}" y="{height_mm - cy:.3f}" font-size="4" text-anchor="middle" '
                             f'dominant-baseline="middle" fill="{stroke}" stroke="none">{_esc(part.name)}</text>')
        lines.append("  </g>")
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def write_parts_svg(path: str | FsPath, parts: list[Part], gap_mm: float = 5.0, operations=None, labels=False) -> FsPath:
    """Loose parts in a row (no sheet): for a single rib or a quick look."""
    placed, x = [], gap_mm
    h_max = 0.0
    for part in parts:
        p = part.at_origin().transformed(0.0, x, gap_mm)
        w, h = p.size()
        placed.append(p)
        x += w + gap_mm
        h_max = max(h_max, h)
    return write_svg(path, placed, x, h_max + 2 * gap_mm, operations, labels)


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_TOKEN = re.compile(rf"([MLZmlz])|({_NUM})")


def read_svg_paths(path: str | FsPath) -> dict:
    """Read an SVG written by Atropos (absolute M/L/Z paths in groups named by operation): page size, the paths with
    their operation and points (sheet coordinates, y up), and the length per operation."""
    root = ET.parse(path).getroot()
    ns = {"svg": "http://www.w3.org/2000/svg"}
    w = float(re.sub(r"[a-z%]+$", "", root.get("width", "0")))
    h = float(re.sub(r"[a-z%]+$", "", root.get("height", "0")))
    paths, groups = [], {}
    for g in root.findall("svg:g", ns):
        op = g.get("id", "cut")
        n = 0
        for el in g.findall("svg:path", ns):
            pts, closed = [], False
            cmd = None
            nums = []
            for m in _TOKEN.finditer(el.get("d", "")):
                if m.group(1):
                    if m.group(1) in "Zz":
                        closed = True
                    cmd = m.group(1)
                else:
                    nums.append(float(m.group(2)))
                    if len(nums) == 2:
                        pts.append((nums[0], h - nums[1]))
                        nums = []
            if len(pts) >= 2:
                paths.append({"op": op, "closed": closed, "points": np.array(pts), "part": el.get("data-part", "")})
                n += 1
        groups[op] = n
    length = {}
    for p in paths:
        pl = Path(p["points"], p["closed"], p["op"])
        length[p["op"]] = length.get(p["op"], 0.0) + pl.length()
    return {"width_mm": w, "height_mm": h, "groups": groups, "paths": paths, "length_mm": length}
