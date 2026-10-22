"""GRBL G-code for a diode laser (xTool D1 Pro and the like): ``G21 G90``, one ``M4 S`` block per operation in the
engineer's order, ``G0`` travels, ``G1 ... F`` cuts, the passes repeated, ``M5`` between parts, home at the end.
Holes are cut before the contour that holds them, parts are visited nearest-first. The time estimate is path
length over feed plus travel over the machine's rapid feed: no acceleration model (the machine is slower)."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path as FsPath

import numpy as np

from .geometry import Part, Path
from .settings import CutSettings, Operation


@dataclass
class GcodeJob:
    path: FsPath | None
    bbox: tuple[float, float, float, float]
    cut_length_mm: dict[str, float] = field(default_factory=dict)      # per operation, passes included
    travel_length_mm: float = 0.0
    time_s: dict[str, float] = field(default_factory=dict)             # per operation, plus "travel"
    passes_total: int = 0
    lines: int = 0

    @property
    def total_time_s(self) -> float:
        return sum(self.time_s.values())

    @property
    def total_cut_length_mm(self) -> float:
        return sum(self.cut_length_mm.values())


def _ops_in_order(settings: CutSettings, parts: list[Part]) -> list[Operation]:
    needed = {p.op for part in parts for p in part.paths()}
    for op in needed:
        settings.operation(op)             # raises when a part asks for an operation the settings do not state
    return sorted((settings.operations[o] for o in needed), key=lambda o: (o.order, o.name))


def _part_paths(part: Part, op: str) -> list[Path]:
    """Paths of one operation on a part: holes first, the outer contour last."""
    if op == "cut":
        return [*part.holes, part.outer]
    return [p for p in part.paths() if p.op == op]


def _nearest_order(parts: list[Part], op: str, start: np.ndarray) -> list[Part]:
    todo = [p for p in parts if _part_paths(p, op)]
    out, here = [], start
    while todo:
        i = min(range(len(todo)), key=lambda k: float(np.linalg.norm(_part_paths(todo[k], op)[0].points[0] - here)))
        out.append(todo.pop(i))
        here = _part_paths(out[-1], op)[-1].points[0]
    return out


def plan(parts: list[Part], settings: CutSettings) -> GcodeJob:
    """Lengths and times without writing a file."""
    return _generate(parts, settings, None)


def write_gcode(path: str | FsPath, parts: list[Part], settings: CutSettings, title: str = "") -> GcodeJob:
    """Write the G-code for ``parts`` (sheet coordinates, origin at the front-left of the sheet, which must lie at
    the machine's origin) and return the job's lengths and times."""
    return _generate(parts, settings, FsPath(path), title)


def _generate(parts: list[Part], settings: CutSettings, path: FsPath | None, title: str = "") -> GcodeJob:
    m = settings.machine
    for part in parts:
        x0, y0, x1, y1 = part.bbox()
        if x0 < -1e-6 or y0 < -1e-6 or x1 > m.bed_width_mm + 1e-6 or y1 > m.bed_height_mm + 1e-6:
            raise ValueError(f"part {part.name!r} lies outside the bed ({m.bed_width_mm:g} x {m.bed_height_mm:g} mm)")
    ops = _ops_in_order(settings, parts)
    out = [f"; {title or 'atropos job'}", f"; machine: {m.name}; sheet: {settings.sheet.name}",
           "; operations: " + "; ".join(f"{o.name} {o.power_pct:g} % {o.feed_mm_min:g} mm/min x {o.passes}" for o in ops),
           "G21 G90 G17", "M5", f"G0 F{m.travel_feed_mm_min:g}"]
    job = GcodeJob(path, (0.0, 0.0, 0.0, 0.0))
    here = np.zeros(2)
    travel = 0.0
    allpts = []
    for op in ops:
        s = int(round(op.power_pct / 100 * m.s_max))
        out.append(f"; --- {op.name}: S{s} F{op.feed_mm_min:g} passes {op.passes} kerf {op.kerf_mm:g} mm")
        out.append(f"M4 S{s}")
        length = 0.0
        for part in _nearest_order(parts, op.name, here):
            out.append(f"; part {part.name}")
            for p in _part_paths(part, op.name):
                pts = p.points
                allpts.append(pts)
                travel += float(np.linalg.norm(pts[0] - here))
                out.append(f"G0 X{pts[0][0]:.3f} Y{pts[0][1]:.3f}")
                seq = np.vstack([pts[1:], pts[:1]]) if p.closed else pts[1:]
                for k in range(op.passes):
                    first = True
                    for x, y in seq:
                        out.append(f"G1 X{x:.3f} Y{y:.3f}" + (f" F{op.feed_mm_min:g} S{s}" if first else ""))
                        first = False
                    if k < op.passes - 1 and not p.closed:          # an open path: back to its start for the next pass
                        out.append(f"G0 X{pts[0][0]:.3f} Y{pts[0][1]:.3f}")
                        travel += p.length()
                length += p.length() * op.passes
                here = pts[0] if p.closed else pts[-1]
            out.append("M5")
            out.append(f"M4 S{s}")
        out.pop()                                                   # no M4 after the last part of the operation
        job.cut_length_mm[op.name] = length
        job.time_s[op.name] = length / op.feed_mm_min * 60.0
        job.passes_total += op.passes
    travel += float(np.linalg.norm(here))
    out += ["M5", "G0 X0 Y0", "M2"]
    job.travel_length_mm = travel
    job.time_s["travel"] = travel / m.travel_feed_mm_min * 60.0
    if allpts:
        a = np.vstack(allpts)
        job.bbox = (float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 0].max()), float(a[:, 1].max()))
    job.lines = len(out)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(out) + "\n")
    return job


def estimate_time(parts: list[Part], settings: CutSettings) -> dict[str, float]:
    """Seconds per operation and for travel, as the G-code writer would count them."""
    return dict(plan(parts, settings).time_s)


_WORD = re.compile(r"([GMXYZFS])\s*([-+]?\d*\.?\d+)")


def read_gcode(path: str | FsPath) -> dict:
    """A small reader for GRBL files: the bounding box of the G1 moves, G1 length per S value and in total, G0 travel,
    the feeds and powers seen, and the time estimate from those feeds."""
    x = y = 0.0
    s = f = None
    mode = None
    g1_len, g0_len, time_s = 0.0, 0.0, 0.0
    by_s: dict[int, float] = {}
    pts = []
    feeds, powers, m4 = set(), set(), 0
    for raw in FsPath(path).read_text().splitlines():
        line = raw.split(";", 1)[0].strip().upper()
        if not line:
            continue
        words = dict()
        for k, v in _WORD.findall(line):
            words.setdefault(k, []).append(float(v))
        if "M" in words:
            for mv in words["M"]:
                if int(mv) in (3, 4):
                    m4 += 1
        if "G" in words:
            g = int(words["G"][0])
            if g in (0, 1):
                mode = g
            else:
                mode = None if g not in (0, 1) and ("X" not in words and "Y" not in words) else mode
        if "S" in words:
            s = int(words["S"][0])
            powers.add(s)
        if "F" in words:
            f = words["F"][0]
            feeds.add(f)
        if mode is None or ("X" not in words and "Y" not in words):
            continue
        nx = words["X"][0] if "X" in words else x
        ny = words["Y"][0] if "Y" in words else y
        d = math.hypot(nx - x, ny - y)
        if mode == 1:
            g1_len += d
            by_s[s or 0] = by_s.get(s or 0, 0.0) + d
            pts.append((nx, ny))
            if f:
                time_s += d / f * 60.0
        else:
            g0_len += d
            if f:
                pass
        x, y = nx, ny
    bbox = None
    if pts:
        a = np.array(pts)
        bbox = (float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 0].max()), float(a[:, 1].max()))
    return {"bbox": bbox, "g1_length_mm": g1_len, "g0_length_mm": g0_len, "length_by_s_mm": by_s, "feeds": sorted(feeds),
            "powers": sorted(powers), "max_s": max(powers) if powers else None, "laser_on_blocks": m4,
            "cut_time_s": time_s}
