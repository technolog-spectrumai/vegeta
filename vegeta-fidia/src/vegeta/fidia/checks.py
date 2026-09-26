"""Deterministic checks on a built model: the only thing that makes a revision valid or invalid.

Each check is ``pass``, ``warn`` or ``fail``; any ``fail`` makes the revision invalid (it can never become the
best output, whatever the reviewer says). *Fail*: B-rep invalid, no solid, no volume, not watertight, inconsistent
or inverted winding, over the triangle budget, no parts, a broken export round-trip. *Warn*: degenerate faces,
several bodies in one part, planned parts missing, size off the plan, not resting on z = 0, floating parts, parts
without a colour.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .mesh import Part, bounds

PASS, WARN, FAIL = "pass", "warn", "fail"


@dataclass
class Check:
    name: str
    status: str
    message: str = ""
    part: str | None = None
    value: Any = None
    expected: Any = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def line(self) -> str:
        where = f"[{self.part}] " if self.part else ""
        return f"{self.status.upper():4s} {self.name}: {where}{self.message}"


def _part_checks(part: Part) -> list[Check]:
    out: list[Check] = []
    b = part.brep or {}
    n = part.name

    def add(name, ok, message, status_bad=FAIL, **kw):
        out.append(Check(name, PASS if ok else status_bad, message if not ok else kw.pop("ok_message", "ok"), n, **kw))

    if b:
        add("brep_valid", b.get("valid", False), "the CAD solid is not valid (OpenCASCADE check)")
        add("solid", b.get("n_solids", 0) >= 1, "the part contains no closed solid (only faces or wires)",
            value=b.get("n_solids", 0), expected=">= 1")
        add("volume", b.get("volume_mm3", 0.0) > 0, "the part has no volume", value=b.get("volume_mm3"))
    if len(part.triangles) == 0:
        out.append(Check("mesh", FAIL, "the part has no triangles", n))
        return out
    import trimesh

    m = part.to_trimesh()
    add("watertight", m.is_watertight, "the mesh has holes or open edges (not watertight)")
    consistent = m.is_winding_consistent
    add("winding", consistent and (not m.is_watertight or m.volume > 0),
        "the triangle winding is inconsistent" if not consistent else "the mesh is inside out (negative volume)")
    scale = float(np.linalg.norm(m.extents)) or 1.0
    degenerate = int((m.area_faces < 1e-10 * scale * scale).sum())
    add("degenerate", degenerate == 0, f"{degenerate} degenerate (zero-area) triangle(s)", WARN, value=degenerate)
    bodies = len(trimesh.graph.connected_components(m.face_adjacency, nodes=np.arange(len(m.faces)), engine="scipy"))
    add("components", bodies <= 1, f"{bodies} separate bodies in one part (split it or join them)", WARN, value=bodies)
    add("color", part.color_given, "no colour given; a default colour was used", WARN)
    return out


def _floating(parts: Sequence[Part], contacts: Mapping | None) -> tuple[list[str], str]:
    """Names of parts not connected to the largest connected group, and how contact was decided."""
    n = len(parts)
    if n < 2:
        return [], "one part"
    adj = {i: set() for i in range(n)}
    if contacts and "pairs" in contacts and not contacts.get("skipped"):
        method = f"exact B-rep distance <= {contacts.get('gap_mm', 0):.2g} mm"
        for i, j, _ in contacts["pairs"]:
            adj[i].add(j)
            adj[j].add(i)
    else:  # without the runner's contacts: bounding boxes that touch (optimistic)
        boxes = [p.bounds for p in parts]
        gap = 2e-3 * float(np.linalg.norm(np.ptp(bounds(list(parts)), axis=0))) + 1e-6
        method = "bounding boxes (approximate)"
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(boxes[i][0] - gap <= boxes[j][1]) and np.all(boxes[j][0] - gap <= boxes[i][1]):
                    adj[i].add(j)
                    adj[j].add(i)
    groups, seen = [], set()
    for s in range(n):
        if s in seen:
            continue
        stack, g = [s], set()
        while stack:
            k = stack.pop()
            if k not in g:
                g.add(k)
                stack.extend(adj[k] - g)
        seen |= g
        groups.append(g)
    main = max(groups, key=lambda g: sum(float(parts[k].brep.get("volume_mm3", 0.0) or len(parts[k].triangles)) for k in g))
    return [parts[k].name for k in range(n) if k not in main], method


def check_parts(parts: Sequence[Part], plan: Mapping | None = None, *, contacts: Mapping | None = None,
                max_triangles: int = 200_000, size_tol: float = 0.35) -> list[Check]:
    """All geometry checks for one revision (``plan`` optional: planned parts, ``size_mm``, ``floating_ok``)."""
    plan = plan or {}
    if not parts:
        return [Check("parts", FAIL, "the model has no parts")]
    out: list[Check] = []
    for p in parts:
        out += _part_checks(p)
    total = int(sum(len(p.triangles) for p in parts))
    out.append(Check("triangles", PASS if total <= max_triangles else FAIL,
                     f"{total} triangles" + ("" if total <= max_triangles else f" (budget {max_triangles}; simplify)"),
                     value=total, expected=max_triangles))
    lo, hi = bounds(list(parts))
    size = hi - lo
    names = [p.name for p in parts]
    planned = [str(q.get("name", "")) for q in plan.get("parts", []) if q.get("name")]
    if planned:
        missing = [q for q in planned if q not in names]
        out.append(Check("planned_parts", WARN if missing else PASS,
                         f"missing planned parts: {', '.join(missing)}" if missing else f"all {len(planned)} planned parts present",
                         value=names, expected=planned))
    want = plan.get("size_mm")
    if want and len(want) == 3 and all(float(w) > 0 for w in want):
        want = np.array(want, dtype=float)

        def off(s):
            return float(np.max(np.abs(s - want) / want))

        err = min(off(size), off(size[[1, 0, 2]]))  # a model turned 90 degrees about Z is the same size
        out.append(Check("size", PASS if err <= size_tol else WARN,
                         f"size {size.round(1).tolist()} mm vs planned {want.round(1).tolist()} mm ({100 * err:.0f} % off)",
                         value=size.round(2).tolist(), expected=want.tolist()))
    scale = float(np.linalg.norm(size)) or 1.0
    out.append(Check("grounded", PASS if abs(lo[2]) <= 0.01 * scale + 0.5 else WARN,
                     f"lowest point at z = {lo[2]:.1f} mm (should rest on z = 0)", value=round(float(lo[2]), 3), expected=0.0))
    floating, method = _floating(parts, contacts)
    if plan.get("floating_ok"):
        out.append(Check("floating", PASS, "separate parts allowed by the plan", value=floating))
    else:
        out.append(Check("floating", WARN if floating else PASS,
                         (f"not attached to the rest: {', '.join(floating)} ({method})" if floating
                          else f"all parts connected ({method})"), value=floating))
    return out


def check_export(report: Mapping) -> list[Check]:
    """One check per exported format from ``export.reimport_check``'s report."""
    out = []
    for fmt, r in report.get("formats", {}).items():
        out.append(Check(f"export_{fmt}", PASS if r.get("ok") else FAIL,
                         "round-trip ok" if r.get("ok") else "; ".join(r.get("problems", [])) or "failed"))
    if not out:
        out.append(Check("export", FAIL, report.get("error") or "nothing was exported"))
    return out


def summarize(checks: Sequence[Check]) -> dict[str, Any]:
    fails = [c for c in checks if c.status == FAIL]
    warns = [c for c in checks if c.status == WARN]
    return {"valid": not fails, "fails": len(fails), "warnings": len(warns), "passes": len(checks) - len(fails) - len(warns),
            "problems": [c.line() for c in fails + warns]}
