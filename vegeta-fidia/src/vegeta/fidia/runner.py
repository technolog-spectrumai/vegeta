"""The subprocess that runs the model's code: ``python -I -m vegeta.fidia.runner design.py outdir [params.json]``.

It re-checks the source, loads the Dedalus design, calls ``build()`` itself (``Design.generate()`` would flatten an
assembly and lose part names and colours), splits the result into parts (placing each with its location), measures
each part's B-rep, tessellates with a size-relative tolerance, and writes ``result.json`` + ``parts.npz`` +
``model.step`` into ``outdir``. It always exits 0 with a ``result.json`` unless the operating system kills it.
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
import traceback
from pathlib import Path

PALETTE = [(0.62, 0.72, 0.82), (0.82, 0.64, 0.62), (0.64, 0.82, 0.62), (0.81, 0.76, 0.62), (0.70, 0.62, 0.82), (0.62, 0.82, 0.80)]


def safe_name(name: str, taken: set[str]) -> str:
    """The last path segment as a glTF/OBJ-safe identifier, unique within the model."""
    base = re.sub(r"[^A-Za-z0-9_]+", "_", name.split("/")[-1]).strip("_") or "part"
    if base[0].isdigit():
        base = "p_" + base
    out, n = base, 2
    while out in taken:
        out, n = f"{base}_{n}", n + 1
    taken.add(out)
    return out


def split_parts(obj):
    """[(name, shape, rgba or None)] from an Assembly, Workplane or Shape — placed, not raw."""
    import cadquery as cq

    if isinstance(obj, cq.Assembly):
        out = []
        for shape, name, loc, color in obj:
            out.append((name, shape.moved(loc), color.toTuple() if color is not None else None))
        return out
    if isinstance(obj, cq.Workplane):
        shapes = [v for v in obj.vals() if isinstance(v, cq.Shape)]
        if not shapes:
            raise ValueError("build() returned a Workplane without any shape")
        shape = shapes[0] if len(shapes) == 1 else cq.Compound.makeCompound(shapes)
        return [("body", shape, None)]
    if isinstance(obj, cq.Shape):
        return [("body", obj, None)]
    raise TypeError(f"build() must return a cq.Assembly, cq.Workplane or cq.Shape, not {type(obj).__name__}")


def brep_summary(shape) -> dict:
    solids = shape.Solids()
    bb = shape.BoundingBox()
    volume = float(sum(s.Volume() for s in solids)) if solids else 0.0
    return {"valid": bool(shape.isValid()), "n_solids": len(solids), "volume_mm3": volume, "area_mm2": float(shape.Area()),
            "bbox_min": [bb.xmin, bb.ymin, bb.zmin], "bbox_max": [bb.xmax, bb.ymax, bb.zmax]}


def contacts(parts, boxes, gap: float, max_parts: int = 60) -> dict:
    """Pairs of parts closer than ``gap`` mm (exact B-rep distance), for the floating-part check."""
    pairs = []
    if len(parts) > max_parts:
        return {"gap_mm": gap, "pairs": pairs, "skipped": f"more than {max_parts} parts"}
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            a, b = boxes[i], boxes[j]
            if (a.xmin - gap > b.xmax or b.xmin - gap > a.xmax or a.ymin - gap > b.ymax or b.ymin - gap > a.ymax
                    or a.zmin - gap > b.zmax or b.zmin - gap > a.zmax):
                continue
            d = float(parts[i][1].distance(parts[j][1]))
            if d <= gap:
                pairs.append([i, j, round(d, 4)])
    return {"gap_mm": gap, "pairs": pairs}


def main(argv=None) -> int:
    import numpy as np

    argv = list(sys.argv[1:] if argv is None else argv)
    design_path, outdir = Path(argv[0]), Path(argv[1])
    params = json.loads(Path(argv[2]).read_text()) if len(argv) > 2 else {}
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    result: dict = {"status": "ok", "parts": [], "error": None, "traceback": None, "parameters": {}}

    def done(status: str | None = None, error: str | None = None, tb: str | None = None) -> int:
        if status:
            result.update(status=status, error=error, traceback=tb)
        result["duration_s"] = time.monotonic() - t0
        (outdir / "result.json").write_text(json.dumps(result, indent=2, default=str))
        return 0

    from vegeta.fidia.contract import validate_source

    problems = validate_source(design_path.read_text())
    if problems:
        return done("rejected", "; ".join(problems))
    try:
        from vegeta.dedalus.loading import load_design

        dd = load_design(str(design_path))
        values = dd.resolve(**params)
        result["parameters"] = values
        parts = split_parts(dd.build(values))
    except MemoryError:
        return done("resource_limit", "out of memory while building")
    except Exception as exc:  # the model's code failed: tell it exactly where
        tb = traceback.format_exc()
        return done("build_error", f"{type(exc).__name__}: {exc}", "\n".join(tb.strip().splitlines()[-25:]))
    try:
        import cadquery as cq

        breps = [brep_summary(s) for _, s, _ in parts]  # before tessellating: meshing loosens OCC bounding boxes
        boxes = [s.BoundingBox() for _, s, _ in parts]
        mins = np.array([[b.xmin, b.ymin, b.zmin] for b in boxes]).min(0)
        maxs = np.array([[b.xmax, b.ymax, b.zmax] for b in boxes]).max(0)
        diag = float(np.linalg.norm(maxs - mins)) or 1.0
        tol = min(max(1e-3 * diag, 0.01), 1.0)
        arrays, taken = {}, set()
        for i, (name, shape, color) in enumerate(parts):
            verts, tris = shape.tessellate(tol, 0.2)
            v = np.array([[p.x, p.y, p.z] for p in verts], dtype=np.float64).reshape(-1, 3)
            t = np.array(tris, dtype=np.int64).reshape(-1, 3)
            arrays[f"v{i}"], arrays[f"t{i}"] = v, t
            rgba = list(color) if color is not None else list(PALETTE[i % len(PALETTE)]) + [1.0]
            result["parts"].append({"index": i, "name": safe_name(name, taken), "source_name": name, "color": rgba,
                                    "color_given": color is not None, "brep": breps[i],
                                    "n_vertices": len(v), "n_triangles": len(t)})
        np.savez_compressed(outdir / "parts.npz", **arrays)
        result["bbox_min"], result["bbox_max"] = mins.tolist(), maxs.tolist()
        result["tessellation"] = {"linear_mm": tol, "angular_rad": 0.2}
        result["contacts"] = contacts(parts, boxes, max(0.5, 2e-3 * diag))
        compound = cq.Compound.makeCompound([s for _, s, _ in parts])
        cq.exporters.export(compound, str(outdir / "model.step"))
    except MemoryError:
        return done("resource_limit", "out of memory while meshing")
    except Exception as exc:
        tb = traceback.format_exc()
        return done("build_error", f"meshing failed: {type(exc).__name__}: {exc}", "\n".join(tb.strip().splitlines()[-25:]))
    if not math.isfinite(result["bbox_max"][0]):
        return done("build_error", "the model has no geometry")
    return done()


if __name__ == "__main__":
    sys.exit(main())
