"""Write the model for other tools, and prove the files re-import.

``export_scene(parts, outdir)`` writes ``model.glb``, ``model.gltf`` + ``model.bin``, ``model.obj`` + ``model.mtl``,
``model.stl`` and (when the runner made one) ``model.step``, plus ``manifest.json``. One glTF node and one
base-colour material per part; in OBJ one ``o`` object per part and one ``newmtl`` (``Kd``) per colour. glTF/GLB are **Y up, metres**
(the glTF convention; the transform is in the manifest); OBJ, STL and STEP stay **Z up, millimetres** like the CAD.

``reimport_check(outdir, parts)`` reads every file back with two independent readers (trimesh and, when installed,
pyvista/VTK), checks the sidecars (``buffers[*].uri`` exist with the right ``byteLength``; ``mtllib`` resolves and
every ``usemtl`` has a ``Kd``) and compares part count, names, triangles, bounds and colours with what was written.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .mesh import Part, bounds

# CAD (Z up, mm) -> glTF (Y up, m): rotate -90 degrees about X (z -> y, y -> -z), then scale by 1/1000
GLTF_FROM_CAD = np.array([[0.001, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.001, 0.0],
                          [0.0, -0.001, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 1.0]])
FORMATS = ("glb", "gltf", "obj", "stl", "step")


def _scene(parts: Sequence[Part], transform: np.ndarray | None = None):
    import trimesh
    from trimesh.visual import TextureVisuals
    from trimesh.visual.material import PBRMaterial

    scene = trimesh.Scene()
    for p in parts:
        # unmerged vertices (one set per CAD face): crisp edges and smooth curved faces in viewers
        m = trimesh.Trimesh(np.asarray(p.vertices, float), np.asarray(p.triangles), process=False)
        if transform is not None:
            m.apply_transform(transform)
        rgba = p.rgba8
        m.visual = TextureVisuals(material=PBRMaterial(
            name=p.name, baseColorFactor=rgba, metallicFactor=0.0, roughnessFactor=0.6,
            alphaMode="BLEND" if rgba[3] < 255 else "OPAQUE", doubleSided=False))
        scene.add_geometry(m, node_name=p.name, geom_name=p.name)
    return scene


def export_scene(parts: Sequence[Part], outdir: str | Path, *, step: str | Path | None = None) -> dict[str, Any]:
    """Write all formats into ``outdir`` and return the manifest (also saved as ``manifest.json``)."""
    import trimesh

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, Any] = {}

    gl = _scene(parts, GLTF_FROM_CAD)
    (out / "model.glb").write_bytes(gl.export(file_type="glb"))
    files["glb"] = "model.glb"

    written = trimesh.exchange.gltf.export_gltf(gl, include_normals=True, merge_buffers=True)
    tree = json.loads(written.pop("model.gltf"))
    renames = {old: ("model.bin" if len(written) == 1 else f"model_{i}.bin") for i, old in enumerate(sorted(written))}
    for buf in tree.get("buffers", []):
        if buf.get("uri") in renames:
            buf["uri"] = renames[buf["uri"]]
    for old, new in renames.items():
        (out / new).write_bytes(written[old])
    tree.setdefault("asset", {})["generator"] = "vegeta-fidia (trimesh)"
    (out / "model.gltf").write_text(json.dumps(tree, indent=1))
    files["gltf"] = "model.gltf"
    files["gltf_buffers"] = sorted(renames.values())

    cad = _scene(parts)
    text, extra = trimesh.exchange.obj.export_obj(cad, include_normals=True, include_color=True, include_texture=True,
                                                  return_texture=True, mtl_name="model.mtl")
    (out / "model.obj").write_text(text)
    for name, data in extra.items():
        (out / name).write_bytes(data)
    files["obj"], files["mtl"] = "model.obj", "model.mtl"

    whole = trimesh.util.concatenate([p.to_trimesh() for p in parts])
    (out / "model.stl").write_bytes(whole.export(file_type="stl"))
    files["stl"] = "model.stl"
    if step and Path(step).is_file():
        shutil.copyfile(step, out / "model.step")
        files["step"] = "model.step"

    lo, hi = bounds(list(parts))
    manifest = {
        "files": files,
        "units": {"glb": "m", "gltf": "m", "obj": "mm", "stl": "mm", "step": "mm"},
        "up_axis": {"glb": "Y", "gltf": "Y", "obj": "Z", "stl": "Z", "step": "Z"},
        "gltf_from_cad": GLTF_FROM_CAD.tolist(),
        "bounds_mm": [lo.tolist(), hi.tolist()],
        "n_triangles": int(sum(len(p.triangles) for p in parts)),
        "parts": [{"name": p.name, "color_rgba": [round(c, 4) for c in p.color], "n_triangles": int(len(p.triangles)),
                   "n_vertices": int(len(p.vertices))} for p in parts],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


# -- re-import -----------------------------------------------------------------------------------------------------
def _gltf_sidecars(path: Path) -> list[str]:
    problems = []
    tree = json.loads(path.read_text())
    for i, buf in enumerate(tree.get("buffers", [])):
        uri = buf.get("uri")
        if uri is None or uri.startswith("data:"):
            continue
        f = path.parent / uri
        if not f.is_file():
            problems.append(f"buffer {i}: sidecar {uri} is missing")
        elif f.stat().st_size != buf.get("byteLength"):
            problems.append(f"buffer {i}: {uri} has {f.stat().st_size} bytes, expected {buf.get('byteLength')}")
    for img in tree.get("images", []):
        if img.get("uri") and not img["uri"].startswith("data:") and not (path.parent / img["uri"]).is_file():
            problems.append(f"image {img['uri']} is missing")
    return problems


def _obj_sidecars(path: Path) -> list[str]:
    text = path.read_text()
    libs = re.findall(r"^mtllib\s+(.+?)\s*$", text, re.M)
    used = set(re.findall(r"^usemtl\s+(.+?)\s*$", text, re.M))
    if not libs:
        return ["no mtllib line (colours would be lost)"] if used else []
    problems, defined = [], {}
    for lib in libs:
        f = path.parent / lib
        if not f.is_file():
            problems.append(f"material library {lib} is missing")
            continue
        current = None
        for line in f.read_text().splitlines():
            if line.startswith("newmtl "):
                current = line.split(None, 1)[1].strip()
                defined[current] = False
            elif line.startswith("Kd ") and current:
                defined[current] = True
    problems += [f"material {m} is used but not defined" for m in sorted(used - set(defined))]
    problems += [f"material {m} has no Kd colour" for m in sorted(used) if defined.get(m) is False]
    return problems


def _color(geom) -> list[int] | None:
    mat = getattr(getattr(geom, "visual", None), "material", None)
    for attr in ("baseColorFactor", "diffuse"):  # trimesh gives both as uint8 RGBA
        c = getattr(mat, attr, None)
        if c is not None:
            return [int(v) for v in np.asarray(c).ravel()[:4]]
    return None


def _to_cad(bounds_: np.ndarray, fmt: str) -> np.ndarray:
    """Bounds read from a file, back in CAD millimetres (Z up)."""
    if fmt not in ("glb", "gltf"):
        return bounds_
    inv = np.linalg.inv(GLTF_FROM_CAD)
    corners = np.array([[x, y, z, 1.0] for x in bounds_[:, 0] for y in bounds_[:, 1] for z in bounds_[:, 2]])
    back = (inv @ corners.T).T[:, :3]
    return np.array([back.min(0), back.max(0)])


def _trimesh_read(path: Path, fmt: str, parts: Sequence[Part], tol: float) -> dict[str, Any]:
    import trimesh

    # OBJ: one geometry per `o` object (by default trimesh groups faces by material, and parts of one colour share one)
    kw = {"split_objects": True, "group_material": False} if fmt == "obj" else {}
    scene = trimesh.load(str(path), force="scene", process=False, **kw)
    geoms = dict(scene.geometry)
    tris = int(sum(len(g.faces) for g in geoms.values()))
    got = _to_cad(np.asarray(scene.bounds), fmt)
    want = bounds(list(parts))
    info: dict[str, Any] = {"parts": len(geoms), "triangles": tris, "bounds_mm": got.round(4).tolist()}
    problems = []
    n_tri = int(sum(len(p.triangles) for p in parts))
    if tris != n_tri:
        problems.append(f"{tris} triangles read back, {n_tri} written")
    if np.max(np.abs(got - want)) > tol:
        problems.append(f"bounds {got.round(3).tolist()} mm differ from {want.round(3).tolist()} mm")
    if fmt in ("glb", "gltf", "obj"):
        names = {p.name for p in parts}
        if set(geoms) != names:
            problems.append(f"parts read back {sorted(geoms)} differ from {sorted(names)}")
        else:
            bad = []
            for p in parts:
                c = _color(geoms[p.name])
                if c is None or max(abs(a - b) for a, b in zip(c[:3], p.rgba8[:3])) > 2:
                    bad.append(f"{p.name} {c} != {p.rgba8}")
            if bad:
                problems.append("colours differ: " + "; ".join(bad))
    info["problems"] = problems
    return info


def _pyvista_read(path: Path, fmt: str, parts: Sequence[Part], tol: float) -> dict[str, Any]:
    import pyvista as pv

    data = pv.read(str(path))
    blocks = []

    def leaves(d):
        if isinstance(d, pv.MultiBlock):
            for b in d:
                if b is not None:
                    leaves(b)
        else:
            blocks.append(d)
    leaves(data)
    tris = int(sum(b.extract_surface(algorithm=None).triangulate().n_cells if hasattr(b, "extract_surface") else b.n_cells
                   for b in blocks))
    b = np.array(data.bounds).reshape(3, 2).T
    got = _to_cad(b, fmt)
    want = bounds(list(parts))
    n_tri = int(sum(len(p.triangles) for p in parts))
    problems = []
    if tris != n_tri:
        problems.append(f"{tris} triangles read back by VTK, {n_tri} written")
    if np.max(np.abs(got - want)) > tol:
        problems.append(f"VTK bounds {got.round(3).tolist()} mm differ from {want.round(3).tolist()} mm")
    return {"blocks": len(blocks), "triangles": tris, "bounds_mm": got.round(4).tolist(), "problems": problems}


def reimport_check(outdir: str | Path, parts: Sequence[Part], *, readers: Sequence[str] = ("trimesh", "pyvista")) -> dict[str, Any]:
    """Read every exported file back and compare it with ``parts``; saved as ``reimport.json``."""
    out = Path(outdir)
    manifest = json.loads((out / "manifest.json").read_text())
    lo, hi = bounds(list(parts))
    tol = max(1e-3, 1e-4 * float(np.linalg.norm(hi - lo)))
    report: dict[str, Any] = {"formats": {}, "readers": []}
    try:
        import pyvista  # noqa: F401
        have_vtk = "pyvista" in readers
    except ImportError:
        have_vtk = False
    report["readers"] = [r for r in readers if r == "trimesh" or (r == "pyvista" and have_vtk)]
    for fmt in ("glb", "gltf", "obj", "stl", "step"):
        name = manifest["files"].get(fmt)
        if not name:
            continue
        path = out / name
        entry: dict[str, Any] = {"file": name, "problems": []}
        if not path.is_file():
            entry["problems"].append(f"{name} is missing")
        elif fmt == "step":
            head = path.read_bytes()[:64]
            if not head.startswith(b"ISO-10303-21"):
                entry["problems"].append("not a STEP file (no ISO-10303-21 header)")
            entry["bytes"] = path.stat().st_size
        else:
            if fmt == "gltf":
                entry["problems"] += _gltf_sidecars(path)
            if fmt == "obj":
                entry["problems"] += _obj_sidecars(path)
            for reader, fn in (("trimesh", _trimesh_read), ("pyvista", _pyvista_read)):
                if reader not in report["readers"]:
                    continue
                if reader == "pyvista" and fmt == "gltf" and entry["problems"]:
                    continue  # VTK aborts on a missing buffer
                try:
                    info = fn(path, fmt, parts, tol)
                except Exception as exc:  # a reader that cannot read the file is the finding
                    info = {"problems": [f"{reader} could not read {name}: {type(exc).__name__}: {exc}"]}
                entry[reader] = info
                entry["problems"] += info.get("problems", [])
        entry["ok"] = not entry["problems"]
        report["formats"][fmt] = entry
    report["ok"] = bool(report["formats"]) and all(e["ok"] for e in report["formats"].values())
    (out / "reimport.json").write_text(json.dumps(report, indent=2, default=str))
    return report
