#!/usr/bin/env python
"""Regis monsters: Vegeta designs exported as static glTF models for the Regis real-time combat screen.

Regis (technolog-spectrumai/regis) shows each monster kind as a stationary, deliberately robotic 3D model. The models
are Vegeta CAD designs, built and made light here, offline; Regis packages the GLBs and never needs Vegeta at runtime.

    ghoul   -> Humanoid              (labs/legged)    wraith -> Balloon            (labs/regis, new)
    vampire -> Quadcopter            (labs/aero)      wyvern -> Drako hexacopter   (labs/aero)
    hound   -> RobotDog (Cerberus)   (labs/legged)    leshy  -> Onager Sentinel    (labs/land)

Each design's ``build()`` is split into its named, coloured parts (``vegeta.fidia.runner.split_parts``), tessellated
with ``vegeta.labkit.visual`` and simplified by vertex clustering to one budget for the whole model (``MAX_TRIANGLES``),
then written by ``vegeta.fidia.export.export_scene``: glTF 2.0 binary, Y up, metres, one node and one base-colour
material per part (parts without a colour of their own take the monster's colour). The designs are this repository's
own, so they are loaded directly, not through Fidia's sandbox contract for generated code.

    python labs/regis/export_monsters.py [--out exports/regis] [kind ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAX_TRIANGLES = 6000

# kind -> (design "path:Class" relative to the repository, rgba for parts without a colour, what it is)
MONSTERS = {
    "ghoul": ("labs/legged/designs/humanoid.py:Humanoid", (0.50, 0.68, 0.35, 1.0), "the humanoid walker"),
    "wraith": ("labs/regis/designs/balloon.py:Balloon", (0.56, 0.72, 0.91, 1.0), "a tethered survey balloon"),
    "vampire": ("labs/aero/designs/quadcopter_assembly.py:Quadcopter", (0.66, 0.20, 0.24, 1.0), "the X-frame quadcopter"),
    "wyvern": ("labs/aero/designs/drako.py:Drako", (0.80, 0.55, 0.20, 1.0), "the Drako heavy-lift hexacopter"),
    "hound": ("labs/legged/designs/robot_dog.py:RobotDog", (0.62, 0.64, 0.68, 1.0), "the Cerberus robot dog"),
    "leshy": ("labs/land/designs/onager.py:OnagerSentinel", (0.42, 0.52, 0.30, 1.0), "the Onager Sentinel land robot"),
}


def _paths():
    libs = ["libs/vegeta-cli/src", "libs/vegeta-core/src", "libs/vegeta-ai/src", "libs/vegeta-fidia/src", "libs/labkit/src",
            "libs/ogma/src"]
    designs = sorted({str(Path(d.split(":")[0]).parent) for d, _, _ in MONSTERS.values()})
    for p in libs + designs:
        if str(ROOT / p) not in sys.path:
            sys.path.insert(0, str(ROOT / p))


def parts_of(kind: str):
    """The monster's design as Fidia parts (millimetres, Z up), at most MAX_TRIANGLES triangles in all."""
    import numpy as np
    from vegeta.dedalus.loading import load_design
    from vegeta.fidia.mesh import Part
    from vegeta.fidia.runner import safe_name, split_parts
    from vegeta.labkit.visual import cluster, tessellate

    spec, tint, _ = MONSTERS[kind]
    path, cls = spec.split(":")
    design = load_design(f"{ROOT / path}:{cls}")
    meshes, taken = [], set()
    for name, shape, color in split_parts(design.build(design.resolve())):
        pts, faces = tessellate(shape, 0.5)
        meshes.append((safe_name(name, taken), tuple(color) if color is not None else tint, pts, faces))
    total = sum(len(f) for *_, f in meshes)
    if total > MAX_TRIANGLES:
        # one clustering cell for the whole model, grown until it fits (as labkit.visual.mesh does for one part)
        lo = np.min([m[2].min(0) for m in meshes], axis=0)
        hi = np.max([m[2].max(0) for m in meshes], axis=0)
        cell = float(np.max(hi - lo)) / 400
        while True:
            light = [(n, c, *cluster(p, f, cell)) for n, c, p, f in meshes]
            if sum(len(f) for *_, f in light) <= MAX_TRIANGLES:
                meshes = light
                break
            cell *= 1.25
    return [Part(name=n, color=c, vertices=np.asarray(p, float) * 1e3, triangles=np.asarray(f, np.int64))
            for n, c, p, f in meshes if len(f)]


def export(kind: str, out: Path) -> dict:
    import numpy as np
    import trimesh
    from vegeta.fidia.export import export_scene

    parts = parts_of(kind)
    with tempfile.TemporaryDirectory() as tmp:
        export_scene(parts, tmp)
        target = out / f"{kind}.glb"
        shutil.copyfile(Path(tmp) / "model.glb", target)
    data = target.read_bytes()
    scene = trimesh.load(target, file_type="glb")  # read back: what Regis will load
    lo, hi = scene.bounds
    spec, _, what = MONSTERS[kind]
    return {"file": target.name, "design": spec, "what": what, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            "parts": [p.name for p in parts], "triangles": int(sum(len(p.triangles) for p in parts)),
            "bounds_m": {"min": np.round(lo, 4).tolist(), "max": np.round(hi, 4).tolist()},
            "frame": "glTF: Y up, metres, x forward; rest pose (no joints or motions)"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(ROOT / "exports" / "regis"))
    ap.add_argument("kinds", nargs="*", default=list(MONSTERS))
    a = ap.parse_args(argv)
    _paths()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"format": "vegeta-regis-monsters/1", "monsters": {}}
    for kind in a.kinds:
        manifest["monsters"][kind] = info = export(kind, out)
        print(f"{kind:8} {info['triangles']:5d} triangles  {info['bytes']:7d} bytes  {info['what']}")
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
