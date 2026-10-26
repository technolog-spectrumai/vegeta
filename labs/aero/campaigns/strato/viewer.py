"""Strato — a self-contained interactive 3D page of the CAD (Stage V8.14): every part's mesh embedded in one HTML
file, drawn with three.js (loaded from jsDelivr), orbit/zoom/pan with the mouse, a part toggle and a ship selector.

    python labs/aero/campaigns/strato/viewer.py [--out FILE] [--lengths 15 30 50 80]
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "designs"))
sys.path.insert(0, str(HERE))

import strato  # noqa: E402
from cad import COLORS  # noqa: E402

THREE = "https://cdn.jsdelivr.net/npm/three@0.160.0"


def mesh_of(geometry, target_faces: int = 6000) -> dict:
    """Triangles of a part [m], decimated to about ``target_faces``, as base64 float32 positions and uint32 indices."""
    from vegeta.dedalus.viz import to_pyvista
    m = to_pyvista(geometry, tolerance=max(geometry.dimensions) / 3000, angular_tolerance=0.2).triangulate()
    if m.n_cells > target_faces:
        m = m.decimate(1 - target_faces / m.n_cells)
    return _pack(np.asarray(m.points) * 1e-3, np.asarray(m.faces).reshape(-1, 4)[:, 1:])


def _pack(pts, faces) -> dict:
    return {"p": base64.b64encode(np.asarray(pts, np.float32).tobytes()).decode(),
            "i": base64.b64encode(np.asarray(faces, np.uint32).tobytes()).decode()}


def revolved(hull, x0: float, x1: float, half_deg: float = 180.0, offset: float = 0.0, nx: int = 160, nphi: int = 96) -> dict:
    """A smooth surface of revolution of the analytic hull profile (exactly the CAD's shape, without the
    tessellation and decimation noise): stations ``x0``–``x1`` [m], ±``half_deg`` about the top, ``offset`` [m] out."""
    u = 0.5 * (1 - np.cos(np.linspace(0.0, np.pi, nx)))
    x = x0 + (x1 - x0) * u
    r = hull.r(x) + offset
    phi = np.radians(np.linspace(-half_deg, half_deg, nphi))
    X, P = np.meshgrid(x, phi, indexing="ij")
    R = np.broadcast_to(r[:, None], X.shape)
    pts = np.stack([X, R * np.sin(P), R * np.cos(P)], -1).reshape(-1, 3)
    i, j = np.meshgrid(np.arange(nx - 1), np.arange(nphi - 1), indexing="ij")
    a = (i * nphi + j).ravel()
    faces = np.concatenate([np.stack([a, a + nphi, a + 1], 1), np.stack([a + 1, a + nphi, a + nphi + 1], 1)])
    return _pack(pts, faces)


def build(lengths, out: Path) -> Path:
    import strato_geometry as sg
    ships = {}
    for L in lengths:
        parts = strato.parts(**strato.for_length(L))
        model = sg.DEMONSTRATOR.scaled(L)
        h, lay = model.hull, model.layout
        ship = {}
        for name, g in parts.items():
            if name == "hull":
                m = revolved(h, 0.0, h.length)
            elif name == "solar":
                m = revolved(h, lay.solar_x0 * h.length, lay.solar_x1 * h.length, lay.solar_half_angle_deg,
                             offset=0.004 * h.diameter, nx=120, nphi=64)
            else:
                m = mesh_of(g, 9000 if name in ("propellers", "structure") else 3000)
            ship[name] = {**m, "color": COLORS[name]}
        ships[f"{L:g} m"] = ship
    html = PAGE.replace("__DATA__", json.dumps(ships)).replace("__THREE__", THREE)
    out.write_text(html)
    return out


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Strato 3D</title>
<style>
:root{--surface:#fcfcfb;--panel:#ffffff;--text:#0b0b0b;--muted:#52514e;--line:#e4e3de}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--surface:#1a1a19;--panel:#222221;--text:#f0efec;--muted:#c3c2b7;--line:#3a3a38}}
:root[data-theme="dark"]{--surface:#1a1a19;--panel:#222221;--text:#f0efec;--muted:#c3c2b7;--line:#3a3a38}
html,body{margin:0;height:100%;background:var(--surface);color:var(--text);font:14px/1.4 system-ui,sans-serif}
#ui{position:absolute;top:12px;left:12px;right:12px;display:flex;flex-wrap:wrap;gap:8px;align-items:center;pointer-events:none}
#ui>*{pointer-events:auto}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:6px 10px}
select,button{font:inherit;color:var(--text);background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:3px 8px}
label{margin-right:8px;white-space:nowrap}.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px}
#hint{position:absolute;bottom:10px;left:12px;color:var(--muted);font-size:12px}
canvas{display:block}
</style></head><body>
<div id="ui"><div class="card"><b>Strato</b> <select id="ship"></select></div><div class="card" id="parts"></div></div>
<div id="hint">drag to orbit · right-drag to pan · wheel to zoom · units m, x aft from the nose</div>
<script type="importmap">{"imports":{"three":"__THREE__/build/three.module.js","three/addons/":"__THREE__/examples/jsm/"}}</script>
<script type="module">
import * as THREE from "three";
import {OrbitControls} from "three/addons/controls/OrbitControls.js";
const DATA = __DATA__;
const b64 = (s, T) => { const b = atob(s), u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return new T(u.buffer); };
const renderer = new THREE.WebGLRenderer({antialias: true}); renderer.setPixelRatio(devicePixelRatio);
document.body.appendChild(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 5000); camera.up.set(0, 0, 1);
const controls = new OrbitControls(camera, renderer.domElement);
scene.add(new THREE.HemisphereLight(0xffffff, 0x666670, 2.2));
const sun = new THREE.DirectionalLight(0xffffff, 1.6); sun.position.set(-1, -2, 3); scene.add(sun);
let group = null; const visible = {};
function bg() { scene.background = new THREE.Color(getComputedStyle(document.documentElement).getPropertyValue("--surface").trim()); }
function load(name) {
  if (group) scene.remove(group);
  group = new THREE.Group();
  for (const [part, m] of Object.entries(DATA[name])) {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(b64(m.p, Float32Array), 3));
    g.setIndex(new THREE.BufferAttribute(b64(m.i, Uint32Array), 1)); g.computeVertexNormals();
    const mesh = new THREE.Mesh(g, new THREE.MeshStandardMaterial({color: m.color, roughness: 0.6, metalness: 0.05,
      transparent: part === "gondola", opacity: part === "gondola" ? 0.55 : 1}));
    mesh.name = part; mesh.visible = visible[part] ?? true; group.add(mesh);
  }
  scene.add(group);
  const box = new THREE.Box3().setFromObject(group), c = box.getCenter(new THREE.Vector3()), s = box.getSize(new THREE.Vector3()).length();
  controls.target.copy(c); camera.position.set(c.x - 0.55 * s, c.y - 0.75 * s, c.z + 0.35 * s); camera.near = s / 500; camera.far = s * 20;
  camera.updateProjectionMatrix(); controls.update();
}
const sel = document.getElementById("ship");
for (const k of Object.keys(DATA)) sel.add(new Option(k, k));
sel.onchange = () => load(sel.value);
const parts = document.getElementById("parts");
for (const [part, m] of Object.entries(Object.values(DATA)[0])) {
  const l = document.createElement("label"); l.innerHTML = `<input type="checkbox" checked> <span class="sw" style="background:${m.color}"></span>${part}`;
  l.querySelector("input").onchange = e => { visible[part] = e.target.checked; group.getObjectByName(part).visible = e.target.checked; };
  parts.appendChild(l);
}
function resize() { renderer.setSize(innerWidth, innerHeight); camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); }
addEventListener("resize", resize); matchMedia("(prefers-color-scheme: dark)").addEventListener("change", bg);
resize(); bg(); load(sel.value);
renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });
</script></body></html>
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=HERE / "cad" / "strato_3d.html")
    ap.add_argument("--lengths", type=float, nargs="+", default=[15.0, 30.0, 50.0, 80.0])
    a = ap.parse_args(argv)
    p = build(a.lengths, a.out)
    print(f"{p} ({p.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
