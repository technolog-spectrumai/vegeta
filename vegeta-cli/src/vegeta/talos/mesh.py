"""Meshing with Gmsh and reading the mesh back for CalculiX."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._gmsh import gmsh_session, import_step

# gmsh element type -> (CalculiX type, node count, reorder gmsh->ccx)
VOLUME_TYPES = {
    4: ("C3D4", 4, [0, 1, 2, 3]),
    11: ("C3D10", 10, [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]),
}
# gmsh triangle types -> node count
SURFACE_TYPES = {2: 3, 9: 6}
# CalculiX tetrahedron faces (0-based corner indices) -> face label S1..S4
TET_FACES = {1: (0, 1, 2), 2: (0, 3, 1), 3: (1, 3, 2), 4: (2, 3, 0)}


@dataclass(frozen=True)
class MeshSettings:
    """Explicit meshing choices. ``element_size`` is in model length units."""

    element_size: float
    order: int = 2
    min_element_size: float | None = None
    optimize: bool = True
    algorithm_3d: int = 1  # Gmsh 3D algorithm: 1 Delaunay, 4 Frontal, 10 HXT
    high_order_optimize: int = 1  # order 2 only: 0 off, 1 optimise, 2 elastic + optimise (fixes curved-node inversions)

    def __post_init__(self):
        if self.element_size <= 0:
            raise ValueError("element_size must be > 0")
        if self.order not in (1, 2):
            raise ValueError("order must be 1 (C3D4) or 2 (C3D10); use 2 for bending")
        if self.high_order_optimize not in (0, 1, 2, 3, 4):
            raise ValueError("high_order_optimize must be a Gmsh Mesh.HighOrderOptimize value 0-4")


@dataclass
class MeshData:
    """Mesh read back from ``mesh.msh`` in CalculiX conventions."""

    node_ids: np.ndarray          # (N,)
    coords: np.ndarray            # (N, 3)
    element_type: str             # C3D4 or C3D10
    element_ids: np.ndarray       # (E,)
    connectivity: np.ndarray      # (E, n) node ids in CalculiX order
    region_nodes: dict[str, np.ndarray]
    region_triangles: dict[str, np.ndarray]  # (T, 3|6) node ids, gmsh triangle order

    def coord_map(self) -> dict[int, np.ndarray]:
        return {int(n): self.coords[i] for i, n in enumerate(self.node_ids)}


def generate_mesh(geometry: Path, units, regions, settings: MeshSettings, msh_path: Path, log_file: Path,
                  progress=None) -> dict:
    """Import, tag regions as physical groups, mesh and write ``msh_path``. Returns statistics."""
    progress = progress or (lambda *a, **k: None)
    with gmsh_session(log_file) as gmsh:
        progress("import STEP", 0.05)
        info = import_step(gmsh, geometry, units)
        model = gmsh.model
        region_tags = {}
        for region in regions:
            tags = region.select(model, info["diag"])
            if not tags:
                raise ValueError(f"region {region.name!r} selects no surfaces; check it with talos.inspect_step")
            region_tags[region.name] = tags
            pg = model.addPhysicalGroup(2, tags)
            model.setPhysicalName(2, pg, region.name)
        pv = model.addPhysicalGroup(3, info["volumes"])
        model.setPhysicalName(3, pv, "SOLID")

        opt = gmsh.option
        opt.setNumber("Mesh.MeshSizeMax", settings.element_size)
        opt.setNumber("Mesh.MeshSizeMin", settings.min_element_size or 0.0)
        opt.setNumber("Mesh.ElementOrder", settings.order)
        opt.setNumber("Mesh.Optimize", 1 if settings.optimize else 0)
        opt.setNumber("Mesh.Algorithm3D", settings.algorithm_3d)
        opt.setNumber("Mesh.HighOrderOptimize", settings.high_order_optimize if settings.order == 2 else 0)
        opt.setNumber("Mesh.SaveAll", 0)
        opt.setNumber("Mesh.MshFileVersion", 4.1)
        progress("mesh 2D", 0.2)
        model.mesh.generate(2)
        progress("mesh 3D", 0.4)
        model.mesh.generate(3)  # Mesh.ElementOrder makes this second order when requested
        progress("write mesh", 0.9)
        gmsh.write(str(msh_path))

        etypes, etags, _ = model.mesh.getElements(3)
        n_elem = int(sum(len(t) for t in etags))
        qual = model.mesh.getElementQualities(np.concatenate(etags).tolist(), "minSICN") if n_elem else []
        node_tags, _, _ = model.mesh.getNodes()
        stats = {
            "n_nodes": int(len(node_tags)),
            "n_elements": n_elem,
            "element_type": VOLUME_TYPES.get(int(etypes[0]), ("?",))[0] if len(etypes) else None,
            "min_quality_sicn": float(np.min(qual)) if len(qual) else None,
            "mean_quality_sicn": float(np.mean(qual)) if len(qual) else None,
            "region_surfaces": region_tags,
            "bbox": info["bbox"],
            "gmsh_version": gmsh.__version__,
        }
    return stats


def read_mesh(msh_path: Path) -> MeshData:
    """Read ``mesh.msh`` written by :func:`generate_mesh`."""
    with gmsh_session() as gmsh:
        gmsh.open(str(msh_path))
        model = gmsh.model
        tags, coords, _ = model.mesh.getNodes()
        coords = np.asarray(coords).reshape(-1, 3)
        all_pos = {int(t): i for i, t in enumerate(tags)}

        etypes, etags, enodes = model.mesh.getElements(3)
        if len(etypes) != 1 or int(etypes[0]) not in VOLUME_TYPES:
            raise ValueError(f"unsupported volume element types {list(etypes)}; expected tetrahedra only")
        ccx_type, nn, order = VOLUME_TYPES[int(etypes[0])]
        conn = np.asarray(enodes[0], dtype=np.int64).reshape(-1, nn)[:, order]
        used = np.unique(conn)
        idx = np.array([all_pos[int(n)] for n in used])

        region_nodes, region_tris = {}, {}
        for dim, pg in model.getPhysicalGroups(2):
            name = model.getPhysicalName(dim, pg)
            region_nodes[name] = np.asarray(model.mesh.getNodesForPhysicalGroup(dim, pg)[0], dtype=np.int64)
            tris = []
            for ent in model.getEntitiesForPhysicalGroup(dim, pg):
                st, _, sn = model.mesh.getElements(2, ent)
                for t, nodes in zip(st, sn):
                    if int(t) not in SURFACE_TYPES:
                        raise ValueError(f"unsupported surface element type {t}")
                    tris.append(np.asarray(nodes, dtype=np.int64).reshape(-1, SURFACE_TYPES[int(t)]))
            region_tris[name] = np.vstack(tris) if tris else np.zeros((0, 3), dtype=np.int64)
        return MeshData(
            node_ids=used, coords=coords[idx], element_type=ccx_type,
            element_ids=np.asarray(etags[0], dtype=np.int64), connectivity=conn,
            region_nodes=region_nodes, region_triangles=region_tris,
        )


def triangle_areas(tris: np.ndarray, cmap: dict[int, np.ndarray]) -> np.ndarray:
    a = np.array([cmap[int(n)] for n in tris[:, 0]])
    b = np.array([cmap[int(n)] for n in tris[:, 1]])
    c = np.array([cmap[int(n)] for n in tris[:, 2]])
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def consistent_nodal_forces(tris: np.ndarray, cmap: dict[int, np.ndarray], total: np.ndarray) -> dict[int, np.ndarray]:
    """Distribute a total force as uniform traction over triangles (consistent nodal loads).

    Linear triangles: A/3 per corner. Quadratic triangles: 0 at corners, A/3 per mid-side node.
    The distributed forces sum exactly to ``total``.
    """
    areas = triangle_areas(tris, cmap)
    atot = areas.sum()
    if atot <= 0:
        raise ValueError("load region has zero area")
    weights: dict[int, float] = {}
    carriers = tris[:, :3] if tris.shape[1] == 3 else tris[:, 3:6]
    for row, a in zip(carriers, areas):
        for n in row:
            weights[int(n)] = weights.get(int(n), 0.0) + a / 3.0
    return {n: np.asarray(total, dtype=float) * w / atot for n, w in weights.items()}


def element_faces(mesh: MeshData, tris: np.ndarray) -> list[tuple[int, int]]:
    """Map surface triangles to (element id, face number 1..4) of the adjacent tetrahedron."""
    lookup: dict[tuple, tuple[int, int]] = {}
    corners = mesh.connectivity[:, :4]
    for eid, c in zip(mesh.element_ids, corners):
        for face, (i, j, k) in TET_FACES.items():
            lookup[tuple(sorted((int(c[i]), int(c[j]), int(c[k]))))] = (int(eid), face)
    out = []
    for t in tris[:, :3]:
        key = tuple(sorted(int(n) for n in t))
        if key not in lookup:
            raise ValueError("surface triangle without adjacent tetrahedron (non-conformal mesh?)")
        out.append(lookup[key])
    return out
