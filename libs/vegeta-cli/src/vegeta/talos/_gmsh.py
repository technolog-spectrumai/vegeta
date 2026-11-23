"""Gmsh session handling and STEP import shared by inspection and meshing."""
from __future__ import annotations

import contextlib
import math
from pathlib import Path

from .units import UnitSystem


def gmsh_module():
    try:
        import gmsh
    except (ImportError, OSError) as exc:  # OSError: missing shared libraries such as libGLU
        raise RuntimeError(f"gmsh is not usable: {exc}") from exc
    return gmsh


@contextlib.contextmanager
def gmsh_session(log_file: Path | None = None):
    """Initialise Gmsh quietly, collect its log, always finalise."""
    gmsh = gmsh_module()
    if gmsh.isInitialized():
        gmsh.finalize()
    gmsh.initialize(readConfigFiles=False)
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.logger.start()
    try:
        yield gmsh
    finally:
        try:
            if log_file is not None:
                Path(log_file).write_text("\n".join(gmsh.logger.get()) + "\n")
            gmsh.logger.stop()
        finally:
            gmsh.finalize()


def import_step(gmsh, path: str | Path, units: UnitSystem) -> dict:
    """Import a STEP file into the current Gmsh model and return basic facts about it.

    Multiple volumes are fragmented so they share interfaces (conformal mesh). Inspection and
    meshing both go through this function, so surface tags always agree.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"STEP file not found: {path}")
    gmsh.option.setString("Geometry.OCCTargetUnit", units.occ_unit)
    gmsh.model.add(path.stem)
    gmsh.model.occ.importShapes(str(path))
    gmsh.model.occ.synchronize()
    vols = gmsh.model.getEntities(3)
    if not vols:
        raise ValueError(f"{path} contains no solid volumes; Talos needs closed solids")
    if len(vols) > 1:
        gmsh.model.occ.fragment(vols[:1], vols[1:])
        gmsh.model.occ.synchronize()
        vols = gmsh.model.getEntities(3)
    bb = gmsh.model.getBoundingBox(-1, -1)
    diag = math.dist(bb[:3], bb[3:])
    return {"volumes": [t for _, t in vols], "bbox": list(bb), "diag": diag}
