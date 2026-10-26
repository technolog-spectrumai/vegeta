"""VTK ImageData (``.vti``) export for ParaView.

The built-in writer needs only NumPy (XML, inline base64 binary, point data at the lattice nodes).
:func:`to_pyvista` builds a ``pyvista.ImageData`` when PyVista is installed (the ``vtk`` extra).
"""
from __future__ import annotations

import base64
from pathlib import Path

import numpy as np


def _vtk_order(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a).ravel(order="F")       # VTK runs x fastest; our arrays run z fastest


def _encode(data: np.ndarray) -> str:
    raw = np.ascontiguousarray(data).tobytes()
    header = np.array([len(raw)], dtype=np.uint64).tobytes()
    return base64.b64encode(header).decode() + base64.b64encode(raw).decode()


_TYPES = {np.dtype(np.float32): "Float32", np.dtype(np.float64): "Float64", np.dtype(np.uint8): "UInt8"}


def write_vti(path: str | Path, scalars: dict[str, np.ndarray], vectors: dict[str, np.ndarray] | None = None,
              spacing: float = 1.0, origin: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Path:
    """Write point fields: scalars ``[nx, ny, nz]``, vectors ``[3, nx, ny, nz]``."""
    path = Path(path)
    vectors = vectors or {}
    shape = next(iter(scalars.values())).shape if scalars else next(iter(vectors.values())).shape[1:]
    extent = f"0 {shape[0] - 1} 0 {shape[1] - 1} 0 {shape[2] - 1}"
    arrays = []
    for name, a in scalars.items():
        a = np.asarray(a)
        a = a.astype(np.uint8) if a.dtype == bool else a
        arrays.append(f'<DataArray type="{_TYPES[a.dtype]}" Name="{name}" format="binary">'
                      f"{_encode(_vtk_order(a))}</DataArray>")
    for name, v in vectors.items():
        v = np.asarray(v)
        flat = np.stack([_vtk_order(v[k]) for k in range(3)], axis=1)
        arrays.append(f'<DataArray type="{_TYPES[flat.dtype]}" Name="{name}" NumberOfComponents="3" '
                      f'format="binary">{_encode(flat)}</DataArray>')
    xml = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian" header_type="UInt64">\n'
        f'<ImageData WholeExtent="{extent}" Origin="{origin[0]} {origin[1]} {origin[2]}" '
        f'Spacing="{spacing} {spacing} {spacing}">\n'
        f'<Piece Extent="{extent}">\n<PointData>\n' + "\n".join(arrays) + "\n</PointData>\n</Piece>\n"
        "</ImageData>\n</VTKFile>\n")
    path.write_text(xml)
    return path


def to_pyvista(scalars: dict[str, np.ndarray], vectors: dict[str, np.ndarray] | None = None,
               spacing: float = 1.0):
    """A ``pyvista.ImageData`` with the given point fields (needs PyVista)."""
    import pyvista as pv

    vectors = vectors or {}
    shape = next(iter(scalars.values())).shape if scalars else next(iter(vectors.values())).shape[1:]
    grid = pv.ImageData(dimensions=shape, spacing=(spacing,) * 3)
    for name, a in scalars.items():
        grid.point_data[name] = _vtk_order(np.asarray(a).astype(np.uint8) if a.dtype == bool else a)
    for name, v in vectors.items():
        grid.point_data[name] = np.stack([_vtk_order(v[k]) for k in range(3)], axis=1)
    return grid
