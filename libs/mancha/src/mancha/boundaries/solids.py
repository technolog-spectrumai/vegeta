"""Static solid obstacles as voxel masks (True = solid). Bounce-back puts the wall halfway between a fluid node
and a solid voxel, so a curved body becomes a staircase; its effective size differs from the nominal one by up
to half a cell."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..errors import InvalidParameterError


def _grid(shape: Sequence[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.meshgrid(*(np.arange(n, dtype=np.float64) for n in shape), indexing="ij", sparse=True)


def sphere(shape: Sequence[int], center: Sequence[float], radius: float) -> np.ndarray:
    """Voxels whose centres (integer coordinates) lie within ``radius`` of ``center`` (lattice units)."""
    x, y, z = _grid(shape)
    return (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2 <= radius ** 2


def box(shape: Sequence[int], lower: Sequence[int], upper: Sequence[int]) -> np.ndarray:
    """Voxels with ``lower <= index < upper`` on every axis."""
    mask = np.zeros(tuple(shape), dtype=bool)
    mask[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]] = True
    return mask


def validate_mask(mask: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.shape != shape:
        raise InvalidParameterError(f"the solid mask has shape {mask.shape}, the grid {shape}")
    if mask.dtype != bool:
        if not np.isin(mask, (0, 1)).all():
            raise InvalidParameterError("the solid mask must be boolean (or 0/1)")
        mask = mask.astype(bool)
    if mask.all():
        raise InvalidParameterError("the solid mask covers the whole domain")
    return mask
