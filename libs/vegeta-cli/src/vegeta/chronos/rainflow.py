"""Rainflow cycle counting (ASTM E1049, three-point method with residue counted as half cycles)."""
from __future__ import annotations

import numpy as np


def reversals(series) -> np.ndarray:
    """Peaks and valleys of a series (first and last point kept)."""
    x = np.asarray(series, dtype=float)
    if len(x) < 2:
        return x
    keep = [0]
    for i in range(1, len(x) - 1):
        if (x[i] - x[i - 1]) * (x[i + 1] - x[i]) < 0:
            keep.append(i)
        elif x[i] != x[i - 1] and x[i + 1] == x[i]:
            keep.append(i)
    keep.append(len(x) - 1)
    r = x[keep]
    d = np.diff(r)
    return r[np.concatenate([[True], d != 0])] if len(r) > 1 else r


def rainflow(series) -> list[tuple[float, float, float]]:
    """``[(range, mean, count), ...]`` with count 1.0 for closed cycles and 0.5 for the residue."""
    pts = list(reversals(series))
    out: list[tuple[float, float, float]] = []
    stack: list[float] = []
    for p in pts:
        stack.append(p)
        while len(stack) >= 3:
            x = abs(stack[-1] - stack[-2])
            y = abs(stack[-2] - stack[-3])
            if x < y:
                break
            if len(stack) == 3:                       # residue at the start: half cycle
                out.append((y, 0.5 * (stack[0] + stack[1]), 0.5))
                stack.pop(0)
            else:
                out.append((y, 0.5 * (stack[-2] + stack[-3]), 1.0))
                del stack[-3:-1]
    for a, b in zip(stack, stack[1:]):
        out.append((abs(b - a), 0.5 * (a + b), 0.5))
    return [c for c in out if c[0] > 0]
