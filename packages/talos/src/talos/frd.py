"""Parsers for CalculiX ASCII ``.frd`` results and ``.dat`` printouts."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class FieldResults:
    """Nodal results read from an ``.frd`` file (last increment of each field)."""

    node_ids: np.ndarray                   # (N,)
    coords: np.ndarray                     # (N, 3) undeformed
    fields: dict[str, np.ndarray] = field(default_factory=dict)  # name -> (N, k), aligned with node_ids

    @property
    def displacement(self) -> np.ndarray:
        return self.fields["DISP"][:, :3]

    @property
    def stress(self) -> np.ndarray:
        """(N, 6): SXX SYY SZZ SXY SYZ SZX."""
        return self.fields["STRESS"][:, :6]

    @property
    def displacement_magnitude(self) -> np.ndarray:
        return np.linalg.norm(self.displacement, axis=1)

    @property
    def von_mises(self) -> np.ndarray:
        return von_mises(self.stress)


def von_mises(s: np.ndarray) -> np.ndarray:
    """Von Mises stress from (N, 6) = xx, yy, zz, xy, yz, zx."""
    s = np.atleast_2d(s)
    xx, yy, zz, xy, yz, zx = (s[:, i] for i in range(6))
    return np.sqrt(0.5 * ((xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2) + 3.0 * (xy ** 2 + yz ** 2 + zx ** 2))


def _floats(text: str, width: int = 12) -> list[float]:
    """Fixed-width E12.5 fields; falls back to whitespace splitting."""
    text = text.rstrip("\n")
    vals = []
    for i in range(0, len(text), width):
        chunk = text[i:i + width].strip()
        if chunk:
            vals.append(float(chunk))
    return vals


def read_frd(path: str | Path) -> FieldResults:
    path = Path(path)
    nodes: dict[int, list[float]] = {}
    fields: dict[str, dict[int, list[float]]] = {}
    with path.open("r", errors="replace") as fh:
        lines = iter(fh)
        for line in lines:
            if line.startswith("    2C"):
                for rec in lines:
                    if rec.startswith(" -3"):
                        break
                    if rec.startswith(" -1"):
                        nid = int(rec[3:13])
                        nodes[nid] = _floats(rec[13:])
            elif line.startswith("  100C"):
                name = None
                data: dict[int, list[float]] = {}
                for rec in lines:
                    if rec.startswith(" -4"):
                        name = rec.split()[1]
                    elif rec.startswith(" -1"):
                        data[int(rec[3:13])] = _floats(rec[13:])
                    elif rec.startswith(" -2"):  # continuation of the previous node
                        last = next(reversed(data))
                        data[last].extend(_floats(rec[13:]))
                    elif rec.startswith(" -3"):
                        break
                if name:
                    fields[name] = data  # later increments overwrite earlier ones
    if not nodes:
        raise ValueError(f"{path}: no node block found")
    ids = np.array(sorted(nodes), dtype=np.int64)
    coords = np.array([nodes[i][:3] for i in ids])
    out = {}
    for name, data in fields.items():
        width = max(len(v) for v in data.values())
        arr = np.full((len(ids), width), np.nan)
        pos = {n: i for i, n in enumerate(ids)}
        for n, v in data.items():
            if n in pos:
                arr[pos[n], :len(v)] = v
        out[name] = arr
    return FieldResults(node_ids=ids, coords=coords, fields=out)


_TOTAL = re.compile(r"total force \(fx,fy,fz\) for set (\S+) and time\s+(\S+)", re.I)


def read_dat_reactions(path: str | Path) -> dict[str, list[float]]:
    """Total reaction force per node set (``*NODE PRINT, TOTALS=YES``); last time wins."""
    text = Path(path).read_text(errors="replace").splitlines()
    out: dict[str, list[float]] = {}
    for i, line in enumerate(text):
        m = _TOTAL.search(line)
        if not m:
            continue
        for nxt in text[i + 1:i + 5]:
            parts = nxt.split()
            if len(parts) == 3:
                try:
                    out[m.group(1).upper()] = [float(p) for p in parts]
                    break
                except ValueError:
                    continue
    return out
