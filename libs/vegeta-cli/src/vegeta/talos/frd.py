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


def _read_blocks(path: Path):
    """Nodes and the list of result blocks ``(step_value, name, {node: values})`` in file order."""
    nodes: dict[int, list[float]] = {}
    blocks: list[tuple[float, str, dict[int, list[float]]]] = []
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
                try:
                    step_value = float(line[12:24])
                except ValueError:
                    step_value = float("nan")
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
                    blocks.append((step_value, name, data))
    if not nodes:
        raise ValueError(f"{path}: no node block found")
    return nodes, blocks


def _assemble(nodes, fields) -> FieldResults:
    ids = np.array(sorted(nodes), dtype=np.int64)
    coords = np.array([nodes[i][:3] for i in ids])
    out = {}
    pos = {n: i for i, n in enumerate(ids)}
    for name, data in fields.items():
        width = max(len(v) for v in data.values())
        arr = np.full((len(ids), width), np.nan)
        for n, v in data.items():
            if n in pos:
                arr[pos[n], :len(v)] = v
        out[name] = arr
    return FieldResults(node_ids=ids, coords=coords, fields=out)


def read_frd(path: str | Path) -> FieldResults:
    """Last increment of every field."""
    nodes, blocks = _read_blocks(Path(path))
    fields: dict[str, dict[int, list[float]]] = {}
    for _, name, data in blocks:
        fields[name] = data  # later increments overwrite earlier ones
    return _assemble(nodes, fields)


def read_frd_steps(path: str | Path) -> list[tuple[float, FieldResults]]:
    """Every increment (a mode in a ``*FREQUENCY`` job) as ``(step_value, FieldResults)``."""
    nodes, blocks = _read_blocks(Path(path))
    steps: list[tuple[float, dict]] = []
    for value, name, data in blocks:
        if not steps or steps[-1][0] != value or name in steps[-1][1]:
            steps.append((value, {}))
        steps[-1][1][name] = data
    return [(v, _assemble(nodes, f)) for v, f in steps]


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


def read_dat_eigen(path: str | Path) -> dict:
    """Eigenvalues (``*FREQUENCY``): frequencies in Hz plus, when printed, participation factors and
    effective modal masses per mode (x, y, z, rx, ry, rz)."""
    text = Path(path).read_text(errors="replace")
    out: dict = {"frequencies_hz": [], "participation": [], "effective_modal_mass": [], "total_effective_mass": None}
    m = re.search(r"E I G E N V A L U E   O U T P U T(.*?)(?:\n\s*\n\s*\n|P A R T I C I P A T I O N)", text, re.S)
    if m:
        for line in m.group(1).splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[0].isdigit():
                out["frequencies_hz"].append(float(parts[3]))
    for key, title in (("participation", "P A R T I C I P A T I O N   F A C T O R S"),
                       ("effective_modal_mass", "E F F E C T I V E   M O D A L   M A S S")):
        m = re.search(re.escape(title) + r"(.*?)(?:\n\s*\n\s*\n|T O T A L|E F F E C T I V E|\Z)", text, re.S)
        if m:
            for line in m.group(1).splitlines():
                parts = line.split()
                if len(parts) == 7 and parts[0].isdigit():
                    out[key].append([float(x) for x in parts[1:]])
    m = re.search(r"T O T A L   E F F E C T I V E   M A S S(.*?)\Z", text, re.S)
    if m:
        for line in m.group(1).splitlines():
            parts = line.split()
            if len(parts) == 6:
                try:
                    out["total_effective_mass"] = [float(x) for x in parts]
                    break
                except ValueError:
                    continue
    return out
