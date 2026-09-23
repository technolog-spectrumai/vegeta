"""Parsers for OpenFOAM outputs: forceCoeffs, forces, solver residuals, checkMesh."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

COEFF_FILES = ("coefficient.dat", "forceCoeffs.dat")
ALIASES = {"CmPitch": "Cm"}


@dataclass
class CoefficientHistory:
    columns: list[str]
    data: np.ndarray  # (n, len(columns))
    files: list[Path]

    def __contains__(self, name: str) -> bool:
        return name in self.columns

    def __getitem__(self, name: str) -> np.ndarray:
        return self.data[:, self.columns.index(name)]

    @property
    def iterations(self) -> np.ndarray:
        return self["Time"]

    def to_dataframe(self):
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover
            raise ImportError("to_dataframe needs pandas: pip install 'vegeta-cli[pandas]'") from exc
        return pd.DataFrame(self.data, columns=self.columns)


def find_coefficient_files(case: Path) -> list[Path]:
    pp = Path(case) / "postProcessing"
    files = []
    for name in COEFF_FILES:
        files += pp.glob(f"*/*/{name}")
    # order by the start-time directory so restarted runs concatenate correctly
    return sorted(set(files), key=lambda p: (p.parent.parent.name, _time_key(p.parent.name)))


def _time_key(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        return float("inf")


def read_coefficients(files: list[Path] | Path) -> CoefficientHistory:
    files = [files] if isinstance(files, (str, Path)) else list(files)
    columns: list[str] | None = None
    rows: list[list[float]] = []
    for f in files:
        header = None
        for line in Path(f).read_text(errors="replace").splitlines():
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                parts = s.lstrip("#").split()
                if parts and parts[0] == "Time":
                    header = [ALIASES.get(p, p) for p in parts]
                continue
            if header is None:
                raise ValueError(f"{f}: data before a '# Time ...' header")
            try:
                vals = [float(v) for v in s.split()]
            except ValueError:
                continue
            if len(vals) == len(header):
                rows.append(vals)
        if header is None:
            raise ValueError(f"{f}: no '# Time ...' header")
        if columns is not None and header != columns:
            raise ValueError(f"inconsistent coefficient columns between files: {columns} vs {header}")
        columns = header
    if columns is None:
        raise ValueError("no coefficient files")
    data = np.array(rows, dtype=float).reshape(-1, len(columns))
    # keep the last value written for each iteration (restarts may overlap)
    _, last = np.unique(data[:, 0][::-1], return_index=True)
    data = data[::-1][last]
    return CoefficientHistory(columns, data[np.argsort(data[:, 0])], [Path(f) for f in files])


_RESIDUAL = re.compile(r"Solving for (\w+), Initial residual = ([-+\d.eE]+)")
_TIME = re.compile(r"^Time = ([-+\d.eE]+)\s*s?\s*$")   # openfoam.org prints "Time = 1s"


@dataclass
class SolverLog:
    iterations: int
    converged: bool
    residuals: dict[str, np.ndarray]  # field -> initial residual per iteration (first solve in the iteration)
    completed: bool                    # the solver reached "End"
    version: str | None


def read_solver_log(path: str | Path) -> SolverLog:
    text = Path(path).read_text(errors="replace").splitlines()
    res: dict[str, list[float]] = {}
    iteration = 0
    seen: set[str] = set()
    for line in text:
        m = _TIME.match(line.strip())
        if m:
            iteration += 1
            seen = set()
            continue
        for fld, val in _RESIDUAL.findall(line):
            if fld not in seen:
                seen.add(fld)
                res.setdefault(fld, []).append(float(val))
    joined = "\n".join(text)
    return SolverLog(
        iterations=iteration,
        converged="solution converged in" in joined,
        residuals={k: np.array(v) for k, v in res.items()},
        completed=bool(re.search(r"^End\s*$", joined, re.M)),
        version=read_version(joined),
    )


def read_version(log_text: str) -> str | None:
    for pat in (r"^Build\s*:\s*(.+)$", r"^\|\s*Version:\s*(\S+)"):
        m = re.search(pat, log_text, re.M)
        if m:
            return m.group(1).strip()
    return None


@dataclass
class MeshCheck:
    cells: int | None
    ok: bool
    failed_checks: int
    messages: list[str]


def read_checkmesh(path: str | Path) -> MeshCheck:
    text = Path(path).read_text(errors="replace")
    m = re.search(r"^\s*cells:\s+(\d+)", text, re.M)
    failed = re.search(r"Failed (\d+) mesh checks", text)
    msgs = [l.strip() for l in text.splitlines() if l.strip().startswith("***")]
    return MeshCheck(
        cells=int(m.group(1)) if m else None,
        ok="Mesh OK." in text and not failed,
        failed_checks=int(failed.group(1)) if failed else 0,
        messages=msgs,
    )


# -- forces function object (force.dat / moment.dat) ------------------------------------------------
_NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def find_force_files(case: Path, name: str = "force.dat") -> list[Path]:
    pp = Path(case) / "postProcessing"
    return sorted(set(pp.glob(f"*/*/{name}")), key=lambda p: (p.parent.parent.name, _time_key(p.parent.name)))


def read_force_history(files: list[Path] | Path) -> np.ndarray:
    """``(n, 4)`` array ``time, x, y, z`` of the TOTAL force (or moment) from ``force.dat``/``moment.dat``.

    Both OpenFOAM forks write ``time`` followed by the total vector, then the pressure and viscous parts,
    with or without parentheses; only the first three components after the time are used."""
    files = [files] if isinstance(files, (str, Path)) else list(files)
    rows = []
    for f in files:
        for line in Path(f).read_text(errors="replace").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            nums = [float(x) for x in _NUM.findall(line)]
            if len(nums) >= 4:
                rows.append(nums[:4])
    if not rows:
        raise ValueError("no force data found")
    return np.array(rows, dtype=float)
