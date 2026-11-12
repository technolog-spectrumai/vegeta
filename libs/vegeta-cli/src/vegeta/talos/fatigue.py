"""Fatigue damage from linear unit-load stress fields and a load spectrum (stress-life, Miner's rule).

The structure is linear: the stress tensor under any load level of a pattern is that level times the
tensor of the unit case. For every node and every spectrum block the block's amplitude and mean are
scaled by the node's unit stress, an S-N curve (Basquin, optional endurance limit, Goodman mean
correction) gives cycles to failure, and damage adds up linearly. Patterns are treated as acting
one at a time (no phase between them), which is stated in the result's messages.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ._process import utc_now
from .frd import read_frd, von_mises
from .result import Result


@dataclass(frozen=True)
class FatigueCurve:
    """Stress-life curve: amplitude S_a = sigma_f * (2N)^b (Basquin), in the model's stress unit.

    ``ultimate`` enables the Goodman mean-stress correction; ``endurance_limit`` is an amplitude below
    which no damage is counted (None: every cycle counts). Values are engineer inputs, ideally from
    coupon tests of the printed/machined material in the build orientation used.
    """

    name: str
    sigma_f: float
    b: float
    ultimate: float | None = None
    endurance_limit: float | None = None
    source: str = ""

    def __post_init__(self):
        if self.sigma_f <= 0 or not (-0.5 < self.b < 0):
            raise ValueError("sigma_f must be > 0 and b in (-0.5, 0) (typically -0.05 .. -0.15)")
        if self.ultimate is not None and self.ultimate <= 0:
            raise ValueError("ultimate must be > 0")

    def cycles_to_failure(self, amplitude, mean=0.0):
        """Cycles to failure (array-safe; ``inf`` below the endurance limit or at zero amplitude)."""
        a = np.abs(np.asarray(amplitude, dtype=float))
        m = np.asarray(mean, dtype=float)
        if self.ultimate is not None:
            factor = np.clip(1.0 - np.clip(m, 0.0, None) / self.ultimate, 1e-6, 1.0)   # Goodman (tension only)
            a = a / factor
        with np.errstate(divide="ignore", over="ignore"):
            n = 0.5 * (a / self.sigma_f) ** (1.0 / self.b)
        n = np.where(a <= 0, np.inf, n)
        if self.endurance_limit is not None:
            n = np.where(a < self.endurance_limit, np.inf, n)
        return n


@dataclass
class FatigueResult:
    """Per-node damage for one spectrum (one mission) plus the derived life."""

    node_ids: np.ndarray
    coords: np.ndarray
    damage: np.ndarray                 # per node, per pass of the spectrum
    contributions: dict                # block source -> damage at the hotspot
    result: Result

    @property
    def hotspot(self) -> int:
        return int(np.nanargmax(self.damage))


def _unit_tensor(case, load_value: float):
    """Signed stress-per-unit-load field ``(node_ids, coords, tensor/load, signed_vm/load)``."""
    frd = case.artifacts["frd"] if hasattr(case, "artifacts") else Path(case)
    fr = read_frd(frd)
    if "STRESS" not in fr.fields:
        raise ValueError(f"{frd}: no stress field")
    if load_value == 0:
        raise ValueError("load_value of a unit case must be non-zero")
    t = np.nan_to_num(fr.stress) / load_value
    vm = von_mises(t)
    sign = np.where(t[:, :3].sum(axis=1) < 0, -1.0, 1.0)     # sign of the hydrostatic part
    return fr.node_ids, fr.coords, t, vm * sign


def assess(unit_cases: dict, spectrum, curve: FatigueCurve, *, workdir: str | Path | None = None) -> FatigueResult:
    """Damage per node for one pass of ``spectrum`` (a ``vegeta.chronos`` spectrum dict or JSON path).

    ``unit_cases`` maps pattern name -> ``(solve result or .frd path, load value used in that run)``.
    Every pattern in the spectrum must have a unit case. Returns a ``FatigueResult`` whose ``.result``
    is the usual Result (metrics: damage per pass, passes to failure, hotspot).
    """
    t0 = time.monotonic()
    spec = json.loads(Path(spectrum).read_text()) if isinstance(spectrum, (str, Path)) else dict(spectrum)
    res = Result(kind="talos.fatigue", metadata={"curve": asdict(curve), "spectrum": spec.get("mission"),
                                                 "started_at": utc_now(), "patterns": sorted(unit_cases)})
    blocks = spec.get("blocks", [])
    missing = sorted({b["pattern"] for b in blocks} - set(unit_cases))
    if missing:
        res.fail(f"spectrum uses patterns without a unit case: {missing}")
        return FatigueResult(np.array([]), np.array([]), np.array([]), {}, res)
    fields = {name: _unit_tensor(case, value) for name, (case, value) in unit_cases.items()}
    ids0, coords0 = next(iter(fields.values()))[:2]
    for name, (ids, *_rest) in fields.items():
        if not np.array_equal(ids, ids0):
            res.fail(f"unit case {name!r} is on a different mesh than the others")
            return FatigueResult(np.array([]), np.array([]), np.array([]), {}, res)
    damage = np.zeros(len(ids0))
    per_block = {}
    for blk in blocks:
        svm = fields[blk["pattern"]][3]
        amp = np.abs(blk["amplitude"] * svm)
        mean = blk["mean"] * svm
        d = blk["cycles"] / curve.cycles_to_failure(amp, mean)
        d = np.nan_to_num(d, nan=0.0, posinf=np.inf)
        damage += d
        per_block[blk.get("source", blk["pattern"])] = d
    hot = int(np.nanargmax(damage)) if len(damage) else 0
    d_pass = float(damage[hot]) if len(damage) else 0.0
    contributions = {k: float(v[hot]) for k, v in per_block.items()}
    passes = (1.0 / d_pass) if d_pass > 0 else float("inf")
    hours = spec.get("duration_s", 0.0) / 3600.0
    res.metrics = {"damage_per_pass": d_pass, "passes_to_failure": passes,
                   "hours_to_failure": passes * hours if np.isfinite(passes) else float("inf"),
                   "spectrum_duration_h": hours, "blocks": len(blocks),
                   "hotspot_node": int(ids0[hot]) if len(ids0) else None,
                   "hotspot_location": coords0[hot].tolist() if len(ids0) else None,
                   "nodes_above_1pct": int(np.sum(damage > 0.01 * d_pass)) if d_pass > 0 else 0}
    res.messages += [
        "linear superposition: stress at any load level = level x unit-case stress; patterns act one at a time",
        "damage is evaluated on nodal (extrapolated, averaged) stresses: the hotspot at a bolt hole or a "
        "sharp corner is mesh-dependent; compare designs, and validate the absolute life with a test",
    ]
    if workdir is not None:
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(workdir / "damage.npz", node_ids=ids0, coords=coords0, damage=damage)
        res.artifacts["damage"] = workdir / "damage.npz"
        res.artifacts["summary"] = res.save_json(workdir / "fatigue_summary.json")
    res.duration_s = time.monotonic() - t0
    return FatigueResult(ids0, coords0, damage, contributions, res)
