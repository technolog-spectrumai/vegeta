"""Turn a mission into a load spectrum: blocks of (pattern, mean, amplitude, cycles)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .dynamics import Structure
from .mission import Mission
from .rainflow import rainflow


@dataclass(frozen=True)
class Block:
    pattern: str
    mean: float
    amplitude: float
    cycles: float
    source: str = ""


@dataclass
class LoadSpectrum:
    mission: str
    duration_s: float
    blocks: list = field(default_factory=list)
    patterns: list = field(default_factory=list)
    notes: str = ""
    structure: dict | None = None

    def to_dict(self) -> dict:
        return {"tool": "vegeta.chronos", "mission": self.mission, "duration_s": self.duration_s,
                "patterns": list(self.patterns), "blocks": [asdict(b) for b in self.blocks], "notes": self.notes,
                "structure": self.structure, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "LoadSpectrum":
        d = json.loads(Path(path).read_text())
        return cls(d["mission"], d["duration_s"], [Block(**b) for b in d["blocks"]], d.get("patterns", []),
                   d.get("notes", ""), d.get("structure"))

    @property
    def total_cycles(self) -> float:
        return float(sum(b.cycles for b in self.blocks))

    def table(self):
        rows = [asdict(b) for b in self.blocks]
        try:
            import pandas as pd

            return pd.DataFrame(rows)
        except ImportError:  # pragma: no cover
            return rows

    def plot(self, ax=None):
        """Amplitude vs cycles per block, one marker per pattern (log x)."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 3.6))
        for p in self.patterns:
            bl = [b for b in self.blocks if b.pattern == p]
            if bl:
                ax.scatter([b.cycles for b in bl], [b.amplitude for b in bl], s=30, alpha=0.8, label=p)
        ax.set(xscale="log", xlabel="cycles per mission", ylabel="amplitude [load units]",
               title=f"load spectrum: {self.mission} ({len(self.blocks)} blocks, {self.total_cycles:.3g} cycles)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
        return ax.figure


def build_spectrum(mission: Mission, structure: Structure | None = None, *, ground_level: float = 0.0,
                   min_range: float = 0.0) -> LoadSpectrum:
    """Low-cycle blocks from rainflow counting of each pattern's level sequence (starting and ending
    at ``ground_level``) plus high-cycle blocks from every excitation (amplitude x dynamic
    amplification when a ``structure`` is given, cycles = frequency x duration)."""
    blocks: list[Block] = []
    for p in mission.patterns:
        # level sequence: a repeated segment is N excursions from the preceding level (punch-outs
        # from hover, gusts from cruise), so every repeat closes a cycle
        series = [ground_level]
        for s in mission.segments:
            level, prev = s.loads.get(p, 0.0), series[-1]
            for _ in range(s.repeat - 1):
                series += [level, prev]
            series.append(level)
        series.append(ground_level)
        merged: dict[tuple, float] = {}
        for rng, mean, count in rainflow(series):
            if rng >= min_range and rng > 0:
                key = (round(float(mean), 9), round(0.5 * float(rng), 9))
                merged[key] = merged.get(key, 0.0) + count
        for (mean, amp), count in merged.items():
            blocks.append(Block(p, mean, amp, count, f"manoeuvre cycles of {p}"))
    for s in mission.segments:
        for e in s.excitations:
            amp = e.amplitude * (float(structure.amplification(e.frequency_hz)[0]) if structure else 1.0)
            blocks.append(Block(e.pattern, s.loads.get(e.pattern, 0.0), amp, e.frequency_hz * s.duration_s * s.repeat,
                                f"{s.name}: {e.name} at {e.frequency_hz:.0f} Hz"))
    return LoadSpectrum(mission.name, mission.duration_s, blocks, mission.patterns,
                        structure=asdict(structure) if structure else None)
