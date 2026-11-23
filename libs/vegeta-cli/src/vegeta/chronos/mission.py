"""Missions as sequences of segments with explicit load levels and vibratory excitations."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Excitation:
    """A vibratory force of ``amplitude`` (load units of ``pattern``) at ``frequency_hz`` during a segment,
    e.g. a rotor unbalance on the lateral pattern of one motor mount."""

    name: str
    frequency_hz: float
    amplitude: float
    pattern: str

    def __post_init__(self):
        if self.frequency_hz <= 0 or self.amplitude < 0:
            raise ValueError(f"excitation {self.name!r}: frequency must be > 0 and amplitude >= 0")


@dataclass(frozen=True)
class Segment:
    """``duration_s`` at steady ``loads`` (pattern -> level, in the load unit of the pattern) with
    ``excitations`` superimposed. ``repeat`` = this segment happens N times as excursions from the
    preceding level (20 punch-outs from hover), each closing a load cycle."""

    name: str
    duration_s: float
    loads: dict = field(default_factory=dict)
    excitations: tuple = ()
    repeat: int = 1
    notes: str = ""

    def __post_init__(self):
        if self.duration_s <= 0 or self.repeat < 1:
            raise ValueError(f"segment {self.name!r}: duration must be > 0 and repeat >= 1")
        for k, v in self.loads.items():
            if not isinstance(v, (int, float)):
                raise ValueError(f"segment {self.name!r}: load level of {k!r} must be a number")


@dataclass(frozen=True)
class Mission:
    name: str
    segments: tuple
    description: str = ""

    def __post_init__(self):
        if not self.segments:
            raise ValueError("a mission needs at least one segment")

    @property
    def duration_s(self) -> float:
        return float(sum(s.duration_s * s.repeat for s in self.segments))

    @property
    def duration_h(self) -> float:
        return self.duration_s / 3600.0

    @property
    def patterns(self) -> list[str]:
        out: list[str] = []
        for s in self.segments:
            for k in list(s.loads) + [e.pattern for e in s.excitations]:
                if k not in out:
                    out.append(k)
        return out

    def expanded(self) -> list[Segment]:
        """Segments with ``repeat`` unrolled, in flight order."""
        return [s for s in self.segments for _ in range(s.repeat)]

    def table(self):
        rows = [{"segment": s.name, "duration_s": s.duration_s, "repeat": s.repeat, **{f"load:{k}": v for k, v in s.loads.items()},
                 "excitations": ", ".join(f"{e.name}@{e.frequency_hz:.0f}Hz" for e in s.excitations)} for s in self.segments]
        try:
            import pandas as pd

            return pd.DataFrame(rows).set_index("segment")
        except ImportError:  # pragma: no cover
            return rows

    def profile(self, patterns=None, ax=None):
        """Step plot of the load levels over the mission (matplotlib)."""
        import matplotlib.pyplot as plt

        patterns = patterns or self.patterns
        if ax is None:
            _, ax = plt.subplots(figsize=(9, 3.2))
        t = 0.0
        times, levels = [0.0], {p: [0.0] for p in patterns}
        for s in self.expanded():
            for p in patterns:
                levels[p] += [s.loads.get(p, 0.0), s.loads.get(p, 0.0)]
            times += [t, t + s.duration_s]
            t += s.duration_s
        times.append(t)
        for p in patterns:
            levels[p].append(0.0)
            ax.plot([x / 60 for x in times], levels[p], label=p, drawstyle="steps-post")
        ax.set(xlabel="time [min]", ylabel="load level", title=f"mission {self.name}: {self.duration_h * 60:.0f} min")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        return ax.figure

    def describe(self) -> dict:
        return {"name": self.name, "description": self.description, "duration_s": self.duration_s,
                "segments": [{"name": s.name, "duration_s": s.duration_s, "repeat": s.repeat, "loads": dict(s.loads),
                              "excitations": [e.__dict__ for e in s.excitations], "notes": s.notes} for s in self.segments]}
