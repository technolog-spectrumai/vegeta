"""Explicit PrusaSlicer settings defined in Python and written to the .ini files PrusaSlicer loads."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Settings Mellonia insists on being stated explicitly (in any of the three sections).
REQUIRED_KEYS = {
    "printer": ("bed_shape", "nozzle_diameter", "max_print_height"),
    "filament": ("filament_diameter", "temperature", "bed_temperature"),
    "print": ("layer_height", "first_layer_height", "perimeters", "fill_density"),
}


def ini_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (list, tuple)):
        return ",".join(ini_value(v) for v in value)
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def read_ini(path: str | Path) -> dict[str, str]:
    """Flat ``key = value`` reader for PrusaSlicer ini exports (sections and comments ignored)."""
    out: dict[str, str] = {}
    for line in Path(path).read_text(errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith(("#", ";", "[")) or "=" not in s:
            continue
        k, v = s.split("=", 1)
        out[k.strip()] = v.strip()
    return out


@dataclass
class PrintSettings:
    """Printer, filament and print settings as PrusaSlicer keys (``layer_height``, ``fill_density`` ...).

    Keys not given use PrusaSlicer's built-in defaults; every slice records the full effective
    configuration so nothing is hidden. Minimum explicit keys: see ``REQUIRED_KEYS``.
    """

    printer: dict[str, Any] = field(default_factory=dict)
    filament: dict[str, Any] = field(default_factory=dict)
    print: dict[str, Any] = field(default_factory=dict)
    name: str = "settings"
    source: str = ""

    def __post_init__(self):
        merged = self.merged()
        missing = [k for keys in REQUIRED_KEYS.values() for k in keys if k not in merged]
        if missing:
            raise ValueError(f"print settings {self.name!r} must state explicitly: {missing}")
        dup = (set(self.printer) & set(self.filament)) | (set(self.printer) & set(self.print)) | (
            set(self.filament) & set(self.print))
        if dup:
            raise ValueError(f"settings given in more than one section: {sorted(dup)}")

    def merged(self) -> dict[str, Any]:
        return {**self.printer, **self.filament, **self.print}

    def write(self, directory: str | Path) -> list[Path]:
        """Write ``printer.ini``, ``filament.ini`` and ``print.ini``; returns them in load order."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        for section in ("printer", "filament", "print"):
            values = getattr(self, section)
            p = directory / f"{section}.ini"
            p.write_text(f"# written by mellonia from PrintSettings {self.name!r} ({section})\n"
                         + "".join(f"{k} = {ini_value(v)}\n" for k, v in values.items()))
            paths.append(p)
        return paths

    @classmethod
    def from_ini(cls, printer: str | Path, filament: str | Path, print: str | Path, name: str = "") -> "PrintSettings":
        """Settings exported from PrusaSlicer (one ini per preset)."""
        return cls(printer=read_ini(printer), filament=read_ini(filament), print=read_ini(print),
                   name=name or Path(print).stem, source=f"{printer}, {filament}, {print}")

    def replace(self, **sections) -> "PrintSettings":
        """Copy with some keys changed, e.g. ``replace(print={"fill_density": "40%"})``."""
        new = {s: dict(getattr(self, s)) for s in ("printer", "filament", "print")}
        for section, updates in sections.items():
            if section not in new:
                raise ValueError(f"unknown section {section!r}")
            new[section].update(updates)
        return PrintSettings(**new, name=self.name, source=self.source)


@dataclass(frozen=True)
class Orientation:
    """Rotation in degrees chosen by the engineer, passed to PrusaSlicer as --rotate-x, --rotate-y,
    --rotate (z), in that order. PrusaSlicer places the rotated part on the bed. No optimisation."""

    rotate_x: float = 0.0
    rotate_y: float = 0.0
    rotate_z: float = 0.0

    def cli_args(self) -> list[str]:
        args = []
        for flag, v in (("--rotate-x", self.rotate_x), ("--rotate-y", self.rotate_y), ("--rotate", self.rotate_z)):
            if v:
                args += [flag, f"{v:g}"]
        return args
