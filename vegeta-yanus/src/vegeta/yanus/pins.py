"""pins — the explicit choice of which run (or file) a unit reads: ``notebooks/pins/<unit>.json``,
``{"kestrel/airframe-sweep@4Q-2": "side_labs/runs/kestrel/airframe-sweep/<run id>"}``. Yanus never picks one by
itself; ``yanus pin`` writes them, ``yanus pin --suggest`` only prints candidates."""
from __future__ import annotations

import json
from pathlib import Path

from .stamps import write_json_atomic

DIR = Path("notebooks") / "pins"


def pin_file(root, unit: str) -> Path:
    return Path(root) / DIR / f"{unit.replace('/', '__')}.json"


def read_pins(root, unit: str) -> dict:
    try:
        d = json.loads(pin_file(root, unit).read_text())
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _rel(root, path) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = (Path.cwd() / p)
    try:
        return str(p.resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(p)


def set_pin(root, unit: str, key: str, path) -> Path:
    pins = read_pins(root, unit)
    pins[key] = None if path is None else _rel(root, path)
    f = pin_file(root, unit)
    f.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(f, dict(sorted(pins.items())))
    return f


def clear_pin(root, unit: str, key: str) -> bool:
    pins = read_pins(root, unit)
    if key not in pins:
        return False
    del pins[key]
    write_json_atomic(pin_file(root, unit), dict(sorted(pins.items())))
    return True


def resolve(root, value) -> Path | None:
    if value is None:
        return None
    p = Path(value)
    return p if p.is_absolute() else Path(root) / p
