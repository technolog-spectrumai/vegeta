"""Small file helpers: write-once JSON, append-only JSON lines, environment snapshot."""
from __future__ import annotations

import re

import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any


class ImmutableError(RuntimeError):
    """Raised when something tries to rewrite a write-once record."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "tolist"):
        return _jsonable(obj.tolist())
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


def write_once(path: Path, data: dict) -> Path:
    """Create ``path`` with JSON ``data``; refuses to overwrite (records are immutable)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "x") as fh:
            json.dump(_jsonable(data), fh, indent=2)
            fh.write("\n")
    except FileExistsError:
        raise ImmutableError(f"{path} already exists and is immutable") from None
    return path


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def append_jsonl(path: Path, data: dict) -> None:
    with open(path, "a") as fh:
        fh.write(json.dumps(_jsonable(data)) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def environment() -> dict:
    versions = {}
    for dist in ("vegeta-cli", "vegeta-core", "cadquery", "gmsh", "numpy"):
        try:
            versions[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            versions[dist] = None
    return {"python": sys.version.split()[0], "platform": platform.platform(), "packages": versions}

_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")

def check_name(kind: str, name: str) -> str:
    if not _NAME.match(name or ""):
        raise ValueError(f"{kind} name {name!r} must start with a letter and use letters, digits, '_' or '-'")
    return name
