"""Mock outputs live apart from real ones."""
from __future__ import annotations

import os
from pathlib import Path

SUFFIX = "-mock"


def isolated(path, suffix: str = SUFFIX):
    """``runs`` → ``runs-mock`` (idempotent; returns the same type it was given)."""
    s = str(path).rstrip("/")
    out = s if s.endswith(suffix) else s + suffix
    return Path(out) if isinstance(path, Path) else out


def isolate_env(defaults: dict, suffix: str = SUFFIX) -> dict:
    """For each ``{ENV_VAR: default path}``: set the variable to the isolated form of its current value (or the
    default); returns what was set."""
    done = {}
    for key, default in defaults.items():
        os.environ[key] = done[key] = str(isolated(os.environ.get(key) or str(default), suffix))
    return done
