"""The external interpreter a real API runs in — or this one, which runs the stubs, under mock."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def interpreter(*, mock: bool, env: str | None = None, candidates=()) -> Path | None:
    """``sys.executable`` when ``mock``; else the first that exists of ``$env`` and ``candidates``; else None."""
    if mock:
        return Path(sys.executable)
    for c in ([os.environ.get(env, "")] if env else []) + [str(c) for c in candidates]:
        if c and Path(c).exists():
            return Path(c)
    return None
