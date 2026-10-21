"""Vegeta's smoke suite (``./test.sh --smoke``): broad, fast, deterministic.

Goldens: ``golden(name, value, rel=...)`` compares ``value`` with ``golden.json``; ``SMOKE_UPDATE_GOLDEN=1`` rewrites
it (do that only when a change of result is intended, and say so in the commit).
"""
import json
import math
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "libs/vegeta-cli" / "src", ROOT / "libs/vegeta-core" / "src",       # tests/ may import any lab
          *sorted(ROOT.glob("labs/*/designs")), *sorted(ROOT.glob("labs/*/exp")), ROOT / "labs" / "aero" / "sims"):
    if str(p) not in sys.path:
        sys.path.append(str(p))
os.environ.setdefault("MPLBACKEND", "Agg")

GOLDEN = Path(__file__).with_name("golden.json")
_UPDATE = os.environ.get("SMOKE_UPDATE_GOLDEN") == "1"
_store = json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {}


def _close(a, b, rel, abs_):
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_close(a[k], b[k], rel, abs_) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(_close(x, y, rel, abs_) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return math.isclose(a, b, rel_tol=rel, abs_tol=abs_)
    return a == b


def _plain(v):
    if hasattr(v, "tolist"):
        return v.tolist()
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v


@pytest.fixture(scope="session")
def golden():
    def check(name, value, rel=1e-9, abs=1e-12):
        value = _plain(value)
        if _UPDATE:
            _store[name] = value
            return
        assert name in _store, f"no golden value {name!r}: run with SMOKE_UPDATE_GOLDEN=1 once"
        assert _close(value, _store[name], rel, abs), f"{name}: {value!r} != golden {_store[name]!r}"
    yield check
    if _UPDATE:
        GOLDEN.write_text(json.dumps(dict(sorted(_store.items())), indent=1) + "\n")
