"""Ozaru's tests run from the source tree without an install: each library's src/ on the path."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for lib in ("porunga", "mimesis", "yanus"):
    p = ROOT / lib / "src"
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import mimesis  # noqa: E402,F401  bound now: the mimesis/ directory itself would otherwise import as an empty namespace package
import porunga  # noqa: E402,F401
import yanus  # noqa: E402,F401
