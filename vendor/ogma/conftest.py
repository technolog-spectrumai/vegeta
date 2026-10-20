"""Ogma's tests run from the source tree without an install: each distribution's src/ on the path."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for lib in ("yax", "ogma"):
    p = ROOT / lib / "src"
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import ogma  # noqa: E402,F401  bound now: the ogma/ directory itself would otherwise import as a namespace package
import yax  # noqa: E402,F401
