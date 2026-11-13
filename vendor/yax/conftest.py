"""Yax's tests run from the source tree without an install (``tests/helpers.py`` importable too)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for p in (ROOT / "src", ROOT / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
