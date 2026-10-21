"""Ogma's tests run from the source tree without an install (Yax next to it)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE / "src", HERE.parent / "yax" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
