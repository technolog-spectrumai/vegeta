"""Tests across labs (tests/ may import any lab): every lab's designs/ on sys.path."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _d in sorted((ROOT / "labs").glob("*/designs")):
    if str(_d) not in sys.path:
        sys.path.append(str(_d))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: a heavy design compared in full (seconds to a minute)")
