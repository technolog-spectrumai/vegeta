"""``import components`` from any working directory (the package lives at the repository root)."""
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: a heavy design compared in full (seconds to a minute)")
