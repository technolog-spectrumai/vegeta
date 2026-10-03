"""Make ``import assemblies`` work from any working directory (the package lives at the repository root)."""
import sys
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[2])
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_ccx: needs the CalculiX ccx executable")
    config.addinivalue_line("markers", "slow: long-running test")
