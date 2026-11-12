"""Put ``side_labs/`` and ``designs/`` on the path, as the scripts do."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
for p in (ROOT / "labs" / "aero" / "campaigns" / "side_labs", ROOT / "labs" / "aero" / "campaigns" / "side_labs" / "kestrel", ROOT / "labs" / "aero" / "designs"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long-running test (CAD export, a whole campaign prepared)")
