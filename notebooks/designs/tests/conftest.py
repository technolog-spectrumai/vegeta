"""Put the design files on sys.path (``import gait``, ``import robot_dog_robot``, ``import myropod_robot``), as the
notebooks do."""
import sys
from pathlib import Path

DESIGNS = str(Path(__file__).resolve().parents[1])
if DESIGNS not in sys.path:
    sys.path.insert(0, DESIGNS)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long-running test (a whole simulated mission)")
