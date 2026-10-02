"""Put the Onager design files and the shared notebook designs (actuators.py, gait.py) on sys.path."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
for d in (HERE.parents[1], HERE.parents[3] / "notebooks" / "designs"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
