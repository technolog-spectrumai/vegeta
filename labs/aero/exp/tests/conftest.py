"""Put designs/ and designs/exp/ on sys.path, as the notebooks do."""
import sys
from pathlib import Path

for _p in (Path(__file__).resolve().parents[2] / "designs", Path(__file__).resolve().parents[1]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
