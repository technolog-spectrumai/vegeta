"""python -m pinna_launch [clip ...|all] [--out DIR]  (from labs/marine/sims, or with it on PYTHONPATH)."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
try:
    import vegeta.dedalus  # noqa: F401
except ImportError:                      # not under activate.sh: the Vegeta libraries from the source tree
    for _d in ("libs/vegeta-core/src", "libs/labkit/src", "libs/ogma/src", "libs/vegeta-cli/src"):
        sys.path.insert(0, str(_ROOT / _d))

from pinna_launch.clips import main  # noqa: E402

main()
