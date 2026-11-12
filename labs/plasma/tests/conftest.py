"""The plasma lab's folders on sys.path (designs, common, exp, sims), as its notebooks have them."""
import sys

from vegeta.labkit import bootstrap

for _p in reversed(bootstrap.folders("plasma")):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
