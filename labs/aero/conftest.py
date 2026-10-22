"""Every test in the aero lab (tests/, sims/, campaigns/, exp/) sees the lab's folders, as its notebooks do."""
import sys

from vegeta.labkit import bootstrap

for _p in reversed(bootstrap.folders("aero")):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
