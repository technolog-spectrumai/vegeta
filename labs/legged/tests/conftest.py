"""The legged lab's folders on sys.path (designs, common, exp, sims), as its notebooks have them."""
import sys

from vegeta.labkit import bootstrap

for _p in reversed(bootstrap.folders("legged")):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long-running test (a whole simulated mission)")
