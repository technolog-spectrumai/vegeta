"""Make ``import assemblies`` work from any working directory (the package lives at the repository root), and run every
test with the solvers mocked (``stubs.py``)."""
import importlib
import pkgutil
import sys
from pathlib import Path

import pytest

REPO = str(Path(__file__).resolve().parents[2])
if REPO not in sys.path:
    sys.path.insert(0, REPO)
HERE = str(Path(__file__).resolve().parent)
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_ccx: needs the CalculiX ccx executable")
    config.addinivalue_line("markers", "slow: long-running test")
    config.addinivalue_line("markers", "real_solvers: runs the real CalculiX / OpenFOAM / MuJoCo instead of the mocks")


@pytest.fixture(autouse=True)
def solvers(request, monkeypatch):
    """Every test runs with the solver run functions mocked and answered by stubs (``stubs.py``), unless it is marked
    ``real_solvers``. Take ``solvers`` as an argument to look at the mocks' calls (``solvers.solve_models``,
    ``solvers.run_cases``, ``solvers.run_scene``; ``solvers.fea``/``cfd``/``sim`` list what was solved)."""
    if request.node.get_closest_marker("real_solvers"):
        yield None
        return
    import assemblies.workflows as wf
    for m in pkgutil.iter_modules(wf.__path__):          # imported first, so the mocks replace the names they hold
        try:
            importlib.import_module(f"assemblies.workflows.{m.name}")
        except ImportError:                              # an optional tool missing: its tests skip anyway
            pass
    import stubs
    yield stubs.install(monkeypatch)
