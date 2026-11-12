import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
DATA = Path(__file__).parent / "data"


def _openfoam_envs() -> dict:
    """Test harness only: one usable installation per OpenFOAM flavor.
    AEROMANT_TEST_OPENFOAM_PREFIX (e.g. "micromamba run -p /opt/foam") or AEROMANT_TEST_OPENFOAM_BASHRC
    add an installation explicitly; otherwise OpenFOAMEnvironment.candidates() is used."""
    import os
    import shlex

    from vegeta.aeromant import OpenFOAMEnvironment

    envs = []
    if os.environ.get("AEROMANT_TEST_OPENFOAM_PREFIX"):
        envs.append(OpenFOAMEnvironment(prefix=shlex.split(os.environ["AEROMANT_TEST_OPENFOAM_PREFIX"])))
    if os.environ.get("AEROMANT_TEST_OPENFOAM_BASHRC"):
        envs.append(OpenFOAMEnvironment(bashrc=os.environ["AEROMANT_TEST_OPENFOAM_BASHRC"]))
    envs += OpenFOAMEnvironment.candidates()
    by_flavor = {}
    for env in envs:
        flavor = env.flavor()
        if flavor and env.available() and flavor not in by_flavor:
            by_flavor[flavor] = env
    return by_flavor


OPENFOAM_BY_FLAVOR = _openfoam_envs()
OPENFOAM = OPENFOAM_BY_FLAVOR.get("openfoam.com") or next(iter(OPENFOAM_BY_FLAVOR.values()), None)


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "requires_openfoam" in item.keywords and OPENFOAM is None:
            item.add_marker(pytest.mark.skip(reason="OpenFOAM not found"))


@pytest.fixture(params=sorted(OPENFOAM_BY_FLAVOR) or ["none"])
def openfoam(request):
    """Runs the test once per available OpenFOAM flavor (openfoam.com, openfoam.org)."""
    if request.param == "none":
        pytest.skip("OpenFOAM not found")
    return OPENFOAM_BY_FLAVOR[request.param]


@pytest.fixture
def sphere_stl(tmp_path):
    from _sphere import icosphere
    from vegeta.aeromant import write_stl_ascii

    return write_stl_ascii(icosphere(0.5, 3), tmp_path / "sphere.stl")


