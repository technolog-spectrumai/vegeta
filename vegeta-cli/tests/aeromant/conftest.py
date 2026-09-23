import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
DATA = Path(__file__).parent / "data"


def _openfoam_env():
    """Test harness only: AEROMANT_TEST_OPENFOAM_PREFIX (e.g. "micromamba run -p /opt/foam") or
    AEROMANT_TEST_OPENFOAM_BASHRC select the installation; otherwise OpenFOAMEnvironment.detect()."""
    import os
    import shlex

    from vegeta.aeromant import OpenFOAMEnvironment

    if os.environ.get("AEROMANT_TEST_OPENFOAM_PREFIX"):
        return OpenFOAMEnvironment(prefix=shlex.split(os.environ["AEROMANT_TEST_OPENFOAM_PREFIX"]))
    if os.environ.get("AEROMANT_TEST_OPENFOAM_BASHRC"):
        return OpenFOAMEnvironment(bashrc=os.environ["AEROMANT_TEST_OPENFOAM_BASHRC"])
    try:
        env = OpenFOAMEnvironment.detect()
    except RuntimeError:
        return None
    return env if env.available() else None


OPENFOAM = _openfoam_env()


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "requires_openfoam" in item.keywords and OPENFOAM is None:
            item.add_marker(pytest.mark.skip(reason="OpenFOAM not found"))


@pytest.fixture
def sphere_stl(tmp_path):
    from _sphere import icosphere
    from vegeta.aeromant import write_stl_ascii

    return write_stl_ascii(icosphere(0.5, 3), tmp_path / "sphere.stl")


@pytest.fixture
def openfoam():
    if OPENFOAM is None:
        pytest.skip("OpenFOAM not found")
    return OPENFOAM
