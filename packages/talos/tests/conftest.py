import shutil
from pathlib import Path

import pytest


def _gmsh_ok():
    try:
        import gmsh  # noqa: F401
        return True
    except (ImportError, OSError):
        return False


HAVE_GMSH = _gmsh_ok()
HAVE_CCX = shutil.which("ccx") is not None
DATA = Path(__file__).parent / "data"


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "requires_gmsh" in item.keywords and not HAVE_GMSH:
            item.add_marker(pytest.mark.skip(reason="gmsh Python module not available"))
        if "requires_ccx" in item.keywords and not HAVE_CCX:
            item.add_marker(pytest.mark.skip(reason="CalculiX 'ccx' not on PATH"))


def make_box_step(path: Path, x0, y0, z0, dx, dy, dz) -> Path:
    """Write a box STEP with Gmsh's OpenCascade kernel (no CAD package needed)."""
    import gmsh

    gmsh.initialize(readConfigFiles=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("box")
        gmsh.model.occ.addBox(x0, y0, z0, dx, dy, dz)
        gmsh.model.occ.synchronize()
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    return path


@pytest.fixture(scope="session")
def beam_step(tmp_path_factory):
    if not HAVE_GMSH:
        pytest.skip("gmsh not available")
    return make_box_step(tmp_path_factory.mktemp("geom") / "beam.step", 0, -10, -5, 200, 20, 10)


@pytest.fixture(scope="session")
def bar_step(tmp_path_factory):
    if not HAVE_GMSH:
        pytest.skip("gmsh not available")
    return make_box_step(tmp_path_factory.mktemp("geom") / "bar.step", 0, 0, 0, 100, 10, 10)
