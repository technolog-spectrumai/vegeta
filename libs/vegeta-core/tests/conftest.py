import shutil

import pytest


def _gmsh_ok():
    try:
        import gmsh  # noqa: F401
        return True
    except (ImportError, OSError):
        return False


HAVE = {
    "requires_gmsh": _gmsh_ok(),
    "requires_ccx": shutil.which("ccx") is not None,
    "requires_prusaslicer": shutil.which("prusa-slicer") is not None,
}


def pytest_collection_modifyitems(config, items):
    for item in items:
        for marker, ok in HAVE.items():
            if marker in item.keywords and not ok:
                item.add_marker(pytest.mark.skip(reason=f"{marker}: tool not available"))
        if "requires_openfoam" in item.keywords:
            try:
                from vegeta.aeromant import OpenFOAMEnvironment

                if not OpenFOAMEnvironment.detect().available():
                    raise RuntimeError
            except RuntimeError:
                item.add_marker(pytest.mark.skip(reason="OpenFOAM not found"))


DESIGN = '''
import cadquery as cq
from vegeta.dedalus import Design, Parameter

class Plate(Design):
    parameters = [Parameter("width", 30.0, "mm", min=1), Parameter("thickness", 2.0, "mm", min=0.5)]
    def build(self, p):
        return cq.Workplane().box(p["width"], 10, p["thickness"])
'''


@pytest.fixture
def ws(tmp_path):
    from vegeta import core

    return core.Workspace.create(tmp_path / "ws", name="test")


@pytest.fixture
def plate_file(tmp_path):
    f = tmp_path / "designs_src" / "plate.py"
    f.parent.mkdir()
    f.write_text(DESIGN)
    return f
