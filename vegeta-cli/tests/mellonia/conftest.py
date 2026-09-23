import shutil
from pathlib import Path

import pytest

DATA = Path(__file__).parent / "data"
HAVE_PRUSA = shutil.which("prusa-slicer") is not None


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "requires_prusaslicer" in item.keywords and not HAVE_PRUSA:
            item.add_marker(pytest.mark.skip(reason="prusa-slicer not on PATH"))


def box_stl(path: Path, dx: float, dy: float, dz: float) -> Path:
    """ASCII STL of an axis-aligned box on z=0 (no CAD package needed)."""
    import itertools

    v = {c: (c[0] * dx, c[1] * dy, c[2] * dz) for c in itertools.product((0, 1), repeat=3)}
    quads = [((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)), ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)),
             ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)), ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)),
             ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)), ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1))]
    lines = ["solid box"]
    for a, b, c, d in quads:
        for tri in ((a, b, c), (a, c, d)):
            lines += ["facet normal 0 0 0", "outer loop"] + [f"vertex {x} {y} {z}" for x, y, z in (v[p] for p in tri)]
            lines += ["endloop", "endfacet"]
    lines.append("endsolid box")
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def cube_stl(tmp_path):
    return box_stl(tmp_path / "cube.stl", 20, 20, 20)


@pytest.fixture
def tall_stl(tmp_path):
    return box_stl(tmp_path / "tall.stl", 10, 10, 40)
