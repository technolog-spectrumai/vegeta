import numpy as np
import pytest
from _helpers import box_part
from PIL import Image

from vegeta.fidia.render import VIEWS, render_views


@pytest.fixture
def parts():
    return [box_part("base", size=(100, 60, 10), color=(0.8, 0.1, 0.1, 1.0)),
            box_part("post", size=(20, 20, 50), offset=(30, 0, 10), color=(0.1, 0.2, 0.9, 1.0))]


def _check(out, info, size):
    assert set(info["views"]) == set(VIEWS)
    for name in VIEWS:
        img = np.asarray(Image.open(out / f"{name}.png"))
        assert img.shape[:2] == (size, size)
        assert img.std() > 5, name  # not blank
    sheet = Image.open(out / "sheet.png")
    assert sheet.size == (3 * size, 2 * size)


def test_matplotlib_backend(tmp_path, parts):
    info = render_views(parts, tmp_path, size=160, backend="matplotlib", title="t")
    assert info["backend"] == "matplotlib"
    _check(tmp_path, info, 160)


@pytest.mark.requires_render
def test_pyvista_backend(tmp_path, parts):
    info = render_views(parts, tmp_path, size=160, backend="pyvista")
    assert info["backend"] == "pyvista"
    _check(tmp_path, info, 160)
    red = np.asarray(Image.open(tmp_path / "iso.png")).astype(int)
    assert ((red[:, :, 0] > 120) & (red[:, :, 1] < 80)).any()  # the base's colour is visible
