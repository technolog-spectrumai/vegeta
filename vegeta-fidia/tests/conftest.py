import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from _helpers import DESIGN  # noqa: E402

@pytest.fixture
def plate(tmp_path):
    f = tmp_path / "plate.py"
    f.write_text(DESIGN)
    return f


HAVE_KEY = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def pytest_collection_modifyitems(config, items):
    render = None
    for item in items:
        if "requires_api_key" in item.keywords and not HAVE_KEY:
            item.add_marker(pytest.mark.skip(reason="no ANTHROPIC_API_KEY"))
        if "requires_render" in item.keywords:
            if render is None:
                from vegeta.fidia.render import pyvista_works
                render = pyvista_works()
            if not render:
                item.add_marker(pytest.mark.skip(reason="no off-screen pyvista rendering here"))
