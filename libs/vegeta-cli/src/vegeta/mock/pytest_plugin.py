"""``pytest -p vegeta.mock.pytest_plugin`` (the mock-only user tests): with ``VEGETA_DEBUG`` on, install
``vegeta.mock`` for the whole session — FEA, CFD, slicing, MuJoCo (Chiron), Gmsh meshes and the aero models mocked —
and skip the tests listed in ``mock_incompatible.txt`` (they check the real engines' numbers)."""
import os
from pathlib import Path

LIST = Path(__file__).with_name("mock_incompatible.txt")


def _on() -> bool:
    return os.environ.get("VEGETA_DEBUG", "").strip().lower() in ("1", "true", "on", "yes")


def pytest_configure(config):
    if _on():
        from vegeta import mock
        mock.install()


def pytest_collection_modifyitems(config, items):
    if not _on() or not LIST.is_file():
        return
    import pytest
    names = {l.strip() for l in LIST.read_text().splitlines() if l.strip() and not l.startswith("#")}
    for item in items:
        nid = item.nodeid.split("/tests/", 1)[-1] if "/tests/" in item.nodeid else item.nodeid
        if item.nodeid in names or nid in names or item.name in names or getattr(item, "originalname", None) in names:
            item.add_marker(pytest.mark.skip(reason="checks the real engine; skipped under the mocks"))
