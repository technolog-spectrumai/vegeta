"""Publishing: exports/<machine>.yx2 with a manifest a consumer can verify without Vegeta (tools/publish.py). A mock
machine, a CadQuery box as its design; no lab design is built."""
import json
import sys
from pathlib import Path

import pytest

yax = pytest.importorskip("yax")
pytest.importorskip("mujoco")
cq = pytest.importorskip("cadquery")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import publish  # noqa: E402
from test_export_visual import _design, _robot  # noqa: E402


@pytest.fixture
def registered():
    from vegeta.labkit import machines
    machines.register("mockpub", _robot, visual=_design)
    yield
    machines._REGISTRY.pop("mockpub", None)


def test_publish_and_check(registered, tmp_path):
    consumer = tmp_path / "sim.json"
    consumer.write_text(json.dumps({"consumer": "sim", "required": ["00core", "00geom", "00mass"], "machines": ["mockpub"]}))
    picky = tmp_path / "fem.json"
    picky.write_text(json.dumps({"consumer": "fem", "required": ["00str"]}))
    man = publish.publish(["mockpub"], tmp_path / "exports", consumers=[consumer, picky], strict=True)
    e = man["machines"]["mockpub"]
    assert e["file"] == "mockpub.yx2" and e["visual_triangles"] > 0 and e["links"] == 2
    assert e["consumers"] == {"sim": True, "fem": False}            # fem does not name it: reported, not fatal
    assert man["format"] == "vegeta-exports/1" and "commit" in man["vegeta"]
    assert json.loads((tmp_path / "exports" / "MANIFEST.json").read_text()) == man
    assert publish.check(tmp_path / "exports") == []
    (tmp_path / "exports" / "mockpub.yx2").write_text((tmp_path / "exports" / "mockpub.yx2").read_text() + "\n")
    (tmp_path / "exports" / "stray.yx2").write_text("")
    problems = publish.check(tmp_path / "exports")
    assert any("differs from the manifest" in p for p in problems) and any("stray.yx2: not in the manifest" in p for p in problems)


def test_strict_fails_a_named_consumer(registered, tmp_path):
    needy = tmp_path / "needy.json"
    needy.write_text(json.dumps({"consumer": "needy", "required": ["00str"], "machines": ["mockpub"]}))
    with pytest.raises(SystemExit):
        publish.publish(["mockpub"], tmp_path / "exports", consumers=[needy], strict=True)
