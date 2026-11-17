"""The published exports (``exports/``): every file matches MANIFEST.json and is valid YX2, and every machine a
consumer names in its requirements file (``tools/consumers/*.json``, ``machines``) is published and serves it.
Only Yax and the published files are needed."""
import json
import sys
from pathlib import Path

import pytest

yax = pytest.importorskip("yax")
from yax import tags as tg  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import publish  # noqa: E402

CONSUMERS = sorted((ROOT / "tools" / "consumers").glob("*.json"))
NAMED = [(c.stem, n) for c in CONSUMERS for n in json.loads(c.read_text()).get("machines", [])]


def test_files_match_the_manifest():
    assert publish.check(ROOT / "exports") == []


@pytest.mark.parametrize("consumer,name", NAMED)
def test_named_machines_are_published_and_served(consumer, name):
    man = publish.load_manifest(ROOT / "exports")
    assert name in man["machines"], f"{name} is not published"
    assert man["machines"][name]["consumers"][consumer]
    m, _ = yax.read(ROOT / "exports" / man["machines"][name]["file"])
    assert tg.check(m, tg.Requirements.load(ROOT / "tools" / "consumers" / f"{consumer}.json"), "yx2").ok


@pytest.mark.parametrize("name", ["pumba", "timon"])
def test_pumba_and_timon_are_published_for_montezuma(name):
    assert ("montezuma", name) in NAMED
    e = publish.load_manifest(ROOT / "exports")["machines"][name]
    assert e["lab"] == "land" and e["mass_kg"] > 0 and e["visual_triangles"] > 0
