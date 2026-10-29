"""The Regis monster exports: the manifest matches the files, every design exists, and the balloon builds."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("export_monsters", ROOT / "labs/regis/export_monsters.py")
em = importlib.util.module_from_spec(spec)
spec.loader.exec_module(em)


def test_every_monster_has_a_design_and_an_exported_glb():
    manifest = json.loads((ROOT / "exports/regis/MANIFEST.json").read_text())["monsters"]
    assert sorted(manifest) == sorted(em.MONSTERS)
    for kind, (design, _, _) in em.MONSTERS.items():
        path, cls = design.split(":")
        assert f"class {cls}(" in (ROOT / path).read_text(), design
        info = manifest[kind]
        data = (ROOT / "exports/regis" / info["file"]).read_bytes()
        assert data[:4] == b"glTF", kind  # glTF 2.0 binary
        assert hashlib.sha256(data).hexdigest() == info["sha256"], f"{kind}: re-run labs/regis/export_monsters.py"
        assert 0 < info["triangles"] <= em.MAX_TRIANGLES


def test_the_balloon_builds_into_named_coloured_parts():
    cq = pytest.importorskip("cadquery")
    em._paths()
    from balloon import Balloon

    d = Balloon()
    built = d.build(d.resolve())
    assert isinstance(built, cq.Assembly)
    names = [name.split("/")[-1] for _, name, _, _ in built]
    assert names == ["envelope", "neck", "load_ring", "lines", "gondola"]
