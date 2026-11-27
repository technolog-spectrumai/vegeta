"""The fictional alien vessels: the brief's dimensions, every component separate and valid, the emblem on the hull
(ship) and both sides of the sail (submarine), the flag at the top, the export and the rigid bodies."""
import json

import pytest

cq = pytest.importorskip("cadquery")

import alien_vessels as av  # noqa: E402


def _bb(D, part):
    return D().build(D.p({"part": part})).val().BoundingBox()


def test_ship_dimensions_and_components():
    h, sup = _bb(av.AlienShip, "hull"), _bb(av.AlienShip, "superstructure")
    assert h.xlen / 1000 == pytest.approx(140, abs=1.5) and h.ylen / 1000 == pytest.approx(17, abs=0.1)
    assert h.zmax / 1000 > 8.0 and abs(min(av.AlienShip.section(av.AlienShip.p(), 0)[1], 99) - 8.0) < 1e-6      # 8 m visible at midship
    assert sup.xlen / 1000 == pytest.approx(56, abs=0.5) and sup.ylen / 1000 == pytest.approx(13, abs=0.1)
    assert _bb(av.AlienShip, "flag").zmax / 1000 == pytest.approx(31, abs=0.2)
    for k in av.SHIP_PARTS:
        assert av.AlienShip().build(av.AlienShip.p({"part": k})).val().isValid(), k
    e = _bb(av.AlienShip, "emblem")
    assert e.ymin < 0 < e.ymax and e.zmax / 1000 < 9.0                       # both sides of the hull


def test_submarine_dimensions_and_components():
    h, sail = _bb(av.AlienSubmarine, "sub_hull"), _bb(av.AlienSubmarine, "sail")
    assert h.xlen / 1000 == pytest.approx(95, abs=0.5) and h.ylen / 1000 == pytest.approx(10, abs=0.1) and h.zlen / 1000 == pytest.approx(10, abs=0.1)
    assert (sail.zmax - h.zmin) / 1000 == pytest.approx(17, abs=0.1) and sail.ylen / 1000 == pytest.approx(4, abs=0.1)
    assert sail.xlen / 1000 == pytest.approx(12, abs=1.0)
    e = _bb(av.AlienSubmarine, "emblem")
    assert e.ymin < 0 < e.ymax and sail.zmin < e.zmin
    for k in av.SUB_PARTS:
        assert av.AlienSubmarine().build(av.AlienSubmarine.p({"part": k})).val().isValid(), k


def test_emblem_has_eyes_and_bones():
    w, b = av.emblem_2d(1000.0, 10.0)
    bw, bb = w.val().BoundingBox(), b.val().BoundingBox()
    assert bw.ylen == pytest.approx(1000, rel=0.05) and bb.zmin >= 0 and bb.ylen < bw.ylen / 3
    assert w.val().Volume() > 0 and b.val().Volume() > 0


def test_export_and_rigid_bodies(tmp_path):
    import ketos_export
    import sea_robots as sr
    ketos_export.export(str(tmp_path), {"alien_ship", "alien_submarine"})
    for n, parts in (("alien_ship", av.SHIP_PARTS), ("alien_submarine", av.SUB_PARTS)):
        c = json.load(open(tmp_path / f"{n}.card.json"))
        assert c["fictional"] and not c["functional"] and c["components"] == list(parts)
        for m in c["parts"]:
            assert (tmp_path / f"{m['model']}.json").exists()
        assert sr.alien(n).total_mass() == pytest.approx(c["mass_kg"], rel=1e-3)
