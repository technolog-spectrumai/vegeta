"""YX2 (docs/yx2.md): a validated USD subset — units and frame stated, stable ids, feature metadata always present,
canonical text; write(read(f)) == f; selected machines keep what they hold; an invalid file is refused with its
problems; the other conversions still work from and to it."""
import json
from pathlib import Path

import numpy as np
import pytest

import yax
from yax import features as ft
from yax import hub
from yax.formats import yx2
from helpers import assert_same
from test_features import _rich

HERE = Path(__file__).parent
ROVER = HERE / "fixtures" / "rover.xml"
pytest.importorskip("mujoco")


def test_yx2_round_trip_is_exact_and_idempotent(tmp_path):
    m = _rich()
    f = tmp_path / "rich.yx2"
    assert yax.save(m, f).dropped == []
    assert yx2.validate(f) == []
    text = f.read_text()
    assert text.startswith("#usda 1.0") and 'dictionary yx2' in text and '"included"' in text and 'string yx2:id = "link:base"' in text
    assert 'yx2:id = "joint:shoulder"' in text and 'yx2:id = "shape:base/hull"' in text and 'yx2:id = "mass:base/battery"' in text
    back, _ = yax.read(f)
    assert_same(m, back)
    assert back.data == m.data and back.producer == m.producer
    assert back.features == {"included": list(ft.FEATURES), "excluded": []}
    assert back.joints[0].drive == m.joints[0].drive and back.joints[0].data == m.joints[0].data
    assert back.links[0].data == m.links[0].data and back.links[0].point_masses == [("battery", 0.3, (0.0, 0.0, 0.05))]
    yax.save(back, tmp_path / "again.yx2")
    assert (tmp_path / "again.yx2").read_text() == text                      # write(read(f)) == f
    assert yx2.text(m) == text                                                # the same machine, the same bytes


def test_yx2_keeps_a_selection_and_reports_nothing(tmp_path):
    m = ft.select(_rich(), exclude=("strength", "connections", "sensors"))
    f = tmp_path / "cut.yx2"
    yax.save(m, f)
    text = f.read_text()
    assert "allowable_pa" not in text and "M3" not in text and "cam" not in text
    back, _ = yax.read(f)
    assert back.features["excluded"] == ["connections", "sensors", "strength"]
    inv = ft.inventory(back)
    assert inv["strength"].status == ft.EXCLUDED and inv["materials"].status == ft.AVAILABLE and inv["motors"].status == ft.AVAILABLE
    assert back.data["materials"] == m.data["materials"] and back.data["propulsion"] == m.data["propulsion"]
    v = ft.check(back, ft.Requirements(required=("geometry", "mass", "materials"), optional=("strength",)), fmt="yx2")
    assert v.ok and v.statuses["strength"].status == ft.EXCLUDED
    yax.save(back, tmp_path / "again.yx2")
    assert (tmp_path / "again.yx2").read_text() == text


def test_yx2_refuses_what_is_outside_the_profile(tmp_path):
    m, _ = yax.read(ROVER)
    f = tmp_path / "rover.yx2"
    yax.save(m, f)
    assert yx2.validate(f) == []
    plain = tmp_path / "rover.usda"
    yax.save(m, plain)                                                        # plain USD: no layer data, no ids
    bad = tmp_path / "bad.yx2"
    bad.write_text(plain.read_text())
    problems = yx2.validate(bad)
    assert problems == ["no yx2 layer data (customLayerData['yx2']): not a YX2 file"]
    with pytest.raises(yax.Invalid):
        yax.read(bad)
    text = f.read_text()
    for wrong, expect in ((('string version = "0.1"', 'string version = "9"'), "yx2.version"),
                          (('string length = "m"', 'string length = "mm"'), "yx2.units"),
                          (('string yx2:id = "link:base"', 'string yx2:id = "link:other"'), "yx2:id must be 'link:base'"),
                          (('custom string yx2:id = "joint:', 'custom string yx2:extra = "x"\n    custom string yx2:id = "joint:'), "outside the YX2 vocabulary")):
        assert wrong[0] in text, wrong[0]
        bad.write_text(text.replace(wrong[0], wrong[1], 1))
        assert any(expect in p for p in yx2.validate(bad)), (expect, yx2.validate(bad))
    two = _rich()
    two.links[1].name = "base"                                                # two links of one name: no identity
    with pytest.raises(yax.Invalid, match="not unique"):
        yax.save(two, tmp_path / "two.yx2")
    not_text = tmp_path / "bin.yx2"
    not_text.write_bytes(b"\x00\x01")
    assert yx2.validate(not_text) and not yx2.validate(f)


def test_conversions_through_yx2_still_work(tmp_path):
    """The existing conversions take a .yx2 as a source and a target like any file: MJCF compiles to the same model,
    URDF and SDF round-trip, the reports are unchanged."""
    import mujoco
    f = tmp_path / "rover.yx2"
    rep = yax.convert(ROVER, f)
    assert len(rep.dropped) == 1 and "world geom" in rep.dropped[0]
    rep = yax.convert(f, tmp_path / "rover.xml")
    assert rep.dropped == []
    a, b = mujoco.MjModel.from_xml_path(str(ROVER)), mujoco.MjModel.from_xml_path(str(tmp_path / "rover.xml"))
    assert a.nbody == b.nbody and a.njnt == b.njnt and np.allclose(a.body_mass, b.body_mass)
    for ext in ("urdf", "sdf", "usda"):
        yax.convert(f, tmp_path / f"rover.{ext}")
        back, _ = yax.read(tmp_path / f"rover.{ext}")
        assert_same(yax.read(f)[0], back, tol=1e-9, skip_joints=("wrist",) if ext == "urdf" else ())   # URDF: the ball joint is fixed
        yax.convert(tmp_path / f"rover.{ext}", tmp_path / f"back_{ext}.yx2")
        assert yx2.validate(tmp_path / f"back_{ext}.yx2") == []
    from yax import cli
    assert cli.main(["check", str(f)]) == 0 and cli.main(["check", str(tmp_path / "rover.xml")]) == 1
    assert cli.main(["show", str(f)]) == 0 and cli.main(["features", str(f), "--to", "yx2", "--require", "geometry,motors"]) == 0
    assert yax.detect(f) == "yx2" and "yx2" in yax.FORMATS


def test_yx2_meshes_are_inline(tmp_path):
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    m = hub.Machine("tet", links=[hub.Link("body", mass=1.0, inertia=np.eye(3) * 0.1, shapes=[hub.Shape("skin", "mesh", points=pts, faces=faces)])])
    f = tmp_path / "tet.yx2"
    yax.save(m, f)
    assert list(tmp_path.iterdir()) == [f]                                    # no side files
    back, _ = yax.read(f)
    np.testing.assert_allclose(back.links[0].shapes[0].points, pts)
    assert back.links[0].shapes[0].faces.tolist() == faces.tolist()


def test_rotated_poses_round_trip_exactly(tmp_path):
    """USD's matrix decomposition rounds a quaternion; the exact pose shadows make the round trip bit-exact and the
    text idempotent for links, shapes and point masses at odd angles."""
    q = hub.qnorm((0.3, 0.5, -0.7, 0.11))
    lk = hub.Link("b", pos=(0.1234567890123, -2 / 3, 1e-5), quat=q, mass=1.0, inertia=np.eye(3) * 0.01,
                  shapes=[hub.Shape("s", "box", (0.1, 0.2, 0.3), pos=(1 / 7, 1 / 9, 1 / 11), quat=hub.qnorm((0.9, -0.2, 0.3, 0.4)))],
                  point_masses=[("p", 0.1, (1 / 13, 0.0, -1 / 17))])
    m = hub.Machine("odd", links=[lk])
    f = tmp_path / "odd.yx2"
    yax.save(m, f)
    back, _ = yax.read(f)
    assert back.links[0].pos == lk.pos and back.links[0].quat == q and back.links[0].shapes[0].pos == lk.shapes[0].pos
    assert back.links[0].shapes[0].quat == lk.shapes[0].quat and back.links[0].point_masses == lk.point_masses
    yax.save(back, tmp_path / "again.yx2")
    assert (tmp_path / "again.yx2").read_text() == f.read_text()
