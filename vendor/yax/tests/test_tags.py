"""Tags (docs/interchange.md): every property has a path and carries system tags Yax assigns and producer tags of
1–8 alphanumerics; the 00 namespace is reserved; select keeps or drops by tag without inventing anything and the
file records what was withheld; a consumer checks the tags it needs against the file as it is and a format's support;
the command line does the same."""
from pathlib import Path

import pytest

import yax
from yax import hub
from yax import tags as tg
from helpers import rich

HERE = Path(__file__).parent
ROVER = HERE / "fixtures" / "rover.xml"
pytest.importorskip("mujoco")


def test_properties_and_system_tags():
    m = rich()
    paths = {p.path: p for p in tg.properties(m)}
    assert paths["link:base/pose"].system == ("00core",) and paths["link:base/mass"].system == ("00mass",)
    assert paths["link:base/inertia"].system == ("00inert",) and paths["link:base/collision:hull"].system == ("00geom",)
    assert paths["link:base/pmass:battery"].system == ("00mass",) and paths["link:base/data:material"].system == ("00mat",)
    assert paths["link:base/data:strength/allowable_pa"].system == ("00str",) and paths["link:base/data:sdf/sensor"].system == ("00sensor",)
    assert paths["link:base/data:sdf/plugin"].system == ("00fmt",)
    assert paths["joint:shoulder/tree"].system == ("00core",) and paths["joint:shoulder/articulation"].system == ("00joint",)
    assert paths["joint:shoulder/drive"].system == ("00motor",) and paths["joint:shoulder/data:motor/stall_nm"].system == ("00motor",)
    assert paths["joint:shoulder/data:connection"].system == ("00conn",) and paths["joint:shoulder/data:damping"].system == ("00joint",)
    assert paths["data:propulsion/propellers/p/coefficients"].system == ("00prop",) and paths["data:materials/al6061"].system == ("00mat",)
    assert paths["data:propulsion/motors/mo"].system == ("00prop", "00motor")       # a property in two categories
    assert paths["producer"].system == ("00prov",)
    t = tg.table(m)
    assert t["link:base/mass"] == ["00mass"] and "link:base/pose" in t and set(tg.inventory(m)) >= set(tg.SYSTEM) - {"00detach"}   # no assembly in the fixture
    assert all(tg.CODE.match(c) for c in tg.SYSTEM) and all(tg.is_system(c) for c in tg.SYSTEM)
    assert "00core" in tg.describe(m) and "00geom: 2 properties" in tg.describe(m)


def test_producer_tags_and_their_rules():
    m = rich()
    assert tg.tag(m, "data:propulsion/propellers/*", "boreas") == 6                 # the propeller and everything under it
    assert tg.tag(m, "link:*/mass", "nb08") == 2
    assert tg.table(m)["data:propulsion/propellers/p"] == ["00prop", "boreas"] and tg.table(m)["link:arm/mass"] == ["00mass", "nb08"]
    assert {"data:propulsion/propellers/p", "data:propulsion/propellers/p/coefficients/CT"} <= set(tg.inventory(m)["boreas"])
    assert tg.tag(m, "data:propulsion/propellers/p", "apc") == 1                      # a pattern without a star: one path
    for bad in ("00mine", "nine_char", "toolongcode", "", "a b", "é"):
        with pytest.raises(ValueError):
            tg.tag(m, "link:*/mass", bad)
    with pytest.raises(ValueError, match="no property matches"):
        tg.tag(m, "link:*/colour", "ok")
    with pytest.raises(ValueError):
        tg.Requirements(required=("toolongcode",))
    assert tg.check_code("00geom", system_ok=True) == "00geom"


def test_select_keeps_and_drops_by_tag_without_inventing(tmp_path):
    m = rich()
    tg.tag(m, "data:propulsion/*", "boreas")
    cut = tg.select(m, drop=("00str", "00conn", "boreas"))
    assert "strength" not in cut.links[0].data and "connections" not in cut.data and "connection" not in cut.joints[0].data
    assert cut.data["propulsion"] == {}                                        # its tagged children went; the key stays, empty
    assert cut.tags["withheld"] == ["00conn", "00str", "boreas"] and "link:base/data:strength" not in cut.tags["table"]
    assert tg.inventory(cut)["00str"] is None and tg.inventory(cut)["00mat"]
    assert tg.describe(m) != tg.describe(cut) and "strength" in m.links[0].data   # the source is untouched
    only = tg.select(m, keep=("00geom", "00mass"))
    assert all(lk.inertia is None for lk in only.links) and all(j.kind == "fixed" and j.drive is None for j in only.joints)
    assert only.producer == {} and only.data == {} and [lk.name for lk in only.links] == ["base", "arm"]
    assert "00core" not in only.tags["withheld"] and "00inert" in only.tags["withheld"] and "00joint" in only.tags["withheld"]
    assert only.links[0].mass == 2.0 and only.links[0].point_masses == [("battery", 0.3, (0.0, 0.0, 0.05))] and len(only.links[0].shapes) == 1
    with pytest.raises(ValueError):
        tg.select(m, drop=("00core",))
    with pytest.raises(ValueError):
        tg.remove(only, "link:base/pose")
    nomass = tg.select(m, drop=("00mass",))
    rep = yax.save(nomass, tmp_path / "nomass.sdf")
    assert "<inertial>" not in (tmp_path / "nomass.sdf").read_text() and any("tag 00mass: withheld" in d for d in rep.dropped)
    yax.save(cut, tmp_path / "cut.usda")
    back, _ = yax.read(tmp_path / "cut.usda")
    assert back.tags == cut.tags and "allowable_pa" not in (tmp_path / "cut.usda").read_text()


def test_consumer_check_against_system_and_producer_tags():
    m = rich()
    tg.tag(m, "data:propulsion/*", "boreas")
    req = tg.Requirements(required=("00geom", "00mass", "00inert", "boreas"), optional=("00sensor", "fea"), consumer="piccolo")
    v = tg.check(m, req)
    assert v.ok and v.statuses["boreas"].status == tg.PRESENT and v.statuses["fea"].status == tg.ABSENT and str(v).startswith("piccolo: served")
    v = tg.check(m, req, fmt="urdf")                                          # URDF holds no part data
    assert not v.ok and v.statuses["boreas"].status == tg.UNSUPPORTED and v.statuses["00mass"].status == tg.PRESENT
    v = tg.check(m, tg.Requirements(required=("00sensor",)), fmt="sdf")
    assert v.ok and "1 of 2 properties reach sdf" in v.statuses["00sensor"].detail     # the verbatim SDF sensor, not data:sensors
    v = tg.check(m, tg.Requirements(required=("00joint",)), fmt="urdf")
    assert v.ok and "ball joints as fixed" in v.statuses["00joint"].detail
    cut = tg.select(m, drop=("boreas",))
    v = tg.check(cut, req)
    assert not v.ok and [s.tag for s in v.unmet] == ["boreas"] and v.statuses["boreas"].status == tg.WITHHELD
    rover, _ = yax.read(ROVER)
    v = tg.check(rover, tg.Requirements(required=("00str",)))
    assert v.statuses["00str"].status == tg.ABSENT and "NOT served" in str(v)
    v = tg.check(rover, tg.Requirements(required=("00motor",)), fmt="mjcf")
    assert v.ok and "position actuators" in v.statuses["00motor"].detail
    v = tg.check(m, tg.Requirements(required=("00prop",)), fmt="jsbsim")
    assert v.ok and "coefficients against J" in v.statuses["00prop"].detail


def test_cli_tags_and_select(tmp_path, capsys):
    from yax import cli
    assert cli.main(["tags", str(ROVER)]) == 0
    out = capsys.readouterr().out
    assert "00geom:" in out and "00core:" in out and "00mat" not in out
    assert cli.main(["tags", str(ROVER), "--require", "00geom,00mass", "--optional", "00sensor", "--to", "sdf"]) == 0
    assert "served" in capsys.readouterr().out
    assert cli.main(["tags", str(ROVER), "--require", "00str"]) == 1
    assert cli.main(["select", str(ROVER), str(tmp_path / "r.yx2"), "--tag", "link:*/mass=weighed", "--drop", "00inert,00motor"]) == 0
    m, _ = yax.read(tmp_path / "r.yx2")
    assert m.tags["withheld"] == ["00inert", "00motor"] and all(lk.inertia is None for lk in m.links) and not any(j.drive for j in m.joints)
    assert all("weighed" in m.tags["table"][f"link:{lk.name}/mass"] for lk in m.links)
    assert cli.main(["show", str(tmp_path / "r.yx2")]) == 0 and "withheld: 00inert, 00motor" in capsys.readouterr().out
    (tmp_path / "req.json").write_text('{"consumer": "sim", "required": ["00inert"], "optional": ["weighed"]}')
    assert cli.main(["tags", str(tmp_path / "r.yx2"), "--requirements", str(tmp_path / "req.json")]) == 1
    out = capsys.readouterr().out
    assert "sim: NOT served" in out and "00inert: withheld" in out and "weighed: present" in out


def test_old_files_read_with_implied_tags(tmp_path):
    m, _ = yax.read(ROVER)
    yax.save(m, tmp_path / "plain.usda")
    assert "yax:tags" not in (tmp_path / "plain.usda").read_text()
    back, _ = yax.read(tmp_path / "plain.usda")
    assert back.tags == {} and tg.inventory(back)["00geom"] and "00sensor" not in tg.inventory(back)
    assert hub.lump(tg.select(m, drop=("00motor",))).tags["withheld"] == ["00motor"]


def test_a_container_stays_while_a_kept_property_is_under_it():
    m = rich()
    tg.tag(m, "data:propulsion/propellers*", "boreas")
    only = tg.select(m, keep=("00core", "boreas"))
    assert set(only.data) == {"propulsion"} and set(only.data["propulsion"]) == {"propellers"}
    assert only.data["propulsion"]["propellers"]["p"]["diameter_m"] == 0.2 and "00prop" in only.tags["withheld"]
    none = tg.select(m, keep=("00core",))
    assert none.data == {}
