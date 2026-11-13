"""V6.3: a machine leaves Vegeta only as standard files written through Yax (``tools/export.py``), from what the labs
register (``vegeta.labkit.machines``)."""
import sys
from pathlib import Path

import numpy as np
import pytest

yax = pytest.importorskip("yax")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import export  # noqa: E402


def test_drongo_exports_with_its_rotors(tmp_path):
    reps = export.export("drongo", tmp_path / "drongo.usda", also=("sdf",))
    assert (tmp_path / "drongo.usda").exists() and (tmp_path / "drongo.sdf").exists()
    assert any("ogma" in d for d in reps[-1].dropped)                     # SDF cannot hold part data: reported
    m, _ = yax.read(tmp_path / "drongo.usda")
    assert m.producer["machine"] == "drongo"
    rot = m.data["ogma"]["rotors"]
    assert len(rot) == 4 and sorted(r["spin"] for r in rot) == [-1, -1, 1, 1]
    for r in rot:
        assert r["kF"] * r["w_max"] ** 2 == pytest.approx(r["T_max"])
        assert r["tau_down"] > r["tau_up"] > 0


def test_lumped_drongo_is_the_body_minus_the_rotors():
    m, _ = export.machine("drongo")
    ml, _ = export.machine("drongo", lump=True)
    rot = ml.data["ogma"]["rotors"]
    assert len(ml.links) == 1 and not ml.joints
    total = sum(lk.mass for lk in m.links)
    assert ml.links[0].mass == pytest.approx(total - sum(r["kg"] for r in rot), rel=1e-12)
    assert np.all(np.linalg.eigvalsh(ml.links[0].inertia) > 0)


def test_unknown_machine_is_refused():
    with pytest.raises(KeyError):
        export.machine("no_such_machine")


@pytest.mark.parametrize("name", sorted(set(export.registry().names()) - {"drongo"}))
def test_every_machine_goes_through_yax(name, tmp_path):
    m, _ = export.machine(name)
    assert m.links and all(lk.mass is None or lk.mass >= 0 for lk in m.links)
    yax.save(m, tmp_path / f"{name}.usda")
    m2, _ = yax.read(tmp_path / f"{name}.usda")
    assert [lk.name for lk in m2.links] == [lk.name for lk in m.links]


# ------------------------------------------------------------------------------------- propellers as part data
WITH_PROPELLERS = ("drongo", "nisus", "nisus_plus", "falco")


def test_drongo_propulsion_block_matches_its_rotors(tmp_path):
    """Drongo's propulsion block: the 5x4.3's BEMT map (coefficients against J, the grid), the 2306 on 4S, and the
    reduced constants that ``ogma.rotors`` carries — the same numbers, checked at hover; the block survives USD and
    SDF reports it as the reduction."""
    reps = export.export("drongo", tmp_path / "drongo.usda", also=("sdf",))
    m, _ = yax.read(tmp_path / "drongo.usda")
    blk, rot = m.data["propulsion"], m.data["ogma"]["rotors"]
    assert [r["name"] for r in blk["rotors"]] == ["rotor0", "rotor1", "rotor2", "rotor3"]
    for r, o in zip(blk["rotors"], rot):
        assert r["pos"] == o["pos"] and r["spin"] == o["spin"] and r["axis"] == [0.0, 0.0, 1.0]
        red = blk["reduced"][r["name"]]
        assert red["kF"] == o["kF"] and red["kM"] == o["kM"] and red["w_max"] == o["w_max"] and red["tau_up"] == o["tau_up"]
        assert 0 <= red["residual"] < 1 and "hover" not in red["residual_at"] and red["residual_at"].endswith("0 m/s")
    (prop,) = blk["propellers"].values()
    co, grid = prop["coefficients"], prop["grid"]
    assert prop["blades"] == 3 and prop["diameter_m"] == pytest.approx(5 * 0.0254) and len(prop["stations"]) >= 3
    assert np.all(np.diff(co["J"]) > 0) and co["J"][0] == 0 and 0.05 < co["CT"][0] < 0.3 and 0 < co["CP"][0] < 0.2
    assert np.asarray(grid["thrust_n"]).shape == (len(grid["rpm"]), len(grid["airspeed_m_s"])) and grid["rho"] == 1.225
    (mot,) = blk["motors"].values()
    assert mot["kv_rpm_per_v"] == 2400 and 0 < mot["power_w"] < mot["voltage_v"] * mot["max_current_a"]
    assert blk["battery"]["cells"] == 4
    assert any("propulsion: rotor rotor0: its propeller map reduced to kF" in d for d in reps[-1].dropped)
    assert any("ogma" in d for d in reps[-1].dropped)


@pytest.mark.parametrize("name", WITH_PROPELLERS)
def test_propulsion_blocks_are_consistent(name):
    """Every rotor names a propeller and a motor the block holds; the propeller's coefficients come from its grid
    (CT at J = 0 is the grid's static thrust coefficient); the motor's power is a point of the grid within the pack."""
    from vegeta.labkit import propulsion as lp
    m, _ = export.machine(name)
    blk = m.data["propulsion"]
    for r in blk["rotors"]:
        assert r["propeller"] in blk["propellers"] and r["motor"] in blk["motors"] and r["spin"] in (-1, 1)
        assert np.linalg.norm(r["axis"]) == pytest.approx(1.0) and r["name"] in blk["reduced"]
    for p in blk["propellers"].values():
        g, D = p["grid"], p["diameter_m"]
        n0 = g["rpm"][-1] / 60
        ct0 = lp.at(g, "thrust_n", g["rpm"][-1], 0.0) / (g["rho"] * n0 ** 2 * D ** 4)
        assert p["coefficients"]["CT"][0] == pytest.approx(ct0, rel=0.05)      # Boreas: no Reynolds effect, CT(0) ~ constant
        assert p["inertia_kgm2"] > 0 and p["mass_kg"] > 0
    for mo in blk["motors"].values():
        assert mo["power_w"] > 0 and (("max_current_a" not in mo) or mo["power_w"] < mo["voltage_v"] * mo["max_current_a"])
    for red in blk["reduced"].values():
        assert red["kF"] > 0 and red["kM"] > 0 and red["w_max"] > 0 and red["residual"] >= 0


def test_nisus_flies_in_jsbsim_on_its_own_map(tmp_path):
    """``--also jsbsim`` writes NISUS as a JSBSim aircraft with its 9x6 on the fitted X2216: JSBSim's thrust at its
    settled rpm and advance ratio is the exported grid's (the propeller file holds the map), and the reaction torque
    on the airframe follows the assumed spin."""
    jsbsim = pytest.importorskip("jsbsim")
    from vegeta.labkit import propulsion as lp
    reps = export.export("nisus", tmp_path / "nisus.usda", also=("jsbsim",))
    assert not any("propulsion" in d for d in reps[-1].dropped)
    root = tmp_path / "jsbsim"
    assert (root / "aircraft" / "nisus" / "nisus.xml").exists() and list((root / "aircraft" / "nisus" / "Engines").glob("prop_*.xml"))
    for d in ("engine", "systems"):
        (root / d).mkdir()
    m, _ = yax.read(tmp_path / "nisus.usda")
    blk = m.data["propulsion"]
    (prop,) = blk["propellers"].values()
    (rot,) = blk["rotors"]
    f = jsbsim.FGFDMExec(str(root))
    f.set_debug_level(0)
    assert f.load_model("nisus")
    f.set_dt(1e-4)
    f.set_property_value("ic/h-sl-ft", 0.0)
    f.set_property_value("ic/vt-kts", 0.0)
    f.set_property_value("fcs/throttle-cmd-norm", 1.0)
    f.run_ic()
    for _ in range(5000):
        f.run()
    g = f.get_property_value
    rpm, J = g("propulsion/engine/propeller-rpm"), g("propulsion/engine/advance-ratio")
    V = max(J, 0.0) * rpm / 60 * prop["diameter_m"]
    T = g("propulsion/engine/thrust-lbs") * 4.4482216152605
    assert prop["grid"]["rpm"][0] < rpm < prop["grid"]["rpm"][-1]
    assert T == pytest.approx(lp.at(prop["grid"], "thrust_n", rpm, V), rel=0.03)
    assert g("moments/l-prop-lbsft") * rot["spin"] > 0                       # drag torque on the frame along spin · axis (+x forward)


# ------------------------------------------------------------------------------------- selective export (tags)
from yax import tags as tg  # noqa: E402


def test_selective_export_withholds_and_says_so(tmp_path):
    """Dropped tags reach no file, no metadata and no side file; the file records the table and what was withheld; the
    inventory says present / withheld / absent; the structure (links, joint tree) stays."""
    reps = export.export("drongo", tmp_path / "drongo.usda", also=("sdf", "jsbsim"), drop=("00motor", "00prov"))
    text = (tmp_path / "drongo.usda").read_text()
    for marker in ("kF", "propulsion", "yax:producer", "sources", "2306"):
        assert marker not in text, marker
    m, _ = yax.read(tmp_path / "drongo.usda")
    assert m.tags["withheld"] == ["00motor", "00prov", "assumed", "boreas"] and m.producer == {}   # nb08 stays: the materials carry it
    assert "propulsion" not in m.data and "ogma" not in m.data and m.data["materials"]
    inv = tg.inventory(m)
    assert inv["00motor"] is None and inv["00prov"] is None and inv["00matl"] and inv["00geom"] and "00sensor" not in inv
    assert [lk.name for lk in m.links] == [lk.name for lk in export.machine("drongo")[0].links]
    assert not list((tmp_path / "jsbsim" / "aircraft" / "drongo").rglob("prop_*.xml"))      # no propeller files
    assert any("tag 00motor: withheld by the producer" in d for d in reps[2].dropped)       # the SDF report says so
    only, _ = export.machine("nisus", keep=("00geom", "00mass"))
    assert all(lk.inertia is None for lk in only.links) and all(j.kind == "fixed" for j in only.joints)
    assert {"00inert", "00motor", "00prov", "boreas"} <= set(only.tags["withheld"]) and "00core" not in only.tags["withheld"]   # NISUS has no joints
    yax.save(only, tmp_path / "only.sdf")
    assert "<inertia>" not in (tmp_path / "only.sdf").read_text()


def test_design_and_user_tags(tmp_path):
    """The designs tag what they know about their numbers; the user tags more; both are in the table and selectable."""
    m, _ = export.machine("drongo")
    t = m.tags["table"] if m.tags else tg.table(m)
    t = tg.table(m)
    assert "boreas" in t["data:propulsion/propellers"] and "nb08" in t["data:materials"] and "assumed" in t["data:ogma/rotors"]
    assert t["data:propulsion/reduced"] == ["00motor", "assumed"] and t["link:drongo/mass"] == ["00mass"]
    m, _ = export.machine("nisus")
    assert "fitted" in tg.table(m)["data:propulsion/motors"] and "boreas" in tg.table(m)["data:propulsion/propellers"]
    m, _ = export.machine("drongo", tag=(("link:*/mass", "weighed"),), drop=("assumed",))
    assert all("weighed" in m.tags["table"][f"link:{lk.name}/mass"] for lk in m.links)
    assert "ogma" not in m.data and "reduced" not in m.data["propulsion"] and m.data["propulsion"]["propellers"]
    assert m.tags["withheld"] == ["assumed"]
    with pytest.raises(ValueError):
        export.machine("drongo", tag=(("link:*/mass", "00mine"),))


def test_full_is_the_default_and_the_flags_are_checked():
    full, _ = export.machine("drongo")
    assert not full.tags.get("withheld") and tg.inventory(full)["00geom"] and tg.inventory(full)["00core"]
    assert export.selection(full=True) == (None, (), []) and export.selection() == (None, (), [])
    assert export.selection(["00geom"], ["00mass"], spec={"keep": ["00joint"], "drop": ["00inert"], "tag": {"link:*/mass": "w"}},
                            tag=["data:*=x"]) == (("00geom", "00joint"), ("00mass", "00inert"), [("data:*", "x"), ("link:*/mass", "w")])
    with pytest.raises(ValueError):
        export.selection(["00geom"], full=True)
    with pytest.raises(ValueError):
        export.selection(["ninechars"])
    with pytest.raises(ValueError):
        export.selection(tag=["nocode"])
    with pytest.raises(ValueError):
        export.selection(tag=["link:*/mass=00sys"])


def test_tags_file_json_and_yaml(tmp_path):
    (tmp_path / "f.json").write_text('{"drop": ["00strgth", "00conn"]}')
    assert export.tags_from_file(tmp_path / "f.json") == {"keep": [], "drop": ["00strgth", "00conn"], "tag": {}}
    pytest.importorskip("yaml")
    (tmp_path / "f.yaml").write_text("keep:\n  - 00geom\n  - 00mass\ndrop:\n  - 00motor\ntag:\n  'link:*/mass': weighed\n  'data:*': [a, b]\n")
    assert export.tags_from_file(tmp_path / "f.yaml") == {"keep": ["00geom", "00mass"], "drop": ["00motor"],
                                                           "tag": {"link:*/mass": ["weighed"], "data:*": ["a", "b"]}}
    (tmp_path / "bad.yaml").write_text("features: [geometry]\n")
    with pytest.raises(ValueError):
        export.tags_from_file(tmp_path / "bad.yaml")


def test_consumer_requirements_report_and_never_override_the_flags(tmp_path, capsys):
    piccolo = export.ROOT / "tools" / "consumers" / "piccolo.json"
    reps = export.export("drongo", tmp_path / "d.yx2", also=("sdf",), drop=("00inert",), consumers=(piccolo,))
    verdicts = [r for r in reps if isinstance(r, tg.Compatibility)]
    assert [v.fmt for v in verdicts] == ["yx2", "sdf"] and not verdicts[0].ok and verdicts[0].statuses["00inert"].status == tg.WITHHELD
    assert verdicts[0].statuses["boreas"].status == tg.PRESENT and verdicts[1].statuses["boreas"].status == tg.UNSUPPORTED
    m, _ = yax.read(tmp_path / "d.yx2")
    assert all(lk.inertia is None for lk in m.links)                           # the requirement changed nothing
    out = tmp_path / "e.yx2"
    assert export.main(["drongo", "--to", str(out), "--drop", "00inert", "--consumer", str(piccolo)]) == 0
    assert export.main(["drongo", "--to", str(out), "--drop", "00inert", "--consumer", str(piccolo), "--strict"]) == 1
    assert export.main(["drongo", "--to", str(out), "--full", "--consumer", str(piccolo), "--strict"]) == 0
    printed = capsys.readouterr().out
    assert "piccolo (yx2): NOT served" in printed and "piccolo (yx2): served" in printed and "exported:" in printed
    assert "00matl:" in printed and "boreas:" in printed and "00motor:" in printed
    (tmp_path / "f.yaml").write_text("drop: [00motor]\ntag: {'link:*/mass': weighed}\n")
    pytest.importorskip("yaml")
    assert export.main(["drongo", "--to", str(out), "--tags", str(tmp_path / "f.yaml"), "--tag", "data:materials=cat"]) == 0
    printed = capsys.readouterr().out
    assert "00motor: withheld" in printed and "weighed: 3 properties" in printed and "cat: 1 properties" in printed
    with pytest.raises(ValueError):
        export.main(["drongo", "--to", str(out), "--full", "--drop", "00mass"])


@pytest.mark.parametrize("name", sorted(export.registry().names()))
def test_every_machine_exports_its_structure_alone(name, tmp_path):
    """A real design with only the structure (00core, 00geom, 00mass, 00inert, 00joint): the rest withheld, round trip exact."""
    keep = ("00geom", "00mass", "00inert", "00joint")
    m, _ = export.machine(name, keep=keep)
    assert "00core" not in m.tags["withheld"] and m.producer == {} and "00prov" in m.tags["withheld"]
    assert not any(k in m.data for k in ("propulsion", "ogma", "materials", "provenance"))
    yax.save(m, tmp_path / f"{name}.usda")
    back, _ = yax.read(tmp_path / f"{name}.usda")
    assert back.tags == m.tags and [lk.name for lk in back.links] == [lk.name for lk in m.links]
    assert set(tg.inventory(back)) - {"00core", "00fmt"} <= set(keep) | {c for c, v in tg.inventory(back).items() if v is None}


# ------------------------------------------------------------------------------------------------ YX2 (stage 0)
from yax.formats import yx2  # noqa: E402


def _yx2_round_trip(name, tmp_path, **sel):
    """Vegeta → YX2 → Yax → YX2: the second file is the first, byte for byte; what Yax read is what was selected."""
    f = tmp_path / f"{name}.yx2"
    m0, _ = export.machine(name, **sel)
    reps = export.write(m0, f)
    assert reps[0].dropped == [] and yx2.validate(f) == []
    m1, _ = yax.read(f)
    yax.save(m1, tmp_path / "again.yx2")
    assert (tmp_path / "again.yx2").read_text() == f.read_text()
    assert m1.data == m0.data and m1.producer == m0.producer and m1.tags["withheld"] == (m0.tags or {}).get("withheld", [])
    assert [lk.name for lk in m1.links] == [lk.name for lk in m0.links] and [j.name for j in m1.joints] == [j.name for j in m0.joints]
    for a, b in zip(m0.links, m1.links):
        assert a.mass == b.mass and tuple(a.com) == tuple(b.com) and (a.inertia is None) == (b.inertia is None)
        if a.inertia is not None:
            np.testing.assert_array_equal(a.inertia, b.inertia)
        assert [(s.name, s.kind, s.role) for s in a.shapes] == [(s.name, s.kind, s.role) for s in b.shapes]
    for a, b in zip(m0.joints, m1.joints):
        assert (a.kind, a.parent, a.child, a.lower, a.upper, a.drive, a.data) == (b.kind, b.parent, b.child, b.lower, b.upper, b.drive, b.data)
    assert tg.table(m1) == tg.table(m0) and tg.inventory(m1) == tg.inventory(m0)
    return f, m1


def test_yx2_is_the_main_file_and_keeps_every_selected_property(tmp_path):
    f, m = _yx2_round_trip("drongo", tmp_path)
    text = f.read_text()
    assert 'dictionary yx2' in text and 'string yx2:id = "link:drongo"' in text and '"withheld"' in text
    assert m.tags["withheld"] == [] and m.tags["table"]["data:propulsion/propellers"] == ["00motor", "boreas"] and "materials" in m.data
    f, m = _yx2_round_trip("drongo", tmp_path / "cut", drop=("00motor", "00prov"))
    text = f.read_text()
    assert "propulsion" not in text and "yax:producer" not in text and m.tags["withheld"][:2] == ["00motor", "00prov"]
    assert tg.inventory(m)["00motor"] is None and tg.inventory(m)["00matl"]
    piccolo = export.ROOT / "tools" / "consumers" / "piccolo.json"
    reps = export.export("drongo", tmp_path / "c" / "drongo.yx2", also=("sdf",), consumers=(piccolo,))
    assert [r.fmt for r in reps if isinstance(r, tg.Compatibility)] == ["yx2", "sdf"] and all(r.ok for r in reps if isinstance(r, tg.Compatibility))
    assert export.main(["drongo", "--to", str(tmp_path / "m" / "d.yx2"), "--also", "usda", "sdf", "--drop", "00inert"]) == 0
    assert yx2.validate(tmp_path / "m" / "d.yx2") == [] and (tmp_path / "m" / "d.usda").exists() and (tmp_path / "m" / "d.sdf").exists()
    assert "<inertia>" not in (tmp_path / "m" / "d.sdf").read_text()


@pytest.mark.parametrize("name", sorted(export.registry().names()))
def test_every_machine_round_trips_through_yx2(name, tmp_path):
    _yx2_round_trip(name, tmp_path)
    _yx2_round_trip(name, tmp_path / "structure", keep=("00geom", "00mass", "00inert", "00joint"))
