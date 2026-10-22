"""Stage 4B — detachable components through the real Vegeta → Yax export on a mock assembly: a Chiron robot with a
body, an arm, a motor and a propeller as links on fixed and spinning joints, whose parts function states which come
off, through which connection, with which separation limits (the fixture Yax's own tests use). The assembly comes
from the robot's joint tree (labkit.detach); the physics stays on the links. No MuJoCo simulation, no Ukko."""
import json
import sys
from pathlib import Path

import pytest

yax = pytest.importorskip("yax")
pytest.importorskip("mujoco")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import export  # noqa: E402
from yax import schema  # noqa: E402
from yax import tags as tg  # noqa: E402

GOLD = json.loads((Path(__file__).parent / "fixtures" / "export2" / "mock_detachable.json").read_text())


def _mock_robot():
    from vegeta.chiron import Geom, Joint, Link, Robot
    prop = Link("prop", pos=(0.0, 0.0, 0.016), joints=[Joint("motor_prop", axis=(0, 0, 1), range=None)],        # in the motor's frame
                geoms=[Geom("disk", "cylinder", (0.0635, 0.0005), mass=0.005, role="visual")])
    motor = Link("motor", pos=(0.11, 0.0, 0.005), geoms=[Geom("can", "cylinder", (0.014, 0.008), mass=0.03)], children=[prop])
    arm = Link("arm", pos=(0.09, 0.09, 0.0), quat=(0.9238795, 0, 0, 0.3826834), geoms=[Geom("tube", "capsule", (0.006, 0.055), mass=0.03)], children=[motor])
    root = Link("body", geoms=[Geom("hub", "box", (0.04, 0.04, 0.015), mass=0.6)], children=[arm])
    return Robot("mockquadarm", root=root, sources={"assembly": "mock CAD"})


def _parts(m):
    from vegeta.labkit import detach as ld
    g = GOLD["data:assembly"]["components"]
    for lk in m.links:
        if lk.name in ("body", "arm"):
            lk.data["material"] = "petg-cf"
    det = {name: {"connection": g[name]["connection"], "separation": g[name]["separation"], "source": g[name]["source"]} for name in ("arm", "motor", "prop")}
    asm = ld.from_joint_tree(m, name="mock quad arm", source="mock assembly", kinds={n: g[n]["kind"] for n in g}, detachable=det)
    return {"assembly": asm, "connections": json.loads(json.dumps(GOLD["data:connections"])), "materials": json.loads(json.dumps(GOLD["data:materials"])),
            "tags": {"data:assembly/components/prop*": "pulltest"}}


@pytest.fixture(scope="module")
def registered():
    from vegeta.labkit import machines
    machines.register("mockquadarm", _mock_robot, parts=_parts)
    return machines


def test_assembly_from_the_joint_tree_matches_the_oracle(registered, tmp_path):
    m, _ = export.machine("mockquadarm")
    assert schema.problems(m) == []
    asm = m.data["assembly"]
    g = GOLD["data:assembly"]["components"]
    assert asm["root"] == "body" and set(asm["components"]) == {"body", "arm", "motor", "prop"}
    for name in ("arm", "motor", "prop"):
        c = asm["components"][name]
        assert c["parent"] == g[name]["parent"] and c["detachable"] is True and c["connection"] == g[name]["connection"]
        assert c["separation"] == g[name]["separation"] and c["kind"] == g[name]["kind"]
        assert c["attachment"]["pos"] == pytest.approx(g[name]["attachment"]["pos"], abs=1e-9)   # the joint, in the parent's frame
    assert "detachable" not in asm["components"]["body"] and asm["components"]["body"]["material"] == "petg-cf"
    assert asm["components"]["prop"]["attachment"]["axis"] == pytest.approx([0, 0, 1], abs=1e-9)
    assert m.link("prop").mass == 0.005 and m.link("motor").mass == 0.03                        # physics on the links, not repeated
    export.write(m, tmp_path / "q.yx2", also=("usda", "sdf"))
    for f in ("q.yx2", "q.usda"):
        back, _ = yax.read(tmp_path / f)
        assert back.data["assembly"] == asm and back.data["connections"] == GOLD["data:connections"] and schema.problems(back) == []
        t = tg.table(back)
        assert t["data:assembly/components/prop/separation"] == ["00detach", "pulltest"] and t["data:assembly/components/arm"] == ["00detach"]
    assert "separation" not in (tmp_path / "q.sdf").read_text() and "<link name=\"prop\">" in (tmp_path / "q.sdf").read_text()


def test_ukko_style_selection_and_validation(registered, tmp_path):
    ukko = export.ROOT / "tools" / "consumers" / "ukko.json"
    m, _ = export.machine("mockquadarm", keep=("00core", "00geom", "00mass", "00inert", "00mat", "00conn", "00detach"))
    reps = export.write(m, tmp_path / "u.yx2", consumers=(ukko,))
    v = next(r for r in reps if isinstance(r, tg.Compatibility))
    assert v.ok and v.requirements.consumer == "ukko" and v.statuses["00detach"].status == tg.PRESENT
    assert "pulltest" in m.tags["withheld"] or "pulltest" in tg.inventory(m)              # a producer tag on kept properties stays
    m, _ = export.machine("mockquadarm", drop=("00detach",))
    assert "assembly" not in m.data and m.data["connections"]
    export.write(m, tmp_path / "n.yx2", consumers=(ukko,))
    text = (tmp_path / "n.yx2").read_text()
    assert "separation" not in text and "mock pull test" not in text
    v = tg.check(yax.read(tmp_path / "n.yx2")[0], tg.Requirements.load(ukko))
    assert not v.ok and v.statuses["00detach"].status == tg.WITHHELD
