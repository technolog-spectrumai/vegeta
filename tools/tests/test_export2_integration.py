"""The whole data path on one mock machine holding every engineering block: mock Vegeta object → tools/export.py →
Yax property model → YX2 and USD → Yax → consumer validation (Piccolo's and Ukko's requirement files). The expected
blocks are the stage fixtures; nothing after the mock producer is mocked. No solver runs."""
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

FIX = Path(__file__).parent / "fixtures" / "export2"
MOTOR = json.loads((FIX / "mock_motor_vegeta.json").read_text())
PROP = json.loads((FIX / "mock_propulsion_vegeta.json").read_text())["data:propulsion"]
MAT = json.loads((FIX / "mock_assembly_materials.json").read_text())
AERO = json.loads((FIX / "mock_wing_aero.json").read_text())
DET = json.loads((FIX / "mock_detachable.json").read_text())


def _mock_robot():
    from ogma import actuators as act
    from vegeta.chiron import Geom, Joint, Link, Robot, Servo
    unit = act.Actuator("mock gearmotor 20:1", "gearmotor", 120.0, 2.5, 0.8, 12.0, 24.0, 6000.0, False,
                        "mock datasheet (the brief's example): a 20:1 planetary, efficiency 0.85", 20.0, 0.85)
    if not any(a.key == unit.key for a in act.CATALOG):
        act.CATALOG.append(unit)
    servo = Servo.from_actuator(unit, kp=30.0, kd=2.0, armature=0.048)
    prop = Link("prop", pos=(0.0, 0.0, 0.016), joints=[Joint("motor_prop", axis=(0, 0, 1), range=None)], geoms=[Geom("disk", "cylinder", (0.0635, 0.0005), mass=0.005, role="visual")])
    motor = Link("motor", pos=(0.11, 0.0, 0.005), geoms=[Geom("can", "cylinder", (0.014, 0.008), mass=0.03)], children=[prop])
    arm = Link("arm", pos=(0.09, 0.09, 0.0), joints=[Joint("elbow", axis=(0, 0, 1), range=(-1.5, 1.5), servo=servo)],
               geoms=[Geom("tube", "capsule", (0.006, 0.055), mass=0.03)], children=[motor])
    wing = Link("wing", pos=(0.0, 0.0, 0.05), geoms=[Geom("plank", "box", (0.1, 0.8, 0.005), mass=0.4)])
    panel = Link("panel", pos=(0.1, 0.0, 0.03), geoms=[Geom("sheet", "box", (0.05, 0.05, 0.001), mass=0.12)])
    root = Link("body", geoms=[Geom("hub", "box", (0.04, 0.04, 0.015), mass=0.6)], children=[arm, wing, panel])
    return Robot("mockall", root=root, notes="every block, mocked", sources={"all": "the stage fixtures"})


def _parts(m):
    from vegeta.labkit import detach as ld
    mats = json.loads(json.dumps(MAT["data:materials"]))
    mats.update(json.loads(json.dumps(DET["data:materials"])))
    m.link("body").data["material"] = "petg-cf"
    m.link("panel").data["material"] = "cfrp-t300"
    m.link("body").data["strength"] = json.loads(json.dumps(MAT["link:frame/data:strength"]))
    m.link("wing").data["aero"] = json.loads(json.dumps(AERO["link:wing/data:aero"]))
    conns = {k: dict(v, links=["body", "panel"], frame="link:body") for k, v in MAT["data:connections"].items()}
    conns.update(json.loads(json.dumps(DET["data:connections"])))
    g = DET["data:assembly"]["components"]
    det = {n: {"connection": g[n]["connection"], "separation": g[n]["separation"], "source": g[n]["source"]} for n in ("arm", "motor", "prop")}
    asm = ld.from_joint_tree(m, name="mock all", source="mock", kinds={"body": "body", "arm": "arm", "motor": "motor", "prop": "propeller", "wing": "wing", "panel": "panel"}, detachable=det)
    prop = json.loads(json.dumps(PROP))
    prop["rotors"][0]["link"] = "prop"
    return {"propulsion": prop, "materials": mats, "connections": conns, "aerodynamics": json.loads(json.dumps(AERO["data:aerodynamics"])), "assembly": asm,
            "tags": {"data:propulsion/propellers*": "mockbemt", "data:aerodynamics*": "cfd", "link:wing/data:aero*": "analytic"}}


@pytest.fixture(scope="module")
def registered():
    from vegeta.labkit import machines
    machines.register("mockall", _mock_robot, parts=_parts)
    return machines


def test_the_whole_data_path(registered, tmp_path):
    m, _ = export.machine("mockall")
    assert schema.problems(m) == []
    inv = tg.inventory(m)
    for tag in ("00core", "00geom", "00mass", "00inert", "00joint", "00motor", "00prop", "00mat", "00str", "00conn", "00aero", "00detach", "00prov",
                "mockbemt", "cfd", "analytic"):
        assert inv.get(tag), tag
    piccolo, ukko = export.ROOT / "tools" / "consumers" / "piccolo.json", export.ROOT / "tools" / "consumers" / "ukko.json"
    reps = export.write(m, tmp_path / "all.yx2", also=("usda", "sdf", "xml"), consumers=(piccolo, ukko))
    verdicts = [r for r in reps if isinstance(r, tg.Compatibility)]
    assert [(v.requirements.consumer, v.fmt, v.ok) for v in verdicts] == [("piccolo", "yx2", True), ("piccolo", "usd", True), ("piccolo", "sdf", True), ("piccolo", "mjcf", True),
                                                                        ("ukko", "yx2", True), ("ukko", "usd", True), ("ukko", "sdf", False), ("ukko", "mjcf", False)]
    assert verdicts[6].statuses["00detach"].status == tg.UNSUPPORTED and verdicts[6].statuses["00conn"].status == tg.UNSUPPORTED   # SDF: read them from the YX2
    for f in ("all.yx2", "all.usda"):
        back, _ = yax.read(tmp_path / f)
        assert schema.problems(back) == []
        j = next(j for j in back.joints if j.name == "elbow")
        assert j.data["motor"]["stall_torque_nm"] == 2.5 and j.data["motor"]["transmission"]["reflected_inertia_kgm2"] == 0.048
        assert back.data["motors"]["mock gearmotor 20:1"]["transmission"]["gear_ratio"] == 20.0
        assert back.data["propulsion"]["propellers"]["p5"]["coefficients"]["CQ"] == PROP["propellers"]["p5"]["coefficients"]["CQ"]
        assert back.data["materials"]["cfrp-t300"]["orthotropic"]["E1_pa"] == 135e9 and back.link("body").data["strength"]["safety_factor"] == 1.5
        assert back.data["connections"]["frame-panel-bolts"]["fasteners"]["preload_n"] == 1200.0
        assert back.link("wing").data["aero"]["tables"][0]["values"]["CD"][2] == [0.02, 0.018] and back.data["aerodynamics"]["coefficients"]["CL"] == 0.42
        assert back.data["assembly"]["components"]["prop"]["separation"]["limits"]["tension_n"] == 60.0 and back.data["assembly"]["components"]["prop"]["parent"] == "motor"
        assert back.link("prop").mass == 0.005 and back.producer["machine"] == "mockall" and back.data["provenance"]["sources"] == {"all": "the stage fixtures"}
        assert tg.inventory(back) == inv
    assert (tmp_path / "all.yx2").read_text() == yax.formats.yx2.text(yax.read(tmp_path / "all.yx2")[0])
    sdf_rep = reps[2]
    for loss in ("joint elbow: its motor model", "data:propulsion: the propellers' maps", "data:materials: the materials catalogue", "data:connections: the connections",
                 "link body: its strength limits", "link wing: its aerodynamics", "data:aerodynamics: the aerodynamics", "data:assembly: the assembly"):
        assert any(loss in d for d in sdf_rep.dropped), loss
    sdf = (tmp_path / "all.sdf").read_text()
    for leak in ("stall_torque", "preload", "tsai_wu", "CLalpha", "separation", "CQ"):
        assert leak not in sdf, leak


def test_selective_export_of_the_whole_machine(registered, tmp_path):
    m, _ = export.machine("mockall", keep=("00core", "00geom", "00mass", "00inert", "00conn", "00detach"))
    assert set(m.data) == {"connections", "assembly"} and all(k not in m.link("body").data for k in ("material", "strength"))
    assert not any("motor" in j.data for j in m.joints)
    assert sorted(schema.problems(m)) == sorted([                                # dependents of the withheld catalogue, each named
        "data:connections/panel-bond: material 'al6061-t6' is not in data['materials']",
        "data:assembly/components/body: material 'petg-cf' is not in data['materials']",
        "data:assembly/components/panel: material 'cfrp-t300' is not in data['materials']"])
    export.write(m, tmp_path / "ukko.yx2", consumers=(export.ROOT / "tools" / "consumers" / "ukko.json",))
    text = (tmp_path / "ukko.yx2").read_text()
    for leak in ("stall_torque", "youngs_modulus", "CLalpha", "kv_rpm_per_v", "allowable_stress"):
        assert leak not in text, leak
    assert {"00motor", "00prop", "00mat", "00str", "00aero", "00prov", "mockbemt", "cfd", "analytic"} <= set(m.tags["withheld"])
    back, _ = yax.read(tmp_path / "ukko.yx2")
    assert tg.check(back, tg.Requirements.load(export.ROOT / "tools" / "consumers" / "ukko.json")).ok
