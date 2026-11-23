"""Stage 1B — motors and propulsion through the real Vegeta → Yax export (tools/export.py), on mocks: a two-body Chiron
robot with one servo joint made from a mock Ogma actuator (the catalogue entry stated like Pekari's drive), and a mock
parts function with a propulsion block of fixed numbers. Everything after the mock producer is production code; the
expected blocks are written in tools/tests/fixtures/export2/*.json from vendor/yax/docs/engineering.md, not read back.
MuJoCo compiles the two-body model (milliseconds); no BEMT, CAD or simulation runs."""
import json
import math
import sys
from pathlib import Path

import pytest

yax = pytest.importorskip("yax")
pytest.importorskip("mujoco")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import export  # noqa: E402
from yax import schema  # noqa: E402
from yax import tags as tg  # noqa: E402

HERE = Path(__file__).parent
GOLD = HERE / "fixtures" / "export2"


def _mock_robot():
    """A body with one servo-driven arm. The servo is Chiron's model of the mock gearmotor; its numbers are the
    catalogue entry's at the output (stall 2.5 N m, 6000 rpm no-load … as the brief's example)."""
    from ogma import actuators as act
    from vegeta.chiron import Geom, Joint, Link, Robot, Servo
    unit = act.Actuator("mock gearmotor 20:1", "gearmotor", 120.0, 2.5, 0.8, 12.0, 24.0, 6000.0, False,
                        "mock datasheet (the brief's example): a 20:1 planetary, efficiency 0.85", 20.0, 0.85)
    if not any(a.key == unit.key for a in act.CATALOG):
        act.CATALOG.append(unit)
    servo = Servo.from_actuator(unit, kp=30.0, kd=2.0, armature=0.00012 * 20 ** 2)
    arm = Link("arm", pos=(0.1, 0.0, 0.0), joints=[Joint("elbow", axis=(0, 0, 1), range=(-1.5, 1.5), servo=servo)],
               geoms=[Geom("rod", "capsule", (0.01, 0.05), mass=0.2)])
    root = Link("body", geoms=[Geom("box", "box", (0.05, 0.05, 0.05), mass=1.0)], children=[arm])
    return Robot("mockarm", root=root, notes="a mock", sources={"servo": "mock datasheet"})


PROPULSION = json.loads((GOLD / "mock_propulsion_vegeta.json").read_text())["data:propulsion"]


def _parts(m):
    return {"propulsion": json.loads(json.dumps(PROPULSION)), "tags": {"data:propulsion/propellers*": "mockbemt"}}


@pytest.fixture(scope="module")
def registered():
    from vegeta.labkit import machines
    machines.register("mockarm", _mock_robot, parts=_parts)
    machines.register("mockarm_bare", _mock_robot)
    return machines


def test_servo_leaves_as_a_motor_block_not_a_pd_drive(registered):
    m, _ = export.machine("mockarm_bare")
    j = next(j for j in m.joints if j.name == "elbow")
    gold = json.loads((GOLD / "mock_motor_vegeta.json").read_text())
    mo = j.data["motor"]
    assert mo["stall_torque_nm"] == 2.5 and mo["rated_torque_nm"] == 0.8 and mo["voltage_v"] == 24.0 and mo["stall_current_a"] == 12.0
    assert mo["no_load_speed_rad_s"] == pytest.approx(6000 * 2 * math.pi / 60)   # Chiron holds the speed in rad/s: exported as such
    assert mo["transmission"] == {"reflected_inertia_kgm2": 0.048, "source": "the servo model's armature (rotor inertia × gear ratio²)"}
    assert mo["control"]["kp"] == 30.0 and mo["control"]["kd"] == 2.0 and mo["derived_from"] == "mock gearmotor 20:1"
    assert mo["kt_nm_per_a"] == pytest.approx(2.5 / 12.0) and mo["resistance_ohm"] == pytest.approx(24.0 / 12.0) and "estimates" in mo["source"]
    assert set(mo) == set(gold["joint:elbow/data:motor"])
    cat = m.data["motors"]["mock gearmotor 20:1"]
    assert cat == gold["data:motors/mock gearmotor 20:1"]                        # the bought unit: exactly the entry's numbers
    assert cat["no_load_speed_rpm"] == 6000.0 and cat["transmission"]["gear_ratio"] == 20.0 and cat["transmission"]["efficiency"] == 0.85
    assert j.drive == {"stiffness": 1.0, "damping": 2.0}                       # the PD Chiron wrote into MJCF stays what it was
    assert schema.problems(m) == []
    t = tg.table(m)
    assert t["joint:elbow/data:motor"] == ["00motor"] and t["data:motors/mock gearmotor 20:1/transmission/gear_ratio"] == ["00motor"]


def test_vegeta_to_yax_to_usd_to_yax(registered, tmp_path):
    m, _ = export.machine("mockarm")
    reps = export.write(m, tmp_path / "mockarm.yx2", also=("usda", "xml", "sdf"))
    gold = json.loads((GOLD / "mock_motor_vegeta.json").read_text())
    for f in ("mockarm.yx2", "mockarm.usda"):
        back, _ = yax.read(tmp_path / f)
        j = next(j for j in back.joints if j.name == "elbow")
        assert j.data["motor"]["stall_torque_nm"] == 2.5 and j.data["motor"]["transmission"]["reflected_inertia_kgm2"] == 0.048
        assert back.data["motors"]["mock gearmotor 20:1"] == gold["data:motors/mock gearmotor 20:1"]
        assert back.data["propulsion"] == PROPULSION                           # the design's block, as the fixture states it
        assert back.producer["machine"] == "mockarm" and schema.problems(back) == []
        t = tg.table(back)
        assert t["data:propulsion/propellers/p5/coefficients/CQ"] == ["00prop", "mockbemt"] and t["data:propulsion/motors/m2306"] == ["00motor", "00prop"]
    assert (tmp_path / "mockarm.yx2").read_text() == yax.formats.yx2.text(yax.read(tmp_path / "mockarm.yx2")[0])   # idempotent
    rep_mjcf = reps[2]
    assert any(d.startswith("joint elbow: its motor model (stall_torque_nm, no_load_speed_rad_s, transmission)") for d in rep_mjcf.dropped)
    assert any("data:propulsion: the propellers' maps (1 propellers, 1 rotors)" in d for d in rep_mjcf.dropped)
    assert any("data:motors: the motor catalogue (1 motors)" in d for d in rep_mjcf.dropped)


def test_tag_selection_withholds_motors_without_leaking(registered, tmp_path):
    m, _ = export.machine("mockarm", drop=("00motor",))
    j = next(j for j in m.joints if j.name == "elbow")
    assert "motor" not in j.data and j.drive is None and "motors" not in m.data
    assert "motors" not in m.data["propulsion"] and m.data["propulsion"]["propellers"]   # 00prop stays; its motors were 00motor too
    assert "00motor" in m.tags["withheld"]
    export.write(m, tmp_path / "cut.yx2", also=("xml",))
    for f in ("cut.yx2", "cut.xml"):
        text = (tmp_path / f).read_text()
        for marker in ("mock gearmotor", "kv_rpm_per_v", "0.048", "stall_torque"):
            i = text.find(marker)
            assert i < 0, (f, marker, text[max(0, i - 120):i + 60])
    m, _ = export.machine("mockarm", keep=("00core", "00geom", "00mass", "00inert", "00joint", "mockbemt"))
    assert "propulsion" in m.data and set(m.data["propulsion"]) == {"propellers"} and "motor" not in next(j for j in m.joints if j.name == "elbow").data
    with pytest.raises(ValueError):
        export.machine("mockarm", tag=(("joint:*/data:motor", "00mine"),))


def test_consumer_validation_of_the_received_file(registered, tmp_path):
    m, _ = export.machine("mockarm")
    piccolo = export.ROOT / "tools" / "consumers" / "piccolo.json"
    reps = export.write(m, tmp_path / "m.yx2", also=("sdf",), consumers=(piccolo,))
    v = [r for r in reps if isinstance(r, tg.Compatibility)]
    assert v[0].ok and v[0].statuses["00motor"].status == tg.PRESENT and v[1].statuses["00motor"].status == tg.UNSUPPORTED
    r = tg.Requirements(required=("00motor", "00prop"), optional=("mockbemt",), consumer="mock-sim")
    back, _ = yax.read(tmp_path / "m.yx2")
    v = tg.check(back, r)
    assert v.ok and "properties" in v.statuses["00prop"].detail and v.statuses["mockbemt"].status == tg.PRESENT
