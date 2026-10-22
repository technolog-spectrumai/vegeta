"""Stage 1A — motors, actuators, propulsion (docs/engineering.md is the oracle). Mock specifications with known
numbers go into a tiny machine, through the real tagging, USD/YX2 serialization and deserialization, and come back
compared field by field against values written here, not against the object that went in; losses are reported by
every format that cannot carry them; invalid blocks are refused with their reason. No motor or propeller runs."""
import json
from pathlib import Path

import numpy as np
import pytest

import yax
from yax import hub, schema
from yax import tags as tg

HERE = Path(__file__).parent
GOLD = HERE / "fixtures" / "export2"

#: The brief's mock motor, exactly as a datasheet would give it; the oracle says what the file must then hold.
MOTOR = {"name": "mock-gearmotor-20", "kind": "gearmotor", "stall_torque_nm": 2.5, "no_load_speed_rpm": 6000, "rated_torque_nm": 0.8,
         "rated_speed_rpm": 5200, "voltage_v": 24.0, "stall_current_a": 12.0, "no_load_current_a": 0.3, "resistance_ohm": 2.0,
         "kt_nm_per_a": 0.2083, "rotor_inertia_kgm2": 0.00012, "mechanical_time_constant_s": 0.015,
         "torque_speed": {"speed_rpm": [0, 1500, 3000, 4500, 6000], "torque_nm": [2.5, 1.9, 1.25, 0.6, 0.0], "current_a": [12.0, 9.1, 6.0, 3.1, 0.3],
                          "voltage_v": 24.0, "source": "mock datasheet curve (linear line read at five points)"},
         "limits": {"max_torque_nm": 2.5, "continuous_torque_nm": 0.8, "max_speed_rpm": 6000, "max_current_a": 12.0, "max_temperature_c": 85},
         "transmission": {"type": "planetary", "gear_ratio": 20, "efficiency": 0.85, "backlash_rad": 0.004, "max_torque_nm": 50.0,
                          "stages": [{"ratio": 4}, {"ratio": 5}], "source": "mock gearbox datasheet"},
         "source": "mock datasheet (the brief's example)", "validity": {"voltage_v": [20.0, 25.2], "note": "room temperature"}}


def _machine():
    base = hub.Link("base", mass=1.0, inertia=np.eye(3) * 0.01, shapes=[hub.Shape("box", "box", (0.1, 0.1, 0.1))])
    arm = hub.Link("arm", pos=(0.1, 0.0, 0.0), mass=0.2, inertia=np.eye(3) * 0.001, shapes=[hub.Shape("rod", "capsule", (0.01, 0.1))])
    j = hub.Joint("elbow", "revolute", "base", "arm", pos=(0.1, 0.0, 0.0), axis=(0.0, 0.0, 1.0), lower=-1.5, upper=1.5,
                  drive={"stiffness": 30.0, "damping": 2.0, "max_force": 2.5}, data={"motor": json.loads(json.dumps(MOTOR))})
    m = hub.Machine("mockarm", links=[base, arm], joints=[j], producer={"tool": "test", "revision": "0"})
    tg.tag(m, "joint:elbow/data:motor", "drive")
    return m


def test_mock_motor_through_usd_matches_the_oracle(tmp_path):
    """Export → USD → import: every value as written in this test (not as read back), both tags, nothing extra."""
    m = _machine()
    assert schema.problems(m) == []
    for fmt in ("usda", "yx2"):
        f = tmp_path / f"arm.{fmt}"
        yax.save(m, f)
        back, _ = yax.read(f)
        mo = back.joints[0].data["motor"]
        assert mo["name"] == "mock-gearmotor-20" and mo["kind"] == "gearmotor"
        assert mo["stall_torque_nm"] == 2.5 and mo["no_load_speed_rpm"] == 6000 and mo["rated_torque_nm"] == 0.8
        assert mo["transmission"]["gear_ratio"] == 20 and mo["transmission"]["efficiency"] == 0.85
        assert mo["rotor_inertia_kgm2"] == 0.00012 and mo["voltage_v"] == 24.0 and mo["resistance_ohm"] == 2.0
        assert mo["torque_speed"]["speed_rpm"] == [0, 1500, 3000, 4500, 6000] and mo["torque_speed"]["torque_nm"] == [2.5, 1.9, 1.25, 0.6, 0.0]
        assert mo["limits"]["max_temperature_c"] == 85 and mo["validity"]["voltage_v"] == [20.0, 25.2]
        assert set(mo) == set(MOTOR)                                           # no unrelated properties appear
        assert "no_load_speed_rad_s" not in mo and "rated_speed_rad_s" not in mo   # units as given, not converted
        t = tg.table(back)                                                    # plain USD implies the system tags; YX2 materialises them
        assert t["joint:elbow/data:motor"] == ["00motor", "drive"] and t["joint:elbow/data:motor/transmission/gear_ratio"] == ["00motor", "drive"]
        if fmt == "yx2":
            assert back.tags["table"]["joint:elbow/data:motor"] == ["00motor", "drive"]
        assert back.joints[0].drive == {"stiffness": 30.0, "damping": 2.0, "max_force": 2.5}   # the PD stays a PD
        assert schema.problems(back) == []
    # the serialized text holds the numbers as written
    text = (tmp_path / "arm.yx2").read_text().replace('\\"', '"')             # usda escapes the quotes of a JSON string
    for s in ('"stall_torque_nm":2.5', '"no_load_speed_rpm":6000', '"gear_ratio":20', '"efficiency":0.85', '"rotor_inertia_kgm2":0.00012'):
        assert s in text, s


def test_motor_golden_fixture_is_what_the_file_holds():
    """The expected joint block, written by hand in the fixture, is exactly the block the hub reads from the file."""
    gold = json.loads((GOLD / "mock_motor.json").read_text())
    m = _machine()
    stage = hub.to_stage(m)
    back = hub.from_stage(stage)
    assert back.joints[0].data["motor"] == gold["joint:elbow/data:motor"]
    assert tg.table(back)["joint:elbow/data:motor"] == gold["tags"]["joint:elbow/data:motor"]


def test_catalogue_motor_shared_by_joints(tmp_path):
    m = _machine()
    m.data["motors"] = {"m20": json.loads(json.dumps(MOTOR))}
    m.joints[0].data["motor"] = "m20"
    m.joints.append(hub.Joint("wrist", "revolute", "arm", "arm", pos=(0.2, 0, 0), axis=(0, 1, 0), data={"motor": "m20"}))
    m.links.append(hub.Link("hand", pos=(0.2, 0, 0), mass=0.05, inertia=np.eye(3) * 1e-4))
    m.joints[-1].child = "hand"
    assert schema.problems(m) == []
    paths = {p.path: p.system for p in tg.properties(m)}
    assert paths["data:motors/m20"] == ("00motor",) and paths["data:motors/m20/torque_speed"] == ("00motor",) and paths["joint:wrist/data:motor"] == ("00motor",)
    yax.save(m, tmp_path / "cat.yx2")
    back, _ = yax.read(tmp_path / "cat.yx2")
    assert back.data["motors"]["m20"]["transmission"]["gear_ratio"] == 20 and back.joints[1].data["motor"] == "m20"
    m.joints[1].data["motor"] = "m99"
    assert schema.problems(m) == ["joint:wrist/data:motor: motor 'm99' is not in data['motors']"]


def test_invalid_and_incomplete_motors_are_refused_with_reasons():
    m = _machine()
    mo = m.joints[0].data["motor"]
    mo["no_load_speed_rad_s"] = 628.3                                          # both speed units
    mo["torque_speed"]["torque_nm"] = [2.5, 1.9]                               # ragged curve
    mo["transmission"]["efficiency"] = 1.2
    mo["stall_torque_nm"] = "2.5"
    mo["peak_power"] = 60.0                                                    # a number without a unit
    p = schema.problems(m)
    assert any("not both" in x for x in p) and any("one length" in x for x in p) and any("efficiency must be in (0, 1]" in x for x in p)
    assert any("stall_torque_nm must be a number" in x for x in p) and any("peak_power: a number needs a unit suffix" in x for x in p)
    m = _machine()
    del m.joints[0].data["motor"]["torque_speed"]                              # optional: absent is fine, never defaulted
    del m.joints[0].data["motor"]["transmission"]
    assert schema.problems(m) == [] and "torque_speed" not in m.joints[0].data["motor"]
    m.joints[0].data["motor"] = {"stall_torque_nm": -1}
    assert schema.problems(m) == ["joint:elbow/data:motor: stall_torque_nm must be ≥ 0"]
    m.joints[0].data["motor"] = {"torque_speed": {"torque_nm": [1, 2]}}
    assert any("speed_rpm or speed_rad_s is required" in x for x in schema.problems(m))


def test_losses_are_reported_never_silent(tmp_path):
    m = _machine()
    for ext, keeps in (("xml", "the PD drive written is not the motor model"), ("urdf", "effort limit"), ("sdf", "Gazebo plugin")):
        rep = yax.save(m, tmp_path / f"arm.{ext}")
        line = next(d for d in rep.dropped if d.startswith("joint elbow: its motor model"))
        assert "stall_torque_nm, no_load_speed_rpm, torque_speed, transmission, limits" in line and keeps in line
    text = (tmp_path / "arm.xml").read_text()
    assert 'kp="30"' in text and "6000" not in text                            # the PD went out; the motor did not pretend to
    rep = yax.save(m, tmp_path / "arm.usda")
    assert rep.dropped == []


def test_selection_by_tag_withholds_the_motor_without_a_trace(tmp_path):
    m = _machine()
    cut = tg.select(m, drop=("00motor",))
    assert "motor" not in cut.joints[0].data and cut.joints[0].drive is None
    assert cut.tags["withheld"] == ["00motor", "drive"]
    for ext in ("yx2", "usda", "xml", "sdf"):
        yax.save(cut, tmp_path / f"cut.{ext}")
        text = (tmp_path / f"cut.{ext}").read_text()
        assert "6000" not in text and "0.85" not in text and "mock-gearmotor" not in text and "planetary" not in text, ext
    kept = tg.select(m, keep=("00core", "00geom", "00mass", "00inert", "00joint", "drive"))
    assert kept.joints[0].data["motor"]["stall_torque_nm"] == 2.5                 # a user tag keeps the property
    v = tg.check(yax.read(tmp_path / "cut.yx2")[0], tg.Requirements(required=("00motor",), optional=("drive",)))
    assert not v.ok and v.statuses["00motor"].status == tg.WITHHELD and v.statuses["drive"].status == tg.WITHHELD


def test_propulsion_block_with_torque_coefficient_and_references(tmp_path):
    """A mock propeller with CT, CQ and CP against J, a geometry reference to its link, a motor from the catalogue."""
    m = _machine()
    m.links.append(hub.Link("prop0", pos=(0.1, 0.1, 0.05), mass=0.004, inertia=np.diag([1e-6, 1e-6, 2e-6]),
                            shapes=[hub.Shape("disk", "cylinder", (0.0635, 0.001), role="visual")]))
    m.joints.append(hub.Joint("prop0_spin", "continuous", "base", "prop0", pos=(0.1, 0.1, 0.05), axis=(0, 0, 1)))
    block = {"rotors": [{"name": "r0", "link": "prop0", "pos": [0.1, 0.1, 0.05], "axis": [0, 0, 1], "spin": 1, "propeller": "p5", "motor": "m2306"}],
             "propellers": {"p5": {"blades": 3, "diameter_m": 0.127, "hub_radius_m": 0.006, "mass_kg": 0.004, "inertia_kgm2": 2e-6, "geometry_ref": "prop0",
                                   "coefficients": {"J": [0.0, 0.2, 0.4, 0.6], "CT": [0.12, 0.10, 0.07, 0.03], "CQ": [0.011, 0.010, 0.008, 0.005],
                                                    "CP": [0.069, 0.063, 0.050, 0.031], "rho_ref_kg_m3": 1.225, "rpm_band": [8000, 24000], "source": "mock BEMT"},
                                   "grid": {"rpm": [10000, 20000], "airspeed_m_s": [0.0, 5.0, 10.0], "rho": 1.225,
                                            "thrust_n": [[1.2, 1.0, 0.7], [4.8, 4.3, 3.6]], "torque_nm": [[0.015, 0.013, 0.010], [0.060, 0.055, 0.047]]},
                                   "source": "mock", "validity": {"rpm": [8000, 24000]}}},
             "motors": {"m2306": {"name": "2306-2400KV", "kind": "bldc", "kv_rpm_per_v": 2400, "resistance_ohm": 0.06, "no_load_current_a": 1.2,
                                  "max_current_a": 40, "rotor_inertia_kgm2": 1.5e-6, "limits": {"max_power_w": 330.0}, "source": "mock"}},
             "source": "mock propulsion"}
    m.data["propulsion"] = json.loads(json.dumps(block))
    assert schema.problems(m) == []
    paths = {p.path: p.system for p in tg.properties(m)}
    assert paths["data:propulsion/propellers/p5/coefficients"] == ("00prop",) and paths["data:propulsion/motors/m2306"] == ("00prop", "00motor")
    assert paths["data:propulsion/motors/m2306/kv_rpm_per_v"] == ("00prop", "00motor")
    yax.save(m, tmp_path / "quad.yx2")
    back, _ = yax.read(tmp_path / "quad.yx2")
    pe = back.data["propulsion"]["propellers"]["p5"]
    assert pe["coefficients"]["CQ"] == [0.011, 0.010, 0.008, 0.005] and pe["grid"]["thrust_n"][1] == [4.8, 4.3, 3.6] and pe["geometry_ref"] == "prop0"
    assert back.data["propulsion"]["motors"]["m2306"]["kv_rpm_per_v"] == 2400 and pe["validity"] == {"rpm": [8000, 24000]}
    gold = json.loads((GOLD / "mock_propulsion.json").read_text())
    assert back.data["propulsion"] == gold["data:propulsion"]
    # losses: SDF and MJCF say what they cannot carry; JSBSim says what of the block does not reach it
    rep = yax.save(m, tmp_path / "quad.sdf")
    assert any(d.startswith("data:propulsion: the propellers' maps (1 propellers, 1 rotors)") for d in rep.dropped)
    rep = yax.save(m, tmp_path / "quad.xml")
    assert any("data:propulsion: the propellers' maps" in d for d in rep.dropped)
    # references are checked
    m.data["propulsion"]["rotors"][0]["propeller"] = "p9"
    m.data["propulsion"]["propellers"]["p5"]["geometry_ref"] = "nowhere"
    m.data["propulsion"]["propellers"]["p5"]["coefficients"]["J"] = [0.0, 0.4, 0.2, 0.6]
    p = schema.problems(m)
    assert "data:propulsion/rotors/0: propeller 'p9' is not in the block" in p
    assert "data:propulsion/propellers/p5: geometry_ref 'nowhere' names no link or shape" in p
    assert "data:propulsion/propellers/p5/coefficients: J must be increasing" in p
    only = tg.select(m, keep=("00core", "00prop"))
    assert "motor" not in only.joints[0].data and "propulsion" in only.data and only.data["propulsion"]["motors"]   # 00prop keeps its motors
    none = tg.select(m, drop=("00prop",))
    assert "propulsion" not in none.data and "motor" in none.joints[0].data           # the joint motor is 00motor alone


def test_cli_validate(tmp_path, capsys):
    from yax import cli
    m = _machine()
    yax.save(m, tmp_path / "arm.yx2")
    assert cli.main(["validate", str(tmp_path / "arm.yx2")]) == 0 and "ok" in capsys.readouterr().out
    m.joints[0].data["motor"]["transmission"]["gear_ratio"] = -1
    yax.save(m, tmp_path / "bad.yx2")
    assert cli.main(["validate", str(tmp_path / "bad.yx2")]) == 1
    assert "gear_ratio must be > 0" in capsys.readouterr().out
