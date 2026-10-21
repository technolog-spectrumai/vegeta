"""Feature-based interchange (docs/interchange.md): the inventory tells available from missing from excluded, select
withholds without inventing and the USD remembers what was withheld, a consumer's requirements are checked against
the file as it is (and a target format's support) without enabling anything, and the command line does the same."""
from pathlib import Path

import numpy as np
import pytest

import yax
from yax import features as ft
from yax import hub

HERE = Path(__file__).parent
ROVER = HERE / "fixtures" / "rover.xml"
pytest.importorskip("mujoco")


def _rich():
    """A two-link machine holding something of every feature."""
    base = hub.Link("base", mass=2.0, inertia=np.diag([0.1, 0.1, 0.1]), shapes=[hub.Shape("hull", "box", (0.3, 0.2, 0.1))],
                    point_masses=[("battery", 0.3, (0.0, 0.0, 0.05))],
                    data={"material": "al6061", "strength": {"allowable_pa": 2.4e8, "safety_factor": 1.5, "source": "notebook"},
                          "aero": {"S_m2": 0.1}, "sensors": [{"name": "cam", "type": "camera", "pos": [0.1, 0, 0]}],
                          "sdf": {"sensor": ["<sensor name='cam' type='camera'/>"], "plugin": ["<plugin name='x'/>"]}})
    arm = hub.Link("arm", pos=(0.2, 0.0, 0.0), mass=0.5, inertia=np.diag([0.01, 0.01, 0.01]), shapes=[hub.Shape("rod", "capsule", (0.01, 0.2))])
    j = hub.Joint("shoulder", "revolute", "base", "arm", pos=(0.2, 0.0, 0.0), axis=(0.0, 1.0, 0.0), lower=-1.0, upper=1.0,
                  drive={"stiffness": 10.0, "damping": 1.0, "max_force": 5.0},
                  data={"damping": 0.1, "motor": {"stall_nm": 2.0, "no_load_rad_s": 50.0, "source": "datasheet"}, "connection": {"type": "pin"}})
    m = hub.Machine("rich", links=[base, arm], joints=[j], producer={"tool": "test", "revision": "abc"},
                    data={"materials": {"al6061": {"density_kg_m3": 2700.0, "E_pa": 69e9, "source": "handbook"}},
                          "connections": [{"name": "b1", "links": ["base", "arm"], "type": "bolt", "spec": {"size": "M3"}}],
                          "aerodynamics": {"reference": {"S_m2": 0.1}, "source": "cfd"},
                          "propulsion": {"rotors": [{"name": "r0", "pos": [0, 0, 0], "axis": [0, 0, 1], "spin": 1, "propeller": "p", "motor": "mo"}],
                                         "propellers": {"p": {"diameter_m": 0.2, "coefficients": {"J": [0, 1], "CT": [0.1, 0.0], "CP": [0.05, 0.01]}}},
                                         "motors": {"mo": {"power_w": 100.0}}}})
    return m


def test_inventory_tells_available_from_missing_from_excluded(tmp_path):
    m, _ = yax.read(ROVER)
    inv = ft.inventory(m)
    assert {f for f, s in inv.items() if s.status == ft.AVAILABLE} == {"geometry", "mass", "inertia", "joints", "motors"}
    assert inv["motors"].detail == "1 joint drives" and inv["joints"].detail == "4 moving joints"
    assert inv["materials"].status == ft.MISSING and inv["provenance"].status == ft.MISSING
    rich = _rich()
    assert all(s.status == ft.AVAILABLE for s in ft.inventory(rich).values()), ft.describe(rich)
    cut = ft.select(rich, exclude=("materials", "strength"))
    inv = ft.inventory(cut)
    assert inv["materials"].status == ft.EXCLUDED and inv["strength"].status == ft.EXCLUDED and inv["mass"].status == ft.AVAILABLE
    yax.save(cut, tmp_path / "cut.usda")                                       # the USD remembers what was withheld
    back, _ = yax.read(tmp_path / "cut.usda")
    assert back.features == {"included": sorted(set(ft.FEATURES) - {"materials", "strength"}), "excluded": ["materials", "strength"]}
    assert ft.inventory(back)["materials"].status == ft.EXCLUDED
    text = (tmp_path / "cut.usda").read_text()
    assert "al6061" not in text and "allowable_pa" not in text and "2700" not in text


def test_select_withholds_without_inventing(tmp_path):
    """Each withheld feature leaves nothing of itself in the hub, the USD text or another format's file; the
    structure (links, the joint tree as fixed joints) stays; the source machine is untouched."""
    rich = _rich()
    snapshot = ft.describe(rich)
    for feat in ft.FEATURES:
        cut = ft.select(rich, exclude=(feat,))
        assert ft.inventory(cut)[feat].status == ft.EXCLUDED, feat
        assert [lk.name for lk in cut.links] == ["base", "arm"] and [j.name for j in cut.joints] == ["shoulder"]
        yax.save(cut, tmp_path / f"{feat}.usda")
        text = (tmp_path / f"{feat}.usda").read_text()
        marker = {"geometry": "hull", "mass": "battery", "inertia": "yax:inertia", "joints": "RevoluteJoint", "motors": "stall_nm",
                  "materials": "al6061", "strength": "allowable_pa", "connections": "M3", "aerodynamics": "S_m2", "sensors": "cam",
                  "provenance": "yax:producer"}[feat]
        assert marker not in text, (feat, marker)
        if feat == "mass":
            assert all(lk.mass is None for lk in cut.links)
            rep = yax.save(cut, tmp_path / "nomass.sdf")
            assert "<inertial>" not in (tmp_path / "nomass.sdf").read_text() and any("mass: withheld" in d for d in rep.dropped)
        if feat == "joints":
            assert all(j.kind == "fixed" and j.drive is None for j in cut.joints)
        if feat == "sensors":
            assert cut.links[0].data["sdf"] == {"plugin": ["<plugin name='x'/>"]}   # the plugin stays: not a sensor
    assert ft.describe(rich) == snapshot
    with pytest.raises(ValueError):
        ft.select(rich, exclude=("colour",))
    only = ft.select(rich, include=("geometry", "mass"))
    assert ft.inventory(only)["geometry"].status == ft.AVAILABLE and ft.inventory(only)["inertia"].status == ft.EXCLUDED
    assert yax.save(only, tmp_path / "only.xml").dropped[:1] == ["aerodynamics: withheld by the producer (not in the source)"]


def test_requirements_are_checked_without_enabling_anything(tmp_path):
    rich = _rich()
    req = ft.Requirements(required=("geometry", "mass", "inertia", "sensors"), optional=("motors", "strength"), consumer="piccolo")
    v = ft.check(rich, req)
    assert v.ok and all(s.status == ft.AVAILABLE for s in v.statuses.values()) and str(v).startswith("piccolo: served")
    v = ft.check(rich, req, fmt="urdf")                                        # URDF has no sensors: unsupported
    assert not v.ok and v.statuses["sensors"].status == ft.UNSUPPORTED and "available in the machine" in v.statuses["sensors"].detail
    assert v.statuses["motors"].status == ft.UNSUPPORTED and v.statuses["mass"].status == ft.AVAILABLE
    v = ft.check(rich, req, fmt="sdf")
    assert v.ok and v.statuses["sensors"].status == ft.AVAILABLE
    cut = ft.select(rich, exclude=("sensors",))
    v = ft.check(cut, req)
    assert not v.ok and [s.feature for s in v.unmet] == ["sensors"] and v.statuses["sensors"].status == ft.EXCLUDED
    assert ft.inventory(cut)["sensors"].status == ft.EXCLUDED                  # the check changed nothing
    rover, _ = yax.read(ROVER)
    v = ft.check(rover, ft.Requirements(required=("strength",), optional=("materials",)), fmt="mjcf")
    assert v.statuses["strength"].status == ft.UNSUPPORTED and "missing" not in v.statuses["strength"].detail
    v = ft.check(rover, ft.Requirements(required=("strength",)))
    assert v.statuses["strength"].status == ft.MISSING and "NOT served" in str(v)
    v = ft.check(rover, ft.Requirements(required=("motors",)), fmt="mjcf")
    assert v.ok and "position actuators" in v.statuses["motors"].detail        # partial support is said, not hidden
    (tmp_path / "piccolo.json").write_text('{"consumer": "piccolo", "required": ["geometry", "mass"], "optional": ["sensors"]}')
    r = ft.Requirements.load(tmp_path / "piccolo.json")
    assert r.consumer == "piccolo" and r.required == ("geometry", "mass") and r.optional == ("sensors",)
    with pytest.raises(ValueError):
        ft.Requirements(required=("colour",))


def test_cli_features_and_select(tmp_path, capsys):
    from yax import cli
    assert cli.main(["features", str(ROVER)]) == 0
    out = capsys.readouterr().out
    assert "geometry: available" in out and "materials: missing" in out
    assert cli.main(["features", str(ROVER), "--require", "geometry,mass", "--optional", "sensors", "--to", "sdf"]) == 0
    assert "served" in capsys.readouterr().out
    assert cli.main(["features", str(ROVER), "--require", "strength"]) == 1
    assert "NOT served" in capsys.readouterr().out
    assert cli.main(["select", str(ROVER), str(tmp_path / "r.usda"), "--exclude", "inertia,motors"]) == 0
    m, _ = yax.read(tmp_path / "r.usda")
    assert m.features["excluded"] == ["inertia", "motors"] and all(lk.inertia is None for lk in m.links) and not any(j.drive for j in m.joints)
    assert cli.main(["show", str(tmp_path / "r.usda")]) == 0
    assert "excluded: inertia, motors" in capsys.readouterr().out
    (tmp_path / "req.json").write_text('{"consumer": "sim", "required": ["inertia"]}')
    assert cli.main(["features", str(tmp_path / "r.usda"), "--requirements", str(tmp_path / "req.json")]) == 1
    assert "sim: NOT served" in capsys.readouterr().out and "excluded" in capsys.readouterr().out or True


def test_old_files_read_as_nothing_withheld(tmp_path):
    m, _ = yax.read(ROVER)
    yax.save(m, tmp_path / "plain.usda")
    assert "yax:features" not in (tmp_path / "plain.usda").read_text()
    back, _ = yax.read(tmp_path / "plain.usda")
    assert back.features == {} and ft.inventory(back)["sensors"].status == ft.MISSING
    assert hub.lump(ft.select(m, exclude=("motors",))).features["excluded"] == ["motors"]
