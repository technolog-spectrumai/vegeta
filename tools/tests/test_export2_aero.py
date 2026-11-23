"""Stage 3B — aerodynamics through the real Vegeta → Yax export on a mock wing: a two-body Chiron robot (pod and wing)
whose parts function states the wing's aerodynamics with known numbers (the fixture Yax's own tests use) and the
machine's CFD coefficients; the labkit helper that reads an aero-lab design file is fed a tiny in-memory document.
No CFD, no Aeromant."""
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

GOLD = json.loads((Path(__file__).parent / "fixtures" / "export2" / "mock_wing_aero.json").read_text())


def _mock_robot():
    from vegeta.chiron import Geom, Joint, Link, Robot
    wing = Link("wing", pos=(0.0, 0.0, 0.05), joints=[Joint("wing_flex", axis=(1, 0, 0), range=(-0.05, 0.05))],
                geoms=[Geom("plank", "box", (0.1, 0.8, 0.005), mass=0.4)])
    root = Link("body", geoms=[Geom("pod", "capsule", (0.04, 0.15), mass=1.5)], children=[wing])
    return Robot("mockplane", root=root, sources={"aerodynamics": "mock"})


def _parts(m):
    from vegeta.labkit import aero as la
    g = GOLD["link:wing/data:aero"]
    m.link("wing").data["aero"] = la.block(name=g["name"], method=g["method"], reference=g["reference"], coefficients=g["coefficients"],
                                           derivatives=g["derivatives"], tables=g["tables"], frame=g["frame"], axes=g["axes"],
                                           reference_point_m=g["reference_point_m"], center_of_pressure_m=g["center_of_pressure_m"],
                                           assumptions=g["assumptions"], source=g["source"], validity=g["validity"])
    return {"aerodynamics": json.loads(json.dumps(GOLD["data:aerodynamics"])), "tags": {"data:aerodynamics*": "cfd", "link:wing/data:aero/tables*": "polar"}}


@pytest.fixture(scope="module")
def registered():
    from vegeta.labkit import machines
    machines.register("mockplane", _mock_robot, parts=_parts)
    return machines


def test_wing_aero_leaves_as_stated(registered, tmp_path):
    m, _ = export.machine("mockplane")
    assert schema.problems(m) == []
    assert m.link("wing").data["aero"] == GOLD["link:wing/data:aero"]            # the helper assembles exactly the contract's block
    assert m.data["aerodynamics"] == GOLD["data:aerodynamics"]
    export.write(m, tmp_path / "p.yx2", also=("usda", "xml"))
    for f in ("p.yx2", "p.usda"):
        back, _ = yax.read(tmp_path / f)
        a = back.link("wing").data["aero"]
        assert a["reference"]["area_m2"] == 0.32 and a["coefficients"]["CD0"] == 0.012 and a["reference_point_m"] == [0.05, 0.0, 0.0]
        assert a["tables"][0]["values"]["CL"] == GOLD["link:wing/data:aero"]["tables"][0]["values"]["CL"]
        t = tg.table(back)
        assert t["link:wing/data:aero/tables"] == ["00aero", "polar"] and t["data:aerodynamics/coefficients/CL"] == ["00aero", "cfd"]
    assert "0.012" not in (tmp_path / "p.xml").read_text()
    m, _ = export.machine("mockplane", drop=("00aero",))
    assert "aero" not in m.link("wing").data and "aerodynamics" not in m.data and "polar" in m.tags["withheld"]
    m, _ = export.machine("mockplane", keep=("00core", "00mass", "polar"))
    assert set(m.link("wing").data["aero"]) == {"tables"} and "aerodynamics" not in m.data


def test_design_file_helper_on_a_tiny_document():
    from vegeta.labkit import aero as la
    doc = {"layout": {"S_ref": 0.6, "mac": 253.3333, "b2": 1200.0}, "derivatives": {"CL0": 0.3932, "CLa": 5.3982, "CLq": 3.2509, "CD0": 0.0253}}
    b = la.from_design_json(doc, name="mock plane", source="mock_design.json")
    assert b["reference"] == {"area_m2": 0.6, "chord_m": pytest.approx(0.2533333), "span_m": 2.4, "source": "mock_design.json: layout"}
    assert b["coefficients"] == {"CL0": 0.3932, "CD0": 0.0253} and b["derivatives"] == {"CLa": 5.3982, "CLq": 3.2509}
    assert b["method"] == "analytic" and b["axes"]["z"] == "down" and b["frame"] == "machine"
    with pytest.raises(ValueError):
        la.block(name="x", method="analytic", reference={}, derivatives={"CLa": "big"})
