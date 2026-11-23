"""Stage 3A — aerodynamics (docs/engineering.md is the oracle): a mock wing with a known reference area, drag and lift
coefficients, a reference point and a small alpha × Re table on its link, and the whole machine's CFD coefficients;
through tagging and serialization and back against the fixture; losses reported, tables never replaced by constants;
wrong tables refused with reasons. No CFD runs."""
import json
from pathlib import Path

import numpy as np
import pytest

import yax
from yax import hub, schema
from yax import tags as tg

GOLD = json.loads((Path(__file__).parent / "fixtures" / "export2" / "mock_wing_aero.json").read_text())


def _plane():
    body = hub.Link("body", mass=1.5, inertia=np.diag([0.02, 0.05, 0.06]), shapes=[hub.Shape("pod", "capsule", (0.04, 0.3))])
    wing = hub.Link("wing", pos=(0.0, 0.0, 0.05), mass=0.4, inertia=np.diag([0.08, 0.002, 0.08]), shapes=[hub.Shape("plank", "box", (0.2, 1.6, 0.01))],
                    data={"aero": json.loads(json.dumps(GOLD["link:wing/data:aero"]))})
    m = hub.Machine("mockplane", links=[body, wing], joints=[hub.Joint("wing_mount", "fixed", "body", "wing", pos=(0, 0, 0.05))],
                    producer={"tool": "test"}, data={"aerodynamics": json.loads(json.dumps(GOLD["data:aerodynamics"]))})
    return m


def test_wing_aero_matches_the_oracle(tmp_path):
    m = _plane()
    assert schema.problems(m) == []
    for ext in ("usda", "yx2"):
        yax.save(m, tmp_path / f"p.{ext}")
        back, _ = yax.read(tmp_path / f"p.{ext}")
        a = back.link("wing").data["aero"]
        assert a == GOLD["link:wing/data:aero"] and back.data["aerodynamics"] == GOLD["data:aerodynamics"]
        assert a["reference"]["area_m2"] == 0.32 and a["coefficients"]["CD0"] == 0.012 and a["coefficients"]["CLalpha"] == 5.1
        assert a["reference_point_m"] == [0.05, 0.0, 0.0] and a["frame"] == "link:wing"
        assert a["tables"][0]["values"]["CL"][4] == [1.2, 1.25] and a["tables"][0]["inputs"]["re"] == [100000, 300000]
        assert schema.problems(back) == []
    t = tg.table(m)
    assert t["link:wing/data:aero/tables"] == ["00aero"] and t["data:aerodynamics/coefficients/CD"] == ["00aero"]
    text = (tmp_path / "p.yx2").read_text().replace('\\"', '"')
    assert '"area_m2":0.32' in text and '"CLalpha":5.1' in text and '"re":[100000,300000]' in text


def test_losses_say_no_constant_replaced_a_table(tmp_path):
    m = _plane()
    for ext in ("xml", "sdf", "urdf"):
        rep = yax.save(m, tmp_path / f"p.{ext}")
        assert any(d.startswith("link wing: its aerodynamics (8 coefficients, 1 tables)") for d in rep.dropped)
        assert any(d.startswith("data:aerodynamics: the aerodynamics (3 coefficients)") for d in rep.dropped)
        assert "0.012" not in (tmp_path / f"p.{ext}").read_text()
    m.data["aerodynamics"]["tables"] = m.link("wing").data["aero"]["tables"]
    rep = yax.save(m, tmp_path / "p.xml")
    assert any("a table is not a constant: none was substituted" in d for d in rep.dropped)
    rep = yax.save(m, tmp_path / "jsb" / "aircraft" / "p" / "p.xml", fmt="jsbsim")
    assert any("not written as JSBSim aerodynamics" in d for d in rep.dropped)


def test_filter_and_refusals():
    m = _plane()
    cut = tg.select(m, drop=("00aero",))
    assert "aero" not in cut.link("wing").data and "aerodynamics" not in cut.data and cut.tags["withheld"] == ["00aero"]
    assert cut.link("wing").mass == 0.4 and cut.link("wing").shapes                    # geometry and mass untouched
    a = m.link("wing").data["aero"]
    a["method"] = "guess"
    a["frame"] = "link:fin"
    a["coefficients"]["CL"] = [0.1, 0.2]
    a["tables"][0]["inputs"]["alpha_deg"] = [0, -4, 4, 8, 12]
    a["tables"][0]["values"]["CD"] = [[0.02, 0.016]]
    a["tables"].append({"inputs": {"temperature_c": [0, 10]}, "values": {"CL": [0.1, 0.2]}})
    p = schema.problems(m)
    for expect in ("method must be one of", "frame must be 'machine' or 'link:<name>'", "CL must be a number (a table goes in 'tables')",
                   "inputs/alpha_deg: must be increasing", "values/CD: shape must be [5, 2] (alpha_deg × re), got [1, 2]",
                   "inputs ['temperature_c'] are not among"):
        assert any(expect in x for x in p), (expect, p)
    m = _plane()
    del m.link("wing").data["aero"]["tables"]                                  # a wing with constants only is fine
    del m.link("wing").data["aero"]["center_of_pressure_m"]                   # absent, not derived from the tables
    assert schema.problems(m) == [] and "center_of_pressure_m" not in m.link("wing").data["aero"]
