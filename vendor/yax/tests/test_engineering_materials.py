"""Stage 2A — materials, strength, connections (docs/engineering.md is the oracle): a mock aluminium frame and a
composite panel, bolted and bonded, with known numbers; through the real tagging and serialization and back, compared
against the fixture written by hand; filtered independently by 00mat, 00str and 00conn; refused with reasons when
wrong; losses reported by the formats that cannot carry them. No FEA runs."""
import json
from pathlib import Path

import numpy as np
import pytest

import yax
from yax import hub, schema
from yax import tags as tg

GOLD = Path(__file__).parent / "fixtures" / "export2" / "mock_assembly_materials.json"


def _assembly():
    g = json.loads(GOLD.read_text())
    frame = hub.Link("frame", mass=0.8, inertia=np.diag([0.01, 0.02, 0.02]), shapes=[hub.Shape("bar", "box", (0.3, 0.04, 0.04))],
                     data={"material": g["link:frame/data:material"], "strength": g["link:frame/data:strength"]})
    panel = hub.Link("panel", pos=(0.1, 0.0, 0.03), mass=0.12, inertia=np.diag([0.001, 0.001, 0.002]), shapes=[hub.Shape("sheet", "box", (0.1, 0.1, 0.002))],
                     data={"material": g["link:panel/data:material"]})
    j = hub.Joint("panel_mount", "fixed", "frame", "panel", pos=(0.1, 0.0, 0.03))
    m = hub.Machine("mockframe", links=[frame, panel], joints=[j], producer={"tool": "test"},
                    data={"materials": g["data:materials"], "connections": g["data:connections"]})
    return m, g


def test_materials_strength_connections_match_the_oracle(tmp_path):
    m, g = _assembly()
    assert schema.problems(m) == []
    for ext in ("usda", "yx2"):
        f = tmp_path / f"frame.{ext}"
        yax.save(m, f)
        back, _ = yax.read(f)
        assert back.data["materials"] == g["data:materials"] and back.data["connections"] == g["data:connections"]
        assert back.link("frame").data == {"material": "al6061-t6", "strength": g["link:frame/data:strength"]}
        assert back.link("panel").data == {"material": "cfrp-t300"}
        al, cf = back.data["materials"]["al6061-t6"], back.data["materials"]["cfrp-t300"]
        assert al["youngs_modulus_pa"] == 68.9e9 and al["poisson_ratio"] == 0.33 and al["fracture_toughness_pa_sqrt_m"] == 29e6
        assert cf["orthotropic"]["E1_pa"] == 135e9 and cf["composite"]["layup"][1] == {"angle_deg": 45} and cf["failure"]["calibrated"] is True
        bolts = back.data["connections"]["frame-panel-bolts"]
        assert bolts["fasteners"]["preload_n"] == 1200.0 and bolts["geometry"]["pattern"][3] == [0.11, 0.02, 0.02] and bolts["frame"] == "link:frame"
        assert back.data["connections"]["panel-bond"]["limits"] == {"shear_n": 1600.0, "tension_n": 900.0, "basis": "assumed"}
        assert back.link("frame").data["strength"]["analysis"]["kind"] == "static"       # a static result stays a static result
        assert schema.problems(back) == []
    t = tg.table(m)
    assert t["data:materials/al6061-t6/youngs_modulus_pa"] == ["00mat"] and t["link:frame/data:material"] == ["00mat"]
    assert t["link:frame/data:strength/analysis"] == ["00str"] and t["data:connections/panel-bond/limits"] == ["00conn"]
    text = (tmp_path / "frame.yx2").read_text().replace('\\"', '"')
    for s in ('"youngs_modulus_pa":68900000000.0', '"preload_n":1200.0', '"angle_deg":45', '"criterion":"tsai_wu"'):
        assert s in text, s


def test_independent_filtering_of_mat_str_conn(tmp_path):
    m, g = _assembly()
    for drop, gone, stays in ((("00mat",), ("density_kg_m3", "68900000000", "orthotropic", "tsai_wu"), ("preload_n", "allowable_stress_pa")),
                              (("00str",), ("allowable_stress_pa", "4 g landing"), ("preload_n", "cfrp-t300")),
                              (("00conn",), ("preload_n", "panel-bond", "tsai_wu_never"), ("allowable_stress_pa", "cfrp-t300"))):
        cut = tg.select(m, drop=drop)
        assert cut.tags["withheld"] == list(drop) and schema.problems(cut) == [] or drop == ("00mat",)
        for ext in ("yx2", "sdf"):
            yax.save(cut, tmp_path / f"cut.{ext}")
            text = (tmp_path / f"cut.{ext}").read_text()
            for s in gone:
                assert s not in text, (drop, s)
            for s in stays:
                assert (s in text) == (ext == "yx2"), (drop, s, ext)
    cut = tg.select(m, drop=("00mat",))
    assert "material" not in cut.link("frame").data and "materials" not in cut.data                   # the link's material is 00mat too
    assert cut.link("frame").data["strength"]["material"] == "al6061-t6"                              # the strength's reference stays: it names, it does not hold
    assert schema.problems(cut) == ["link:frame/data:strength: material 'al6061-t6' is not in data['materials']",     # dependents of excluded data, said
                                    "data:connections/panel-bond: material 'al6061-t6' is not in data['materials']"]


def test_invalid_blocks_are_refused_with_reasons():
    m, g = _assembly()
    m.data["materials"]["al6061-t6"]["poisson_ratio"] = 0.7
    m.data["materials"]["cfrp-t300"]["anisotropy"] = "weird"
    m.data["materials"]["cfrp-t300"]["composite"]["layup"].append({"thickness_m": 1e-4})
    m.data["materials"]["cfrp-t300"]["failure"] = {"parameters": {}}
    m.data["connections"]["panel-bond"]["type"] = "glue"
    m.data["connections"]["frame-panel-bolts"]["links"] = ["frame", "roof"]
    m.data["connections"]["frame-panel-bolts"]["frame"] = "link:roof"
    m.link("panel").data["material"] = "unobtainium"
    m.link("frame").data["strength"]["safety_factor"] = 0
    p = schema.problems(m)
    for expect in ("poisson_ratio must be in (-1, 0.5)", "anisotropy must be", "composite/layup/4: a ply is", "failure: criterion is required",
                   "type must be one of", "link 'roof' is not a link", "frame must be 'machine' or 'link:<name>'",
                   "material 'unobtainium' is not in data['materials']", "safety_factor must be > 0"):
        assert any(expect in x for x in p), (expect, p)
    m, _ = _assembly()
    del m.data["materials"]["al6061-t6"]["shear_modulus_pa"]                   # optional: absent, not derived
    assert schema.problems(m) == [] and "shear_modulus_pa" not in m.data["materials"]["al6061-t6"]


def test_losses_reported_by_every_other_format(tmp_path):
    m, _ = _assembly()
    for ext in ("sdf", "urdf", "xml"):
        rep = yax.save(m, tmp_path / f"f.{ext}")
        lines = "\n".join(rep.dropped)
        assert "data:materials: the materials catalogue (2 entries)" in lines and "data:connections: the connections (2 entries)" in lines
        assert "link frame: its material" in lines and "link frame: its strength limits" in lines and "link panel: its material" in lines
        assert "machine data 'materials'" not in lines                        # said once, specifically
        text = (tmp_path / f"f.{ext}").read_text()
        assert "al6061" not in text and "preload" not in text
    assert yax.save(m, tmp_path / "f.usda").dropped == [] and yax.save(m, tmp_path / "f.yx2").dropped == []
