"""Stage 2B — materials, strength and connections through the real Vegeta → Yax export on a mock assembly: a frame and
a panel (a two-body Chiron robot), the materials, strength and connections a mock parts function states with known
numbers (the same fixture Yax's own tests use), Pekari-style axles from the labkit helper. Compared against the fixture,
not against what went in. No FEA, no CAD."""
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

GOLD = json.loads((Path(__file__).parent / "fixtures" / "export2" / "mock_assembly_materials.json").read_text())


def _mock_robot():
    from vegeta.chiron import Geom, Joint, Link, Robot
    panel = Link("panel", pos=(0.1, 0.0, 0.03), joints=[Joint("panel_hinge", axis=(0, 1, 0), range=(-0.2, 0.2))],
                 geoms=[Geom("sheet", "box", (0.05, 0.05, 0.001), mass=0.12)])
    root = Link("frame", geoms=[Geom("bar", "box", (0.15, 0.02, 0.02), mass=0.8)], children=[panel])
    return Robot("mockframe", root=root, notes="a mock", sources={"materials": "mock handbook"})


def _parts(m):
    from vegeta.labkit import materials as lmat
    for lk in m.links:
        lk.data["material"] = GOLD[f"link:{lk.name}/data:material"]
    m.link("frame").data["strength"] = json.loads(json.dumps(GOLD["link:frame/data:strength"]))
    conns = json.loads(json.dumps(GOLD["data:connections"]))
    conns.update(lmat.axles(m, diameter_m=0.01, length_m=0.08, source="mock DESIGN['axle_diameter']"))
    return {"materials": json.loads(json.dumps(GOLD["data:materials"])), "connections": conns, "tags": {"data:materials*": "handbook"}}


@pytest.fixture(scope="module")
def registered():
    from vegeta.labkit import machines
    machines.register("mockframe", _mock_robot, parts=_parts)
    return machines


def test_materials_strength_connections_leave_as_stated(registered, tmp_path):
    m, _ = export.machine("mockframe")
    assert schema.problems(m) == []
    assert m.data["materials"] == GOLD["data:materials"] and m.link("frame").data["strength"] == GOLD["link:frame/data:strength"]
    assert m.link("panel").data["material"] == "cfrp-t300"
    axle = m.data["connections"]["axle:panel_hinge"]
    assert axle["type"] == "pin" and axle["links"] == ["frame", "panel"] and axle["geometry"] == {"diameter_m": 0.01, "length_m": 0.08}
    assert axle["pos"] == [0.1, 0.0, 0.03] and axle["axis"] == [0.0, 1.0, 0.0] and "limits" not in axle    # none stated, none written
    export.write(m, tmp_path / "f.yx2", also=("usda", "sdf"))
    for f in ("f.yx2", "f.usda"):
        back, _ = yax.read(tmp_path / f)
        assert back.data["materials"] == GOLD["data:materials"] and back.data["connections"]["panel-bond"] == GOLD["data:connections"]["panel-bond"]
        assert back.link("frame").data["strength"]["analysis"]["max_stress_pa"] == 95e6 and schema.problems(back) == []
        t = tg.table(back)
        assert t["data:materials/cfrp-t300/orthotropic/E1_pa"] == ["00mat", "handbook"] and t["data:connections/axle:panel_hinge"] == ["00conn"]
        assert t["link:frame/data:strength/analysis"] == ["00str"]
    assert "preload" not in (tmp_path / "f.sdf").read_text()


def test_filter_each_block_on_its_own(registered, tmp_path):
    for drop, gone in ((("00mat",), ("density_kg_m3", "tsai_wu")), (("00str",), ("allowable_stress_pa",)), (("00conn",), ("preload_n", "axle:"))):
        m, _ = export.machine("mockframe", drop=drop)
        export.write(m, tmp_path / "cut.yx2")
        text = (tmp_path / "cut.yx2").read_text()
        for s in gone:
            assert s not in text, (drop, s)
        assert list(drop)[0] in m.tags["withheld"]
    m, _ = export.machine("mockframe", keep=("00core", "00geom", "00mass", "00inert", "handbook"))
    assert set(m.data) == {"materials"} and "strength" not in m.link("frame").data and "material" not in m.link("frame").data


def test_talos_material_and_density_helpers():
    from vegeta.labkit import materials as lmat
    from vegeta import talos
    al = talos.Material("al6061", youngs_modulus=68900.0, poissons_ratio=0.33, density=2.7e-9, yield_strength=276.0, source="mock, mm-N-t")
    e = lmat.from_talos(al, stress_pa=1e6, density_kg_m3=1e12, cls="metal")
    assert e == {"name": "al6061", "anisotropy": "isotropic", "youngs_modulus_pa": 68.9e9, "poisson_ratio": 0.33, "class": "metal",
                 "density_kg_m3": pytest.approx(2700.0), "yield_strength_pa": 276e6, "source": "mock, mm-N-t"}
    assert "shear_modulus_pa" not in e                                         # not derived
    d = lmat.from_densities({"PETG-CF": 1.25e-3}, kg_m3_per_unit=1e6, source="nb08")
    assert d == {"PETG-CF": {"name": "PETG-CF", "density_kg_m3": 1250.0, "source": "nb08"}}
