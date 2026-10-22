"""Temperature loads: a radial temperature field, thermal expansion and stiffness/yield at temperature in the CalculiX
deck, and the model's checks (no CalculiX needed)."""
import numpy as np
import pytest

from vegeta.talos import (Centrifugal, FixedSupport, Material, MeshSettings, RadialTemperature, StructuralModel,
                          SurfacesOnPlane)
from vegeta.talos.ccx import write_inp
from vegeta.talos.mesh import MeshData

INCONEL = Material("IN713C", 200000, 0.3, density=7.91e-9, yield_strength=740, units="mm-N-MPa",
                   thermal_expansion=1.4e-5, reference_temperature=293.15,
                   temperature_table=((293.15, 200000, 0.30, 740), (873.15, 170000, 0.31, 700), (1073.15, 158000, 0.32, 620)))


def tet():
    return MeshData(node_ids=np.array([1, 2, 3, 4]), coords=np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10.0]]),
                    element_type="C3D4", element_ids=np.array([1]), connectivity=np.array([[1, 2, 3, 4]]),
                    region_nodes={"fix": np.array([1, 2, 3])}, region_triangles={"fix": np.array([[1, 2, 3]])})


def test_radial_temperature_profile():
    t = RadialTemperature(radii=(5.0, 30.0), temperatures=(500.0, 1000.0), axis=(1, 0, 0))
    got = t.at([[0, 0, 0], [7, 17.5, 0], [0, 0, 30], [0, 40, 0]])
    assert got == pytest.approx([500.0, 750.0, 1000.0, 1000.0])
    with pytest.raises(ValueError):
        RadialTemperature(radii=(5.0, 3.0), temperatures=(1.0, 2.0))
    with pytest.raises(ValueError):
        RadialTemperature(radii=(5.0,), temperatures=(-1.0,))


def test_material_at_temperature():
    assert INCONEL.yield_at(973.15) == pytest.approx(660.0)
    assert Material("a", 1, 0.3, yield_strength=5).yield_at(1000.0) == pytest.approx(5)
    with pytest.raises(ValueError, match="temperature_table"):
        Material("b", 1, 0.3, temperature_table=((300, 1, 0.3, None), (200, 1, 0.3, None)))
    with pytest.raises(ValueError, match="thermal_expansion"):
        Material("c", 1, 0.3, thermal_expansion=0.0)


def test_deck_has_the_thermal_cards(tmp_path):
    t = RadialTemperature(radii=(0.0, 10.0), temperatures=(400.0, 900.0), axis=(0, 0, 1))
    book = write_inp(tmp_path / "m.inp", tet(), INCONEL, [FixedSupport("fix")], [Centrifugal(100000.0), t])
    deck = (tmp_path / "m.inp").read_text()
    assert "*EXPANSION, ZERO=293.15\n1.4e-05" in deck
    assert "158000, 0.32, 1073.15" in deck
    i_ic, i_step, i_temp = deck.index("*INITIAL CONDITIONS, TYPE=TEMPERATURE"), deck.index("*STEP"), deck.index("*TEMPERATURE\n")
    assert i_ic < i_step < i_temp and "NALL, 293.15" in deck
    assert "2, 900" in deck and "4, 400" in deck
    assert book["temperature_range"] == pytest.approx([400.0, 900.0])
    plain = write_inp(tmp_path / "p.inp", tet(), INCONEL, [FixedSupport("fix")], [Centrifugal(100000.0)])
    assert "*EXPANSION" not in (tmp_path / "p.inp").read_text() and plain["temperature_range"] is None


def test_model_checks_for_temperature_loads():
    kw = dict(geometry="x.step", units="mm-N-MPa", regions=[SurfacesOnPlane("fix", "x", 0)],
              supports=[FixedSupport("fix")], mesh_settings=MeshSettings(1.0))
    t = RadialTemperature(radii=(0.0,), temperatures=(800.0,))
    StructuralModel(material=INCONEL, loads=[t], **kw)
    with pytest.raises(ValueError, match="thermal_expansion"):
        StructuralModel(material=Material("s", 210000, 0.3, units="mm-N-MPa"), loads=[t], **kw)
    with pytest.raises(ValueError, match="one temperature field"):
        StructuralModel(material=INCONEL, loads=[t, t], **kw)
