"""The sonic blowdown study (notebook 50): the case files, the mesh blocks, the choked-throat relations against the
textbook numbers, the 0-D blowdown's mass balance (no OpenFOAM, no CAD export)."""
import math

import pytest

import sonic_flow as sf


def test_choked_air_matches_the_isentropic_tables():
    G, u, ps, Ts = sf.choked_ideal(sf.AIR, 50e5, 293.15)
    assert ps / 50e5 == pytest.approx(0.5283, abs=2e-4)          # p*/p0 at gamma 1.4
    assert Ts / 293.15 == pytest.approx(2 / (sf.AIR.gamma + 1), rel=1e-9) == pytest.approx(2 / 2.4, rel=1e-4)
    assert u == pytest.approx(313.4, abs=0.5)
    # mdot = 0.0404 A p0 / sqrt(T0) for air
    assert G == pytest.approx(0.04042 * 50e5 / math.sqrt(293.15), rel=3e-3)


def test_container_volume_and_mesh_blocks_follow_the_rig():
    rig, mesh = sf.Rig(), sf.Mesh()
    assert math.pi / 4 * rig.container_bore ** 2 * rig.container_length == pytest.approx(1e6)
    blocks = mesh.blocks(rig)
    # neighbouring blocks share their cell counts across common faces
    cont_core, cont_out, up, dn, amb_core, amb_out = blocks
    assert cont_core[4] == cont_out[4] and cont_core[5] == up[5] == dn[5] == amb_core[5]
    assert amb_core[4] == amb_out[4]


def test_case_is_complete(tmp_path):
    c = sf.write_case(tmp_path / "air", sf.AIR)
    for f in ("system/blockMeshDict", "system/controlDict", "system/topoSetDict", "system/setFieldsDict",
              "constant/thermophysicalProperties", "0/p", "0/T", "0/U", "0/k", "0/omega", "0/nut", "0/alphat"):
        assert (c / f).is_file(), f
    control = (c / "system/controlDict").read_text()
    assert "rhoCentralFoam" in control and "adjustTimeStep  no;" in control and "$exitPlane;" in control
    assert "molWeight 44.01" in sf.thermophysical(sf.CO2)
    assert "(0.02 10.0 10.0)" in (c / "system/setFieldsDict").read_text()


def test_ideal_blowdown_conserves_mass():
    b = sf.blowdown(sf.AIR, n=200)
    released = sum(0.5 * (b["mdot"][i] + b["mdot"][i + 1]) * (b["t"][i + 1] - b["t"][i]) for i in range(len(b["t"]) - 1))
    assert released == pytest.approx(b["released_mass"][-1], rel=2e-3)
    assert b["p"][-1] < b["p"][0] and b["T"][-1] < b["T"][0]
