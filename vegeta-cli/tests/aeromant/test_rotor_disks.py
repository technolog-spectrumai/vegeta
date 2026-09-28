"""aircraft_rotor_disks: the case is written for both OpenFOAM forks (prepare only; no OpenFOAM run)."""
import numpy as np
import pytest

from vegeta import aeromant
from vegeta.aeromant import CFDCase, get_template

BLADE = [[0.02, 30.0, 0.018], [0.06, 18.0, 0.022], [0.114, 10.0, 0.008]]
POLAR = [[a, 0.02 + 0.04 * min(1.0, abs(a) / 15) ** 2, float(np.clip(0.1 * (a + 2), -1.1, 1.1))]
         for a in range(-180, 181, 10)]


def values(**over):
    v = dict(velocity=14.0, kinematic_viscosity=1.5e-5, density=1.2, reference_area=0.17, reference_length=0.25,
             center_of_rotation=(0.05, 0, 0), disk1_center=(-0.06, -0.3, 0.0), disk2_center=(-0.06, 0.3, 0.0),
             disk_axis=(-1, 0, 0), diameter=0.2286, rpm=7000.0, blades=2, blade=BLADE, polar=POLAR)
    v.update(over)
    return v


def test_template_is_registered_and_describes_itself():
    t = get_template("aircraft_rotor_disks")
    assert "rotor disks" in t.describe() and t.flavor_names == ["openfoam.com", "openfoam.org"]


@pytest.mark.parametrize("version,files", [
    (None, ("system/fvOptions", "constant/triSurface/body.stl")),
    ("14", ("constant/fvModels", "constant/geometry/body.stl")),
])
def test_prepare_writes_both_disks(tmp_path, sphere_stl, version, files):
    env = (aeromant.OpenFOAMEnvironment(env={"PATH": str(tmp_path)}) if version is None else
           aeromant.OpenFOAMEnvironment(bashrc=_bashrc(tmp_path, version)))
    case = CFDCase("aircraft_rotor_disks", sphere_stl, values(), tmp_path / "c", geometry_units="m", environment=env)
    res = case.prepare()
    assert res.ok, res.messages
    c = tmp_path / "c"
    for f in files:
        assert (c / f).is_file(), f
    for f in c.rglob("*"):
        if f.is_file() and f.suffix != ".stl" and f.name != "aeromant_case.json":
            assert "{{" not in f.read_text().replace("{{...}}", ""), f
    src = (c / files[0]).read_text()
    assert src.count("type            rotorDisk;") == 2
    assert "cellZone        diskLeft;" in src and "cellZone        diskRight;" in src
    assert "(section (0.02 30 0.018))" in src and "(-180 " in src
    assert ("rpm             7000;" in src and "rpm             -7000;" in src) or ("7000 [rpm]" in src and "-7000 [rpm]" in src)
    assert "axis            (-1 0 0);" in src and "refDirection    (0 0 1);" in src
    snappy = (c / "system/snappyHexMeshDict").read_text()
    assert snappy.count("cellZone diskLeft;") == 1 and "point1 (-0.050856 -0.3 0)" in snappy


def test_bad_disk_inputs_are_refused(tmp_path, sphere_stl):
    for over, msg in ((dict(blade=[[0.02, 30, 0.02], [0.2, 10, 0.01]]), "diameter / 2"),
                      (dict(polar=[[-10, 0.02, -0.8], [10, 0.02, 1.0]]), "-90..90"),
                      (dict(disk1_center=(50.0, 0, 0)), "inside the domain"),
                      (dict(rotation1=0.5), "rotation1")):
        case = CFDCase("aircraft_rotor_disks", sphere_stl, values(**over), tmp_path / msg[:4], geometry_units="m")
        res = case.prepare()
        assert not res.ok and any(msg in m for m in res.messages), (over, res.messages)
    with pytest.raises(ValueError, match="rows of 3"):
        CFDCase("aircraft_rotor_disks", sphere_stl, values(blade=[[1, 2]]), tmp_path / "x", geometry_units="m")


def _bashrc(tmp_path, version):
    rc = tmp_path / "bashrc"
    rc.write_text(f"export WM_PROJECT_VERSION={version}\n")
    return str(rc)
