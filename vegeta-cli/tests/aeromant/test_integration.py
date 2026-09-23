"""OpenFOAM integration: a small known case validated against a drag correlation."""
import json
import math

import pytest

from vegeta import aeromant
from vegeta.aeromant import CFDCase

pytestmark = [pytest.mark.requires_openfoam, pytest.mark.slow]


def schiller_naumann(re):
    """Sphere drag correlation, accurate to a few percent for Re < 800."""
    return 24.0 / re * (1 + 0.15 * re ** 0.687)


def test_laminar_sphere_re100(tmp_path, openfoam):
    from _sphere import icosphere

    stl = aeromant.write_stl_ascii(icosphere(0.5, 4), tmp_path / "sphere.stl")
    params = dict(velocity=1.0, kinematic_viscosity=0.01, density=1.0, reference_area=math.pi / 4,
                  reference_length=1.0, center_of_rotation=(0, 0, 0))
    case = CFDCase("laminar_external", stl, params, tmp_path / "case", geometry_units="m",
                   environment=openfoam)
    assert case.prepare().ok
    res = case.run()
    assert res.ok, res.messages
    m = res.metrics
    assert m["mesh_ok"] and m["mesh_cells"] > 10000
    assert m["converged"], "SIMPLE residual target not reached"
    # 49k-cell castellated/snapped mesh without layers: observed +0.7 %; allow 10 %
    assert m["Cd"] == pytest.approx(schiller_naumann(100.0), rel=0.10)
    assert abs(m["Cl"]) < 0.02  # symmetric body
    assert m["Cd_std_last50"] < 1e-3
    # native case and logs are preserved
    for f in ("log.blockMesh", "log.snappyHexMesh", "log.checkMesh", "log.solver", "constant/polyMesh/boundary",
              "system/controlDict", "inputs/sphere.stl"):
        assert (tmp_path / "case" / f).exists(), f
    assert json.loads((tmp_path / "case/summary.json").read_text())["status"] == "success"
    # results can be re-read later without running anything
    again = case.results()
    assert again.metrics["Cd"] == pytest.approx(m["Cd"])


def test_run_selected_steps_only(tmp_path, openfoam, sphere_stl):
    params = dict(velocity=1.0, kinematic_viscosity=0.01, density=1.0, reference_area=math.pi / 4,
                  reference_length=1.0, center_of_rotation=(0, 0, 0), surface_level=2, near_level=2, wake_level=1)
    case = CFDCase("laminar_external", sphere_stl, params, tmp_path / "c", geometry_units="m",
                   environment=openfoam)
    case.prepare()
    res = case.run(steps=["blockMesh", "checkMesh"])
    assert res.ok and res.metrics["mesh_cells"] > 0
    assert not (tmp_path / "c/log.solver").exists()  # nothing beyond the chosen steps ran
    assert "Cd" not in res.metrics
    assert not case.results().ok  # NOT RUN
