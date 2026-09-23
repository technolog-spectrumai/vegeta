"""Gmsh + CalculiX integration tests validated against closed-form solutions."""
import json

import numpy as np
import pytest

from talos import (Acceleration, Displacement, FixedSupport, Force, Material, MeshSettings, Pressure,
                   StructuralModel, SurfacesOnPlane, Surfaces, inspect_step, read_frd)

E, NU = 210000.0, 0.3
STEEL = Material("steel", E, NU, density=7.85e-9, yield_strength=235.0)
pytestmark = pytest.mark.requires_gmsh


def test_inspect_step_lists_box_faces(beam_step):
    info = inspect_step(beam_step, "mm-N-MPa")
    assert len(info.surfaces) == 6 and len(info.volumes) == 1
    assert info.volumes[0]["volume"] == pytest.approx(200 * 20 * 10)
    end = [s for s in info.surfaces if s.centroid[0] == pytest.approx(200)]
    assert len(end) == 1 and end[0].area == pytest.approx(200) and end[0].normal[0] == pytest.approx(1)


def test_inspect_step_in_metres(beam_step):
    info = inspect_step(beam_step, "m-N-Pa")
    assert info.volumes[0]["volume"] == pytest.approx(200 * 20 * 10 * 1e-9)


def test_mesh_writes_msh_and_region_groups(beam_step, tmp_path):
    m = _cantilever(beam_step, 10.0)
    res = m.mesh(tmp_path)
    assert res.ok, res.messages
    assert (tmp_path / "mesh.msh").stat().st_size > 0
    assert res.metrics["element_type"] == "C3D10" and res.metrics["min_quality_sicn"] > 0
    assert res.metadata["region_surfaces"]["fixed"] and res.metadata["region_surfaces"]["tip"]


def test_region_selecting_nothing_fails_meshing(beam_step, tmp_path):
    m = StructuralModel(beam_step, "mm-N-MPa", STEEL, regions=[SurfacesOnPlane("fix", "x", 0),
                        SurfacesOnPlane("ghost", "x", 123.0)], supports=[FixedSupport("fix")],
                        loads=[Force("ghost", fz=-1)], mesh_settings=MeshSettings(20.0))
    res = m.mesh(tmp_path)
    assert res.status == "failed" and "ghost" in res.messages[0]


def test_unknown_surface_tag_fails_meshing(beam_step, tmp_path):
    m = StructuralModel(beam_step, "mm-N-MPa", STEEL, regions=[Surfaces("fix", [1]), Surfaces("load", [99])],
                        supports=[FixedSupport("fix")], loads=[Force("load", fz=-1)], mesh_settings=MeshSettings(20.0))
    res = m.mesh(tmp_path)
    assert not res.ok and "99" in res.messages[0]


def _cantilever(step, size=5.0, **mat):
    return StructuralModel(
        step, "mm-N-MPa", Material("steel", E, NU, **mat) if mat else STEEL,
        regions=[SurfacesOnPlane("fixed", "x", 0.0), SurfacesOnPlane("tip", "x", 200.0)],
        supports=[FixedSupport("fixed")], loads=[Force("tip", fz=-100.0)], mesh_settings=MeshSettings(size),
    )


@pytest.mark.requires_ccx
def test_cantilever_tip_deflection_matches_beam_theory(beam_step, tmp_path):
    m = _cantilever(beam_step)
    assert m.mesh(tmp_path).ok
    res = m.solve(tmp_path)
    assert res.ok, res.messages
    L, b, h, F = 200.0, 20.0, 10.0, 100.0
    I = b * h ** 3 / 12
    delta = F * L ** 3 / (3 * E * I)                       # Euler-Bernoulli: 0.7619 mm
    assert res.metrics["displacement_min"][2] == pytest.approx(-delta, rel=0.03)
    sigma_root = F * L * (h / 2) / I                        # 60 MPa bending stress at the root
    assert res.metrics["max_von_mises"] == pytest.approx(sigma_root, rel=0.10)
    # equilibrium: reactions balance the applied load
    assert res.metrics["reaction_total"] == pytest.approx([0, 0, F], abs=1e-3 * F)
    assert res.metrics["safety_factor_yield"] == pytest.approx(235.0 / res.metrics["max_von_mises"])
    for key in ("mesh", "inp", "frd", "dat", "ccx_log", "summary"):
        assert res.artifacts[key].is_file()
    assert json.loads(res.artifacts["summary"].read_text())["status"] == "success"
    assert res.execution[0].command[:2] == ["ccx", "-i"] and res.execution[0].returncode == 0


def _axial(step, load, order=2):
    return StructuralModel(
        step, "mm-N-MPa", STEEL,
        regions=[SurfacesOnPlane("x0", "x", 0), SurfacesOnPlane("y0", "y", 0), SurfacesOnPlane("z0", "z", 0),
                 SurfacesOnPlane("end", "x", 100)],
        # symmetry-type supports: free lateral contraction -> exact uniaxial stress state
        supports=[Displacement("x0", ux=0.0), Displacement("y0", uy=0.0), Displacement("z0", uz=0.0)],
        loads=[load], mesh_settings=MeshSettings(4.0, order=order),
    )


@pytest.mark.requires_ccx
@pytest.mark.parametrize("load, order", [
    (Force("end", fx=1000.0), 2),
    (Pressure("end", -10.0), 2),   # negative pressure pulls: 10 MPa x 100 mm^2 = 1000 N
    (Force("end", fx=1000.0), 1),  # linear tetrahedra represent constant strain exactly
])
def test_axial_bar_uniform_stress(bar_step, tmp_path, load, order):
    m = _axial(bar_step, load, order)
    assert m.mesh(tmp_path).ok
    res = m.solve(tmp_path)
    assert res.ok, res.messages
    sigma, delta = 1000.0 / 100.0, 1000.0 * 100.0 / (E * 100.0)
    assert res.metrics["displacement_max"][0] == pytest.approx(delta, rel=0.005)
    fr = read_frd(res.artifacts["frd"])
    assert np.median(fr.von_mises) == pytest.approx(sigma, rel=0.005)
    assert res.metrics["max_von_mises"] == pytest.approx(sigma, rel=0.03)
    assert res.metrics["reactions"]["x0"][0] == pytest.approx(-1000.0, rel=0.005)
    assert res.metrics["safety_factor_yield"] == pytest.approx(23.5, rel=0.03)


@pytest.mark.requires_ccx
def test_gravity_reaction_equals_weight(bar_step, tmp_path):
    m = StructuralModel(bar_step, "mm-N-MPa", STEEL, regions=[SurfacesOnPlane("x0", "x", 0)],
                        supports=[FixedSupport("x0")], loads=[Acceleration(az=-9810.0)],
                        mesh_settings=MeshSettings(5.0))
    assert m.mesh(tmp_path).ok
    res = m.solve(tmp_path)
    assert res.ok, res.messages
    weight = 7.85e-9 * (100 * 10 * 10) * 9810.0  # t/mm^3 * mm^3 * mm/s^2 = N
    # CalculiX RF excludes the body-force share lumped on the supported nodes themselves
    # (~0.5 % here, shrinking with mesh refinement), so the reaction is slightly below the weight.
    assert weight * 0.985 < res.metrics["reaction_total"][2] <= weight * 1.0001
    assert any("CalculiX RF" in s for s in res.messages)


@pytest.mark.requires_ccx
def test_reaction_excludes_load_on_supported_nodes(bar_step, tmp_path):
    m = StructuralModel(bar_step, "mm-N-MPa", STEEL, regions=[SurfacesOnPlane("x0", "x", 0),
                        SurfacesOnPlane("end", "x", 100)], supports=[FixedSupport("x0")],
                        loads=[Force("x0", fz=-50.0), Force("end", fz=-10.0)], mesh_settings=MeshSettings(10.0))
    assert m.mesh(tmp_path).ok
    res = m.solve(tmp_path)
    assert res.metrics["reaction_total"][2] == pytest.approx(10.0, rel=1e-3)
    assert any("['x0']" in s for s in res.messages)


@pytest.mark.requires_ccx
def test_no_yield_strength_means_no_safety_factor(beam_step, tmp_path):
    m = _cantilever(beam_step, 20.0, density=None)
    assert m.mesh(tmp_path).ok
    res = m.solve(tmp_path)
    assert res.ok and res.metrics["safety_factor_yield"] is None
    assert any("no yield_strength" in s for s in res.messages)


def test_changed_model_needs_new_mesh(beam_step, tmp_path):
    assert _cantilever(beam_step, 20.0).mesh(tmp_path).ok
    res = _cantilever(beam_step, 10.0).solve(tmp_path)
    assert res.status == "failed" and "out of date" in res.messages[0]


def test_missing_solver_is_failed_result(beam_step, tmp_path):
    m = _cantilever(beam_step, 20.0)
    assert m.mesh(tmp_path).ok
    res = m.solve(tmp_path, executable="ccx_not_installed_here")
    assert res.status == "failed" and "not found" in res.messages[0]
    assert res.execution[0].returncode is None and res.artifacts["inp"].is_file()


@pytest.mark.requires_ccx
def test_plots(beam_step, tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    from talos import plot_along_axis, plot_deformed, plot_von_mises_histogram

    m = _cantilever(beam_step, 20.0)
    m.mesh(tmp_path)
    res = m.solve(tmp_path)
    for fn in (plot_deformed, plot_von_mises_histogram, plot_along_axis):
        assert fn(res) is not None
