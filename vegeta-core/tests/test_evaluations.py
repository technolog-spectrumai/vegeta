"""Evaluations through the real tools (skipped when a tool is missing)."""
import json
import math

import pytest

from vegeta import aeromant, core, mellonia, talos
from vegeta.mellonia.examples import GENERIC_PLA_0_2MM


def beam_model(rev):
    L = rev.params["length"]
    return talos.StructuralModel(
        rev.step, "mm-N-MPa", talos.Material("steel", 210000, 0.3, yield_strength=235),
        regions=[talos.SurfacesOnPlane("fixed", "x", 0.0), talos.SurfacesOnPlane("tip", "x", L)],
        supports=[talos.FixedSupport("fixed")], loads=[talos.Force("tip", fz=-100.0)],
        mesh_settings=talos.MeshSettings(5.0))


@pytest.fixture
def beam_rev(ws):
    r = ws.add_design("beam", "vegeta.dedalus.examples:CantileverBeam").new_revision(length=200.0)
    assert r.generate().ok
    return r


@pytest.mark.requires_gmsh
@pytest.mark.requires_ccx
def test_fea_evaluation_recorded_and_validated(ws, beam_rev):
    ev = beam_rev.run_fea("static", beam_model)
    assert ev.ok and ev.recorded, ev.messages
    delta = 100 * 200 ** 3 / (3 * 210000 * 20 * 10 ** 3 / 12)
    assert -ev.metrics["displacement_min"][2] == pytest.approx(delta, rel=0.03)
    rec = json.loads((ev.directory / "evaluation.json").read_text())
    assert rec["revision"] == "r1" and rec["config"]["units"] == "mm-N-MPa"
    assert "def beam_model" in rec["factory"]["source"] and rec["tool_versions"]["gmsh"]
    assert any(c["command"][0] == "ccx" for c in rec["commands"])
    assert (ev.directory / "model.frd").is_file()  # native results kept
    again = beam_rev.run_fea("static", beam_model)
    assert not again.ok and "already exists" in again.messages[0]
    assert "SF=" in ws.status().rows[0]["fea:static"]
    r2 = beam_rev.branch(height=12.0)
    assert ws.status().rows[1]["fea:static"] == core.NOT_RUN


@pytest.mark.requires_gmsh
def test_fea_model_must_use_revision_geometry(ws, beam_rev, tmp_path):
    other = beam_rev.branch(length=150.0)
    other.generate()
    with pytest.raises(ValueError, match="this revision's STEP"):
        beam_rev.run_fea("wrong", lambda rev: beam_model(other))


@pytest.mark.requires_prusaslicer
def test_print_evaluation(ws, beam_rev):
    ev = beam_rev.run_print("flat", GENERIC_PLA_0_2MM, mellonia.Orientation())
    assert ev.ok and ev.metrics["layer_count"] == 50
    assert ev.record["config"]["orientation"] == {"rotate_x": 0.0, "rotate_y": 0.0, "rotate_z": 0.0}
    assert "print:flat" in ws.status().columns


@pytest.mark.requires_openfoam
def test_cfd_evaluation_mesh_steps_only(ws, beam_rev):
    def case(rev, workdir):
        return aeromant.CFDCase(
            "laminar_external_simplefoam", rev.stl,
            dict(velocity=1.0, kinematic_viscosity=1e-3, density=1.0, reference_area=2e-4,
                 reference_length=0.2, center_of_rotation=(0, 0, 0)),
            workdir=workdir, geometry_units="mm", environment=aeromant.OpenFOAMEnvironment.detect())

    ev = beam_rev.run_cfd("mesh_only", case, steps=["blockMesh", "checkMesh"])
    assert ev.ok, ev.messages
    assert ev.metrics["mesh_cells"] > 0 and ev.record["steps"] == ["blockMesh", "checkMesh"]
    assert "cells" in ws.status().rows[0]["cfd:mesh_only"]
