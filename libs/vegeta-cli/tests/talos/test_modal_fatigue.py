"""Modal analysis against beam theory; fatigue against a hand calculation."""
import math

import numpy as np
import pytest

from vegeta import talos
from vegeta.talos.ccx import write_inp
from vegeta.talos.mesh import read_mesh

E, NU, RHO = 210000.0, 0.3, 7.85e-9
STEEL = talos.Material("S235", E, NU, density=RHO, yield_strength=235)


def beam_model(step, loads=(), masses=()):
    return talos.StructuralModel(step, "mm-N-MPa", STEEL,
                                 [talos.SurfacesOnPlane("fixed", "x", 0.0), talos.SurfacesOnPlane("tip", "x", 200.0)],
                                 [talos.FixedSupport("fixed")], list(loads), talos.MeshSettings(5.0), masses=list(masses))


@pytest.mark.requires_gmsh
@pytest.mark.requires_ccx
def test_modes_match_beam_theory(beam_step, tmp_path):
    m = beam_model(beam_step, loads=[talos.Force("tip", fz=-100.0)])
    assert m.mesh(tmp_path).ok
    res = m.solve_modes(tmp_path, 4)
    assert res.ok, res.messages
    f = res.metrics["frequencies_hz"]
    A, L = 20 * 10, 200
    f_z = 1.875**2 / (2 * math.pi) * math.sqrt(E * (20 * 10**3 / 12) / (RHO * A * L**4))
    f_y = 1.875**2 / (2 * math.pi) * math.sqrt(E * (10 * 20**3 / 12) / (RHO * A * L**4))
    assert f[0] == pytest.approx(f_z, rel=0.02) and f[1] == pytest.approx(f_y, rel=0.02)
    assert len(res.metrics["effective_modal_mass"]) == 4 and res.artifacts["frd"].is_file()
    steps = talos.read_frd_steps(res.artifacts["frd"])
    assert len(steps) == 4 and steps[0][0] == pytest.approx(f[0], rel=1e-3)
    # the first mode bends in z: tip displacement is almost purely uz
    u = steps[0][1].displacement
    tip = np.nanargmax(np.abs(u[:, 2]))
    assert abs(u[tip, 2]) > 20 * max(abs(u[tip, 0]), abs(u[tip, 1]))


@pytest.mark.requires_gmsh
@pytest.mark.requires_ccx
def test_tip_mass_lowers_frequency_like_rayleigh(beam_step, tmp_path):
    mb = RHO * 200 * 200
    m = beam_model(beam_step, masses=[talos.PointMass("tip", 0.5 * mb)])      # modal-only model, no loads
    assert m.mesh(tmp_path).ok
    res = m.solve_modes(tmp_path, 2)
    assert res.ok, res.messages
    f_ray = 1 / (2 * math.pi) * math.sqrt(3 * E * (20 * 10**3 / 12) / ((0.5 * mb + 0.2357 * mb) * 200**3))
    assert res.metrics["frequencies_hz"][0] == pytest.approx(f_ray, rel=0.03)
    assert res.metrics["point_mass_total"] == pytest.approx(0.5 * mb)
    assert not m.solve(tmp_path).ok                                           # static needs loads
    inp = (tmp_path / "modes.inp").read_text()
    assert "*FREQUENCY" in inp and "TYPE=MASS" in inp and "*CLOAD" not in inp


def test_modal_deck_and_validation(tmp_path):
    with pytest.raises(ValueError, match="positive"):
        talos.PointMass("tip", 0.0)
    with pytest.raises(ValueError, match="no loads"):
        talos.StructuralModel("x.step", "mm-N-MPa", STEEL, [talos.SurfacesOnPlane("f", "x", 0.0)],
                              [talos.FixedSupport("f")], [], talos.MeshSettings(5.0))
    with pytest.raises(ValueError, match="unknown region"):
        talos.StructuralModel("x.step", "mm-N-MPa", STEEL, [talos.SurfacesOnPlane("f", "x", 0.0)],
                              [talos.FixedSupport("f")], [], talos.MeshSettings(5.0), masses=[talos.PointMass("nope", 1.0)])


def test_fatigue_curve():
    c = talos.FatigueCurve("t", sigma_f=100.0, b=-0.1, ultimate=200.0, endurance_limit=10.0)
    assert c.cycles_to_failure(100.0) == pytest.approx(0.5)
    n = c.cycles_to_failure([50.0, 50.0, 5.0, 0.0], [0.0, 100.0, 0.0, 0.0])
    assert n[1] < n[0] and math.isinf(n[2]) and math.isinf(n[3])
    with pytest.raises(ValueError):
        talos.FatigueCurve("t", 100.0, 0.1)


@pytest.mark.requires_gmsh
@pytest.mark.requires_ccx
def test_fatigue_assessment_matches_hand_calculation(beam_step, tmp_path):
    m = beam_model(beam_step, loads=[talos.Force("tip", fz=-100.0)])
    assert m.mesh(tmp_path).ok
    res = m.solve(tmp_path)
    assert res.ok
    curve = talos.FatigueCurve("t", sigma_f=500.0, b=-0.1)
    spec = {"mission": "one", "duration_s": 3600.0,
            "blocks": [{"pattern": "bend", "mean": 0.0, "amplitude": 50.0, "cycles": 1000.0, "source": "b1"}]}
    fat = talos.assess_fatigue({"bend": (res, 100.0)}, spec, curve, workdir=tmp_path / "fat")
    assert fat.result.ok, fat.result.messages
    s_max = res.metrics["max_von_mises"]                     # at 100 N -> amplitude at 50 N is s_max/2
    expected = 1000.0 / float(curve.cycles_to_failure(s_max / 2))
    assert fat.result.metrics["damage_per_pass"] == pytest.approx(expected, rel=1e-6)
    assert fat.result.metrics["passes_to_failure"] == pytest.approx(1 / expected, rel=1e-6)
    assert fat.result.metrics["hours_to_failure"] == pytest.approx(1 / expected, rel=1e-6)
    assert (tmp_path / "fat" / "damage.npz").is_file()
    assert fat.contributions["b1"] == pytest.approx(expected, rel=1e-6)
    bad = talos.assess_fatigue({}, spec, curve)
    assert not bad.result.ok and "without a unit case" in bad.result.messages[0]
