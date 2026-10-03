"""(Copied from notebooks/designs/tests with only the imports changed, and files loaded by path loaded by module
name: the promoted copy must pass the same tests.)
Onager Manus (notebook 22) in ChironLab: design/CAD consistency, masses, arm kinematics, standing, and the
mission — cut the wire across the track, lift the log off it, drive on.

Run: cd /home/user/vegeta && python3 -m pytest -q notebooks/designs/tests/test_onager_manus.py
"""
import math

import numpy as np
import pytest

pytest.importorskip("mujoco")

from assemblies.components import onager_controller as oc  # noqa: E402
from assemblies.components import onager_manus_robot as omr  # noqa: E402
from assemblies.components import onager_manus_scenario as oms  # noqa: E402
from assemblies.components import onager_robot as orb  # noqa: E402
from vegeta import chiron as ch  # noqa: E402


def test_design_defaults_match_the_dedalus_design():
    onager_manus = pytest.importorskip("assemblies.components.onager_manus")
    defaults = {prm.name: prm.default for prm in onager_manus.OnagerManus.parameters if prm.name != "part"}
    assert {k: defaults[k] for k in omr.MANUS} == omr.MANUS


def test_stored_cad_numbers_match_a_rebuild():
    pytest.importorskip("cadquery")
    fresh = omr.cad_numbers(recompute=True)
    for k in omr.CAD:
        assert fresh[k] == pytest.approx(omr.CAD[k], rel=0.01)


def test_mass_budget():
    r = omr.manus()
    assert r.total_mass() == pytest.approx(sum(omr.mass_budget().values()), abs=1e-6)
    assert 430 < r.total_mass() < 490
    assert len(r.actuated_joints()) == 12 + 12


@pytest.mark.parametrize("side", ["L", "R"])
@pytest.mark.parametrize("target,elev", [((1.6, 0.28, 0.0), -math.pi / 2), ((1.7, -0.28, -0.2), 0.0), ((1.4, 0.6, 0.3), -0.5)])
def test_arm_ik_round_trip(side, target, elev):
    q = omr.arm_ik(target, elev, side)
    assert q is not None
    pos, e = omr.arm_fk(q, side)
    assert np.allclose(pos, target, atol=1e-9) and e == pytest.approx(elev)


def test_arm_ik_out_of_reach():
    assert omr.arm_ik((4.0, 0.0, 0.0), 0.0, "L") is None


def test_standing_with_arms_stowed():
    lab = omr.manus_lab(ch.Flat())
    ep = lab.run(oc.Stand(), duration=1.5, rules=None, settle=0.0)
    fz = np.asarray(ep.log["foot_force"])[-1, :, 2]
    assert fz.sum() == pytest.approx(lab.total_mass * orb.G, rel=0.03)
    names = list(ep.log["joints"])
    q = np.asarray(ep.log["q"])
    for j in ("L_shoulder", "R_elbow", "L_wrist"):
        assert abs(q[-1, names.index(j)] - q[0, names.index(j)]) < 0.05


def test_cut_force_input():
    assert oms.F_CUT == pytest.approx(0.8 * 1200 * math.pi / 4 * 3.15 ** 2, rel=1e-9)


@pytest.mark.slow
def test_mission_cuts_the_wire_and_clears_the_log():
    scene = oms.Scene()
    lab = oms.make_lab(scene)
    ep = oms.run(lab, scene, duration=70.0)
    assert ep.log["mission_finished"]
    cut = [e for e in ep.log["events"] if e[1] == "wire"]
    assert cut and "cut" in cut[0][2]                      # the jaws squeezed hard enough
    ts = oms.timeseries(ep)
    assert ts.wire_a_deg.abs().iloc[-1] > 60 and ts.wire_b_deg.abs().iloc[-1] > 30   # the halves fell
    assert ts.log_z.max() > 0.4                            # the log was lifted
    assert abs(ts.log_y.iloc[-1]) > 0.75                   # and put down beside the track
    assert ts.x.iloc[-1] > scene.x_end - 0.2               # the robot drove on
    assert not [e for e in ep.log["events"] if e[1] == "mujoco"]
