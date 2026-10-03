"""(Copied from notebooks/designs/tests with only the imports changed, and files loaded by path loaded by module
name: the promoted copy must pass the same tests.)
Onager Atlas (notebook 21) in ChironLab: design/CAD consistency, masses, the logistics stance, the forklift's
joints, four-quadrant drives, and the pallet job (pick up, carry, set down).

Run: cd /home/user/vegeta && python3 -m pytest -q notebooks/designs/tests/test_onager_atlas.py
"""
import math

import numpy as np
import pytest

pytest.importorskip("mujoco")

from assemblies.components import onager_atlas_robot as oar  # noqa: E402
from assemblies.components import onager_atlas_scenario as oas  # noqa: E402
from assemblies.components import onager_controller as oc  # noqa: E402
from assemblies.components import onager_robot as orb  # noqa: E402
from vegeta import chiron as ch  # noqa: E402


@pytest.fixture(scope="module")
def robot():
    return oar.atlas()


def test_design_defaults_match_the_dedalus_design():
    onager_atlas = pytest.importorskip("assemblies.components.onager_atlas")
    defaults = {prm.name: prm.default for prm in onager_atlas.OnagerAtlas.parameters if prm.name != "part"}
    assert {k: defaults[k] for k in oar.ATLAS} == oar.ATLAS
    assert {k: defaults[k] for k in oar.CHASSIS} == oar.CHASSIS


def test_stored_cad_numbers_match_a_rebuild():
    pytest.importorskip("cadquery")
    fresh = oar.cad_numbers(recompute=True)
    for k in oar.CAD:
        assert fresh[k] == pytest.approx(oar.CAD[k], rel=0.01)


def test_mass_budget_and_geometry(robot):
    assert robot.total_mass() == pytest.approx(sum(oar.mass_budget().values()), abs=1e-6)
    assert 650 < robot.total_mass() < 800
    g, f = robot.geometry, robot.forklift
    wheel_front = g["shoulder_x"] + g["axle_x"] + g["r_wheel"]
    mast_rear = f["pivot"][0] - f["upright"][0] / 2
    assert wheel_front < mast_rear                       # the front wheels clear the mast
    assert all(s.four_quadrant for s in (oar.lift_servo(), oar.tilt_servo()))


def test_logistics_stance_cuts_the_knee_lever():
    sentinel, atlas = orb.geometry(), orb.geometry(oar.design_params())
    lever = lambda g: g["L2"] * math.sin(g["a2"])          # noqa: E731  knee to wheel contact, horizontally
    assert lever(atlas) < 0.4 * lever(sentinel)


def test_standing_and_the_lift(robot):
    lab = oar.atlas_lab(ch.Flat(), robot=robot)
    ep = lab.run(oc.Stand(), duration=1.5, rules=None, settle=0.0)
    fz = np.asarray(ep.log["foot_force"])[-1, :, 2]
    assert fz.sum() == pytest.approx(lab.total_mass * orb.G, rel=0.03)
    z0 = float(lab.data.geom_xpos[lab._geom_id("fork_L"), 2])

    class Lift:
        def reset(self, lab, seed=None):
            self.stand = oc.Stand()
            self.stand.reset(lab, seed)

        def __call__(self, obs):
            cmd = self.stand(obs)
            cmd.q_target["lift"] = 0.30
            return cmd

    lab.run(Lift(), duration=4.0, rules=None, settle=0.0)
    lab.observe(sync=True)
    z1 = float(lab.data.geom_xpos[lab._geom_id("fork_L"), 2])
    assert z1 - z0 == pytest.approx(0.30, abs=0.02)


@pytest.mark.slow
def test_pallet_job():
    scene = oas.Scene()
    lab = oas.make_lab(scene)
    ep = oas.run(lab, scene, duration=50.0)
    ts = oas.timeseries(ep)
    assert ts.pallet_z.max() > 0.15                       # it was lifted
    assert abs(ts.pallet_x.iloc[-1] - scene.x_drop) < 0.3  # set down near the mark
    assert ts.pallet_tilt_deg.iloc[-1] < 2.0              # flat on the ground
    assert ts.pallet_z.iloc[-1] < 0.02
    carry = ts[(ts.pallet_z > 0.1)]
    assert carry.pallet_tilt_deg.max() < 8.0               # it rode on the forks, tilted back with the mast
    assert not ep.log.get("events")                        # no MuJoCo instability
