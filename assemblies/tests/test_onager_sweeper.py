"""(Copied from notebooks/designs/tests with only the imports changed, and files loaded by path loaded by module
name: the promoted copy must pass the same tests.)
Onager Sweeper (notebook 23): design/CAD consistency, masses, the suction template and the litter pick-up
analysis, standing, the vacuum hook, and the mission — sweep, brick and box into the basket (slow).

Run: cd /home/user/vegeta && python3 -m pytest -q notebooks/designs/tests/test_onager_sweeper.py
"""
import math

import numpy as np
import pytest

pytest.importorskip("mujoco")

from assemblies.components import onager_controller as oc  # noqa: E402
from assemblies.components import onager_manus_robot as omr  # noqa: E402
from assemblies.components import onager_robot as orb  # noqa: E402
from assemblies.components import onager_sweeper_cfd as cfd  # noqa: E402
from assemblies.components import onager_sweeper_robot as osr  # noqa: E402
from assemblies.components import onager_sweeper_scenario as oss  # noqa: E402
from vegeta import chiron as ch  # noqa: E402


def test_design_defaults_match_the_dedalus_design():
    onager_sweeper = pytest.importorskip("assemblies.components.onager_sweeper")
    defaults = {prm.name: prm.default for prm in onager_sweeper.OnagerSweeper.parameters if prm.name != "part"}
    assert {k: defaults[k] for k in osr.SWEEPER} == osr.SWEEPER
    assert {k: defaults[k] for k in osr.CHASSIS} == osr.CHASSIS
    assert {k: defaults[k] for k in osr.ARMS} == osr.ARMS


def test_stored_cad_numbers_match_a_rebuild():
    pytest.importorskip("cadquery")
    fresh = osr.cad_numbers(recompute=True)
    for k in osr.CAD:
        assert fresh[k] == pytest.approx(osr.CAD[k], rel=0.01)


def test_mass_budget_and_balance():
    r = osr.sweeper()
    assert r.total_mass() == pytest.approx(sum(osr.mass_budget().values()), abs=1e-6)
    assert 540 < r.total_mass() < 640
    assert len(r.actuated_joints()) == 12 + 12 + 1
    assert abs(r.battery_x) < osr.orb.BATTERY_X_MAX            # the battery balances the arms and the basket inside its compartment


def test_arm_reaches_the_road_and_the_basket():
    p = osr.design_params()
    g = omr.arm_geometry(p)
    sg = osr.sweeper_geometry(p)
    down = -math.pi / 2
    assert omr.arm_ik((g["x"] + 0.75, 0.0, 0.07 + 0.17 - sg["hull_z"]), down, "L", p) is not None   # the brick ahead of the wheels
    bz1 = sg["basket"][5]
    for side in omr.SIDES:
        q = omr.arm_ik((-0.45, omr.SIDES[side] * (g["y"] + 0.02), bz1 + 0.36), down, side, p)
        assert q is not None and abs(q[0]) > math.radians(170)                                      # swung right back


def test_pickup_analysis():
    t = cfd.pickup_table()
    assert t.loc["dry leaf", "hood lifts it"] and t.loc["cigarette butt", "duct carries it"]
    assert not t.loc["pebble 15 mm", "hood lifts it"]                 # a stone stays for the broom
    assert cfd.terminal_velocity(cfd.DEBRIS["gravel chip 5 mm"]) > cfd.terminal_velocity(cfd.DEBRIS["dry leaf"])


def test_suction_template_renders(tmp_path):
    pytest.importorskip("cadquery")
    from vegeta import aeromant, dedalus

    d = dedalus.load_design("assemblies.components.onager_sweeper:OnagerSweeper")
    p = d.resolve()
    stl = d.generate(part="hood").export(tmp_path / "hood", stl_tolerance=0.5).artifacts["stl"]
    case = cfd.suction_case(stl, tmp_path / "case", p, flow_rate=osr.FAN_FLOW, environment=aeromant.OpenFOAMEnvironment())
    res = case.prepare()
    assert res.ok, res.messages
    derived = aeromant.open_case(tmp_path / "case")["derived"]
    assert float(derived["SUCTION_W"]) == pytest.approx(osr.FAN_FLOW / (p["duct_inner"] / 1000) ** 2)
    assert float(derived["X1"]) < p["hood_x"] / 1000 < float(derived["X2"])
    assert "{{" not in (tmp_path / "case/system/blockMeshDict").read_text().replace("{{...}}", "")


def test_standing_with_the_broom_running():
    lab = osr.sweeper_lab(ch.Flat())
    ep = lab.run(oc.Stand(), duration=1.5, rules=None, settle=0.0)
    fz = np.asarray(ep.log["foot_force"])[-1, :, 2]
    assert fz.sum() == pytest.approx(lab.total_mass * orb.G, rel=0.03)
    assert fz.max() / fz.min() < 1.15                               # balanced on its four wheels


def test_vacuum_hook_collects_a_can():
    scene = oss.Scene(litter=[("can", -0.1, 0.0), ("packet", 2.5, 0.0)], pickups=[], rough_rms=0.0)   # a can under the hood of the standing robot
    lab = oss.make_lab(scene)
    lab.fan_on = True
    lab.run(oc.Stand(), duration=1.0, rules=None, settle=0.0)
    assert "can_0" in lab.vacuum.collected and "packet_1" not in lab.vacuum.collected
    assert [e for e in lab.events if e[1] == "vacuum"]
    lab.fan_on = False
    lab.run(oc.Stand(), duration=0.5, rules=None, settle=0.0)
    assert not lab.vacuum.collected                                   # the fan off: nothing moves


@pytest.mark.slow
def test_mission_sweeps_and_loads_the_basket():
    scene = oss.Scene()
    lab = oss.make_lab(scene)
    ep = oss.run(lab, scene, duration=95.0)
    assert ep.log["mission_finished"]
    assert oss.in_basket(ep, "brick") and oss.in_basket(ep, "box")
    assert len(ep.log["collected"]) >= 3                            # the cans were vacuumed (the denser packets pass the hood at 1 m/s)
    ts = oss.timeseries(ep)
    assert ts.x.iloc[-1] > scene.x_end - 0.3
    assert not [e for e in ep.log["events"] if e[1] == "mujoco"]
