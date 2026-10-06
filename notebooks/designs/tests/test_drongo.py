"""Drongo (notebook 08b): the design and its numbers, the propulsion (assumed, or notebook 08's Boreas export), the grip
and net arithmetic, the flight controller's pieces, a hover in ChironLab, and (slow) the two deliveries — dropped into
the net and put down on the zone — with the judge that says whether the potato and the cream arrived unspoilt.

Run: cd /home/user/vegeta && python3 -m pytest -q notebooks/designs/tests/test_drongo.py         (add -m "not slow")
"""
import math

import numpy as np
import pytest

pytest.importorskip("mujoco")

import drongo_controller as dc  # noqa: E402
import drongo_robot as dr  # noqa: E402
import drongo_scenario as ds  # noqa: E402
from vegeta import chiron as ch  # noqa: E402


# ----------------------------------------------------------------------------------------------- design and numbers
def test_design_defaults_match_the_dedalus_design():
    pytest.importorskip("cadquery")
    import drongo

    defaults = {prm.name: prm.default for prm in drongo.Drongo.parameters}
    assert {k: defaults[k] for k in dr.DRONGO} == dr.DRONGO


def test_stored_cad_numbers_match_a_rebuild():
    pytest.importorskip("cadquery")
    fresh = dr.cad_numbers(recompute=True)
    for k in dr.CAD:
        assert fresh[k] == pytest.approx(dr.CAD[k], rel=0.01)


def test_drongo_design_checks_its_clearances():
    pytest.importorskip("cadquery")
    import drongo
    from vegeta.dedalus.design import BuildError

    with pytest.raises(BuildError, match="clear the tallest item"):
        drongo.Drongo().generate(part="gear", skid_height=90.0)


def test_mass_budget_is_the_robot():
    r = dr.drongo()
    assert r.total_mass() == pytest.approx(sum(dr.mass_budget().values()) / 1000, abs=1e-6)
    assert 0.55 < r.total_mass() < 0.62                              # notebook 08's quad + 0.1 kg of gear and pincer
    assert sorted(r.actuated_joints()) == sorted(dr.JAW_JOINTS)


def test_items():
    assert dr.POTATO["mass_kg"] == 0.200 and dr.POTATO["diameter_m"] == pytest.approx(0.0707, abs=5e-4)
    assert dr.CREAM["mass_kg"] == pytest.approx(0.412)
    volume = math.pi / 4 * dr.CREAM["diameter_m"] ** 2 * dr.CREAM["height_m"]
    assert dr.CREAM["content_kg"] / dr.CREAM["density_kg_m3"] < volume          # the cream fits in its cup
    for it in dr.ITEMS.values():                                      # both items fit between the open pads
        assert it["diameter_m"] < dr.pad_gap((dr.geometry()["travel"],) * 2) - 0.04


def test_jaw_drive_cannot_crush_the_cream():
    s = dr.jaw_servo()
    assert s.stall_torque < dr.CREAM["squeeze_limit_N"] < dr.POTATO["squeeze_limit_N"]
    assert s.stall_torque == pytest.approx(0.55 / (2 * dr.PINION_RADIUS_M))


# ----------------------------------------------------------------------------------------------- propulsion
def test_assumed_propulsion(tmp_path):
    prop = dr.propulsion(tmp_path / "none.json")
    assert prop.max_thrust_N == 8.0 and "assumed" in prop.source
    for t_g, eff in dr.ASSUMED_PROPULSION["efficiency_g_per_W"]:
        assert prop.electrical_power(t_g * dr.G / 1000) == pytest.approx(t_g / eff, rel=1e-9)
    T = np.linspace(0.1, 9.0, 50)
    assert np.all(np.diff(prop.electrical_power(T)) > 0) and prop.electrical_power(0.0) == 0.0
    assert prop.usable_Wh == pytest.approx(4 * 3.7 * 1.5 * 0.8)


def test_propulsion_from_notebook_08s_export(tmp_path):
    """Notebook 08's propeller, motor and battery through Boreas, exported as notebook 08 exports them."""
    from vegeta import boreas

    d, p = boreas.inches(5, 4.3)
    prop = boreas.Propeller.from_pitch("5x4.3 tri-blade", d, p, blades=3, chord_root_m=0.010, chord_max_m=0.016,
                                       chord_tip_m=0.006, mass_kg=0.0045, rotor_mass_kg=0.020)
    airfoil = boreas.Airfoil(name="thin cambered section", cl_alpha=2 * math.pi * 0.9, alpha0_deg=-3.0, cl_max=1.1,
                             cd0=0.025, k=0.045)
    motor = boreas.Motor("2306-2400KV", kv_rpm_per_volt=2400, resistance_ohm=0.06, no_load_current_a=1.2,
                         max_current_a=40, mass_kg=0.030)
    battery = boreas.Battery("4S 1500 mAh", cells=4, capacity_ah=1.5, usable_fraction=0.8, mass_kg=0.180)
    system = boreas.Propulsion(prop, airfoil, motor, battery, rho=1.2)
    points = {"hover": system.for_thrust(1.5), "cruise": system.for_thrust(2.0), "full": system.at_throttle(1.0)}
    path = tmp_path / "quad_5x43.json"
    boreas.export(path, prop, airfoil, points=points, motor=motor, battery=battery)
    pr = dr.propulsion(path)
    assert "export" in pr.source
    assert pr.max_thrust_N == pytest.approx(points["full"].thrust)
    assert pr.electrical_power(points["hover"].thrust) == pytest.approx(points["hover"].electrical_power, rel=1e-6)
    assert pr.torque_per_thrust_m == pytest.approx(points["hover"].aero.torque / points["hover"].thrust)


# ----------------------------------------------------------------------------------------------- grip and net
@pytest.mark.parametrize("name", ["potato", "cream"])
def test_grip_window(name):
    it = dr.ITEMS[name]
    F = dr.GRIP_N[name]
    assert dr.grip_needed(it, 4.0, 2.0, sf=2.0) <= F <= it["squeeze_limit_N"] / 1.5
    assert dr.grip_needed(it, dr.grip_holds(it, F), 0.0) == pytest.approx(F, rel=1e-9)   # round trip


def test_net_catch():
    c = dr.net_catch(dr.POTATO, 4.0)
    assert c["speed at the net [m/s]"] == pytest.approx(math.sqrt(2 * dr.G * 4.0))
    assert c["net holds"] and c["item survives"]
    assert c["load on the item [N]"] == pytest.approx(0.2 * dr.G * (1 + 4.0 / dr.NET["stretch_m"]))
    assert not dr.net_catch(dr.CREAM, 5.5)["net holds"]


# ----------------------------------------------------------------------------------------------- the flight controller
def test_profile_reaches_its_distance_within_its_limits():
    for dist in (0.4, 3.0, 36.5):
        p = dc.Profile(dist, 6.0, 3.0)
        assert p.at(p.T)[0] == pytest.approx(dist) and p.at(p.T + 5)[1] == 0.0
        assert p.s[-1] == pytest.approx(dist, rel=1e-6)
        assert np.max(p.v) <= 6.0 * 1.001 and np.max(np.abs(p.a)) <= 3.0 * 1.05
        assert np.all(np.diff(p.s) >= -1e-12)


def test_mixer_reproduces_thrust_and_torques():
    lab = ds.make_lab(ds.Scene())
    f = dc.Flight(dr.propulsion())
    f.reset(lab)
    want = np.array([6.0, 0.05, -0.04, 0.01])
    T, sat = f.mix(want[0], want[1:])
    assert not sat and f.A @ T == pytest.approx(want, abs=1e-9)
    T, sat = f.mix(31.0, np.array([0.8, 0.0, 0.0]))                   # near full thrust: limited, never above T_max
    assert sat and T.max() <= dr.ASSUMED_PROPULSION["max_thrust_N"] + 1e-9 and T.sum() == pytest.approx(31.0, rel=1e-6)


def test_takeoff_and_hop_hover_on_the_weight():
    """The first seconds of the mission: the rotors lift Drongo to the hop height and hold it there on its weight."""
    scene = ds.Scene("drop")
    lab = ds.make_lab(scene)
    ep = ds.run(lab, scene, duration=4.4)
    ts = ds.timeseries(ep)
    hover = ts[(ts.t > 3.9) & (ts.t < 4.1)]                           # over the potato, before it descends
    assert hover.z.mean() == pytest.approx(1.5, abs=0.02)                  # the frame at the hop height
    assert hover.x.mean() == pytest.approx(scene.potato_at[0], abs=0.02)
    assert hover.y.mean() == pytest.approx(scene.potato_at[1], abs=0.02)
    thrust = hover[["T1", "T2", "T3", "T4"]].sum(axis=1).mean()
    assert thrust == pytest.approx(lab.total_mass * dr.G, rel=0.03)
    assert ts.tilt_deg.max() < dc.Flight(dr.propulsion()).tilt_max * 180 / math.pi
    assert all(r["delivered"] is None and not r["damage"] for r in ep.log["records"].values())


# ----------------------------------------------------------------------------------------------- the deliveries
@pytest.mark.slow
@pytest.mark.parametrize("variant", ["drop", "place"])
def test_delivery_arrives_unspoilt(variant):
    scene = ds.Scene(variant)
    lab = ds.make_lab(scene)
    ep = ds.run(lab, scene, duration=100.0)
    assert ep.log["mission_finished"] and ep.outcome["success"], ep.outcome
    table = ds.deliveries(ep)
    assert table["unspoilt"].all()
    assert (table["peak squeeze [N]"] < table["squeeze limit [N]"]).all()
    waits = ds.wait_times(ep)
    assert waits["potato [s]"] < waits["cream [s]"] == waits["both [s]"] < waits["Drongo home [s]"]
    if variant == "drop":
        assert (table["fall onto the net [m]"] <= dr.NET["max_drop_m"]).all()
        assert (table["how"] == "caught in the net").all()
    else:
        assert (table["how"] == "put down on the zone").all()
        assert (table["touchdown speed [m/s]"] < 1.2 * dc.Plan().v_place).all()
    assert ep.log["energy_Wh"] < dr.propulsion().usable_Wh


@pytest.mark.slow
def test_a_drop_from_too_high_tears_the_net():
    scene = ds.Scene("drop")
    lab = ds.make_lab(scene)
    ep = ds.run(lab, scene, dc.Plan(release_above_net=5.6, items=("potato",)), duration=40.0)
    rec = ep.log["records"]["potato"]
    assert any("tore the net" in d for d in rec["damage"])
    assert any("smashed on the ground" in d for d in rec["damage"]) and rec["delivered"] is None
    assert not ep.outcome["success"]


@pytest.mark.slow
def test_a_weak_grip_leaves_the_cream_behind():
    """Half a newton per pad cannot lift 412 g at μ 0.5: Drongo flies off and the cream stays by the kitchen."""
    scene = ds.Scene("drop")
    lab = ds.make_lab(scene)
    plan = dc.Plan(items=("cream",), grip={"cream": 0.5})
    ep = ds.run(lab, scene, plan, duration=25.0)
    rec = ep.log["records"]["cream"]
    assert rec["delivered"] is None
    assert np.linalg.norm(np.array(rec["final_pos"][:2]) - np.array(scene.cream_at)) < 0.1
