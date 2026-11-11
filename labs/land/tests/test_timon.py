"""Timon, the servicing rover (designs/timon_robot.py, timon.py) and the fleet's service tasks (fleet.py:
swap_battery, recover; fleet_scenario with_timon).

Run: python tools/lab.py test land -- -k timon
"""
import pytest

import fleet as fl
import fleet_scenario as fs
import pekari_family as pf
import pumba_robot as pu
import timon_robot as tr


def test_timon_meets_every_service_requirement():
    c = tr.check(tr.TIMON)
    assert set(c) == set(tr.REQUIREMENTS)
    assert all(ok for _, _, ok in c.values()), {k: v for k, v in c.items() if not v[2]}


def test_timon_is_the_lightest_size_that_passes():
    rows = tr.sweep(lengths=(1150.0, 1200.0, 1250.0), widths=(600.0, 620.0, 640.0))
    best = next(r for r in rows if r[3])
    assert (best[0], best[1]) == (1200.0, 620.0)
    assert dict(tr.CHASSIS) == tr.chassis(1200.0, 620.0)
    # the deck, not stability, decides: a shorter or narrower hull fails only the deck
    for L, W, _, ok, fails in rows:
        if not ok:
            assert "deck fits 2 modules + tool rack" in fails


def test_winching_needs_the_rear_anchor():
    no_spade = tr.design(1200.0, 620.0, cad=tr.CAD, spade=False)
    assert not tr.check(no_spade)["holds it on the winch on a 15 deg slope"][2]
    assert tr.winch_hold(tr.TIMON) > tr.tow_force(pu.PUMBA, 15.0)


def test_timon_in_the_fleet():
    prof = fl.profile_from_platform(tr.TIMON)
    assert {"service", "tow", "dual_arm"} <= prof.tags
    assert prof.width_m <= fl.profile_from_platform(pu.PUMBA).width_m
    assert pf.total_mass(tr.TIMON) < pf.total_mass(pu.PUMBA) + 5.0


def test_timon_stored_cad():
    pytest.importorskip("cadquery")
    fresh = pf.cad_numbers(tr.TIMON, recompute=True)
    for k, v in tr.CAD.items():
        assert fresh[k] == (pytest.approx(v, abs=0.5) if isinstance(v, tuple) else pytest.approx(v, rel=0.01))


def test_timon_holds_a_module_in_mujoco():
    pytest.importorskip("mujoco")
    r = pf.stand_test(tr.TIMON, "deployed", loads={"main": tr.MODULE["kg"]}, duration=1.0)
    assert r["tilt_deg"] < 0.5 and r["arm_err_deg"] < 2.0 and not r["belly"]


@pytest.fixture(scope="module")
def serviced():
    return fs.run(with_timon=True)


def test_scenario_with_timon_swaps_and_recovers(serviced):
    t = serviced["tasks"]
    assert t["swap_battery_P1"]["status"] == "done" and t["swap_battery_P1"]["agent"] == "TIMON"
    assert t["recover_P2"]["status"] == "done" and t["recover_P2"]["agent"] == "TIMON"
    assert serviced["agents"]["P1"]["soc"] > 0.95                   # a fresh module (a little used since)
    assert serviced["agents"]["P2"]["state"] == "recovered"
    assert all(v["status"] in ("done", "cancelled") for v in t.values())


def test_scenario_without_timon_unchanged():
    r = fs.run()
    assert not any(k.startswith(("swap_battery", "recover")) for k in r["tasks"])
