"""The held-out validation of the autopilot (races.validate): the scenario set, the metrics and grades; one MuJoCo
check that a neutral scenario is the nominal lap (skipped without MuJoCo)."""
import pytest

from races import validate


def test_scenarios_are_seeded_and_held_out():
    a, b, c = validate.scenarios(5), validate.scenarios(5), validate.scenarios(5, seed=1)
    assert a == b and a != c
    assert sum(s["family"] == "nominal" for s in a) == 1 and len(a) == 1 + 5 * (len(validate.FAMILIES) - 1)
    nom = next(s for s in a if s["family"] == "nominal")
    assert nom["mult"] == {} and nom["scenario"] == {}
    only = validate.scenarios(5, families=("nominal", "wind ≤ 2 m/s"))
    assert [s for s in a if s["family"] == "wind ≤ 2 m/s"] == [s for s in only if s["family"] == "wind ≤ 2 m/s"]   # a family's draws do not depend on the others


def test_scenario_ranges():
    for s in validate.scenarios(50):
        sc, m = s["scenario"], s["mult"]
        assert all(0.98 <= f <= 1.02 for f in m.values())
        assert 0.95 <= sc.get("mass_scale", 1.0) <= 1.05 and 0.95 <= sc.get("thrust_scale", 1.0) <= 1.05
        assert sum(w * w for w in sc.get("wind", (0, 0, 0))) ** 0.5 <= 2.0 + 1e-9
        dx, dy, dz = sc.get("start_offset", (0, 0, 0))
        assert abs(dx) <= 0.5 and abs(dy) <= 0.5 and abs(dz) <= 0.2
        if s["family"] == "combined":
            assert set(sc) == {"start_offset", "mass_scale", "thrust_scale", "wind"} and len(m) == 6


def test_apply_keeps_no_cap():
    s = validate.apply({"safety": 0.5, "v_cap": None}, {"safety": 1.02, "v_cap": 0.98})
    assert s == {"safety": 0.51, "v_cap": None}


def _row(fam, lap, coll=0):
    return {"family": fam, "lap time [s]": lap, "collisions": coll}


def test_summary_and_grades():
    rows = [_row("nominal", 30.0)] + [_row("wind ≤ 2 m/s", 31.0)] * 8 + [_row("wind ≤ 2 m/s", 33.0, 1), _row("wind ≤ 2 m/s", None, 6)] \
        + [_row("mass ±5 %", 30.0)] * 5 + [_row("mass ±5 %", None)] * 5
    s = validate.summarise(rows)
    w, m = s["families"]["wind ≤ 2 m/s"], s["families"]["mass ±5 %"]
    assert (w["clean rate"], w["finish rate"], w["median lap [s]"], w["gap [s]"], w["grade"]) == (0.8, 0.9, 31.0, 1.0, "pass")
    assert (m["clean rate"], m["grade"]) == (0.5, "weak")
    assert s["verdict"] == "weak" and s["weakest family"] == "mass ±5 %" and s["held-out clean rate"] == 0.65
    assert validate.summarise([_row("nominal", 30.0), _row("mass ±5 %", 30.0)])["weakest family"] == "none"
    assert validate.grade(0.49) == "fail" and validate.summarise(rows + [_row("thrust ±5 %", None)])["verdict"] == "fail"


def test_neutral_scenario_is_the_nominal_lap():
    pytest.importorskip("mujoco")
    from races import courses, sim
    kw = dict(control_dt=0.004, duration=8.0)
    a = sim.run_mujoco(courses.industrial(), **kw)
    b = sim.run_mujoco(courses.industrial(), scenario={"start_offset": [0, 0, 0], "mass_scale": 1.0, "thrust_scale": 1.0, "wind": [0, 0, 0]}, **kw)
    assert (a["history"] == b["history"]).all() and a["stats"]["zones passed"] == b["stats"]["zones passed"]
