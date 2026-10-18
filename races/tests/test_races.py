"""Smoke tests: the scenes, the zones' clearance, the mock laps, the rules, the CLI. No MuJoCo run."""
import xml.etree.ElementTree as ET

import pytest

from races import cli, courses, mock, refquad, replay, rules


@pytest.fixture(scope="module", params=list(courses.COURSES))
def course(request):
    return courses.build(request.param)


def test_reference_quad_budget_and_drive():
    s = refquad.summary()
    assert 0.5 < s["all-up mass [kg]"] < 0.65 and s["motor"].endswith("KV") and "6S" in s["voltage"]
    assert 8 < s["hover thrust / weight"] < 16 and 50 < s["hover power [W]"] < 200


def test_scene_builds_and_parses(course, tmp_path):
    xml = courses.to_mjcf(course)
    root = ET.fromstring(xml)
    names = {g.get("name") for g in root.iter("geom")}
    assert "ground" in names and "plate" in names and any(n and n.startswith("zone_") for n in names)
    cams = {c.get("name") for c in root.iter("camera")}
    assert {"overview", "grandstand", "onboard", "chase"} <= cams


def test_zones_clear_the_scene_and_the_line_is_flyable(course):
    ck = course.clearance_check()
    assert ck["ok"].all(), ck[~ck["ok"]]
    assert course.zones[0].kind == "start" and course.zones[-1].kind == "finish"
    assert any(z.kind == "checkpoint" for z in course.zones)
    assert len(course.start_grid) == 4
    p = mock.schedule(course)
    assert p.v.min() >= 2.0 and p.v.max() <= 28.0 and p.time_estimate() < 120


def test_mock_completes_every_course(course):
    r = mock.run_mock(course)
    st = r["stats"]
    assert st["status"] == "finished", (course.slug, st, r["events"][-5:])
    assert st["collisions"] == 0 and st["missed gates"] == 0
    assert st["lap time [s]"] < 1.2 * st["schedule time [s]"] + 10
    assert 0 < st["energy [Wh]"] < refquad.ENERGY_WH
    assert len(r["gate times"]) == len(course.zones) - 1


def test_rules_reset_count_stop():
    c = courses.farm()
    r = mock.run_mock(c, crash_at=5)
    assert r["stats"]["status"] == "finished" and r["stats"]["resets"] == 1 and r["stats"]["penalty [s]"] == 5.0
    r = mock.run_mock(c, crash_at=5, rules=rules.Rules(collision="count"))
    assert r["stats"]["status"] == "finished" and r["stats"]["collisions"] == 1 and r["stats"]["penalty [s]"] == 0.0
    r = mock.run_mock(c, crash_at=5, rules=rules.Rules(collision="stop"))
    assert r["stats"]["status"] == "dnf" and r["stats"]["reason"] == "collision"


def test_cli_list_build_and_mock_run(tmp_path, capsys):
    assert cli.main(["list"]) == 0
    assert cli.main(["build", "--out", str(tmp_path / "scenes")]) == 0
    assert (tmp_path / "scenes" / "quarry.xml").exists()
    assert cli.main(["run", "--course", "hangar", "--mock", "--out", str(tmp_path / "runs")]) == 0
    assert (tmp_path / "runs" / "hangar_refquad_mock" / "stats.json").exists()
    out = capsys.readouterr().out
    assert "finished" in out
