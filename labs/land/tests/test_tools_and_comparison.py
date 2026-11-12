"""Interchangeable tools on the family's arms (pekari_arms.TOOLS / with_tool, pekari_family.with_tool,
pumba_robot.PUMBA_SAW) and the land lab comparison (designs/land_comparison.py).

Run: python tools/lab.py test land -- -k "tool or comparison"
"""
import math

import pytest

import pekari_arms as pa
import pekari_family as pf
import pumba_robot as pu


def test_swapping_back_to_the_grab_restores_the_arm():
    saw = pa.with_tool(pa.PUMBA_MAIN_ARM, "chainsaw 2 kW")
    back = pa.with_tool(saw, "two-jaw grab")
    assert back.tool == pytest.approx(pa.PUMBA_MAIN_ARM.tool)
    assert pa.total_mass(back) == pytest.approx(pa.total_mass(pa.PUMBA_MAIN_ARM))
    assert pa.rated_load(back, 1.0) == pytest.approx(pa.rated_load(pa.PUMBA_MAIN_ARM, 1.0))


def test_the_saw_replaces_the_jaws():
    saw = pa.with_tool(pa.PUMBA_MAIN_ARM, "chainsaw 2 kW")
    assert "grip" not in saw.drives and saw.saw == "chainsaw 2 kW, 36 V" and saw.rated_payload_kg == 0.0
    assert pa.flange(saw) == pytest.approx(pa.flange(pa.PUMBA_MAIN_ARM))
    assert pa.reach(saw) > pa.reach(pa.PUMBA_MAIN_ARM)
    q = pa.qpos(saw, "main_", pa.pose(saw, "cut"))
    assert "main_grip_a" not in q
    assert pa.fk(saw, pa.pose(saw, "cut"))["elevation"] == pytest.approx(0.0, abs=1e-9)   # the bar level


def test_couplers_must_match():
    with pytest.raises(ValueError, match="coupler"):
        pa.with_tool(pa.PEKARI_ARM, "chainsaw 2 kW")
    pa.with_tool(pa.CALYDON_MAIN_ARM, "chainsaw 2 kW")


def test_cut_check_against_perseus_measure():
    saw = pa.with_tool(pa.PUMBA_MAIN_ARM, "chainsaw 2 kW")
    c = pa.cut_check(saw, 0.95 * pa.reach(saw), dynamic=1.5)
    assert all(v > 1.2 for v in c["sf_stall"].values())             # holds the cut with margin
    assert c["max_branch_m"] == pytest.approx(0.40)
    light = pa.cut_check(pa.with_tool(pa.PEKARI_ARM, "pruning saw 0.6 kW"), dynamic=1.0)
    assert min(light["sf_stall"].values()) < 1.0                    # the light arm cannot hold a saw's cut


def test_pumba_saw_platform():
    assert pf.capabilities(pu.PUMBA_SAW)["tags"] >= {"cut", "dual_arm"}
    assert "heavy_lift" not in pf.capabilities(pu.PUMBA_SAW)["tags"]
    assert abs(pf.total_mass(pu.PUMBA_SAW) - pf.total_mass(pu.PUMBA)) < 2.0
    rows = dict((r[0], r) for r in pf.bom(pu.PUMBA_SAW))
    assert "drive: jaw screw 6 kN" not in rows and not any("two-jaw grab" in k for k in rows)


def test_pumba_saw_chiron_model():
    pytest.importorskip("mujoco")
    r = pu.pumba(tool="chainsaw 2 kW")
    names = {g.name for g in r.geoms()}
    assert "main_saw_bar" in names and "main_grip_a" not in names
    assert r.total_mass() == pytest.approx(pf.total_mass(pu.PUMBA_SAW, pu.PUMBA.payload_kg), rel=1e-6)


def test_comparison_table():
    import land_comparison as lc

    r = lc.rows(cad=False)
    assert set(r) == set(lc.MACHINES)
    assert r["Pumba"]["chassis_kg"] <= pu.WILK_REFERENCE["empty_kg"]
    assert r["Perseus"]["empty_kg"] == pytest.approx(180.6, abs=1.0)
    assert r["Calydon"]["empty_kg"] > r["Pumba"]["empty_kg"] > r["Perseus"]["empty_kg"] > r["Pekari-2M"]["empty_kg"]
    assert all(row["climb20"] for row in r.values())
    df = lc.table(cad=False)
    assert list(df.columns) == list(lc.MACHINES) and len(df) == len(lc.LABELS)
