"""The small craft: each in its class's band (length, speed), the displacement craft held by their hull speed, the
planing ones by their power and - the skiff - by its crew's impacts; speed falls with the sea; the signatures say
who is quiet; no weapons; the CAD builds."""
import pytest

import small_craft as sc

BANDS = {"fishing_boat": ((15, 25), (6, 10.5)), "pilot_boat": ((15, 17), (20, 25.5)), "yacht": ((10, 14), (4, 8.5)),
         "skiff": ((6, 8), (25, 40))}


@pytest.mark.parametrize("name", list(sc.CRAFT))
def test_in_class_band(name):
    s = sc.sizing(name)
    (l0, l1), (v0, v1) = BANDS[name]
    assert l0 <= s["length_m"] <= l1 and v0 <= s["max_kn"] <= v1, (s["length_m"], s["max_kn"])
    t = [r["kn"] for r in s["speed_vs_hs"]]
    assert t == sorted(t, reverse=True) and t[-1] > 0
    assert s["displacement_t"] > 0 and s["accel_0_cruise_s"] > 0 and s["basis"]
    assert s["crew"][0] >= 1 and s["rcs_m2"]["beam"] >= s["rcs_m2"]["bow"]


def test_who_is_quiet_and_who_is_big():
    f, sk = sc.sizing("fishing_boat"), sc.sizing("skiff")
    assert f["ais_typical"] and sc.sizing("pilot_boat")["ais_typical"] and not sk["ais_typical"] and not sc.sizing("yacht")["ais_typical"]
    assert sk["rcs_m2"]["beam"] < sc.sizing("yacht")["rcs_m2"]["beam"] < sc.sizing("pilot_boat")["rcs_m2"]["beam"] < f["rcs_m2"]["beam"]
    assert sk["speed_vs_hs"][2]["limited_by"].startswith("slamming") and sk["max_kn_light"] >= sk["max_kn"]


def test_barracuda_outruns_the_skiff_in_a_sea():
    from barracuda import Barracuda
    b = {r["hs_m"]: r["sustained_kn"] for r in Barracuda.sizing()["speed_vs_hs"]}
    for r in sc.sizing("skiff")["speed_vs_hs"]:
        assert b[r["hs_m"]] >= r["kn"]


def test_cad_builds():
    pytest.importorskip("cadquery")
    for n in sc.CRAFT:
        assert sc.build(n).val().isValid()
