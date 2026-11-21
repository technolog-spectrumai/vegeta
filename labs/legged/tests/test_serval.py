"""Serval (designs/serval.py): light, fast, stable with both turrets; the tools' numbers; the flak's back-blast rule."""
import gorilla_quad as gq
import serval


def test_lighter_and_faster_than_kerchaks_quad():
    s = serval.sizing()
    assert s["mass_kg"] < gq.sizing()["mass_total_kg"] - 300          # no gorilla aboard
    assert s["top_speed_kmh"] >= 79.0
    assert s["t_0_70_s"] < 6.0
    assert s["power_at_top_speed_kW"] < 4 * serval.DRIVE["cont_kW"]   # it cruises flat out on the continuous rating


def test_stable_with_both_turrets():
    s = serval.sizing()
    assert s["ssf"] > 1.2                                             # it slides (0.8 g) before it rolls
    assert s["rollover_lat_g"] > serval.MU


def test_the_tools():
    s = serval.sizing()
    assert s["golf"]["range_m_at_70"] > 100.0
    assert s["flak"]["shots"] >= 400
    assert serval.FLAK_MOUNT["pivot"][2] > serval.GOLF["ring"][2] + 0.6   # the flak on top


def test_the_back_blast_rule():
    assert serval.blast_clear(0, 10)                                  # level over the bow: the blast goes back over everything
    assert not serval.blast_clear(0, 80)                              # steeply up: the blast comes down on the golf turret
    assert serval.BLAST_KEEP_OUT[90] >= 55                            # to the sides it may fire high
    assert all(v is not None and v >= 30 for v in serval.BLAST_KEEP_OUT.values())


def test_hit_and_run():
    h = serval.hit_and_run()
    assert h["flak_shots"] > 20 and h["runs_per_magazine"] >= 3
