"""K4TQ-F3 "Freddy 3" and K4TQ-G3 "Ginna 3": the Jetson, bigger off-the-shelf motors than the parents', a top speed at
or above the 15-inch FPV quad's with its payload, the range of F2 / G2 kept — with the hover rules, R3 and the root
within their ratings."""
import k4tq
import k4tq_fast as kf
import k4tq_scaled as ks
import k4tq_sizing as sz


def test_fast_designs_meet_the_fpv_speed_and_keep_the_range():
    V_t = kf.target_speed()
    assert 30.0 < V_t < 40.0
    for d in ("F3", "G3"):
        p = k4tq.params(d)
        cat = k4tq.CATALOGUE
        assert p["autonomy"] and kf.bigger(d, p["motor"])
        assert cat["motor"][p["motor"]].get("off_the_shelf") and cat["propeller"][p["prop"]].get("off_the_shelf")
        assert cat["battery"][p["battery"]]["cells_s"] <= cat["motor"][p["motor"]]["cells_max"]
        e = kf.evaluate(sz.vehicle(d))
        assert e["top speed [m/s]"] >= kf.SPEED_MARGIN * V_t
        assert e["hover ok"] and e["R3 closes"] and e["R3 margin [Wh]"] >= kf.MARGIN_MIN and e["root ok"]
        assert e["still-air range [km]"] >= kf.RANGE_KEEP * kf.parent_range(d)
    assert sz.vehicle("G3")["mass"]["mass_kg"] <= ks.LIMITS["G"] + 0.01
    assert k4tq.params("F3")["payload_kg"] == 1.0
