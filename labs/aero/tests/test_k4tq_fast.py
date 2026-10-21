"""K4TQ-F3 "Freddy 3" and K4TQ-G3 "Ginna 3": the Jetson, bigger motors than the parents', and a top speed at or above
the 15-inch FPV quad's with its payload — with the hover rules, R3 and the root within their ratings."""
import k4tq
import k4tq_fast as kf
import k4tq_scaled as ks
import k4tq_sizing as sz


def test_fast_designs_meet_the_fpv_speed():
    V_t = kf.target_speed()
    assert 30.0 < V_t < 40.0
    for d in ("F3", "G3"):
        p = k4tq.params(d)
        assert p["autonomy"] and kf.bigger(d, p["motor"])
        e = kf.evaluate(sz.vehicle(d))
        assert e["top speed [m/s]"] >= kf.SPEED_MARGIN * V_t
        assert e["hover ok"] and e["R3 closes"] and e["R3 margin [Wh]"] >= kf.MARGIN_MIN and e["root ok"]
    assert sz.vehicle("G3")["mass"]["mass_kg"] <= ks.LIMITS["G"] + 0.01
    assert k4tq.params("F3")["payload_kg"] == 1.0
