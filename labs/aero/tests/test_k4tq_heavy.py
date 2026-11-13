"""K4TQ-H3 "Hanna 3" and K4TQ-J3 "Joanna 3": the payloads, the hover rules, R3 and Ginna 3's range kept, off the shelf."""
import k4tq
import k4tq_fast as kf
import k4tq_heavy as kh
import k4tq_sizing as sz


def test_heavy_designs():
    r_g3 = kh.ginna3_range()
    for d, pay in (("H3", 3.5), ("J3", 4.5)):
        p = k4tq.params(d)
        assert p["payload_kg"] >= pay and p["autonomy"]
        for key in ("prop", "prop_rear"):
            assert k4tq.CATALOGUE["propeller"][p[key]].get("off_the_shelf")
        assert k4tq.CATALOGUE["propeller"][p["prop_rear"]]["folding"]         # the rear pair stops and folds in cruise
        e = kf.evaluate(sz.vehicle(d))
        assert e["hover ok"] and e["R3 closes"] and e["root ok"]
        assert e["still-air range [km]"] >= kh.RANGE_KEEP * r_g3
        assert e["take-off mass [kg]"] < 24.0 and e["stall [m/s]"] < 18.2
