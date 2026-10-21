"""K4TQ-G "Ginna" and K4TQ-B "Bertha": the scaling keeps Freddy unchanged, scales every length, and the sized designs
stay under their mass limits with the hover rules, the R3 mission and Freddy's stall speed."""
import k4tq
import k4tq_scaled as ks
import k4tq_sizing as sz


def test_scaling_lengths_and_freddy_unchanged():
    p1, pB = k4tq.params("F1"), k4tq.params("B")
    assert p1.get("scale", 1.0) == 1.0 and p1["battery"] == "6S3P-P45B"
    s = pB["scale"]
    for k in ("fuselage_length", "span", "motor_y", "root_chord_front"):
        assert abs(pB[k] - s * p1[k]) < 1e-9
    assert pB["cargo_box"][0] == s * p1["cargo_box"][0]


def test_scaled_designs_within_limits():
    for d in ("G", "B"):
        v = sz.vehicle(d)
        assert v["mass"]["mass_kg"] <= ks.LIMITS[d] + 0.05
        ok, h = ks.hover_ok(v)
        assert ok
        assert ks.closes(v)[0]
        assert sz.stall_speed(v) <= ks.STALL_MAX + 0.1
        r = sz.root_loads(v)
        assert r["actuator reserve"] >= 1.0 and r["bearing static reserve"] >= 1.0
    assert k4tq.params("B")["payload_kg"] > k4tq.params("G")["payload_kg"] > k4tq.params("F1")["payload_kg"]
