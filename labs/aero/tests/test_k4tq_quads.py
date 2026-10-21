"""K4TQ against the quadcopters: same pack, the K4TQ wins the R3 radius; the quads win the hover; the same-mass quad
carries the bigger pack within K4TQ's take-off mass."""
import k4tq_quads as kq
import k4tq_sizing as sz


def test_quads_against_cathy():
    k4 = sz.vehicle("A1")
    r_k4 = kq.k4_radius("A1")
    q17 = kq.quad("Q-17", "17x5.8", k4, in_footprint=True)
    assert abs(q17["plan box [m]"][0] - k4["L"]["width_overall"]) < 1e-9
    assert q17["T/W"] >= kq.TW_MIN
    assert r_k4 > q17["mission radius [km]"]
    assert q17["hover minutes"] > sz.hover(k4)["hover minutes on the usable pack"]
    qh = kq.quad_same_mass("Q-17 heavy", "17x5.8", k4)
    assert qh["mass [kg]"] <= k4["mass"]["mass_kg"] and qh["pack"] != k4["p"]["battery"]
    assert qh["mission radius [km]"] > q17["mission radius [km]"]
