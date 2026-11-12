"""K4TQ-A2 / F2 flight envelope: climb, descent and manoeuvre stay physical (fast: one design, a short descent)."""
import k4tq_envelope as ke
import k4tq_sizing as sz


def test_envelope_physics():
    veh = sz.vehicle("F2")
    cl = ke.climb(veh)
    assert cl["rate of climb [m/s]"].max() > 3.0
    gl = ke.glide(veh)
    assert 6.0 < gl["glide ratio"].max() < 16.0 and gl["sink [m/s]"].min() > 0.5
    vd = ke.vertical_descent(veh, gl)
    assert vd["safe vertical descent [m/s]"] < vd["induced velocity [m/s]"]
    r = ke.roll(veh)
    assert 50.0 < r["steady roll rate [deg/s]"] < 600.0


def test_powered_descent_continuation():
    veh = sz.vehicle("F2")
    d = ke.powered_descent(veh, speeds=(12.0, 26.0), step=2.0, gamma_min=-10.0)
    steep = ke.steepest_powered(d)
    assert steep[12.0] <= -10.0                     # prop-borne at 12 m/s: the tilt is the brake
    assert -8.0 < steep[26.0] < 0.0                 # wing-borne: only as steep as the zero-thrust glide
    ok = d[d["feasible"]]
    assert (ok["thrust total [N]"] >= 0.0).all()
