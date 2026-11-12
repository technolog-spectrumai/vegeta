"""The KPI table's physics: K4TQ out-ranges the 15-inch FPV quad, the quad out-hovers and out-sprints it, payload slows the
hover clock, and nothing reaches 100 km/h straight up."""
import math

import k4tq_kpi as kk
import k4tq_sizing as sz


def test_kpi_rows_are_physical():
    k = kk._k4_row("F3")
    f = kk._fpv_row(1.0)
    assert k["R3 radius with the payload [km]"] > 2 * f["R3 radius with the payload [km]"]
    assert f["hover to an empty pack, with payload [min]"] > k["hover to an empty pack, with payload [min]"]
    assert k["hover to an empty pack, without payload [min]"] > k["hover to an empty pack, with payload [min]"]
    assert k["top speed (3 s), with payload [km/h]"] > f["top speed (3 s), with payload [km/h]"]
    t_k, _ = kk._accel_k4(sz.vehicle("F3"), up=False)
    t_f, _ = kk._accel_fpv(1.0, up=False)
    assert math.isfinite(t_k) and math.isfinite(t_f) and t_f < t_k < 10.0
    assert not math.isfinite(kk._accel_k4(sz.vehicle("F3"), up=True)[0])
