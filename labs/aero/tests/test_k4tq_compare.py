"""K4TQ-A1 / A2 / F1 / F2 compared: the per-wing sections reach the stall rule and the CAD, the Jetson reaches the power
budget, the spar rule tells A from F (fast: no corridor, no extended mission)."""
import k4tq
import k4tq_compare as kc


def test_four_designs_compare():
    ev = {v: kc.evaluate(v, corridor=False, extended=False) for v in kc.DESIGNS}
    df = kc.table(ev)
    a1, a2, f, f2 = (df.loc[kc.name(v)] for v in kc.DESIGNS)
    assert float(f["CL max (aircraft)"]) > float(a1["CL max (aircraft)"])            # the 4415 front wing stalls later
    assert not bool(a1["Ø16 spar fits"]) and bool(f["Ø16 spar fits"])                # 2412 on the 120 mm tip: 14 mm < 18
    assert float(a2["electronics [W]"]) - float(a1["electronics [W]"]) > 20.0        # the Jetson and the perception
    assert float(f2["take-off mass [kg]"]) > float(f["take-off mass [kg]"])
    assert all(ev[v]["veh"]["p"]["cad_mass_status"] == "CAD" for v in kc.DESIGNS)
    assert k4tq.section_of(ev["F1"]["veh"]["p"], "front") == "NACA 4415" and k4tq.section_of(ev["F1"]["veh"]["p"], "rear") == "NACA 2415"
