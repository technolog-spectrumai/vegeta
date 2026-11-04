"""Hall thrusters: the 0-D chain against the SPT-100 and the BHT-200, the scaling, the magnetised checks, the
circuit by reluctance and solved on FiPy."""
import math

import pytest

from vegeta.fulgur.hall import (BHT200, SPT100, CircuitSection, HallPoint, design_coils, magnetised, point_for_power,
                                reluctance_circuit, scale_channel)


def test_chain_against_the_references():
    spt = HallPoint(SPT100["v_d"], SPT100["mdot_a"])
    assert spt.thrust == pytest.approx(SPT100["thrust"], rel=0.15) and spt.isp == pytest.approx(SPT100["isp"], rel=0.15)
    assert spt.power == pytest.approx(SPT100["power"], rel=0.15)
    bht = HallPoint(BHT200["v_d"], BHT200["mdot_a"])
    assert bht.thrust == pytest.approx(BHT200["thrust"], rel=0.15) and bht.isp == pytest.approx(BHT200["isp"], rel=0.15)
    assert 0.4 < spt.efficiency < 0.6 and 50 < spt.thrust_per_watt * 1e6 < 80          # mN/kW of the class


def test_gyrus_300_point_and_krypton():
    g = point_for_power(300.0, 280.0)
    assert g.power == pytest.approx(300.0) and 15e-3 < g.thrust < 25e-3 and 1300 < g.isp < 1700
    k = point_for_power(300.0, 280.0, "krypton")
    assert k.isp > g.isp and k.thrust < g.thrust                                        # lighter ions: faster, less thrust per watt


def test_scaling_from_the_spt100():
    ch = scale_channel(300.0, 280.0)
    assert 0.035 < ch.d < 0.05 and 0.005 < ch.h < 0.01 and ch.b_exit > SPT100["channel"].b_exit
    assert scale_channel(SPT100["power"], SPT100["v_d"]).d == pytest.approx(SPT100["channel"].d)
    m = magnetised(ch, 280.0)
    assert m["electrons_magnetised"] and m["ions_unmagnetised"] and m["hall_ok"]


def test_reluctance_and_coils():
    ch = scale_channel(300.0, 280.0)
    rc = reluctance_circuit(ch)
    assert 300 < rc["ampere_turns"] < 1500 and rc["iron_share"] < 0.05
    coils = design_coils(rc["ampere_turns"], inner_radius=0.011, outer_radius=0.03)
    assert coils["current_density_A_mm2"] < 3.0                                         # vacuum: no convection
    assert coils["power_W"] < 10.0


def test_magnetostatics_hits_the_target():
    pytest.importorskip("fipy")
    from vegeta.fulgur.hall import tune_ampere_turns
    ch = scale_channel(300.0, 280.0)
    sec = CircuitSection(ch)
    sol = tune_ampere_turns(sec, ch.b_exit, 500.0, lc_fine=0.6e-3, lc_far=6e-3)
    assert sol["B_r_exit"] == pytest.approx(ch.b_exit, rel=0.10)                        # the exit field on target
    assert sol["B_r_anode"] < 0.2 * sol["B_r_exit"]                                     # and little at the anode
    assert 150 < sol["ampere_turns"] < 1500
