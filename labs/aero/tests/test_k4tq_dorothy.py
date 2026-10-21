"""K4TQ-D1 "Dorothy": the powered fold's mass reaches the budget, the tuck trades gust response for control and stops
where the propellers do, folding adds drag (it cannot go faster), the narrow hover keeps the quad's arms, the fold drive
holds the tuck's hinge moment. Fast (no CAD Booleans; the folded CAD's clash check is in the slow set)."""
import pytest

import k4tq
import k4tq_sizing as sz


@pytest.fixture(scope="module")
def D1():
    return sz.vehicle("D1")


def test_the_fold_drive_costs_mass(D1):
    F1 = sz.vehicle("F1")
    fold = sum(v["mass_kg"] for v in k4tq.CATALOGUE["fold_drive"].values())
    assert 0.5 < D1["mass"]["mass_kg"] - F1["mass"]["mass_kg"] < 0.9 and sum(1 for i in D1["items"] if i["item"].startswith("fold drive")) == 4
    assert abs(sum(i["mass_kg"] for i in D1["items"] if i["item"].startswith("fold drive")) - 4 * fold) < 1e-12


def test_smoke_transit_trades_gusts_for_control(D1):
    sm = sz.smoke_transit(D1)
    assert list(sm["gust bump (× unfolded)"]) == sorted(sm["gust bump (× unfolded)"], reverse=True)
    assert sm.loc[30.0, "within the propellers' speed"] and not sm.loc[60.0, "within the propellers' speed"]
    assert 0.6 < sm.loc[30.0, "roll authority (× unfolded)"] < 0.7 and sm.loc[30.0, "yaw authority (× unfolded)"] > 0.85


def test_folding_adds_drag(D1):
    fl = sz.fold_level(D1, folds=(0.0, 30.0), speeds=(26.0, 28.0))
    for V in (26.0, 28.0):
        assert fl.loc[(30.0, V), "drag [N]"] > fl.loc[(0.0, V), "drag [N]"]


def test_hover_fold_and_drive(D1):
    hf = sz.hover_fold(D1)
    assert hf["width over the discs [m]"].is_monotonic_decreasing and (hf["pitch arm (× unfolded)"] == 1.0).all()
    assert sz.fold_drive_loads(D1)["reserve"] > 1.0


@pytest.mark.slow
def test_folds_clear_in_cad(D1):
    pytest.importorskip("cadquery")
    import k4tq_cad
    for t, f in ((0.0, 30.0), (90.0, 30.0)):
        assert k4tq_cad.interferences(k4tq_cad.assembly(D1["p"], t, f, items=D1["items"], discs=True)) == []
