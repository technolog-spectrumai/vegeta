"""The mini raptors (notebook 49): the layouts' disc clearances, the mass budget, the performance model's sanity, the
quad baseline, the targets table, the CAD parts (quick: no export)."""
import math

import pytest

import mini_raptor as mr


@pytest.fixture(scope="module")
def PERF():
    return {v: mr.performance(v) for v in mr.VARIANTS}


@pytest.fixture(scope="module")
def BASE():
    return mr.quad()


def test_layouts_clear_the_discs_and_keep_their_briefs():
    for v in mr.VARIANTS:
        L = mr.layout(v)
        assert mr.checks(L) == [], (v, mr.checks(L))
        assert L["disc_gap_vertical"] >= mr.DISC_CLEARANCE and L["disc_gap_span"] >= mr.DISC_CLEARANCE
    assert mr.layout("Gavilan")["span"] <= 0.31 and mr.layout("Gavilan")["x_pod_nose"] > 0           # the pod starts behind the discs
    assert mr.layout("Alcotan")["layout"] == "cruciform" and mr.layout("Apus Mini")["span_over_gate"] <= 0.34


def test_mass_budget_is_explicit_and_in_the_band(PERF):
    for v, p in PERF.items():
        b = mr.budget(p["L"])
        assert set(b["group"]) == {"propulsion", "structure", "energy", "systems"}
        assert 0.45 < p["summary"]["mass [kg]"] < 0.62, (v, p["summary"]["mass [kg]"])
        assert abs(b["mass_g"].sum() / 1e3 - p["summary"]["mass [kg]"]) < 1e-3
    assert 0.45 < sum(r["mass_kg"] for r in mr.quad_mass_table()) < 0.55


def test_performance_sanity(PERF, BASE):
    q = BASE["summary"]
    a, c, g = (PERF[k]["summary"] for k in ("Apus Mini", "Alcotan", "Gavilan"))
    assert a["Wh/km at 20 m/s"] < 0.5 * q["Wh/km at 20 m/s"]                     # the economy brief
    assert c["sustained top speed [km/h]"] > a["sustained top speed [km/h]"] > q["sustained top speed [km/h]"]
    assert c["sustained top speed [km/h]"] <= c["top speed [km/h]"] and c["full-throttle endurance at top speed [s]"] < 120
    assert g["span [m]"] < a["span [m]"] and g["stall speed [m/s]"] < 12
    for s in (a, c, g):
        assert 5 < s["hover thrust / weight"] < 15 and s["thrust / weight at 1.2 Vs"] > 2
        assert s["brake 20→4 m/s distance [m]"] < 10 and s["hover pitch-over 90° [s]"] < 0.5
        assert s["turn radius at 20 m/s (banked) [m]"] > 0
    assert 30 < q["top speed [m/s]"] < 60 and 2 < q["best range [km]"] < 15                   # the racer's band, roughly
    t = mr.targets(PERF, BASE)
    assert {"target", "value", "met"} <= set(t.columns) and len(t) >= 9


def test_drag_buildup_and_induced(PERF):
    for v, p in PERF.items():
        d = mr.drag_buildup(p["L"])
        assert 0.015 < d["Cd0"] < 0.08 and "wings" in d["parts_m2"]
        k = mr.induced_factor(p["L"])
        assert 0.05 < k["k"] < 0.3


def test_cad_parts_build_valid():
    for v in mr.VARIANTS:
        prt = mr.parts(mr.cad_params(v))
        assert all(s.val().isValid() for s, _ in prt.values())
        assert {"pod", "wings", "motors", "propeller discs", "battery", "flight stack", "camera"} <= set(prt)
        assert ("posts" in prt) == (mr.layout(v)["layout"] == "biplane") and ("fins" in prt) == (mr.layout(v)["layout"] == "cruciform")
