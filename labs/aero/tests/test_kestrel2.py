"""Kestrel-4Q-2 against 4Q-1 and the case for 4Q-3 (kestrel2, notebook 42): the aircraft rebuilt from notebook 40's
recipe, the back-transition remedies (reverse thrust closes the balloon; a descent into it does not), the level
deceleration, the performance and pursuit tables, the 4Q-3 proposal.

Run: cd /home/user/vegeta && python3 -m pytest -q designs/tests/test_kestrel2.py
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "designs"))

import kestrel2 as k2
import kestrel_transition as kt


@pytest.fixture(scope="module")
def V():
    return k2.vehicles(which=("4Q-1", "4Q-2", "4Q-3"))


def test_vehicles_follow_notebook_38(V):
    assert V["4Q-1"].pol.cd0 > 2 * V["4Q-2"].pol.cd0 and V["4Q-2"].mass < V["4Q-1"].mass
    assert V["4Q-3"].elevon_y_in == pytest.approx(0.25) and V["4Q-2"].elevon_y_in == pytest.approx(0.275) and V["4Q-1"].elevon_y_in is None
    assert 0.3 < V["4Q-2"].elevon_tau < 0.6


def test_remedies_reverse_thrust_closes_the_balloon(V):
    v2 = V["4Q-2"]
    base = kt.simulate(v2, "back", t_tr=3.0).metrics
    rev = kt.simulate(v2, "back", t_tr=3.0, reverse_N=k2.REVERSE_N).metrics
    slow = kt.simulate(v2, "back", t_tr=3.0, V_entry=13.0).metrics
    brake = kt.simulate(v2, "back", t_tr=3.0, airbrake_deg=20.0).metrics
    assert base["height gain [m]"] > 10.0 and not base["pass"]
    assert rev["height gain [m]"] < 1.0 and rev["pass"] and rev["energy [Wh]"] > base["energy [Wh]"]
    assert slow["pass"] and brake["height gain [m]"] < base["height gain [m]"]
    desc = kt.simulate(v2, "back", t_tr=3.0, descent_entry=2.0).metrics
    assert desc["height loss [m]"] < 1.0                      # measured against the descending reference
    df, eps = k2.remedy_matrix(v2, remedies={"as is": {}, "reverse": {"reverse_N": 3.0}}, durations=(3.0,), headwinds=(0.0,))
    ver = k2.remedy_verdict(df)
    assert ver.loc["reverse", "passes"] == "1 of 1" and ver.loc["as is", "passes"] == "0 of 1" and np.isfinite(ver.loc["reverse", "hover reached, 3 s [s]"])
    assert ("reverse", 3.0, 0.0) in eps and eps[("reverse", 3.0, 0.0)].log["T_lo"].min() < 0


def test_decelerate_level(V):
    v2 = V["4Q-2"]
    idle, brake, rev = k2.decelerate_level(v2), k2.decelerate_level(v2, airbrake_deg=20.0), k2.decelerate_level(v2, reverse_N=3.0)
    assert idle["reached"] and idle["time [s]"] > 5 * brake["time [s]"] > 3 * rev["time [s]"] and rev["energy [Wh]"] < 0.1
    v1 = V["4Q-1"]
    assert k2.decelerate_level(v1)["time [s]"] < idle["time [s]"]           # the draggy aircraft slows by itself


def test_performance_and_pursuit(V):
    P = {k: k2.performance(v) for k, v in V.items()}
    s1, s2 = P["4Q-1"]["summary"], P["4Q-2"]["summary"]
    assert 28 < s1["V max, thrust = drag [m/s]"] < 32 and s2["V max, thrust = drag [m/s]"] > s1["V max, thrust = drag [m/s]"]
    assert s2["best endurance [min]"] > 1.3 * s1["best endurance [min]"] and s2["terminal dive (idle) [m/s]"] > s1["terminal dive (idle) [m/s]"]
    assert s2["turn radius at the corner [m]"] < s1["turn radius at the corner [m]"] and 1 < s1["sustained load factor at 16 m/s"] < 4
    assert s2["time to climb 100 m [s]"] < s1["time to climb 100 m [s]"] and math.isfinite(s1["accelerate 13 → 25 m/s [s]"])
    t = k2.pursuit(P)
    assert list(t.columns) == ["4Q-1", "4Q-2", "4Q-3"] and t.loc["corner speed [m/s]", "4Q-1"] == pytest.approx(2 * s1["stall speed [m/s]"])
    assert float(t.loc["steady roll rate at 16 m/s [deg/s]", "4Q-3"]) > float(t.loc["steady roll rate at 16 m/s [deg/s]", "4Q-2"])
    prop = k2.propose_4q3()
    assert {"airframe", "drives", "elevons", "autopilot", "what it buys", "open"} <= set(prop)


def test_kestrel_fast_compares_two_candidates():
    """The fast comparator: Zephyr + Boreas points + beam rules, no solver; a bigger wing lowers the stall speed and the
    wing loading, the structure rules give finite reserves, the targets table counts."""
    import kestrel_fast as kfa
    base = kfa.Candidate("base")
    big = kfa.Candidate("big", span=1.2, chord=0.16, gap=0.30, prop="10x7")
    df = kfa.compare([base, big])
    assert df.loc["big", "stall speed [m/s]"] < df.loc["base", "stall speed [m/s]"]
    assert df.loc["big", "wing loading [N/m²]"] < df.loc["base", "wing loading [N/m²]"]
    assert df.loc["big", "V max, thrust-limited [m/s]"] > df.loc["base", "V max, thrust-limited [m/s]"]       # the 10x7's pitch speed
    assert (df["spar reserve (ultimate)"] > 1).all() and (df["post buckling margin"] > 1).all() and (df["layout rules broken"] == 0).all()
    v = kfa.verdicts(df)
    assert "targets met" in v.columns and int(v.loc["base", "targets met"]) >= 8
    assert abs(kfa.ratios(df, "base").loc["base"] - 1).max() < 1e-12


def test_interference_measure_and_monoplane_equivalents():
    """The analytic wing-interference measure: the factor is below 1 and rises with the gap, Prandtl's σ falls with it,
    the pair beats the equal-span equal-area monoplane on induced drag, and the equivalents are consistent."""
    import kestrel_flight as kf
    r = kf.interference()
    d, sw, eq = r["design"], r["sweep"], r["equivalents"]
    assert 0.8 < d["lift interference factor"] < 0.95 and 0.3 < d["Prandtl σ"] < 0.5
    assert 0.6 < d["induced drag vs monoplane of the same span and area, lattice"] < 0.8
    assert sw["lift interference factor"].is_monotonic_increasing and sw["Prandtl σ"].is_monotonic_decreasing
    assert abs(kf.prandtl_sigma(0.0) - 1 / 1.05) < 1e-9
    assert eq.loc["the pair as built", "area [m²]"] == 2 * eq.loc["monoplane, same span and chord (one wing)", "area [m²]"]
    assert eq.loc["monoplane with the pair's induced drag, same area", "span [m]"] > eq.loc["the pair as built", "span [m]"]
    assert sw["design"].sum() == 1


def test_other_machines_scaled_to_kestrel_size():
    """kestrel_compare: Kestrel three ways and the design files on disk, all at one mass and speed; the sentences."""
    import kestrel_compare as kcm
    import kestrel_campaigns as kcp
    ms = kcm.machines()
    assert {"falco", "nisus_plus", "merlin", "peregrine"} <= set(ms)
    df = kcm.compare(kcp.params("4Q-2"))
    assert len(df) == 3 + len(ms) and df.index[0].startswith("Kestrel biplane")
    bi, m2c, one = df.iloc[0], df.iloc[1], df.iloc[2]
    assert abs(bi["area scaled [m²]"] - m2c["area scaled [m²]"]) < 1e-9 and abs(one["area scaled [m²]"] * 2 - bi["area scaled [m²]"]) < 1e-9
    assert bi["k = CDi/CL²"] < m2c["k = CDi/CL²"]                              # the biplane's induced-drag edge over the thick monoplane
    assert one["stall speed [m/s]"] > bi["stall speed [m/s]"] * 1.3            # the second wing is the stall speed
    assert (df["span scaled [m]"].iloc[3:] > 1.1).all()                         # the monoplanes buy their L/D with span
    v = kcm.verdict(df)
    assert len(v) == 4 and "span" in v[2]
