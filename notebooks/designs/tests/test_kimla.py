"""Notebook 48: KIMLA's machines as hypothetical welding platforms — the scenarios carry their own optics, the welds are
capped and flagged, the throughput counts its overheads, the stray beam meets referenced limits without a blink, and
no barrier passes without evidence."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pytest.importorskip("vegeta.hyperion")

import kimla as km  # noqa: E402


def test_specs_and_scenarios_are_explicit():
    assert set(km.SCENARIOS) == {"3 kW", "6 kW", "12 kW", "20 kW", "30 kW"}
    assert all("not retrieved" in s["status"] for s in km.SPECS["sources"]["kimla_documents_not_retrieved"]) and "conflicts" in km.SPECS
    assert "hypothetical" in km.SPECS["hypothetical_welding_head"]
    for k, s in km.SCENARIOS.items():
        assert s.bpp_low < s.bpp_nominal < s.bpp_high and "not an orderable" in s.to_dict()["status"]
    o3, o30 = km.optics(km.SCENARIOS["3 kW"]), km.optics(km.SCENARIOS["30 kW"])
    assert o30["spot_radius"] > o3["spot_radius"] and o30["m2"] > o3["m2"] > 5 and o3["regime"].startswith("multimode")   # no single-mode optics carried over
    assert o3["spot_interval"][0] < o3["spot_radius"] < o3["spot_interval"][1]


def test_weld_cases_capped_and_flagged():
    s = km.SCENARIOS["12 kW"]
    c = km.weld_case(s, "carbon steel", 1e-3, "butt")
    assert c["regime"] == "keyhole" and c["feasible"] and c["capped"] and c["cap"] in ("humping onset", "gantry feed")
    assert c["speed"] == pytest.approx(min(km._a("max_feed_m_per_s").nominal, km._a("humping_m_per_s").nominal)) and c["v_model"] > c["speed"]
    assert c["depth_at_speed"] > c["depth_needed"] and "capped" in c["limitation"] and not c["validated"]
    f = km.weld_case(s, "carbon steel", 1e-3, "fillet")
    assert f["depth_needed"] == pytest.approx(1.2 * math.sqrt(2) * 1e-3)
    slow = km.weld_case(km.SCENARIOS["3 kW"], "carbon steel", 3e-3, "butt")
    assert not slow["capped"] and slow["speed"] == pytest.approx(slow["v_model"]) and slow["speed_interval"][0] < slow["speed"] < slow["speed_interval"][1]
    al = km.weld_case(km.SCENARIOS["3 kW"], "aluminium", 2e-3, "butt")
    assert al["cooling_rate_K_per_s"] > 0 and al["haz_each_side"] > 0 and al["pool_width"] > 0


def test_weld_matrix_shape_and_table():
    cases = km.weld_matrix(scenarios={k: km.SCENARIOS[k] for k in ("3 kW", "30 kW")})
    assert len(cases) == 2 * 3 * 3 * 2
    tab = km.weld_table(cases)
    assert set(tab["joint"]) == {"butt", "fillet"} and tab["v [m/min]"].notna().all() and (tab["v [m/min]"] <= tab["v model [m/min]"] + 1e-9).all()
    assert tab["limitation"].notna().all()


def test_throughput_and_economics():
    c = km.weld_case(km.SCENARIOS["6 kW"], "carbon steel", 2e-3, "butt")
    tp = km.throughput(c, km.JOBS["assembly_1m"])
    assert tp["arc_on_s"] == pytest.approx(1.0 / c["speed"]) and tp["practical_s"] > tp["arc_on_s"] and 0 < tp["arc_on_share"] < 1
    half = km.throughput(c, km.JOBS["assembly_1m"], {k: (v / 2 if isinstance(v, float) else v) for k, v in km.OVERHEADS.items()})
    assert half["parts_per_h_practical"] > tp["parts_per_h_practical"]
    cx = km.capex(km.SCENARIOS["30 kW"])
    assert cx["nominal"]["source"] == 25000 * 30 and cx["low"]["total"] < cx["nominal"]["total"] < cx["high"]["total"]
    ox = km.opex_per_hour(km.SCENARIOS["6 kW"])
    assert ox["low"]["total"] < ox["nominal"]["total"] < ox["high"]["total"] and ox["nominal"]["kW_drawn"] > 6000 / 0.35 / 1e3
    econ = km.economics_table([c])
    assert len(econ) == 2 and (econ["€/m practical"] > econ["€/m arc-on"]).all() and (econ["parts/h practical"] < econ["parts/h arc-on"]).all()


def test_stray_cases_limits_and_no_blink():
    df = km.stray_cases(scenarios={"6 kW": km.SCENARIOS["6 kW"]})
    assert len(df) == (1 + 3) * 4 * len(km.EXPOSURES) and (~df["aversion_assumed"]).all()
    div = df[(df["kind"] == "diverging") & (df["exposure"] == "T2 (100 s)")].sort_values("distance_m")
    assert (np.diff(div["irradiance_W_per_m2"]) < 0).all() and div["mpe_eye_W_per_m2"].iloc[0] == 50.0 and div["mpe_skin_W_per_m2"].iloc[0] == 10000.0
    assert div["over_mpe_eye"].iloc[0] > 1 and div["od_required_eye"].iloc[0] > 0 and div["nohd_m"].iloc[0] > 20
    refl = df[(df["kind"] == "reflected") & (df["distance_m"] == 20.0) & (df["exposure"] == "T2 (100 s)")]
    assert refl[refl["material"] == "aluminium"]["power_W"].iloc[0] > refl[refl["material"] == "carbon steel"]["power_W"].iloc[0]
    assert all(lo <= E <= hi for lo, E, hi in zip(refl["irradiance_interval_W_per_m2"].str[0], refl["irradiance_W_per_m2"], refl["irradiance_interval_W_per_m2"].str[1]))
    fault = km.collimated_fault(scenarios={"6 kW": km.SCENARIOS["6 kW"]})
    assert len(fault) == 4 and (fault["over_mpe_eye"] > div["over_mpe_eye"].max()).all() and fault["note"].str.contains("fault").all()


def test_barriers_never_pass():
    df = km.barrier_assessment(scenarios={"3 kW": km.SCENARIOS["3 kW"]})
    assert len(df) == 4 * 4 and (df["verdict"] != "PASS").all() and df["interlock_required"].all()
    assert not any("burn" in c or "failure_time" in c for c in df.columns)


def test_sensitivity_and_record():
    sens = km.capacity_sensitivity()
    assert {"melting_efficiency", "overheads ×2"} <= set(sens["assumption"]) and (sens[sens["assumption"] == "overheads ×2"]["change [%]"] < 0).all()
    s = {"3 kW": km.SCENARIOS["3 kW"]}
    cases = km.weld_matrix(scenarios=s)
    rec = km.record({k: km.optics(v) for k, v in km.SCENARIOS.items()}, cases, km.economics_table(cases), km.stray_cases(scenarios=s), km.collimated_fault(scenarios=s),
                    km.barrier_assessment(scenarios=s), sens, km.OVERHEADS)
    assert set(rec) >= {"meta", "scenarios", "welding", "throughput", "economics", "safety", "sensitivity", "validation_open"} and rec["validation_open"]
    import json
    json.dumps(rec, default=km.sc._json_default)


def test_shield_endurance_screening_and_fire():
    rows = km.shield_endurance(scenarios={"6 kW": km.SCENARIOS["6 kW"]}, distances=(20.0, 200.0), run=False)
    assert len(rows) == 2 * 4 * 2 and {r["shield"] for r in rows} == {"aluminium 1 mm", "aluminium 3 mm", "black polycarbonate 3 mm", "mild steel 3 mm"}
    div = [r for r in rows if r["kind"] == "diverging" and r["distance_m"] == 200.0]
    assert all(r["holds_s"] == math.inf and r["T3"] for r in div)                                   # metres wide past the focus at 200 m: every sheet holds by a bound
    assert all(r["holds_s"] != r["holds_s"] or r["holds_s"] == math.inf for r in rows)               # without the recorded run nothing is a finite time
    col = [r for r in rows if r["kind"] == "collimated" and r["material"] == "black polycarbonate" and r["distance_m"] == 20.0][0]
    assert col["decision"].startswith("undecided") and math.isnan(col["holds_s"]) and col["t_burn_through_lower_bound_s"] > 0 and not col["simulated"]
    assert col["combustible"] and col["fire_risk"].startswith("high") and 0 < col["t_ignition_bound_s"] < 10
    metal = [r for r in rows if r["material"] == "aluminium"][0]
    assert not metal["combustible"] and metal["t_ignition_bound_s"] == math.inf
    assert all("not protection evidence" in r["note"] for r in rows)
    assert km.shield_endurance_table(rows).shape == (8, 2)
    assert km.load_endurance(Path("/nonexistent/kimla_endurance.json")) is None


def test_mock_endurance_record_exercises_the_read_back_path(tmp_path):
    import json
    rec = km.mock_endurance_record(scenarios={"3 kW": km.SCENARIOS["3 kW"], "30 kW": km.SCENARIOS["30 kW"]})
    assert rec["mock"] and rec["meta"]["made"] == "MOCK" and len(rec["rows"]) == 2 * 2 * 4 * 4
    assert all(r["holds_s"] == r["holds_s"] for r in rec["rows"])                                   # nothing undecided after the mock
    assert all("MOCK" in r["source"] for r in rec["rows"] if r["decision"].startswith("undecided")) and not any(r["simulated"] for r in rec["rows"])
    tab = km.shield_endurance_table(rec["rows"])
    assert tab.shape == (2 * 2 * 4, 4) and tab.notna().all().all()
    pc = [r for r in rec["rows"] if r["material"] == "black polycarbonate" and r["kind"] == "collimated" and r["scenario"] == "30 kW" and r["distance_m"] == 20.0][0]
    assert pc["holds_s"] == pytest.approx(3 * pc["t_burn_through_lower_bound_s"]) and not pc["T2"] and pc["combustible"]
    p = tmp_path / "kimla_endurance.json"                                                           # the same serialisation and read-back as the real record
    p.write_text(json.dumps({**rec, "rows": [{kk: (None if isinstance(vv, float) and math.isnan(vv) else ("inf" if vv == math.inf else vv)) for kk, vv in r.items()} for r in rec["rows"]]}, indent=1))
    back = km.load_endurance(p)
    assert back["mock"] and len(back["rows"]) == len(rec["rows"]) and all(r["holds_s"] in (math.inf,) or r["holds_s"] > 0 for r in back["rows"])
    assert not km.ENDURANCE_PATH.exists() or json.loads(km.ENDURANCE_PATH.read_text()).get("mock") is not True   # the mock never lands on the data path
