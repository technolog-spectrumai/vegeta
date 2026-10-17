"""Svarog III: the published limits stay as published, the optics and the energy balance close where the sheet lets them,
the welds and the shielding matrix have their counts, and no verdict says PASS without evidence."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pytest.importorskip("vegeta.hyperion")

import svarog_commercial as sc  # noqa: E402


def test_specs_are_verbatim_and_unknowns_preserved():
    assert set(sc.LASERS) == {"YLS-SM", "FL2000C", "FL2000CSM", "WFL-3000"}
    y, c, csm, w = (sc.LASERS[k] for k in ("YLS-SM", "FL2000C", "FL2000CSM", "WFL-3000"))
    assert y.m2_max == 1.1 and y.bpp_max_mm_mrad is None and y.electrical_max_kW is None and y.connected_kVA is None and y.cooling_kW is None
    assert c.m2_max is None and c.bpp_max_mm_mrad == 2.5 and c.electrical_max_kW == 6.2 and c.cooling_kW == 4.4 and c.connected_kVA == 6.4
    assert csm.mode == "single" and csm.bpp_max_mm_mrad == 0.4 and csm.m2_max is None
    assert w.power_W == 3000 and w.modulation_kHz == 20 and w.stability_pct == 1.0 and w.electrical_max_kW is None
    assert all(s.wavelength == pytest.approx(1070e-9) for s in sc.LASERS.values())
    assert "limits" in y.source


def test_optics_m2_bpp_consistency_and_regimes():
    o = {k: sc.optics(s) for k, s in sc.LASERS.items()}
    assert sc.bpp_from_m2(1.0) == pytest.approx(1070e-9 / math.pi * 1e6)
    assert sc.m2_from_bpp(sc.bpp_from_m2(3.3)) == pytest.approx(3.3)
    assert o["FL2000CSM"]["m2"] == pytest.approx(math.pi * 0.4e-6 / 1070e-9) and o["FL2000CSM"]["regime"].startswith("Gaussian")
    assert o["FL2000C"]["m2"] > 5 and o["FL2000C"]["regime"].startswith("multimode") and o["WFL-3000"]["regime"].startswith("multimode")
    assert all(v["m2_bpp_consistent"] and v["mode_matches_m2"] for v in o.values())
    assert o["YLS-SM"]["spot_radius"] < o["FL2000C"]["spot_radius"] and o["YLS-SM"]["cone_half_angle"] < o["FL2000C"]["cone_half_angle"]
    assert o["FL2000C"]["spot_radius"] == pytest.approx(sc.HEAD.spot_radius(o["FL2000C"]["fibre"]))
    assert o["FL2000C"]["delivered_W"] < 2000 and o["FL2000C"]["delivered_interval_W"][0] < o["FL2000C"]["delivered_W"] < o["FL2000C"]["delivered_interval_W"][1]
    r = sc.beam_radius_at(o["YLS-SM"], [0.0, 100.0], "collimated")
    assert r[0] == pytest.approx(o["YLS-SM"]["collimated_radius"]) and r[1] > r[0]
    assert o["YLS-SM"]["power_band_W"] is None and o["FL2000C"]["power_band_W"] == (1980.0, 2020.0)
    with pytest.raises(ValueError):
        sc.optics(sc.Spec("x", "a", "b", 1000.0, 1070e-9, "multi"))


def test_energy_balance_closes_where_published_and_bounds_elsewhere():
    c = sc.energy_balance(sc.LASERS["FL2000C"])
    assert c["electrical_W"]["high"] == 6200.0 and c["waste_heat_W"]["high"] == pytest.approx(4200.0) and c["cooling_consistent"]
    assert c["wall_plug"]["low"] == pytest.approx(2000 / 6200)
    y = sc.energy_balance(sc.LASERS["YLS-SM"])
    assert y["electrical_W"]["low"] < y["electrical_W"]["nominal"] < y["electrical_W"]["high"] and y["cooling_published_W"] is None and "no published" in y["cooling_note"]
    w = sc.energy_balance(sc.LASERS["WFL-3000"])
    assert w["electrical_W"]["nominal"] > y["electrical_W"]["nominal"] and w["total_electrical_W"]["high"] > w["electrical_W"]["high"]
    sens = sc.energy_sensitivity(sc.LASERS["WFL-3000"])
    assert len(sens) == 10 and sens["electrical_W"].max() > sens["electrical_W"].min()


def test_welding_cases_count_and_physics():
    cases = sc.welding_cases(sc.LASERS["FL2000C"])
    assert len(cases) == 6 and {(c["family"], c["thickness"]) for c in cases} == {(f, t) for f in ("steel", "aluminium") for t in sc.THICKNESSES}
    steel = [c for c in cases if c["family"] == "steel"]
    assert all(c["feasible"] for c in steel) and steel[0]["speed"] > steel[1]["speed"] > steel[2]["speed"]
    assert steel[0]["t85_s"] < steel[2]["t85_s"] and steel[0]["haz_each_side"] < steel[2]["haz_each_side"]
    assert all(not c["validated"] for c in cases) and steel[1]["speed_interval"][0] < steel[1]["speed"] < steel[1]["speed_interval"][1]
    tab = sc.welding_table(cases)
    assert len(tab) == 6 and "feasible (model)" in tab and tab["heat input [J/mm]"].notna().all()
    unc = sc.welding_uncertainty(cases)
    assert (unc["validated"] == "no").all()


def test_mpe_nohd_and_od():
    assert float(sc.mpe_eye_1070(10.0)) == 50.0 and float(sc.mpe_eye_1070(100.0)) == 50.0 and float(sc.mpe_eye_1070(3e4)) == 50.0
    assert float(sc.mpe_eye_1070(1.0)) == pytest.approx(90.0) and float(sc.mpe_eye_1070(0.1)) > float(sc.mpe_eye_1070(1.0))
    assert math.isnan(float(sc.mpe_eye_1070(1e-7)))
    assert float(sc.mpe_skin_1070(100.0)) == 10000.0
    o = sc.optics(sc.LASERS["WFL-3000"]); o2 = sc.optics(sc.LASERS["FL2000C"])
    assert sc.nohd(o) > sc.nohd(o2) > 1000.0                                                 # a collimated kilowatt: kilometres
    assert sc.nohd(o, kind="reflected") < sc.nohd(o)
    od = sc.od_required(1e6, 50.0)
    assert od["od_required"] == pytest.approx(math.log10(2e4)) and od["en207_lb_scale"] == 3 and od["en207_tested_at_W_per_m2"] == 1e6
    assert sc.od_required(10.0, 50.0)["od_required"] == 0.0


def test_shield_matrix_counts_and_verdict_rule():
    df = sc.shield_cases()
    assert len(df) == 60 and df["base"].all() and set(df["laser"]) == set(sc.LASERS) and set(df["distance_m"]) == set(sc.SHIELD_DISTANCES)
    full = sc.shield_cases(variants=True)
    assert len(full) == 180 and full["base"].sum() == 60
    assert not any(k for k in full.columns if "burn" in k or "failure_time" in k or "holds" in k)   # no destruction times anywhere
    assert (full["verdict"] != "PASS").all()                                                        # no evidence in this module → never PASS
    assert (full[full["variant"] == "clear"]["verdict"] == "FAIL").all()
    cert = full[full["variant"] == "certified laser-protective"]
    assert (cert["verdict"] == "INCOMPLETE").all() and cert["reason"].str.contains("EN 12254").all()
    assert (full["interlock_required"]).all() and (full["over_mpe_eye"] > 1).all()
    wood = full[full["family"] == "wood"]
    assert wood["fire_risk"].str.startswith(("high", "moderate", "low")).all() and (full[full["family"] != "wood"]["fire_risk"] == "n/a").all()
    assert wood[wood["variant"] == "fire-retardant"]["fire_note"].str.contains("INCOMPLETE").all()
    far = full[(full["laser"] == "YLS-SM") & (full["distance_m"] == 100.0) & (full["variant"] == "bare")].iloc[0]
    near = full[(full["laser"] == "YLS-SM") & (full["distance_m"] == 5.0) & (full["variant"] == "bare")].iloc[0]
    assert far["irradiance_W_per_cm2"] < near["irradiance_W_per_cm2"] and far["verdict"] == "INCOMPLETE"         # distance alone never gives PASS
    refl = sc.shield_cases(kinds=("collimated", "reflected"))
    assert len(refl) == 120 and (refl[refl["beam"] == "reflected"]["power_W"] < refl[refl["beam"] == "collimated"]["power_W"].max()).all()
    vm = sc.verdict_matrix(full)
    assert vm.shape == (36, 5)
    c = dict(full.iloc[0]); c["evidence"] = "certified"; c["seam"] = False
    assert sc.verdict(c)[0] in ("PASS", "FAIL")                                                     # evidence entered by hand can pass unless it transmits
    c["transmitted_fraction"] = 0.0
    assert sc.verdict(c)[0] == "PASS"
    c["seam"] = True
    assert sc.verdict(c)[0] == "FAIL"


def test_thermal_response_is_nondestructive_bounds():
    beam = sc.stray_beam(sc.LASERS["FL2000C"], 10.0)
    r = sc.thermal_response(sc.SHIELDS["aluminium"][0], beam)
    assert r["surface_rise_T2_bound_K"] > r["surface_rise_T1_bound_K"] > 0 and r["steady_rise_bound_K"] > 0 and set(r) >= {"bound_ratio", "cannot_reach_damage"}
    assert r["reflected_fraction"] == pytest.approx(1 - 0.12) and r["transmitted_fraction"] == 0.0
    leak = sc.seam_leak(beam)
    assert 0 < leak["seam_share"] <= 1 and leak["seam_power_W"] <= beam.power
    deg = sc.degraded_sensitivity(sc.LASERS["FL2000C"])
    assert len(deg) == 6 and (deg["verdict"] != "PASS").all()


def test_record_keys(tmp_path):
    optic = {k: sc.optics(s) for k, s in sc.LASERS.items()}
    energy = {k: sc.energy_balance(s) for k, s in sc.LASERS.items()}
    welds = {"FL2000C": sc.welding_cases(sc.LASERS["FL2000C"], optic["FL2000C"])}
    df = sc.shield_cases()
    rec = sc.record(sc.LASERS, optic, energy, welds, df)
    assert set(rec) >= {"meta", "specs", "optics", "energy", "welding", "shielding", "validation_open"} and rec["validation_open"]
    assert rec["specs"]["YLS-SM"]["cooling_kW"] is None and rec["shielding"]["base_count"] == 60
    import json
    json.dumps(rec, default=sc._json_default)
    paths = sc.write_csvs(tmp_path, {"shield": df})
    assert paths[0].exists()


def test_endurance_screening_without_transients(tmp_path):
    rows = sc.endurance(sc.LASERS["FL2000C"], distances=(5.0, 100.0), run=False)
    assert len(rows) == 3 * 3 * 2 and {r["family"] for r in rows} == {"polycarbonate", "steel", "aluminium"}
    al = [r for r in rows if r["family"] == "aluminium"]
    assert all(r["holds_s"] == math.inf and r["T3"] for r in al)                                      # the steady bound decides aluminium
    pc = [r for r in rows if r["family"] == "polycarbonate"]
    assert all(r["decision"] == "simulate" and math.isnan(r["holds_s"]) for r in pc)                  # undecided without the transient
    assert all("not protection evidence" in r["note"] for r in rows)
    assert sc.endurance_table(rows).shape == (9, 2)
    assert sc.load_endurance(tmp_path / "none.json") is None


def test_prices_are_user_supplied_and_the_per_keur_metrics():
    assert set(sc.PRICES["allowance_new_EUR"]) == set(sc.LASERS) and "unverified" in sc.PRICES["source"] and sc.PRICES["listings_USD"]["FL2000C"]["used"] is None
    assert sc.price_keur("FL2000C") == (25.0, 50.0)
    cases = sc.welding_cases(sc.LASERS["FL2000C"])
    spk = sc.speed_per_keur(cases)
    assert len(spk) == 6 and (spk["v per k€, at the low allowance [m/min/k€]"] > spk["v per k€, at the high allowance [m/min/k€]"]).all()
    rows = sc.endurance(sc.LASERS["FL2000C"], distances=(5.0, 100.0), run=False)
    for r in rows:                                                                              # stand in for the transients: PC holds 2 s per mm, steel 1 mm fails at 5 m
        if r["family"] == "polycarbonate":
            r["holds_s"] = 2.0 * r["thickness_mm"]
        if r["family"] == "steel" and math.isnan(r["holds_s"]):
            r["holds_s"] = 4.0
    sfe = sc.shield_for_escape(rows)
    assert len(sfe) == 3 * 2 * 3
    pc3 = sfe[(sfe["family"] == "polycarbonate") & (sfe["escape_s"] == 3.0)].iloc[0]
    assert pc3["shield_needed_mm"] == 2.0 and pc3["mm per k€ (low allowance)"] == pytest.approx(2.0 / 25.0)
    assert (sfe[sfe["family"] == "aluminium"]["shield_needed_mm"] == 1.0).all()
    pc5 = sfe[(sfe["family"] == "polycarbonate") & (sfe["escape_s"] == 5.0)]
    assert (pc5["shield_needed_mm"] == 3.0).all()
    epk = sc.escape_per_keur(rows)
    assert len(epk) == len(rows) and (epk[epk["family"] == "aluminium"]["s per k€ (low allowance)"] == math.inf).all()


def test_economics_are_sourced_and_close():
    tab = sc.econ_table()
    assert {"source_capex_EUR", "electricity_EUR_per_kWh", "labour_EUR_per_h", "certified_sheet_EUR_per_m2"} <= set(tab.index)
    assert (tab["low"] <= tab["nominal"]).all() and (tab["nominal"] <= tab["high"]).all() and tab["source"].notna().all()
    assert sum(tab["source"] == "assumed") < len(tab) / 2                                         # most parameters carry a source
    s = sc.LASERS["FL2000C"]; e = sc.energy_balance(s)
    cx = sc.capex(s)
    assert cx["low"]["total"] < cx["nominal"]["total"] < cx["high"]["total"] and cx["nominal"]["total"] == 50000 + 3000 + 2500
    assert sc.capex(sc.LASERS["WFL-3000"])["nominal"]["source"] == 75000
    ox = sc.opex_per_hour(s, e)
    assert ox["nominal"]["electricity"] == pytest.approx(e["total_electrical_W"]["nominal"] / 1e3 * 0.1837)
    assert ox["low"]["total"] < ox["nominal"]["total"] < ox["high"]["total"] and ox["nominal"]["total"] == pytest.approx(sum(ox["nominal"][k] for k in ("electricity", "gas", "maintenance", "diodes", "labour")))
    cpm = sc.cost_per_metre(sc.welding_cases(s), e)
    assert len(cpm) == 6 and (cpm["€/m with operator"] > cpm["€/m machine"]).all() and (cpm["€/m machine"] > 0).all()
    assert sc.panel_cost_per_m2("steel", 2.0) == pytest.approx(0.9 * 7850 * 2e-3) and sc.panel_cost_per_m2("certified", 5.0) == 2600
    rows = sc.endurance(s, distances=(5.0, 100.0), run=False)
    for r in rows:
        if r["family"] == "polycarbonate":
            r["holds_s"] = 2.0 * r["thickness_mm"]
        if math.isnan(r["holds_s"]):
            r["holds_s"] = 4.0
    need, per = sc.shield_economics(rows)
    assert len(need) == 18 and len(per) == len(rows)
    pc3 = need[(need["family"] == "polycarbonate") & (need["escape_s"] == 3.0)].iloc[0]
    assert pc3["shield_needed_mm"] == 2.0 and pc3["panel €/m²"] == 24.0 and pc3["panel € per k€ of source"] == pytest.approx(24.0 / 50.0)
    assert (per[per["family"] == "aluminium"]["escape s per € of panel"] == math.inf).all()
