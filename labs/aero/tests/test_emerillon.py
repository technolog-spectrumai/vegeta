"""ÉMERILLON (notebook 37): the market's bookkeeping, the compressible propeller against Boreas, the drive's limits, the
trim, the top speed, the trade and its front, the CAD against the flight model's geometry.

Run: .venv/bin/python -m pytest -q labs/aero/tests/test_emerillon.py   (half a minute; the CAD tests need CadQuery)
"""
import dataclasses
import math

import pytest

import emerillon_flight as ef
import emerillon_market as mk

DESIGN = ef.DESIGN
FAST = ef.RECORD_BUILD


# ------------------------------------------------------------------------------------------------------------ market
def test_catalogue_rows_are_priced_and_say_what_they_are():
    for p in mk.CATALOGUE.values():
        assert p.price_usd > 0 and p.kind in ("sourced", "partly", "assumed"), p.key
        if p.kind != "assumed":
            assert p.url.startswith("https://"), p.key
        if p.seen:
            assert p.seen[0] <= p.seen[1] and p.seen[0] <= p.price_usd <= p.seen[1] + 1e-9, p.key
    assert mk.part("v3115-900").kv == 900.0
    with pytest.raises(AttributeError):
        mk.part("v3115-900").no_such_number


def test_bom_adds_up_and_the_stack_brings_its_flight_controller():
    rows = mk.bom(DESIGN)
    assert mk.price(DESIGN) == pytest.approx(sum(r["line [USD]"] for r in rows), abs=0.01)
    assert not any("H743" in r["part"] for r in rows)                      # the SpeedyBee stack has its own
    assert any("H743" in r["part"] for r in mk.bom(FAST))
    assert next(r for r in rows if r["group"] == "propeller")["qty"] == 8  # a set and a spare set
    assert next(r for r in mk.bom(FAST) if r["group"] == "battery")["qty"] == 4
    assert mk.price(DESIGN, which="low") <= mk.price(DESIGN) <= mk.price(DESIGN, which="high")


def test_esc_choice_follows_voltage_and_current():
    assert mk.esc_for(6, 45.0).key == "speedybee-f405v4-55"
    assert mk.esc_for(12, 110.0).key == "apd-120f3"
    assert mk.esc_for(12, 200.0) is None
    assert all(c.cells <= mk.part(c.motor).max_cells and c.cells <= mk.part(c.esc).max_cells for c in ef.candidates())


# --------------------------------------------------------------------------------------------------------- propeller
def test_mach_section_is_boreas_at_low_mach_and_loses_efficiency_at_transonic_tips():
    slow_c, slow_i = ef.solve_point("apc7x10", 6000, 15.0), ef.solve_point("apc7x10", 6000, 15.0, compressible=False)
    assert slow_c.tip_mach < 0.2
    assert slow_c.thrust == pytest.approx(slow_i.thrust, rel=0.03) and slow_c.power == pytest.approx(slow_i.power, rel=0.03)
    J = 1.15                                                                # one advance ratio, two tip speeds
    D = 7 * 0.0254
    eff = [ef.solve_point("apc7x10", rpm, J * rpm / 60 * D).efficiency for rpm in (15000, 36000)]
    assert eff[1] < eff[0] - 0.05
    assert ef.solve_point("apc7x10", 36000, J * 600 * D).power > 1.15 * ef.solve_point("apc7x10", 36000, J * 600 * D, compressible=False).power


def test_map_lookup_reproduces_the_solver():
    pm = ef.propeller_map("apc7x10")
    rpm, V = 24000.0, 95.0
    n, D = rpm / 60, pm["D"]
    ct, cp = ef.coefficients(pm, V / (n * D), math.pi * n * D / ef.A_SOUND)
    o = ef.solve_point("apc7x10", rpm, V)
    assert ct == pytest.approx(o.ct, abs=0.004) and cp == pytest.approx(o.cp, abs=0.006)
    assert ef.coefficients(pm, 99.0, 9.0) == ef.coefficients(pm, pm["J"][-1], pm["Mt"][-1])   # held at the edges


# ------------------------------------------------------------------------------------------------------------- drive
def test_drive_respects_the_current_limits_and_the_pack_sags():
    d = ef.drive(FAST, 150.0)
    motor, esc = mk.part(FAST.motor), mk.part(FAST.esc)
    assert d["current"] <= min(motor.max_a, esc.max_a) * 1.001
    assert d["cell_v"] >= ef.CELL_V_MIN - 0.005 and d["cell_v"] < ef.CELL_OCV
    assert d["limit"] in ("voltage", "motor current", "ESC current", "battery")
    one = ef.drive(dataclasses.replace(FAST, n_parallel=1), 150.0)                    # half the packs: the battery holds it back
    assert one["limit"] == "battery" and one["thrust"] < d["thrust"]
    assert d["electric_w"] > d["shaft_w"] > d["thrust"] * 150.0 > 0


def test_trim_carries_the_weight_with_a_few_degrees_at_speed():
    m = ef.mass_budget(DESIGN).mass
    cda = ef.drag_buildup(DESIGN, 100.0)["CdA"]
    a, T = ef.trim(DESIGN, 100.0, m, cda)
    q = 0.5 * ef.RHO * 100.0 ** 2
    assert T * math.sin(a) + q * ef.normal_force_area(DESIGN, a) * math.cos(a) == pytest.approx(m * ef.G, rel=1e-3)
    assert math.degrees(a) < 8.0 and T == pytest.approx(q * cda, rel=0.08)             # the thrust is the drag, nearly
    a_slow, T_slow = ef.trim(DESIGN, 5.0, m, cda)
    assert math.degrees(a_slow) > 70.0 and T_slow == pytest.approx(m * ef.G, rel=0.1)  # near the hover it hangs on its propellers


def test_drag_buildup_is_the_sum_of_its_parts_and_scales_with_the_margin():
    d = ef.drag_buildup(DESIGN, 100.0)
    assert d["clean"] == pytest.approx(d["body"] + d["arms"] + d["pods"] + d["fins"] + d["base"])
    assert d["CdA"] == pytest.approx(d["clean"] * ef.DRAG_MARGIN)
    assert 0.0015 < d["CdA"] < 0.006
    assert ef.drag_buildup(DESIGN, 100.0, slip=1.5)["CdA"] > d["CdA"]


# -------------------------------------------------------------------------------------------------- speed and trade
def test_top_speed_balances_and_follows_pitch_and_drag():
    top = ef.top_speed(DESIGN)
    assert 300.0 < top["kmh"] < 450.0 and abs(top["excess"]) < 0.02 * top["thrust"]
    assert top["limit"] == "voltage" and top["J"] > 1.0                               # the pitch speed sets it (Kestrel's lesson)
    assert ef.top_speed(dataclasses.replace(DESIGN, propeller="apc7x7e"))["kmh"] < top["kmh"] - 40.0
    dirty = ef.balance(DESIGN, top["V"])                                              # 40 % more drag: no longer level at that speed
    cda = ef.drag_buildup(DESIGN, top["V"], margin=ef.DRAG_MARGIN * 1.4)["CdA"]
    assert ef.trim(DESIGN, top["V"], dirty["mass"], cda)[1] > dirty["thrust"]


def test_evaluate_names_what_fails():
    r = ef.evaluate(DESIGN)
    assert r["feasible"] and r["within_rpm_limit"] and r["static_tw"] > ef.STATIC_TW_MIN and r["passes"] >= 2.0
    assert r["price"] == mk.price(DESIGN) and not r["assumed_parts"]
    starved = ef.evaluate(dataclasses.replace(FAST, n_parallel=1, battery="cnhl-1300-100c"))
    assert "energy" in starved["fails"] or "motor heat" in starved["fails"] or starved["limit"] == "battery"
    fast = ef.evaluate(FAST)
    assert fast["kmh"] > 550.0 and not fast["within_rpm_limit"] and fast["tip_mach"] > 0.95
    assert fast["assumed_parts"]                                                      # the special wind and the own propeller


def test_front_is_monotone_and_the_picks_are_on_it():
    rows = ef.load_trade()
    pk = ef.picks(rows)
    front = pk["front"]
    assert len(front) >= 5 and all(b["price"] > a["price"] and b["kmh"] > a["kmh"] for a, b in zip(front, front[1:]))
    assert all(r["feasible"] for r in front)
    assert pk["budget"] is front[0] and pk["fastest"] is front[-1] and pk["knee"] in front
    assert pk["knee"]["config"] == DESIGN                                             # the design is the knee of the stored trade
    assert all(m["usd_per_kmh"] > 0 for m in ef.marginal(front))
    inside = ef.picks(rows, within_rpm_limit=True)
    assert inside["fastest"]["kmh"] < pk["fastest"]["kmh"] and all(r["within_rpm_limit"] for r in inside["front"])


def test_stored_trade_is_the_models():
    stored = {r["config"]: r for r in ef.load_trade()}
    for c in (DESIGN, FAST):
        now = ef.evaluate(c)
        assert stored[c]["kmh"] == pytest.approx(now["kmh"], rel=0.005) and stored[c]["price"] == pytest.approx(now["price"], abs=0.01)


# --------------------------------------------------------------------------------------------------------------- CAD
def test_cad_is_the_flight_models_airframe():
    em = pytest.importorskip("emerillon")
    assert {k: em.resolve()[k] for k in ef.cad_parameters(DESIGN)} == ef.cad_parameters(DESIGN)   # the defaults are the design
    g, geo = em.Emerillon().generate(), ef.geometry(DESIGN)
    assert g.shape.isValid() and len(g.shape.Solids()) == 1
    assert g.dimensions[0] == pytest.approx(geo["length"] * 1e3, rel=0.01)
    wetted = geo["s_body"] + geo["s_arms"] + geo["s_pods"] + geo["s_fins"]
    assert g.surface_area / 1e6 == pytest.approx(wetted, rel=0.08)
    L = em.Emerillon.layout(em.resolve())
    assert L["tip_to_body"] > 3.0 and L["tip_to_tip"] > 3.0 and L["x_prop"] < L["x_arm"]         # the blades clear everything
    for part in ("arm", "nose"):
        s = em.Emerillon().generate(part=part)
        assert s.shape.isValid() and len(s.shape.Solids()) == 1
    with pytest.raises(Exception):
        em.Emerillon().generate(arm_radius=100.0)                                                 # tips into the body
    fast = em.Emerillon().generate(**ef.cad_parameters(FAST))
    assert fast.dimensions[0] > g.dimensions[0] + 250.0                                           # four packs in line


def test_mass_budget_and_printed_parts():
    b = ef.mass_budget(DESIGN)
    groups = b.by_group()
    assert b.mass == pytest.approx(sum(groups.values())) and 0.9 < b.mass < 1.5
    assert groups["airframe"] * 1e3 == pytest.approx(sum(ef.printed_masses(DESIGN).values()), rel=1e-6)
    x_cg, geo = b.properties().com[0], ef.geometry(DESIGN)
    assert geo["x_prop"] < x_cg < geo["nose"] + geo["cyl"]                                        # it hangs below its propellers


# ------------------------------------------------------------------------------------------ ER1, ER2, altitude, flights
def test_variants_meet_their_speeds_and_the_air_thins():
    for v, c in ef.VARIANTS.items():
        assert ef.top_speed(c)["kmh"] == pytest.approx(ef.TARGET[v]["kmh"], abs=4.0)
    sea, high = ef.air_at(0.0), ef.air_at(7000.0)
    assert high["rho"] == pytest.approx(0.59, abs=0.01) and high["a"] < sea["a"] and high["T_C"] == pytest.approx(-30.5, abs=0.5)
    d0, d7 = ef.drive(ef.ER2, 0.5), ef.drive(ef.ER2, 0.5, air=high)
    assert d7["thrust"] < d0["thrust"] and d7["thrust"] / (ef.mass_budget(ef.ER2).mass * ef.G) > 2.0      # thrust is not what stops it
    half = ef.drive(ef.ER2, 30.0, throttle=0.5)
    assert half["limit"] == "throttle" and half["thrust"] < 0.7 * ef.drive(ef.ER2, 30.0)["thrust"]


def test_envelope_is_the_drive_and_inverts():
    mi = pytest.importorskip("emerillon_mission")
    env = mi.envelope("ER1")
    v = env["V"][4]
    d = ef.drive(ef.ER1, v, air=ef.air_at(mi.H_GRID[2]), throttle=mi.U_GRID[3])
    T, P, _ = mi.thrust_at(env, mi.H_GRID[2], v, mi.U_GRID[3])
    assert T == pytest.approx(d["thrust"], rel=1e-6) and P == pytest.approx(d["electric_w"], rel=1e-6)
    u = mi.throttle_for(env, 1000.0, 20.0, 0.5 * mi.thrust_at(env, 1000.0, 20.0, 1.0)[0])
    assert mi.thrust_at(env, 1000.0, 20.0, u)[0] == pytest.approx(0.5 * mi.thrust_at(env, 1000.0, 20.0, 1.0)[0], rel=0.01)
    assert mi.throttle_for(env, 0.0, 0.0, 1e6) == 1.0 and mi.thrust_at(env, 0.0, 0.0, 0.0)[0] == 0.0


def test_ceilings_are_set_by_the_pack_not_the_thrust():
    mi = pytest.importorskip("emerillon_mission")
    for v in ef.VARIANTS:
        c = mi.ceiling(v)
        assert c["static_tw_at_target"] > 2.5 and c["climb_wh"] > c["potential_wh"]          # it can hang there; the climb costs more than m g h
        assert not c["reaches_target"] and c["energy_ceiling_m"] < c["target_m"]             # as built, neither gets there with the reserve
        assert mi.ceiling(v, mode="fast")["climb_wh"] > c["climb_wh"]                        # full throttle is the dear way up
    assert mi.ceiling(ef.HIGH["ER1"], 4000.0)["reaches_target"] and mi.ceiling(ef.HIGH["ER2"], 7000.0)["reaches_target"]
    rows = mi.ceiling_options("ER2")
    assert sum(r["as built"] for r in rows) == 1 and any(r["target met"] for r in rows)


@pytest.mark.parametrize("variant,scenario", [("ER1", "palace"), ("ER2", "mountain"), ("ER2", "sea")])
def test_scenarios_fly_and_land(variant, scenario):
    mi = pytest.importorskip("emerillon_mission")
    f = mi.fly(variant, scenario)
    m = f.metrics
    assert f.ok, f.why
    assert m["touchdown_sink_ms"] <= 1.2 and m["touchdown_miss_m"] <= 1.0 and m["reserve"] >= mi.RESERVE
    assert m["max_kmh"] <= 1.25 * ef.top_speed(ef.VARIANTS[variant])["kmh"] and m["max_g"] <= 7.0
    if scenario == "palace":
        assert f.hist["z"][-1] == pytest.approx(mi.PALACE["roof"], abs=0.1) and m["min_obstacle_clearance_m"] > 3.0
        assert m["max_altitude_m"] > mi.PALACE["ground_asl"] + 1100.0
    else:
        assert m["t_caught_s"] < 60.0


def test_the_climb_scenario_agrees_with_the_slices():
    mi = pytest.importorskip("emerillon_mission")
    f = mi.fly("ER1", "climb", dt=0.05)
    top = next(e for e in f.events if e["leg"] == "top of climb")
    assert f.metrics["max_altitude_m"] == pytest.approx(4000.0, abs=30.0)
    assert top["wh"] == pytest.approx(mi.ceiling("ER1")["climb_wh"], rel=0.12)
    assert not f.ok and any("pack" in w for w in f.why)                                      # as built: up, but not down with the reserve
    assert mi.fly("ER1", "climb", dt=0.05, config=ef.HIGH["ER1"]).ok


# --------------------------------------------------------------------------------------------- the workflow book (E)
def test_campaigns_catalogue_and_fingerprint():
    ec = pytest.importorskip("emerillon_campaigns")
    assert len(ec.cfd_specs("ER2", "drag")) == 3 and len(ec.cfd_specs("ER1", "alpha")) == 8
    assert max(s["V"] for s in ec.cfd_specs("ER2", "drag").values()) == pytest.approx(ec.vehicle("ER2").V_top, abs=0.01)
    with pytest.raises(ValueError):
        ec.cfd_specs("ER2", "everything")
    fp = ec.fingerprint("ER2")
    assert ec.compare(fp, fp)["level"] == "compatible" and ec.compare(fp, ec.fingerprint("ER1"))["level"] == "incompatible"
    assert ec.compare(fp, dict(fp, mass_kg=fp["mass_kg"] * 1.1))["level"] == "incompatible" and ec.compare(fp, None)["level"] == "incompatible"
    assert "Mach" in ec.list_footer(ec.vehicle("ER2"), "cfd") and "PA6-CF" in ec.list_footer(ec.vehicle("ER2"), "fea")


def test_arm_boxes_take_one_face_each_and_the_cases_are_built(tmp_path):
    ec = pytest.importorskip("emerillon_campaigns")
    em = pytest.importorskip("emerillon")
    for v in ("ER1", "ER2"):
        arm = em.Emerillon().generate(part="arm", **ec.params(v)).shape
        for name, b in ec.arm_regions(ec.params(v)).items():
            inside = [f for f in arm.Faces() if all(b[i] <= lo and hi <= b[i + 3] for i, (lo, hi) in enumerate(
                zip((f.BoundingBox().xmin, f.BoundingBox().ymin, f.BoundingBox().zmin), (f.BoundingBox().xmax, f.BoundingBox().ymax, f.BoundingBox().zmax))))]
            assert len(inside) == 1, (v, name)
    cases = ec.fea_cases("ER2", "arm", tmp_path)
    assert [c.name for c in cases] == ["arm_A", "arm_B", "arm_C", "arm_modes"] and cases[-1].kind == "modes"
    assert (tmp_path / "cad" / "emerillon_ER2_arm.step").exists() and len({ec.fea_case_key(c) for c in cases}) == 4
    beam = ec.arm_beam("ER2")
    assert beam["stress_side_MPa"] < ec.PA6CF["strength"] and beam["f1_thin_Hz"] < beam["rotor_1P_Hz"]   # the rotor runs through the arm's first mode


def test_book_is_registered_and_prepares_cfd_cases_without_a_solver(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[3]
    side = root / "labs" / "aero" / "campaigns" / "side_labs"
    sys.path.insert(0, str(side))
    import library
    assert library.BOOKS["emerillon"]["code"] == "E" and library.by_code("e3")[1] == "arm"
    out = tmp_path / "run"
    r = subprocess.run([sys.executable, str(side / "emerillon" / "airframe_cfd.py"), "--variant", "ER1", "--chapter", "drag", "--prepare-only", "--cases", "100 %",
                        "--out", str(out), "--quiet"], capture_output=True, text=True, timeout=300)
    assert r.returncode in (0, 3), r.stdout[-800:] + r.stderr[-800:]
    case = next((out / "cases").glob("ER1_a00.0_V*_screening"))
    assert (case / "system" / "controlDict").exists() and (case / "system" / "snappyHexMeshDict").exists() and list((out / "cases" / "stl").glob("*.stl"))
