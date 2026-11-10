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
