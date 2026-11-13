"""og.lifting_gas: ideal-gas buoyancy, sealed and vented cells (hand calculations and conservation)."""
import pytest

from ogma import lifting_gas as lg
from ogma import stratosphere as st


def test_helium_lift_at_sea_level():
    # ρ_air − ρ_He at 101325 Pa, 288.15 K: 1.2250 − 0.16928 = 1.0557 kg/m³ (the textbook ~1.06 kg/m³)
    assert lg.net_lift_per_volume(101325.0, 288.15) == pytest.approx(101325 / (8.314462618 / 0.0289644 * 288.15) - 101325 / (8.314462618 / 0.0040026 * 288.15), rel=1e-12)
    assert lg.net_lift_per_volume(101325.0, 288.15) == pytest.approx(1.0557, rel=2e-3)


def test_lift_at_20km_is_7_percent_of_sea_level_per_volume():
    s = st.state(20000.0)
    ratio = lg.net_lift_per_volume(s.pressure, s.temperature) / lg.net_lift_per_volume(101325.0, 288.15)
    assert ratio == pytest.approx(s.density / 1.225, rel=1e-3)   # both scale with p/T


def test_superheat_and_superpressure_signs():
    s = st.state(20000.0)
    base = lg.net_lift_per_volume(s.pressure, s.temperature)
    assert lg.net_lift_per_volume(s.pressure, s.temperature, dT=20.0) > base     # warm gas is lighter
    assert lg.net_lift_per_volume(s.pressure, s.temperature, dp=500.0) < base    # compressed gas is heavier


def test_sealed_cell_keeps_mass_and_lift_while_full():
    s = st.state(20000.0)
    V = 1000.0
    m = lg.fill_for_full(V, s.pressure, s.temperature, dp=100.0)
    cell = lg.GasCell(V, m)
    night, day = cell.state(s.pressure, s.temperature, 0.0), cell.state(s.pressure, s.temperature, 25.0)
    assert night["full"] and day["full"]
    assert night["superpressure"] == pytest.approx(100.0, rel=1e-9)
    assert day["superpressure"] > night["superpressure"]
    assert day["net_lift"] == pytest.approx(night["net_lift"])          # full: same volume, same gas → same lift
    assert cell.vent(s.pressure, s.temperature, 25.0) == 0.0             # sealed cells never vent


def test_slack_below_pressure_altitude():
    s = st.state(20000.0)
    cell = lg.GasCell(1000.0, lg.fill_for_full(1000.0, s.pressure, s.temperature))
    low = st.state(12000.0)
    out = cell.state(low.pressure, low.temperature)
    assert not out["full"] and out["superpressure"] == 0.0 and out["fill"] < 1.0
    # partially full: lift equals the full-cell lift (the gas displaces its own volume, Archimedes, same p and T ratio)
    assert out["net_lift"] == pytest.approx(cell.gas_mass * (lg.M_AIR / lg.GASES["helium"] - 1), rel=1e-6)


def test_vented_cell_loses_gas_by_day():
    s = st.state(20000.0)
    cell = lg.GasCell(1000.0, lg.fill_for_full(1000.0, s.pressure, s.temperature), vented=True, valve_dp=10.0)
    m0 = cell.gas_mass
    released = cell.vent(s.pressure, s.temperature, 20.0)
    assert released > 0 and cell.gas_mass == pytest.approx(m0 - released)
    assert cell.state(s.pressure, s.temperature, 20.0)["superpressure"] == pytest.approx(10.0, abs=1e-6)
    assert cell.state(s.pressure, s.temperature, 0.0)["net_lift"] < lg.GasCell(1000.0, m0).state(s.pressure, s.temperature)["net_lift"]


def test_purity_and_unknown_gas():
    assert lg.specific_gas_constant("helium", 0.95) < lg.specific_gas_constant("helium")
    with pytest.raises(ValueError):
        lg.specific_gas_constant("neon")
