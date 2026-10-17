"""The plate solver against Carslaw & Jaeger's closed forms and the energy balance; the welding numbers against their
closed forms and the textbook's orders of magnitude."""
import math

import numpy as np
import pytest

from vegeta import hyperion as hy
from vegeta.hyperion import thermal as th
from vegeta.hyperion import welding as wd


def test_solids_and_closed_forms():
    s = th.solid("aluminium")
    assert s is th.ALUMINIUM and th.solid(s) is s
    assert s.diffusivity == pytest.approx(200 / (2700 * 950))
    assert s.enthalpy_to_melt() == pytest.approx(2700 * (950 * (933 - th.T_ROOM) + 3.97e5))
    assert th.CLEAR_PC.transmitted(5e-3) == pytest.approx(0.95 ** 2 * math.exp(-5e-3 / 0.05))
    assert th.BLACK_PC.transmitted(5e-3) == 0.0 and th.LASER_PC.transmitted(5e-3) < 1e-6
    assert th.semi_infinite_surface_rise(s, 1e6, 1.0) == pytest.approx(2e6 * math.sqrt(1 / (math.pi * 2700 * 950 * 200)))
    assert th.thin_plate_time_to(s, 1.5e-3, 1e5, 500.0) == pytest.approx(2700 * 1.5e-3 * 950 * (500 - th.T_ROOM) / 1e5)
    assert th.gaussian_steady_rise(s, 100.0, 1e-3) == pytest.approx(100 / (math.sqrt(2 * math.pi) * 200 * 1e-3))
    r = np.linspace(0, 5e-3, 500)
    g = th.irradiance(1000.0, 1e-3, r)
    assert np.trapezoid(g * 2 * math.pi * r, r) == pytest.approx(1000.0, rel=1e-3) if hasattr(np, "trapezoid") else True
    assert th.irradiance(1000.0, 1e-3, 0.0, "tophat") == pytest.approx(1000 / (math.pi * 1e-6))
    with pytest.raises(ValueError):
        th.solid("unobtainium")


def test_semi_infinite_limit():
    """A thick plate, a wide top-hat, a short time, no losses: the surface follows 2q√(t/πρck)."""
    s = th.MILD_STEEL.with_(emissivity=0.0)
    q = 2e6                                                                        # absorbed W/m²
    res = th.heat_plate(s, 0.02, q / s.absorptivity * math.pi * 0.05 ** 2, 0.05, 2.0, shape="tophat", h=0.0, nr=10, nz=40, r_max=0.06, stop_at_failure=False)
    assert res.T_top[-1] - th.T_ROOM == pytest.approx(float(th.semi_infinite_surface_rise(s, q, res.t[-1])), rel=0.08)
    assert res.T_bottom[-1] < th.T_ROOM + 5                                        # the back has not noticed yet


def test_thin_plate_lumped_heating_and_burn_through():
    """A uniform beam over the whole plate, no losses, the far edge insulated by its distance: dT/dt = q/(ρ c d), and the
    melt-through at ρ d (c ΔT + L)/q."""
    s = th.ALUMINIUM.with_(emissivity=0.0)
    d, q = 1.5e-3, 5e5
    P = q / s.absorptivity * math.pi * 0.2 ** 2
    res = th.heat_plate(s, d, P, 0.2, 60.0, shape="tophat", h=0.0, nr=6, nz=4, r_max=0.2)
    t_melt = th.thin_plate_time_to(s, d, q, s.T_melt - 1e-6)                        # to the melting point, without the latent heat
    i = np.searchsorted(res.t, 0.5 * t_melt)
    assert res.T_top[i] - th.T_ROOM == pytest.approx(q * res.t[i] / (s.rho * s.c * d), rel=0.03)
    assert res.failed and res.failure_mode.startswith("melt-through")
    assert res.failure_time == pytest.approx(th.thin_plate_time_to(s, d, q, s.T_melt), rel=0.08)   # with the latent heat: fully molten
    assert res.surface_damage_time == pytest.approx(t_melt, rel=0.08)


def test_melting_plateau_and_steady_state():
    s = th.ALUMINIUM
    res = th.heat_plate(s, 3e-3, 300.0, 0.012, 60.0, h=10.0, nr=20, nz=5, r_max=0.06)
    assert not res.failed and res.steady and res.T_max[-1] < s.T_melt             # 300 W on 3 mm aluminium: warm, never molten
    assert res.survives(100.0) and res.survives(30000.0)
    hot = th.heat_plate(s, 1.5e-3, 3000.0, 0.003, 60.0, h=10.0, nr=24, nz=5, r_max=0.06)   # the same power in a 3 mm spot melts through
    assert hot.failed and hot.surface_damage_time < hot.failure_time
    window = (hot.t > hot.surface_damage_time) & (hot.t < hot.failure_time)
    plateau = hot.T_top[window & (hot.melt_depth == 0.0)]
    assert len(plateau) > 0 and np.all(np.abs(plateau - s.T_melt) < 1.0)          # the face holds at the melting point until its cell is fully molten
    assert np.all(hot.T_top[window] >= s.T_melt - 1e-6)                            # then the liquid superheats while the cells below melt in turn
    assert hot.T_bottom[window][-1] == pytest.approx(s.T_melt, abs=1.0)            # the back face is on its plateau at melt-through
    assert not hot.survives(100.0) and hot.survives(hot.failure_time * 0.5)


def test_steady_solver_against_the_transient_and_the_bound():
    s = th.ALUMINIUM
    st = th.steady_plate(s, 3e-3, 300.0, 0.012, h=10.0, nr=20, nz=5, r_max=0.06)
    tr = th.heat_plate(s, 3e-3, 300.0, 0.012, 60.0, h=10.0, nr=20, nz=5, r_max=0.06)
    assert st["converged"] and not st["reaches_damage"]
    assert st["T_max"] == pytest.approx(tr.T_max[-1], abs=3.0)                     # the transient settled where the steady solve says
    # without losses the steady centre rise is bounded by the thin-plate Green's function
    lossless = th.steady_plate(s.with_(emissivity=0.0), 3e-3, 300.0, 0.012, h=0.0, nr=20, nz=5, r_max=0.06)
    bound = 0.12 * 300 * (math.log(0.06 / 0.012) + 0.5) / (2 * math.pi * 200 * 3e-3)
    assert 0.5 * bound < lossless["T_max"] - th.T_ROOM < 1.05 * bound
    hot = th.steady_plate(s, 1.5e-3, 3000.0, 0.003, h=10.0, nr=24, nz=5, r_max=0.06)
    assert hot["reaches_damage"]                                                   # the transient above melts through: the steady solve agrees
    brick = th.heat_plate(th.BRICK, 0.02, 5000.0, 0.012, 30.0, nr=12, nz=6, stop_at_failure=False)
    assert brick.T_max.max() <= th.BRICK.T_fail + 1e-6                             # the boiling plateau caps the temperature


def test_ablation_and_transmission():
    pc = th.BLACK_PC
    res = th.heat_plate(pc, 5e-3, 1000.0, 0.012, 600.0, h=10.0, nr=20, nz=10)
    assert res.failed and res.failure_mode.startswith("ablated") and res.ablated_depth[-1] == pytest.approx(5e-3, abs=1e-9)
    clear = th.heat_plate(th.CLEAR_PC, 5e-3, 1000.0, 0.012, 20.0, h=10.0, nr=20, nz=10, stop_at_failure=False)
    assert clear.transmitted_fraction == pytest.approx(th.CLEAR_PC.transmitted(5e-3))
    assert not clear.failed and clear.ablated_depth[-1] == 0.0                     # the clear sheet passes the beam and does not char in 20 s
    assert res.ablated_depth[min(len(res.t) - 1, 5)] > 0 or res.failure_time < 20  # the black one is already being eaten
    assert th.heat_plate(pc, 5e-3, 0.0, 0.012, 2.0, nr=8, nz=4).T_max[-1] == pytest.approx(th.T_ROOM)


def test_delivery_fibre_and_head():
    f = wd.DeliveryFibre(25e-6, 0.12)
    assert f.bpp == pytest.approx(3e-6) and f.m2(1070e-9) == pytest.approx(math.pi * 3e-6 / 1070e-9)
    head = wd.WeldHead(0.100, 0.200)
    assert head.spot_radius(f) == pytest.approx(50e-6) and head.cone_half_angle(f) == pytest.approx(0.06)
    assert head.collimated_radius(f) == pytest.approx(0.012) and head.rayleigh_length(f) == pytest.approx(50e-6 / 0.06)
    b = head.focused_beam(f, 1070e-9)
    assert b.w0 == pytest.approx(50e-6) and b.divergence == pytest.approx(0.06, rel=1e-9)   # M² λ/(π w₀) = NA/M


def test_keyhole_rosenthal_and_penetration():
    th_steel = wd.keyhole_threshold("mild steel", 50e-6)
    assert th_steel["P_threshold_W"] == pytest.approx(math.sqrt(2 * math.pi) * 40 * 50e-6 * (3134 - th.T_ROOM) / 0.38)
    assert 1e5 < th_steel["I_threshold_W_per_cm2"] < 2e6                             # the textbook's ~1 MW/cm²
    assert wd.keyhole_threshold("aluminium", 50e-6)["P_threshold_W"] > 5 * th_steel["P_threshold_W"]   # reflective and conductive
    pool = wd.rosenthal_pool("mild steel", 300.0, 0.01)
    assert pool["length_behind"] == pytest.approx(300 / (2 * math.pi * 40 * (1810 - th.T_ROOM)))
    assert 0 < pool["length_ahead"] < pool["length_behind"] and 0 < pool["depth"] < pool["length_behind"]
    slow = wd.rosenthal_pool("mild steel", 300.0, 0.002)
    assert slow["width"] > pool["width"]
    p = wd.penetration("mild steel", 1000.0, 1 / 60, 50e-6)
    assert p["mode"] == "keyhole" and 1.5e-3 < p["depth"] < 4.5e-3                  # 1 kW at 1 m/min: a few millimetres
    assert p["width"] > 2 * 50e-6
    curve = wd.penetration_curve("mild steel", 1000.0, [0.5 / 60, 1 / 60, 3 / 60], 50e-6)
    assert np.all(np.diff(curve["depth"]) < 0) and np.all(np.diff(curve["heat_input_J_per_m"]) < 0)
    v = wd.speed_for_penetration("mild steel", 1000.0, 2e-3, 50e-6)
    assert wd.penetration("mild steel", 1000.0, v, 50e-6)["depth"] == pytest.approx(2e-3, rel=1e-6)
    assert math.isnan(wd.speed_for_penetration("mild steel", 1000.0, 0.1, 50e-6))
    win = wd.process_window("aluminium", 1000.0, [1e-3, 4e-3], 50e-6, absorption=0.5)
    assert win[0]["v_full_penetration"] > win[1]["v_full_penetration"] or math.isnan(win[1]["v_full_penetration"])
    weak = wd.penetration("mild steel", 30.0, 0.01, 50e-6)
    assert weak["mode"] == "conduction" and "pool_length" in weak


def test_backend_dispatch_and_fenicsx_when_present():
    from vegeta.hyperion import thermal_fenicsx as tf
    with pytest.raises(ValueError):
        th.steady_plate(th.ALUMINIUM, 3e-3, 300.0, 0.012, backend="nope")
    st = th.steady_plate(th.ALUMINIUM, 3e-3, 300.0, 0.012, nr=12, nz=4, r_max=0.06, backend="numpy")
    assert st["backend"] == "numpy"
    if not tf.available():
        with pytest.raises(ImportError):
            th.steady_plate(th.ALUMINIUM, 3e-3, 300.0, 0.012, backend="fenicsx")
        pytest.skip("dolfinx not installed: the FEniCSx backend is not exercised here (NOT RUN)")
    fx = th.steady_plate(th.ALUMINIUM, 3e-3, 300.0, 0.012, nr=24, nz=6, r_max=0.06, backend="fenicsx")
    assert fx["backend"] == "fenicsx" and fx["converged"]
    assert fx["T_max"] == pytest.approx(th.steady_plate(th.ALUMINIUM, 3e-3, 300.0, 0.012, nr=24, nz=6, r_max=0.06, backend="numpy")["T_max"], abs=5.0)
    hot = th.heat_plate(th.BLACK_PC, 5e-3, 1000.0, 0.012, 60.0, nr=20, nz=10, backend="fenicsx")
    ref = th.heat_plate(th.BLACK_PC, 5e-3, 1000.0, 0.012, 60.0, nr=20, nz=10, backend="numpy")
    assert hot.failed and hot.failure_time == pytest.approx(ref.failure_time, rel=0.25)
