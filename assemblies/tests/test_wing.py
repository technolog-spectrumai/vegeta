"""The wing component against the notebooks it came from: planform (09a cell 10), Pratt's gust (26 cell 12 and the
promoted AGUYA wing), the lower skins and regions on FixedWing's own wing."""
import math

import pytest

from assemblies.components import aguya_wing, wing
from assemblies.components.fixed_wing import FixedWing


def test_spec_from_fixed_wing_defaults_and_planform():
    p = FixedWing().resolve()
    s = wing.WingSpec.from_params(p)
    b, c0, lam = p["span"] / 1000, p["root_chord"] / 1000, p["taper"]
    S = b * c0 * (1 + lam) / 2
    pl = s.planform()
    assert pl["area_m2"] == pytest.approx(S) and pl["aspect_ratio"] == pytest.approx(b ** 2 / S)
    assert s.type == "tapered" and s.naca == f"NACA {round(p['camber'] * 100)}{round(p['camber_pos'] * 10)}{round(p['thickness'] * 100):02d}"
    assert wing.WingSpec(1000, 200, taper=1.0).type == "rectangular"


def test_spec_refuses_what_the_geometry_cannot_make():
    with pytest.raises(ValueError, match="delta"):
        wing.WingSpec(1000, 300, taper=0.0)
    with pytest.raises(TypeError):
        wing.WingSpec(1000, 300, sweep_deg=30)


def test_gust_is_notebook_26s():
    # notebook 26 cell 12, written out with its numbers: W_MAX, S_W, root chord 270, taper 0.6
    W_MAX, S_W, RHO, G = 2.3 * 9.81, 0.30, 1.225, 9.81
    V_GUST, U_GUST, A_LIFT = 60.0, 7.5, 2 * math.pi * 0.85 * 6.5 / (6.5 + 2)
    MU = 2 * W_MAX / S_W / (RHO * 270 * (1 + 0.6) / 2000 * A_LIFT * G)
    K_G = 0.88 * MU / (5.3 + MU)
    N_GUST = 1 + K_G * RHO * V_GUST * U_GUST * A_LIFT * S_W / (2 * W_MAX)
    g = wing.pratt_gust(W_MAX, S_W, 270 * (1 + 0.6) / 2000, wing.lift_slope(6.5, 0.85), V_GUST, g=G)
    assert g["load_factor"] == pytest.approx(N_GUST) and g["alleviation"] == pytest.approx(K_G)


def test_gust_is_the_aguya_wings():
    from assemblies.components import aguya, aguya_flight as F
    from vegeta.boreas import microjet as mj
    e = mj.from_catalogue("140 N class")
    P = aguya.for_engine(e)
    af = F.airframe(P, F.jet_unit(e))
    a = aguya_wing.gust(P, af, 160.0)
    b = wing.pratt_gust((af.dry_mass_kg + af.tank_kg) * 9.80665, af.wing_area_m2, P["root_chord"] * (1 + P["taper"]) / 2000,
                        wing.lift_slope(af.aspect_ratio), 160.0, limit_load_factor=6.0)
    for k in ("load_factor", "alleviation", "penetration_speed", "mass_ratio"):
        assert b[k] == pytest.approx(a[k])


def test_lower_skins_and_model_on_the_fixed_wing(tmp_path):
    pytest.importorskip("cadquery")
    pytest.importorskip("gmsh")
    fw = FixedWing()
    p = fw.resolve(part="wing")
    step = fw.generate(part="wing").export(tmp_path, formats=("step",), basename="wing").artifacts["step"]
    tags = wing.lower_skins(step)
    assert tags == aguya_wing.lower_skins(step) and len(set(tags)) == 2
    regions = [wing.root_region(p["fuselage_diameter"]), *wing.motor_regions(p["nacelle_y"], p["nacelle_diameter"],
                                                                             p["nacelle_forward"])]
    import vegeta.talos as talos
    m = wing.wing_model(step, regions + [talos.Surfaces("lift", tags)], [talos.Pressure("lift", 1e-3)], element_mm=10.0)
    assert m.material.name.startswith("LW-PLA") and [r.name for r in m.regions] == ["root", "motor_left", "motor_right", "lift"]
