"""The propeller, in air or water: one definition for the blade-element model (Boreas), the CAD (Dedalus) and the CFD
and FEA cases built on them.

Every notebook typed its propeller three times: the ``boreas.Propeller.from_pitch`` call (metres), the
``dedalus.examples.Propeller`` keyword arguments (millimetres, kept consistent by hand) and the tables of the rotor-disk
CFD. A ``PropellerSpec`` holds the numbers once, exactly as the notebooks have them (the CAD chords are stored, not
derived: notebook 12's CAD blade is narrower at the root than its BEMT blade). ``CATALOGUE`` holds the propellers the
notebooks use. The functions below are the cells that were repeated across notebooks 08, 09a, 12, 13, 25, 26b and
``scenarios/run_scenario.py``, lifted as they were.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
from vegeta import aeromant, boreas, talos

from ..vida import digest

# ------------------------------------------------------------------------------------------------- sections and drives
SECTIONS = {
    # notebook 08 cell 46
    "quad 5 in": dict(name="thin cambered section", cl_alpha=2 * math.pi * 0.9, alpha0_deg=-3.0, cl_max=1.1, cd0=0.025,
                      k=0.045, source="assumed for a moulded 5-inch blade at Re ~ 1e5"),
    # notebook 09a cell 42 = scenarios/run_scenario.py air()
    "electric 9 in": dict(name="thin cambered section", cl_alpha=2 * math.pi * 0.9, alpha0_deg=-2.5, cl_max=1.1, cd0=0.02,
                          k=0.04, source="assumed for a 9-inch blade at Re ~ 1.5e5"),
    # notebook 25 cell 4
    "electric 10 in": dict(name="thin cambered section", cl_alpha=2 * math.pi * 0.9, alpha0_deg=-3.0, cl_max=1.1, cd0=0.018,
                           k=0.04, source="assumed for a 10-inch blade at Re ~ 1e5 (fit a measured polar here)"),
    # notebooks 12 cell 15, 13 cell 13 = scenarios/run_scenario.py sub() and boat()
    "marine": dict(name="marine blade section", cl_alpha=5.5, alpha0_deg=-2.0, cl_max=1.0, cd0=0.02, k=0.05, source="assumed"),
}
MOTORS = {
    "2306-2400KV": dict(kv_rpm_per_volt=2400, resistance_ohm=0.06, no_load_current_a=1.2, max_current_a=40, mass_kg=0.030),  # 08
    "2212-920KV": dict(kv_rpm_per_volt=920, resistance_ohm=0.12, no_load_current_a=0.6, max_current_a=20, mass_kg=0.055),    # 09a
    "2836-500KV (water-cooled pod)": dict(kv_rpm_per_volt=500, resistance_ohm=0.08, no_load_current_a=0.8, max_current_a=30,
                                          mass_kg=0.12),                                                                  # 12
    "thruster 100KV": dict(kv_rpm_per_volt=100, resistance_ohm=0.6, no_load_current_a=0.3, max_current_a=5.0, mass_kg=0.35),  # 13
}
BATTERIES = {
    "4S 1500 mAh": dict(cells=4, capacity_ah=1.5, usable_fraction=0.8, mass_kg=0.180),        # 08
    "3S 5000 mAh": dict(cells=3, capacity_ah=5.0, usable_fraction=0.8, mass_kg=0.380),        # 09a
    "4S 10 Ah": dict(cells=4, capacity_ah=10.0, usable_fraction=0.8, mass_kg=0.9),            # 12
    "7S Li-ion 20 Ah": dict(cells=7, capacity_ah=20.0, usable_fraction=0.85, mass_kg=6.0),    # 13
}
MEDIA = {"air": dict(density=1.225, kinematic_viscosity=1.5e-5), "sea water": dict(density=1025.0, kinematic_viscosity=1.05e-6)}


def airfoil(section: str | dict) -> boreas.Airfoil:
    return boreas.Airfoil(**(SECTIONS[section] if isinstance(section, str) else section))


def motor(name: str) -> boreas.Motor:
    return boreas.Motor(name, **MOTORS[name])


def battery(name: str) -> boreas.Battery:
    return boreas.Battery(name, **BATTERIES[name])


# ------------------------------------------------------------------------------------------------- the spec
@dataclass(frozen=True)
class PropellerSpec:
    """One propeller: the ``from_pitch`` arguments (SI), its section, and the CAD keyword arguments (mm) as the
    notebook that proved it had them."""

    name: str
    diameter_m: float
    pitch_m: float
    blades: int
    chord_root_m: float
    chord_max_m: float
    chord_tip_m: float
    section: str
    cad: dict = field(default_factory=dict)
    mass_kg: float = 0.0
    rotor_mass_kg: float = 0.0
    hub_radius_m: float | None = None
    notes: str = ""

    def model(self) -> boreas.Propeller:
        return boreas.Propeller.from_pitch(self.name, self.diameter_m, self.pitch_m, blades=self.blades,
                                           chord_root_m=self.chord_root_m, chord_max_m=self.chord_max_m,
                                           chord_tip_m=self.chord_tip_m, hub_radius_m=self.hub_radius_m,
                                           mass_kg=self.mass_kg, rotor_mass_kg=self.rotor_mass_kg, notes=self.notes)

    def airfoil(self) -> boreas.Airfoil:
        return airfoil(self.section)

    def cad_kw(self, blades: int | None = None) -> dict:
        """The ``dedalus.examples.Propeller`` arguments (mm); ``blades=1`` for the single blade of the FEA."""
        if not self.cad:
            raise ValueError(f"{self.name} has no CAD definition")
        return dict(self.cad, blades=self.blades if blades is None else blades)

    def with_blades(self, blades: int) -> "PropellerSpec":
        return replace(self, blades=blades, name=self.name.replace(f"{self.blades} blades", f"{blades} blades"))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PropellerSpec":
        return cls(**d)


def _in(d_in, p_in):
    return boreas.inches(d_in, p_in)


CATALOGUE = {
    # notebook 08 cells 46 / 49: the quadcopter's
    "5x4.3 tri-blade": PropellerSpec(
        "5x4.3 tri-blade", *_in(5, 4.3), blades=3, chord_root_m=0.010, chord_max_m=0.016, chord_tip_m=0.006,
        section="quad 5 in", mass_kg=0.0045, rotor_mass_kg=0.020,
        notes="generic planform; fit chord/beta to the real propeller for better numbers",
        cad=dict(diameter=_in(5, 4.3)[0] * 1000, pitch=_in(5, 4.3)[1] * 1000, hub_diameter=12, hub_height=7, bore=5,
                 chord_root=10, chord_max=16, chord_tip=6, thickness=0.10, camber=0.05)),
    # notebook 09a cells 42 / 44 = 09b cell 4 = scenarios/run_scenario.py air(): the fixed wing's
    "9x6 electric": PropellerSpec(
        "9x6 electric", *_in(9, 6), blades=2, chord_root_m=0.014, chord_max_m=0.022, chord_tip_m=0.006,
        section="electric 9 in", mass_kg=0.012, rotor_mass_kg=0.045, notes="generic planform",
        cad=dict(diameter=_in(9, 6)[0] * 1000, pitch=_in(9, 6)[1] * 1000, hub_diameter=16, hub_height=9, bore=5,
                 chord_root=14, chord_max=22, chord_tip=6, thickness=0.09, camber=0.04)),
    # notebook 25 cell 4 (prop_of / cad_kw), 2 blades; with_blades(3) for the other one
    "10x6, 2 blades": PropellerSpec(
        "10x6, 2 blades", 0.254, 0.1524, blades=2, chord_root_m=0.018, chord_max_m=0.026, chord_tip_m=0.010,
        section="electric 10 in", hub_radius_m=0.010,
        cad=dict(diameter=254.0, pitch=152.4, chord_root=18.0, chord_max=26.0, chord_tip=10.0, thickness=0.10, camber=0.04,
                 stations=8, hub_diameter=20.0, hub_height=10.0, bore=5.0)),
    # notebook 12 cells 15 / 40 = scenarios/run_scenario.py boat(): the survey boat's
    "60 mm 3-blade marine": PropellerSpec(
        "60 mm 3-blade marine", 0.060, 0.050, blades=3, chord_root_m=0.012, chord_max_m=0.018, chord_tip_m=0.008,
        section="marine", mass_kg=0.02, rotor_mass_kg=0.05, notes="generic planform; fit to the real propeller",
        cad=dict(diameter=60.0, pitch=50.0, hub_diameter=12.0, hub_height=8.0, bore=4.0, chord_root=10.0, chord_max=18.0,
                 chord_tip=8.0, thickness=0.12, camber=0.05, stations=8)),
    # notebook 13 cells 13 / 18 = scenarios/run_scenario.py sub(): the submarine's
    "120 mm 3-blade": PropellerSpec(
        "120 mm 3-blade", 0.120, 0.100, blades=3, chord_root_m=0.018, chord_max_m=0.030, chord_tip_m=0.012,
        section="marine", mass_kg=0.06, rotor_mass_kg=0.15, notes="generic planform",
        cad=dict(diameter=120.0, pitch=100.0, hub_diameter=24.0, hub_height=16.0, bore=8.0, chord_root=18.0, chord_max=30.0,
                 chord_tip=12.0, thickness=0.12, camber=0.05, stations=8)),
}


def get(name: str) -> PropellerSpec:
    if name not in CATALOGUE:
        raise KeyError(f"no propeller {name!r}; the catalogue has {sorted(CATALOGUE)}")
    return CATALOGUE[name]


# ------------------------------------------------------------------------------------------------- rotor-disk tables
def polar_table(section: boreas.Airfoil) -> list[list[float]]:
    """``[[alpha_deg, cd, cl], ...]`` over -180..180 deg for Aeromant's rotor-disk templates (09a cell 72, 26b cell 16,
    ``scenarios/run_scenario._polar_table``)."""
    alphas = np.unique(np.r_[np.arange(-180, -30, 10), np.arange(-30, 31, 1), np.arange(40, 181, 10)]).astype(float)
    cl, cd = section.coefficients(np.radians(alphas))
    return [[float(a), float(d), float(l)] for a, l, d in zip(alphas, cl, cd)]


def blade_table(prop: boreas.Propeller) -> list[list[float]]:
    """``[[r_m, beta_deg, chord_m], ...]`` for Aeromant's rotor-disk templates."""
    return [[float(r), float(b), float(c)] for r, b, c in zip(prop.r, prop.beta_deg, prop.chord)]


# ------------------------------------------------------------------------------------------------- CAD files
def cad_files(spec: PropellerSpec, out: Path, *, blades: int | None = None, stl_tolerance: float = 0.02) -> dict[str, Path]:
    """The propeller (or one blade) as STEP and STL in ``out``, plus the STL turned so the axis is +x (the frame of the
    rotor templates: CAD axis z -> +x, notebooks 08 cell 67, 09a cell 60). Files made from the same arguments are kept
    (``cad.params.json``): a new STEP carries a new time stamp, and would change every FEA mesh key made from it."""
    from vegeta import dedalus
    from vegeta.dedalus.examples import Propeller as PropellerCAD

    out = Path(out)
    kw = spec.cad_kw(blades)
    stamp, want = out / "cad.params.json", digest({"cad": kw, "stl_tolerance": stl_tolerance})
    paths = {"step": out / "propeller.step", "stl": out / "propeller.stl", "stl_axis_x": out / "propeller_axis_x.stl"}
    if stamp.is_file() and json.loads(stamp.read_text()).get("params") == want and all(p.is_file() for p in paths.values()):
        return paths
    out.mkdir(parents=True, exist_ok=True)
    cad = PropellerCAD().generate(**kw)
    res = cad.export(out, formats=("step", "stl"), basename="propeller", stl_tolerance=stl_tolerance)
    res.raise_for_status()
    turned = dedalus.Geometry.from_cadquery(cad.shape.rotate((0, 0, 0), (0, 1, 0), 90), name="propeller_axis_x")
    turned.export_stl(paths["stl_axis_x"], tolerance=stl_tolerance)
    stamp.write_text(json.dumps({"params": want, "volume_mm3": float(cad.volume)}) + "\n")
    return paths


# ------------------------------------------------------------------------------------------------- rotor CFD
ROTOR_FIDELITY = {
    "smoke": dict(iterations=60, cells_per_diameter=5.0, surface_level=3, near_level=2, rotor_level=2, wake_level=1),   # 25 smoke
    "quick": dict(iterations=400, cells_per_diameter=6.0, surface_level=3, near_level=2, rotor_level=2, wake_level=1),  # 08/09a/12/13
    "full": dict(iterations=600, cells_per_diameter=8.0, surface_level=4, near_level=3, rotor_level=3, wake_level=2),   # 25 full
}


def rotor_params(spec: PropellerSpec, rpm: float, *, airspeed: float | None = None, medium: str = "air",
                 fidelity: str = "quick", rotation: int = 1) -> dict:
    """The CFD values of the resolved-blade rotor case at an operating point (no ``airspeed``: hover,
    ``rotor_mrf_static``; with it: ``rotor_mrf``)."""
    if fidelity not in ROTOR_FIDELITY:
        raise ValueError(f"fidelity must be one of {tuple(ROTOR_FIDELITY)}")
    m = MEDIA[medium]
    p = dict(rpm=float(rpm), diameter=spec.diameter_m, kinematic_viscosity=m["kinematic_viscosity"], density=m["density"],
             rotation=rotation, **ROTOR_FIDELITY[fidelity])
    if airspeed is not None:
        p["airspeed"] = float(airspeed)
    return p


def rotor_case(stl_axis_x: Path, params: dict, workdir: Path, env: aeromant.OpenFOAMEnvironment) -> aeromant.CFDCase:
    template = "rotor_mrf" if "airspeed" in params else "rotor_mrf_static"
    return aeromant.CFDCase(template, stl_axis_x, params, workdir=workdir, geometry_units="mm", environment=env)


# ------------------------------------------------------------------------------------------------- blade FEA
BLADE_MATERIAL = dict(name="PA6-GF30 (moulded blade)", youngs_modulus=8000.0, poissons_ratio=0.35, density=1.35e-9,
                      yield_strength=100.0, source="assumed moulded glass-filled nylon")      # 08 cell 76 = 09a cell 68


def blade_loads(prop: boreas.Propeller, point) -> dict:
    """Thrust and tangential force per blade at an operating point (the tangential at 0.7 R from the torque). ``point``
    is a drive point (``Propulsion.at_throttle``/``for_thrust``, the torque in ``point.aero``) or a blade-element one."""
    aero = getattr(point, "aero", point)
    return {"thrust_N": float(point.thrust / prop.blades),
            "tangential_N": float(aero.torque / (prop.blades * 0.7 * prop.radius)), "rpm": float(point.rpm)}


def blade_model(spec: PropellerSpec, step: Path, loads: dict, *, element_mm: float, material: dict | None = None,
                name: str = "blade", root_gap_mm: float = 1.5, half_width_mm: float | None = None) -> talos.StructuralModel:
    """One blade on its hub: the hub fixed, the blade's thrust and tangential force on the blade (08 cell 76,
    09a cell 68; notebook 12 cell 48 boxes the blade at ``half_width_mm=25``, 13 cell 25 at ``root_gap_mm=2``)."""
    kw = spec.cad_kw(1)
    hub_r, hub_h, R_tip = kw["hub_diameter"] / 2, kw["hub_height"], kw["diameter"] / 2
    w = R_tip if half_width_mm is None else half_width_mm
    regions = [talos.SurfacesInBox("hub", (-hub_r - 0.5, -hub_r - 0.5, -hub_h / 2 - 0.5, hub_r + 0.5, hub_r + 0.5, hub_h / 2 + 0.5)),
               talos.SurfacesInBox("blade", (hub_r - root_gap_mm, -w, -w, R_tip + 1.0, w, w))]
    mat = talos.Material(**(material or BLADE_MATERIAL))
    return talos.StructuralModel(step, "mm-N-MPa", mat, regions, [talos.FixedSupport("hub")],
                                 [talos.Force("blade", fz=loads["thrust_N"], fy=-loads["tangential_N"])],
                                 talos.MeshSettings(element_size=element_mm), name=name)
