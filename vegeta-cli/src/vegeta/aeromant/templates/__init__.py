"""Known-good OpenFOAM template cases and their specifications.

A template holds the engineering decisions (domain, patches, physics model, boundary conditions,
meshing and solver settings). Its :class:`TemplateSpec` states which values the engineer must
supply and which values the template decides (visible defaults that can be explicitly overridden).
Each template ships its case files in both OpenFOAM dialects (``com/`` for openfoam.com,
``org/`` for openfoam.org) because the two forks use different dictionaries and solvers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

TEMPLATE_ROOT = Path(__file__).parent
REQUIRED = object()


@dataclass(frozen=True)
class TemplateParameter:
    name: str
    description: str
    units: str = ""
    default: Any = REQUIRED  # REQUIRED -> the engineer must give it; otherwise a template decision
    kind: str = "float"      # float | int | vector

    @property
    def required(self) -> bool:
        return self.default is REQUIRED

    def validate(self, value: Any) -> Any:
        if self.kind == "vector":
            arr = np.asarray(value, dtype=float)
            if arr.shape != (3,):
                raise ValueError(f"template parameter {self.name!r} needs 3 components, got {value!r}")
            return [float(v) for v in arr]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"template parameter {self.name!r} must be a number, got {value!r}")
        if self.kind == "int":
            if int(value) != value or value < 1:
                raise ValueError(f"template parameter {self.name!r} must be a positive integer")
            return int(value)
        if not math.isfinite(value):
            raise ValueError(f"template parameter {self.name!r} must be finite")
        return float(value)


@dataclass(frozen=True)
class Step:
    name: str
    argv: tuple[str, ...]
    kind: str = "openfoam"  # openfoam | internal
    description: str = ""


@dataclass(frozen=True)
class Flavor:
    """Case files and pipeline for one OpenFOAM fork (the dialects differ)."""

    name: str            # "openfoam.com" | "openfoam.org"
    directory: str       # subdirectory of the template holding the case files
    geometry_dir: str    # where the STL goes inside the case
    pipeline: tuple[Step, ...]
    versions: str        # human note, e.g. "v1912+" or "12+"


@dataclass(frozen=True)
class TemplateSpec:
    name: str
    description: str
    parameters: tuple[TemplateParameter, ...]
    flavors: tuple[Flavor, ...]
    derive: Callable[[dict, np.ndarray, np.ndarray], dict[str, Any]]
    max_body_extent: float  # largest body dimension allowed, in multiples of reference_length
    patches: dict[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    @property
    def directory(self) -> Path:
        return TEMPLATE_ROOT / self.name

    @property
    def flavor_names(self) -> list[str]:
        return [f.name for f in self.flavors]

    def flavor(self, name: str | None) -> Flavor:
        """Case files for an OpenFOAM flavor; ``None`` selects the first (openfoam.com)."""
        if name is None:
            return self.flavors[0]
        for f in self.flavors:
            if f.name == name:
                return f
        raise ValueError(f"template {self.name!r} has no case files for {name!r}; available: {self.flavor_names}")

    def case_dir(self, flavor: str | None) -> Path:
        return self.directory / self.flavor(flavor).directory

    def step_names(self, flavor: str | None = None) -> list[str]:
        return [s.name for s in self.flavor(flavor).pipeline]

    def resolve(self, values: dict[str, Any]) -> dict[str, Any]:
        """Validate engineer-supplied values; add template decisions for the rest."""
        known = {p.name: p for p in self.parameters}
        unknown = sorted(set(values) - set(known))
        if unknown:
            raise ValueError(f"template {self.name!r} has no parameter(s) {unknown}; known: {sorted(known)}")
        missing = [p.name for p in self.parameters if p.required and p.name not in values]
        if missing:
            raise ValueError(f"template {self.name!r} requires explicit value(s) for {missing}")
        out = {}
        for p in self.parameters:
            out[p.name] = p.validate(values[p.name]) if p.name in values else p.default
        for positive in ("velocity", "kinematic_viscosity", "density", "reference_area", "reference_length", "diameter", "rpm"):
            if positive in out and out[positive] <= 0:
                raise ValueError(f"{positive} must be > 0")
        return out

    def describe(self) -> str:
        lines = [f"{self.name}", f"  {self.description}", "  parameters:"]
        for p in self.parameters:
            d = "REQUIRED" if p.required else f"template default = {p.default}"
            lines.append(f"    {p.name:<22} [{p.units}] {d} — {p.description}")
        for f in self.flavors:
            lines.append(f"  {f.name} ({f.versions}): " + " -> ".join(s.name for s in f.pipeline))
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)


# -- shared definitions --------------------------------------------------------------------------
COMMON_REQUIRED = (
    TemplateParameter("velocity", "free-stream speed along +x", "m/s"),
    TemplateParameter("kinematic_viscosity", "fluid kinematic viscosity", "m^2/s"),
    TemplateParameter("density", "fluid density (for dimensional forces; coefficients use it consistently)", "kg/m^3"),
    TemplateParameter("reference_area", "reference area A_ref for coefficients", "m^2"),
    TemplateParameter("reference_length", "reference length L_ref for Cm; also scales domain and mesh", "m"),
    TemplateParameter("center_of_rotation", "moment reference point", "m", kind="vector"),
)

COM_PIPELINE = (
    Step("blockMesh", ("blockMesh",), description="background hex mesh of the domain"),
    Step("features", ("surfaceFeatureExtract",), description="feature edges of body.stl"),
    Step("snappyHexMesh", ("snappyHexMesh", "-overwrite"), description="castellate and snap around the body"),
    Step("checkMesh", ("checkMesh",), description="mesh quality report"),
    Step("restore0", ("copy", "0.orig", "0"), kind="internal", description="initial fields for the meshed case"),
    Step("solver", ("simpleFoam",), description="steady incompressible solver with forceCoeffs"),
)
ORG_PIPELINE = (
    Step("blockMesh", ("blockMesh",), description="background hex mesh of the domain"),
    Step("features", ("surfaceFeatures",), description="feature edges of body.stl"),
    Step("snappyHexMesh", ("snappyHexMesh", "-overwrite"), description="castellate and snap around the body"),
    Step("checkMesh", ("checkMesh",), description="mesh quality report"),
    Step("restore0", ("copy", "0.orig", "0"), kind="internal", description="initial fields for the meshed case"),
    Step("solver", ("foamRun",), description="steady incompressibleFluid solver with forceCoeffs"),
)
EXTERNAL_FLAVORS = (
    Flavor("openfoam.com", "com", "constant/triSurface", COM_PIPELINE, "v1912 and later"),
    Flavor("openfoam.org", "org", "constant/geometry", ORG_PIPELINE, "12 and later"),
)

EXTERNAL_PATCHES = {
    "inlet": "fixed velocity (+x), zero-gradient pressure",
    "outlet": "fixed pressure 0, inletOutlet velocity",
    "sides": "slip (four lateral faces)",
    "body*": "no-slip wall; forceCoeffs patch",
}


def _fmt(v) -> str:
    if isinstance(v, (list, tuple, np.ndarray)):
        return " ".join(_fmt(x) for x in v)
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f"{float(v):.9g}"


def _external_derive(p: dict, bmin: np.ndarray, bmax: np.ndarray) -> dict[str, str]:
    """Domain and refinement boxes placed around the body bounding box, sized in L_ref."""
    L = p["reference_length"]
    lo = np.array([bmin[0] - p["upstream"] * L, bmin[1] - p["lateral"] * L, bmin[2] - p["lateral"] * L])
    hi = np.array([bmax[0] + p["downstream"] * L, bmax[1] + p["lateral"] * L, bmax[2] + p["lateral"] * L])
    h = L / p["cells_per_length"]
    n = np.maximum(1, np.ceil((hi - lo) / h)).astype(int)
    hi = lo + n * h  # keep background cells cubic
    near = 0.25 * L
    wake_min = bmin - np.array([near, near, near])
    wake_max = bmax + np.array([p["wake_length"] * L, 0.5 * L, 0.5 * L])
    loc = lo + np.array([0.37 * p["upstream"] * L, 0.123 * (hi[1] - lo[1]), 0.211 * (hi[2] - lo[2])])
    out = {
        "XMIN": lo[0], "YMIN": lo[1], "ZMIN": lo[2], "XMAX": hi[0], "YMAX": hi[1], "ZMAX": hi[2],
        "NX": n[0], "NY": n[1], "NZ": n[2],
        "NEAR_MIN": bmin - near, "NEAR_MAX": bmax + near,
        "WAKE_MIN": wake_min, "WAKE_MAX": np.minimum(wake_max, hi - h),
        "LOCATION_IN_MESH": loc,
        "SURFACE_LEVEL_MIN": p["surface_level"], "SURFACE_LEVEL_MAX": p["surface_level"],
        "FEATURE_LEVEL": p["surface_level"], "NEAR_LEVEL": p["near_level"], "WAKE_LEVEL": p["wake_level"],
        "VELOCITY": p["velocity"], "KINEMATIC_VISCOSITY": p["kinematic_viscosity"], "DENSITY": p["density"],
        "REFERENCE_AREA": p["reference_area"], "REFERENCE_LENGTH": p["reference_length"],
        "CENTER_OF_ROTATION": p["center_of_rotation"], "ITERATIONS": p["iterations"],
        "RESIDUAL_TARGET": p["residual_target"],
    }
    if "turbulence_intensity" in p:
        k = 1.5 * (p["velocity"] * p["turbulence_intensity"]) ** 2
        out["K_INLET"] = k
        out["OMEGA_INLET"] = k / (p["kinematic_viscosity"] * p["viscosity_ratio"])
    return {key: _fmt(v) for key, v in out.items()}


def _mesh_and_domain(iterations: int) -> tuple[TemplateParameter, ...]:
    return (
        TemplateParameter("iterations", "maximum SIMPLE iterations", "", iterations, "int"),
        TemplateParameter("residual_target", "stop when all initial residuals are below", "", 1e-5),
        TemplateParameter("upstream", "domain inlet distance ahead of the body", "L_ref", 5.0),
        TemplateParameter("downstream", "domain outlet distance behind the body", "L_ref", 12.0),
        TemplateParameter("lateral", "domain half-width beyond the body", "L_ref", 5.0),
        TemplateParameter("wake_length", "refined wake region behind the body", "L_ref", 3.0),
        TemplateParameter("cells_per_length", "background cells per L_ref", "", 2.0),
        TemplateParameter("surface_level", "snappy refinement level at the body", "", 4, "int"),
        TemplateParameter("near_level", "refinement level close to the body", "", 3, "int"),
        TemplateParameter("wake_level", "refinement level in the wake box", "", 2, "int"),
    )


LAMINAR = TemplateSpec(
    name="laminar_external",
    description="Steady laminar external flow around a single body in a box domain (flow +x, lift +z).",
    parameters=COMMON_REQUIRED + _mesh_and_domain(400),
    flavors=EXTERNAL_FLAVORS,
    derive=_external_derive,
    max_body_extent=4.0,
    patches=EXTERNAL_PATCHES,
    notes=(
        "valid only for steady laminar flow (roughly Re = U L / nu below a few hundred for bluff bodies)",
        "no boundary-layer prism layers; results are mesh dependent — run a refinement study",
        "drag +x, lift +z, pitch axis +y; forceCoeffs on all patches named body*",
    ),
)

RANS_KSST = TemplateSpec(
    name="rans_ksst_external",
    description="Steady RANS k-omega SST external flow with wall functions (flow +x, lift +z).",
    parameters=COMMON_REQUIRED + _mesh_and_domain(600) + (
        TemplateParameter("turbulence_intensity", "inlet turbulence intensity", "-", 0.005),
        TemplateParameter("viscosity_ratio", "inlet eddy/molecular viscosity ratio", "-", 10.0),
    ),
    flavors=EXTERNAL_FLAVORS,
    derive=_external_derive,
    max_body_extent=4.0,
    patches=EXTERNAL_PATCHES,
    notes=(
        "wall functions without prism layers: y+ is not controlled; use for trend comparison, not absolute drag",
        "drag +x, lift +z, pitch axis +y; forceCoeffs on all patches named body*",
    ),
)

# -- rotating propeller / rotor in a rotating reference frame (MRF) --------------------------------
ROTOR_COMMON = (
    TemplateParameter("rpm", "rotor speed", "rpm"),
    TemplateParameter("diameter", "rotor diameter D (reference length; also scales domain and mesh)", "m"),
    TemplateParameter("kinematic_viscosity", "fluid kinematic viscosity", "m^2/s"),
    TemplateParameter("density", "fluid density (forces are reported dimensional)", "kg/m^3"),
    TemplateParameter("rotation", "sense of rotation about +x by the right-hand rule: 1 or -1", "", 1.0),
    TemplateParameter("center", "rotor centre; the axis is +x through this point (place the STL accordingly)", "m", [0.0, 0.0, 0.0], "vector"),
    TemplateParameter("iterations", "maximum SIMPLE iterations", "", 500, "int"),
    TemplateParameter("residual_target", "stop when all initial residuals are below", "", 1e-4),
    TemplateParameter("upstream", "domain inlet distance ahead of the rotor centre", "D", 3.0),
    TemplateParameter("downstream", "domain outlet distance behind the rotor centre", "D", 6.0),
    TemplateParameter("lateral", "domain half-width around the rotor centre", "D", 3.0),
    TemplateParameter("zone_radius", "MRF cell-zone radius", "D", 0.6),
    TemplateParameter("zone_length", "MRF cell-zone length along the axis", "D", 0.4),
    TemplateParameter("wake_length", "refined wake region behind the rotor", "D", 2.0),
    TemplateParameter("cells_per_diameter", "background cells per D", "", 8.0),
    TemplateParameter("surface_level", "snappy refinement level on the blades", "", 4, "int"),
    TemplateParameter("near_level", "refinement level in a box around the rotor", "", 3, "int"),
    TemplateParameter("rotor_level", "refinement level inside the MRF zone", "", 3, "int"),
    TemplateParameter("wake_level", "refinement level in the wake", "", 2, "int"),
    TemplateParameter("turbulence_intensity", "far-field turbulence intensity", "-", 0.01),
    TemplateParameter("viscosity_ratio", "far-field eddy/molecular viscosity ratio", "-", 10.0),
)
ROTOR_PATCHES = {
    "inlet": "fixed axial inflow: airspeed (rotor_mrf) or inflow_fraction x tip speed (rotor_mrf_static)",
    "outlet": "fixed pressure 0, inletOutlet velocity",
    "sides": "slip (four lateral faces)",
    "body*": "no-slip wall inside the rotating zone; forces patch",
}


def _rotor_derive(p: dict, bmin: np.ndarray, bmax: np.ndarray) -> dict[str, str]:
    """Domain, MRF cylinder and refinement regions around the rotor, sized in diameters; the rotor axis is
    +x through ``center`` (the STL must be placed on it)."""
    if p["rpm"] <= 0:
        raise ValueError("rpm must be > 0 (use rotation=-1 for the other sense of rotation)")
    if p["rotation"] not in (1.0, -1.0):
        raise ValueError("rotation must be 1 or -1")
    if p.get("airspeed", 1.0) <= 0:
        raise ValueError("airspeed must be > 0 (use rotor_mrf_static for a rotor without inflow)")
    D = p["diameter"]
    omega = p["rotation"] * p["rpm"] * 2 * math.pi / 60
    tip_speed = abs(omega) * D / 2
    airspeed = p["airspeed"] if "airspeed" in p else p["inflow_fraction"] * tip_speed   # static: a small residual inflow
    c = np.asarray(p["center"], dtype=float)
    if np.any(bmin > c + 0.1 * D) or np.any(bmax < c - 0.1 * D):
        raise ValueError(f"the rotor centre {c.tolist()} is not inside the STL bounding box {bmin.tolist()}..{bmax.tolist()}: "
                         "place the STL with its axis on +x through 'center' (default the origin)")
    lo = c - np.array([p["upstream"] * D, p["lateral"] * D, p["lateral"] * D])
    hi = c + np.array([p["downstream"] * D, p["lateral"] * D, p["lateral"] * D])
    h = D / p["cells_per_diameter"]
    n = np.maximum(1, np.ceil((hi - lo) / h)).astype(int)
    hi = lo + n * h
    near = 0.3 * D
    u_ref = max(airspeed, 0.1 * tip_speed)
    k = 1.5 * (u_ref * p["turbulence_intensity"]) ** 2
    out = {
        "XMIN": lo[0], "YMIN": lo[1], "ZMIN": lo[2], "XMAX": hi[0], "YMAX": hi[1], "ZMAX": hi[2],
        "NX": n[0], "NY": n[1], "NZ": n[2],
        "ROTOR_CENTER": c, "OMEGA": omega,
        "ZONE_P1": c - np.array([0.5 * p["zone_length"] * D, 0, 0]), "ZONE_P2": c + np.array([0.5 * p["zone_length"] * D, 0, 0]),
        "ZONE_RADIUS": p["zone_radius"] * D,
        "NEAR_MIN": bmin - near, "NEAR_MAX": bmax + near,
        "WAKE_MIN": np.array([bmin[0] - 0.2 * D, c[1] - 0.7 * D, c[2] - 0.7 * D]),
        "WAKE_MAX": np.minimum(np.array([c[0] + p["wake_length"] * D, c[1] + 0.7 * D, c[2] + 0.7 * D]), hi - h),
        "LOCATION_IN_MESH": lo + np.array([0.37 * p["upstream"] * D, 0.123 * (hi[1] - lo[1]), 0.211 * (hi[2] - lo[2])]),
        "SURFACE_LEVEL_MIN": p["surface_level"], "SURFACE_LEVEL_MAX": p["surface_level"],
        "FEATURE_LEVEL": p["surface_level"], "NEAR_LEVEL": p["near_level"], "ROTOR_LEVEL": p["rotor_level"],
        "WAKE_LEVEL": p["wake_level"],
        "AIRSPEED": airspeed, "KINEMATIC_VISCOSITY": p["kinematic_viscosity"], "DENSITY": p["density"],
        "ITERATIONS": p["iterations"], "RESIDUAL_TARGET": p["residual_target"],
        "K_INLET": k, "OMEGA_INLET": k / (p["kinematic_viscosity"] * p["viscosity_ratio"]),
    }
    return {key: _fmt(v) for key, v in out.items()}


ROTOR_NOTES = (
    "rotor axis +x through 'center' (default the origin); place the STL so its hub sits there",
    "thrust is the force on the rotor along -x (the rotor pushes fluid toward +x); torque about +x; both from the forces function object",
    "rotation=1 turns the rotor by the right-hand rule about +x; if the thrust comes out negative the blade pitch is handed the other way: use rotation=-1",
    "steady MRF: no blade-passing unsteadiness, no tip-vortex resolution; wall functions without prism layers — compare with blade element theory, expect tens of percent",
)

ROTOR_MRF = TemplateSpec(
    name="rotor_mrf",
    description="Propeller/rotor in axial inflow (along +x) in a rotating reference frame, steady k-omega SST.",
    parameters=(TemplateParameter("airspeed", "axial inflow speed along +x", "m/s"),) + ROTOR_COMMON,
    flavors=EXTERNAL_FLAVORS,
    derive=_rotor_derive,
    max_body_extent=1.5,
    patches=ROTOR_PATCHES,
    notes=ROTOR_NOTES + ("efficiency = thrust x airspeed / shaft power",),
)

ROTOR_MRF_STATIC = TemplateSpec(
    name="rotor_mrf_static",
    description="Propeller/rotor at static thrust (hover): rotating frame, steady k-omega SST, a small residual axial inflow for stability.",
    parameters=ROTOR_COMMON + (
        TemplateParameter("inflow_fraction", "residual axial inflow as a fraction of the tip speed (keeps the steady solution stable; "
                          "thrust changes by ~1 % at the default)", "-", 0.02),),
    flavors=EXTERNAL_FLAVORS,
    derive=_rotor_derive,
    max_body_extent=1.5,
    patches=ROTOR_PATCHES,
    notes=ROTOR_NOTES + ("figure of merit = ideal hover power / shaft power",
                         "open (total-pressure) far-field boundaries were tried and diverged with SIMPLE; the small fixed inflow is the stable choice"),
)

TEMPLATES: dict[str, TemplateSpec] = {t.name: t for t in (LAMINAR, RANS_KSST, ROTOR_MRF, ROTOR_MRF_STATIC)}
ALIASES = {"laminar_external_simplefoam": "laminar_external", "rans_ksst_external_simplefoam": "rans_ksst_external"}


def get_template(name: str | TemplateSpec) -> TemplateSpec:
    if isinstance(name, TemplateSpec):
        return name
    name = ALIASES.get(name, name)
    if name not in TEMPLATES:
        raise ValueError(f"unknown template {name!r}; available: {sorted(TEMPLATES)}")
    return TEMPLATES[name]


def list_templates() -> list[str]:
    return sorted(TEMPLATES)
