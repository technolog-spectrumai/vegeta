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
    kind: str = "float"      # float | int | vector | table (rows of 3 numbers)

    @property
    def required(self) -> bool:
        return self.default is REQUIRED

    def validate(self, value: Any) -> Any:
        if self.kind == "table":
            arr = np.asarray(value, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 3 or len(arr) < 2 or not np.all(np.isfinite(arr)):
                raise ValueError(f"template parameter {self.name!r} needs at least 2 rows of 3 finite numbers")
            return arr.tolist()
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
    static_geometry: bool = False   # True: a second STL (a pod, an airframe) that stands still next to the rotor
    surfaces: tuple[str, ...] = ()  # further named STL surfaces the case needs (an engine intake, a casing...)

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
        for positive in ("velocity", "kinematic_viscosity", "density", "reference_area", "reference_length", "diameter", "rpm",
                         "pressure", "temperature"):
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

# -- a propeller installed on a pod / airframe: rotating blades (MRF) next to a standing body ------------------
def _installed_derive(p: dict, bmin: np.ndarray, bmax: np.ndarray) -> dict[str, str]:
    """``_rotor_derive`` for the propeller (``bmin``/``bmax``), then the domain grown to hold the standing body
    (``p["_static_bbox"]``, given by ``CFDCase.prepare``) with ``static_margin`` diameters around it, and a refinement
    box around that body."""
    out = _rotor_derive(p, bmin, bmax)
    if "_static_bbox" not in p:
        raise ValueError("rotor_mrf_installed needs the standing body: CFDCase(..., static_geometry=<STL>)")
    smin, smax = (np.asarray(v, dtype=float) for v in p["_static_bbox"])
    D, c = p["diameter"], np.asarray(p["center"], dtype=float)
    h = D / p["cells_per_diameter"]
    lo = np.minimum(c - np.array([p["upstream"] * D, p["lateral"] * D, p["lateral"] * D]), smin - p["static_margin"] * D)
    hi = np.maximum(c + np.array([p["downstream"] * D, p["lateral"] * D, p["lateral"] * D]), smax + p["static_margin"] * D)
    n = np.maximum(1, np.ceil((hi - lo) / h)).astype(int)
    hi = lo + n * h
    loc = lo + np.array([0.37 * p["upstream"] * D, 0.123 * (hi[1] - lo[1]), 0.211 * (hi[2] - lo[2])])
    if np.all(loc > smin) and np.all(loc < smax):
        raise ValueError("the mesh seed point falls inside the standing body's bounding box; enlarge 'lateral'")
    out.update({k: _fmt(v) for k, v in {
        "XMIN": lo[0], "YMIN": lo[1], "ZMIN": lo[2], "XMAX": hi[0], "YMAX": hi[1], "ZMAX": hi[2],
        "NX": n[0], "NY": n[1], "NZ": n[2], "LOCATION_IN_MESH": loc,
        "STATIC_BOX_MIN": smin - 0.1 * D, "STATIC_BOX_MAX": smax + 0.1 * D,
        "STATIC_LEVEL": p["static_level"], "STATIC_BOX_LEVEL": p["static_box_level"],
        "WAKE_MAX": np.minimum(np.array([c[0] + p["wake_length"] * D, c[1] + 0.7 * D, c[2] + 0.7 * D]), hi - h),
    }.items()})
    return out


ROTOR_MRF_INSTALLED = TemplateSpec(
    name="rotor_mrf_installed",
    description="A propeller installed on a standing body (a pod, a nacelle with its pylon, an airframe) in axial inflow "
                "(along +x): the blades in a rotating reference frame, the body standing still, steady k-omega SST; "
                "forces on the propeller and on the body separately (tractor: body behind the disc, pusher: ahead of it).",
    parameters=(TemplateParameter("airspeed", "axial inflow speed along +x (flight speed)", "m/s"),) + ROTOR_COMMON + (
        TemplateParameter("static_margin", "free space around the standing body (the domain grows to keep it)", "D", 1.5),
        TemplateParameter("static_level", "snappy refinement level on the standing body", "", 3, "int"),
        TemplateParameter("static_box_level", "refinement level in a box around the standing body", "", 1, "int"),
    ),
    flavors=EXTERNAL_FLAVORS,
    derive=_installed_derive,
    max_body_extent=1.5,
    patches=dict(ROTOR_PATCHES, **{"static*": "no-slip wall, not rotating even inside the MRF zone (nonRotatingPatches); "
                                              "its own forces function object (staticForces)"}),
    notes=ROTOR_NOTES + (
        "geometry = the propeller (with its spinner) on the axis +x through 'center'; static_geometry = the body",
        "leave a small gap between spinner and body, or let the body reach into the spinner: two surfaces must not touch face to face",
        "thrust/torque are the propeller's (forces); body_force_N is the standing body's (staticForces); net_thrust_N = thrust - body drag",
        "efficiency = thrust x airspeed / shaft power; net efficiency uses the net thrust",
        "steady MRF with a standing body: the blades are frozen at one angle relative to the body (frozen rotor); "
        "the unsteady blade-passing interaction needs a sliding mesh — read the body's wake from a body-alone run",
    ),
    static_geometry=True,
)

# -- a whole aircraft with its two propellers as rotor disks (blade-element source terms) ---------------
def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n == 0:
        raise ValueError("disk_axis must not be zero")
    return v / n


def _disks_derive(p: dict, bmin: np.ndarray, bmax: np.ndarray) -> dict[str, str]:
    """The external RANS case plus two disk cell zones and their rotor-disk sources (blade data and the
    section polar are written as tables)."""
    out = _external_derive(p, bmin, bmax)
    D, axis = p["diameter"], _unit(p["disk_axis"])
    ref = np.array([0.0, 0.0, 1.0]) - axis[2] * axis        # "up" in the disk plane: psi = 0 reference
    if np.linalg.norm(ref) < 1e-6:
        ref = np.array([0.0, 1.0, 0.0]) - axis[1] * axis
    ref = _unit(ref)
    lo = np.array([float(out["XMIN"]), float(out["YMIN"]), float(out["ZMIN"])])
    hi = np.array([float(out["XMAX"]), float(out["YMAX"]), float(out["ZMAX"])])
    half = 0.5 * p["disk_thickness"] * D * axis
    extra: dict[str, Any] = {}
    disks = [i for i in (1, 2) if f"disk{i}_center" in p]            # two disks (aircraft) or one (hull_rotor_disk)
    for i in disks:
        key = f"disk{i}_center"
        c = np.asarray(p[key], dtype=float)
        if np.any(c - D < lo) or np.any(c + D > hi):
            raise ValueError(f"{key} {c.tolist()} is not well inside the domain {lo.tolist()}..{hi.tolist()}")
        extra[f"DISK{i}_CENTER"], extra[f"DISK{i}_P1"], extra[f"DISK{i}_P2"] = c, c - half, c + half
        extra[f"DISK{i}_RPM"] = p["rpm"] * p[f"rotation{i}"]
    for i in disks:
        if p[f"rotation{i}"] not in (1.0, -1.0):
            raise ValueError(f"rotation{i} must be 1 or -1")
    blade = np.asarray(p["blade"], dtype=float)
    if np.any(np.diff(blade[:, 0]) <= 0) or blade[0, 0] <= 0 or blade[-1, 0] > 0.5 * D * 1.001:
        raise ValueError("blade rows are (radius [m], twist [deg], chord [m]) with increasing radius up to diameter / 2")
    if np.any(blade[:, 2] <= 0):
        raise ValueError("blade chord must be positive")
    polar = np.asarray(p["polar"], dtype=float)
    polar = polar[np.argsort(polar[:, 0])]
    if polar[0, 0] > -90 or polar[-1, 0] < 90:
        raise ValueError("polar rows are (alpha [deg], Cd, Cl) and must cover at least -90..90 deg")
    extra.update({
        "DISK_AXIS": axis, "REF_DIRECTION": ref, "DISK_ZONE_RADIUS": p["zone_radius"] * D,
        "DISK_LEVEL": p["disk_level"], "BLADES": p["blades"], "TIP_EFFECT": p["tip_effect"],
        "INLET_VELOCITY": np.array([p["velocity"], 0.0, 0.0]),
    })
    out.update({k: _fmt(v) for k, v in extra.items()})
    out["BLADE_DATA"] = "\n".join(f"            (section ({_fmt(r)} {_fmt(t)} {_fmt(c)}))" for r, t, c in blade)
    out["PROFILE_DATA"] = "\n".join(f"                ({_fmt(a)} {_fmt(cd)} {_fmt(cl)})" for a, cd, cl in polar)
    return out


AIRCRAFT_ROTOR_DISKS = TemplateSpec(
    name="aircraft_rotor_disks",
    description="Whole aircraft in steady RANS k-omega SST flow (flow +x, lift +z) with two propellers as rotor "
                "disks: blade-element source terms from the blade geometry and section polar (e.g. Boreas).",
    parameters=COMMON_REQUIRED + _mesh_and_domain(800) + (
        TemplateParameter("turbulence_intensity", "inlet turbulence intensity", "-", 0.005),
        TemplateParameter("viscosity_ratio", "inlet eddy/molecular viscosity ratio", "-", 10.0),
        TemplateParameter("disk1_center", "centre of the left propeller disk (in the flow frame, metres)", "m", kind="vector"),
        TemplateParameter("disk2_center", "centre of the right propeller disk", "m", kind="vector"),
        TemplateParameter("disk_axis", "thrust direction of both disks (usually about -x: upstream)", "-", kind="vector"),
        TemplateParameter("diameter", "propeller diameter", "m"),
        TemplateParameter("rpm", "propeller speed", "rpm"),
        TemplateParameter("blades", "blades per propeller", "", kind="int"),
        TemplateParameter("blade", "blade rows (radius [m], twist [deg], chord [m]), hub to tip", "", kind="table"),
        TemplateParameter("polar", "section polar rows (alpha [deg], Cd, Cl), covering -180..180 deg", "", kind="table"),
        TemplateParameter("rotation1", "left disk sense of rotation about disk_axis: 1 or -1", "", 1.0),
        TemplateParameter("rotation2", "right disk sense of rotation: -1 counter-rotates", "", -1.0),
        TemplateParameter("tip_effect", "normalised radius above which the blades carry no lift (tip loss)", "-", 0.97),
        TemplateParameter("disk_thickness", "thickness of each disk cell zone", "D", 0.08),
        TemplateParameter("zone_radius", "radius of each disk cell zone", "D", 0.52),
        TemplateParameter("disk_level", "refinement level inside the disk zones", "", 5, "int"),
    ),
    flavors=EXTERNAL_FLAVORS,
    derive=_disks_derive,
    max_body_extent=6.0,
    patches=EXTERNAL_PATCHES,
    notes=(
        "propellers are rotor disks (momentum and swirl from blade elements), not resolved rotating blades",
        "forceCoeffs are on the airframe (body*) only: the airframe sees the slipstream; propeller thrust is "
        "the disks' (printed by the rotorDisk source in the solver log)",
        "sign check: with a correct setup the disks accelerate the flow (+x) behind them; if not, flip disk_axis",
        "wall functions without prism layers: y+ is not controlled; use for trend comparison, not absolute drag",
    ),
)

_ONE_DISK_DROP = ("disk2_center", "rotation2")
HULL_ROTOR_DISK = TemplateSpec(
    name="hull_rotor_disk",
    description="A hull (submarine, boat double body, any body) in steady RANS k-omega SST flow (flow +x) with one "
                "propeller as a rotor disk: blade-element source terms from the blade geometry and section polar.",
    parameters=tuple(
        (TemplateParameter("disk1_center", "centre of the propeller disk (in the flow frame, metres)", "m", kind="vector")
         if t.name == "disk1_center" else
         TemplateParameter("rotation1", "sense of rotation about disk_axis: 1 or -1", "", 1.0) if t.name == "rotation1" else t)
        for t in AIRCRAFT_ROTOR_DISKS.parameters if t.name not in _ONE_DISK_DROP),
    flavors=EXTERNAL_FLAVORS,
    derive=_disks_derive,
    max_body_extent=6.0,
    patches=EXTERNAL_PATCHES,
    notes=(
        "the propeller is a rotor disk (momentum and swirl from blade elements), not resolved rotating blades",
        "forceCoeffs are on the hull (body*) only: the hull sees the propeller's inflow; the thrust is the disk's "
        "(printed by the rotorDisk source in the solver log)",
        "sign check: the flow behind the disk must be faster than around it; if not, flip disk_axis",
        "wall functions without prism layers: y+ is not controlled; use for trend comparison, not absolute drag",
    ),
)

# -- a suction hood under a vehicle floor (a street sweeper's vacuum nozzle) --------------------------------------
SUCTION_PATCHES = {
    "ground": "no-slip wall (moving at -ground_speed along x: the road passing under the vehicle)",
    "floor": "no-slip wall (the vehicle's floor, the domain top outside the duct)",
    "suction": "fixed velocity out of the domain, +z (flow_rate over the duct's inner section), zero-gradient pressure",
    "sides": "fixed pressure 0 with pressureInletOutletVelocity (the atmosphere under the vehicle's edges)",
    "body*": "no-slip wall (the hood, duct, broom...; surfaceFieldValue patch)",
}


def _suction_derive(p: dict, bmin: np.ndarray, bmax: np.ndarray) -> dict[str, str]:
    """A 3 x 3 block domain between the road and the floor: the middle block is the duct's column (its top face
    the suction patch), the eight others end on the floor; refinement boxes around the body and in the gap under
    it."""
    L, H = p["reference_length"], p["floor_height"]
    di, dw = p["duct_inner"], p["duct_wall"]
    cx, cy = float(p["duct_center"][0]), float(p["duct_center"][1])
    if bmax[2] <= H:
        raise ValueError(f"the STL must carry the duct through the floor (floor_height {H} m): its top is at {bmax[2]:.4g} m")
    if bmin[2] < -1e-6:
        raise ValueError(f"the STL goes below the road (z = 0): its bottom is at {bmin[2]:.4g} m")
    if H <= 0 or di <= 0 or dw <= 0:
        raise ValueError("floor_height, duct_inner and duct_wall must be > 0")
    half = (di + dw) / 2                                    # the block boundary runs through the duct's wall
    lo = np.array([bmin[0] - p["margin_ahead"] * L, bmin[1] - p["margin_aside"] * L])
    hi = np.array([bmax[0] + p["margin_behind"] * L, bmax[1] + p["margin_aside"] * L])
    x1, x2, y1, y2 = cx - half, cx + half, cy - half, cy + half
    if not (lo[0] < x1 and x2 < hi[0] and lo[1] < y1 and y2 < hi[1]):
        raise ValueError("the duct column must lie inside the domain: check duct_center against the STL bounding box")
    h = L / p["cells_per_length"]
    n = lambda a, b: int(max(1, math.ceil((b - a) / h)))        # noqa: E731
    gap_top = min(H, bmin[2] + 0.5 * (bmax[2] - bmin[2]))
    near = 0.2 * L
    out = {
        "X0": lo[0], "X1": x1, "X2": x2, "X3": hi[0], "Y0": lo[1], "Y1": y1, "Y2": y2, "Y3": hi[1], "ZTOP": H,
        "NX0": n(lo[0], x1), "NX1": max(2, n(x1, x2)), "NX2": n(x2, hi[0]),
        "NY0": n(lo[1], y1), "NY1": max(2, n(y1, y2)), "NY2": n(y2, hi[1]), "NZ": n(0.0, H),
        "NEAR_MIN": np.array([bmin[0] - near, bmin[1] - near, 0.0]), "NEAR_MAX": np.array([bmax[0] + near, bmax[1] + near, H]),
        "GAP_MIN": np.array([bmin[0] - 0.05 * L, bmin[1] - 0.05 * L, 0.0]),
        "GAP_MAX": np.array([bmax[0] + 0.05 * L, bmax[1] + 0.05 * L, gap_top]),
        "LOCATION_IN_MESH": np.array([lo[0] + 0.5 * (bmin[0] - lo[0]), lo[1] + 0.37 * (bmin[1] - lo[1]), 0.5 * H]),
        "SURFACE_LEVEL_MIN": p["surface_level"], "SURFACE_LEVEL_MAX": p["surface_level"],
        "FEATURE_LEVEL": p["surface_level"], "NEAR_LEVEL": p["near_level"], "GAP_LEVEL": p["gap_level"],
        "SUCTION_W": p["flow_rate"] / (di * di), "GROUND_U": -p["ground_speed"],
        "KINEMATIC_VISCOSITY": p["kinematic_viscosity"], "DENSITY": p["density"],
        "ITERATIONS": p["iterations"], "RESIDUAL_TARGET": p["residual_target"],
    }
    u_ref = out["SUCTION_W"]
    k = 1.5 * (u_ref * p["turbulence_intensity"]) ** 2
    out["K_INLET"] = k
    out["OMEGA_INLET"] = k / (p["kinematic_viscosity"] * p["viscosity_ratio"])
    return {key: _fmt(v) for key, v in out.items()}


SUCTION_HOOD = TemplateSpec(
    name="suction_hood",
    description="A suction hood under a vehicle floor (a sweeper's vacuum nozzle): the road and the floor are walls, "
                "a square duct draws flow_rate out through the floor, the sides are the atmosphere; steady k-omega SST.",
    parameters=(
        TemplateParameter("flow_rate", "volume flow the fan draws through the duct", "m^3/s"),
        TemplateParameter("duct_center", "the duct's axis (x, y; z ignored): the STL's duct must run through the floor here", "m", kind="vector"),
        TemplateParameter("duct_inner", "the square duct's inner width (the suction patch is this square)", "m"),
        TemplateParameter("duct_wall", "the duct's wall thickness (the mesh blocks meet inside it)", "m"),
        TemplateParameter("floor_height", "the vehicle floor over the road: the domain's top", "m"),
        TemplateParameter("kinematic_viscosity", "fluid kinematic viscosity", "m^2/s"),
        TemplateParameter("density", "fluid density (pressures are reported in Pa)", "kg/m^3"),
        TemplateParameter("reference_length", "the hood's width: scales the domain margins and the background mesh", "m"),
        TemplateParameter("ground_speed", "the vehicle's speed over the road (the road moves at -x under it)", "m/s", 0.0),
        TemplateParameter("iterations", "maximum SIMPLE iterations", "", 500, "int"),
        TemplateParameter("residual_target", "stop when all initial residuals are below", "", 1e-4),
        TemplateParameter("margin_ahead", "domain edge ahead of the body (-x)", "L_ref", 2.0),
        TemplateParameter("margin_behind", "domain edge behind the body (+x)", "L_ref", 2.0),
        TemplateParameter("margin_aside", "domain edge beside the body", "L_ref", 2.0),
        TemplateParameter("cells_per_length", "background cells per L_ref", "", 12.0),
        TemplateParameter("surface_level", "snappy refinement level at the body", "", 3, "int"),
        TemplateParameter("near_level", "refinement level in a box around the body, road to floor", "", 2, "int"),
        TemplateParameter("gap_level", "refinement level in the gap under the body (road to half its height)", "", 3, "int"),
        TemplateParameter("turbulence_intensity", "turbulence intensity of the air drawn in", "-", 0.05),
        TemplateParameter("viscosity_ratio", "eddy/molecular viscosity ratio of the air drawn in", "-", 10.0),
    ),
    flavors=EXTERNAL_FLAVORS,
    derive=_suction_derive,
    max_body_extent=6.0,
    patches=SUCTION_PATCHES,
    notes=(
        "place the STL on the road: z = 0 is the road, the hood's lips just above it, the duct rising through the floor (above floor_height)",
        "the duct's inner square must be centred on duct_center with the wall of duct_wall around it: the block boundary runs through that wall",
        "no boundary-layer prism layers; the gap flow under the lips is resolved by the gap box's refinement level",
        "metrics: the depression at the duct (fan static pressure), the flow drawn (a check of the suction velocity), the air power Q x dp",
    ),
)

# -- compressible cases (openfoam.com, rhoSimpleFoam): a jet aircraft and a radial compressor ---------------------
GAS_CONSTANT = 287.05
COMPRESSIBLE_FLAVORS = (
    Flavor("openfoam.com", "com", "constant/triSurface", tuple(
        Step(s.name, ("rhoSimpleFoam",), description="steady compressible solver (SIMPLE)") if s.name == "solver" else s
        for s in COM_PIPELINE), "v2106 and later (surfaceFieldValue 'names')"),
)


def _flag(v) -> str:
    return "yes" if float(v) else "no"


def _surface(p: dict, name: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Bounding box and area of a named extra surface (given by ``CFDCase.prepare``)."""
    if "_surface_bboxes" not in p or name not in p["_surface_bboxes"]:
        raise ValueError(f"this template needs the surface {name!r}: CFDCase(..., surfaces={{{name!r}: <STL>, ...}})")
    lo, hi = (np.asarray(v, dtype=float) for v in p["_surface_bboxes"][name])
    return lo, hi, float(p["_surface_areas"][name])


def _jet_derive(p: dict, bmin: np.ndarray, bmax: np.ndarray) -> dict[str, str]:
    """``_external_derive`` with the density of the free stream, the engine's flows at the intake and nozzle faces,
    refinement around both faces and along the hot jet, and the sampling line down the plume."""
    for k in ("pressure", "temperature", "exhaust_temperature", "intake_mass_flow", "exhaust_mass_flow"):
        if p[k] <= 0:
            raise ValueError(f"{k} must be > 0")
    if p["exhaust_mass_flow"] < p["intake_mass_flow"]:
        raise ValueError("exhaust_mass_flow is the air plus the fuel: it cannot be below intake_mass_flow")
    rho = p["pressure"] / (GAS_CONSTANT * p["temperature"])
    q = dict(p, density=rho)
    out = _external_derive(q, bmin, bmax)
    L = p["reference_length"]
    ilo, ihi, _ = _surface(p, "intake")
    elo, ehi, e_area = _surface(p, "exhaust")
    lo = np.array([float(out[k]) for k in ("XMIN", "YMIN", "ZMIN")])
    hi = np.array([float(out[k]) for k in ("XMAX", "YMAX", "ZMAX")])
    h = L / p["cells_per_length"]
    for name, (a, b) in (("intake", (ilo, ihi)), ("exhaust", (elo, ehi))):
        if np.any(a < bmin - 0.05 * L) or np.any(b > bmax + 0.05 * L):
            raise ValueError(f"the {name} face lies outside the body's bounding box: it must close an opening of the body")
    d_jet = math.sqrt(4 * e_area / math.pi)
    centre = 0.5 * (elo + ehi)
    reach = p["plume_length"] * L
    half = 0.15 * L + d_jet
    start = centre + np.array([0.5 * d_jet, 0.0, 0.0])
    end = np.array([min(centre[0] + reach, hi[0] - h), centre[1], centre[2]])
    if end[0] <= start[0]:
        raise ValueError("the domain ends at the nozzle: raise downstream")
    margin = 0.1 * L
    out.update({k: _fmt(v) for k, v in {
        "PRESSURE": p["pressure"], "TEMPERATURE": p["temperature"], "DENSITY": rho,
        "INTAKE_MASS_FLOW": p["intake_mass_flow"], "EXHAUST_MASS_FLOW": p["exhaust_mass_flow"],
        "EXHAUST_TEMPERATURE": p["exhaust_temperature"],
        "EXHAUST_DENSITY": p["pressure"] / (GAS_CONSTANT * p["exhaust_temperature"]),
        "JET_INTENSITY": p["jet_intensity"], "JET_MIXING_LENGTH": 0.07 * d_jet,
        "ENGINE_MIN": np.minimum(ilo, elo) - margin, "ENGINE_MAX": np.maximum(ihi, ehi) + margin,
        "ENGINE_LEVEL": p["engine_level"], "PLUME_LEVEL": p["plume_level"],
        "PLUME_MIN": np.array([elo[0] - margin, centre[1] - half, centre[2] - half]),
        "PLUME_MAX": np.minimum(np.array([centre[0] + reach, centre[1] + half, centre[2] + half]), hi - h),
        "PLUME_START": start, "PLUME_END": end, "PLUME_POINTS": p["plume_points"],
        "K_INLET": max(float(out["K_INLET"]), 1e-8),
    }.items()})
    out["TRANSONIC"] = _flag(p["transonic"])
    return out


JET_EXTERNAL = TemplateSpec(
    name="jet_external",
    description="An aircraft with a running jet engine in steady compressible RANS k-omega SST (rhoSimpleFoam, flow +x, "
                "lift +z): the engine draws its air through the intake face and blows the hot jet out of the nozzle "
                "face; a passive tracer follows the exhaust (dilution behind the aircraft).",
    parameters=(
        TemplateParameter("velocity", "flight speed along +x", "m/s"),
        TemplateParameter("pressure", "ambient static pressure (absolute)", "Pa"),
        TemplateParameter("temperature", "ambient static temperature", "K"),
        TemplateParameter("reference_area", "reference area A_ref for coefficients (the wing area)", "m^2"),
        TemplateParameter("reference_length", "reference length for Cm; also scales domain and mesh (span / 4 for an aircraft)", "m"),
        TemplateParameter("center_of_rotation", "moment reference point", "m", kind="vector"),
        TemplateParameter("intake_mass_flow", "air the engine swallows through the intake face", "kg/s"),
        TemplateParameter("exhaust_mass_flow", "gas leaving the nozzle face: air plus fuel", "kg/s"),
        TemplateParameter("exhaust_temperature", "static temperature of the jet at the nozzle face (cycle model)", "K"),
        TemplateParameter("kinematic_viscosity", "only for the Reynolds number report (the solver uses Sutherland's law)", "m^2/s", 1.5e-5),
    ) + _mesh_and_domain(1500) + (
        TemplateParameter("turbulence_intensity", "inlet turbulence intensity", "-", 0.005),
        TemplateParameter("viscosity_ratio", "inlet eddy/molecular viscosity ratio", "-", 10.0),
        TemplateParameter("jet_intensity", "turbulence intensity of the jet at the nozzle", "-", 0.05),
        TemplateParameter("engine_level", "refinement level on the intake and nozzle faces and around them", "", 5, "int"),
        TemplateParameter("plume_level", "refinement level in the box along the jet", "", 3, "int"),
        TemplateParameter("plume_length", "length of the refined, sampled jet behind the nozzle", "L_ref", 4.0),
        TemplateParameter("plume_points", "sample points along the jet's axis", "", 200, "int"),
        TemplateParameter("transonic", "1: transonic pressure formulation (above about Mach 0.6 anywhere); 0: off", "-", 0.0),
    ),
    flavors=COMPRESSIBLE_FLAVORS,
    derive=_jet_derive,
    max_body_extent=6.0,
    patches=dict(EXTERNAL_PATCHES, **{
        "intake*": "the engine face: the engine's air leaves the domain (flowRateOutletVelocity, the intake mass flow)",
        "exhaust*": "the nozzle face: the hot jet enters (flowRateInletVelocity with the exhaust mass flow, fixed temperature, tracer = 1)"}),
    notes=(
        "three STLs, one closed surface together: the body (airframe and nacelle, with the intake and the nozzle open), "
        "the intake face and the nozzle exit face (surfaces={'intake': ..., 'exhaust': ...})",
        "forceCoeffs are on the body only, with the pressure relative to the ambient: the airframe and nacelle drag with "
        "the engine running (inlet spillage, the jet's suction on the afterbody); the engine's thrust is the cycle's",
        "the jet is air with a hot temperature (no fuel species, one Cp): its temperature and the 'exhaust' tracer along "
        "the sampled line show how fast the jet cools and dilutes",
        "compressible with wall functions and no prism layers: drag for trends, not to the last percent",
        "openfoam.com only (rhoSimpleFoam)",
    ),
    surfaces=("intake", "exhaust"),
)


def _compressor_derive(p: dict, bmin: np.ndarray, bmax: np.ndarray) -> dict[str, str]:
    """A closed background box around the casing; the MRF cylinder around the impeller (``bmin``/``bmax``); the inlet
    turbulence from the inlet face's size."""
    if p["rpm"] <= 0:
        raise ValueError("rpm must be > 0 (use rotation=-1 for the other sense of rotation)")
    if p["rotation"] not in (1.0, -1.0):
        raise ValueError("rotation must be 1 or -1")
    for k in ("inlet_total_pressure", "inlet_total_temperature", "outlet_pressure"):
        if p[k] <= 0:
            raise ValueError(f"{k} must be > 0 (absolute)")
    D = p["diameter"]
    c = np.asarray(p["center"], dtype=float)
    if np.any(bmin > c + 0.1 * D) or np.any(bmax < c - 0.1 * D):
        raise ValueError(f"the impeller axis point {c.tolist()} is not inside the impeller's bounding box: the axis is +x through 'center'")
    boxes = [(bmin, bmax)] + [_surface(p, n)[:2] for n in ("shroud", "inlet", "outlet")]
    alo = np.min([b[0] for b in boxes], axis=0)
    ahi = np.max([b[1] for b in boxes], axis=0)
    loc = np.asarray(p["location_in_mesh"], dtype=float)
    if np.any(loc <= alo) or np.any(loc >= ahi):
        raise ValueError("location_in_mesh must lie in the flow passage, inside the casing's bounding box")
    h = D / p["cells_per_diameter"]
    lo = alo - 0.1 * D
    n = np.maximum(1, np.ceil((ahi + 0.1 * D - lo) / h)).astype(int)
    hi = lo + n * h
    omega = p["rotation"] * p["rpm"] * 2 * math.pi / 60
    tip = abs(omega) * D / 2
    m = p["zone_margin"] * D
    axial = 0.1 * tip
    _, _, a_in = _surface(p, "inlet")
    mix = 0.07 * math.sqrt(4 * a_in / math.pi)
    k = 1.5 * (axial * p["turbulence_intensity"]) ** 2
    out = {
        "XMIN": lo[0], "YMIN": lo[1], "ZMIN": lo[2], "XMAX": hi[0], "YMAX": hi[1], "ZMAX": hi[2],
        "NX": n[0], "NY": n[1], "NZ": n[2],
        "ROTOR_CENTER": c, "OMEGA": omega,
        "ZONE_P1": np.array([bmin[0] - m, c[1], c[2]]), "ZONE_P2": np.array([bmax[0] + m, c[1], c[2]]),
        "ZONE_RADIUS": 0.5 * D + m,
        "LOCATION_IN_MESH": loc,
        "SURFACE_LEVEL": p["surface_level"], "SHROUD_LEVEL": p["shroud_level"], "PORT_LEVEL": p["port_level"],
        "ROTOR_LEVEL": p["rotor_level"],
        "INLET_TOTAL_PRESSURE": p["inlet_total_pressure"], "INLET_TOTAL_TEMPERATURE": p["inlet_total_temperature"],
        "OUTLET_PRESSURE": p["outlet_pressure"], "AXIAL_GUESS": axial,
        "TURBULENCE_INTENSITY": p["turbulence_intensity"], "MIXING_LENGTH": mix,
        "K_INLET": k, "OMEGA_INLET": math.sqrt(k) / (0.09 ** 0.25 * mix),
        "ITERATIONS": p["iterations"], "RESIDUAL_TARGET": p["residual_target"],
    }
    out = {key: _fmt(v) for key, v in out.items()}
    out["TRANSONIC"] = _flag(p["transonic"])
    out["U_SCHEME"] = "upwind" if float(p["first_order"]) else "linearUpwind limited"
    return out


COMPRESSOR_MRF = TemplateSpec(
    name="compressor_mrf",
    description="A radial compressor impeller in its casing (inlet duct, shroud, vaneless diffuser) in a rotating "
                "reference frame, steady compressible k-omega SST (rhoSimpleFoam): total pressure and temperature at the "
                "inlet face, a static back pressure at the outlet face. One back pressure = one point of a speed line.",
    parameters=(
        TemplateParameter("rpm", "impeller speed", "rpm"),
        TemplateParameter("diameter", "impeller tip (exducer) diameter: scales the mesh", "m"),
        TemplateParameter("inlet_total_pressure", "total pressure at the inlet face (absolute)", "Pa"),
        TemplateParameter("inlet_total_temperature", "total temperature at the inlet face", "K"),
        TemplateParameter("outlet_pressure", "static back pressure at the outlet face (absolute): sets the flow", "Pa"),
        TemplateParameter("location_in_mesh", "a point in the flow passage, inside the casing and outside the impeller's metal", "m", kind="vector"),
        TemplateParameter("rotation", "sense of rotation about +x by the right-hand rule: 1 or -1", "", 1.0),
        TemplateParameter("center", "a point on the impeller axis (+x through it)", "m", [0.0, 0.0, 0.0], "vector"),
        TemplateParameter("kinematic_viscosity", "only for the Reynolds number report (the solver uses Sutherland's law)", "m^2/s", 1.5e-5),
        TemplateParameter("iterations", "maximum SIMPLE iterations", "", 3000, "int"),
        TemplateParameter("residual_target", "stop when all initial residuals are below", "", 1e-5),
        TemplateParameter("cells_per_diameter", "background cells per impeller diameter", "", 10.0),
        TemplateParameter("surface_level", "snappy refinement level on the impeller", "", 4, "int"),
        TemplateParameter("shroud_level", "snappy refinement level on the casing (resolves the tip gap)", "", 4, "int"),
        TemplateParameter("port_level", "refinement level on the inlet and outlet faces", "", 2, "int"),
        TemplateParameter("rotor_level", "refinement level inside the MRF zone", "", 3, "int"),
        TemplateParameter("zone_margin", "MRF cylinder beyond the impeller (stay inside the casing's tip gap)", "D", 0.002),
        TemplateParameter("turbulence_intensity", "inlet turbulence intensity", "-", 0.03),
        TemplateParameter("transonic", "1: transonic pressure formulation; 0: off (start with 0)", "-", 0.0),
        TemplateParameter("first_order", "1: first-order upwind momentum (robust start); 0: linearUpwind", "-", 1.0),
    ),
    flavors=COMPRESSIBLE_FLAVORS,
    derive=_compressor_derive,
    max_body_extent=2.0,
    patches={
        "inlet*": "total pressure and total temperature (pressureInletOutletVelocity)",
        "outlet*": "fixed static back pressure, inletOutlet velocity and temperature",
        "body*": "the impeller: no-slip wall turning with the MRF zone; forces patch (torque -> shaft power)",
        "shroud*": "the casing, the static hub after the impeller, the diffuser walls: no-slip, not rotating",
        "farfield": "the background box (not part of the flow when the casing is closed)",
    },
    notes=(
        "four STLs forming one closed passage: the impeller (geometry), the casing (shroud), the inlet face and the "
        "outlet face (surfaces={'shroud': ..., 'inlet': ..., 'outlet': ...}); location_in_mesh in the passage",
        "the MRF cylinder must hold the impeller and stay off the casing: keep zone_margin below the tip gap / D",
        "steady MRF (frozen rotor), wall functions without prism layers: pressure ratio and efficiency within ~5-10 % "
        "of a test rig; the speed line's choke end is reliable, surge is not (an unsteady phenomenon)",
        "start a speed line near choke (low back pressure) with first_order=1, then raise the back pressure",
        "openfoam.com only (rhoSimpleFoam)",
    ),
    surfaces=("shroud", "inlet", "outlet"),
)

TEMPLATES: dict[str, TemplateSpec] = {t.name: t for t in (LAMINAR, RANS_KSST, ROTOR_MRF, ROTOR_MRF_STATIC, ROTOR_MRF_INSTALLED,
                                                         AIRCRAFT_ROTOR_DISKS, HULL_ROTOR_DISK, SUCTION_HOOD,
                                                         JET_EXTERNAL, COMPRESSOR_MRF)}
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
