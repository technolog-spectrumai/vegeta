"""Known-good OpenFOAM template cases and their specifications.

A template holds the engineering decisions (domain, patches, physics model, boundary conditions,
meshing and solver settings). Its :class:`TemplateSpec` states which values the engineer must
supply and which values the template decides (visible defaults that can be explicitly overridden).
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
class TemplateSpec:
    name: str
    description: str
    openfoam_flavor: str
    parameters: tuple[TemplateParameter, ...]
    pipeline: tuple[Step, ...]
    derive: Callable[[dict, np.ndarray, np.ndarray], dict[str, Any]]
    max_body_extent: float  # largest body dimension allowed, in multiples of reference_length
    patches: dict[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    @property
    def directory(self) -> Path:
        return TEMPLATE_ROOT / self.name

    @property
    def step_names(self) -> list[str]:
        return [s.name for s in self.pipeline]

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
        for positive in ("velocity", "kinematic_viscosity", "density", "reference_area", "reference_length"):
            if positive in out and out[positive] <= 0:
                raise ValueError(f"{positive} must be > 0")
        return out

    def describe(self) -> str:
        lines = [f"{self.name} ({self.openfoam_flavor})", f"  {self.description}", "  parameters:"]
        for p in self.parameters:
            d = "REQUIRED" if p.required else f"template default = {p.default}"
            lines.append(f"    {p.name:<22} [{p.units}] {d} — {p.description}")
        lines.append("  pipeline: " + " -> ".join(self.step_names))
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

EXTERNAL_PIPELINE = (
    Step("blockMesh", ("blockMesh",), description="background hex mesh of the domain"),
    Step("surfaceFeatureExtract", ("surfaceFeatureExtract",), description="feature edges of body.stl"),
    Step("snappyHexMesh", ("snappyHexMesh", "-overwrite"), description="castellate and snap around the body"),
    Step("checkMesh", ("checkMesh",), description="mesh quality report"),
    Step("restore0", ("copy", "0.orig", "0"), kind="internal", description="initial fields for the meshed case"),
    Step("simpleFoam", ("simpleFoam",), description="steady incompressible solver with forceCoeffs"),
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
    name="laminar_external_simplefoam",
    description="Steady laminar external flow around a single body in a box domain (flow +x, lift +z).",
    openfoam_flavor="openfoam.com v1912+",
    parameters=COMMON_REQUIRED + _mesh_and_domain(400),
    pipeline=EXTERNAL_PIPELINE,
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
    name="rans_ksst_external_simplefoam",
    description="Steady RANS k-omega SST external flow with wall functions (flow +x, lift +z).",
    openfoam_flavor="openfoam.com v1912+",
    parameters=COMMON_REQUIRED + _mesh_and_domain(600) + (
        TemplateParameter("turbulence_intensity", "inlet turbulence intensity", "-", 0.005),
        TemplateParameter("viscosity_ratio", "inlet eddy/molecular viscosity ratio", "-", 10.0),
    ),
    pipeline=EXTERNAL_PIPELINE,
    derive=_external_derive,
    max_body_extent=4.0,
    patches=EXTERNAL_PATCHES,
    notes=(
        "wall functions without prism layers: y+ is not controlled; use for trend comparison, not absolute drag",
        "drag +x, lift +z, pitch axis +y; forceCoeffs on all patches named body*",
    ),
)

TEMPLATES: dict[str, TemplateSpec] = {t.name: t for t in (LAMINAR, RANS_KSST)}


def get_template(name: str | TemplateSpec) -> TemplateSpec:
    if isinstance(name, TemplateSpec):
        return name
    if name not in TEMPLATES:
        raise ValueError(f"unknown template {name!r}; available: {sorted(TEMPLATES)}")
    return TEMPLATES[name]


def list_templates() -> list[str]:
    return sorted(TEMPLATES)
