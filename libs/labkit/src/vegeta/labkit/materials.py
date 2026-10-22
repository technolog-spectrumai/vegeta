"""Materials, strength and connections as Yax part data (``vendor/yax/docs/engineering.md``: the ``00mat``, ``00str``,
``00conn`` blocks) from what a design states: a Talos ``Material`` with the factors that bring it to SI, a design module's
density table, a fastener or pin the module names. Nothing is derived (no G from E and ν), nothing is sized here."""
from __future__ import annotations


def from_talos(material, *, name: str | None = None, stress_pa: float = 1.0, density_kg_m3: float = 1.0, cls: str | None = None) -> dict:
    """A catalogue entry from a Talos ``Material`` (isotropic, linear elastic). Talos holds the numbers in the model's
    unit system: ``stress_pa`` and ``density_kg_m3`` are what one of its stress and density units is in SI (1 for SI,
    1e6 and 1e12 for an mm–N–t model's MPa and t/mm³) — stated by the caller, never guessed."""
    out = {"name": name or material.name, "anisotropy": "isotropic", "youngs_modulus_pa": float(material.youngs_modulus) * stress_pa,
           "poisson_ratio": float(material.poissons_ratio)}
    if cls:
        out["class"] = cls
    if material.density is not None:
        out["density_kg_m3"] = float(material.density) * density_kg_m3
    if material.yield_strength is not None:
        out["yield_strength_pa"] = float(material.yield_strength) * stress_pa
    if getattr(material, "thermal_expansion", None) is not None:
        out["thermal"] = {"expansion_per_k": float(material.thermal_expansion), "reference_temperature_k": float(material.reference_temperature)}
    out["source"] = material.source or "talos.Material"
    return out


def from_densities(table: dict, *, kg_m3_per_unit: float, source: str, classes: dict | None = None) -> dict:
    """A catalogue of density-only entries from a design module's table (``{name: density}`` in the module's unit)."""
    out = {}
    for name, rho in table.items():
        e = {"name": name, "density_kg_m3": float(rho) * kg_m3_per_unit, "source": source}
        if classes and name in classes:
            e["class"] = classes[name]
        out[name] = e
    return out


def connection(name: str, type_: str, links: tuple, *, frame: str = "machine", pos=None, quat=None, axis=None, geometry: dict | None = None,
               fasteners: dict | None = None, material: str | None = None, limits: dict | None = None, detachable: bool | None = None,
               source: str = "") -> dict:
    """A connection entry: only what is given goes in."""
    out = {"name": name, "type": type_, "links": [str(links[0]), str(links[1])], "frame": frame}
    for k, v in (("pos", pos), ("quat", quat), ("axis", axis)):
        if v is not None:
            out[k] = [float(x) for x in v]
    for k, v in (("geometry", geometry), ("fasteners", fasteners), ("material", material), ("limits", limits), ("detachable", detachable)):
        if v is not None:
            out[k] = v
    if source:
        out["source"] = source
    return out


def axles(m, *, diameter_m: float, length_m: float | None = None, material: str | None = None, which=None, source: str = "") -> dict:
    """One pinned connection per joint of the machine that carries a wheel or roller (``which(joint) -> bool``, default:
    every revolute/continuous joint): the axle the design states, at the joint, along its axis."""
    out = {}
    for j in m.joints:
        if j.parent is None or j.kind not in ("revolute", "continuous"):
            continue
        if which is not None and not which(j):
            continue
        geo = {"diameter_m": float(diameter_m)}
        if length_m is not None:
            geo["length_m"] = float(length_m)
        out[f"axle:{j.name}"] = connection(f"axle:{j.name}", "pin", (j.parent, j.child), pos=j.pos, axis=j.axis, geometry=geo,
                                           material=material, source=source)
    return out
