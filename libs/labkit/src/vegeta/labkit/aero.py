"""Aerodynamics as Yax part data (``vendor/yax/docs/engineering.md``, the ``00aero`` block) from what a design holds: a
set of non-dimensional derivatives and the reference geometry they are normalised with (the aero lab's ``*_design.json``,
``*_flight.derivatives``), a CFD campaign's coefficients, a section polar. Only assembly: no coefficient is computed here."""
from __future__ import annotations

#: A design's derivative names that are plain coefficients (the rest are derivatives).
COEFFICIENT_NAMES = ("CD", "CL", "CY", "Cl", "Cm", "Cn", "CD0", "CL0", "Cm0", "k", "e", "CLmax")


def block(*, name: str, method: str, reference: dict, derivatives: dict | None = None, coefficients: dict | None = None,
          tables: list | None = None, frame: str = "machine", axes: dict | None = None, reference_point_m=None,
          center_of_pressure_m=None, assumptions=None, source: str = "", validity: dict | None = None) -> dict:
    """An aerodynamics block. ``derivatives`` may hold plain coefficients too (``COEFFICIENT_NAMES``): they are sorted
    into ``coefficients``. Only finite numbers go in; a name that is not a number is an error, not a guess."""
    out = {"name": name, "method": method, "reference": dict(reference), "frame": frame}
    if axes:
        out["axes"] = dict(axes)
    for k, v in (("reference_point_m", reference_point_m), ("center_of_pressure_m", center_of_pressure_m)):
        if v is not None:
            out[k] = [float(x) for x in v]
    co, de = dict(coefficients or {}), {}
    for k, v in (derivatives or {}).items():
        if not isinstance(v, (int, float)):
            raise ValueError(f"derivative {k!r} is not a number: {v!r}")
        (co if k in COEFFICIENT_NAMES else de)[k] = float(v)
    if co:
        out["coefficients"] = co
    if de:
        out["derivatives"] = de
    if tables:
        out["tables"] = list(tables)
    if assumptions:
        out["assumptions"] = list(assumptions)
    if source:
        out["source"] = source
    if validity:
        out["validity"] = dict(validity)
    return out


def from_design_json(doc: dict, *, name: str, source: str, method: str = "analytic") -> dict:
    """The block of an aero-lab ``*_design.json``: its ``derivatives`` (body axes, per radian, rates with b/2V and c/2V)
    and the reference geometry of its ``layout`` (``S_ref`` [m²], ``mac`` and ``b2`` [mm])."""
    lay = doc["layout"]
    ref = {"area_m2": float(lay["S_ref"]), "chord_m": float(lay["mac"]) / 1000.0, "span_m": 2 * float(lay["b2"]) / 1000.0, "source": f"{source}: layout"}
    return block(name=name, method=method, reference=ref, derivatives=doc["derivatives"],
                 axes={"x": "forward", "y": "right", "z": "down", "note": "body axes; per radian; rates non-dimensionalised with b/2V and c/2V"},
                 source=f"{source}: derivatives (about the centre of gravity the design sized for)")
