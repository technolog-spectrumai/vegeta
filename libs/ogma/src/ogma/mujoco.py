"""og.mujoco — Ogma's MuJoCo strategy: parts → MJCF fragments.

A strategy only translates: the masses and inertias are ``og.mass``'s (its solid shapes use MuJoCo's own
``inertiafromgeom`` formulas, so a geom and its budget item agree). The strings need no MuJoCo; only ``compile_check``
imports it (the optional ``mujoco`` extra). Contact, servo and integration policy are not here — they are the
simulator's (Chiron's).

- ``geom(shape, name, pos, kg=None, **attrs)``: a ``<geom/>`` for Box, Sphere, Cylinder, Capsule, Ellipsoid (MuJoCo's
  half sizes; cylinders and capsules along z, turned by ``quat`` for an x or y axis).
- ``inertial(props)``: an explicit ``<inertial/>`` (mass, COM, full tensor) from a ``MassProperties`` — a budget's,
  or a CAD solid's from ``og.cad.properties``.
- ``body(name, budget, pos)``: a ``<body>`` carrying the budget's inertial and visual geoms for its shaped items.
- ``model(*bodies)`` / ``compile_check(xml)``: a minimal document, and what MuJoCo makes of it (mass, COM, principal
  inertias per body).
"""
from __future__ import annotations

import math

import numpy as np

from . import mass as om

_H = math.sqrt(0.5)
_QUAT = {"z": (1.0, 0.0, 0.0, 0.0), "x": (_H, 0.0, _H, 0.0), "y": (_H, -_H, 0.0, 0.0)}     # z → the axis


def _f(v) -> str:
    return " ".join(f"{float(x):.12g}" for x in np.ravel(v))


def mjcf_type_size(shape) -> tuple:
    """(MuJoCo type, size, quat) of an ``og.mass`` solid shape."""
    if isinstance(shape, om.Box):
        return "box", (shape.sx / 2, shape.sy / 2, shape.sz / 2), _QUAT["z"]
    if isinstance(shape, om.Sphere):
        return "sphere", (shape.radius,), _QUAT["z"]
    if isinstance(shape, om.Cylinder):
        return "cylinder", (shape.radius, shape.length / 2), _QUAT[shape.axis]
    if isinstance(shape, om.Capsule):
        return "capsule", (shape.radius, shape.length / 2), _QUAT[shape.axis]
    if isinstance(shape, om.Ellipsoid):
        return "ellipsoid", (shape.a, shape.b, shape.c), _QUAT["z"]
    raise NotImplementedError(f"og.mujoco: no MuJoCo primitive for {type(shape).__name__} (thin shapes are budget "
                              "idealisations: give the body an explicit inertial instead)")


def geom(shape, name: str, pos=(0.0, 0.0, 0.0), kg: float | None = None, **attrs) -> str:
    """A ``<geom/>`` of ``shape`` at ``pos`` [m]; ``kg`` sets its mass (MuJoCo derives the density); extra
    attributes (``rgba``, ``contype``, ``group`` …) are written as given."""
    typ, size, quat = mjcf_type_size(shape)
    a = [f'name="{name}"', f'type="{typ}"', f'size="{_f(size)}"', f'pos="{_f(pos)}"']
    if quat != _QUAT["z"]:
        a.append(f'quat="{_f(quat)}"')
    if kg is not None:
        a.append(f'mass="{float(kg):.12g}"')
    a += [f'{k}="{_f(v) if not isinstance(v, str) else v}"' for k, v in attrs.items()]
    return "<geom " + " ".join(a) + "/>"


def inertial(props: om.MassProperties) -> str:
    """An explicit ``<inertial/>``: mass, COM (in the body frame) and the full tensor about it (products as tensor
    entries, MuJoCo's ``fullinertia`` order Ixx Iyy Izz Ixy Ixz Iyz)."""
    I = np.asarray(props.inertia, float)
    full = (I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2])
    return f'<inertial pos="{_f(props.com)}" mass="{props.mass:.12g}" fullinertia="{_f(full)}"/>'


def body(name: str, budget: om.Budget, pos=(0.0, 0.0, 0.0), *, free: bool = True, rgba=(0.6, 0.6, 0.7, 1.0)) -> str:
    """A ``<body>`` whose inertial is the budget's (items' positions in the body frame) with visual, non-colliding
    geoms for the items that have a MuJoCo shape (thin shapes and point masses count in the inertial only)."""
    lines = [f'<body name="{name}" pos="{_f(pos)}">']
    if free:
        lines.append(f'  <freejoint name="{name}_free"/>')
    lines.append("  " + inertial(budget.properties()))
    for i, it in enumerate(budget.items):
        try:
            g = geom(it.shape, f"{name}_g{i}", it.pos, contype=0, conaffinity=0, group=1, rgba=rgba)
        except NotImplementedError:
            continue
        lines.append("  " + g)
    lines.append("</body>")
    return "\n".join(lines)


def model(*bodies: str, name: str = "ogma") -> str:
    """A minimal MJCF document holding ``bodies`` in the world (no gravity-dependent defaults changed)."""
    inner = "\n".join("    " + line for b in bodies for line in b.splitlines())
    return f'<mujoco model="{name}">\n  <worldbody>\n{inner}\n  </worldbody>\n</mujoco>'


def compile_check(xml: str) -> dict:
    """What MuJoCo makes of ``xml``: per body (world left out) its mass, COM in the body frame and principal
    inertias (MuJoCo's ``body_inertia``, about ``body_iquat``'s axes)."""
    import mujoco

    m = mujoco.MjModel.from_xml_string(xml)
    out = {}
    for b in range(1, m.nbody):
        out[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)] = {
            "mass": float(m.body_mass[b]), "com": m.body_ipos[b].copy(), "inertia": m.body_inertia[b].copy(),
            "iquat": m.body_iquat[b].copy()}
    return out


__all__ = ["geom", "inertial", "body", "model", "compile_check", "mjcf_type_size"]
