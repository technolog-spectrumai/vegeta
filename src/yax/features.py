"""Feature-based interchange: what a machine file holds, what a producer withheld, what a consumer needs, and what a
format can carry — per feature category (docs/interchange.md, the contract).

Every physical fact in the hub belongs to one **feature**::

    geometry      the links' shapes (collision and visual), meshes
    mass          the links' masses, centres of mass and point masses
    inertia       the links' inertia tensors
    joints        the articulation: joint kinds, axes, limits, passive damping, effort and velocity limits
    motors        what drives the joints and rotors: joint drives, joint ``data["motor"]`` (the nonlinear motor:
                  torque–speed line, stall, no-load speed, kt, R, I0), the ``propulsion`` block (rotors, propellers,
                  motors, reduced constants), ``ogma`` rotor constants
    materials     ``data["materials"]`` (the catalogue: name → density, E, ν, …) and each link's ``data["material"]``
    strength      ``data["strength"]`` and each link's ``data["strength"]``: allowables, safety factors, analysis results
    connections   ``data["connections"]``: fasteners, welds, bonds between named links, with their specifications
    aerodynamics  ``data["aerodynamics"]`` (coefficients, derivatives, polars), the ``jsbsim`` sections, links' ``data["aero"]``
    sensors       the links' ``data["sensors"]`` and the SDF sensors carried verbatim (``data["sdf"]["sensor"]``)
    provenance    the producer stamp (tool, revision, machine, identity) and ``data["provenance"]``

A feature's **status** for a machine is one of ``available`` (the machine holds it), ``missing`` (it holds none and
nobody withheld it), ``excluded`` (the producer withheld it: ``Machine.features["excluded"]``), and, relative to a
target format, ``unsupported`` (the format has no place for it; Yax reports the loss when writing).

``inventory(m)`` lists the statuses; ``select(m, include=…, exclude=…)`` returns a copy holding only the chosen
features, recording the rest as excluded — nothing is invented, the structure (links, the joint tree as fixed
joints) stays; ``check(m, Requirements(required, optional), fmt)`` tells a consumer whether the file serves it,
feature by feature, without enabling anything. Units are SI, poses are in the machine frame at q = 0, and names
(links, joints, shapes, part entries) are the identities a consumer maps on (docs/interchange.md).
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

from .hub import Joint, Machine, Report

FEATURES = ("geometry", "mass", "inertia", "joints", "motors", "materials", "strength", "connections", "aerodynamics",
            "sensors", "provenance")
AVAILABLE, MISSING, EXCLUDED, UNSUPPORTED = "available", "missing", "excluded", "unsupported"

#: Machine-level data keys by feature (the contract); a key not listed here is format data (``mjcf``, ``sdf``) or free.
MACHINE_DATA = {"motors": ("propulsion", "ogma"), "materials": ("materials",), "strength": ("strength",),
                "connections": ("connections",), "aerodynamics": ("aerodynamics", "jsbsim"), "sensors": ("sensors",),
                "provenance": ("provenance",)}
#: Link-level data keys by feature.
LINK_DATA = {"motors": ("motor",), "materials": ("material",), "strength": ("strength",), "aerodynamics": ("aero",),
             "sensors": ("sensors",)}
#: Joint-level data keys by feature (the rest of a joint's data — limits, damping — is ``joints``).
JOINT_DATA = {"motors": ("motor",), "connections": ("connection",)}

#: What each format can hold, natively or verbatim through USD (the writers report the rest).
SUPPORT = {
    "usd": set(FEATURES),
    "yx2": set(FEATURES),
    "mjcf": {"geometry", "mass", "inertia", "joints", "motors"},
    "urdf": {"geometry", "mass", "inertia", "joints"},
    "sdf": {"geometry", "mass", "inertia", "joints", "sensors"},
    "jsbsim": {"mass", "inertia", "motors", "aerodynamics"},
    "step": {"geometry"},
    "stl": {"geometry"},
    "package": {"geometry", "mass", "inertia", "provenance"},
}
#: Where a format's support is partial: what of the feature it keeps.
PARTIAL = {("mjcf", "motors"): "joint drives as position actuators (kp, kv, force range); no motor curves, no propellers",
           ("urdf", "joints"): "ball joints as fixed; the drive's max force as the effort limit",
           ("sdf", "joints"): "no drives (a plugin carries them)",
           ("jsbsim", "mass"): "one rigid body: the links' masses lumped at q = 0",
           ("jsbsim", "inertia"): "one rigid body: the links' inertias lumped at q = 0",
           ("jsbsim", "motors"): "propellers and electric engines from the propulsion block; no joint drives",
           ("package", "mass"): "the whole body's mass, centre and inertia"}


@dataclass
class Status:
    feature: str
    status: str
    detail: str = ""

    def __str__(self):
        return f"{self.feature}: {self.status}" + (f" ({self.detail})" if self.detail else "")


# ---------------------------------------------------------------------------------------------------- inventory
def _count(*pairs) -> str:
    return ", ".join(f"{n} {what}" for n, what in pairs if n)


def _held(m: Machine) -> dict[str, str]:
    """Feature → the detail of what the machine holds of it ('' when nothing)."""
    held = {}
    held["geometry"] = _count((sum(len(lk.shapes) for lk in m.links), "shapes"))
    held["mass"] = _count((sum(1 for lk in m.links if lk.mass is not None), "link masses"),
                          (sum(len(lk.point_masses) for lk in m.links), "point masses"))
    held["inertia"] = _count((sum(1 for lk in m.links if lk.inertia is not None), "link inertias"))
    moving = sum(1 for j in m.joints if j.kind != "fixed")                   # fixed joints are structure, not articulation
    held["joints"] = _count((moving, "moving joints"), (sum(1 for j in m.joints if j.kind == "fixed") if moving else 0, "fixed"))
    motors = [(sum(1 for j in m.joints if j.drive), "joint drives"), (sum(1 for j in m.joints if "motor" in j.data), "joint motors")]
    prop = m.data.get("propulsion") or {}
    motors += [(len(prop.get("rotors", [])), "rotors"), (len(prop.get("propellers", {})), "propellers"), (len(prop.get("motors", {})), "motors")]
    motors += [(len((m.data.get("ogma") or {}).get("rotors", [])), "ogma rotors")]
    held["motors"] = _count(*motors)
    for feat in ("materials", "strength", "connections", "aerodynamics", "sensors"):
        n_m = sum(1 for k in MACHINE_DATA.get(feat, ()) if m.data.get(k))
        n_l = sum(1 for lk in m.links for k in LINK_DATA.get(feat, ()) if lk.data.get(k))
        n_j = sum(1 for j in m.joints for k in JOINT_DATA.get(feat, ()) if j.data.get(k))
        if feat == "sensors":
            n_l += sum(1 for lk in m.links if (lk.data.get("sdf") or {}).get("sensor"))
        held[feat] = _count((n_m, "machine entries"), (n_l, "link entries"), (n_j, "joint entries"))
    held["provenance"] = _count((1 if m.producer else 0, "producer stamp"), (1 if m.data.get("provenance") else 0, "provenance data"))
    return held


def inventory(m: Machine) -> dict[str, Status]:
    """Each feature's status for the machine: available (with what it holds), excluded, or missing."""
    excluded = set(m.features.get("excluded", ()))
    out = {}
    for feat, detail in _held(m).items():
        if detail:
            out[feat] = Status(feat, AVAILABLE, detail)
        elif feat in excluded:
            out[feat] = Status(feat, EXCLUDED, "withheld by the producer")
        else:
            out[feat] = Status(feat, MISSING)
    return out


def describe(m: Machine) -> str:
    return "\n".join(str(s) for s in inventory(m).values())


# ------------------------------------------------------------------------------------------------------ select
def select(m: Machine, include=None, exclude=()) -> Machine:
    """A copy of ``m`` holding only the chosen features (``include``: a set of features, default all; ``exclude``
    wins). Nothing is invented: a withheld mass is ``None``, withheld joints become fixed (the tree stays, so links
    keep their names and poses), withheld data keys are gone, a withheld provenance is an empty producer. The copy
    records what was withheld in ``features["excluded"]`` and what it holds in ``features["included"]``."""
    include = set(FEATURES if include is None else include)
    unknown = (include | set(exclude)) - set(FEATURES)
    if unknown:
        raise ValueError(f"unknown features {sorted(unknown)}; the features are {FEATURES}")
    keep = include - set(exclude)
    out = copy.deepcopy(m)
    drop = set(FEATURES) - keep
    if "geometry" in drop:
        for lk in out.links:
            lk.shapes = []
    if "mass" in drop:
        for lk in out.links:
            lk.mass, lk.com, lk.point_masses = None, (0.0, 0.0, 0.0), []
    if "inertia" in drop:
        for lk in out.links:
            lk.inertia = None
    if "joints" in drop:
        out.joints = [Joint(j.name, "fixed", j.parent, j.child, pos=j.pos, axis=j.axis) for j in out.joints]
        for lk in out.links:
            lk.data.pop("free_joint", None)
    if "motors" in drop:
        for j in out.joints:
            j.drive = None
    for feat in drop:
        for k in MACHINE_DATA.get(feat, ()):
            out.data.pop(k, None)
        for lk in out.links:
            for k in LINK_DATA.get(feat, ()):
                lk.data.pop(k, None)
            if feat == "sensors" and "sdf" in lk.data:
                lk.data["sdf"].pop("sensor", None)
                if not lk.data["sdf"]:
                    del lk.data["sdf"]
        for j in out.joints:
            for k in JOINT_DATA.get(feat, ()):
                j.data.pop(k, None)
    if "provenance" in drop:
        out.producer = {}
    out.features = {"included": sorted(keep), "excluded": sorted(set(m.features.get("excluded", ())) | drop)}
    return out


# ------------------------------------------------------------------------------------------- requirements
@dataclass
class Requirements:
    """What a consumer needs (``required``: the file is useless without them) and uses when present (``optional``)."""
    required: tuple = ()
    optional: tuple = ()
    consumer: str = ""

    def __post_init__(self):
        self.required, self.optional = tuple(self.required), tuple(self.optional)
        unknown = (set(self.required) | set(self.optional)) - set(FEATURES)
        if unknown:
            raise ValueError(f"unknown features {sorted(unknown)}; the features are {FEATURES}")

    @classmethod
    def load(cls, path) -> "Requirements":
        """From a JSON file ``{"consumer": "...", "required": [...], "optional": [...]}``."""
        d = json.loads(Path(path).read_text())
        return cls(d.get("required", ()), d.get("optional", ()), d.get("consumer", Path(path).stem))


@dataclass
class Compatibility:
    """The verdict of ``check``: each required and optional feature's status for the consumer, and whether the file
    serves it (every required feature available and, given a format, supported)."""
    requirements: Requirements
    fmt: str | None
    statuses: dict = field(default_factory=dict)

    @property
    def unmet(self) -> list:
        return [s for f, s in self.statuses.items() if f in self.requirements.required and s.status != AVAILABLE]

    @property
    def ok(self) -> bool:
        return not self.unmet

    def __str__(self):
        who = self.requirements.consumer or "the consumer"
        head = f"{who}{f' ({self.fmt})' if self.fmt else ''}: {'served' if self.ok else 'NOT served'}"
        lines = [f"  {'required' if f in self.requirements.required else 'optional'} {s}" for f, s in self.statuses.items()]
        return "\n".join([head] + lines)


def check(m: Machine, req: Requirements, fmt: str | None = None) -> Compatibility:
    """Does the machine serve the consumer? A feature is ``unsupported`` when the target format cannot hold it (the
    consumer would read it from the USD instead), else its inventory status; partial support is noted. Nothing is
    enabled or changed: the verdict is about the file as it is."""
    inv = inventory(m)
    out = Compatibility(req, fmt)
    for feat in (*req.required, *(f for f in req.optional if f not in req.required)):
        s = inv[feat]
        if fmt and feat not in SUPPORT.get(fmt, set()):
            s = Status(feat, UNSUPPORTED, f"{fmt} has no place for it" + (f"; it is {s.status} in the machine" if s.status != MISSING else ""))
        elif fmt and (fmt, feat) in PARTIAL and s.status == AVAILABLE:
            s = Status(feat, AVAILABLE, f"{s.detail}; {fmt} keeps {PARTIAL[(fmt, feat)]}")
        out.statuses[feat] = s
    return out


def report_excluded(m: Machine, rep: Report) -> None:
    """Writers call this: a withheld feature is a loss the file's reader should know about."""
    for feat in m.features.get("excluded", ()):
        rep.drop(f"{feat}: withheld by the producer (not in the source)")
