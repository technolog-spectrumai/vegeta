"""Tags: property groups a producer marks and a consumer validates (docs/interchange.md).

Every **property** of a machine has a path and may carry any number of **tags** — codes of one to eight alphanumeric
characters. A producer tags properties as it pleases (``tag(m, pattern, code)``), keeps or drops properties by tag
(``select``), and writes the table into the file; a consumer validates the file against the tags it needs
(``check``) — it does not have to know the producer's groups, only the codes the two agreed on. Codes that start
with ``00`` are **system tags**, reserved to Yax: it assigns them itself from what a property is, so a consumer can
always validate against them::

    00core    the skeleton — the machine's name, the links' identities and poses, the joint tree (never withheld)
    00geom    shapes (collision and visual, meshes)                 00mass    masses, centres of mass, point masses
    00inert   inertia tensors                                       00joint   articulation: kinds, axes, limits, damping
    00motor   drives, motor models, the propulsion motors, ogma     00prop    propulsion: rotors, propellers, maps, reduced constants
    00mat     materials (catalogue and per link)                    00str     structural strength
    00conn    connections                                           00aero    aerodynamics, JSBSim sections, per-link aero
    00sensor  sensors (and SDF sensors carried verbatim)         00detach  the assembly: parent–child, attachments, separation
    00prov    the producer stamp, provenance data                   00fmt     another format's own data carried verbatim

Property paths (``properties(m)``)::

    link:<name>/pose           link:<name>/mass          link:<name>/inertia
    link:<name>/<role>:<shape> link:<name>/pmass:<name>  link:<name>/data:<key>[/<key>…]
    joint:<name>/tree          joint:<name>/articulation joint:<name>/drive     joint:<name>/data:<key>[/<key>…]
    data:<key>[/<key>…]        producer

``data:`` paths reach every dictionary key at any depth; a tag on a path covers everything under it (a tag on
``data:propulsion`` covers the block, a tag on ``data:propulsion/propellers/apc`` one propeller and its fields). ``Machine.tags`` is ``{"table": {path: [codes]}, "withheld":
[codes]}``: the table lists every tagged property (system tags included, so the file stands on its own), ``withheld``
the tags the producer dropped. Nothing is invented: a dropped property is absent (a mass ``None``, a joint fixed, a
key gone), never defaulted. A container (a data block) stays while a property under it is kept: dropping ``00prop``
with a producer tag kept on one propeller leaves the block holding that propeller and nothing else.
"""
from __future__ import annotations

import copy
import fnmatch
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .hub import Joint, Machine, Report

CODE = re.compile(r"^[A-Za-z0-9]{1,8}$")
SYSTEM_PREFIX = "00"
SYSTEM = {"00core": "the skeleton: the machine's name, the links' identities and poses, the joint tree",
          "00geom": "geometry: shapes (collision and visual), meshes", "00mass": "mass properties: masses, centres of mass, point masses",
          "00inert": "inertia tensors", "00joint": "articulation: joint kinds, axes, limits, passive damping",
          "00motor": "motors and actuators: joint drives, motor models (stall, no-load, curves, transmission), the propulsion motors",
          "00prop": "propulsion: rotors, propellers (geometry, CT/CP/CQ against J, performance maps), reduced rotor constants",
          "00mat": "materials: the catalogue and each link's", "00str": "structural strength: allowables, safety factors, analysis results",
          "00conn": "physical connections between links", "00aero": "aerodynamics: coefficients, tables, reference points, JSBSim sections",
          "00sensor": "sensors, SDF sensors carried verbatim", "00prov": "the producer stamp and provenance data",
          "00fmt": "another format's own data carried verbatim (mjcf, sdf plugins)",
          "00detach": "detachable components: the assembly's parent–child relations, attachments, separation limits"}
#: Machine data keys → system tag (the contract); unknown keys get no system tag (a producer tags them).
DATA_TAGS = {"propulsion": ("00prop",), "ogma": ("00prop", "00motor"), "motors": ("00motor",), "materials": ("00mat",), "assembly": ("00detach",),
             "strength": ("00str",), "connections": ("00conn",), "aerodynamics": ("00aero",), "jsbsim": ("00aero",),
             "sensors": ("00sensor",), "provenance": ("00prov",), "mjcf": ("00fmt",), "sdf": ("00fmt",)}
#: Deeper data paths that carry more system tags than their block (a property may belong to several categories).
DATA_PATH_TAGS = {"data:propulsion/motors": ("00motor",)}
LINK_DATA_TAGS = {"motor": ("00motor",), "material": ("00mat",), "strength": ("00str",), "aero": ("00aero",), "sensors": ("00sensor",),
                  "sdf": ("00fmt",), "free_joint": ("00core",)}
JOINT_DATA_TAGS = {"motor": ("00motor",), "connection": ("00conn",), "damping": ("00joint",), "effort": ("00joint",), "velocity": ("00joint",)}
KINDS = ("pose", "mass", "inertia", "shape", "pmass", "tree", "articulation", "drive", "data", "producer")
#: What each format holds, by property kind (a tag is unsupported when none of its properties' kinds are).
SUPPORT = {"usd": set(KINDS), "yx2": set(KINDS),
           "mjcf": {"pose", "mass", "inertia", "shape", "pmass", "tree", "articulation", "drive"},
           "urdf": {"pose", "mass", "inertia", "shape", "pmass", "tree", "articulation"},
           "sdf": {"pose", "mass", "inertia", "shape", "pmass", "tree", "articulation"},
           "jsbsim": {"mass", "inertia", "pmass"}, "step": {"pose", "shape"}, "stl": {"pose", "shape"},
           "package": {"pose", "shape", "mass", "inertia", "producer"}}
#: Data paths a format holds beyond the kinds above (prefix match).
SUPPORT_DATA = {"sdf": ("data:sdf", "link:*/data:sdf"), "jsbsim": ("data:propulsion", "data:jsbsim"), "mjcf": ("data:mjcf",)}
PARTIAL = {("mjcf", "00motor"): "joint drives as position actuators; no motor models, no propellers",
           ("urdf", "00joint"): "ball joints as fixed; the drive's max force as the effort limit",
           ("sdf", "00joint"): "no drives (a plugin carries them)",
           ("jsbsim", "00mass"): "one rigid body: the links' masses lumped at q = 0",
           ("jsbsim", "00motor"): "propellers and electric engines from the propulsion block; no joint drives",
           ("jsbsim", "00prop"): "the propellers' coefficients against J and the motors' power; not the grids or stations"}


def is_system(code: str) -> bool:
    return code.startswith(SYSTEM_PREFIX)


def check_code(code: str, *, system_ok: bool = False) -> str:
    """A tag code as a producer may give it: 1–8 alphanumerics, not in the ``00`` namespace unless ``system_ok``."""
    if not isinstance(code, str) or not CODE.match(code):
        raise ValueError(f"tag {code!r}: a tag is 1 to 8 alphanumeric characters")
    if is_system(code) and not system_ok:
        raise ValueError(f"tag {code!r}: codes starting with {SYSTEM_PREFIX!r} are system tags, Yax's to assign")
    return code


# ---------------------------------------------------------------------------------------------------- properties
@dataclass
class Property:
    path: str
    kind: str
    system: tuple                 # the system tags Yax assigns (a property may belong to several categories); () under an unknown key


def _data_paths(prefix: str, d, system, out: list):
    for k, v in d.items():
        p = f"{prefix}/{k}"
        sys_ = tuple(system) + tuple(t for pre, ts in DATA_PATH_TAGS.items() if p == pre or p.startswith(pre + "/") for t in ts if t not in system)
        out.append(Property(p, "data", sys_))
        if isinstance(v, dict):
            _data_paths(p, v, sys_, out)


def properties(m: Machine) -> list:
    """Every property of the machine with its path, kind and system tag (docs/interchange.md)."""
    out = []
    for lk in m.links:
        L = f"link:{lk.name}"
        out.append(Property(f"{L}/pose", "pose", ("00core",)))
        if lk.mass is not None:
            out.append(Property(f"{L}/mass", "mass", ("00mass",)))
        if lk.inertia is not None:
            out.append(Property(f"{L}/inertia", "inertia", ("00inert",)))
        for s in lk.shapes:
            out.append(Property(f"{L}/{s.role}:{s.name}", "shape", ("00geom",)))
        for n, _, _ in lk.point_masses:
            out.append(Property(f"{L}/pmass:{n}", "pmass", ("00mass",)))
        for k, v in lk.data.items():
            sys_ = LINK_DATA_TAGS.get(k, ())
            out.append(Property(f"{L}/data:{k}", "data", sys_))
            if k == "sdf" and isinstance(v, dict):                             # verbatim SDF: sensors are sensors
                for kk, vv in v.items():
                    out.append(Property(f"{L}/data:sdf/{kk}", "data", ("00sensor",) if kk == "sensor" else ("00fmt",)))
            elif isinstance(v, dict):
                _data_paths(f"{L}/data:{k}", v, sys_, out)
    for j in m.joints:
        J = f"joint:{j.name}"
        out.append(Property(f"{J}/tree", "tree", ("00core",)))
        if j.kind != "fixed" or j.lower is not None:
            out.append(Property(f"{J}/articulation", "articulation", ("00joint",)))
        if j.drive:
            out.append(Property(f"{J}/drive", "drive", ("00motor",)))
        for k, v in j.data.items():
            sys_ = JOINT_DATA_TAGS.get(k, ())
            out.append(Property(f"{J}/data:{k}", "data", sys_))
            if isinstance(v, dict):
                _data_paths(f"{J}/data:{k}", v, sys_, out)
    for k, v in m.data.items():
        sys_ = DATA_TAGS.get(k, ())
        out.append(Property(f"data:{k}", "data", sys_))
        if isinstance(v, dict):
            _data_paths(f"data:{k}", v, sys_, out)
    if m.producer:
        out.append(Property("producer", "producer", ("00prov",)))
    return out


def table(m: Machine) -> dict:
    """The full tag table: each property's system tag plus the producer's tags (``m.tags["table"]``), sorted."""
    given = (m.tags or {}).get("table", {})
    out = {}
    for p in properties(m):
        codes = set(p.system)
        parts = p.path.split("/")
        for i in range(1, len(parts) + 1):                                     # a producer's tag covers the subtree under its path
            codes |= {c for c in given.get("/".join(parts[:i]), ()) if not is_system(c) or i == len(parts)}
        if codes:
            out[p.path] = sorted(codes)
    return out


def tag(m: Machine, pattern: str, code: str) -> int:
    """Give every property whose path matches ``pattern`` (fnmatch: ``link:*/mass``, ``data:propulsion/*``) the
    producer tag ``code``; the number of properties tagged (0 is an error: nothing matched)."""
    check_code(code)
    if not m.tags:
        m.tags = {"table": {}, "withheld": []}
    t = m.tags.setdefault("table", {})
    n = 0
    for p in properties(m):
        if fnmatch.fnmatchcase(p.path, pattern):
            codes = t.setdefault(p.path, [])
            if code not in codes:
                codes.append(code)
                codes.sort()
            n += 1
    if n == 0:
        raise ValueError(f"tag {code!r}: no property matches {pattern!r}")
    return n


def inventory(m: Machine) -> dict:
    """Tag → the paths carrying it (system tags included); withheld tags map to ``None``."""
    out = {}
    for path, codes in table(m).items():
        for c in codes:
            out.setdefault(c, []).append(path)
    for c in (m.tags or {}).get("withheld", []):
        out.setdefault(c, None)
    return {c: (sorted(v) if v is not None else None) for c, v in sorted(out.items())}


def describe(m: Machine) -> str:
    lines = []
    for c, paths in inventory(m).items():
        what = f"withheld by the producer" if paths is None else f"{len(paths)} properties"
        lines.append(f"{c}: {what}" + (f" — {SYSTEM[c]}" if c in SYSTEM else ""))
    return "\n".join(lines)


# -------------------------------------------------------------------------------------------------------- remove
def _del(d: dict, keys: list):
    for k in keys[:-1]:
        d = d[k]
    del d[keys[-1]]


def remove(m: Machine, path: str) -> None:
    """Take a property out of the machine (in place), never leaving a default behind. ``00core`` properties
    (poses, the tree) cannot be removed."""
    head, _, rest = path.partition("/")
    kind, _, name = head.partition(":")
    if kind == "producer":
        m.producer = {}
        return
    if kind == "data":
        _del(m.data, [name, *rest.split("/")] if rest else [name])
        return
    if kind == "link":
        lk = m.link(name)
        what, _, arg = rest.partition(":")
        if what == "mass":
            lk.mass, lk.com, lk.point_masses = None, (0.0, 0.0, 0.0), []
        elif what == "inertia":
            lk.inertia = None
        elif what in ("collision", "visual"):
            lk.shapes = [s for s in lk.shapes if not (s.role == what and s.name == arg)]
        elif what == "pmass":
            lk.point_masses = [pm for pm in lk.point_masses if pm[0] != arg]
        elif what == "data":
            _del(lk.data, arg.split("/"))
        elif what == "pose":
            raise ValueError(f"{path}: a 00core property cannot be removed")
        else:
            raise KeyError(path)
        return
    if kind == "joint":
        j = next(x for x in m.joints if x.name == name)
        what, _, arg = rest.partition(":")
        if what == "articulation":
            i = m.joints.index(j)
            m.joints[i] = Joint(j.name, "fixed", j.parent, j.child, pos=j.pos, axis=j.axis,
                                data={k: v for k, v in j.data.items() if k not in JOINT_DATA_TAGS or JOINT_DATA_TAGS[k] != "00joint"})
        elif what == "drive":
            j.drive = None
        elif what == "data":
            _del(j.data, arg.split("/"))
        elif what == "tree":
            raise ValueError(f"{path}: a 00core property cannot be removed")
        else:
            raise KeyError(path)
        return
    raise KeyError(path)


def select(m: Machine, keep=None, drop=()) -> Machine:
    """A copy holding the properties whose tags meet ``keep`` (any of; default: every tag) and none of ``drop``;
    ``00core`` always stays. The copy's table lists what it holds, ``withheld`` the tags that lost properties."""
    keep = None if keep is None else set(keep)
    drop = set(drop)
    for c in set(keep or ()) | drop:
        check_code(c, system_ok=True)
    if "00core" in drop:
        raise ValueError("00core cannot be dropped")
    out = copy.deepcopy(m)
    full = table(out)
    removed, withheld = [], set()
    for p in properties(out):
        codes = set(full.get(p.path, ()))
        if "00core" in codes:
            continue
        ok = (keep is None or bool(codes & keep)) and not (codes & drop)
        if not ok:
            removed.append(p.path)
            withheld |= codes - (keep or set()) if keep is not None else codes & drop
    gone = set(removed)                                                        # a container stays while something under it is kept
    removed = [r for r in removed if not any(k.startswith(r + "/") for k in (p.path for p in properties(out)) if k not in gone)]
    removed.sort(key=lambda s: -s.count("/"))                                  # deepest first: a parent's removal is enough
    for path in removed:
        try:
            remove(out, path)
        except (KeyError, StopIteration):
            pass                                                               # gone with its parent
    kept = table(out)
    gone = {c for cs in full.values() for c in cs} - {c for cs in kept.values() for c in cs}   # tags that lost every property
    old = set((m.tags or {}).get("withheld", []))
    out.tags = {"table": {p: c for p, c in kept.items()}, "withheld": sorted((old | withheld | drop | gone) - {"00core"})}
    return out


# ------------------------------------------------------------------------------------------- requirements
@dataclass
class Requirements:
    """What a consumer needs: tags whose properties the file must hold (``required``) and tags it uses when present."""
    required: tuple = ()
    optional: tuple = ()
    consumer: str = ""

    def __post_init__(self):
        self.required, self.optional = tuple(self.required), tuple(self.optional)
        for c in (*self.required, *self.optional):
            check_code(c, system_ok=True)

    @classmethod
    def load(cls, path) -> "Requirements":
        d = json.loads(Path(path).read_text())
        return cls(d.get("required", ()), d.get("optional", ()), d.get("consumer", Path(path).stem))


PRESENT, ABSENT, WITHHELD, UNSUPPORTED = "present", "absent", "withheld", "unsupported"


@dataclass
class Status:
    tag: str
    status: str
    detail: str = ""

    def __str__(self):
        return f"{self.tag}: {self.status}" + (f" ({self.detail})" if self.detail else "")


@dataclass
class Compatibility:
    requirements: Requirements
    fmt: str | None
    statuses: dict = field(default_factory=dict)

    @property
    def unmet(self) -> list:
        return [s for t, s in self.statuses.items() if t in self.requirements.required and s.status != PRESENT]

    @property
    def ok(self) -> bool:
        return not self.unmet

    def __str__(self):
        who = self.requirements.consumer or "the consumer"
        head = f"{who}{f' ({self.fmt})' if self.fmt else ''}: {'served' if self.ok else 'NOT served'}"
        return "\n".join([head] + [f"  {'required' if t in self.requirements.required else 'optional'} {s}" for t, s in self.statuses.items()])


def _supported(fmt: str, path: str, kind: str) -> bool:
    if fmt in ("usd", "yx2"):
        return True
    if kind != "data":
        return kind in SUPPORT.get(fmt, set())
    return any(fnmatch.fnmatchcase(path, pre) or fnmatch.fnmatchcase(path, pre + "/*") for pre in SUPPORT_DATA.get(fmt, ()))


def check(m: Machine, req: Requirements, fmt: str | None = None) -> Compatibility:
    """Does the file serve the consumer? Per tag: present (with how many properties), absent, withheld by the
    producer, or unsupported by the target format (none of its properties' kinds have a place there). Changes nothing."""
    inv = inventory(m)
    kinds = {p.path: p.kind for p in properties(m)}
    out = Compatibility(req, fmt)
    for t in (*req.required, *(c for c in req.optional if c not in req.required)):
        paths = inv.get(t, "absent")
        if paths is None:
            s = Status(t, WITHHELD, "withheld by the producer")
        elif paths == "absent":
            s = Status(t, ABSENT)
        else:
            s = Status(t, PRESENT, f"{len(paths)} properties")
            if fmt:
                held = [p for p in paths if _supported(fmt, p, kinds.get(p, "data"))]
                if not held:
                    s = Status(t, UNSUPPORTED, f"{fmt} has no place for any of its {len(paths)} properties; read them from the YX2")
                elif len(held) < len(paths):
                    s = Status(t, PRESENT, f"{len(held)} of {len(paths)} properties reach {fmt}" + (f"; {PARTIAL[(fmt, t)]}" if (fmt, t) in PARTIAL else ""))
                elif (fmt, t) in PARTIAL:
                    s = Status(t, PRESENT, f"{len(paths)} properties; {fmt} keeps {PARTIAL[(fmt, t)]}")
        out.statuses[t] = s
    return out


def report_withheld(m: Machine, rep: Report) -> None:
    for c in (m.tags or {}).get("withheld", []):
        rep.drop(f"tag {c}: withheld by the producer (its properties are not in the source)")
