"""``.yx2`` — the Yax 2 exchange file: a validated subset of OpenUSD (docs/yx2.md).

One self-contained ``usda`` text layer holding a machine the way the hub holds it, plus what an exchange needs and
plain USD leaves open: the file says its **units** and **frame** outright (layer data ``yx2``), every component
carries a **stable identifier** (``yx2:id``: ``link:<name>``, ``joint:<name>``, ``shape:<link>/<name>``,
``mass:<link>/<name>``) that survives USD's path mangling, the **feature metadata** (``yax:features``: what the
producer shared and withheld) is always present, and the **serialization is canonical**: usda text, prims in
machine order, JSON part data with sorted keys and no whitespace, so the same machine gives the same bytes and
``write(read(f)) == f``.

``validate(path)`` lists the problems (``[]`` when the file is a YX2); ``read`` refuses an invalid file
(``yax.Invalid``), ``write`` refuses to write one. USD itself reads a ``.yx2`` after a rename to ``.usda`` (or
``Sdf.Layer.ImportFromString``): nothing in it is outside USD.
"""
from __future__ import annotations

import json
from pathlib import Path

from .. import hub
from ..hub import JOINTS, Machine, Report
from ..package import Invalid

VERSION = "0.1"
UNITS = {"length": "m", "mass": "kg", "time": "s", "angle": "rad"}
FRAME = {"up": "+z", "forward": "+x", "handed": "right", "pose": "machine frame at q = 0"}
#: The closed vocabulary of Yax attributes (the subset): anything else in the ``yax:`` or ``yx2:`` namespace is a problem.
ATTRS = {"yax:name", "yax:kind", "yax:pos", "yax:quat", "yax:axis", "yax:limits", "yax:drive", "yax:data", "yax:mass",
         "yax:com", "yax:inertia", "yax:role", "yax:uri", "yax:producer", "yax:features", "yx2:id"}
ROLES = ("collision", "visual")


def _canonical(s: str) -> bool:
    try:
        return json.dumps(json.loads(s), sort_keys=True, separators=(",", ":")) == s
    except ValueError:
        return False


def _layer_from_text(text: str):
    from pxr import Sdf
    from pxr import Tf
    layer = Sdf.Layer.CreateAnonymous("yx2.usda")
    try:
        ok = layer.ImportFromString(text)
    except Tf.ErrorException as e:                                            # USD's parser throws on what is not usda
        raise Invalid([f"not a usda text layer: {str(e).strip().splitlines()[-1]}"]) from None
    if not ok:
        raise Invalid(["not a usda text layer"])
    return layer


# -------------------------------------------------------------------------------------------------------- validate
def validate_stage(stage) -> list:
    """The problems of a stage against the YX2 profile; ``[]`` when it is one."""
    from pxr import UsdGeom, UsdPhysics
    from ..features import FEATURES
    p = []
    ld = stage.GetRootLayer().customLayerData or {}
    y = ld.get("yx2")
    if not isinstance(y, dict):
        return ["no yx2 layer data (customLayerData['yx2']): not a YX2 file"]
    if y.get("version") != VERSION:
        p.append(f"yx2.version: expected {VERSION!r}, got {y.get('version')!r}")
    if dict(y.get("units") or {}) != UNITS:
        p.append(f"yx2.units: expected {UNITS}, got {dict(y.get('units') or {})}")
    if dict(y.get("frame") or {}) != FRAME:
        p.append(f"yx2.frame: expected {FRAME}, got {dict(y.get('frame') or {})}")
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        p.append("stage upAxis: expected Z")
    if UsdGeom.GetStageMetersPerUnit(stage) != 1.0:
        p.append("stage metersPerUnit: expected 1")
    if UsdPhysics.GetStageKilogramsPerUnit(stage) != 1.0:
        p.append("stage kilogramsPerUnit: expected 1")
    root = stage.GetDefaultPrim()
    if not root:
        return p + ["no default prim (the machine)"]
    if root.GetCustomDataByKey("yax") != 2:
        p.append("root customData yax: expected 2")
    if not root.GetCustomDataByKey("name"):
        p.append("root customData name: the machine's name is required")
    feats = hub._get(root, "yax:features")
    try:
        f = json.loads(feats) if feats else None
    except ValueError:
        f = None
    if not isinstance(f, dict) or not isinstance(f.get("included"), list) or not isinstance(f.get("excluded"), list):
        p.append("root yax:features: 'included' and 'excluded' lists are required")
    else:
        bad = sorted((set(f["included"]) | set(f["excluded"])) - set(FEATURES))
        if bad:
            p.append(f"root yax:features: unknown features {bad}")
        if set(f["included"]) & set(f["excluded"]):
            p.append("root yax:features: a feature cannot be both included and excluded")
    for key in ("yax:producer", "yax:data", "yax:features"):
        v = hub._get(root, key)
        if v and not _canonical(v):
            p.append(f"root {key}: not canonical JSON (sorted keys, no whitespace)")
    links, joints = {}, {}
    from pxr import Usd
    for prim in Usd.PrimRange(root):
        for a in prim.GetAttributes():
            n = a.GetName()
            if (n.startswith("yax:") or n.startswith("yx2:")) and n not in ATTRS:
                p.append(f"{prim.GetPath()}: attribute {n} is outside the YX2 vocabulary")
        if prim == root:
            continue
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            name = hub._get(prim, "yax:name")
            if not name:
                p.append(f"{prim.GetPath()}: a link without yax:name")
                continue
            if name in links:
                p.append(f"link {name!r}: the name is not unique (also {links[name]})")
            links[name] = prim.GetPath()
            if hub._get(prim, "yx2:id") != f"link:{name}":
                p.append(f"link {name!r}: yx2:id must be 'link:{name}', got {hub._get(prim, 'yx2:id')!r}")
            if prim.HasAPI(UsdPhysics.MassAPI) and hub._get(prim, "yax:mass") is None:
                p.append(f"link {name!r}: MassAPI without the exact yax:mass")
            d = hub._get(prim, "yax:data")
            if d and not _canonical(d):
                p.append(f"link {name!r}: yax:data is not canonical JSON")
            shapes = set()
            for child in prim.GetChildren():
                cname = hub._get(child, "yax:name")
                if hub._get(child, "yax:kind") == "point_mass":
                    if hub._get(child, "yx2:id") != f"mass:{name}/{cname}":
                        p.append(f"link {name!r}: point mass {cname!r} without its yx2:id")
                    continue
                if child.IsA(UsdGeom.Gprim):
                    if not cname:
                        p.append(f"link {name!r}: a shape without yax:name ({child.GetPath()})")
                        continue
                    if cname in shapes and hub._get(child, "yax:role") in ROLES and (cname, hub._get(child, "yax:role")) in shapes:
                        p.append(f"link {name!r}: shape {cname!r} ({hub._get(child, 'yax:role')}) is not unique")
                    shapes.add((cname, hub._get(child, "yax:role")))
                    shapes.add(cname)
                    if hub._get(child, "yax:role") not in ROLES:
                        p.append(f"link {name!r}: shape {cname!r}: yax:role must be one of {ROLES}")
                    if hub._get(child, "yx2:id") != f"shape:{name}/{cname}":
                        p.append(f"link {name!r}: shape {cname!r} without its yx2:id")
        elif prim.IsA(UsdPhysics.Joint):
            name = hub._get(prim, "yax:name")
            if not name:
                p.append(f"{prim.GetPath()}: a joint without yax:name")
                continue
            if name in joints:
                p.append(f"joint {name!r}: the name is not unique")
            joints[name] = prim.GetPath()
            if hub._get(prim, "yax:kind") not in JOINTS:
                p.append(f"joint {name!r}: yax:kind must be one of {JOINTS}")
            if hub._get(prim, "yax:pos") is None or hub._get(prim, "yax:axis") is None:
                p.append(f"joint {name!r}: yax:pos and yax:axis are required")
            if hub._get(prim, "yx2:id") != f"joint:{name}":
                p.append(f"joint {name!r}: yx2:id must be 'joint:{name}'")
            for key in ("yax:drive", "yax:data"):
                v = hub._get(prim, key)
                if v and not _canonical(v):
                    p.append(f"joint {name!r}: {key} is not canonical JSON")
    if not links:
        p.append("no links (rigid bodies)")
    for j, path in joints.items():
        uj = UsdPhysics.Joint(stage.GetPrimAtPath(path))
        for rel in (uj.GetBody0Rel().GetTargets(), uj.GetBody1Rel().GetTargets()):
            if rel and rel[0] not in links.values():
                p.append(f"joint {j!r}: body {rel[0]} is not a link of this machine")
    return p


def validate(path) -> list:
    """The problems of a ``.yx2`` file; ``[]`` when it is one."""
    from pxr import Usd
    try:
        layer = _layer_from_text(Path(path).read_text())
    except (OSError, UnicodeDecodeError, Invalid) as e:
        return [str(e)]
    return validate_stage(Usd.Stage.Open(layer))


# -------------------------------------------------------------------------------------------------------- read, write
def stage_of(m: Machine):
    """The machine as a YX2 stage (in memory): the hub's stage plus the layer data, the ids and the feature metadata."""
    from pxr import Sdf, Usd
    from ..features import FEATURES
    m2 = Machine(m.name, m.links, m.joints, m.data, m.producer,
                 {"included": list(m.features.get("included", FEATURES)), "excluded": list(m.features.get("excluded", []))})
    stage = hub.to_stage(m2)
    root = stage.GetDefaultPrim()
    for prim in Usd.PrimRange(root):
        if prim == root:
            continue
        name = hub._get(prim, "yax:name")
        if hub._get(prim, "yax:kind") == "point_mass":
            link = hub._get(prim.GetParent(), "yax:name")
            hub._attr(prim, "yx2:id", Sdf.ValueTypeNames.String, f"mass:{link}/{name}")
        elif prim.GetParent() == root and name and prim.GetName() != "joints":
            hub._attr(prim, "yx2:id", Sdf.ValueTypeNames.String, f"link:{name}")
        elif prim.GetParent().GetName() == "joints":
            hub._attr(prim, "yx2:id", Sdf.ValueTypeNames.String, f"joint:{name}")
        elif name:
            hub._attr(prim, "yx2:id", Sdf.ValueTypeNames.String, f"shape:{hub._get(prim.GetParent(), 'yax:name')}/{name}")
    for key in ("yax:producer", "yax:data", "yax:features"):
        a = root.GetAttribute(key)
        if a and a.HasAuthoredValue():
            a.Set(json.dumps(json.loads(a.Get()), sort_keys=True, separators=(",", ":")))
    for prim in Usd.PrimRange(root):
        for key in ("yax:data", "yax:drive"):
            a = prim.GetAttribute(key)
            if prim != root and a and a.HasAuthoredValue():
                a.Set(json.dumps(json.loads(a.Get()), sort_keys=True, separators=(",", ":")))
    layer = stage.GetRootLayer()
    layer.customLayerData = {"yx2": {"version": VERSION, "units": dict(UNITS), "frame": dict(FRAME)}}
    return stage


def problems_of(m: Machine) -> list:
    """What keeps a machine out of the profile before a stage is even built: names that are no identities."""
    p = []
    if not m.name:
        p.append("the machine has no name")
    seen = set()
    for lk in m.links:
        if not lk.name:
            p.append("a link without a name")
        elif lk.name in seen:
            p.append(f"link {lk.name!r}: the name is not unique")
        seen.add(lk.name)
        roles = set()
        for sh in lk.shapes:
            if not sh.name:
                p.append(f"link {lk.name!r}: a shape without a name")
            elif (sh.name, sh.role) in roles:
                p.append(f"link {lk.name!r}: shape {sh.name!r} ({sh.role}) is not unique")
            roles.add((sh.name, sh.role))
            if sh.role not in ROLES:
                p.append(f"link {lk.name!r}: shape {sh.name!r}: role must be one of {ROLES}")
        pm = [n for n, _, _ in lk.point_masses]
        if len(set(pm)) != len(pm):
            p.append(f"link {lk.name!r}: point mass names are not unique")
    seen = set()
    for j in m.joints:
        if not j.name:
            p.append("a joint without a name")
        elif j.name in seen:
            p.append(f"joint {j.name!r}: the name is not unique")
        seen.add(j.name)
        if j.kind not in JOINTS:
            p.append(f"joint {j.name!r}: kind must be one of {JOINTS}")
    return p


def text(m: Machine) -> str:
    """The machine's YX2 text (validated)."""
    problems = problems_of(m)
    if problems:
        raise Invalid(problems)
    stage = stage_of(m)
    problems = validate_stage(stage)
    if problems:
        raise Invalid(problems)
    return stage.GetRootLayer().ExportToString()


def write(m: Machine, path, rep: Report | None = None) -> Report:
    path = Path(path)
    rep = rep or Report("usd", str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text(m))
    return rep


def read(path) -> tuple[Machine, Report]:
    """A validated ``.yx2`` as a machine (``yax.Invalid`` lists the problems of one that is not)."""
    from pxr import Usd
    path = Path(path)
    stage = Usd.Stage.Open(_layer_from_text(path.read_text()))
    problems = validate_stage(stage)
    if problems:
        raise Invalid(problems)
    return hub.from_stage(stage), Report(str(path), "usd")
