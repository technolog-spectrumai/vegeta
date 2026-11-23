"""JSBSim aircraft (``<fdm_config>``). An aircraft is one rigid body: ``mass_balance`` becomes the machine's one link
(empty mass, CG and inertia plus the point masses, in SI and Yax's frame); every other section — metrics,
aerodynamics, propulsion, ground reactions, systems, flight control — is kept verbatim in ``yax:data`` (with the
engine, thruster and system files it references), so a round trip through USD loads in JSBSim and flies the same.

Frames: JSBSim's structural frame is x aft, y right, z up (inches by default); Yax's machine frame is x forward,
y left, z up: x = −x_s, y = −y_s, z = z_s. JSBSim's body axes are x forward, y right, z down; its inertia input
(measured on the C172X against JSBSim's own totals) is the tensor [[ixx, −ixy, ixz], [−ixy, iyy, −iyz], [ixz, −iyz, izz]]
in body axes.
"""
from __future__ import annotations

import copy
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .. import propulsion as pr
from ..hub import Link, Machine, Report

LENGTH = {"IN": 0.0254, "FT": 0.3048, "M": 1.0, "CM": 0.01, "MM": 0.001}
MASS = {"LBS": 0.45359237, "KG": 1.0, "SLUG": 14.593902937206364}
INERTIA = {"SLUG*FT2": 14.593902937206364 * 0.3048 ** 2, "KG*M2": 1.0}
_T = np.diag([1.0, -1.0, -1.0])                    # body (x fwd, y right, z down) ↔ machine (x fwd, y left, z up)
_S = np.diag([-1.0, -1.0, 1.0])                    # structural (x aft, y right, z up) → machine


def _num(el, tag, table, default_unit, default=0.0):
    t = el.find(tag)
    if t is None or not (t.text or "").strip():
        return default
    return float(t.text) * table[(t.get("unit") or default_unit).upper()]


def _loc(el) -> np.ndarray:
    """A <location> (structural, inches by default) in the machine frame [m]."""
    k = LENGTH[(el.get("unit") or "IN").upper()]
    return _S @ (np.array([float(el.find(a).text) for a in "xyz"]) * k)


def _body_tensor(el, unit_default="SLUG*FT2") -> np.ndarray:
    """JSBSim's ixx … iyz of ``el`` as the inertia tensor in the machine frame [kg m²]."""
    g = {k: _num(el, k, INERTIA, unit_default) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")}
    Jb = np.array([[g["ixx"], -g["ixy"], g["ixz"]], [-g["ixy"], g["iyy"], -g["iyz"]], [g["ixz"], -g["iyz"], g["izz"]]])
    return _T @ Jb @ _T


def _shift(m, d) -> np.ndarray:
    d = np.asarray(d, float)
    return m * (d @ d * np.eye(3) - np.outer(d, d))


def _find(name: str, kind: str, aircraft_dir: Path, root: Path | None):
    cands = {"engine": [aircraft_dir / "Engines", (root / "engine") if root else None],
             "system": [aircraft_dir / "Systems", aircraft_dir, (root / "systems") if root else None],
             "aircraft": [aircraft_dir]}[kind]
    for d in filter(None, cands):
        p = d / (name if name.endswith(".xml") else name + ".xml")
        if p.exists():
            return p
    return None


def read(path, *, root=None) -> tuple[Machine, Report]:
    """Read an aircraft file; ``root`` (JSBSim's root directory, default: two levels above the aircraft directory)
    locates engine and system files."""
    path = Path(path)
    tree = ET.parse(path)
    fdm = tree.getroot()
    if fdm.tag != "fdm_config":
        raise ValueError(f"{path}: not a JSBSim aircraft (<fdm_config>)")
    rep = Report(str(path), "usd")
    root = Path(root) if root else path.parent.parent.parent
    mb = fdm.find("mass_balance")
    if mb is None:
        raise ValueError(f"{path}: no <mass_balance>")
    me = _num(mb, "emptywt", MASS, "LBS")
    cg_el = next((l for l in mb.findall("location") if l.get("name", "CG") == "CG"), None)
    ecg = _loc(cg_el) if cg_el is not None else np.zeros(3)
    Je = _body_tensor(mb)
    pms = []
    for pm in mb.findall("pointmass"):
        kg = _num(pm, "weight", MASS, "LBS")
        J = _body_tensor(pm) if any(pm.find(k) is not None for k in ("ixx", "iyy", "izz")) else np.zeros((3, 3))
        if pm.find("form") is not None:
            rep.drop(f"point mass {pm.get('name')}: its <form> shape (kept verbatim; its inertia as JSBSim computes it is not in the link)")
        pms.append({"name": pm.get("name") or f"pm{len(pms)}", "kg": kg, "pos": _loc(pm.find("location")).tolist(), "inertia": J.tolist()})
    M = me + sum(p["kg"] for p in pms)
    com = (me * ecg + sum(p["kg"] * np.asarray(p["pos"]) for p in pms)) / M
    J = Je + _shift(me, ecg - com) + sum(np.asarray(p["inertia"]) + _shift(p["kg"], np.asarray(p["pos"]) - com) for p in pms)
    link = Link(name="airframe", mass=float(M), com=tuple(com), inertia=J,
                point_masses=[(p["name"], p["kg"], tuple(p["pos"])) for p in pms])
    sections = []
    for el in fdm:
        if el.tag == "mass_balance":
            sections.append(["mass_balance", None])
            continue
        sections.append([el.tag, ET.tostring(el, encoding="unicode")])
    files = {}
    for el in fdm.iter():
        f = el.get("file")
        if not f:
            continue
        kind = "engine" if el.tag in ("engine", "thruster") else "system" if el.tag in ("system", "autopilot", "flight_control") else None
        if kind is None:
            continue
        found = _find(f, kind, path.parent, root)
        if found is None:
            rep.drop(f"<{el.tag} file={f!r}>: not found (the converted aircraft needs it beside it)")
            continue
        where = "aircraft" if found.parent == path.parent else kind              # written back where JSBSim looks
        files[f"{where}/{found.name}"] = found.read_text()
    mb_rest = [ET.tostring(c, encoding="unicode") for c in mb
               if c.tag not in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz", "emptywt", "location", "pointmass")]
    metrics = fdm.find("metrics")
    si = {}
    if metrics is not None:
        for tag, table, unit in (("wingarea", {"FT2": 0.09290304, "M2": 1.0, "IN2": 0.00064516}, "FT2"), ("wingspan", LENGTH, "FT"),
                                 ("chord", LENGTH, "FT")):
            if metrics.find(tag) is not None:
                si[tag] = _num(metrics, tag, table, unit)
        for loc in metrics.findall("location"):
            si[loc.get("name")] = _loc(loc).tolist()
    m = Machine(name=Path(path).stem, links=[link])
    m.data["jsbsim"] = {"attrib": dict(fdm.attrib), "sections": sections, "files": files, "mass_balance_extra": mb_rest,
                        "empty": {"kg": me, "cg": ecg.tolist(), "inertia": Je.tolist()}, "point_masses": pms, "metrics_si": si,
                        "title": fdm.get("name")}
    return m, rep


def _rigid(m: Machine, rep: Report) -> tuple:
    """The whole machine as one rigid body at q = 0: mass, centre of mass, inertia about it (machine frame)."""
    M = sum(lk.mass or 0.0 for lk in m.links)
    if M <= 0:
        return 0.0, np.zeros(3), np.zeros((3, 3))
    from .. import hub
    parts = []
    for lk in m.links:
        if not lk.mass:
            continue
        Rl = hub.R(lk.quat)
        c = np.asarray(lk.pos) + Rl @ np.asarray(lk.com)
        I = Rl @ np.asarray(lk.inertia if lk.inertia is not None else np.zeros((3, 3))) @ Rl.T
        parts.append((lk.mass, c, I))
    com = sum(mm * c for mm, c, _ in parts) / M
    J = sum(I + _shift(mm, c - com) for mm, c, I in parts)
    if len(m.links) > 1 or m.joints:
        rep.drop("links and joints (a JSBSim aircraft is one rigid body: their mass at q = 0 is lumped into mass_balance)")
    return M, com, J


def _fmt(v):
    return f"{float(v):.17g}"


def write(m: Machine, path, rep: Report | None = None) -> Report:
    """Write ``m`` as a JSBSim aircraft at ``path`` (its engine and system files in ``Engines/`` and ``Systems/``
    beside it). The mass balance is written in SI from the link(s); the other sections come back verbatim."""
    path = Path(path)
    rep = rep or Report("usd", str(path))
    js = m.data.get("jsbsim")
    M, com, J = _rigid(m, rep)
    pms = js.get("point_masses", []) if js else []
    me = M - sum(p["kg"] for p in pms)
    ecg = (M * com - sum(p["kg"] * np.asarray(p["pos"]) for p in pms)) / me if me > 0 else com
    Je = J - _shift(me, ecg - com) - sum(np.asarray(p["inertia"]) + _shift(p["kg"], np.asarray(p["pos"]) - com) for p in pms)
    Jb = _T @ Je @ _T
    vals = {"ixx": Jb[0, 0], "iyy": Jb[1, 1], "izz": Jb[2, 2], "ixy": -Jb[0, 1], "ixz": Jb[0, 2], "iyz": -Jb[1, 2]}
    # JSBSim's native units (its own SI conversion factors are approximate: KG*M2 is 9e-5 off)
    ki, km, kl = INERTIA["SLUG*FT2"], MASS["LBS"], LENGTH["IN"]
    mb = ET.Element("mass_balance")
    for k, v in vals.items():
        ET.SubElement(mb, k, unit="SLUG*FT2").text = _fmt(v / ki)
    ET.SubElement(mb, "emptywt", unit="LBS").text = _fmt(me / km)
    s = _S @ ecg / kl                                                          # back to the structural frame
    loc = ET.SubElement(mb, "location", name="CG", unit="IN")
    for a, v in zip("xyz", s):
        ET.SubElement(loc, a).text = _fmt(v)
    for p in pms:
        e = ET.SubElement(mb, "pointmass", name=p["name"])
        Jp = _T @ np.asarray(p["inertia"]) @ _T
        if np.any(Jp):
            for k, v in {"ixx": Jp[0, 0], "iyy": Jp[1, 1], "izz": Jp[2, 2], "ixy": -Jp[0, 1], "ixz": Jp[0, 2], "iyz": -Jp[1, 2]}.items():
                ET.SubElement(e, k, unit="SLUG*FT2").text = _fmt(v / ki)
        ET.SubElement(e, "weight", unit="LBS").text = _fmt(p["kg"] / km)
        pl = ET.SubElement(e, "location", unit="IN")
        for a, v in zip("xyz", _S @ np.asarray(p["pos"]) / kl):
            ET.SubElement(pl, a).text = _fmt(v)
    if js:
        for extra in js.get("mass_balance_extra", []):
            mb.append(ET.fromstring(extra))
    fdm = ET.Element("fdm_config", (js or {}).get("attrib") or {"name": m.name, "version": "2.0", "release": "ALPHA"})
    files = dict((js or {}).get("files", {}))
    if js:
        for tag, raw in js["sections"]:
            fdm.append(mb if tag == "mass_balance" else ET.fromstring(raw))
    else:
        blk = pr.block(m)
        rep.drop(("aerodynamics, ground reactions" if blk else "aerodynamics, propulsion, ground reactions")
                 + " (the source has none: a mass-and-balance shell, which JSBSim needs metrics for)")
        met = ET.SubElement(fdm, "metrics")
        for tag, unit in (("wingarea", "M2"), ("wingspan", "M"), ("chord", "M")):
            ET.SubElement(met, tag, unit=unit).text = "1"
        for name in ("AERORP", "EYEPOINT", "VRP"):
            l = ET.SubElement(met, "location", name=name, unit="M")
            for a, v in zip("xyz", _S @ com):
                ET.SubElement(l, a).text = _fmt(v)
        fdm.append(mb)
        if blk:
            fdm.append(_propulsion_xml(blk, rep, files))
        aero = ET.SubElement(fdm, "aerodynamics")                               # JSBSim loads nothing without them
        ET.SubElement(ET.SubElement(ET.SubElement(aero, "axis", name="DRAG"), "function", name="aero/force/zero"), "value").text = "0"
        ET.SubElement(fdm, "ground_reactions")
    if js and pr.block(m):
        rep.drop("propulsion part data: not written (the aircraft's own propulsion section is kept verbatim)")
    from .. import hub
    hub.report_data(m, rep, keep=("jsbsim", pr.KEY))
    for lk in m.links:
        if lk.shapes:
            rep.drop(f"{lk.name}: its shapes (a JSBSim aircraft has no geometry)")
            break
    ET.indent(fdm, space="  ")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('<?xml version="1.0"?>\n' + ET.tostring(fdm, encoding="unicode") + "\n")
    for rel, text in files.items():
        kind, name = rel.split("/", 1)
        d = path.parent / {"engine": "Engines", "system": "Systems", "aircraft": "."}[kind]
        d.mkdir(exist_ok=True)
        (d / name).write_text(text)
    return rep


# ------------------------------------------------------------------------------------- propellers from part data
def _orient(axis) -> tuple:
    """JSBSim's thruster orientation (roll, pitch, yaw) [deg]: its +x (the thrust axis) along ``axis`` (machine
    frame). Body axes are x forward, y right, z down: the x axis after yaw ψ and pitch θ is (cos θ cos ψ, cos θ sin ψ,
    −sin θ)."""
    b = _T @ (np.asarray(axis, float) / np.linalg.norm(axis))
    return 0.0, math.degrees(math.asin(max(-1.0, min(1.0, -b[2])))) + 0.0, math.degrees(math.atan2(b[1], b[0])) + 0.0


def _location(parent, pos, unit="IN"):
    l = ET.SubElement(parent, "location", unit=unit)
    for a, v in zip("xyz", _S @ np.asarray(pos, float) / LENGTH[unit]):
        ET.SubElement(l, a).text = _fmt(v)


def _table(parent, name, x, y):
    t = ET.SubElement(parent, "table", name=name, type="internal")
    ET.SubElement(t, "tableData").text = "\n" + "".join(f"      {_fmt(a)}  {_fmt(b)}\n" for a, b in zip(x, y)) + "    "


def _propulsion_xml(blk: dict, rep: Report, files: dict):
    """The <propulsion> section of the ``propulsion`` part data (``yax.propulsion``): one electric engine per rotor,
    its propeller's C_THRUST and C_POWER against the advance ratio; the engine and propeller files go to ``files``.
    JSBSim's sense is +1 for a propeller turning clockwise seen from behind (its reaction torque on the airframe is
    −sense along the thrust axis); Yax's spin is +1 when the drag torque on the frame is along +axis: sense = −spin."""
    sec = ET.Element("propulsion")
    props = blk.get("propellers", {})
    for r in pr.rotors(blk):
        prop = props.get(r.get("propeller") or "")
        if prop is None:
            raise ValueError(f"rotor {r['name']}: no propeller {r.get('propeller')!r} in propulsion.propellers")
        pname, mname = r["propeller"], r.get("motor") or f"{r['name']}_motor"
        if f"engine/prop_{pname}.xml" not in files:
            co, note = pr.coefficients(prop)
            if note:
                rep.drop(f"propulsion: propeller {pname}: {note}")
            pe = ET.Element("propeller", name=pname)
            ET.SubElement(pe, "ixx", unit="SLUG*FT2").text = _fmt(float(prop.get("inertia_kgm2", 0.0)) / INERTIA["SLUG*FT2"])
            ET.SubElement(pe, "diameter", unit="IN").text = _fmt(float(prop["diameter_m"]) / LENGTH["IN"])
            ET.SubElement(pe, "numblades").text = str(int(prop.get("blades", 2)))
            _table(pe, "C_THRUST", co["J"], co["CT"])
            _table(pe, "C_POWER", co["J"], co["CP"])
            ET.indent(pe, space="  ")
            files[f"engine/prop_{pname}.xml"] = '<?xml version="1.0"?>\n' + ET.tostring(pe, encoding="unicode") + "\n"
        if f"engine/motor_{mname}.xml" not in files:
            P, note = pr.motor_power(blk, r, prop)
            if note:
                rep.drop(f"propulsion: {note}")
            me = ET.Element("electric_engine", name=mname)
            ET.SubElement(me, "power", unit="WATTS").text = _fmt(P)
            ET.indent(me, space="  ")
            files[f"engine/motor_{mname}.xml"] = '<?xml version="1.0"?>\n' + ET.tostring(me, encoding="unicode") + "\n"
        axis = r.get("axis", (1.0, 0.0, 0.0))
        roll, pitch, yaw = _orient(axis)
        eng = ET.SubElement(sec, "engine", file=f"motor_{mname}")       # JSBSim places the thruster, not the engine
        th = ET.SubElement(eng, "thruster", file=f"prop_{pname}")
        ET.SubElement(th, "sense").text = _fmt(-float(r.get("spin", -1)))
        _location(th, r.get("pos", (0.0, 0.0, 0.0)))
        o = ET.SubElement(th, "orient", unit="DEG")
        for k, v in (("roll", roll), ("pitch", pitch), ("yaw", yaw)):
            ET.SubElement(o, k).text = _fmt(v)
    return sec
