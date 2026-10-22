"""Kestrel-4Q — the structure in Talos (CalculiX FEA): the carbon frame as solids, the printed spar–post joint
(notebook 40 § 8b, run in notebook 39's campaign).

``kestrel_structure`` answers the bracing question with a plane frame of beams and bars; this module puts the same
structure through Talos as solid tubes — the two spars through the pod, the posts at the motor stations, the X of
diagonals (``bracing`` ``x`` / ``posts`` / ``none``) — under the same loads, so the plane frame's spar stress, tip
deflection and first modes have a 3-D check, and the two things the frame cannot see get a number: the **torsion**
of the wing pair when the diagonal pairs' thrust is split (the heading manoeuvre: one diagonal pair at full thrust,
the other at idle) and the stress at the **joints**.

Cases per bracing (``frame_cases``): **A** the 4 g pull-up, the lift elliptic along both spars, the wings' and the
motors' inertia against it; **B** the same with the upper wing carrying 75 % (one wing stalling first); **C** the
heading manoeuvre in the hover (M1 and M4 at full thrust, M2 and M3 at idle, 1 g); **modes** with the motors as point
masses. The pod clamps both spars over its width. The **joint** (``joint_cases``): a printed PETG clamp around the
spar with the post's socket, under the post's largest axial force (ultimate) — the across-layer tension is the
number a printed part fails on.

Units mm-N-MPa (masses in tonnes). Everything is prepared whether or not CalculiX is installed; ``solve(run=False)``
lists NOT RUN, and a case that fails is a result in the table like the CFD's. The CAD (``frame_cad``, ``joint_cad``)
is structural, not the aircraft's skin: tubes and a clamp, meshed as solids.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import kestrel
import kestrel_structure as kst
import kestrel_systems as ks

CARBON = dict(name="carbon tube (axial)", E=kst.CARBON["E"] / 1e6, nu=0.30, rho=kst.CARBON["rho"] * 1e-12, strength=kst.CARBON["sigma_allow"] / 1e6,
              source="pultruded carbon tube, axial modulus 80 GPa, allowable 400 MPa with knock-down (kestrel_structure.CARBON)")
PETG = dict(name="PETG printed (in the layer plane)", E=2000.0, nu=0.38, rho=1.25e-9, strength=45.0, strength_z=27.0,
            source="PETG filament datasheet class; the layer-adhesion strength 60 % of the in-plane (assumed, as NISUS)")
KNOCKDOWN = {"carbon": 0.80, "petg": 0.70}
ULTIMATE = 1.5
N_SEG = 6                         # lift segments per half span (the plane frame's stations)
BRACINGS = {"x": "braced", "posts": "posts", "none": "none"}      # CAD name -> kestrel_structure name


def talos_material(m: dict):
    from vegeta import talos
    return talos.Material(m["name"], youngs_modulus=m["E"], poissons_ratio=m["nu"], density=m["rho"], yield_strength=m["strength"], source=m["source"])


class FEACase:
    """One Talos model with what its result is read against."""

    def __init__(self, name, part, description, model, *, kind="static", strength=0.0, strength_z=None, build_axis=None, exclude=None, bracing=None):
        self.name, self.part, self.description, self.model = name, part, description, model
        self.kind, self.strength, self.strength_z, self.build_axis, self.exclude, self.bracing = kind, strength, strength_z, build_axis, exclude, bracing


# ------------------------------------------------------------------------------------------------- the CAD
def _cyl(a, b, d):
    import cadquery as cq
    a, b = cq.Vector(*a), cq.Vector(*b)
    return cq.Workplane("XY").add(cq.Solid.makeCylinder(d / 2, (b - a).Length, a, b - a))


def _shorten(a, b, by):
    import cadquery as cq
    a, b = cq.Vector(*a), cq.Vector(*b)
    u = (b - a).normalized()
    return (a + u * by).toTuple(), (b - u * by).toTuple()


REGION_MARGIN = 3.0               # mm around a face's bounding box for the region boxes (Gmsh's boxes are not exact)
MOTOR_BAND = 7.0                  # mm: the motor station's load band on each spar, ± about motor_y


def spar_breaks(p=None) -> list:
    """Where each spar tube is split into separate cylinder faces [mm]: the tips, the pod's edges, the lift
    segments' stations (``lift_segments``) and the motor station's band — so that every region box of
    ``frame_cases`` contains **whole** faces (Talos's ``SurfacesInBox`` selects faces lying completely inside the box;
    an unsplit tube is one face from tip to tip and no box of a station would ever hold it)."""
    L = kestrel.layout(p)
    b2, ym, yin = L["span"] / 2 * 1e3, L["motor_y"] * 1e3, L["pod_width"] / 2 * 1e3
    stations = sorted(set(np.round(np.r_[np.linspace(0.0, b2, N_SEG + 1), ym, yin], 6)))
    br = {-b2, b2}
    for st in stations:
        br |= {st, -st}
    br |= {ym - MOTOR_BAND, ym + MOTOR_BAND, -ym - MOTOR_BAND, -ym + MOTOR_BAND}
    return sorted(float(v) for v in br)


def frame_cad(p=None, bracing: str = "x"):
    """The carbon frame as hollow tubes (mm, the aircraft's frame: x aft, y right, z up): both spars tip to tip at the
    spar station, the posts at the motor stations, the diagonals of the X (``bracing`` ``x``), or posts only, or none.
    Each spar is built from cylinder segments between ``spar_breaks`` and the unions keep the seams (``clean=False``),
    so the outer surface is one face per segment — the faces the load regions select. The struts' outer cylinders
    reach the spars' axes (a saddle's overlap) and their bores stop short of the spars, so the spars' walls are whole;
    the bores are cut from the union last, which keeps the solid valid."""
    p = kestrel.resolve(p)
    L = kestrel.layout(p)
    h, b2, ym, yin, x = L["h"] * 1e3, L["span"] / 2 * 1e3, L["motor_y"] * 1e3, L["pod_width"] / 2 * 1e3, L["x_spar"] * 1e3
    od, idd = (v * 1e3 for v in kst.TUBES["spar"])
    br = spar_breaks(p)
    outers = [_cyl((x, a, z), (x, b, z), od) for z in (h, -h) for a, b in zip(br[:-1], br[1:])]
    inners = [_cyl((x, -b2 - 1, h), (x, b2 + 1, h), idd), _cyl((x, -b2 - 1, -h), (x, b2 + 1, -h), idd)]
    struts = []
    if bracing in ("x", "posts"):
        po, pi = (v * 1e3 for v in kst.TUBES["post"])
        for s in (1, -1):
            struts.append(((x, s * ym, -h), (x, s * ym, h), po, pi))
    if bracing == "x":
        do, di = (v * 1e3 for v in kst.TUBES["diagonal"])
        for s in (1, -1):
            struts.append(((x, s * yin, h), (x, s * ym, -h), do, di))
            struts.append(((x, s * yin, -h), (x, s * ym, h), do, di))
    for a, b, d_o, d_i in struts:
        outers.append(_cyl(a, b, d_o))
        a2, b2_ = _shorten(a, b, od / 2 + 1.0)
        inners.append(_cyl(a2, b2_, d_i))
    out = outers[0]
    for o in outers[1:]:
        out = out.union(o, clean=False)
    for i in inners:
        out = out.cut(i, clean=False)
    return out


def frame_faces(p=None, bracing: str = "x") -> list:
    """The frame's faces as (xmin, ymin, zmin, xmax, ymax, zmax, type) — what Gmsh's bounding-box selection sees;
    ``regions_select`` checks the region boxes against it without a mesher."""
    out = []
    for f in frame_cad(p, bracing).val().Faces():
        bb = f.BoundingBox()
        out.append((bb.xmin, bb.ymin, bb.zmin, bb.xmax, bb.ymax, bb.zmax, f.geomType()))
    return out


def regions_select(regions: list, faces: list) -> dict:
    """``{region name: number of faces completely inside its box}`` for Talos ``SurfacesInBox`` regions against
    ``frame_faces`` (or any list of face boxes): zero is the error Talos raises at meshing."""
    out = {}
    for r in regions:
        b = r.box
        out[r.name] = sum(1 for f in faces if f[0] >= b[0] and f[1] >= b[1] and f[2] >= b[2] and f[3] <= b[3] and f[4] <= b[4] and f[5] <= b[5])
    return out


def frame_step(workdir, p=None, bracing: str = "x") -> Path:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    path = workdir / f"kestrel_frame_{bracing}.step"
    if not path.exists():
        frame_cad(p, bracing).val().exportStep(str(path))
    return path


JOINT_FLOOR = 2.0                 # mm of printed material between the socket's floor and the spar bore


def joint_cad(p=None, *, size=(36.0, 26.0, 30.0), wall: float = 3.0):
    """The printed spar–post joint: a block clamped around the spar (bore along y) with the post's socket (bore along
    z from the top face down to a floor ``JOINT_FLOOR`` above the spar bore — the post's end bears on that floor and
    the force goes through the printed material into the spar's saddle), mm; the build axis z."""
    import cadquery as cq
    p = kestrel.resolve(p)
    od_s = kst.TUBES["spar"][0] * 1e3
    od_p = kst.TUBES["post"][0] * 1e3
    lx, ly, lz = size
    block = cq.Workplane("XY").box(lx, ly, lz)
    spar = cq.Workplane("XZ").circle(od_s / 2 + 0.1).extrude(ly + 2.0, both=True)
    z_floor = od_s / 2 + 0.1 + JOINT_FLOOR
    post = cq.Workplane("XY").workplane(offset=z_floor).circle(od_p / 2 + 0.1).extrude(lz)
    return block.cut(spar).cut(post)


def joint_floor_z(p=None) -> float:
    return kst.TUBES["spar"][0] * 1e3 / 2 + 0.1 + JOINT_FLOOR


def joint_step(workdir, p=None) -> Path:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    path = workdir / "kestrel_joint.step"
    if not path.exists():
        joint_cad(p).val().exportStep(str(path))
    return path


# ------------------------------------------------------------------------------------------------- the loads
def masses(p=None) -> dict:
    """What hangs on the frame [kg]: one motor unit (motor, propeller, ESC, nacelle), each wing's panels."""
    L = kestrel.layout(p)
    c = ks.COMPONENTS
    return {"motor_unit": c["motor"]["mass_kg"] + c["propeller"]["mass_kg"] + c["esc"]["mass_kg"] + ks.NACELLE_KG,
            "wing": ks.WING_AREAL_KG_M2 * L["S_wing"] + ks.SPAR_KG_M * L["span"]}


def lift_segments(p=None, *, n: float, mass_kg: float, upper_share: float = 0.5) -> list:
    """The lift (elliptic) less the wing's inertia on each spar segment of each wing at load factor ``n``: rows
    (wing, y0, y1, force_N) for both halves, as the plane frame distributes it."""
    L = kestrel.layout(p)
    b2 = L["span"] / 2
    m = masses(p)
    stations = sorted(set(np.round(np.r_[np.linspace(0.0, b2, N_SEG + 1), L["motor_y"], L["pod_width"] / 2], 6)))
    rows = []
    for wing, share in (("upper", upper_share), ("lower", 1 - upper_share)):
        half = n * mass_kg * ks.G / 2 * share
        for a, b in zip(stations[:-1], stations[1:]):
            ya, yb = a / b2, b / b2
            frac = (2 / math.pi) * ((yb * math.sqrt(1 - yb ** 2) + math.asin(yb)) - (ya * math.sqrt(1 - ya ** 2) + math.asin(ya)))
            inert = n * ks.G * m["wing"] / 2 * (b - a) / b2
            for s in (1, -1):
                rows.append((wing, s * a if s > 0 else -b, s * b if s > 0 else -a, half * frac - inert))
    return rows


def frame_cases(workdir, p=None, *, mass_kg: float, bracing: str = "x", n: float = 4.0, element_size: float = 2.0, drive=None) -> list:
    """The frame's Talos cases for one bracing: A, B, C and the modes (see the module's text)."""
    from vegeta import talos
    p = kestrel.resolve(p)
    L = kestrel.layout(p)
    step = frame_step(Path(workdir) / "cad", p, bracing)
    h, ym, yin, x = L["h"] * 1e3, L["motor_y"] * 1e3, L["pod_width"] / 2 * 1e3, L["x_spar"] * 1e3
    r, m = kst.TUBES["spar"][0] * 1e3 / 2 + REGION_MARGIN, REGION_MARGIN
    regions = [talos.SurfacesInBox("pod_upper", (x - r, -yin - m, h - r, x + r, yin + m, h + r)),
               talos.SurfacesInBox("pod_lower", (x - r, -yin - m, -h - r, x + r, yin + m, -h + r))]
    motors = {"M1": (-1, 1), "M2": (1, 1), "M3": (-1, -1), "M4": (1, -1)}
    for name, (sy, sz) in motors.items():
        regions.append(talos.SurfacesInBox(name, (x - r, sy * ym - MOTOR_BAND - m, sz * h - r, x + r, sy * ym + MOTOR_BAND + m, sz * h + r)))
    segs = {}
    for case_key, share in (("A", 0.5), ("B", 0.75)):
        segs[case_key] = lift_segments(p, n=n, mass_kg=mass_kg, upper_share=share)
    seg_regions = {}
    for i, (wing, y0, y1, _) in enumerate(segs["A"]):
        z = h if wing == "upper" else -h
        ya, yb = sorted((y0 * 1e3, y1 * 1e3))
        if abs(ya - ym) < 1e-6 or abs(ya + ym) < 1e-6 or abs(yb) == ym:      # the segments touching the motor station: the band is the motor's
            lo, hi = ya + (MOTOR_BAND if abs(ya) == ym else 0.0), yb - (MOTOR_BAND if abs(yb) == ym else 0.0)
        else:
            lo, hi = ya, yb
        name = f"lift_{wing[0]}{i:02d}"
        seg_regions[i] = name
        regions.append(talos.SurfacesInBox(name, (x - r, lo - m, z - r, x + r, hi + m, z + r)))
    m = masses(p)
    mu_t = m["motor_unit"] * 1e-3                      # tonnes
    mesh = talos.MeshSettings(element_size=element_size, order=2)
    mat = talos_material(CARBON)
    supports = [talos.FixedSupport("pod_upper"), talos.FixedSupport("pod_lower")]
    out = []
    for key in ("A", "B"):
        loads = [talos.Force(seg_regions[i], fz=ULTIMATE * f) for i, (_, _, _, f) in enumerate(segs[key])]
        loads += [talos.Force(mn, fz=-ULTIMATE * n * ks.G * m["motor_unit"]) for mn in motors]
        desc = {"A": f"{n:g} g pull-up, the lift elliptic on both wings, the motors' and wings' inertia against it (ultimate ×{ULTIMATE})",
                "B": f"{n:g} g with the upper wing carrying 75 % (one wing stalling first) (ultimate ×{ULTIMATE})"}[key]
        out.append(FEACase(f"frame_{bracing}_{key}", "frame", desc, talos.StructuralModel(step, "mm-N-MPa", mat, regions, supports, loads, mesh, name=f"frame_{bracing}_{key}"),
                           strength=CARBON["strength"], bracing=bracing))
    d = drive or ks.drive_table()
    T_hi, T_lo = float(d.T_max[0]), float(d.T_min)
    loads = [talos.Force(mn, fx=-(T_hi if mn in ("M1", "M4") else T_lo), fz=-ks.G * m["motor_unit"]) for mn in motors]
    out.append(FEACase(f"frame_{bracing}_C", "frame", f"the heading manoeuvre in the hover: M1 and M4 at {T_hi:.1f} N, M2 and M3 at {T_lo:.2f} N (the diagonal pairs split), 1 g",
                       talos.StructuralModel(step, "mm-N-MPa", mat, regions, supports, loads, mesh, name=f"frame_{bracing}_C"), strength=CARBON["strength"], bracing=bracing))
    pm = [talos.PointMass(mn, mu_t) for mn in motors]
    out.append(FEACase(f"frame_{bracing}_modes", "frame", "the first modes with the motor units as point masses at the stations",
                       talos.StructuralModel(step, "mm-N-MPa", mat, regions, supports, [], mesh, masses=pm, name=f"frame_{bracing}_modes"), kind="modes", bracing=bracing))
    for c in out:
        c.exclude = lambda xyz, yin=yin, ym=ym: (np.abs(xyz[:, 1]) < yin + 12.0) | (np.abs(np.abs(xyz[:, 1]) - ym) < 12.0)
    return out


def post_force(p=None, *, mass_kg: float, n: float = 4.0) -> float:
    """The post's largest axial force [N] in the plane frame's cases (even lift and the 75 % wing), limit load."""
    best = 0.0
    for share in (0.5, 0.75):
        f = kst.frame(p, n=n, mass_kg=mass_kg, bracing="posts", upper_share=share)
        for b in f["bars"]:
            if b.get("member") == "post":
                best = max(best, abs(float(b["force [N]"])))
    return best


def joint_cases(workdir, p=None, *, mass_kg: float, element_size: float = 1.2) -> list:
    """The printed joint under the post's largest axial force (ultimate) pushing down into its socket, the spar bore
    held: the in-plane stress and the tension across the layers."""
    from vegeta import talos
    p = kestrel.resolve(p)
    step = joint_step(Path(workdir) / "cad", p)
    od_s, od_p = kst.TUBES["spar"][0] * 1e3, kst.TUBES["post"][0] * 1e3
    _, ly, lz = 36.0, 26.0, 30.0
    rs, rp, zf = od_s / 2 + 0.1 + REGION_MARGIN, od_p / 2 + 0.1 + REGION_MARGIN, joint_floor_z(p)
    regions = [talos.SurfacesInBox("spar_bore", (-rs, -ly / 2 - 1, -rs, rs, ly / 2 + 1, rs)),
               talos.SurfacesInBox("socket_floor", (-rp, -rp, zf - 0.5, rp, rp, zf + 0.5))]
    F = ULTIMATE * post_force(p, mass_kg=mass_kg)
    mesh = talos.MeshSettings(element_size=element_size, order=2)
    model = talos.StructuralModel(step, "mm-N-MPa", talos_material(PETG), regions, [talos.FixedSupport("spar_bore")], [talos.Force("socket_floor", fz=-F)], mesh, name="joint_post")
    c = FEACase("joint_post", "joint", f"the post's {F:.1f} N (ultimate) on the socket's floor, the spar bore held; build axis z",
                model, strength=PETG["strength"], strength_z=PETG["strength_z"], build_axis="z")
    c.exclude = lambda xyz, rs=rs: (np.sqrt(xyz[:, 0] ** 2 + xyz[:, 2] ** 2) < rs + 1.5)
    return [c]


def cases(workdir, p=None, *, mass_kg: float, bracings=("x", "posts"), element_size: float = 2.0, drive=None) -> list:
    out = []
    for br in bracings:
        out += frame_cases(workdir, p, mass_kg=mass_kg, bracing=br, element_size=element_size, drive=drive)
    out += joint_cases(workdir, p, mass_kg=mass_kg)
    return out


# ------------------------------------------------------------------------------------------------- solve and read
def solve(cases_: list, workdir, *, run: bool = True, threads: int = 4, n_modes: int = 6, progress: bool = False) -> dict:
    """Mesh and solve each case in ``workdir/<name>``; ``run=False`` reads back what exists and marks the rest NOT RUN.
    A case that fails (no CalculiX, a mesh that will not close, a diverged solve) is a result: its status in the table."""
    out = {}
    it = cases_
    if progress:
        from tqdm.auto import tqdm
        it = tqdm(cases_, desc="FEA")
    for c in it:
        wd = Path(workdir) / c.name
        try:
            if c.kind == "static":
                out[c.name] = c.model.ensure(wd, run=run, threads=threads)
            else:
                if not run and not (wd / "modes.dat").exists():
                    out[c.name] = None
                    continue
                if not (wd / "mesh.msh").exists():
                    c.model.mesh(wd)
                out[c.name] = c.model.solve_modes(wd, n_modes=n_modes, threads=threads)
        except Exception as exc:                          # a failure is a result
            from vegeta.talos.result import Result
            r = Result(kind="talos.static" if c.kind == "static" else "talos.modes", status="failed")
            r.messages.append(f"{type(exc).__name__}: {str(exc)[:200]}")
            out[c.name] = r
    return out


def _status(r) -> str:
    if r is None:
        return "NOT RUN"
    if getattr(r, "ok", False):
        return "solved"
    return "NOT RUN" if any("NOT RUN" in m for m in getattr(r, "messages", [])) else "FAILED"


def _nominal(case, result) -> tuple:
    from vegeta import talos
    f = talos.read_frd(result.artifacts["frd"])
    s = f.stress
    vm = np.sqrt(0.5 * ((s[:, 0] - s[:, 1]) ** 2 + (s[:, 1] - s[:, 2]) ** 2 + (s[:, 2] - s[:, 0]) ** 2) + 3 * (s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2))
    keep = ~case.exclude(f.coords) if case.exclude is not None else np.ones(len(vm), bool)
    return (float(vm[keep].max()) if keep.any() else float("nan")), float(np.percentile(vm, 99.5))


def _across(result, axis) -> float:
    from vegeta import talos
    f = talos.read_frd(result.artifacts["frd"])
    return float(f.stress[:, "xyz".index(axis)].max())


def table(cases_: list, results: dict) -> "pd.DataFrame":
    """One row per case: the status, the peak and the nominal von Mises (away from the clamps and the load
    introductions), the 99.5th percentile, the largest displacement, the margin against the allowable with the
    knock-down; the across-layer tension and its margin for the printed joint; the first frequencies of a modal case.
    A failed case carries its message."""
    import pandas as pd
    rows = {}
    for c in cases_:
        r = results.get(c.name)
        st = _status(r)
        row = {"part": c.part, "bracing": c.bracing or "", "kind": c.kind, "status": st, "description": c.description}
        if st != "solved":
            row["note"] = "" if r is None else (r.messages[-1][:160] if getattr(r, "messages", None) else "")
            rows[c.name] = row
            continue
        m = r.metrics or {}
        if c.kind == "modes":
            fr = list(m.get("frequencies_hz") or [])
            row.update({"f1 [Hz]": fr[0] if fr else np.nan, "f2 [Hz]": fr[1] if len(fr) > 1 else np.nan, "f3 [Hz]": fr[2] if len(fr) > 2 else np.nan, "note": ""})
            rows[c.name] = row
            continue
        vm, disp = m.get("max_von_mises"), m.get("max_displacement")
        try:
            nominal, p995 = _nominal(c, r)
        except Exception:
            nominal, p995 = np.nan, np.nan
        kd = KNOCKDOWN["petg"] if c.strength_z else KNOCKDOWN["carbon"]
        gov = nominal if np.isfinite(nominal) else vm
        row.update({"peak von Mises [MPa]": vm, "nominal von Mises [MPa]": nominal, "99.5th percentile [MPa]": p995, "max displacement [mm]": disp,
                    "allowable [MPa]": c.strength * kd, "margin": c.strength * kd / gov - 1 if gov else np.nan,
                    "reaction [N]": float(np.linalg.norm(m["reaction_total"])) if m.get("reaction_total") is not None else np.nan, "note": ""})
        if c.strength_z:
            try:
                sz = _across(r, c.build_axis)
            except Exception:
                sz = np.nan
            row.update({"across layers [MPa]": sz, "margin (layers)": c.strength_z * kd / sz - 1 if sz and sz > 0 else np.inf})
        rows[c.name] = row
    return pd.DataFrame(rows).T


def summary(tab) -> dict:
    st = tab["status"].astype(str)
    out = {k: int((st == k).sum()) for k in ("solved", "FAILED", "NOT RUN")}
    out["n"] = len(st)
    out["line"] = f"{out['solved']} of {out['n']} FEA cases solved, {out['FAILED']} failed, {out['NOT RUN']} not run"
    return out


def plane_frame_reference(p=None, *, mass_kg: float, n: float = 4.0) -> "pd.DataFrame":
    """What the plane frame (``kestrel_structure``) says for the same cases: the spar's peak stress and tip
    deflection at ``n`` g (even lift and the 75 % wing, limit load) and the first modes per bracing — the numbers
    the FEA is read against (the FEA's loads are ultimate: ×1.5)."""
    import pandas as pd
    rows = {}
    modes = kst.stiffness_and_modes(p)
    for br, label in (("x", "posts + X diagonals"), ("posts", "posts only"), ("none", "no truss (two cantilevers)")):
        for share, key in ((0.5, "A"), (0.75, "B")):
            f = kst.frame(p, n=n, mass_kg=mass_kg, bracing=BRACINGS[br], upper_share=share)
            rows[f"frame_{br}_{key}"] = {"bracing": br, "case": key, "spar stress, limit [MPa]": f["spar stress [MPa]"], "spar stress, ultimate [MPa]": ULTIMATE * f["spar stress [MPa]"],
                                         "tip deflection, upper, limit [mm]": f["tip deflection, upper [mm]"], "tip deflection, ultimate [mm]": ULTIMATE * f["tip deflection, upper [mm]"],
                                         "spar reserve (limit)": f["spar reserve"]}
        if label in modes.index:
            rows[f"frame_{br}_modes"] = {"bracing": br, "case": "modes", "first mode, symmetric [Hz]": float(modes.loc[label, "first mode, symmetric [Hz]"]),
                                         "first mode, antisymmetric [Hz]": float(modes.loc[label, "first mode, antisymmetric [Hz]"])}
    return pd.DataFrame(rows).T


def save(tab, path, *, meta: dict | None = None) -> dict:
    """The FEA table into a JSON (``campaigns["fea"]`` of the CFD results file or its own): statuses, numbers, notes."""
    import json
    rows = {str(k): {kk: (None if vv is None or (isinstance(vv, float) and not np.isfinite(vv)) else (vv.item() if hasattr(vv, "item") else vv))
                     for kk, vv in r.items()} for k, r in tab.to_dict("index").items()}
    doc = {"meta": meta or {}, "summary": {k: v for k, v in summary(tab).items() if k != "line"}, "rows": rows}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, default=str))
    return doc
