"""Kestrel-X-C — the X-wing made competitive (notebook 48): the Kestrel-X at a shallower dihedral with a payload
module, after the civilian X-wing concepts (X-Wing-R/M/U).

What changes against Kestrel-X-A (``kestrel_xwing``) and why:
- **dihedral ±30° instead of ±45°**: a panel gives back cos² φ of its lift vertically — 0.75 instead of 0.50 — and the
  projected span grows; the hover pitch lever shrinks from r sin 45° to r sin 30°, still ahead of the biplane's;
- **panels 0.6 m × 0.15 m** (0.5 × 0.14): the span the shallower X can use, a 23112 reflexed section (the trim drag);
- **faired roots**: the four acute junctions were half of X-A's drag area; at 30° they are blunter and filleted;
- **the payload module inside one blended fuselage**: an elliptical body 162 mm wide and 185 mm deep over 0.76 m
  (fineness 4.4, ``FUSELAGE_STATIONS``: a long nose, the parallel part over the module, a gentle tail cone) carries the
  350 × 150 × 105 mm module on the axis with the battery above it — the medical capsule, a cargo pod or a flotation
  pack (X-Wing-U's idea) — with no box in the free stream and no second junction; the camera sits behind a flush
  window, not in a dome; Kestrel-X's pod could not take the capsule;
- **tapered panels** (root 1.2 and tip 0.8 of the mean chord, the same area) for the span loading, the tip pods as
  the feet: no legs.

Two more variants bound the design: ``X-C45`` (the same airframe at 45°: the dihedral's share) and ``X-CX`` (the
panels 0.21 m in chord instead of 0.15 — longer in x, not in span: 40 % more area for a slower stall, at aspect ratio
2.6 and the induced drag that goes with it). The section's thickness is an argument of ``drag_buildup`` and
``cantilever`` (``t``): thicker panels buy spar depth and cost form drag, nothing in lift slope.

The module gives the layout, the masses, the lattice, the drag, the vehicle for the transition model (all through
``kestrel_xwing`` on the X-C layout), the CAD (``KestrelXC``, Dedalus) for the drawings and the CFD's STL, the CFD
specs and cases (bare sweep, stall) and the FEA case of one panel's spar in its socket — the builders the workflow
book ``kestrel-x`` (code X) runs; the notebook never launches them.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import kestrel
import kestrel_systems as ks
import kestrel_xwing as kx

G = ks.G
RHO, NU = 1.225, 1.46e-5
MAIN = "X-C"
VARIANTS = {"X-C": {"panel_length": 0.60, "chord": 0.15, "motor_r": 0.56, "phi_deg": 30.0, "title": "the shallow X with the module in a blended fuselage"},
            "X-C45": {"panel_length": 0.60, "chord": 0.15, "motor_r": 0.56, "phi_deg": 45.0, "title": "the same airframe at the original 45°: what the dihedral alone is worth"},
            "X-CX": {"panel_length": 0.60, "chord": 0.21, "motor_r": 0.56, "phi_deg": 30.0, "title": "the X-C's panels 40 % longer in x (chord 0.21): the area for a slower stall, the aspect ratio it costs"}}
# the blended fuselage: an elliptical body, its stations (x / length, width / POD_WIDTH, height / POD_HEIGHT); the module
# sits on the axis in the parallel part (0.25–0.72 of the length), the battery and the electronics above it
POD_LENGTH, POD_WIDTH, POD_HEIGHT, X_LE, PROP_X = 0.760, 0.162, 0.185, 0.230, 0.150
FUSELAGE_STATIONS = ((0.0, 0.20, 0.20), (0.07, 0.56, 0.56), (0.16, 0.86, 0.86), (0.25, 1.0, 1.0), (0.72, 1.0, 1.0), (0.82, 0.80, 0.80), (0.92, 0.46, 0.46), (1.0, 0.16, 0.16))
MODULE = (0.350, 0.150, 0.105)        # the payload module [m]: length, width, height, inside the fuselage on its axis
MODULE_X = 0.19                       # its front face from the nose
MODULE_SHELL_KG = 0.100               # the bay's floor and hatch, the longer shell over the pod shell
TAPER = (1.2, 0.8)                    # root and tip chord over the mean chord (the same area as the straight panel)
CAMERA_WINDOW_M2 = 0.4 * math.pi * 0.018 ** 2 * 0.1    # a flush window instead of the dome: a tenth of its drag area
LEG_LENGTH = 0.0                      # no legs: the aircraft stands on its four tip pods
JUNCTION_FACTOR = 0.5                 # the filleted, blunter roots against X-A's acute junction allowance
SECTION = "NACA 23112 (k2/k1 0.03)"


# ================================================================================================= the layout
def layout(variant: str = MAIN) -> dict:
    v = VARIANTS[variant]
    kx.VARIANTS.setdefault(f"xc:{variant}", {"panel_length": v["panel_length"], "chord": v["chord"], "motor_r": v["motor_r"]})
    L = kx.layout(f"xc:{variant}", phi_deg=v["phi_deg"])
    L.update(variant=variant, title=v["title"], pod_length=POD_LENGTH, pod_width=POD_WIDTH, pod_height=POD_HEIGHT, module=MODULE, module_x=MODULE_X,
             x_le=X_LE, x_te=X_LE + v["chord"], x_ac=X_LE + 0.25 * v["chord"], x_spar=X_LE + 0.30 * v["chord"], x_hinge=X_LE + 0.72 * v["chord"], prop_x=PROP_X,
             prop_ahead_of_le=X_LE - PROP_X, root_chord=TAPER[0] * v["chord"], tip_chord=TAPER[1] * v["chord"], x_feet=X_LE + 0.6 * v["chord"] + 0.06, foot_rod_length=0.06, section=SECTION)
    return L


def checks(L: dict) -> list[str]:
    out = kx.checks(L)
    if L["h"] < 0.15:
        out.append(f"the hover pitch lever is {L['h'] * 1e3:.0f} mm: below the biplane's 140 mm plus margin")
    return out


# ================================================================================================= masses, aero, drag, structure
def mass_table(L: dict, **kw) -> list[dict]:
    items = kx.mass_table(L, **kw)
    items.insert(-1, {"item": "module bay (floor, hatch, the deeper shell)", "mass_kg": MODULE_SHELL_KG, "x": MODULE_X + MODULE[0] / 2, "y": 0.0, "z": 0.0, "group": "structure", "rod": None})
    return items


def aero(L: dict) -> dict:
    return kx.aero(L)


def fuselage_geometry(length: float = POD_LENGTH, width: float = POD_WIDTH, height: float = POD_HEIGHT, stations=FUSELAGE_STATIONS) -> dict:
    """The blended fuselage's wetted area and volume from its elliptical stations (the perimeter by Ramanujan,
    the trapezoid rule between stations), its fineness ratio on the equivalent diameter of the largest section."""
    xs = [st[0] * length for st in stations]
    per, area = [], []
    for _, fw, fh in stations:
        a, b = fw * width / 2, fh * height / 2
        h = ((a - b) / (a + b)) ** 2 if a + b else 0.0
        per.append(math.pi * (a + b) * (1 + 3 * h / (10 + math.sqrt(4 - 3 * h))))
        area.append(math.pi * a * b)
    wet = sum(0.5 * (per[i] + per[i + 1]) * (xs[i + 1] - xs[i]) for i in range(len(xs) - 1))
    vol = sum(0.5 * (area[i] + area[i + 1]) * (xs[i + 1] - xs[i]) for i in range(len(xs) - 1))
    d_eq = math.sqrt(width * height)
    return {"length_m": length, "width_m": width, "height_m": height, "wetted_m2": wet, "volume_m3": vol, "d_eq_m": d_eq, "fineness": length / d_eq,
            "frontal_m2": math.pi * width * height / 4}


def drag_buildup(L: dict, *, V: float = 16.0, t: float = 0.12) -> dict:
    """``kestrel_xwing.drag_buildup`` (the panels at thickness ``t``, the nacelles) with the faired roots, the blended
    fuselage on its own wetted area and fineness ratio (the form factor of a streamlined body, the module inside it),
    a flush camera window, the tip skids."""
    import kestrel_flight as kf
    d = kx.drag_buildup(L, V=V, t=t)
    rows = dict(d["parts_m2"])
    rows["wing-pod junctions (4, filleted at 30°)"] = rows.pop("wing-pod junctions (4, acute)") * JUNCTION_FACTOR
    f = fuselage_geometry()
    fr = f["fineness"]
    rows["fuselage with the module inside"] = kf._cf(V * f["length_m"] / NU) * (1 + 60 / fr ** 3 + fr / 400) * f["wetted_m2"]
    rows.pop("pod", None)
    rows["camera window (flush)"] = CAMERA_WINDOW_M2
    rows.pop("camera dome", None)
    rows["foot rods (4), axial"] = 4 * 0.02 * math.pi * 0.004 ** 2 / 4 * 10            # the tip pods' short skids
    total = sum(rows.values()) * kf.INTERFERENCE
    return {"parts_m2": rows, "CdA_m2": total, "Cd0": total / L["S_ref"], "Cd0_on_biplane_Sref": total / 0.28, "fuselage": f}


def cantilever(L: dict, *, mass_kg: float, **kw) -> dict:
    return kx.cantilever(L, mass_kg=mass_kg, **kw)


def stance(L: dict, **kw) -> dict:
    return kx.stance(L, **kw)


def capsule_fit() -> dict:
    cap = (0.160, 0.072, 0.072)
    return {"module [mm]": tuple(round(m * 1e3) for m in MODULE), "capsule [mm]": tuple(round(c * 1e3) for c in cap),
            "capsule fits": all(m >= c + 0.012 for m, c in zip(MODULE, cap)), "capsules in line": int(MODULE[0] // (cap[0] + 0.012)),
            "note": "X-Wing-U's module; the biplane stacks the capsule in its pod, Kestrel-X-A could not take it at all (462 of 430 mm)"}


def make_vehicle(variant: str = MAIN, *, name: str | None = None, extra_kg: float = 0.0, x_cg: float | None = None, elevons: bool = True):
    """The X-C for ``kestrel_transition`` (``kestrel_xwing.make_vehicle`` on this layout, the section's cl max)."""
    import kestrel_flight as kf
    import kestrel_transition as kt
    L = layout(variant)
    table = mass_table(L, x_cg_target=x_cg)
    if extra_kg:
        table.append({"item": "payload", "mass_kg": extra_kg, "x": MODULE_X + MODULE[0] / 2, "y": 0.0, "z": 0.0, "group": "payload", "rod": None})
    ci = ks.cg_inertia(table)
    a = aero(L)
    d = drag_buildup(L)
    pol = kf.Polar(cl_alpha=a["CLa"] / math.cos(L["phi"]) ** 2, cd0=d["Cd0"], k=a["k"] / kf.OSWALD_VISCOUS, cl_max=0.9 * 1.15, cd90=kf.viterna_cd90(L["AR_wing"]), alpha0=-0.03,
                   source="X-C lattice (panel-normal slope), NACA 23112 reflexed cl max")
    phi = L["phi"]
    panels = [{"phi": phi, "sy": sy, "sz": sz, "z": sz * 0.5 * (L["r0"] + L["r1"]) * math.sin(phi), "y0": L["r0"], "y1": L["r1"], "y_motor": L["motor_r"],
               "rotor": "up" if sz > 0 else "lo", "S": L["S_panel"]} for sy, sz in ((-1, 1), (1, 1), (-1, -1), (1, -1))]
    return kt.Vehicle(name or f"Kestrel-{variant}", {"elevons": elevons, "span": 1e3 * L["span_projected"], "chord": 1e3 * L["chord"], "motor_y": 1e3 * L["motor_r"] * math.cos(phi), "gap": 1e3 * max(2 * L["h"], 0.1),
                                                     "pod_length": 1e3 * POD_LENGTH},
                      L, ci["mass_kg"], ci["x_cg"], ci["z_cg"], ci["Iyy"], ci["Ixx"], pol, ks.drive_table(), elevons=elevons, panels=panels,
                      z_rotor=L["motor_r"] * math.sin(phi), elevon_y_in=L["elevon_y_in"], lift_slope_vertical=pol.cl_alpha * math.cos(phi) ** 2)


cruise = kx.cruise


# ================================================================================================= the CAD
def cad_params(variant: str = MAIN) -> dict:
    L = layout(variant)
    return dict(panel_length=L["panel_length"] * 1e3, root_chord=L["root_chord"] * 1e3, tip_chord=L["tip_chord"] * 1e3, motor_r=L["motor_r"] * 1e3, phi_deg=L["phi_deg"], pod_length=POD_LENGTH * 1e3,
                pod_width=POD_WIDTH * 1e3, pod_height=POD_HEIGHT * 1e3, wing_le_x=X_LE * 1e3, prop_x=PROP_X * 1e3, prop_diameter=L["R_prop"] * 2e3,
                module_length=MODULE[0] * 1e3, module_width=MODULE[1] * 1e3, module_height=MODULE[2] * 1e3, module_x=MODULE_X * 1e3, elevon_inner_r=L["elevon_y_in"] * 1e3)


class KestrelXC(kestrel.Design):
    """The X-C: four panels (the reflexed section, extruded along each panel's axis at its dihedral) from a round pod,
    nacelles at the panels' tips, a faired belly module, four tail legs on the lower panels and two tip rods on the
    upper. Parts: ``aircraft`` (with the propeller discs), ``airframe`` (the CFD's body, no discs), ``panel``,
    ``pod``, ``module``."""
    from vegeta.dedalus import Parameter as _P
    parameters = [
        _P("part", "aircraft", choices=("aircraft", "airframe", "panel", "pod", "module")),
        _P("panel_length", 600.0, "mm", min=200), _P("root_chord", 180.0, "mm", min=60), _P("tip_chord", 120.0, "mm", min=40), _P("motor_r", 560.0, "mm", min=100),
        _P("phi_deg", 30.0, "deg", min=10, max=60, description="the panels' dihedral about the pod's axis"),
        _P("pod_length", 760.0, "mm", min=200), _P("pod_width", 162.0, "mm", min=50), _P("pod_height", 185.0, "mm", min=50, description="the blended fuselage's depth: the module on the axis, the battery above it"),
        _P("wing_le_x", 230.0, "mm", min=50), _P("prop_x", 150.0, "mm", min=0),
        _P("prop_diameter", 228.6, "mm", min=100), _P("nacelle_diameter", 32.0, "mm", min=10),
        _P("module_length", 350.0, "mm", min=0), _P("module_width", 150.0, "mm", min=0), _P("module_height", 105.0, "mm", min=0), _P("module_x", 190.0, "mm", min=0),
        _P("elevon_inner_r", 400.0, "mm", min=0), _P("elevon_chord", 0.28, "", min=0.1, max=0.5),
        _P("thickness", 0.12, "", min=0.06, max=0.2), _P("camber", 0.02, "", min=0, max=0.06), _P("camber_pos", 0.3, "", min=0.1, max=0.9),
        _P("show_props", True, ""), _P("angle_of_attack_deg", 0.0, "deg", min=-30, max=120), _P("rotation_x", 240.0, "mm", min=0),
    ]

    def _section_pts(self, p, chord, x_le):
        up, lo = kestrel.naca4(p["camber"], p["camber_pos"], p["thickness"])
        return [(x_le + chord * u, chord * v) for u, v in up] + [(x_le + chord * u, chord * v) for u, v in lo[::-1][1:]]

    def _chord_at(self, p, r):
        r0, r1 = p["pod_width"] / 2 - 6.0, p["pod_width"] / 2 + p["panel_length"]
        t = (r - r0) / (r1 - r0)
        return p["root_chord"] + (p["tip_chord"] - p["root_chord"]) * t

    def _x_le_at(self, p, r):
        """The leading edge keeps the quarter-chord line straight: the taper is shared between the edges."""
        c = self._chord_at(p, r)
        return p["wing_le_x"] + 0.25 * (p["root_chord"] - c)

    def _panel(self, p, angle_deg):
        """One tapered panel: the section at the root and the tip lofted along +y, the elevon bay cut, then rolled
        about the pod's axis by ``angle_deg`` (0 = to the right, 90 = up)."""
        import cadquery as cq
        r0, r1 = p["pod_width"] / 2 - 6.0, p["pod_width"] / 2 + p["panel_length"]
        wp = cq.Workplane("XZ").workplane(offset=-r0).polyline(self._section_pts(p, self._chord_at(p, r0), self._x_le_at(p, r0))).close()
        wp = wp.workplane(offset=-(r1 - r0)).polyline(self._section_pts(p, self._chord_at(p, r1), self._x_le_at(p, r1))).close()
        pan = wp.loft(ruled=True, combine=True)
        cf = p["elevon_chord"]
        c_in, c_tip = self._chord_at(p, p["elevon_inner_r"]), self._chord_at(p, r1)
        x_in, x_tip = self._x_le_at(p, p["elevon_inner_r"]) + (1 - cf) * c_in, self._x_le_at(p, r1) + (1 - cf) * c_tip
        bay = (cq.Workplane("XY").polyline([(x_in - 1.0, p["elevon_inner_r"]), (x_tip - 1.0, r1 + 2.0), (x_tip + c_tip + 5.0, r1 + 2.0), (x_in + c_in + 5.0, p["elevon_inner_r"])])
               .close().extrude(c_in, both=True))
        pan = pan.cut(bay)
        return pan.rotate((0, 0, 0), (1, 0, 0), angle_deg)

    def _elevon(self, p, angle_deg):
        import cadquery as cq
        cf = p["elevon_chord"]
        r1 = p["pod_width"] / 2 + p["panel_length"]
        up, lo = kestrel.naca4(p["camber"], p["camber_pos"], p["thickness"])
        xh = 1 - cf

        def pts(r):
            c, x_le = self._chord_at(p, r) - 1.5, self._x_le_at(p, r) + 1.0
            return [(x_le + c * u, c * v) for u, v in up[::-1] if u >= xh] + [(x_le + c * u, c * v) for u, v in lo[1:] if u >= xh]
        ri = p["elevon_inner_r"] + 1.0
        wp = cq.Workplane("XZ").workplane(offset=-ri).polyline(pts(ri)).close().workplane(offset=-(r1 - 1.0 - ri)).polyline(pts(r1 - 1.0)).close()
        return wp.loft(ruled=True, combine=True).rotate((0, 0, 0), (1, 0, 0), angle_deg)

    def _pod(self, p):
        """The blended fuselage: an elliptical loft through ``FUSELAGE_STATIONS`` (width × depth), its axis the panels'
        axis, the module inside the parallel part; the camera behind a flush window in the nose (no dome)."""
        import cadquery as cq
        Lp, w, h = p["pod_length"], p["pod_width"], p["pod_height"]
        wp = cq.Workplane("YZ")
        x_prev = 0.0
        for fx, fw, fh in FUSELAGE_STATIONS:
            wp = wp.workplane(offset=fx * Lp - x_prev).ellipse(max(fw * w / 2, 1.0), max(fh * h / 2, 1.0))
            x_prev = fx * Lp
        return wp.loft(ruled=False, combine=True)

    def _module(self, p):
        import cadquery as cq
        lm, wm, hm = p["module_length"], p["module_width"], p["module_height"]
        if lm <= 0:
            return None
        z = -(p["pod_height"] / 2 - hm / 2 - 8.0)                     # on the floor of the parallel part, the battery above it
        return cq.Workplane("XY").box(lm, wm - 12.0, hm).edges("|X").fillet(10.0).translate((p["module_x"] + lm / 2, 0, z))

    def _nacelle(self, p, angle_deg, prop):
        import cadquery as cq
        r = p["nacelle_diameter"] / 2
        y = p["motor_r"]
        x0, x1 = p["prop_x"] + 12.0, self._x_le_at(p, y) + 0.95 * self._chord_at(p, y) + 60.0       # the pod runs past the trailing edge: the foot
        can = cq.Workplane("YZ").workplane(offset=x0).center(y, 0).circle(r).extrude(x1 - x0 - 24.0)
        tail = cq.Workplane("XY").add(cq.Solid.makeCone(r, r * 0.45, 24.0, cq.Vector(x1 - 24.0, y, 0), cq.Vector(1, 0, 0)))
        spinner = cq.Workplane("XY").add(cq.Solid.makeCone(0.0, r * 0.8, 30.0, cq.Vector(p["prop_x"] - 18.0, y, 0), cq.Vector(1, 0, 0)))
        out = can.union(spinner).union(tail)
        if prop:
            out = out.union(cq.Workplane("YZ").workplane(offset=p["prop_x"] - 1.0).center(y, 0).circle(p["prop_diameter"] / 2).extrude(2.0))
        return out.rotate((0, 0, 0), (1, 0, 0), angle_deg)

    def build(self, p):
        part = p["part"]
        phi = p["phi_deg"]
        angles = (phi, 180 - phi, 180 + phi, -phi)             # right-up, left-up, left-down, right-down
        if part == "panel":
            return self._panel(p, phi)
        if part == "pod":
            return self._pod(p)
        if part == "module":
            return self._module(p)
        shape = self._pod(p)
        for a in angles:
            shape = shape.union(self._panel(p, a)).union(self._elevon(p, a)).union(self._nacelle(p, a, prop=(part == "aircraft" and p["show_props"])))
        a0 = p["angle_of_attack_deg"]
        if a0:
            x0 = p["rotation_x"]
            shape = shape.rotate((x0, 0, 0), (x0, 1, 0), a0)
        return shape


def export_stl(workdir, variant: str, alpha_deg: float, x_rot_m: float) -> Path:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    path = workdir / f"kestrel_{variant}_a{alpha_deg:+05.1f}.stl"
    if not path.exists():
        KestrelXC().generate(**cad_params(variant), part="airframe", angle_of_attack_deg=float(alpha_deg), rotation_x=x_rot_m * 1000).export_stl(str(path), tolerance=0.3)
    return path


# ================================================================================================= CFD (the book's cfd chapters)
SWEEP_V = 10.0
SWEEP_ALPHAS = (0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 75.0, 90.0)
STALL_ALPHAS = (8.0, 10.0, 12.0, 14.0, 16.0)
CFD_CAMPAIGNS = {"sweep": f"the bare X-C at {len(SWEEP_ALPHAS)} angles 0–90° at {SWEEP_V:g} m/s: the vertical lift slope against the lattice's cos² φ rule, Cd0, the deep stall",
                 "stall": f"the bare X-C at {', '.join(f'{a:g}' for a in STALL_ALPHAS)}°: the panels' CL max (the lattice cannot give it)"}


def cfd_specs(variant: str, campaign: str) -> dict:
    alphas = {"sweep": SWEEP_ALPHAS, "stall": STALL_ALPHAS}[campaign]
    return {f"bare X-C, α {a:.0f}°, {SWEEP_V:g} m/s": {"source": "sweep", "kind": campaign, "alpha_deg": a, "V": SWEEP_V, "T_up": 0.0, "T_lo": 0.0, "t": 0.0, "target_deg": a, "props": False}
            for a in alphas}


def l_ref(L: dict) -> float:
    return max(L["span_projected"], L["x_feet"])


def cfd_case(workdir, spec: dict, *, variant: str, x_cg_m: float, quality: str = "screening", env=None):
    """One bare ``rans_ksst_external`` case of the X-C: the reference area the four panels', the moment about the CG."""
    import kestrel_cfd as kcfd
    from vegeta import aeromant
    L = layout(variant)
    a = spec["alpha_deg"]
    stl = export_stl(Path(workdir) / "stl", variant, a, x_cg_m)
    cor = kestrel.rotate((x_cg_m * 1000, 0.0, 0.0), a, x_cg_m * 1000)
    params = dict(velocity=max(spec["V"], 1.0), kinematic_viscosity=NU, density=RHO, reference_area=L["S_ref"], reference_length=l_ref(L),
                  center_of_rotation=(cor[0] / 1000, 0.0, cor[2] / 1000), **kcfd.QUALITY[quality])
    env = env or aeromant.OpenFOAMEnvironment.detect()
    return aeromant.CFDCase("rans_ksst_external", str(stl), params, workdir=str(Path(workdir) / f"xc_a{a:+05.1f}_V{spec['V']:.1f}_bare_{quality}"), geometry_units="mm", environment=env)


# ================================================================================================= FEA (the book's fea chapter): one panel's spar
N_SEG = 6
FEA_CAMPAIGNS = {"panel": "one panel's carbon spar from its pod socket at a 4 g pull-up (A), with the tip motor's inertia, and its first modes with the motor as a point mass"}


def panel_breaks(L: dict, od_mm: float) -> list:
    """Where the panel's spar tube is split into separate faces [mm along the panel]: the root, the lift stations, the
    motor band, the tip (whole faces for the region boxes, as ``kestrel_fea.spar_breaks``)."""
    import kestrel_fea as kfe
    r0, r1, rm = L["r0"] * 1e3, L["r1"] * 1e3, L["motor_r"] * 1e3
    st = set(np.round(np.linspace(r0, r1, N_SEG + 1), 6)) | {rm - kfe.MOTOR_BAND, rm + kfe.MOTOR_BAND}
    return sorted(float(v) for v in st)


def panel_cad(L: dict, od_mm: float, wall_mm: float = 1.0, socket_mm: float = 40.0):
    """The spar in the panel's own frame (y along the panel from the pod's axis, z normal to the panel): cylinder
    segments between ``panel_breaks`` (``clean=False`` keeps the seams), the bore cut last, a ``socket_mm`` length
    inside the pod as the clamped root."""
    import cadquery as cq
    import kestrel_fea as kfe
    br = panel_breaks(L, od_mm)
    x = (X_LE + 0.30 * L["root_chord"]) * 1e3
    root = L["r0"] * 1e3 - socket_mm
    outers = [kfe._cyl((x, root, 0), (x, br[0], 0), od_mm)] + [kfe._cyl((x, a, 0), (x, b, 0), od_mm) for a, b in zip(br[:-1], br[1:])]
    solid = outers[0]
    for o in outers[1:]:
        solid = solid.union(o, clean=False)
    bore = kfe._cyl((x, root - 1, 0), (x, br[-1] + 1, 0), od_mm - 2 * wall_mm)
    return solid.cut(bore)


def fea_cases(variant: str, campaign: str, workdir, *, mass_kg: float, element_size: float = 2.0, n: float = 4.0) -> list:
    """The panel chapter: case A (the 4 g pull-up, the panel's quarter of the lift elliptic along it, the tip motor's
    inertia, ultimate × 1.5) and the modes, on the spar ``cantilever`` sizes (OD rounded up to a stock tube)."""
    import kestrel_fea as kfe
    from vegeta import talos
    L = layout(variant)
    cl = cantilever(L, mass_kg=mass_kg, n=n)
    od = float(np.ceil(cl["spar OD [mm]"]))
    od = {7.0: 8.0, 9.0: 10.0, 11.0: 12.0}.get(od, od)
    wd = Path(workdir) / "cad"
    wd.mkdir(parents=True, exist_ok=True)
    step = wd / f"xc_panel_spar_{od:g}.step"
    if not step.exists():
        panel_cad(L, od).val().exportStep(str(step))
    x = (X_LE + 0.30 * L["root_chord"]) * 1e3
    r, m = od / 2 + kfe.REGION_MARGIN, kfe.REGION_MARGIN
    br = panel_breaks(L, od)
    root = L["r0"] * 1e3 - 40.0
    regions = [talos.SurfacesInBox("socket", (x - r, root - m, -r, x + r, br[0] + m, r))]
    rm = L["motor_r"] * 1e3
    regions.append(talos.SurfacesInBox("motor", (x - r, rm - kfe.MOTOR_BAND - m, -r, x + r, rm + kfe.MOTOR_BAND + m, r)))
    Lp = L["r1"] - L["r0"]
    lift_panel = n * mass_kg * G / 4 / math.cos(L["phi"])
    wing_m = ks.WING_AREAL_KG_M2 * L["S_panel"]
    loads, i = [], 0
    for a, b in zip(br[:-1], br[1:]):
        if abs(a - (rm - kfe.MOTOR_BAND)) < 1e-6 and abs(b - (rm + kfe.MOTOR_BAND)) < 1e-6:
            continue
        ya, yb = (a / 1e3 - L["r0"]) / Lp, (b / 1e3 - L["r0"]) / Lp
        ya, yb = min(max(ya, 0), 1), min(max(yb, 0), 1)
        frac = (2 / math.pi) * ((yb * math.sqrt(1 - yb ** 2) + math.asin(yb)) - (ya * math.sqrt(1 - ya ** 2) + math.asin(ya)))
        f = lift_panel * frac - n * G * wing_m * (yb - ya)
        name = f"lift_{i:02d}"
        regions.append(talos.SurfacesInBox(name, (x - r, a - m, -r, x + r, b + m, r)))
        loads.append(talos.Force(name, fz=kfe.ULTIMATE * f))
        i += 1
    motor_m = ks.COMPONENTS["motor"]["mass_kg"] + ks.COMPONENTS["propeller"]["mass_kg"] + ks.COMPONENTS["esc"]["mass_kg"] + ks.NACELLE_KG
    loads.append(talos.Force("motor", fz=-kfe.ULTIMATE * n * G * motor_m))
    mesh = talos.MeshSettings(element_size=element_size, order=2)
    mat = kfe.talos_material(kfe.CARBON)
    sup = [talos.FixedSupport("socket")]
    out = [kfe.FEACase("panel_A", "panel", f"{n:g} g pull-up: the panel's quarter of the lift elliptic along it, the tip motor's inertia (ultimate ×{kfe.ULTIMATE}); Ø{od:g} tube",
                       talos.StructuralModel(step, "mm-N-MPa", mat, regions, sup, loads, mesh, name="panel_A"), strength=kfe.CARBON["strength"], bracing="none"),
           kfe.FEACase("panel_modes", "panel", "the first modes with the motor unit as a point mass at the tip", talos.StructuralModel(step, "mm-N-MPa", mat, regions, sup, [], mesh,
                       masses=[talos.PointMass("motor", motor_m * 1e-3)], name="panel_modes"), kind="modes", bracing="none")]
    for c in out:
        c.exclude = lambda xyz, root=root, rm=rm: (xyz[:, 1] < root + 52.0) | (np.abs(xyz[:, 1] - rm) < 12.0)
    return out


# ================================================================================================= against the others
def biplane_reference(path=None):
    return kx.biplane_reference(path)


def xwing_rows() -> dict:
    """X-A and X-B recomputed from ``kestrel_xwing`` (the same modules, so the comparison is like for like)."""
    out = {}
    for v in kx.VARIANTS:
        if v.startswith("xc:"):
            continue
        L = kx.layout(v)
        ci = ks.cg_inertia(kx.mass_table(L))
        a, d = kx.aero(L), kx.drag_buildup(L)
        out[v.split(" ")[0]] = {"L": L, "ci": ci, "aero": a, "drag": d}
    return out
