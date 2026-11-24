"""ASTOR-X — ASTOR-TWIN's body, tail and launcher with four rotors on a short X and a swept wing of 1.2 m span
(2026-11-24). A DRAWING, not yet a design.

The first drawing (commit 4acb977) put four long lifting panels (450 mm) on ASTOR-TWIN's 1.95 m wing. Asked for next:
the propellers close to the fuselage — the X only holds them — and the wing swept back to a span of 1.2 m.

- **Wing**: one piece, no pivots. Span 1.20 m, the AREA KEPT at ASTOR-TWIN's 0.513 m2 (so the wing loading and the
  stall are not given away: a HAND choice; a smaller wing is the alternative), leading edge swept 35 degrees, the
  same cambered section. Its mean quarter chord is where ASTOR-TWIN's was (373 mm from the nose), so the root's
  leading edge is at 74 mm. Aspect ratio 2.8 against 7.4. Flaps inboard, ailerons outboard.
- **X**: four streamlined struts at 45 degrees, 190 mm from the axis to each motor: 15 mm between a disc and the
  body's side. PUSHERS behind the wing's trailing edge and ahead of the tail: so close to the body the discs cross
  the wing's plane, so they cannot sit beside the wing; ahead of it they would stand in front of the net's muzzle
  and the camera. Behind, their wash goes over the tail's roots.
- **Propellers**: four of the two 13-inch discs' total area (9.2 inches).
- Body, tail, fin, launcher and vessel: ASTOR-TWIN's (PROTOHAWK), unchanged.

Every size is a HAND choice. Nothing here has been balanced, analysed for stability, sized structurally or solved in
CFD. ``checks()`` measures the clearances from the solids; ``astor_big`` scores it.
"""
from __future__ import annotations

import math

import cadquery as cq
from vegeta.dedalus import Parameter

import astor_twin as W
import astor_v as V

SPAN_M, AREA_M2, LE_SWEEP_DEG, TAPER = 1.20, 0.513, 35.0, 0.527
_mean = AREA_M2 / SPAN_M * 1e3
WING = dict(half_span=SPAN_M * 500.0, root_chord=round(2 * _mean / (1 + TAPER)), tip_chord=round(2 * _mean * TAPER / (1 + TAPER)), le_x=74.0, sweep_deg=LE_SWEEP_DEG,
            flap=(0.0, 0.50, 0.30), aileron=(0.52, 0.96, 0.25))            # (from, to: shares of the half span; chord share)
X = dict(radius=190.0, strut_chord=60.0, strut_thickness=0.20, disc_x=745.0, pod_d=46.0, pod_len=110.0, prop_diameter=round(330.2 / math.sqrt(2.0), 1))
_NEW = [Parameter("x_radius", X["radius"], "mm", min=100.0, description="from the axis to each motor"),
        Parameter("x_strut_chord", X["strut_chord"], "mm", min=20.0), Parameter("x_strut_thickness", X["strut_thickness"], "", min=0.06, max=0.4),
        Parameter("x_disc_x", X["disc_x"], "mm", description="the propellers' plane from the nose (pushers: behind their motors)"),
        Parameter("x_pod_d", X["pod_d"], "mm", min=20.0), Parameter("x_pod_len", X["pod_len"], "mm", min=40.0),
        Parameter("x_prop_diameter", X["prop_diameter"], "mm", min=50.0),
        Parameter("sw_half_span", WING["half_span"], "mm", min=200.0), Parameter("sw_root_chord", WING["root_chord"], "mm", min=100.0),
        Parameter("sw_tip_chord", WING["tip_chord"], "mm", min=50.0), Parameter("sw_le_x", WING["le_x"], "mm"),
        Parameter("sw_sweep_deg", WING["sweep_deg"], "deg", min=0.0, max=65.0, description="the leading edge's sweep")]
ANGLES = (45.0, 135.0, 225.0, 315.0)


class AstorX(W.AstorTwin):
    """ASTOR-X: ASTOR-TWIN's body and tail; a one-piece swept wing; four pusher motors on a short X behind it."""

    parameters = list(W.AstorTwin.parameters) + _NEW

    # ---- the wing
    def _sw_block(self, p, y_from, y_to, share):
        cr, ct, b, tan = p["sw_root_chord"], p["sw_tip_chord"], p["sw_half_span"], math.tan(math.radians(p["sw_sweep_deg"]))
        hinge = lambda y: y * tan + (1.0 - share) * (cr + (ct - cr) * y / b)
        y0, y1 = max(y_from * b, p["fuselage_diameter"] / 2 + 4.0), y_to * b
        pts = [(hinge(y0), y0), (hinge(y1), y1), (hinge(y1) + cr, y1), (hinge(y0) + cr, y0)]
        return cq.Workplane("XY").polyline(pts).close().extrude(160.0).translate((0, 0, -80.0)), (hinge(y0), y0, 0.0), (hinge(y1), y1, 0.0)

    def _wing_part(self, p, part):
        half = V._panel(p["sw_root_chord"], p["sw_tip_chord"], p["sw_half_span"], p["sw_sweep_deg"], p["panel_thickness"], p["panel_camber"])
        fb, f0, f1 = self._sw_block(p, *WING["flap"])
        ab, a0, a1 = self._sw_block(p, *WING["aileron"])
        if part == "flap":
            half = self._hinged(half, fb, f0, f1, p["inboard_flap_deg"])
        elif part == "aileron":
            half = self._hinged(half, ab, a0, a1, p["flaperon_deg"])
        else:
            half = half.cut(fb).cut(ab)
        return half.union(half.mirror("XZ")).translate((p["sw_le_x"], 0, p["wing_plane_z"]))

    def _inner(self, p, part):
        return self._wing_part(p, part)

    # ---- the X
    def _each(self, shape):
        out = None
        for a in ANGLES:
            s = shape.rotate((0, 0, 0), (1, 0, 0), a)
            out = s if out is None else out.union(s)
        return out

    def _pod_x(self, p):
        return p["x_disc_x"] - 12.0 - p["x_pod_len"]

    def _strut_x(self, p):
        return self._pod_x(p) + 0.45 * p["x_pod_len"] - 0.5 * p["x_strut_chord"]

    def _x_struts(self, p):
        c = p["x_strut_chord"]
        return self._each(V._panel(c, c, p["x_radius"], 0.0, p["x_strut_thickness"]).translate((self._strut_x(p), 0, 0)))

    def _motors(self, p):
        pod = cq.Workplane("YZ").circle(p["x_pod_d"] / 2).extrude(p["x_pod_len"]).faces("<X").edges().fillet(p["x_pod_d"] * 0.35)
        hub = cq.Workplane("YZ").circle(10.0).extrude(12.0).translate((p["x_pod_len"], 0, 0))
        return self._each(pod.union(hub).translate((self._pod_x(p), p["x_radius"], 0)))

    def _props(self, p):
        return self._each(cq.Workplane("YZ").circle(p["x_prop_diameter"] / 2).extrude(1.5).translate((p["x_disc_x"], p["x_radius"], 0)))

    def parts(self, p) -> dict:
        p = {**p, "prop_fold_deg": -1.0}                                       # no folding propellers on the wing
        out = super().parts(p)
        out.pop("shoulder", None)                                              # no pivots: the wing is one piece
        out["wing"], out["ailerons"], out["flaps"] = (self._wing_part(p, k) for k in ("fixed", "aileron", "flap"))
        out["x_struts"] = self._x_struts(p)
        return out


def variant(**over) -> dict:
    return W.variant(**over)


def resolve(q=None) -> dict:
    return AstorX().resolve(**dict(q or variant(net_launchers=1, talon_deploy=-1.0, net_count=0, launcher_depression_deg=0.0)))


def planform(p=None) -> dict:
    p = p or resolve()
    cr, ct, b = p["sw_root_chord"], p["sw_tip_chord"], p["sw_half_span"]
    lam, tan = ct / cr, math.tan(math.radians(p["sw_sweep_deg"]))
    S = (cr + ct) * b / 1e6
    mac = 2 / 3 * cr * (1 + lam + lam * lam) / (1 + lam)
    y_mac = b * (1 + 2 * lam) / (3 * (1 + lam))
    ar = (2 * b / 1e3) ** 2 / S
    tan_half = tan - (cr - ct) / (2 * b)
    return {"span_m": 2 * b / 1e3, "area_m2": round(S, 3), "aspect_ratio": round(ar, 2), "root_chord_mm": cr, "tip_chord_mm": ct, "mean_chord_mm": round(mac),
            "leading_edge_sweep_deg": p["sw_sweep_deg"], "half_chord_sweep_deg": round(math.degrees(math.atan(tan_half)), 1),
            "mean_quarter_chord_from_the_nose_mm": round(p["sw_le_x"] + y_mac * tan + 0.25 * mac),
            "lift_slope_per_rad_helmbold": round(2 * math.pi * ar / (2 + math.sqrt(4 + ar * ar * (1 + tan_half ** 2))), 2),
            "lift_slope_source": "Helmbold's formula with the half-chord sweep (HAND: not the lattice, not CFD)"}


def checks(p=None) -> dict:
    p = p or resolve()
    A = AstorX()
    parts = A.parts(p)
    c45 = math.cos(math.radians(45.0))
    r, d, xd = p["x_radius"], p["x_prop_diameter"], p["x_disc_x"]
    wing = parts["wing"].union(parts["ailerons"]).union(parts["flaps"])
    tail = parts["tail"].union(parts["tail_flaps"]) if "tail_flaps" in parts else parts["tail"]
    tubes = A._each(cq.Workplane("YZ").circle(d / 2 + 0.5).extrude(600.0).translate((xd, r, 0)))                 # the wash, straight back
    sweep = A._each(cq.Workplane("YZ").circle(d / 2 + 3.0).extrude(16.0).translate((xd - 8.0, r, 0)))            # the discs, 3 mm and 8 mm to spare

    def cm3(a, b):
        try:
            return round(a.intersect(b).val().Volume() / 1e3, 1)
        except Exception:
            return 0.0

    def near(shape, lo, hi):                                                    # the solid's extent in x inside the discs' band of y
        box = cq.Workplane("XY").box(4000.0, 2 * (r * c45 + d / 2), 2 * (r * c45 + d / 2)).translate((0, 0, 0))
        bb = shape.intersect(box).val().BoundingBox()
        return bb.xmin, bb.xmax
    w0, w1 = near(wing, 0, 0)
    t0, t1 = near(tail, 0, 0)
    bb = parts["x_struts"].union(parts["props"]).val().BoundingBox()
    return {"what": "ASTOR-X with a short X and a swept wing: a drawing; clearances measured from the solids; every size is a HAND choice",
            "wing": planform(p),
            "x": {"each_motor_from_the_axis_mm": r, "motor_y_and_z_mm": round(r * c45, 1), "strut_chord_mm": p["x_strut_chord"],
                  "box_with_its_propellers_mm": {"y": round(bb.ylen), "z": round(bb.zlen)}},
            "propellers": {"count": 4, "kind": "pushers", "diameter_mm": d, "diameter_in": round(d / 25.4, 1), "disc_plane_from_the_nose_mm": xd,
                           "disc_edge_from_the_fuselage_side_mm": round(r - d / 2 - A.fuselage_radius(p, xd)),
                           "between_neighbouring_discs_mm": round(2 * r * c45 - d),
                           "disc_plane_behind_the_wing_trailing_edge_mm": round(xd - w1), "disc_plane_ahead_of_the_tail_mm": round(t0 - xd),
                           "discs_cross_the_wing_plane": bool(r * c45 - d / 2 < abs(p["wing_plane_z"]) + 10.0)},
            "struck_by_the_discs_cm3": {"wing": cm3(sweep, wing), "tail": cm3(sweep, tail), "fuselage": cm3(sweep, parts["fuselage"]),
                                        "launcher": cm3(sweep, parts["launcher"]) if "launcher" in parts else None},
            "in_the_wash_straight_back_cm3": {"tail": cm3(tubes, tail), "fin": cm3(tubes, parts["fin"]) if "fin" in parts else None,
                                              "note": "the tail's roots are blown: control at low speed; the wing's flaps are not"},
            "overlaps_cm3": {"struts_with_the_wing": cm3(parts["x_struts"], wing), "struts_with_the_tail": cm3(parts["x_struts"], tail)},
            "span_mm": round(2 * p["sw_half_span"]), "length_mm": p["fuselage_length"],
            "not_done": ["balance and mass ledger", "stability and control with four rotors (the motors' arm about the axis is only %d mm)" % round(r * c45),
                         "structure of the struts and of the swept wing", "CFD: the wing is new (aspect ratio 2.8, swept); ASTOR-TWIN's tables do not carry over",
                         "how it stands, launches and lands", "whether the body's wake spoils the pushers"]}


if __name__ == "__main__":
    import pathlib
    import sys
    import yaml
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    here = pathlib.Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent / "reports" / "arigo_catalog"))
    import figures as F
    D = here / "data" / "astor_x"
    D.mkdir(parents=True, exist_ok=True)
    tmp = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else D
    p = resolve()
    parts = AstorX().parts(p)
    asm = cq.Assembly()
    for n, sh in parts.items():
        asm.add(sh, name=n)
    asm.save(str(D / "astor_x.step"))
    col = {**F.PART_COL, "launcher": "#c0392b", "ailerons": "#8a5a1a", "flaps": "#2e7d32", "parachute": "#e67e22", "tail_flaps": "#2e7d32",
           "x_struts": "#1f5fa8", "motors": "#333333"}
    items = [(s, col.get(n, "#888888"), 0.3 if n == "props" else 1.0) for n, s in parts.items() if n != "vessel"]
    views = [("front quarter", (22, -130)), ("from above", (90, -90)), ("from the side", (0, -90)), ("from the front", (0, 180)), ("from below", (-30, -125)), ("rear quarter", (20, -50))]
    f, axs = plt.subplots(3, 2, figsize=(12, 13.5), facecolor="white")
    for ax, (t, (e, a)) in zip(axs.flat, views):
        png = tmp / f"ax_{e}_{a}.png"
        ax.imshow(plt.imread(F.render(str(png), items, e, a, size=(1100, 800))))
        ax.set_axis_off()
        ax.set_title(t, fontsize=11)
        png.unlink()
    f.suptitle("ASTOR-X: span 1.20 m, a swept wing of ASTOR-TWIN's area; four 9.2-inch pushers on a short X (blue) between the wing and the tail", fontsize=11)
    f.tight_layout()
    f.savefig(D / "astor_x_views.png", dpi=115)
    c = checks(p)
    (D / "astor_x_checks.yaml").write_text(yaml.safe_dump(c, sort_keys=False, allow_unicode=True, width=150))
    print(yaml.safe_dump(c, sort_keys=False, allow_unicode=True, width=150))
