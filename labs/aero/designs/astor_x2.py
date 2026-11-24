"""ASTOR-X2 — ASTOR-X as a TAIL-SITTER (2026-11-24). A DRAWING, not yet a design.

ASTOR-X (``designs/astor_x``) hovers on four rotors but had nothing to stand on, and its rotors' arm about the axis
(134 mm) gave little pitch and yaw torque in the hover. A tail-sitter answers both with one part: an X of four FINS at
the tail.

- **X tail**: four swept fins at 45 degrees, 380 mm from the axis to their tips, in place of the two tail halves and
  the upright fin. Each carries an ELEVON (35 % chord): pitch, yaw and roll in flight as an X tail, and the same in
  the hover, where they stand in the propellers' wash.
- **Feet**: a bumper on each fin's tip, behind everything else: the aircraft stands on the four, nose up.
- **Motors**: four TRACTORS on the fins' leading edges, 190 mm from the axis (as ASTOR-X: close to the body), behind
  the wing's trailing edge. Nothing turns near the ground.
- **Wing, body, launcher, vessel**: ASTOR-X's and ASTOR-TWIN's, unchanged. The launch tube points up when it stands.

Every size is a HAND choice. Nothing here has been balanced, analysed for stability, sized structurally or solved in
CFD; the centre of gravity used for the tip-over angle is ASTOR-X's (``astor_x_flight``), which this tail will move.
``checks()`` measures the clearances and the stance from the solids.
"""
from __future__ import annotations

import math

import cadquery as cq
from vegeta.dedalus import Parameter

import astor_v as V
import astor_x as AX

FIN = dict(radius=380.0, root_chord=260.0, tip_chord=160.0, root_le_x=800.0, tip_le_x=950.0, thickness=0.08, elevon_chord=0.35, elevon_from=85.0, elevon_to=355.0,
           foot_len=50.0, foot_d=22.0)
_NEW = [Parameter("fin_radius", FIN["radius"], "mm", min=150.0, description="each X fin, from the axis to its tip (the foot)"),
        Parameter("xfin_root_chord", FIN["root_chord"], "mm", min=60.0), Parameter("xfin_tip_chord", FIN["tip_chord"], "mm", min=40.0),
        Parameter("xfin_root_le_x", FIN["root_le_x"], "mm"), Parameter("xfin_tip_le_x", FIN["tip_le_x"], "mm"),
        Parameter("xfin_thickness", FIN["thickness"], "", min=0.04, max=0.2), Parameter("elevon_chord", FIN["elevon_chord"], "", min=0.0, max=0.5),
        Parameter("elevon_deg", 0.0, "deg", min=-35.0, max=35.0, description="the four elevons, trailing edge turned the same way round (a roll command), for the drawing"),
        Parameter("foot_len", FIN["foot_len"], "mm", min=10.0), Parameter("foot_d", FIN["foot_d"], "mm", min=8.0)]


class AstorX2(AX.AstorX):
    """ASTOR-X2: ASTOR-X's wing and body; an X of four fins with elevons and feet at the tail; four tractor motors on the fins."""

    parameters = list(AX.AstorX.parameters) + _NEW

    def _fin_sweep(self, p):
        return math.degrees(math.atan((p["xfin_tip_le_x"] - p["xfin_root_le_x"]) / p["fin_radius"]))

    def _fin_le(self, p, r):
        return p["xfin_root_le_x"] + (p["xfin_tip_le_x"] - p["xfin_root_le_x"]) * r / p["fin_radius"]

    def _fin_chord(self, p, r):
        return p["xfin_root_chord"] + (p["xfin_tip_chord"] - p["xfin_root_chord"]) * r / p["fin_radius"]

    def _fin_one(self, p, part):
        """One fin along +y in its own plane (fixed part, or its elevon)."""
        cr, ct, b = p["xfin_root_chord"], p["xfin_tip_chord"], p["fin_radius"]
        pan = V._panel(cr, ct, b, self._fin_sweep(p), p["xfin_thickness"])
        f, tan = 1.0 - p["elevon_chord"], math.tan(math.radians(self._fin_sweep(p)))
        hinge = lambda y: y * tan + f * (cr + (ct - cr) * y / b)
        y0, y1 = FIN["elevon_from"], FIN["elevon_to"]
        blk = cq.Workplane("XY").polyline([(hinge(y0), y0), (hinge(y1), y1), (hinge(y1) + cr, y1), (hinge(y0) + cr, y0)]).close().extrude(80.0).translate((0, 0, -40.0))
        out = self._hinged(pan, blk, (hinge(y0), y0, 0.0), (hinge(y1), y1, 0.0), p["elevon_deg"]) if part == "elevon" else pan.cut(blk)
        return out.translate((p["xfin_root_le_x"], 0, 0))

    def _tail(self, p):
        return self._each(self._fin_one(p, "fixed"))

    def _elevons(self, p):
        return self._each(self._fin_one(p, "elevon"))

    def _foot_x(self, p):
        return p["xfin_tip_le_x"] + p["xfin_tip_chord"] + p["foot_len"]

    def _feet(self, p):
        x0 = p["xfin_tip_le_x"] + 0.35 * p["xfin_tip_chord"]
        foot = (cq.Workplane("YZ").circle(p["foot_d"] / 2).extrude(self._foot_x(p) - x0).faces(">X").edges().fillet(p["foot_d"] * 0.4)
                .translate((x0, p["fin_radius"] - p["foot_d"] / 2, 0)))
        return self._each(foot)

    def _pod_x(self, p):                                                      # tractors: the pod's nose, ahead of the fin's leading edge
        return self._fin_le(p, p["x_radius"]) + 25.0 - p["x_pod_len"]

    def _motors(self, p):
        pod = cq.Workplane("YZ").circle(p["x_pod_d"] / 2).extrude(p["x_pod_len"]).faces(">X").edges().fillet(p["x_pod_d"] * 0.35)
        hub = cq.Workplane("YZ").circle(10.0).extrude(-12.0)
        return self._each(pod.union(hub).translate((self._pod_x(p), p["x_radius"], 0)))

    def _disc_x(self, p):
        return self._pod_x(p) - 12.0

    def _props(self, p):
        return self._each(cq.Workplane("YZ").circle(p["x_prop_diameter"] / 2).extrude(1.5).translate((self._disc_x(p), p["x_radius"], 0)))

    def parts(self, p) -> dict:
        out = super().parts(p)
        out.pop("x_struts", None)                                             # the fins hold the motors
        out.pop("tail_flaps", None)
        out["elevons"] = self._elevons(p)
        out["feet"] = self._feet(p)
        return out


def variant(**over) -> dict:
    return AX.variant(**over)


def resolve(q=None) -> dict:
    return AstorX2().resolve(**dict(q or variant(net_launchers=1, talon_deploy=-1.0, net_count=0, launcher_depression_deg=0.0)))


def checks(p=None) -> dict:
    p = p or resolve()
    A = AstorX2()
    parts = A.parts(p)
    c45 = math.cos(math.radians(45.0))
    r, d, xd, R = p["x_radius"], p["x_prop_diameter"], A._disc_x(p), p["fin_radius"]
    wing = parts["wing"].union(parts["ailerons"]).union(parts["flaps"])
    sweep = A._each(cq.Workplane("YZ").circle(d / 2 + 3.0).extrude(16.0).translate((xd - 8.0, r, 0)))
    tubes = A._each(cq.Workplane("YZ").circle(d / 2).extrude(500.0).translate((xd, r, 0)))

    def cm3(a, b):
        try:
            return round(a.intersect(b).val().Volume() / 1e3, 1)
        except Exception:
            return 0.0
    band = cq.Workplane("XY").box(4000.0, 2 * (r * c45 + d / 2), 2 * (r * c45 + d / 2))
    w_te = wing.intersect(band).val().BoundingBox().xmax
    foot_x = A._foot_x(p)
    aft = max(parts[k].val().BoundingBox().xmax for k in parts if k not in ("feet",))
    try:
        import astor_x_flight as XF
        cg, cg_src = XF.balance()["cg_mm"], "ASTOR-X's (astor_x_flight: lattice); this tail will move it"
    except Exception:                                                         # noqa: BLE001
        cg, cg_src = 400.0, "HAND"
    half_width = (R - p["foot_d"] / 2) * c45                                  # the feet make a square: its half side
    h_cg = foot_x - cg
    fin_area = 4 * 0.5 * (A._fin_chord(p, R) + A._fin_chord(p, A.fuselage_radius(p, 900.0))) * (R - A.fuselage_radius(p, 900.0)) / 1e6
    elevon_area = 4 * p["elevon_chord"] * 0.5 * (A._fin_chord(p, FIN["elevon_from"]) + A._fin_chord(p, FIN["elevon_to"])) * (FIN["elevon_to"] - FIN["elevon_from"]) / 1e6
    return {"what": "ASTOR-X2, a tail-sitter: a drawing; measured from the solids; every size is a HAND choice",
            "wing": AX.planform(p),
            "x_tail": {"fins": 4, "each_from_the_axis_mm": R, "root_chord_mm": p["xfin_root_chord"], "tip_chord_mm": p["xfin_tip_chord"],
                       "leading_edge_sweep_deg": round(A._fin_sweep(p), 1), "exposed_area_m2_all_four": round(fin_area, 3),
                       "worth_as_a_flat_tail_m2": round(fin_area / 2, 3), "worth_as_an_upright_fin_m2": round(fin_area / 2, 3),
                       "note": "an X at 45 deg gives half its area in pitch and half in yaw (cos^2 45); ASTOR-X's flat tail was about 0.19 m2",
                       "elevons": {"chord_share": p["elevon_chord"], "area_m2_all_four": round(elevon_area, 3), "inside_the_propellers_wash_cm3": cm3(tubes, parts["elevons"])}},
            "standing": {"height_mm": round(foot_x), "feet_square_side_mm": round(2 * half_width), "cg_from_the_nose_mm": cg, "cg_source": cg_src, "cg_above_the_ground_mm": round(h_cg),
                         "tips_over_beyond_deg": round(math.degrees(math.atan(half_width / h_cg)), 1),
                         "tips_over_beyond_deg_across_a_diagonal": round(math.degrees(math.atan(half_width * math.sqrt(2) / h_cg)), 1),
                         "everything_else_above_the_ground_by_mm": round(foot_x - aft), "note": "the first angle is over a side of the feet's square: the one that matters"},
            "propellers": {"count": 4, "kind": "tractors on the fins", "diameter_mm": d, "motor_from_the_axis_mm": r, "disc_plane_from_the_nose_mm": round(xd),
                           "disc_edge_from_the_fuselage_side_mm": round(r - d / 2 - A.fuselage_radius(p, xd)), "between_neighbouring_discs_mm": round(2 * r * c45 - d),
                           "disc_plane_behind_the_wing_trailing_edge_mm": round(xd - w_te), "discs_above_the_ground_standing_mm": round(foot_x - xd)},
            "struck_by_the_discs_cm3": {"wing": cm3(sweep, wing), "fins": cm3(sweep, parts["tail"]), "fuselage": cm3(sweep, parts["fuselage"]),
                                        "launcher": cm3(sweep, parts["launcher"]) if "launcher" in parts else None},
            "span_mm": round(2 * p["sw_half_span"]), "length_mm": round(foot_x),
            "not_done": ["balance and mass ledger (the X tail and feet against the tail halves and fin they replace)", "the flight model with an X tail (astor_x_flight has ASTOR-X's flat tail)",
                         "hover control: elevon power in the wash against the wing's side area in a gust", "the transition from the hover to wing-borne flight and back",
                         "structure: the fins carry the motors AND the landing loads", "CFD", "whether it can stand in wind: side area of the wing 1 m above the feet"]}


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
    D = here / "data" / "astor_x2"
    D.mkdir(parents=True, exist_ok=True)
    tmp = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else D
    p = resolve()
    A = AstorX2()
    parts = A.parts(p)
    asm = cq.Assembly()
    for n, sh in parts.items():
        asm.add(sh, name=n)
    asm.save(str(D / "astor_x2.step"))
    col = {**F.PART_COL, "launcher": "#c0392b", "ailerons": "#8a5a1a", "flaps": "#2e7d32", "parachute": "#e67e22", "tail": "#1f5fa8", "elevons": "#2e7d32",
           "feet": "#111111", "motors": "#333333"}
    items = [(s, col.get(n, "#888888"), 0.3 if n == "props" else 1.0) for n, s in parts.items() if n != "vessel"]
    foot_x = A._foot_x(p)
    up = lambda s: s.rotate((0, 0, 0), (0, 1, 0), 90.0).translate((0, 0, foot_x))                # nose up, the feet on z = 0
    ground = cq.Workplane("XY").circle(520.0).extrude(-4.0)
    standing = [(up(s), c, o) for s, c, o in items] + [(ground, "#b9c2b0", 1.0)]
    views = [("flying: front quarter", items, (22, -130)), ("flying: from above", items, (90, -90)), ("flying: from the side", items, (0, -90)),
             ("flying: from behind", items, (0, 0)), ("standing on its tail", standing, (12, -60)), ("standing: from the side", standing, (0, -90))]
    f, axs = plt.subplots(3, 2, figsize=(12, 14.5), facecolor="white")
    for k, (ax, (t, its, (e, a))) in enumerate(zip(axs.flat, views)):
        png = tmp / f"ax2_{k}.png"
        ax.imshow(plt.imread(F.render(str(png), its, e, a, size=(1100, 850))))
        ax.set_axis_off()
        ax.set_title(t, fontsize=11)
        png.unlink()
    f.suptitle("ASTOR-X2, a tail-sitter: ASTOR-X's swept wing (1.20 m); an X of four fins (blue) with elevons (green) and feet; four tractors on the fins", fontsize=11)
    f.tight_layout()
    f.savefig(D / "astor_x2_views.png", dpi=115)
    c = checks(p)
    (D / "astor_x2_checks.yaml").write_text(yaml.safe_dump(c, sort_keys=False, allow_unicode=True, width=150))
    print(yaml.safe_dump(c, sort_keys=False, allow_unicode=True, width=150))
