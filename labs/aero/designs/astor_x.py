"""ASTOR-X — ASTOR-TWIN with its two wing motors replaced by four on an X of four panels (2026-11-24). A FIRST DRAWING.

Asked for after ``astor_big``: on the scorecard, four rotors on an X in addition to the wing and the tail let the
aircraft hover, stop and fly below its stall, because thrust on four corners holds the attitude at any airspeed
(GAVILAN-X's arrangement). This file draws that aircraft; it is not yet a design:

- the wing, tail, body, launcher and vessel are ASTOR-TWIN's (PROTOHAWK), unchanged;
- four panels at 45 degrees leave the body at the centre of gravity's station (their quarter chord at 435 mm), so
  the four thrusts act about it; each carries a motor pod on its tip with a tractor propeller;
- four propellers of the two 13-inch discs' total area (9.2 inches each): ``astor_big`` keeps the thrust table;
- every size below is a HAND choice. Nothing here has been balanced, analysed for stability, sized structurally or
  solved in CFD; the mass (+0.65 kg, HAND) is ``astor_big``'s.

``checks()`` measures the clearances from the solids.
"""
from __future__ import annotations

import dataclasses
import math

import cadquery as cq
from vegeta.dedalus import Parameter

import astor_twin as W
import astor_v as V

X = dict(radius=450.0, root_chord=170.0, tip_chord=130.0, quarter_chord_x=435.0, thickness=0.10,     # HAND: astor_big's 4 x 0.45 m x 0.15 m
         pod_d=46.0, pod_len=120.0, prop_diameter=round(330.2 / math.sqrt(2.0), 1))                  # the same total disc area on four
_NEW = [Parameter("x_radius", X["radius"], "mm", min=150.0, description="each X panel, from the axis to its motor"),
        Parameter("x_root_chord", X["root_chord"], "mm", min=40.0), Parameter("x_tip_chord", X["tip_chord"], "mm", min=40.0),
        Parameter("x_quarter_chord_x", X["quarter_chord_x"], "mm", description="the X panels' quarter chord from the nose: ASTOR-TWIN's centre of gravity"),
        Parameter("x_thickness", X["thickness"], "", min=0.04, max=0.2), Parameter("x_pod_d", X["pod_d"], "mm", min=20.0),
        Parameter("x_pod_len", X["pod_len"], "mm", min=40.0), Parameter("x_prop_diameter", X["prop_diameter"], "mm", min=50.0)]
ANGLES = (45.0, 135.0, 225.0, 315.0)


class AstorX(W.AstorTwin):
    """ASTOR-X: ASTOR-TWIN's wing, tail and body; four motors on an X of four panels in place of the two wing motors."""

    parameters = list(W.AstorTwin.parameters) + _NEW

    def _x_le(self, p):
        return p["x_quarter_chord_x"] - 0.25 * p["x_root_chord"]

    def _pod_x(self, p):
        return self._x_le(p) + 0.25 * (p["x_root_chord"] - p["x_tip_chord"]) - 0.60 * p["x_pod_len"]

    def _each(self, shape):
        out = None
        for a in ANGLES:
            s = shape.rotate((0, 0, 0), (1, 0, 0), a)
            out = s if out is None else out.union(s)
        return out

    def _x_panels(self, p):
        sweep = math.degrees(math.atan(0.25 * (p["x_root_chord"] - p["x_tip_chord"]) / p["x_radius"]))       # a straight quarter-chord line
        one = V._panel(p["x_root_chord"], p["x_tip_chord"], p["x_radius"], sweep, p["x_thickness"]).translate((self._x_le(p), 0, 0))
        return self._each(one)

    def _motors(self, p):
        x0 = self._pod_x(p)
        pod = cq.Workplane("YZ").circle(p["x_pod_d"] / 2).extrude(p["x_pod_len"]).faces(">X").edges().fillet(p["x_pod_d"] * 0.35)
        hub = cq.Workplane("YZ").circle(10.0).extrude(-12.0)
        return self._each(pod.union(hub).translate((x0, p["x_radius"], 0)))

    def _props(self, p):
        disc = cq.Workplane("YZ").circle(p["x_prop_diameter"] / 2).extrude(1.5).translate((self._pod_x(p) - 12.0, p["x_radius"], 0))
        return self._each(disc)

    def parts(self, p) -> dict:
        p = {**p, "prop_fold_deg": -1.0}                                       # no folding propellers on the wing
        out = super().parts(p)
        out["x_panels"] = self._x_panels(p)
        return out


def variant(**over) -> dict:
    return W.variant(**over)


def resolve(q=None) -> dict:
    return AstorX().resolve(**dict(q or variant(net_launchers=1, talon_deploy=-1.0, net_count=0, launcher_depression_deg=0.0)))


def checks(p=None) -> dict:
    p = p or resolve()
    A = AstorX()
    parts = A.parts(p)
    c45 = math.cos(math.radians(45.0))
    r, d = p["x_radius"], p["x_prop_diameter"]
    wing = parts["wing"].union(parts["ailerons"])
    if "flaps" in parts:
        wing = wing.union(parts["flaps"])
    sweep_vol = cq.Workplane("YZ").circle(d / 2 + 0.5).extrude(1200.0).translate((A._pod_x(p) - 12.0, r, 0))      # one disc's stream tube, straight back
    tubes = A._each(sweep_vol)

    def cm3(a, b):
        try:
            return round(a.intersect(b).val().Volume() / 1e3, 1)
        except Exception:
            return 0.0
    bb = parts["x_panels"].union(parts["props"]).val().BoundingBox()
    area = 4 * 0.5 * (p["x_root_chord"] + p["x_tip_chord"]) * (r - p["fuselage_diameter"] / 2) / 1e6
    return {"what": "ASTOR-X, a first drawing: clearances measured from the solids; every size is a HAND choice",
            "x_panels": {"each_from_the_axis_mm": r, "root_chord_mm": p["x_root_chord"], "tip_chord_mm": p["x_tip_chord"], "exposed_area_m2_all_four": round(area, 3),
                         "quarter_chord_from_the_nose_mm": p["x_quarter_chord_x"], "motor_y_and_z_mm": round(r * c45, 1)},
            "propellers": {"count": 4, "diameter_mm": d, "diameter_in": round(d / 25.4, 1), "disc_plane_from_the_nose_mm": round(A._pod_x(p) - 12.0),
                           "between_neighbouring_discs_mm": round(2 * r * c45 - d), "disc_edge_from_the_fuselage_side_mm": round(r - d / 2 - p["fuselage_diameter"] / 2),
                           "disc_edge_above_or_below_the_wing_plane_mm": round(r * c45 - d / 2 - abs(p["wing_plane_z"]))},
            "the_propellers_wash_straight_back": {"wing_inside_the_four_stream_tubes_cm3": cm3(tubes, wing), "tail_inside_cm3": cm3(tubes, parts["tail"]),
                                                  "launcher_inside_cm3": cm3(tubes, parts["launcher"]) if "launcher" in parts else None,
                                                  "note": "0 for the wing: the wash passes above and below it, so the wing's flaps are no longer blown"},
            "overlaps_cm3": {"x_panels_with_the_wing": cm3(parts["x_panels"], wing), "x_panels_with_the_launcher": cm3(parts["x_panels"], parts["launcher"]) if "launcher" in parts else None,
                             "note": "the panels' roots cross the wing's root inside the body's fairing; a real joint is not drawn"},
            "box_of_the_x_with_its_propellers_mm": {"y": round(bb.ylen), "z": round(bb.zlen)},
            "span_mm": round(2 * W.WING["half_span"]), "length_mm": p["fuselage_length"],
            "not_done": ["balance and mass ledger (astor_big: +0.65 kg, HAND)", "stability and control with four rotors", "structure of the X and its joint to the wing",
                         "CFD", "how it stands, launches and lands (it would sit on its tail or hang from a rail: not drawn)", "stowing: the X does not fold"]}


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
    col = {**F.PART_COL, "launcher": "#c0392b", "ailerons": "#8a5a1a", "flaps": "#2e7d32", "parachute": "#e67e22", "shoulder": "#222222", "tail_flaps": "#2e7d32",
           "x_panels": "#1f5fa8", "motors": "#333333"}
    items = [(s, col.get(n, "#888888"), 0.3 if n == "props" else 1.0) for n, s in parts.items() if n != "vessel"]
    views = [("front quarter", (22, -130)), ("from above", (90, -90)), ("from the side", (0, -90)), ("from the front", (0, 180)), ("from below", (-30, -125)), ("rear quarter", (20, -50))]
    f, axs = plt.subplots(3, 2, figsize=(12, 13.5), facecolor="white")
    for ax, (t, (e, a)) in zip(axs.flat, views):
        png = tmp / f"ax_{e}_{a}.png"
        ax.imshow(plt.imread(F.render(str(png), items, e, a, size=(1100, 800))))
        ax.set_axis_off()
        ax.set_title(t, fontsize=11)
        png.unlink()
    f.suptitle("ASTOR-X, a first drawing: ASTOR-TWIN's wing, tail and body; four 9.2-inch propellers on an X of four panels (blue) at the centre of gravity", fontsize=11)
    f.tight_layout()
    f.savefig(D / "astor_x_views.png", dpi=115)
    c = checks(p)
    (D / "astor_x_checks.yaml").write_text(yaml.safe_dump(c, sort_keys=False, allow_unicode=True, width=150))
    print(yaml.safe_dump(c, sort_keys=False, allow_unicode=True, width=150))
