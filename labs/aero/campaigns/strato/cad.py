"""Strato CAD: STEP/STL files and pictures of the 15, 30, 50 and 80 m ships (Stage V8.5).

    python labs/aero/campaigns/strato/cad.py [--out DIR] [--lengths 15 30 50 80] [--no-stl]

Writes, per length, ``strato_<L>m.step`` (the whole ship, one solid), ``parts/<part>.step`` (each part on its own,
for editing), ``strato_<L>m.stl`` and a ``summary.json`` (Dedalus's measurements next to the analytic hull numbers);
and the pictures ``views_<L>m.png`` (four views, parts coloured) and ``family.png`` (the four ships to scale).
Pictures render off-screen with pyvista (a virtual display is used when there is none).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "designs"))

import strato  # noqa: E402
import strato_geometry as sg  # noqa: E402

COLORS = {"hull": "#d9dde3", "fins": "#e07a5f", "gondola": "#3d405b", "battery": "#f2cc8f",
          "solar": "#1d3557", "propellers": "#81b29a", "structure": "#9a8c98",
          "tether": "#52514e"}


def export(length: float, out: Path, stl: bool = True) -> dict:
    d = strato.StratoAirship()
    over = strato.for_length(length)
    tag = f"{length:g}m"
    folder = out / tag
    res = d.run(folder, formats=("step", "stl") if stl else ("step",), stl_tolerance=0.002 * length * 1000 / 15,
                stl_angular_tolerance=0.2, **over)
    if not res.ok:
        raise RuntimeError(f"{tag}: {res.messages}")
    for k in ("step", "stl"):
        if k in res.artifacts:
            p = Path(res.artifacts[k])
            p.rename(folder / f"strato_{tag}.{k}")
    geoms = strato.parts(**over)
    for name, g in geoms.items():
        g.export_step(folder / "parts" / f"{name}.step")
    model = sg.DEMONSTRATOR.scaled(length)
    summary = {"analytic": model.summary(), "cad_mm": res.metrics,
               "hull_cad_volume_m3": geoms["hull"].volume / 1e9, "hull_cad_area_m2": geoms["hull"].surface_area / 1e6}
    (folder / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return geoms


def _plotter(shape=(1, 1), size=(1600, 1000)):
    import pyvista as pv
    pv.OFF_SCREEN = True
    pl = pv.Plotter(off_screen=True, shape=shape, window_size=size)
    pl.set_background("white")
    return pl


def _tol(g) -> float:
    """Render tolerance [mm]: 0.4 mm per 15 m of the part's largest dimension, finer than the drawn solar array."""
    return max(0.4 * max(g.dimensions) / 15000, 0.05)


def _add(pl, geoms: dict, shift=(0.0, 0.0, 0.0), scale: float = 1e-3):
    from vegeta.dedalus.viz import to_pyvista
    for name, g in geoms.items():
        mesh = to_pyvista(g, tolerance=_tol(g), angular_tolerance=0.1).scale(scale).translate(shift)
        pl.add_mesh(mesh, color=COLORS[name], smooth_shading=True, specular=0.3,
                    opacity=0.5 if name == "gondola" else 1.0, label=name)


def views(geoms: dict, path: Path, title: str):
    pl = _plotter((2, 2), (1800, 1200))
    cams = [("iso", lambda: pl.view_isometric()), ("side (port)", lambda: pl.view_xz(negative=True)),
            ("top", lambda: pl.view_xy()), ("front", lambda: pl.view_yz())]
    for i, (name, cam) in enumerate(cams):
        pl.subplot(i // 2, i % 2)
        _add(pl, geoms)
        cam()
        if name == "iso":
            pl.camera.azimuth = 200
            pl.camera.elevation = 18
            pl.add_legend(bcolor="white", face="rectangle", size=(0.2, 0.3), loc="lower right")
        pl.add_text(f"{title} — {name}", font_size=11, color="black")
        pl.reset_camera()
        pl.camera.zoom(1.25 if name != "front" else 1.0)
    pl.screenshot(str(path))
    pl.close()


def family(all_geoms: dict, path: Path):
    pl = _plotter((1, 1), (1800, 900))
    z = 0.0
    for length, geoms in sorted(all_geoms.items()):
        _add(pl, {k: v for k, v in geoms.items() if k != "battery"}, shift=(0.0, 0.0, z))
        pl.add_point_labels([[-0.1 * length - 3.0, 0.0, z]], [f"{length:g} m"], font_size=18, text_color="black",
                            shape_opacity=0.0, show_points=False)
        z -= length / (15 / 3.5) * 1.9 + 4.0
    pl.view_xz(negative=True)
    pl.add_text("Strato family to scale (port side): 15, 30, 50, 80 m, fineness 4.3", font_size=12, color="black")
    pl.reset_camera()
    pl.camera.zoom(1.1)
    pl.screenshot(str(path))
    pl.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=HERE / "cad")
    ap.add_argument("--lengths", type=float, nargs="+", default=list(sg.REFERENCE_LENGTHS))
    ap.add_argument("--no-stl", action="store_true")
    a = ap.parse_args(argv)
    all_geoms = {}
    for L in a.lengths:
        geoms = export(L, a.out, stl=not a.no_stl)
        views(geoms, a.out / f"views_{L:g}m.png", f"Strato {L:g} m")
        all_geoms[L] = geoms
        print(f"{L:g} m: {a.out / f'{L:g}m'}")
    if len(all_geoms) > 1:
        family(all_geoms, a.out / "family.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
