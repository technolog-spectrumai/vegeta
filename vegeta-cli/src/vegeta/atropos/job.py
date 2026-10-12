"""A cut job: kerf-compensate the parts, nest them on sheets, write one SVG and one G-code file per sheet, a preview
and a summary; count the sheets, the cut length, the time and the material."""
from __future__ import annotations

import json
import time
from pathlib import Path as FsPath

from .gcode import plan, write_gcode
from .geometry import Part, parts_to_json
from .nest import nest
from .result import Result
from .settings import CutSettings
from .svg import write_svg


def cut_job(parts: list[Part], settings: CutSettings, out_dir: str | FsPath, *, name: str = "job",
            gcode: bool = True, png: bool = True) -> Result:
    """Everything from parts to files. The result's ``nesting`` attribute holds the placements; a part that does not
    fit the sheet gives a failed result (nothing is written but ``parts.json``)."""
    t0 = time.perf_counter()
    out = FsPath(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res = Result(kind="atropos.cut_job", metadata={"name": name, "settings": settings.to_dict()})
    res.nesting = None
    (out / "parts.json").write_text(json.dumps(parts_to_json(parts), indent=1))
    res.artifacts["parts_json"] = out / "parts.json"
    if not parts:
        return res.fail("no parts")
    kerf = settings.operation("cut").kerf_mm
    compensated = [p.kerf_compensated(kerf) for p in parts]
    try:
        nesting = nest(compensated, settings)
    except ValueError as exc:
        res.duration_s = time.perf_counter() - t0
        return res.fail(str(exc))
    res.nesting = nesting
    sheet = settings.sheet
    per_sheet = []
    totals: dict[str, float] = {}
    travel = 0.0
    for i in range(nesting.n_sheets):
        on = nesting.parts_on(i)
        tag = f"{i + 1:02d}"
        svg = write_svg(out / f"sheet_{tag}.svg", on, sheet.width_mm, sheet.height_mm, settings.operations, settings.labels,
                        title=f"{name} sheet {i + 1}/{nesting.n_sheets}")
        res.artifacts[f"svg_{tag}"] = svg
        job = write_gcode(out / f"sheet_{tag}.gcode", on, settings, f"{name} sheet {i + 1}/{nesting.n_sheets}") if gcode \
            else plan(on, settings)
        if gcode:
            res.artifacts[f"gcode_{tag}"] = job.path
        for k, v in job.cut_length_mm.items():
            totals[k] = totals.get(k, 0.0) + v
        travel += job.travel_length_mm
        per_sheet.append({"sheet": i + 1, "parts": len(on), "utilisation": nesting.utilisation[i],
                          "time_s": job.total_time_s, "cut_length_mm": job.total_cut_length_mm,
                          "bbox": job.bbox})
    area = sum(p.area() for p in parts)
    m = res.metrics
    m["parts"] = len(parts)
    m["sheets"] = nesting.n_sheets
    for k, v in sorted(totals.items()):
        m[f"{k}_length_mm"] = v
    m["travel_length_mm"] = travel
    m["time_s"] = sum(s["time_s"] for s in per_sheet)
    m["utilisation"] = sum(nesting.utilisation) / max(len(nesting.utilisation), 1)
    m["parts_area_mm2"] = area
    m["parts_mass_g"] = area * sheet.thickness_mm * 1e-3 * sheet.density_g_cm3
    m["sheets_mass_g"] = nesting.n_sheets * sheet.mass_g
    res.metadata["per_sheet"] = per_sheet
    res.metadata["placements"] = nesting.table()
    if png:
        try:
            import matplotlib

            matplotlib.use("Agg")
            from .plotting import plot_nesting

            fig = plot_nesting(nesting, settings.operations, settings.labels)
            fig.savefig(out / "preview.png", dpi=110)
            import matplotlib.pyplot as plt

            plt.close(fig)
            res.artifacts["preview_png"] = out / "preview.png"
        except ImportError as exc:          # pragma: no cover
            res.messages.append(f"no preview: {exc}")
    res.duration_s = time.perf_counter() - t0
    res.save_json(out / "summary.json")
    res.artifacts["summary_json"] = out / "summary.json"
    return res
