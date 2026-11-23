"""Status overview: every revision against every analysis ever run in the workspace."""
from __future__ import annotations

import html

NOT_RUN = "NOT RUN"
FAILED = "FAILED"


def _fmt(v, spec=".3g"):
    return "?" if v is None else format(v, spec)


def summarise(ev) -> str:
    if ev is None:
        return NOT_RUN
    if not ev.ok:
        return FAILED
    m = ev.metrics
    if ev.kind == "fea":
        text = f"u={_fmt(m.get('max_displacement'))} vM={_fmt(m.get('max_von_mises'))}"
        if m.get("safety_factor_yield") is not None:
            text += f" SF={m['safety_factor_yield']:.2f}"
        return text
    if ev.kind == "cfd":
        if m.get("Cd") is None:
            return f"mesh {m.get('mesh_cells', '?')} cells"
        return f"Cd={_fmt(m.get('Cd'))} Cl={_fmt(m.get('Cl'))}" + ("" if m.get("converged") else " (not converged)")
    if ev.kind == "print":
        g = m.get("filament_used_g")
        return f"{m.get('estimated_time')} {'' if g is None else f'{g:g} g'}".strip()
    return "ok"


class StatusTable:
    def __init__(self, columns: list[str], rows: list[dict]):
        self.columns = columns
        self.rows = rows

    def __str__(self) -> str:
        if not self.rows:
            return "(no revisions)"
        widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in self.rows)) for c in self.columns}
        head = "  ".join(c.ljust(widths[c]) for c in self.columns)
        lines = [head, "  ".join("-" * widths[c] for c in self.columns)]
        lines += ["  ".join(str(r.get(c, "")).ljust(widths[c]) for c in self.columns) for r in self.rows]
        return "\n".join(lines)

    __repr__ = __str__

    def _repr_html_(self) -> str:
        def cell(v):
            style = " style='color:#999'" if v == NOT_RUN else (" style='color:#c62828'" if v == FAILED else "")
            return f"<td{style}>{html.escape(str(v))}</td>"

        head = "".join(f"<th>{html.escape(c)}</th>" for c in self.columns)
        body = "".join("<tr>" + "".join(cell(r.get(c, "")) for c in self.columns) + "</tr>" for r in self.rows)
        return f"<table><tr>{head}</tr>{body}</table>"

    def to_dataframe(self):
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover
            raise ImportError("to_dataframe needs pandas: pip install pandas") from exc
        return pd.DataFrame(self.rows, columns=self.columns)


def status_table(ws, design: str | None = None) -> StatusTable:
    revs = ws.revisions(design)
    analyses: list[tuple[str, str]] = []
    for r in revs:
        for ev in r.evaluations():
            if (ev.kind, ev.name) not in analyses:
                analyses.append((ev.kind, ev.name))
    analyses.sort(key=lambda a: (("fea", "cfd", "print").index(a[0]) if a[0] in ("fea", "cfd", "print") else 9, a[1]))
    columns = ["rev", "design", "parent", "label", "changes", "CAD"] + [f"{k}:{n}" for k, n in analyses]
    rows = []
    for r in revs:
        summary = r.geometry_summary()
        attempts = [a for a in r.annotations() if a.get("type") == "generate"]
        if summary is None:
            cad = FAILED if attempts and attempts[-1].get("status") != "success" else NOT_RUN
        elif summary.get("status") != "success":
            cad = FAILED
        else:
            vol = summary.get("metrics", {}).get("volume")
            cad = "ok" if vol is None else f"V={vol:.4g}"
        row = {"rev": r.id, "design": r.design_name, "parent": r.parent or "-", "label": r.current_label,
               "changes": ", ".join(f"{k}={v}" for k, v in r.changes.items()) or "-", "CAD": cad}
        for kind, name in analyses:
            row[f"{kind}:{name}"] = summarise(r.evaluation(kind, name))
        rows.append(row)
    return StatusTable(columns, rows)
