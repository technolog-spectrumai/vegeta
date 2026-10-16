"""Kestrel — the design dossier: every notebook's numbers exported as Markdown you can paste into a conversation
(notebooks 39, 41, 39, 38).

Each notebook ends with ``write(name, title, sections)``: its tables, series, dicts and verdicts rendered as compact
Markdown — numbers rounded to three significant figures, wide tables trimmed to their first columns, long tables to
their first rows — into ``designs/data/reports/<name>.md``, committed with the repository so the state of the design
travels with it. ``dossier()`` stitches every report on file into ``designs/data/kestrel_dossier.md`` with a header
of what ran and when; ``paste(name)`` prints a report (or the dossier) between fences, ready to copy. A section is
whatever the notebook hands over: a DataFrame, a Series, a dict, a list of dicts, a string, or a list of those.

The same sections go out as **JSON** — ``write_json`` → ``designs/data/reports/<name>.json`` (every table as
``{"index": [...], "columns": [...], "rows": [{...}]}``, numbers rounded to four significant figures, NaN as null),
``dossier_json`` → ``designs/data/kestrel_dossier.json`` with every notebook's report under its name — the machine-
readable copy to paste or to load (``load_json``). ``export`` writes both copies and both dossiers.
"""
from __future__ import annotations

import datetime
import math
from pathlib import Path

import numpy as np

REPORTS = Path(__file__).resolve().parent / "data" / "reports"
DOSSIER = Path(__file__).resolve().parent / "data" / "kestrel_dossier.md"
MAX_COLS = 14
MAX_ROWS = 60
ORDER = ("38_prop_lab", "39_kestrel_tests", "40_kestrel", "41_xwing", "42_kestrel2")


def fmt(v, sig: int = 3) -> str:
    """One value as the dossier prints it: floats to ``sig`` significant figures, booleans as yes/no, NaN as —."""
    if v is None:
        return "—"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if not math.isfinite(float(v)):
            return "—" if math.isnan(float(v)) else ("∞" if v > 0 else "−∞")
        f = float(v)
        if f == 0:
            return "0"
        if abs(f) >= 1e5 or abs(f) < 1e-3:
            return f"{f:.{sig - 1}e}"
        return f"{f:.{sig}g}"
    s = str(v).replace("\n", " ").replace("|", "/")
    return s if len(s) <= 90 else s[:87] + "…"


def _is_frame(x):
    return hasattr(x, "columns") and hasattr(x, "index")


def _is_series(x):
    return hasattr(x, "index") and not hasattr(x, "columns") and hasattr(x, "to_dict")


def table(df, *, max_cols: int = MAX_COLS, max_rows: int = MAX_ROWS, index: bool = True) -> str:
    """A DataFrame as a Markdown table, trimmed: the first ``max_cols`` columns, the first ``max_rows`` rows, a line
    saying what was left out; a MultiIndex flattened to ' | '-joined labels."""
    import pandas as pd
    if len(df) == 0:
        return "*(empty)*\n"
    d = df.copy()
    name = ""
    if isinstance(d.index, pd.MultiIndex):
        name = " · ".join(str(n or "") for n in d.index.names)
        d.index = [" · ".join(fmt(x) for x in k) for k in d.index]
    else:
        name = str(d.index.name or "")
    cols = list(d.columns)
    cut_c = cols[:max_cols]
    rows = d.index[:max_rows]
    head = ([name] if index else []) + [fmt(c) for c in cut_c]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        vals = ([fmt(r)] if index else []) + [fmt(d.at[r, c]) for c in cut_c]
        out.append("| " + " | ".join(vals) + " |")
    notes = []
    if len(cols) > max_cols:
        notes.append(f"{len(cols) - max_cols} more columns not shown: {', '.join(fmt(c) for c in cols[max_cols:max_cols + 8])}{'…' if len(cols) > max_cols + 8 else ''}")
    if len(d) > max_rows:
        notes.append(f"{len(d) - max_rows} more rows not shown")
    return "\n".join(out) + "\n" + ("".join(f"\n*{n}*\n" for n in notes))


def series(s, *, max_rows: int = MAX_ROWS) -> str:
    rows = list(s.items())[:max_rows]
    out = ["| " + (s.name or "") + " | value |", "|---|---|"] + [f"| {fmt(k)} | {fmt(v)} |" for k, v in rows]
    more = f"\n*{len(s) - max_rows} more rows not shown*\n" if len(s) > max_rows else ""
    return "\n".join(out) + "\n" + more


def mapping(d: dict, *, max_rows: int = MAX_ROWS) -> str:
    items = list(d.items())
    if all(isinstance(v, dict) for _, v in items) and items:
        import pandas as pd
        return table(pd.DataFrame(d).T, max_rows=max_rows)
    out = ["| key | value |", "|---|---|"] + [f"| {fmt(k)} | {fmt(v) if not isinstance(v, (list, tuple)) else ', '.join(fmt(x) for x in v[:12])} |" for k, v in items[:max_rows]]
    return "\n".join(out) + "\n"


def render(obj) -> str:
    """Any section object as Markdown."""
    if obj is None:
        return "*(none)*\n"
    if isinstance(obj, str):
        return obj.rstrip("\n") + "\n"
    if _is_frame(obj):
        return table(obj)
    if _is_series(obj):
        return series(obj)
    if isinstance(obj, dict):
        return mapping(obj)
    if isinstance(obj, (list, tuple)):
        if obj and all(isinstance(x, dict) for x in obj):
            import pandas as pd
            return table(pd.DataFrame(obj), index=False)
        return "\n\n".join(render(x) for x in obj) + "\n"
    return fmt(obj) + "\n"


def write(name: str, title: str, sections: dict, *, reports_dir=REPORTS, meta: dict | None = None) -> Path:
    """The report ``<reports_dir>/<name>.md``: a header (the notebook, the date, ``meta``), then every section in
    order — ``{heading: object}``; a heading starting with '#' keeps its level, else it is '## heading'."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "", f"*{name} — exported {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}*", ""]
    if meta:
        lines += [mapping(meta), ""]
    for head, obj in sections.items():
        h = head if head.startswith("#") else f"## {head}"
        lines += [h, "", render(obj), ""]
    path = reports_dir / f"{name}.md"
    path.write_text("\n".join(lines))
    return path


def dossier(*, reports_dir=REPORTS, out=DOSSIER, order=ORDER) -> Path:
    """Every report on file, in notebook order, into one dossier with a table of what is there."""
    reports_dir = Path(reports_dir)
    files = {p.stem: p for p in reports_dir.glob("*.md")}
    names = [n for n in order if n in files] + sorted(n for n in files if n not in order)
    lines = ["# Kestrel — the design dossier", "", f"*stitched {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} from {len(names)} report(s)*", "",
             "| report | lines | exported |", "|---|---|---|"]
    bodies = []
    for n in names:
        text = files[n].read_text()
        stamp = next((l for l in text.splitlines() if l.startswith("*") and "exported" in l), "")
        lines.append(f"| {n} | {len(text.splitlines())} | {stamp.strip('*')} |")
        bodies.append(text.replace("\n# ", "\n## ").replace("\n## ", "\n### ", 0))
    lines += [""]
    for n, b in zip(names, bodies):
        lines += ["", "---", "", b.replace("# ", "## ", 1)]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    return out


def paste(name: str | None = None, *, max_lines: int | None = None, reports_dir=REPORTS) -> str:
    """The report (or the dossier when ``name`` is None) as text, fenced for pasting; printed and returned."""
    path = DOSSIER if name is None else Path(reports_dir) / f"{name}.md"
    text = path.read_text() if path.exists() else f"*(no report {path.name} yet)*"
    if max_lines:
        ls = text.splitlines()
        text = "\n".join(ls[:max_lines]) + (f"\n… {len(ls) - max_lines} more lines in {path}" if len(ls) > max_lines else "")
    block = f"<!-- {path} -->\n{text}"
    print(block)
    return block


# ------------------------------------------------------------------------------------------------- JSON
DOSSIER_JSON = Path(__file__).resolve().parent / "data" / "kestrel_dossier.json"
SIG_JSON = 4


def _num(v):
    f = float(v)
    if not math.isfinite(f):
        return None
    if f == 0:
        return 0.0
    return float(f"{f:.{SIG_JSON}g}")


def plain(obj, *, max_rows: int = 400):
    """Any section object as plain JSON-able data: a DataFrame as index / columns / rows, a Series or dict as a dict,
    numbers rounded to ``SIG_JSON`` significant figures, NaN and inf as null, numpy scalars as Python ones."""
    import pandas as pd
    if obj is None:
        return None
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return _num(obj)
    if isinstance(obj, str):
        return obj
    if _is_frame(obj):
        d = obj
        idx = [" · ".join(str(x) for x in k) if isinstance(k, tuple) else (k if isinstance(k, str) else plain(k)) for k in d.index]
        cols = [str(c) for c in d.columns]
        rows = []
        for k, (_, r) in zip(idx, d.iterrows()):
            row = {"_index": k}
            for c, v in zip(cols, r.values):
                row[c] = plain(v)
            rows.append(row)
        out = {"index": idx[:max_rows], "columns": cols, "rows": rows[:max_rows]}
        if len(rows) > max_rows:
            out["truncated_rows"] = len(rows) - max_rows
        return out
    if _is_series(obj):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [plain(x) for x in list(obj)[:max_rows]]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if hasattr(obj, "__dict__") and not callable(obj):
        return {k: plain(v) for k, v in vars(obj).items() if not k.startswith("_")}
    return str(obj)


def write_json(name: str, title: str, sections: dict, *, reports_dir=REPORTS, meta: dict | None = None) -> Path:
    """The report as JSON: ``{"notebook", "title", "exported", "meta", "sections": {heading: data}}``."""
    import json
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    doc = {"notebook": name, "title": title, "exported": datetime.datetime.now().isoformat(timespec="minutes"), "meta": plain(meta or {}),
           "sections": {str(k).lstrip("# ").strip(): plain(v) for k, v in sections.items()}}
    path = reports_dir / f"{name}.json"
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False, default=str))
    return path


def dossier_json(*, reports_dir=REPORTS, out=DOSSIER_JSON, order=ORDER) -> Path:
    """Every JSON report on file under its notebook's name, with a table of contents."""
    import json
    reports_dir = Path(reports_dir)
    files = {p.stem: p for p in reports_dir.glob("*.json")}
    names = [n for n in order if n in files] + sorted(n for n in files if n not in order)
    doc = {"dossier": "Kestrel", "stitched": datetime.datetime.now().isoformat(timespec="minutes"), "reports": {}}
    for n in names:
        doc["reports"][n] = json.loads(files[n].read_text())
    doc["contents"] = {n: {"title": r.get("title"), "exported": r.get("exported"), "sections": list(r.get("sections", {}))} for n, r in doc["reports"].items()}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False, default=str))
    return out


def load_json(name: str | None = None, *, reports_dir=REPORTS) -> dict | None:
    import json
    path = DOSSIER_JSON if name is None else Path(reports_dir) / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def export(name: str, title: str, sections: dict, *, meta: dict | None = None, reports_dir=REPORTS) -> dict:
    """Both copies of a notebook's report (Markdown and JSON) and both dossiers; returns the paths and the sizes."""
    reports_dir = Path(reports_dir)
    md = write(name, title, sections, reports_dir=reports_dir, meta=meta)
    js = write_json(name, title, sections, reports_dir=reports_dir, meta=meta)
    out_md, out_js = (DOSSIER, DOSSIER_JSON) if reports_dir == REPORTS else (reports_dir.parent / DOSSIER.name, reports_dir.parent / DOSSIER_JSON.name)
    d_md, d_js = dossier(reports_dir=reports_dir, out=out_md), dossier_json(reports_dir=reports_dir, out=out_js)   # a test's reports never restitch the real dossier
    return {"markdown": str(md), "json": str(js), "dossier_md": str(d_md), "dossier_json": str(d_js),
            "json_kB": round(js.stat().st_size / 1024, 1), "dossier_json_kB": round(d_js.stat().st_size / 1024, 1)}
