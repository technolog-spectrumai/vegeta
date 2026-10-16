"""The design dossier (kestrel_report): tables, series, dicts and strings as pasteable Markdown; a report per notebook;
the dossier stitched from what is on file."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import kestrel_report as kr


def test_fmt_and_tables():
    assert kr.fmt(1.23456) == "1.23" and kr.fmt(True) == "yes" and kr.fmt(float("nan")) == "—" and kr.fmt(np.float64(123456.0)) == "1.23e+05" and kr.fmt(3) == "3"
    df = pd.DataFrame({f"c{i}": [i * 1.5, np.nan] for i in range(20)}, index=["a", "b"])
    t = kr.table(df)
    assert t.count("\n") >= 4 and "6 more columns not shown" in t and "| a | 0 | 1.5 |" in t
    mi = pd.DataFrame({"x": [1.0]}, index=pd.MultiIndex.from_tuples([("v", "forward", 3.0)], names=["vehicle", "direction", "t"]))
    assert "| v · forward · 3 | 1 |" in kr.table(mi) and "| vehicle · direction · t | x |" in kr.table(mi)
    assert "| k | 1 |" in kr.mapping({"k": 1.0}) and "| key | value |" in kr.mapping({"a": [1, 2, 3]})
    assert "| col |" in kr.table(pd.DataFrame([{"col": 1}]), index=False)


def test_write_dossier_and_paste(tmp_path, capsys):
    d = tmp_path / "reports"
    p = kr.write("38_kestrel", "Kestrel-4Q", {"spec": {"mass": 2.0}, "table": pd.DataFrame({"a": [1.0]}, index=["r"]), "text": "all hold", "## custom": pd.Series({"x": 1}, name="s")},
                 reports_dir=d, meta={"date": "2026-10-16"})
    kr.write("41_prop_lab", "prop_lab", {"cases": [{"n": 1, "status": "NOT RUN"}]}, reports_dir=d)
    txt = p.read_text()
    assert txt.startswith("# Kestrel-4Q") and "## spec" in txt and "## custom" in txt and "| r | 1 |" in txt
    out = kr.dossier(reports_dir=d, out=tmp_path / "dossier.md")
    doc = out.read_text()
    assert doc.index("38_kestrel") < doc.index("41_prop_lab") and "| report | lines | exported |" in doc and "## Kestrel-4Q" in doc
    block = kr.paste("38_kestrel", reports_dir=d, max_lines=5)
    assert block.startswith("<!--") and "more lines" in block and "Kestrel-4Q" in capsys.readouterr().out
