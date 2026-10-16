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
    p = kr.write("40_kestrel", "Kestrel-4Q", {"spec": {"mass": 2.0}, "table": pd.DataFrame({"a": [1.0]}, index=["r"]), "text": "all hold", "## custom": pd.Series({"x": 1}, name="s")},
                 reports_dir=d, meta={"date": "2026-10-16"})
    kr.write("38_prop_lab", "prop_lab", {"cases": [{"n": 1, "status": "NOT RUN"}]}, reports_dir=d)
    txt = p.read_text()
    assert txt.startswith("# Kestrel-4Q") and "## spec" in txt and "## custom" in txt and "| r | 1 |" in txt
    out = kr.dossier(reports_dir=d, out=tmp_path / "dossier.md")
    doc = out.read_text()
    assert doc.index("40_kestrel") < doc.index("38_prop_lab") and "| report | lines | exported |" in doc and "## Kestrel-4Q" in doc
    block = kr.paste("40_kestrel", reports_dir=d, max_lines=5)
    assert block.startswith("<!--") and "more lines" in block and "Kestrel-4Q" in capsys.readouterr().out


def test_json_export_and_dossier(tmp_path):
    import json
    d = tmp_path / "reports"
    df = pd.DataFrame({"a": [1.23456, np.nan], "b": ["x", "y"]}, index=["r1", "r2"])
    mi = pd.DataFrame({"pass": [True]}, index=pd.MultiIndex.from_tuples([("v", "back", 3.0)], names=["vehicle", "direction", "t"]))
    out = kr.export("40_kestrel", "Kestrel", {"tab": df, "multi": mi, "d": {"m": np.float64(1.6862063), "ok": np.bool_(True)}, "s": pd.Series({"x": 2}), "txt": "hello", "none": None},
                    meta={"mass": 1.686}, reports_dir=d)
    doc = json.loads(Path(out["json"]).read_text())
    assert doc["notebook"] == "40_kestrel" and doc["sections"]["tab"]["rows"][0] == {"_index": "r1", "a": 1.235, "b": "x"} and doc["sections"]["tab"]["rows"][1]["a"] is None
    assert doc["sections"]["multi"]["rows"][0]["_index"] == "v · back · 3.0" and doc["sections"]["multi"]["rows"][0]["pass"] is True
    assert doc["sections"]["d"] == {"m": 1.686, "ok": True} and doc["sections"]["s"] == {"x": 2} and doc["sections"]["txt"] == "hello" and doc["sections"]["none"] is None
    kr.export("38_prop_lab", "prop_lab", {"k": [{"n": 1}]}, reports_dir=d)
    dj = json.loads(Path(kr.dossier_json(reports_dir=d, out=tmp_path / "d.json")).read_text())
    assert list(dj["reports"]) == ["40_kestrel", "38_prop_lab"] and dj["contents"]["40_kestrel"]["sections"][0] == "tab"
    assert kr.load_json("40_kestrel", reports_dir=d)["meta"] == {"mass": 1.686} and kr.load_json("none", reports_dir=d) is None
