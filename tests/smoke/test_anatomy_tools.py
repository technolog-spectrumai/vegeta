"""tools/dump_ast.py → tools/ast_report.py → tools/graph_metrics.py on a tiny tree: the XML parses and holds the modules,
imports, definitions, line counts, entry points and syntax trees; the report and the graph metrics read it back."""
import gzip
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))
import ast_report  # noqa: E402
import dump_ast  # noqa: E402

LIB = '''"""A library."""
import json


def helper(x):
    # a comment
    return json.dumps(x)


class Thing:
    def go(self):
        return helper(1)
'''
APP = '''from lib import helper, Thing
import matplotlib.pyplot as plt


def unused_here():
    return 0


if __name__ == "__main__":
    plt.savefig(helper(Thing().go()))
'''
NB = {"cells": [{"cell_type": "code", "source": ["%matplotlib inline\n", "import lib\n", "lib.helper(2)\n"]},
                {"cell_type": "markdown", "source": ["# title"]}], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}


@pytest.fixture()
def tree(tmp_path):
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "lib.py").write_text(LIB)
    (root / "pkg" / "app.py").write_text(APP)
    (root / "pkg" / "use.ipynb").write_text(json.dumps(NB))
    out = tmp_path / "ast.xml.gz"
    stats = dump_ast.dump(root, out)
    return root, out, stats


def test_the_dump_holds_modules_imports_defs_lines_and_trees(tree):
    root, out, stats = tree
    assert stats["modules"] == 3 and stats["errors"] == 0
    doc = ET.parse(gzip.open(out)).getroot()
    mods = {m.get("path"): m for m in doc.iter("module")}
    assert set(mods) == {"pkg/lib.py", "pkg/app.py", "pkg/use.ipynb"}
    lib = mods["pkg/lib.py"]
    assert (lib.get("comment"), lib.get("docstring"), lib.get("kind")) == ("1", "1", "lib")
    assert int(lib.get("code")) + 1 + 1 + int(lib.get("blank")) == int(lib.get("lines"))
    assert {d.get("qual") for d in lib.iter("def")} == {"helper", "Thing", "Thing.go"}
    app = mods["pkg/app.py"]
    assert app.get("kind") == "script" and app.get("main") == "1"
    assert {i.get("target") for i in app.iter("import") if i.get("target")} == {"pkg/lib.py"}
    assert {i.get("external") for i in app.iter("import") if i.get("external")} == {"matplotlib"}
    assert app.find("ast/Module/FunctionDef").get("name") == "unused_here"
    assert mods["pkg/use.ipynb"].get("kind") == "notebook"                       # the magic did not break parsing
    entries = {(e.get("kind"), e.get("module")) for e in doc.iter("entry")}
    assert entries == {("main", "pkg/app.py"), ("notebook", "pkg/use.ipynb")}


def test_the_report_reads_it_back(tree):
    _, out, _ = tree
    st = ast_report.analyse(ast_report.load(out))
    assert st["totals"]["modules"] == 3 and st["orphans"] == []
    assert [u["def"] for u in st["unreferenced"]] == ["unused_here"]


def test_graph_metrics_answer_serve_purpose_how(tree):
    pytest.importorskip("networkx")
    import graph_metrics
    _, out, _ = tree
    st, dep, calls = graph_metrics.analyse(*graph_metrics.load(out))
    assert dep.has_edge("pkg/app.py", "pkg/lib.py")
    assert calls.has_edge("pkg/app.py::<module>", "pkg/lib.py::helper")
    assert calls.has_edge("pkg/lib.py::Thing.go", "pkg/lib.py::helper")
    assert st["purpose"]["products"]["figures"]["entries"] == 1                  # app.py saves a figure
    assert st["serve"]["areas"]["pkg"]["non_test_entries"] == 2
