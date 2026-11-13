import pytest

from vegeta.fidia.contract import CONTRACT, EXAMPLE_SOURCE, validate_source

HEAD = "import cadquery as cq\nfrom vegeta.dedalus import Design, Parameter\n\n"
BODY = "\nclass M(Design):\n    def build(self, p):\n        return cq.Workplane().box(1, 1, 1)\n"


def test_example_passes():
    assert validate_source(EXAMPLE_SOURCE) == []
    assert "cq.Assembly" in CONTRACT and "Design" in CONTRACT


@pytest.mark.parametrize("line", [
    "import os", "import subprocess", "from pathlib import Path", "import socket", "import cadquery.occ_impl.exporters as ex",
    "from . import x", "from vegeta.dedalus import loading", "from cadquery import exporters", "import vegeta.fidia", "from vegeta import dedalus",
])
def test_imports_rejected(line):
    problems = validate_source(HEAD + line + "\n" + BODY)
    assert problems and any("import" in p or "vegeta.dedalus" in p for p in problems)


@pytest.mark.parametrize("expr", [
    "open('x')", "eval('1')", "exec('1')", "compile('1', 'f', 'exec')", "__import__('os')", "getattr(cq, 'x')",
    "globals()", "breakpoint()", "cq.exporters", "cq.Workplane().val().exportStep('x')".replace("exportStep", "export"),
    "cq.importers", "p.__class__", "(1).__subclasses__", "np.save", "np.load", "x.write_text('a')", "np.ctypeslib",
])
def test_names_and_attributes_rejected(expr):
    src = HEAD + "import numpy as np\n" + BODY.replace("return", f"x = 1\n        y = {expr}\n        return")
    assert validate_source(src), expr


def test_allowed_code_passes():
    src = HEAD + "import math\nimport numpy as np\nfrom typing import Any\n" + BODY.replace(
        "return", "r = math.sqrt(np.float64(4.0))\n        return")
    assert validate_source(src) == []


def test_structure_rules():
    assert any("syntax error" in p for p in validate_source("def f(:\n"))
    assert any("exactly one class" in p for p in validate_source(HEAD + "x = 1\n"))
    assert any("exactly one class" in p for p in validate_source(HEAD + BODY + BODY.replace("class M", "class N")))
    assert any("no build" in p for p in validate_source(HEAD + "class M(Design):\n    pass\n"))
    assert any("global" in p for p in validate_source(HEAD + BODY.replace("return", "global g\n        return")))
