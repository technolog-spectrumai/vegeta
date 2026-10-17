import pytest

from vegeta.yanus import File, Flow, FlowError, Package, Run
from vegeta.yanus.registry import load_flow


def test_codes_and_lookup(repo):
    assert repo.chapter("D2").name == "blown"
    assert repo.chapter("d 3").name == "frame"
    assert repo.chapter("demo/sweep").code == "D1"
    assert repo.unit("20").name == "20_study"
    assert repo.unit("20_study.ipynb").name == "20_study"
    assert repo.books["demo"].chapters["blown"].prerequisites == ("sweep",)
    with pytest.raises(KeyError):
        repo.unit("D9")


def test_validate_errors(tmp_path):
    f = Flow(root=tmp_path)
    b = f.book("x", "X", variants=("A",))
    b.kind("k", title="", script="s", workflow="w")
    b.chapter("one", kind="k", campaign="one", takes=[Run("two")])
    f.notebook("01_a", takes=[Run("x/one", variant="Z"), File("02_b:data/f.json"), File("data/nope.json")])
    f.notebook("02_b", gives=[File("data/other.json")])
    with pytest.raises(FlowError) as e:
        f.validate()
    msg = str(e.value)
    assert "unknown chapter" in msg and "no variant 'Z'" in msg and "does not give it" in msg and "without its producer" in msg


def test_cycle_and_duplicates(tmp_path):
    f = Flow(root=tmp_path)
    f.notebook("01_a", takes=[File("02_b:b.json")], gives=[File("a.json")])
    f.notebook("02_b", takes=[File("01_a:a.json")], gives=[File("b.json")])
    with pytest.raises(FlowError, match="cycle"):
        f.validate()
    with pytest.raises(FlowError, match="twice"):
        f.notebook("01_a")
    b = f.book("x", "X")
    with pytest.raises(FlowError, match="unknown kind"):
        b.chapter("c", kind="nope", campaign="c")


def test_load_by_path_and_env(repo, tmp_path, monkeypatch):
    p = tmp_path / "workflows" / "flow.py"
    fl = load_flow(p, reload=True)
    assert "01_x" in fl.notebooks and fl.root == tmp_path
    monkeypatch.setenv("VEGETA_YANUS_FLOW", str(p))
    assert load_flow() is fl


def test_notebook_folder(tmp_path):
    f = Flow(root=tmp_path)
    nb = f.notebook("15_demo", folder="demo", gives=[File("data/x.json")])
    f.notebook("20_reader", takes=[File("15_demo:data/x.json")])
    f.validate()
    assert nb.path == "notebooks/demo/15_demo.ipynb" and f.unit("15").name == "15_demo"
    from vegeta.yanus.gate import notebook_command
    assert notebook_command(nb) == "scripts/run_notebooks.sh demo/15_demo.ipynb"
