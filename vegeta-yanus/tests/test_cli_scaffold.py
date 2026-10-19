import json
import py_compile

import pytest

from conftest import make_run
from vegeta.yanus import cli, pins, scaffold
from vegeta.yanus.registry import load_flow


def test_status_why_doctor(repo, capsys):
    assert cli.main(["status"]) == 2
    out = capsys.readouterr().out
    assert "D1   sweep" in out and "✘ 20_study" in out and "needs ✘ definitely-not-a-tool-xyz missing" in out
    assert cli.main(["status", "--json"]) == 2
    d = json.loads(capsys.readouterr().out)
    assert d["notebooks"]["20_study"]["state"] == "blocked"
    assert cli.main(["why", "20"]) == 2
    assert "the gate is closed" in capsys.readouterr().out
    assert cli.main(["doctor"]) == 0
    assert "this machine can run" in capsys.readouterr().out


def test_pin_unpin_suggest(repo, capsys):
    d = make_run(repo.root, "demo", "sweep", "r1")
    assert cli.main(["pin", "20_study", "--suggest"]) == 0
    assert "yn pin 20_study D1 --variant A side_labs/runs/demo/sweep/r1" in capsys.readouterr().out
    assert pins.read_pins(repo.root, "20_study") == {}                      # suggesting writes nothing
    assert cli.main(["pin", "20_study", "D1", "--variant", "A", str(d)]) == 0     # the order the gate's fix prints
    assert cli.main(["pin", "--variant", "A", "20_study", "D1", str(d)]) == 0
    assert pins.read_pins(repo.root, "20_study") == {"demo/sweep@A": "side_labs/runs/demo/sweep/r1"}
    assert cli.main(["status", "20_study"]) == 0
    with pytest.raises(SystemExit, match="takes no demo/blown"):
        cli.main(["pin", "20_study", "D2", "--variant", "A", str(d)])
    assert cli.main(["unpin", "20_study", "D1", "--variant", "A"]) == 0
    assert pins.read_pins(repo.root, "20_study") == {}


def test_check(repo, capsys):
    assert cli.main(["check"]) == 0
    pins.set_pin(repo.root, "20_study", "demo/blown@A", repo.root / "x")
    assert cli.main(["check"]) == 1
    assert "not a run take" in capsys.readouterr().out


def test_stamp(repo, capsys):
    (repo.root / "data" / "maker.json").write_text("{}")
    assert cli.main(["stamp", "10_maker", str(repo.root / "data" / "maker.json")]) == 0
    with pytest.raises(SystemExit, match="gives no"):
        cli.main(["stamp", "20_study", str(repo.root / "data" / "maker.json")])


def test_new_notebook(tmp_path):
    (tmp_path / "notebooks").mkdir()
    (tmp_path / "side_labs").mkdir()
    ff = tmp_path / "side_labs" / "flow.py"
    ff.write_text("from vegeta.yanus import Flow, File, Run, Package\nflow = Flow('t')\n")
    out = scaffold.new_notebook(tmp_path, "45_thermal", title="thermal")
    nb = json.loads(out[0].read_text())
    src = "".join("".join(c["source"]) for c in nb["cells"])
    assert 'yanus.enter("45_thermal", debug=DEBUG)' in src and "GATE.leave()" in src and "mock.setup" in src
    try:
        import nbformat
        nbformat.validate(nbformat.reads(out[0].read_text(), as_version=4))
    except ImportError:
        pass
    assert "45_thermal" in load_flow(ff, reload=True).notebooks
    with pytest.raises(FileExistsError):
        scaffold.new_notebook(tmp_path, "45_thermal")
    with pytest.raises(ValueError):
        scaffold.new_notebook(tmp_path, "Thermal")


def test_new_chapter(tmp_path):
    (tmp_path / "side_labs").mkdir()
    ff = tmp_path / "side_labs" / "flow.py"
    ff.write_text("from vegeta.yanus import Flow, Run\nflow = Flow('t')\nb = flow.book('kestrel', 'K')\n"
                  "b.kind('fea', title='FEA', script='s.py', workflow='w')\nb.chapter('frame', kind='fea', campaign='frame')\n")
    scaffold.new_chapter(tmp_path, "kestrel/thermal", kind="heat", new_kind=True, takes=("frame",))
    sp = tmp_path / "side_labs" / "kestrel" / "heat.py"
    py_compile.compile(str(sp), doraise=True)
    fl = load_flow(ff, reload=True)
    ch = fl.chapter("kestrel/thermal")
    assert ch.code == "K2" and ch.prerequisites == ("frame",) and ch.kind.script == "side_labs/kestrel/heat.py"
    scaffold.new_chapter(tmp_path, "kestrel/joint", kind="fea")
    assert load_flow(ff, reload=True).chapter("K3").name == "joint"
    with pytest.raises(FileExistsError):
        scaffold.new_chapter(tmp_path, "kestrel/joint", kind="fea")
