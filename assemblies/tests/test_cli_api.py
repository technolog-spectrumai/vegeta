"""The command line of every workflow is API-compatible with its ``run()``: each option the parser has is sent through
``_cli.main`` to a mock of the real ``run`` (``create_autospec``: a keyword ``run()`` does not take fails), so nothing
is built or solved. Also: ``--show`` on a saved tree, and the defaults alone."""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from unittest import mock

import pytest

from assemblies import _cli
from assemblies.vida import Assembly

WORKFLOWS = ("microjet", "aguya", "quadcopter", "fixed_wing", "propulsors", "merlin", "boat", "submarine", "rover", "onager",
             "walkers")


def every_option(ap: argparse.ArgumentParser, tmp: Path) -> list[str]:
    """argv with every option of ``ap`` set (a value of its type; not --show / --help)."""
    argv = []
    for a in ap._actions:
        if not a.option_strings or a.dest in ("help", "show"):
            continue
        opt = max(a.option_strings, key=len)
        if a.nargs == 0:                                         # store_true / store_false
            argv.append(opt)
        elif a.choices:
            argv += [opt, str(list(a.choices)[0])]
        elif a.type is Path or "vida" in a.dest or a.dest in ("out",):
            argv += [opt, str(tmp / f"{a.dest}.x")]
        elif a.type is int:
            argv += [opt, "2"]
        elif a.type is float:
            argv += [opt, "1.5"]
        elif a.metavar and "=" in str(a.metavar):
            argv += [opt, "span=1"]
        else:
            argv += [opt, "something"]
    return argv


def _entry(mod):
    return getattr(mod, "_run_cli", None) or mod.run


@pytest.mark.parametrize("name", WORKFLOWS)
def test_every_cli_option_is_a_run_keyword(name, tmp_path, monkeypatch, capsys):
    mod = importlib.import_module(f"assemblies.workflows.{name}")
    fake = mock.create_autospec(mod.run, return_value=Assembly(name, "stub"))
    monkeypatch.setattr(mod, "run", fake)
    ap = mod._parser()
    argv = every_option(ap, tmp_path)
    assert _cli.main(_entry(mod), ap, tmp_path / "default.vida", argv=argv) == 0
    assert fake.call_count == 1
    kw = fake.call_args.kwargs
    assert kw["vida_path"] == tmp_path / "vida_path.x"
    for flag, key in (("--no-cfd", "run_cfd"), ("--no-fea", "run_fea"), ("--no-sim", "run_sim"), ("--no-export", "export")):
        if flag in argv:
            assert kw[key] is False, f"{flag} did not reach run() as {key}=False"


@pytest.mark.parametrize("name", WORKFLOWS)
def test_defaults_alone(name, tmp_path, monkeypatch, capsys):
    mod = importlib.import_module(f"assemblies.workflows.{name}")
    fake = mock.create_autospec(mod.run, return_value=Assembly(name, "stub"))
    monkeypatch.setattr(mod, "run", fake)
    assert _cli.main(_entry(mod), mod._parser(), tmp_path / "default.vida", argv=[]) == 0
    assert fake.call_args.kwargs.get("fidelity", "full") == "full"


def test_show_prints_a_saved_tree(tmp_path, capsys):
    from assemblies.workflows import walkers
    path = Assembly("walkers", "walker_family").save(tmp_path / "w.vida")
    assert _cli.main(walkers._run_cli, walkers._parser(), tmp_path / "missing.vida", argv=["--show", "--vida", str(path)]) == 0
    assert "walkers (walker_family)" in capsys.readouterr().out
    assert _cli.main(walkers._run_cli, walkers._parser(), tmp_path / "missing.vida", argv=["--show"]) == 2
