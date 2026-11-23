"""Work the loop: study → design → sim → campaign → export, with exp/ as limbo — one lab at a time.

    python tools/lab.py list                              # the labs and their machines' stages
    python tools/lab.py test aero [-- pytest args]        # the lab's tests, mocks on (VEGETA_DEBUG=1), not the slow ones
    python tools/lab.py run aero notebooks [NAME...]        # its notebooks headless, mocks on (nbconvert, outputs to a temp dir)
    python tools/lab.py run aero sims [NAME...]           # its sim scripts, mocks on (Chiron episodes cut to seconds)
    python tools/lab.py new-exp aero falcon_x             # a limbo study: exp/falcon_x.py + exp/falcon_x.ipynb with the bootstrap
    python tools/lab.py demote aero mini_raptor           # a machine's modules and notebooks to exp/ (git mv)
    python tools/lab.py promote aero kestrel_xwing        # back from exp/ to designs/ (or notebooks/ for a notebook)
    python tools/lab.py promote aero gait --to-libs       # a module two labs need → libs/labkit (vegeta.labkit.<name>)
    python tools/lab.py export drongo --to out/drongo.yx2 --lump    # to Piccolo, as standard files (tools/export.py)
    python tools/lab.py check                             # the lab rules (tools/labs.py)
    python tools/lab.py readme                            # regenerate every lab's README.md (folders, machines' stages)

Never runs a real solver: every run sets VEGETA_DEBUG=1 and VEGETA_NO_SOLVERS=1 (vegeta.mock).
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import labs  # noqa: E402

MOCK_ENV = {"VEGETA_DEBUG": "1", "VEGETA_NO_SOLVERS": "1", "MPLBACKEND": "Agg", "PYVISTA_OFF_SCREEN": "true"}


def _env():
    env = dict(os.environ, **MOCK_ENV)
    env.pop("ANTHROPIC_API_KEY", None)
    return env


def _lab(name: str) -> Path:
    d = ROOT / "labs" / name
    if not d.is_dir():
        sys.exit(f"no lab {name!r}: {', '.join(sorted(p.name for p in (ROOT / 'labs').iterdir() if p.is_dir()))}")
    return d


def _git_mv(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "mv", str(src), str(dst)], cwd=ROOT, check=True)
    print(f"  {src.relative_to(ROOT)} → {dst.relative_to(ROOT)}")


def cmd_list(_):
    for lab, ms in labs.stages().items():
        print(f"[{lab}]")
        for m, s in ms.items():
            print(f"  {m:<18} {' → '.join(s)}")
    return 0


def cmd_test(lab, extra):
    d = _lab(lab)
    return subprocess.call([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "vegeta.mock.pytest_plugin",
                            "--import-mode=importlib", "-m", "not slow", str(d), *extra], cwd=ROOT, env=_env())


def cmd_run(lab, stage, names, timeout=900):
    d = _lab(lab)
    folder = d / stage
    if stage == "sims":
        files = sorted(p for p in folder.glob("*.py") if not p.name.startswith("_"))
    else:
        files = sorted(folder.rglob("*.ipynb"))                     # with the notebook-only subfolders (aero: races/)
    if names:
        files = [f for f in files if any(f.stem.startswith(n) for n in names)]
    status, out_dir = 0, Path(tempfile.mkdtemp(prefix=f"vegeta-{lab}-{stage}-"))
    for f in files:
        if f.suffix == ".py":
            cmd = [sys.executable, str(f)]
        else:
            cmd = [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", str(f), "--output-dir",
                   str(out_dir), f"--ExecutePreprocessor.timeout={timeout}"]
        try:
            rc = subprocess.run(cmd, cwd=f.parent, env=_env(), capture_output=True, text=True, timeout=timeout * 2)
            ok, tail = rc.returncode == 0, (rc.stdout + rc.stderr).strip().splitlines()[-1:] or [""]
        except subprocess.TimeoutExpired:
            ok, tail = False, ["timed out"]
        print(f"  {'ok  ' if ok else 'FAIL'} {f.relative_to(ROOT)}{'' if ok else '  ' + tail[0][:150]}")
        status |= not ok
    print(f"  executed copies and logs: {out_dir}")
    return status


BOOT = '''DEBUG = True                                   # the expensive solvers mocked (vegeta.mock); False for real results
from vegeta.labkit import bootstrap
DEBUG = bootstrap.lab("{lab}", DEBUG)
'''


def cmd_new_exp(lab, name):
    d = _lab(lab) / "exp"
    d.mkdir(exist_ok=True)
    mod, nb = d / f"{name}.py", d / f"{name}.ipynb"
    if mod.exists() or nb.exists():
        sys.exit(f"{name} exists in {d.relative_to(ROOT)}")
    mod.write_text(f'"""{name} — a speculative design in the {lab} lab\'s limbo (exp/). Promote it with '
                   f'``tools/lab.py promote {lab} {name}`` once something imports it."""\nfrom __future__ import annotations\n')
    import json
    cells = [{"cell_type": "markdown", "metadata": {}, "source": [f"# {name}\n", "\n", f"A speculative design ({lab} lab, exp/)."]},
             {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
              "source": (BOOT.format(lab=lab) + f"import {name}\n").splitlines(keepends=True)}]
    nb.write_text(json.dumps({"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
                              "nbformat": 4, "nbformat_minor": 5}, indent=1) + "\n")
    print(f"  {mod.relative_to(ROOT)}\n  {nb.relative_to(ROOT)}")
    return 0


def _members(folder: Path, name: str):
    return sorted(p for p in folder.glob(f"*{name}*") if p.suffix in (".py", ".ipynb"))


def cmd_demote(lab, name):
    d = _lab(lab)
    moved = 0
    for stage in ("designs", "notebooks"):
        for f in _members(d / stage, name):
            _git_mv(f, d / "exp" / f.name)
            moved += 1
    print(f"  {moved} file(s) to limbo; tools/labs.py check says whether anything still imports them")
    return labs.main(["check"])


def cmd_promote(lab, name, to_libs=False):
    d = _lab(lab)
    if to_libs:
        src = d / "designs" / f"{name}.py"
        if not src.exists():
            sys.exit(f"no {src.relative_to(ROOT)}")
        dst = ROOT / "libs" / "labkit" / "src" / "vegeta" / "labkit" / f"{name}.py"
        _git_mv(src, dst)
        src.write_text(f'"""Moved to ``vegeta.labkit.{name}`` (a second lab needs it); re-exported here."""\n'
                       f"from vegeta.labkit.{name} import *  # noqa: F401,F403\n")
        subprocess.run(["git", "add", str(src)], cwd=ROOT, check=True)
        print(f"  import it as vegeta.labkit.{name}; {src.relative_to(ROOT)} re-exports it for this lab")
        return labs.main(["check"])
    for f in _members(d / "exp", name):
        _git_mv(f, d / ("notebooks" if f.suffix == ".ipynb" else "designs") / f.name)
    return labs.main(["check"])


STAGE_COLS = ("notebooks", "designs", "sims", "campaigns", "export", "exp")


def cmd_readme(_=None):
    """Write each lab's README.md: what it is, its folders, its machines' stages (from where their files are)."""
    import tomllib
    st = labs.stages()
    for d in sorted(p for p in (ROOT / "labs").iterdir() if p.is_dir()):
        meta = tomllib.loads((d / "lab.toml").read_text()) if (d / "lab.toml").exists() else {"title": d.name}
        rows = ["| machine | " + " | ".join(STAGE_COLS) + " |", "|---|" + "---|" * len(STAGE_COLS)]
        for m, stages in st.get(d.name, {}).items():
            rows.append(f"| {m} | " + " | ".join("●" if c in stages else "" for c in STAGE_COLS) + " |")
        counts = {s: len([f for f in (d / s).rglob("*") if f.suffix in (".py", ".ipynb") and "__pycache__" not in f.parts])
                  for s in ("notebooks", "designs", "sims", "campaigns", "exp", "common", "tests")}
        text = f"""# {meta.get("title", d.name)}

Generated by `python tools/lab.py readme` from where the files are — run it again after moving things.

| folder | files | what |
|---|---:|---|
| `notebooks/` | {counts["notebooks"]} | the notebooks: one per machine or part (stage 1); notebook-only folders inside it (aero: `races/`) |
| `designs/` | {counts["designs"]} | the machine modules, imported by plain name (`import nisus_flight`), and `machines.py` (what this lab exports) |
| `sims/` | {counts["sims"]} | MuJoCo scenarios and their runners: the designs that graduated to simulation |
| `campaigns/` | {counts["campaigns"]} | data gathering and optimisation loops |
| `exp/` | {counts["exp"]} | limbo: speculative designs and their notebooks — nothing imports them (`tools/lab.py promote` when something should) |
| `common/` | {counts["common"]} | this lab's snippets and tools |
| `tests/` | {counts["tests"]} | `python tools/lab.py test {d.name}` (mocks) |

A notebook starts with::

    DEBUG = True
    from vegeta.labkit import bootstrap
    DEBUG = bootstrap.lab("{d.name}", DEBUG)

## Machines and their stage

""" + ("\n".join(rows) if len(rows) > 2 else "(none yet)") + """

`export` = registered in `designs/machines.py`, sent to Piccolo with `python tools/lab.py export <machine> --to <file>.yx2`.
The rules (`tools/labs.py check`): this lab imports `libs/` and itself, never another lab.
"""
        (d / "README.md").write_text(text)
        print(f"  {(d / 'README.md').relative_to(ROOT)}")
    return 0


def main(argv=None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    if not a or a[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    cmd, rest = a[0], a[1:]
    if cmd == "list":
        return cmd_list(rest)
    if cmd == "check":
        return labs.main(["check"])
    if cmd == "readme":
        return cmd_readme()
    if cmd == "test":
        extra = rest[rest.index("--") + 1:] if "--" in rest else []
        return cmd_test(rest[0], extra)
    if cmd == "run":
        return cmd_run(rest[0], rest[1], rest[2:])
    if cmd == "new-exp":
        return cmd_new_exp(rest[0], rest[1])
    if cmd == "demote":
        return cmd_demote(rest[0], rest[1])
    if cmd == "promote":
        return cmd_promote(rest[0], rest[1], to_libs="--to-libs" in rest)
    if cmd == "export":
        return subprocess.call([sys.executable, str(ROOT / "tools" / "export.py"), *rest], cwd=ROOT, env=_env())
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
