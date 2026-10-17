#!/usr/bin/env python3
"""vegeta workflows — the one entry point to Vegeta's long-running analyses.

Books are designs (``kestrel``, code K); chapters are the numbered studies on them — CFD and FEA alike — chosen by
code, case-insensitive: K1, k2, K10 …

    wf                              # interactive: the tree, pick a chapter by code, the options, confirm, run
    wf list                         # the tree: books → kinds → chapters, with prerequisites and run counts
    wf info K2 [--variant 4Q-2]     # one chapter: what it is, its prerequisites, the exact cases it would run
    wf K2 --variant 4Q-2 --quality fine --detach [--no-cache]      # run chapter K2 (same as: wf run K2 …)
    wf runs                         # every run on disk: chapter code, state, cases done, age
    wf status <run>                 # one run: heartbeat, last log lines, what to do next
    wf tail <run> | wf stop <run> | wf resume <run> | wf notebook <run>
    wf batch batch.yaml [--dry-run] [--from N] [--detach]       # an ordered batch of chapters from one file (wf batch --example)
    wf cache | wf clean-cache [--book kestrel] [-y]
    wf gui                          # the desktop interface (PyQt6): the same tree, Run / Stop, the live log and progress, the runs
    wf import_legacy                # the first CFD campaign's file into K1's folder (once)

(``wf`` = ``workflows/cli.py`` after ``source activate.sh``.) A run is a directory under workflows/runs/<book>/<chapter>/
(the contract: workflows/README.md); solved cases are cached under workflows/cache/<book>/ and reused by later runs of
the book. The scripts (workflows/kestrel/*.py) stay usable directly — this tool only lists, explains, launches and
watches them.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workflows"))
sys.path.insert(0, str(ROOT / "notebooks" / "designs"))
import runkit  # noqa: E402

import library  # noqa: E402

RUNS = library.RUNS
PY = sys.executable
BOOKS = library.BOOKS


# ----------------------------------------------------------------------------------------------- helpers
def _age(ts: str | None) -> str:
    if not ts:
        return ""
    try:
        t = dt.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return ts
    s = (dt.datetime.now(dt.timezone.utc) - t).total_seconds()
    return f"{s / 60:.0f} min ago" if s < 7200 else (f"{s / 3600:.0f} h ago" if s < 2 * 86400 else f"{s / 86400:.0f} d ago")


def _alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def _run_dir(arg: str) -> Path:
    p = Path(arg)
    if p.is_dir() and (p / "manifest.json").exists():
        return p
    for r in _all_runs():
        d = Path(r["run"])
        if d.name == arg or d.name.endswith(arg) or f"{r['book']}/{r['chapter']}/{d.name}" == arg:
            return d
    sys.exit(f"no run directory {arg!r} (see: workflows/cli.py runs)")


def _all_runs() -> list[dict]:
    return library.all_runs()


def _ask(prompt: str, choices: list[str] | None = None, default: str | None = None) -> str:
    while True:
        if choices:
            for i, c in enumerate(choices, 1):
                print(f"  {i}. {c}")
        txt = input(f"{prompt}" + (f" [{default}]" if default else "") + ": ").strip()
        if not txt and default is not None:
            return default
        if choices:
            if txt.isdigit() and 1 <= int(txt) <= len(choices):
                return choices[int(txt) - 1]
            if txt in choices:
                return txt
            print("  (pick a number or a name)")
            continue
        if txt:
            return txt


# ----------------------------------------------------------------------------------------------- commands
def cmd_list(args) -> int:
    print("Books are designs, chapters the numbered studies on them (choose by code: K1, k2 …); a chapter's prerequisites are\n"
          "chapters of the same book. The rule: an analysis over ~15–30 min is a workflow, never a notebook cell (workflows/README.md).\n")
    print(library.tree())
    for name, i in library.IMPORTS.items():
        print(f"\n{name:15s} {i['title']} — wf {name}")
    st = runkit.Cache.stats()
    print("\ncache (solved cases reused across runs of a book, workflows/cache/): " + (", ".join(f"{k}: {v['cases']} case(s), {v['kB']} kB" for k, v in st.items()) if st else "empty"))
    print("\nnext: wf info K2   ·   wf K2 --variant 4Q-2 --quality fine --detach   ·   wf runs")
    return 0


def _chapter(code: str):
    try:
        return library.by_code(code)
    except KeyError as exc:
        sys.exit(str(exc))


def cmd_info(args) -> int:
    if args.chapter is None and (args.book_or_code is None or args.book_or_code in BOOKS):
        print(library.tree(args.book_or_code))
        return 0
    return _info_code(args)


def _info_code(args) -> int:
    book, name, info = _chapter(args.chapter or args.book_or_code)
    kind = library.KINDS[info["kind"]]
    chs = library.chapters(book)
    print(f"{info['code']}  {name}  — book {book} ({BOOKS[book]['title']})\n  {kind['title']}\n  {info['description']}\n  needs: {kind['needs']} · time: {kind['time']}")
    if info["prerequisites"]:
        print("  prerequisites (same book): " + ", ".join(f"{chs[p]['code']} ({p})" for p in info["prerequisites"]))
    rs = library.runs_of(book, name)
    if rs:
        print(f"  runs: {len(rs)} ({sum(1 for r in rs if r['state'] == 'completed')} completed): " + ", ".join(f"{r['run_id']} [{r['variant']}, {r['state']}]" for r in rs[:4]))
    extra = []
    for k, v in vars(args).items():
        if k in ("variant", "quality") and v:
            extra += [f"--{k}", v]
    cmd = [PY, str(ROOT / info["script"]), "--chapter", info["campaign"], *extra, "--list"]
    print("\n$ " + shlex.join([".venv/bin/python", info["script"], "--chapter", info["campaign"], *extra, "--list"]) + "\n")
    return subprocess.call(cmd, cwd=ROOT)


def _launch(script: str, extra: list[str], *, detach: bool, label: str) -> int:
    cmd = [PY, str(ROOT / script), *extra]
    print("$ " + shlex.join([".venv/bin/python", script, *extra]))
    if not detach:
        return subprocess.call(cmd, cwd=ROOT)
    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / f"{label}_{runkit.stamp()}.out"
    with out.open("ab") as f:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    print(f"detached: pid {proc.pid}, terminal output in {out.relative_to(ROOT)}")
    time.sleep(2.0)
    newest = _all_runs()[:1]
    if newest and newest[0].get("created_at"):
        print(f"run directory: {newest[0]['run']}\n  watch:  workflows/cli.py tail {Path(newest[0]['run']).name}\n  status: workflows/cli.py status {Path(newest[0]['run']).name}")
    elif label.startswith("batch_"):
        print(f"the batch folder appears under {RUNS.relative_to(ROOT)}/_batches/ as it starts: its batch.json names every step's run directory and summary file")
    else:
        print("the run directory appears under workflows/runs/ once the cases are built (workflows/cli.py runs)")
    return 0


def cmd_run(args) -> int:
    if args.code in library.IMPORTS:
        return _launch(library.IMPORTS[args.code]["script"], args.rest, detach=args.detach, label=args.code)
    book, name, info = _chapter(args.code)
    ns = library.needs_status(book, name)
    if not ns["ok"] and not ({"--prepare-only", "--list", "-h", "--help"} & set(args.rest)):
        print(f"{info['code']} ({name}) needs {', '.join(ns['missing'])} on this machine — not found:", file=sys.stderr)
        for fx in dict.fromkeys(ns["fixes"]):
            print(f"  → {fx}", file=sys.stderr)
        print(f"(or prepare the cases without solving: wf {info['code']} --prepare-only …; yn doctor shows what this machine can run)", file=sys.stderr)
        return 2
    extra = ["--chapter", info["campaign"], *args.rest] + (["--no-cache"] if args.no_cache and "--no-cache" not in args.rest else [])
    return _launch(info["script"], extra, detach=args.detach, label=f"{info['code']}_{name}")


def cmd_resume(args) -> int:
    d = _run_dir(args.run)
    m = runkit.read_json(d / "manifest.json") or sys.exit(f"{d} has no manifest.json")
    kind = m.get("kind") or {"airframe_cfd": "cfd", "structure_fea": "fea"}.get(m.get("workflow")) or sys.exit(f"unknown kind in manifest of {d.name}")
    st = runkit.read_json(d / "status.json") or {}
    if st.get("state") == "running" and _alive(st.get("pid")):
        sys.exit(f"{d.name} is still running (pid {st.get('pid')}); stop it first or wait")
    return _launch(library.KINDS[kind]["script"], ["--resume", str(d), *args.rest], detach=args.detach, label=f"resume_{m.get('chapter', kind)}")


def cmd_runs(args) -> int:
    runs = _all_runs()
    if args.book:
        if args.book.lower() in {b["code"].lower() for b in BOOKS.values()} or args.book in BOOKS:
            runs = [r for r in runs if r["book"] == args.book or BOOKS[r["book"]]["code"].lower() == args.book.lower()]
        else:
            book, name, _ = _chapter(args.book)
            runs = [r for r in runs if r["book"] == book and r["chapter"] == name]
    if not runs:
        print("no runs under workflows/runs/ yet (workflows/cli.py list; workflows/cli.py run …; or import_legacy)")
        return 0
    print(f"{'code':5s} {'book / chapter':30s} {'run id':44s} {'variant':7s} {'state':10s} {'cases':14s} {'created':12s} summary")
    for r in runs:
        c = r.get("cases") or {}
        st = runkit.read_json(Path(r["run"]) / "status.json") or {}
        state = r["state"] or "?"
        if state == "running" and not _alive(st.get("pid")):
            state = "running? (pid gone)"
        m = runkit.read_json(Path(r["run"]) / "manifest.json") or {}
        var = (m.get("config") or {}).get("variant", "")
        print(f"{r.get('code', '?'):5s} {(r['book'] + ' / ' + r['chapter'])[:30]:30s} {r['run_id'][:44]:44s} {var:7s} {state:10s} {str(c.get('done', 0)) + '/' + str(c.get('planned', '?')) + (' (' + str(c.get('failed')) + ' failed)' if c.get('failed') else ''):14s} {_age(r.get('created_at')):12s} {'yes' if r['summary'] else '-'}")
    return 0


def cmd_status(args) -> int:
    d = _run_dir(args.run)
    m = runkit.read_json(d / "manifest.json") or {}
    st = runkit.read_json(d / "status.json") or {}
    chs = library.chapters(m.get("book")) if m.get("book") in BOOKS else {}
    code = chs.get(m.get("chapter"), {}).get("code", "?")
    print(f"run {d.name}\n  {code}  book {m.get('book')} · chapter {m.get('chapter', m.get('campaign'))} · config {m.get('config')}\n  created {m.get('created_at')} ({_age(m.get('created_at'))}) on {m.get('host')}\n  command {m.get('command')}")
    state = st.get("state")
    pid = st.get("pid")
    alive = _alive(pid)
    c = st.get("cases") or {}
    print(f"  state {state} · cases done {c.get('done', 0)} of {c.get('planned', '?')}" + (f", {c.get('failed')} failed" if c.get("failed") else "") + (f", {c.get('reused')} reused" if c.get("reused") else "")
          + f" · elapsed {st.get('elapsed_s', 0) / 60:.0f} min · pid {pid} {'alive' if alive else 'not running'}")
    hb = st.get("heartbeat")
    if hb:
        print(f"  heartbeat {_age(hb.get('at'))}: case {hb.get('case')!r}, {hb.get('case_elapsed_s', 0) / 60:.1f} min into it; "
              + (f"{hb.get('solver_log')} {hb.get('solver_log_bytes', 0) / 1024:.0f} kB, {'advancing' if hb.get('advancing') else 'quiet for ' + str(hb.get('quiet_for_s')) + ' s'}; {hb.get('last_progress')}" if hb.get("solver_log") else "no solver log yet"))
    if (d / "summary.json").exists():
        s = runkit.read_json(d / "summary.json")
        print(f"  summary.json: {s.get('counts')}")
    elif (d / "summary.partial.json").exists():
        print("  summary.partial.json only: not a completed result")
    log = d / "run.log"
    if log.exists():
        lines = log.read_text(errors="replace").splitlines()
        print("  last log lines:")
        for ln in lines[-args.lines:]:
            print("    " + ln[:200])
    # what to do next
    if state == "running" and alive:
        q = (hb or {}).get("quiet_for_s", 0) or 0
        print("  → healthy" if q < 900 else f"  → the solver log has been quiet for {q / 60:.0f} min: look at the native log before deciding it is stuck")
        print(f"  → follow: workflows/cli.py tail {d.name}")
    elif state == "running" and not alive:
        print(f"  → the process is gone but the state says running (killed? rebooted?): resume with  workflows/cli.py resume {d.name}")
    elif state in ("cancelled", "partial", "prepared"):
        print(f"  → continue with  workflows/cli.py resume {d.name}")
    elif state == "failed":
        print("  → read run.log above and the native logs under cases/; fix; a new run or --resume")
    elif state == "completed":
        print(f"  → use it in the notebook:  workflows/cli.py notebook {d.name}")
    return 0


def cmd_tail(args) -> int:
    d = _run_dir(args.run)
    os.execvp("tail", ["tail", "-n", str(args.lines), "-F", str(d / "run.log")])


def cmd_stop(args) -> int:
    d = _run_dir(args.run)
    st = runkit.read_json(d / "status.json") or {}
    pid = st.get("pid")
    if st.get("state") != "running" or not _alive(pid):
        print(f"{d.name}: no running process (state {st.get('state')})")
        return 1
    os.kill(int(pid), signal.SIGINT)
    print(f"SIGINT sent to pid {pid}: the running solver is stopped by its tool, the run ends 'cancelled' with the finished cases kept; "
          f"then  workflows/cli.py resume {d.name}")
    return 0


def cmd_notebook(args) -> int:
    d = _run_dir(args.run)
    for line in library.pin_commands(d):
        print(line)
    print("# the older RUNS form (notebooks not yet on Yanus pins):")
    m = runkit.read_json(d / "manifest.json") or {}
    cfg = m.get("config") or {}
    rel = d.relative_to(ROOT) if ROOT in d.parents else d
    w = m.get("workflow")
    if w == "airframe_cfd":
        ch = m.get("chapter") or cfg.get("chapter")
        camp = (BOOKS.get(m.get("book"), {}).get("chapters", {}).get(ch) or {}).get("campaign") or cfg.get("campaign", "sweep")
        if camp == "legacy-import":
            camp = "sweep"
        print(f'RUNS["airframe_cfd"]["{cfg.get("variant")}"]["{camp}"] = "{rel}"')
    elif w == "structure_fea":
        print(f'RUNS["structure_fea"]["{cfg.get("variant")}"] = "{rel}"')
    else:
        print(f"# {w}: not read by the notebook")
    st = (runkit.read_json(d / "status.json") or {}).get("state")
    if st != "completed":
        print(f"# note: the run's state is {st!r} — the notebook accepts completed runs only")
    return 0


def cmd_batch(args) -> int:
    import batch as _batch
    if args.example or not args.file:
        print(_batch.EXAMPLE, end="")
        if not args.file and not args.example:
            print("\n(wf batch <file.yaml> runs such a file; wf batch --example > batch.yaml writes this one)", file=sys.stderr)
        return 0
    f = Path(args.file)
    if not f.exists():
        sys.exit(f"no such file: {f}  (wf batch --example prints a template)")
    try:
        b = _batch.load(f)
    except ValueError as exc:
        sys.exit(f"{f}: {exc}")
    if args.detach and not args.dry_run:
        return _launch("workflows/cli.py", ["batch", str(f.resolve()), *(["--from", str(args.start)] if args.start > 1 else [])], detach=True, label=f"batch_{runkit.slug(b.name)}")
    rep = _batch.run(b, source=f, dry_run=args.dry_run, start=args.start)
    return rep["exit_code"]


def cmd_gui(args) -> int:
    try:
        import PyQt6.QtWidgets  # noqa: F401
    except Exception as exc:
        sys.exit(f"the desktop interface needs PyQt6: pip install PyQt6 (or: sudo apt install python3-pyqt6 libegl1 libxkbcommon-x11-0 libxcb-cursor0) — {exc}")
    from gui.app import main as gui_main
    return gui_main([])


def cmd_cache(args) -> int:
    st = runkit.Cache.stats()
    if not st:
        print("the cache is empty (workflows/cache/): a case is stored when a run of its book solves it, and read back by any later run that asks for the same case")
        return 0
    for k, v in st.items():
        print(f"{k:20s} {v['cases']:5d} case(s)  {v['kB']:8.1f} kB   workflows/cache/{k}/")
    return 0


def cmd_clean_cache(args) -> int:
    st = runkit.Cache.stats()
    n = sum(v["cases"] for k, v in st.items() if args.book is None or k == args.book)
    if not n:
        print("nothing to clean")
        return 0
    if not args.yes and _ask(f"remove {n} cached case(s)" + (f" of {args.book}" if args.book else "") + "? the native results stay in their runs", ["yes", "no"], default="no") != "yes":
        return 0
    print(f"removed {runkit.Cache.clean(book=args.book)} cached case(s)")
    return 0


def interactive() -> int:
    print(__doc__.split("\n\n")[0])
    print()
    print(library.tree())
    print("\nalso:  runs — the runs on disk   ·   cache — the cached cases   ·   import_legacy — the first campaign's file into K1")
    pick = _ask("chapter code (K1, k2 …) or runs / cache / import_legacy", default="runs")
    low = pick.strip().lower()
    if low == "cache":
        cmd_cache(None)
        if _ask("clean it?", ["no", "yes"], default="no") == "yes":
            return cmd_clean_cache(argparse.Namespace(book=None, yes=True))
        return 0
    if low == "import_legacy":
        return _launch(library.IMPORTS["import_legacy"]["script"], [], detach=False, label="import_legacy")
    if low == "runs":
        cmd_runs(argparse.Namespace(book=None))
        r = _ask("a run id to inspect (empty to quit)", default="")
        if r:
            cmd_status(argparse.Namespace(run=r, lines=12))
            act = _ask("action", ["nothing", "tail", "resume", "stop", "notebook line"], default="nothing")
            if act == "tail":
                cmd_tail(argparse.Namespace(run=r, lines=30))
            elif act == "resume":
                return cmd_resume(argparse.Namespace(run=r, rest=[], detach=_ask("detach (nohup-style)?", ["yes", "no"], default="yes") == "yes"))
            elif act == "stop":
                return cmd_stop(argparse.Namespace(run=r))
            elif act == "notebook line":
                return cmd_notebook(argparse.Namespace(run=r))
        return 0
    book, name, info = _chapter(pick)
    kind = library.KINDS[info["kind"]]
    chs = library.chapters(book)
    print(f"\n{info['code']}  {name}: {info['description']}\n  {kind['title']}\n  needs: {kind['needs']} · {kind['time']}")
    if info["prerequisites"]:
        print("  prerequisites: " + ", ".join(f"{chs[p]['code']} ({p})" for p in info["prerequisites"]))
    extra = ["--chapter", info["campaign"]]
    for flag, choices, default in kind["options"]:
        v = _ask(flag.lstrip("-"), list(choices) if choices else None, default=default)
        extra += [flag, v]
    mode = _ask("mode", ["run", "prepare only (no solver)", "list the cases only"], default="run")
    if mode.startswith("prepare"):
        extra.append("--prepare-only")
    if mode.startswith("list"):
        extra.append("--list")
        return _launch(info["script"], extra, detach=False, label=info["code"])
    lim = _ask("limit (e.g. '--max-cases 2 --max-minutes 90 --cases \"α 12°\"', empty for none)", default="")
    if lim:
        extra += shlex.split(lim)
    if _ask("use the cache (cases this book already solved are read back)?", ["yes", "no"], default="yes") == "no":
        extra.append("--no-cache")
    detach = _ask("run detached (survives logout; watch with wf tail)?", ["yes", "no"], default="yes") == "yes"
    print("\nabout to run:\n  " + shlex.join([".venv/bin/python", info["script"], *extra]))
    if _ask("confirm", ["yes", "no"], default="yes") != "yes":
        print("nothing run")
        return 0
    return _launch(info["script"], extra, detach=detach, label=f"{info['code']}_{name}")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    import re as _re
    if argv and (_re.fullmatch(r"[A-Za-z]\d+", argv[0]) or argv[0] in library.IMPORTS):     # wf K2 … = wf run K2 …
        argv = ["run", *argv]
    elif argv and argv[0].endswith((".yaml", ".yml")):                                     # wf batch.yaml = wf batch batch.yaml
        argv = ["batch", *argv]
    ap = argparse.ArgumentParser(prog="wf", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("list", help="the tree of books and chapters")
    p = sub.add_parser("info", help="a chapter by code (K2): what it is and the cases it would run; a book name: its tree"); p.add_argument("book_or_code", nargs="?"); p.add_argument("chapter", nargs="?"); p.add_argument("--variant"); p.add_argument("--quality")
    p = sub.add_parser("run", help="run a chapter by code (the rest of the arguments go to its script: --variant, --quality, …)"); p.add_argument("code"); p.add_argument("--detach", action="store_true"); p.add_argument("--no-cache", action="store_true"); p.add_argument("rest", nargs=argparse.REMAINDER)
    p = sub.add_parser("resume", help="continue a run"); p.add_argument("run"); p.add_argument("--detach", action="store_true"); p.add_argument("rest", nargs=argparse.REMAINDER)
    p = sub.add_parser("runs", help="the runs on disk (optionally of one book or chapter code)"); p.add_argument("book", nargs="?")
    p = sub.add_parser("batch", help="run the chapters of a batch.yaml in order (the file: wf batch --example; the GUI writes one)"); p.add_argument("file", nargs="?")
    p.add_argument("--dry-run", action="store_true", help="print the commands only"); p.add_argument("--from", dest="start", type=int, default=1, metavar="N", help="start at step N")
    p.add_argument("--detach", action="store_true", help="run in the background (terminal output under workflows/runs/)"); p.add_argument("--example", action="store_true", help="print a template")
    p.add_argument("--no-progress", action="store_true", help=argparse.SUPPRESS)        # the GUI adds it to every launch; the steps have no TTY anyway
    sub.add_parser("gui", help="the desktop interface (PyQt6): the tree, options, Run / Stop, the live log, the runs")
    sub.add_parser("cache", help="the cached cases per book")
    p = sub.add_parser("clean-cache", help="remove the cached cases (the runs keep their native results)"); p.add_argument("--book"); p.add_argument("-y", "--yes", action="store_true")
    p = sub.add_parser("status", help="one run: state, heartbeat, last log lines, what next"); p.add_argument("run"); p.add_argument("-n", "--lines", type=int, default=8)
    p = sub.add_parser("tail", help="follow a run's log"); p.add_argument("run"); p.add_argument("-n", "--lines", type=int, default=30)
    p = sub.add_parser("stop", help="stop a running run (SIGINT)"); p.add_argument("run")
    p = sub.add_parser("notebook", help="the RUNS line for the notebook"); p.add_argument("run")
    args = ap.parse_args(argv)
    if args.cmd is None:
        return interactive()
    if args.cmd in ("run", "resume"):
        args.rest = list(getattr(args, "rest", []) or [])
        if args.rest[:1] == ["--"]:
            args.rest = args.rest[1:]
        for flag in ("--detach", "--no-cache"):            # may come after the code
            if flag in args.rest:
                args.rest = [a for a in args.rest if a != flag]
                setattr(args, flag.strip("-").replace("-", "_"), True)
    return {"list": cmd_list, "info": cmd_info, "run": cmd_run, "resume": cmd_resume, "runs": cmd_runs, "status": cmd_status, "tail": cmd_tail, "stop": cmd_stop,
            "notebook": cmd_notebook, "cache": cmd_cache, "clean-cache": cmd_clean_cache, "gui": cmd_gui, "batch": cmd_batch}[args.cmd](args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:                                # wf runs | head
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        sys.exit(0)
