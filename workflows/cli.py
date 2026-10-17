#!/usr/bin/env python3
"""vegeta workflows — the one entry point to Vegeta's long-running analyses.

    .venv/bin/python workflows/cli.py                      # interactive: pick a workflow, a campaign, the options; confirm; run
    .venv/bin/python workflows/cli.py list                 # the books (parts) and their chapters (scenarios), prerequisites, runs, cache
    .venv/bin/python workflows/cli.py info kestrel_airframe stall --variant 4Q-2    # the cases a chapter would run
    .venv/bin/python workflows/cli.py run kestrel_airframe --variant 4Q-2 --chapter stall --quality fine --detach [--no-cache]
    .venv/bin/python workflows/cli.py cache                # the cached cases per book;  clean-cache [--book B] removes them
    .venv/bin/python workflows/cli.py runs                 # every run on disk: state, counts, age
    .venv/bin/python workflows/cli.py status <run>         # one run: state, heartbeat, the last log lines, what to do next
    .venv/bin/python workflows/cli.py tail <run>           # follow its run.log
    .venv/bin/python workflows/cli.py stop <run>           # SIGINT to its process: the solver stops, finished cases are kept
    .venv/bin/python workflows/cli.py resume <run>         # continue it
    .venv/bin/python workflows/cli.py notebook <run>       # the RUNS line to paste into 43_kestrel_study

A run is a directory under workflows/runs/<book>/<chapter>/ (the contract: workflows/README.md); solved cases are
cached under workflows/cache/<book>/ and reused by later runs of the book. The scripts themselves
(workflows/kestrel/*.py) stay usable directly — this tool only lists, explains, launches and watches them.
`source activate.sh` once per shell makes `wf` an alias of this tool inside the virtual environment.
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
    print("Books and chapters — a book is the part or assembly simulated, a chapter the scenario solved on it; a chapter's\n"
          "prerequisites are chapters of the same book (the rule: an analysis over ~15–30 min is a workflow — workflows/README.md)\n")
    for name, b in BOOKS.items():
        print(f"{name}\n  {b['title']}\n  part: {b['part']}\n  needs {b['needs']} · {b['time']} · variants {', '.join(b['variants'])} · script {b['script']}")
        try:
            chs = library.chapters(name)
        except Exception as exc:
            chs = {"?": {"description": f"chapter list unavailable ({exc})", "prerequisites": ()}}
        for c, d in chs.items():
            pre = f"  [after: {', '.join(d['prerequisites'])}]" if d["prerequisites"] else ""
            print(f"    {c:13s} {d['description']}{pre}")
        runs = library.runs_of(name)
        done = sum(1 for r in runs if r["state"] == "completed")
        print(f"    runs: {len(runs)} ({done} completed) under workflows/runs/{name}/\n")
    for name, i in library.IMPORTS.items():
        print(f"{name:15s} {i['title']} — {i['script']}")
    st = runkit.Cache.stats()
    print("\ncache (solved cases reused across runs of a book, workflows/cache/): " + (", ".join(f"{k}: {v['cases']} case(s), {v['kB']} kB" for k, v in st.items()) if st else "empty"))
    return 0


def _book(name: str) -> dict:
    if name in BOOKS:
        return BOOKS[name]
    alias = {"airframe_cfd": "kestrel_airframe", "structure_fea": "kestrel_frame"}
    if name in alias:
        return BOOKS[alias[name]]
    sys.exit(f"unknown book {name!r}; one of {list(BOOKS)}")


def cmd_info(args) -> int:
    w = _book(args.book)
    print(f"{args.book}: {w['title']}\npart: {w['part']}\nneeds: {w['needs']}\ntime: {w['time']}\n")
    if args.chapter:
        extra = []
        for k, v in vars(args).items():
            if k in ("variant", "quality") and v:
                extra += [f"--{k}", v]
        pre = w["chapters"].get(args.chapter, {}).get("prerequisites", ())
        if pre:
            print(f"prerequisites (same book): {', '.join(pre)}")
        cmd = [PY, str(ROOT / w["script"]), "--chapter", args.chapter, *extra, "--list"]
        print("$ " + shlex.join(cmd[1:]) + "\n")
        return subprocess.call(cmd, cwd=ROOT)
    for c, d in library.chapters(args.book if args.book in BOOKS else {"airframe_cfd": "kestrel_airframe", "structure_fea": "kestrel_frame"}[args.book]).items():
        print(f"  {c:13s} {d['description']}" + (f"  [after: {', '.join(d['prerequisites'])}]" if d["prerequisites"] else ""))
    return 0


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
    else:
        print("the run directory appears under workflows/runs/ once the cases are built (workflows/cli.py runs)")
    return 0


def cmd_run(args) -> int:
    if args.book in library.IMPORTS:
        return _launch(library.IMPORTS[args.book]["script"], args.rest, detach=args.detach, label=args.book)
    w = _book(args.book)
    extra = list(args.rest) + (["--no-cache"] if args.no_cache and "--no-cache" not in args.rest else [])
    return _launch(w["script"], extra, detach=args.detach, label=args.book)


def cmd_resume(args) -> int:
    d = _run_dir(args.run)
    m = runkit.read_json(d / "manifest.json") or sys.exit(f"{d} has no manifest.json")
    w = BOOKS.get(m.get("book") or {"airframe_cfd": "kestrel_airframe", "structure_fea": "kestrel_frame"}.get(m.get("workflow"))) or sys.exit(f"unknown book in manifest: {m.get('book')}")
    st = runkit.read_json(d / "status.json") or {}
    if st.get("state") == "running" and _alive(st.get("pid")):
        sys.exit(f"{d.name} is still running (pid {st.get('pid')}); stop it first or wait")
    return _launch(w["script"], ["--resume", str(d), *args.rest], detach=args.detach, label=f"resume_{m.get('workflow')}")


def cmd_runs(args) -> int:
    runs = _all_runs()
    if args.book:
        runs = [r for r in runs if r["book"] == args.book or r["workflow"] == args.book]
    if not runs:
        print("no runs under workflows/runs/ yet (workflows/cli.py list; workflows/cli.py run …; or import_legacy)")
        return 0
    print(f"{'book / chapter':30s} {'run id':44s} {'variant':7s} {'state':10s} {'cases':14s} {'created':12s} summary")
    for r in runs:
        c = r.get("cases") or {}
        st = runkit.read_json(Path(r["run"]) / "status.json") or {}
        state = r["state"] or "?"
        if state == "running" and not _alive(st.get("pid")):
            state = "running? (pid gone)"
        m = runkit.read_json(Path(r["run"]) / "manifest.json") or {}
        var = (m.get("config") or {}).get("variant", "")
        print(f"{(r['book'] + ' / ' + r['chapter'])[:30]:30s} {r['run_id'][:44]:44s} {var:7s} {state:10s} {str(c.get('done', 0)) + '/' + str(c.get('planned', '?')) + (' (' + str(c.get('failed')) + ' failed)' if c.get('failed') else ''):14s} {_age(r.get('created_at')):12s} {'yes' if r['summary'] else '-'}")
    return 0


def cmd_status(args) -> int:
    d = _run_dir(args.run)
    m = runkit.read_json(d / "manifest.json") or {}
    st = runkit.read_json(d / "status.json") or {}
    print(f"run {d.name}\n  book {m.get('book')} · chapter {m.get('chapter', m.get('campaign'))} · config {m.get('config')}\n  created {m.get('created_at')} ({_age(m.get('created_at'))}) on {m.get('host')}\n  command {m.get('command')}")
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
    m = runkit.read_json(d / "manifest.json") or {}
    cfg = m.get("config") or {}
    rel = d.relative_to(ROOT) if ROOT in d.parents else d
    w = m.get("workflow")
    if w == "airframe_cfd":
        camp = cfg.get("chapter") or cfg.get("campaign", "sweep")
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
    print("\nBooks (the part simulated) → chapters (the scenario):")
    names = list(BOOKS)
    for i, n in enumerate(names, 1):
        print(f"  {i}. {n:17s} {BOOKS[n]['title']}  [{BOOKS[n]['time']}]")
    print(f"  {len(names) + 1}. runs              see the runs on disk and their state\n  {len(names) + 2}. cache             the cached cases; clean them")
    pick = _ask("choose", [*names, "runs", "cache"], default=names[0])
    if pick == "cache":
        cmd_cache(None)
        if _ask("clean it?", ["no", "yes"], default="no") == "yes":
            return cmd_clean_cache(argparse.Namespace(book=None, yes=True))
        return 0
    if pick == "runs":
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
    w = BOOKS[pick]
    print(f"\n{pick}: {w['title']}\npart: {w['part']}\nneeds: {w['needs']}\n")
    chs = library.chapters(pick)
    extra = []
    print("chapters:")
    for c, d in chs.items():
        print(f"  {c:13s} {d['description']}" + (f"  [after: {', '.join(d['prerequisites'])}]" if d["prerequisites"] else ""))
    camp = _ask("chapter", list(chs), default=list(chs)[0])
    extra += ["--chapter", camp]
    for flag, choices, default in w["options"]:
        v = _ask(flag.lstrip("-"), choices, default=default)
        extra += [flag, v]
    mode = _ask("mode", ["run", "prepare only (no solver)", "list the cases only"], default="run")
    if mode.startswith("prepare"):
        extra.append("--prepare-only")
    if mode.startswith("list"):
        extra.append("--list")
        return _launch(w["script"], extra, detach=False, label=pick)
    lim = _ask("limit (e.g. '--max-cases 2 --max-minutes 90 --cases \"α 12°\"', empty for none)", default="")
    if lim:
        extra += shlex.split(lim)
    if _ask("use the cache (cases this book already solved are read back)?", ["yes", "no"], default="yes") == "no":
        extra.append("--no-cache")
    detach = _ask("run detached (survives logout; watch with tail)?", ["yes", "no"], default="yes") == "yes"
    print("\nabout to run:\n  " + shlex.join([".venv/bin/python", w["script"], *extra]))
    if _ask("confirm", ["yes", "no"], default="yes") != "yes":
        print("nothing run")
        return 0
    return _launch(w["script"], extra, detach=detach, label=pick)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="workflows/cli.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("list", help="the workflows and their campaigns")
    p = sub.add_parser("info", help="a book, or a chapter's cases"); p.add_argument("book"); p.add_argument("chapter", nargs="?"); p.add_argument("--variant"); p.add_argument("--quality")
    p = sub.add_parser("run", help="launch a book's script (the rest of the arguments go to it: --chapter, --variant, …)"); p.add_argument("book"); p.add_argument("--detach", action="store_true"); p.add_argument("--no-cache", action="store_true"); p.add_argument("rest", nargs=argparse.REMAINDER)
    p = sub.add_parser("resume", help="continue a run"); p.add_argument("run"); p.add_argument("--detach", action="store_true"); p.add_argument("rest", nargs=argparse.REMAINDER)
    p = sub.add_parser("runs", help="the runs on disk"); p.add_argument("--book")
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
        for flag in ("--detach", "--no-cache"):            # may come after the book name
            if flag in args.rest:
                args.rest = [a for a in args.rest if a != flag]
                setattr(args, flag.strip("-").replace("-", "_"), True)
    return {"list": cmd_list, "info": cmd_info, "run": cmd_run, "resume": cmd_resume, "runs": cmd_runs, "status": cmd_status, "tail": cmd_tail, "stop": cmd_stop, "notebook": cmd_notebook, "cache": cmd_cache, "clean-cache": cmd_clean_cache}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
