"""yn / vegeta yanus — find and check saved workflow runs, the machine's needs, the registry. It only reads (``new``
writes a scaffold); the commands it prints produce runs when an engineer types them.

    yn find <chapter> [--variant V] [--quality Q] [--run DIR]   the run a notebook would read, or why none (exit 2)
    yn doctor [--json]                                           the needs of every chapter against this machine
    yn validate                                                  the registry (workflows/flow.py), for CI
    yn new chapter <book>/<name> --kind K [--new-kind] [--after CH]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .needs import check_need
from .registry import current, load_flow
from .select import find


def cmd_find(a, flow) -> int:
    sel = find(a.chapter, variant=a.variant, quality=a.quality, run=a.run, flow=flow)
    if a.json:
        print(json.dumps(sel.provenance(), indent=1))
    else:
        print(sel.text())
        if sel.ok and sel.skipped:
            print("  newer runs passed over:\n" + "\n".join(f"    {rid}: {why}" for rid, why in sel.skipped))
        if sel.compatibility == "unverified":
            print("  (compatibility with the current design is checked where a notebook passes its fingerprint)")
    return 0 if sel.ok else 2


def cmd_doctor(a, flow) -> int:
    seen = {}
    for ch in flow.chapters():
        for n in ch.all_needs():
            r = check_need(n)
            seen.setdefault(f"{type(n).__name__.lower()} {n.label}", {"ok": r.ok, "version": r.version, "detail": r.detail, "fix": r.fix,
                                                                     "chapters": []})["chapters"].append(ch.code)
    can = {ch.code: all(check_need(n).ok for n in ch.all_needs()) for ch in flow.chapters()}
    if a.json:
        print(json.dumps({"needs": seen, "can_run": can}, indent=1))
        return 0
    for key, v in seen.items():
        print(f"  {'✔' if v['ok'] else '✘'} {key:26s} {(v['version'] or v['detail'])[:40]:42s} for {', '.join(dict.fromkeys(v['chapters']))}")
        if not v["ok"]:
            print(f"      → {v['fix']}")
    print(f"\nthis machine can run: {', '.join(k for k, ok in can.items() if ok) or 'nothing'}")
    no = [k for k, ok in can.items() if not ok]
    if no:
        print(f"not here (a missing need; a chapter may still --prepare-only, or run under the mocks with VEGETA_DEBUG=1): {', '.join(no)}")
    return 0


def cmd_validate(a, flow) -> int:
    try:
        flow.validate()
    except Exception as exc:
        print(f"error: {exc}")
        return 1
    print(f"ok: {len(flow.books)} book(s), {len(flow.chapters())} chapter(s)")
    return 0


def cmd_new(a, flow) -> int:
    from . import scaffold
    if not a.kind:
        raise SystemExit("yn new chapter <book>/<name> --kind <kind>")
    out = scaffold.new_chapter(flow.root, a.name, kind=a.kind, campaign=a.campaign, new_kind=a.new_kind, title=a.title or "",
                               takes=tuple(a.after or ()))
    for p in out:
        print(f"wrote {Path(p).resolve().relative_to(Path(flow.root).resolve())}")
    print("next: yn validate")
    return 0


def parser(prog: str = "yn") -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=prog, description="Yanus: find and check the saved runs of Vegeta's workflows",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("--flow", default=None, help="the registry file (default $VEGETA_YANUS_FLOW, else workflows/flow.py)")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("find", help="the run a notebook would read, or why none")
    p.add_argument("chapter"); p.add_argument("--variant"); p.add_argument("--quality"); p.add_argument("--run", help="check this run directory only")
    p.add_argument("--json", action="store_true", help="the provenance record a report would carry")
    p = sub.add_parser("doctor", help="the needs of every chapter on this machine"); p.add_argument("--json", action="store_true")
    sub.add_parser("validate", help="the registry (CI)")
    p = sub.add_parser("new", help="scaffold a chapter"); p.add_argument("what", choices=("chapter",)); p.add_argument("name")
    p.add_argument("--title"); p.add_argument("--kind"); p.add_argument("--campaign"); p.add_argument("--new-kind", action="store_true")
    p.add_argument("--after", action="append", help="a prerequisite chapter of the same book (repeatable)")
    return ap


def main(argv=None, prog: str = "yn") -> int:
    ap = parser(prog)
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    if not a.cmd:
        ap.print_help()
        return 0
    flow = load_flow(a.flow) if a.flow else current()
    try:
        return {"find": cmd_find, "doctor": cmd_doctor, "validate": cmd_validate, "new": cmd_new}[a.cmd](a, flow)
    except (KeyError, FileExistsError, ValueError) as exc:
        print(f"yn: {exc.args[0] if exc.args else exc}", file=sys.stderr)
        return 2
    except BrokenPipeError:
        return 0
