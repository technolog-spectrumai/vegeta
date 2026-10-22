"""``vegeta ws ...`` and ``vegeta rev ...`` (registered as vegeta command plug-ins)."""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from pathlib import Path

from .workspace import Workspace


def _load_attr(spec: str):
    """``file.py:name`` or ``package.module:name``."""
    target, sep, attr = spec.partition(":")
    if not sep:
        raise ValueError(f"{spec!r}: expected file.py:name or module:name")
    if target.endswith(".py"):
        path = Path(target)
        if not path.is_file():
            raise FileNotFoundError(f"no such file: {target}")
        name = f"_vegeta_core_user_{abs(hash(str(path.resolve())))}"
        s = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(s)
        sys.modules[name] = module
        s.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    if not hasattr(module, attr):
        raise ValueError(f"{target} has no attribute {attr!r}")
    return getattr(module, attr)


def _overrides(design, items):
    dd = design.load()
    out = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"-p expects name=value, got {item!r}")
        k, v = item.split("=", 1)
        if k not in dd.params:
            raise ValueError(f"unknown parameter {k!r}; known: {[p.name for p in dd.params]}")
        out[k] = dd.params[k].parse(v)
    return out


def _emit_eval(ev, as_json: bool) -> int:
    if as_json:
        print(json.dumps(ev.record or {"kind": ev.kind, "name": ev.name, "status": ev.status,
                                        "messages": ev.messages}, indent=2, default=str))
    else:
        print(ev)
    return 0 if ev.ok else 1


# -- vegeta ws --------------------------------------------------------------------------------
def ws_main(argv=None, prog: str = "vegeta ws") -> int:
    ap = argparse.ArgumentParser(prog=prog, description="Vegeta workspaces")
    ap.add_argument("-w", "--workspace", default=".", help="workspace directory (default: .)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init", help="create a workspace")
    p.add_argument("--name")
    p.add_argument("--description", default="")
    p = sub.add_parser("add-design", help="register a Dedalus design")
    p.add_argument("name")
    p.add_argument("source", help="module:Name or file.py:Name")
    p.add_argument("--description", default="")
    sub.add_parser("designs", help="list designs")
    p = sub.add_parser("status", help="all revisions vs all analyses (missing ones: NOT RUN)")
    p.add_argument("--design")
    p.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "init":
            ws = Workspace.create(args.workspace, name=args.name, description=args.description)
            print(f"created workspace {ws.name} at {ws.root}")
            return 0
        ws = Workspace.open(args.workspace)
        if args.cmd == "add-design":
            d = ws.add_design(args.name, args.source, args.description)
            print(f"added design {d.name} ({d.source}) with parameters "
                  f"{[p['name'] for p in d.record['parameters']]}")
        elif args.cmd == "designs":
            for d in ws.designs():
                print(f"{d.name:<20} {d.source}")
        elif args.cmd == "status":
            table = ws.status(args.design)
            print(json.dumps(table.rows, indent=2) if args.json else table)
        return 0
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as exc:
        print(f"{prog}: error: {exc}", file=sys.stderr)
        return 2


# -- vegeta rev -------------------------------------------------------------------------------
def rev_main(argv=None, prog: str = "vegeta rev") -> int:
    ap = argparse.ArgumentParser(prog=prog, description="Vegeta revisions (each command is one explicit step)")
    ap.add_argument("-w", "--workspace", default=".", help="workspace directory (default: .)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("new", help="new revision of a design (nothing is generated)")
    p.add_argument("design")
    p.add_argument("-p", "--param", action="append", metavar="NAME=VALUE")
    p.add_argument("--note", default="")
    p = sub.add_parser("branch", help="new revision from an existing one with changed parameters")
    p.add_argument("rid")
    p.add_argument("-p", "--param", action="append", metavar="NAME=VALUE")
    p.add_argument("--note", default="")
    p = sub.add_parser("generate", help="generate the revision's geometry (STEP + STL)")
    p.add_argument("rid")
    p.add_argument("--stl-tolerance", type=float, default=0.01)
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("fea", help="run a Talos analysis: --model file.py:fn where fn(rev) -> StructuralModel")
    p.add_argument("rid")
    p.add_argument("--model", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("cfd", help="run an Aeromant case: --case file.py:fn where fn(rev, workdir) -> CFDCase")
    p.add_argument("rid")
    p.add_argument("--case", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--steps", help="comma separated subset of the template pipeline")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("print", help="slice with Mellonia: --settings file.py:NAME or module:NAME")
    p.add_argument("rid")
    p.add_argument("--settings", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--rotate-x", type=float, default=0.0)
    p.add_argument("--rotate-y", type=float, default=0.0)
    p.add_argument("--rotate-z", type=float, default=0.0)
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("label", help="set a human label (history is kept)")
    p.add_argument("rid")
    p.add_argument("label", choices=["preferred", "rejected", "reference", "unclassified"])
    p.add_argument("--note", default="")
    p = sub.add_parser("note", help="add a note (history is kept)")
    p.add_argument("rid")
    p.add_argument("text")
    p = sub.add_parser("show", help="revision record, annotations and evaluations")
    p.add_argument("rid")
    p.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    try:
        ws = Workspace.open(args.workspace)
        if args.cmd == "new":
            d = ws.design(args.design)
            r = d.new_revision(note=args.note, **_overrides(d, args.param))
            print(f"created {r.id} of {d.name}: {r.changes or 'defaults'}")
            return 0
        rev = ws.revision(args.rid)
        if args.cmd == "branch":
            r = rev.branch(note=args.note, **_overrides(rev.design, args.param))
            print(f"created {r.id} from {rev.id}: {r.changes or 'no changes'}")
            return 0
        if args.cmd == "generate":
            res = rev.generate(stl_tolerance=args.stl_tolerance)
            print(res.to_json() if args.json else res)
            return 0 if res.ok else 1
        if args.cmd == "fea":
            return _emit_eval(rev.run_fea(args.name, _load_attr(args.model), threads=args.threads), args.json)
        if args.cmd == "cfd":
            steps = [s.strip() for s in args.steps.split(",")] if args.steps else None
            return _emit_eval(rev.run_cfd(args.name, _load_attr(args.case), steps=steps), args.json)
        if args.cmd == "print":
            from vegeta.mellonia import Orientation

            orient = Orientation(args.rotate_x, args.rotate_y, args.rotate_z)
            return _emit_eval(rev.run_print(args.name, _load_attr(args.settings), orient), args.json)
        if args.cmd == "label":
            rev.label(args.label, args.note)
            print(f"{rev.id} labelled {args.label}")
            return 0
        if args.cmd == "note":
            rev.note(args.text)
            return 0
        if args.cmd == "show":
            data = {"revision": rev.record, "label": rev.current_label, "annotations": rev.annotations(),
                    "generated": rev.is_generated,
                    "evaluations": [{"kind": e.kind, "name": e.name, "status": e.status} for e in rev.evaluations()]}
            if args.json:
                print(json.dumps(data, indent=2, default=str))
            else:
                print(repr(rev))
                print(f"  parameters: {rev.params}")
                print(f"  generated:  {rev.is_generated}")
                for e in data["evaluations"]:
                    print(f"  {e['kind']}:{e['name']}  {e['status']}")
                for a in data["annotations"]:
                    print(f"  [{a['at']}] {a['type']}: {a.get('label', '')} {a.get('note', '')}".rstrip())
            return 0
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as exc:
        print(f"{prog}: error: {exc}", file=sys.stderr)
        return 2
    return 2
