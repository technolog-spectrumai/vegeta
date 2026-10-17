"""yn / vegeta yanus — the door keeper's command line. It reads, checks, pins, stamps and scaffolds; only ``yn auto`` runs a
unit (the commands it prints do that, when an engineer types them).

    yn status [unit|book] [--deep] [--json]   every chapter and notebook: ready / blocked / not run, and its needs
    yn why <unit> [--variant V] [--deep]      one unit's gate: every need and take, its status and the command that fixes it
    yn doctor [--json]                        the needs of every unit against this machine
    yn pin <notebook> <chapter> --variant V <run dir>     the run a notebook reads (explicit; never picked by itself)
    yn pin <notebook> --suggest               the usable runs of each run take (printed, not written)
    yn unpin <notebook> <chapter> --variant V
    yn check [--strict]                       the registry, the pins, the stamps (for CI)
    yn stamp <unit> <file>                    stamp a file a unit gives that was made outside its gate
    yn new notebook <NN_name> [--title …]     a notebook with the common header and both gates, and its registry entry
    yn new chapter <book>/<name> --kind K [--new-kind] [--after CH]
    yn auto <notebook> [--dry-run]            the yanus_auto plan: the chapters (screening quality) and notebooks it would run for its missing takes; runs them without --dry-run
"""
from __future__ import annotations

import argparse
import os
import json
import sys
from pathlib import Path

from . import pins as _pins
from .contract import Chapter, Data, File, Flow, Notebook, Run
from .gate import candidates, enter, runs_root
from .needs import check_need
from .registry import current
from .runs import runs_of
from .stamps import read_stamp, sha256_file, stamp_path, write_stamp

MARK = {"completed": "✔", "running": "▶", "partial": "◐", "cancelled": "◐", "prepared": "○", "failed": "✘"}


def _needs_line(unit) -> tuple[bool, str]:
    res = [check_need(n) for n in unit.all_needs()]
    if not res:
        return True, ""
    bad = [r.need.label for r in res if not r.ok]
    return not bad, ("needs ✔" if not bad else "needs ✘ " + ", ".join(bad) + " missing")


def _chapter_state(flow: Flow, ch: Chapter) -> dict:
    latest = {}
    for rv in runs_of(runs_root(flow), ch.book.name, ch.name):
        latest.setdefault(rv.variant or "?", rv)
    return {v: rv.state for v, rv in sorted(latest.items())}


def status_data(flow: Flow, only: str | None = None, *, deep: bool = False) -> dict:
    out = {"books": {}, "notebooks": {}}
    for b in flow.books.values():
        if only and only not in (b.name, b.code) and not only.upper().startswith(b.code.upper()):
            continue
        rows = {}
        for ch in b.chapters.values():
            if only and only not in (b.name, b.code) and only.upper() != ch.code.upper():
                continue
            ok, needs = _needs_line(ch)
            rows[ch.code] = {"chapter": ch.name, "states": _chapter_state(flow, ch), "after": [b.chapters[p].code for p in ch.prerequisites],
                             "needs_ok": ok, "needs": needs}
        if rows:
            out["books"][b.name] = {"code": b.code, "title": b.title, "chapters": rows}
    for nb in flow.notebooks.values():
        if only and only not in (nb.name, nb.name.split("_", 1)[0]) and only != "notebooks":
            continue
        g = enter(nb, flow=flow, deep=deep, raise_closed=False, auto=False)        # status looks, it never drives
        bad = g.closed
        nr = [r for r in g.rows if r.kind != "need" and not r.usable and r.role == "optional"]
        state = "blocked" if bad else ("ready, some optional inputs NOT RUN" if nr else "ready")
        out["notebooks"][nb.name] = {"state": state, "blocked_by": [f"{r.input}: {r.status}" for r in bad], "fix": [r.fix for r in bad if r.fix],
                                     "not_run": [r.input for r in nr], "gives": [f.path for f in nb.gives], "screening": g.screening, "auto": nb.auto}
    return out


def print_status(d: dict) -> None:
    for bname, b in d["books"].items():
        print(f"{bname}  ({b['code']})  {b['title']}")
        for code, c in b["chapters"].items():
            st = " ".join(f"{MARK.get(s, '?')} {v}" for v, s in c["states"].items()) or "· not run"
            after = f"[after {', '.join(c['after'])}]" if c["after"] else ""
            print(f"  {code:4s} {c['chapter']:22s} {st:28s} {after:14s} {c['needs']}")
    if d["notebooks"]:
        print("notebooks")
        for name, n in d["notebooks"].items():
            mark = "✘" if n["state"] == "blocked" else "✔"
            print(f"  {mark} {name:24s} {n['state']}" + (" (screening)" if n["screening"] else "") +
                  (" — yanus_auto: it runs what is missing by itself, in screening quality" if n.get("auto") and n["state"] == "blocked" else (" (auto)" if n.get("auto") else "")))
            for why, fix in zip(n["blocked_by"], n["fix"] + [""] * len(n["blocked_by"])):
                print(f"      {why}" + (f"\n        → {fix}" if fix else ""))
            if n["not_run"]:
                print(f"      NOT RUN (optional): {', '.join(n['not_run'])}")


def cmd_status(a, flow) -> int:
    d = status_data(flow, a.unit, deep=a.deep)
    if a.json:
        print(json.dumps(d, indent=1))
    else:
        print_status(d)
        if not a.deep:
            print("(compatibility with the current design not checked: --deep builds the fingerprints)")
    return 2 if any(n["state"] == "blocked" for n in d["notebooks"].values()) else 0


def cmd_why(a, flow) -> int:
    g = enter(a.unit, flow=flow, variant=a.variant, deep=a.deep, raise_closed=False)
    print(g.text())
    if not g.ok:
        print("\n" + g.message())
    return 0 if g.ok else 2


def cmd_doctor(a, flow) -> int:
    seen, rows = {}, []
    for u in flow.units():
        for n in u.all_needs():
            r = check_need(n)
            key = f"{type(n).__name__.lower()} {n.label}"
            seen.setdefault(key, {"ok": r.ok, "version": r.version, "detail": r.detail, "fix": r.fix, "units": []})["units"].append(
                u.code if isinstance(u, Chapter) else u.name)
    can = {u.code if isinstance(u, Chapter) else u.name: _needs_line(u)[0] for u in flow.units()}
    if a.json:
        print(json.dumps({"needs": seen, "can_run": can}, indent=1))
        return 0
    for key, v in seen.items():
        print(f"  {'✔' if v['ok'] else '✘'} {key:26s} {(v['version'] or v['detail'])[:40]:42s} for {', '.join(dict.fromkeys(v['units']))}")
        if not v["ok"]:
            print(f"      → {v['fix']}")
    yes = [k for k, ok in can.items() if ok]
    no = [k for k, ok in can.items() if not ok]
    print(f"\nthis machine can run: {', '.join(yes) or 'nothing'}")
    if no:
        print(f"not here (a missing need; a chapter may still --prepare-only): {', '.join(no)}")
    return 0


def _run_key(flow, nb: Notebook, chapter: str, variant: str | None) -> str:
    ch = flow.chapter(chapter)
    key = ch.ref + (f"@{variant}" if variant else "")
    keys = [t.key() for t in nb.takes if isinstance(t, Run)]
    if key not in keys:
        raise SystemExit(f"{nb.name} takes no {key}: its run takes are {', '.join(keys) or 'none'} (workflows/flow.py)")
    return key


def cmd_pin(a, flow) -> int:
    if len(a.rest) > 2:
        raise SystemExit("yn pin <notebook> <chapter> --variant V <run dir>")
    a.chapter, a.path = (a.rest + [None, None])[:2]                 # --variant may come anywhere (argparse fills the rest after it)
    nb = flow.unit(a.unit)
    if not isinstance(nb, Notebook):
        raise SystemExit("pins are a notebook's choice of runs; a chapter takes the newest usable run of its prerequisites")
    if a.suggest or not a.chapter:
        pins = _pins.read_pins(flow.root, nb.name)
        for t in nb.takes:
            if not isinstance(t, Run):
                continue
            ch = flow.run_take_chapter(t, nb)
            key = t.key()
            cand = candidates(flow, ch, t.variant, states=t.states, deep=a.deep)
            print(f"{key}  (pinned: {pins.get(key) or '—'})")
            for rv in cand[:5]:
                rel = str(Path(rv.root).resolve().relative_to(Path(flow.root).resolve())) if str(rv.root).startswith(str(flow.root)) else str(rv.root)
                print(f"    yn pin {nb.name} {ch.code}" + (f" --variant {t.variant}" if t.variant else "") + f" {rel}    # {rv.provenance()}")
            if not cand:
                print("    no usable run on disk")
        return 0
    if not a.path:
        raise SystemExit("yn pin <notebook> <chapter> --variant V <run dir>")
    key = _run_key(flow, nb, a.chapter, a.variant)
    p = Path(a.path)
    p = p if p.is_absolute() else (Path.cwd() / p if (Path.cwd() / p).exists() else Path(flow.root) / p)
    if not p.exists():
        print(f"warning: {a.path} does not exist on this machine (pinned anyway: the gate reports it NOT RUN)", file=sys.stderr)
    f = _pins.set_pin(flow.root, nb.name, key, p)
    print(f"{key} → {_pins.read_pins(flow.root, nb.name)[key]}   ({f.relative_to(flow.root)})")
    return 0


def cmd_unpin(a, flow) -> int:
    nb = flow.unit(a.unit)
    key = _run_key(flow, nb, a.chapter, a.variant)
    print(f"{key}: {'unpinned' if _pins.clear_pin(flow.root, nb.name, key) else 'was not pinned'}")
    return 0


def cmd_check(a, flow) -> int:
    errors, notes = [], []
    try:
        flow.validate()
    except Exception as exc:
        errors.append(str(exc))
    root = Path(flow.root)
    pdir = root / _pins.DIR
    for f in sorted(pdir.glob("*.json")) if pdir.is_dir() else []:
        name = f.stem
        if name not in flow.notebooks:
            errors.append(f"{f.relative_to(root)}: no notebook {name!r} in the flow")
            continue
        keys = {t.key() for t in flow.notebooks[name].takes if isinstance(t, Run)}
        for k, v in _pins.read_pins(root, name).items():
            if k not in keys:
                errors.append(f"{f.relative_to(root)}: {k} is not a run take of {name}")
            elif v and not _pins.resolve(root, v).exists():
                notes.append(f"{name}: {k} → {v} is not on this machine (NOT RUN here)")
    for u in flow.units():
        for g in u.gives:
            fp = root / g.path
            st = read_stamp(fp)
            if fp.exists() and st is not None and sha256_file(fp) != st.get("sha256"):
                (errors if a.strict else notes).append(f"{g.path}: modified since {st.get('producer')} stamped it ({st.get('written_at')})")
            elif fp.exists() and st is None:
                notes.append(f"{g.path}: not stamped yet (run {u.name} through its gate)")
    for u in flow.units():
        for t in u.takes:
            if isinstance(t, Data) and not (root / t.path).exists():
                if t.cache:
                    notes.append(f"{u.name}: cache {t.path} not built yet (computed on first use)")
                elif t.required:
                    errors.append(f"{u.name}: data {t.path} is missing" + (f" — {t.note}" if t.note else ""))
                else:
                    notes.append(f"{u.name}: optional data {t.path} is missing")
    for n in notes:
        print("note: " + n)
    for e in errors:
        print("error: " + e)
    print(f"{'ok' if not errors else 'FAILED'}: {len(flow.units())} units, {len(errors)} error(s), {len(notes)} note(s)")
    return 1 if errors else 0


def cmd_stamp(a, flow) -> int:
    u = flow.unit(a.unit)
    root = Path(flow.root)
    p = Path(a.file).resolve()
    rel = str(p.relative_to(root.resolve()))
    g = next((g for g in u.gives if isinstance(g, File) and g.path == rel), None)
    if g is None:
        raise SystemExit(f"{u.ref} gives no {rel} (its gives: {', '.join(x.path for x in u.gives) or 'none'})")
    st = write_stamp(p, producer=u.name, kind=u.kind_of_unit, schema=g.schema, root=root, inputs=[{"kind": "manual", "note": "yn stamp"}])
    print(f"stamped {rel} ({st['sha256'][:12]}) → {stamp_path(p).relative_to(root.resolve())}")
    return 0


def cmd_auto(a, flow) -> int:
    """What auto mode would run for a notebook (``--dry-run``: the plan only), or run it now, without the notebook."""
    from . import auto
    u = flow.unit(a.unit)
    first = enter(u, flow=flow, variant=a.variant, deep=a.deep, raise_closed=False, auto=False)
    steps = auto.plan(flow, first)
    if not steps:
        print(f"{u.kind_of_unit} {u.ref}: nothing to produce — every take is usable")
        return 0
    print(f"{u.kind_of_unit} {u.ref}: {sum(s.state == 'planned' for s in steps)} step(s) to run in screening quality"
          f"{' (YANUS_AUTO_MINUTES caps each)' if not os.environ.get('YANUS_AUTO_MINUTES') else ''}:")
    for s in steps:
        print(f"  {'·' if s.state == 'skipped' else '▶'} {s.label:40s} {s.because}" + (f"\n      skipped: {s.reason}" if s.reason else "") + f"\n      $ {' '.join(s.argv)}")
    if a.dry_run:
        print("(--dry-run: nothing started; without it the steps run now, one after the other, in the foreground)")
        return 0
    steps = auto.drive(flow, first)
    print(auto.report(steps))
    return 0 if all(s.state != "failed" for s in steps) else 1


def cmd_new(a, flow) -> int:
    from . import scaffold
    if a.what == "notebook":
        out = scaffold.new_notebook(flow.root, a.name, title=a.title or "", auto=a.auto)
    else:
        out = scaffold.new_chapter(flow.root, a.name, kind=a.kind, campaign=a.campaign, new_kind=a.new_kind, title=a.title or "",
                                   takes=tuple(a.after or ()))
    for p in out:
        print(f"wrote {Path(p).resolve().relative_to(Path(flow.root).resolve())}")
    print("next: fill in the contract in workflows/flow.py, then yn check")
    return 0


def parser(prog: str = "yn") -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=prog, description="Yanus, the door keeper: the contracts between notebooks and workflows",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("--flow", default=None, help="the registry file (default $VEGETA_YANUS_FLOW, else workflows/flow.py)")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("status", help="every unit: ready / blocked / not run"); p.add_argument("unit", nargs="?")
    p.add_argument("--deep", action="store_true", help="also check each pinned run's compatibility with the design (builds the fingerprints)")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("why", help="one unit's gate with every fix"); p.add_argument("unit"); p.add_argument("--variant"); p.add_argument("--deep", action="store_true")
    p = sub.add_parser("doctor", help="the needs of every unit on this machine"); p.add_argument("--json", action="store_true")
    p = sub.add_parser("pin", help="the run a notebook reads"); p.add_argument("unit"); p.add_argument("rest", nargs="*", metavar="CHAPTER RUN_DIR")
    p.add_argument("--variant"); p.add_argument("--suggest", action="store_true"); p.add_argument("--deep", action="store_true")
    p = sub.add_parser("unpin", help="forget a pin"); p.add_argument("unit"); p.add_argument("chapter"); p.add_argument("--variant")
    p = sub.add_parser("check", help="registry, pins and stamps (CI)"); p.add_argument("--strict", action="store_true", help="a modified stamped file is an error")
    p = sub.add_parser("stamp", help="stamp a given file made outside the gate"); p.add_argument("unit"); p.add_argument("file")
    p = sub.add_parser("auto", help="what auto mode would produce for a notebook (its yanus_auto plan), or produce it now"); p.add_argument("unit")
    p.add_argument("--variant"); p.add_argument("--dry-run", action="store_true", help="the plan only"); p.add_argument("--deep", action="store_true")
    p = sub.add_parser("new", help="scaffold a notebook or a chapter"); p.add_argument("what", choices=("notebook", "chapter")); p.add_argument("name")
    p.add_argument("--title"); p.add_argument("--kind"); p.add_argument("--campaign"); p.add_argument("--new-kind", action="store_true")
    p.add_argument("--auto", action="store_true", help="notebook: the yanus_auto flag (it runs what it is missing by itself, in screening quality)")
    p.add_argument("--after", action="append", help="a prerequisite chapter of the same book (repeatable)")
    return ap


def main(argv=None, prog: str = "yn") -> int:
    ap = parser(prog)
    a, extra = ap.parse_known_args(sys.argv[1:] if argv is None else argv)
    if extra and a.cmd == "pin" and not any(x.startswith("-") for x in extra):
        a.rest += extra                                                 # yn pin 43 K1 --variant V <run>: the run dir after the option
    elif extra:
        ap.error("unrecognized arguments: " + " ".join(extra))
    if not a.cmd:
        parser(prog).print_help()
        return 0
    from .registry import load_flow
    flow = load_flow(a.flow) if a.flow else current()
    if a.cmd == "new" and a.what == "chapter" and not a.kind:
        raise SystemExit("yn new chapter <book>/<name> --kind <kind>")
    try:
        return {"status": cmd_status, "why": cmd_why, "doctor": cmd_doctor, "pin": cmd_pin, "unpin": cmd_unpin, "check": cmd_check,
                "stamp": cmd_stamp, "new": cmd_new, "auto": cmd_auto}[a.cmd](a, flow)
    except (KeyError, FileExistsError, ValueError) as exc:
        print(f"yn: {exc.args[0] if exc.args else exc}", file=sys.stderr)
        return 2
    except BrokenPipeError:
        return 0
