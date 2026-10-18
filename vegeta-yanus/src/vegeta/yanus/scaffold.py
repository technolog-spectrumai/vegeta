"""scaffold — a new chapter starts from the shape the others share: a registry entry (and, for a new kind, a runkit
script skeleton). Never overwrites a file."""
from __future__ import annotations

from pathlib import Path

CHAPTER_SCRIPT = '''#!/usr/bin/env python
"""{book}/{chapter} ({kind}) — {title}

    .venv/bin/python {script} --variant <variant> --chapter {campaign}
    .venv/bin/python {script} --resume workflows/runs/{book}/<chapter>/<run id>

A thin workflow script (workflows/README.md): explicit inputs, the cases under a budget, one run directory. Its
contract (needs, the prerequisite chapters) is in workflows/flow.py; the entry gate checks it before a case starts.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "workflows", ROOT / "notebooks" / "designs", ROOT / "vegeta-yanus" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import runkit  # noqa: E402
import library  # noqa: E402
from vegeta import yanus  # noqa: E402

BOOK, KIND, WORKFLOW = "{book}", "{kind}", "{workflow}"


def cases(variant: str, campaign: str) -> dict:
    """``{{case name: case spec}}`` — the explicit list of cases of the campaign (never widened by the script)."""
    raise NotImplementedError("list the cases of the campaign here")


def solve(name: str, spec: dict, case_dir: Path, cancel) -> dict:
    """One case: prepare in ``case_dir``, run the tool, return ``{{"status": "solved"|"diverged"|"FAILED", …}}``."""
    raise NotImplementedError("drive the Vegeta tool here")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variant", required=True)
    camps = [c["campaign"] for c in library.BOOKS[BOOK]["chapters"].values() if c["kind"] == KIND]
    ap.add_argument("--chapter", "--campaign", dest="chapter", default=camps[0], choices=tuple(camps))
    ap.add_argument("--prepare-only", action="store_true")
    runkit.add_common_arguments(ap)
    args = ap.parse_args(argv)
    chapter = library.chapter_of(BOOK, KIND, args.chapter)
    gate = yanus.check_chapter(f"{{BOOK}}/{{chapter}}", variant=args.variant, ignore=args.ignore_prerequisites)
    specs = cases(args.variant, args.chapter)
    config = {{"variant": args.variant, "campaign": args.chapter}}
    if args.resume:
        run = runkit.Run.open(args.resume, config=config, heartbeat_s=args.heartbeat, echo=not args.quiet)
    else:
        out = Path(args.out) if args.out else library.run_dir(BOOK, chapter, args.variant)
        run = runkit.Run.create(out, workflow=WORKFLOW, campaign=args.chapter, config=config, planned_cases=list(specs),
                                heartbeat_s=args.heartbeat, echo=not args.quiet)
    run.manifest.update(book=BOOK, chapter=chapter, kind=KIND, prerequisites_ignored=gate["missing"] if args.ignore_prerequisites else [],
                        yanus={{"needs": gate["needs"]}})
    runkit.write_json_atomic(run.root / "manifest.json", run.manifest)
    records = {{}}
    for name, spec in specs.items():
        if run.cancel.is_set():
            break
        records[name] = run.run_case(name, runkit.sha256_json({{"case": name, **spec}}),
                                     lambda d, cancel, n=name, s=spec: solve(n, s, d, cancel), rerun_failed=args.rerun_failed)
    state = "cancelled" if run.cancel.is_set() else "completed"
    code = run.finish({{"cases": {{n: r.to_dict() for n, r in records.items()}}}}, state=state)
    print(f"run directory: {{run.root}}")
    run.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
'''


def _append(flow_file: Path, text: str) -> None:
    src = flow_file.read_text()
    flow_file.write_text(src.rstrip("\n") + "\n" + text)


def new_chapter(root, ref: str, *, kind: str, campaign: str | None = None, flow_file=None, new_kind: bool = False, title: str = "",
                takes: tuple = ()) -> list[Path]:
    """``ref`` = ``book/chapter``. With an existing ``kind`` only the registry entry is added (the kind's script takes
    the new ``--chapter``); with ``new_kind`` the kind and a script skeleton under ``workflows/<book>/`` too."""
    root = Path(root)
    if "/" not in ref:
        raise ValueError("a chapter is book/chapter, e.g. kestrel/thermal")
    book, chapter = ref.split("/", 1)
    campaign = campaign or chapter
    flow_file = Path(flow_file or root / "workflows" / "flow.py")
    src = flow_file.read_text()
    if f'.chapter("{chapter}"' in src:
        raise FileExistsError(f"chapter {chapter!r} is already declared in {flow_file}")
    out = [flow_file]
    lines = ["", f"# yn new chapter {ref} --kind {kind}"]
    if new_kind:
        script = f"workflows/{book}/{kind}.py"
        sp = root / script
        if sp.exists():
            raise FileExistsError(f"{sp} exists — Yanus never overwrites")
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(CHAPTER_SCRIPT.format(book=book, chapter=chapter, kind=kind, title=title or chapter, script=script, campaign=campaign,
                                            workflow=f"{book}_{kind}"))
        sp.chmod(0o755)
        out.insert(0, sp)
        lines.append(f'flow.books["{book}"].kind("{kind}", title={(title or kind)!r}, script="{script}", workflow="{book}_{kind}", needs=[])')
    tk = ", ".join(f'Run("{t}")' for t in takes)
    lines.append(f'flow.books["{book}"].chapter("{chapter}", kind="{kind}", campaign="{campaign}"' + (f", takes=[{tk}]" if tk else "") + ")")
    _append(flow_file, "\n".join(lines) + "\n")
    return out
