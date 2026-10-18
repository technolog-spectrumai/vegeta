"""chapters — the entry check of a workflow script: the chapter's needs on this machine (reported) and a usable run of
each prerequisite chapter (``find``, the same check a notebook gets). It reads; it never starts a prerequisite."""
from __future__ import annotations

import sys

from .contract import Flow
from .needs import check_need
from .registry import current
from .select import find


def check_chapter(chapter, *, variant: str | None, expected: dict | None = None, ignore: bool = False, flow: Flow | None = None,
                  exit_code: int = 2, out=None) -> dict:
    """Exits ``exit_code`` when a prerequisite has no usable run and not ``ignore``; returns ``{"missing": [...],
    "needs": {...}}`` for the manifest. A missing need is only reported (``--prepare-only`` still works)."""
    flow = current(flow)
    ch = flow.chapter(chapter)
    out = out or sys.stdout
    needs = {}
    for n in ch.all_needs():
        r = check_need(n)
        key = f"{type(n).__name__.lower()} {n.label}"
        needs[key] = (r.version or "found") if r.ok else "missing"
        if not r.ok:
            print(f"note: {key} not found on this machine ({r.fix})", file=out)
    missing = []
    for t in ch.takes:
        sel = find(flow.prerequisite(t, ch), variant=variant, fingerprint=expected, flow=flow)
        if not sel.ok:
            missing.append(f"{sel.chapter.code} ({sel.chapter.name}) has no usable run for {variant or 'any variant'}: {'; '.join(sel.reasons)}:  {sel.command}")
    if missing and not ignore:
        print(f"chapter {ch.code} ({ch.name}) of book {ch.book.name!r} needs a completed compatible run of its prerequisite chapter(s) first:", file=out)
        for m in missing:
            print("  - " + m, file=out)
        print("(or --ignore-prerequisites, recorded in the manifest)", file=out)
        sys.exit(exit_code)
    return {"missing": missing, "needs": needs}
