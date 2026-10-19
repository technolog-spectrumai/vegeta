"""Every stub result says it is one."""
from __future__ import annotations

NOTE = "MOCK (mimesis): stub numbers with the real schema — not a result"


def label(result: dict, name: str, note: str = NOTE) -> dict:
    out = dict(result)
    out["mock"] = True
    out.setdefault("note", note)
    out["mimic"] = name
    return out


def is_mock(result) -> bool:
    return isinstance(result, dict) and result.get("mock") is True
