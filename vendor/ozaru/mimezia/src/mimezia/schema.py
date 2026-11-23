"""A real result's shape, recorded once; stubs checked against it."""
from __future__ import annotations

import json
from pathlib import Path

IGNORED = ("mock", "note", "mimic")


class SchemaError(AssertionError):
    pass


def _kind(v):
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, (int, float)) or v is None:
        return "number"          # null stands for NaN in JSON
    if isinstance(v, str):
        return "str"
    if isinstance(v, list):
        return "list"
    if isinstance(v, dict):
        return "dict"
    return type(v).__name__


def schema_of(value, depth: int = 3):
    """``{"type": ..., "keys": {...}}`` for dicts (to ``depth``), the item kind for lists."""
    k = _kind(value)
    if k == "dict" and depth > 0:
        return {"type": "dict", "keys": {key: schema_of(v, depth - 1) for key, v in value.items() if key not in IGNORED}}
    if k == "list" and value and depth > 0:
        return {"type": "list", "item": schema_of(value[0], depth - 1)}
    return {"type": k}


def record_schema(value, path) -> dict:
    s = schema_of(value)
    Path(path).write_text(json.dumps(s, indent=1, sort_keys=True))
    return s


def check(stub, schema, *, path: str = "", extra_ok: bool = True) -> None:
    """Raise ``SchemaError`` when ``stub`` lacks a key of ``schema`` or a value has another kind. Keys the stub adds
    are allowed when ``extra_ok`` (the label's are always ignored)."""
    if isinstance(schema, (str, Path)):
        schema = json.loads(Path(schema).read_text())
    t = schema["type"]
    k = _kind(stub)
    if t == "number" and k == "number" or t == k:
        pass
    else:
        raise SchemaError(f"{path or '/'}: {k}, the real API gives {t}")
    if t == "dict":
        missing = [key for key in schema["keys"] if key not in stub]
        if missing:
            raise SchemaError(f"{path or '/'}: missing {missing}")
        if not extra_ok:
            extra = [key for key in stub if key not in schema["keys"] and key not in IGNORED]
            if extra:
                raise SchemaError(f"{path or '/'}: not in the real API {extra}")
        for key, sub in schema["keys"].items():
            check(stub[key], sub, path=f"{path}/{key}", extra_ok=extra_ok)
    elif t == "list" and "item" in schema and stub:
        check(stub[0], schema["item"], path=f"{path}[0]", extra_ok=extra_ok)
