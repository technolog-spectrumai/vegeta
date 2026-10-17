"""stamps — the sidecar record ``<file>.yanus.json`` of an exported file: who gave it, its schema, its hash, when,
from which commit, with which needs and inputs. The next unit checks the file against it."""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

STAMP_SCHEMA = 1
SUFFIX = ".yanus.json"


def stamp_path(file) -> Path:
    f = Path(file)
    return f.with_name(f.name + SUFFIX)


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_commit(root) -> str | None:
    """The checked-out commit, read from ``.git`` (no subprocess)."""
    g = Path(root) / ".git"
    try:
        if g.is_file():                                       # a worktree: "gitdir: …"
            g = Path(g.read_text().split(":", 1)[1].strip())
        head = (g / "HEAD").read_text().strip()
        if not head.startswith("ref:"):
            return head[:12]
        ref = head.split(" ", 1)[1]
        if (g / ref).exists():
            return (g / ref).read_text().strip()[:12]
        packed = g / "packed-refs"
        if packed.exists():
            for line in packed.read_text().splitlines():
                if line.endswith(" " + ref):
                    return line.split(" ", 1)[0][:12]
    except (OSError, IndexError):
        pass
    return None


def write_json_atomic(p: Path, doc: dict) -> None:
    p = Path(p)
    tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str))
    os.replace(tmp, p)


def write_stamp(file, *, producer: str, kind: str, schema: int | None = None, root=None, needs: dict | None = None,
                inputs: list | None = None, design_fingerprint: dict | None = None) -> dict:
    f = Path(file)
    doc = {"yanus": STAMP_SCHEMA, "file": f.name, "producer": producer, "kind": kind, "schema": schema, "written_at": utc_now(),
           "written_ts": time.time(), "sha256": sha256_file(f), "size": f.stat().st_size, "git": git_commit(root) if root else None,
           "needs": needs or {}, "inputs": inputs or [], "design_fingerprint": design_fingerprint}
    write_json_atomic(stamp_path(f), doc)
    return doc


def read_stamp(file) -> dict | None:
    try:
        d = json.loads(stamp_path(file).read_text())
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) and d.get("yanus") == STAMP_SCHEMA else None


def file_schema(file, stamp: dict | None = None) -> int | None:
    """The file's schema: the stamp's, else a top-level ``"schema"`` of a JSON file."""
    if stamp and stamp.get("schema") is not None:
        return stamp["schema"]
    if str(file).endswith(".json"):
        try:
            v = json.loads(Path(file).read_text()).get("schema")
            return int(v) if v is not None else None
        except (OSError, ValueError, AttributeError, TypeError):
            return None
    return None


def missing_keys(file, keys) -> list[str]:
    if not keys or not str(file).endswith(".json"):
        return []
    try:
        doc = json.loads(Path(file).read_text())
    except (OSError, ValueError):
        return list(keys)
    return [k for k in keys if not isinstance(doc, dict) or k not in doc]


def check_file(file, *, root, min_schema: int | None = None, keys=(), producer: str | None = None) -> dict:
    """``{ok, status, stamp, stale}`` for a taken file: present, its stamp's hash and producer, schema ≥ ``min_schema``,
    the declared keys; **stale** when the file changed after it was stamped or one of the stamp's own file inputs was
    re-stamped later than it."""
    f = Path(root) / file
    out = {"ok": False, "status": "", "stamp": None, "stale": False, "path": f}
    if not f.exists():
        out["status"] = f"missing: {file}"
        return out
    st = read_stamp(f)
    out["stamp"] = st
    notes = []
    if st is None:
        notes.append("unstamped (written before Yanus or outside a gate)")
    else:
        if producer and st.get("producer") != producer:
            out["status"] = f"stamped by {st.get('producer')!r}, expected {producer!r}"
            return out
        if sha256_file(f) != st.get("sha256"):
            out["stale"] = True
            notes.append("modified since it was stamped")
        for inp in st.get("inputs") or []:
            p = inp.get("path")
            if p and inp.get("kind") == "file":
                ist = read_stamp(Path(root) / p)
                if ist and inp.get("sha256") and ist.get("sha256") != inp["sha256"]:
                    out["stale"] = True
                    notes.append(f"its input {p} was rewritten after it")
    sch = file_schema(f, st)
    if min_schema is not None and (sch is None or sch < min_schema):
        out["status"] = f"schema {sch} < {min_schema} required"
        return out
    mk = missing_keys(f, keys)
    if mk:
        out["status"] = f"missing keys: {', '.join(mk)}"
        return out
    when = (st or {}).get("written_at") or datetime.fromtimestamp(f.stat().st_mtime, timezone.utc).strftime("%Y-%m-%d %H:%M")
    if out["stale"]:
        out["status"] = "stale: " + "; ".join(notes)
        return out
    out["ok"] = True
    out["status"] = f"on file ({when}{', schema ' + str(sch) if sch is not None else ''})" + (f" — {'; '.join(notes)}" if notes else "")
    return out
