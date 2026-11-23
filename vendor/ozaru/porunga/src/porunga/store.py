"""Keys, and the on-disk store: ``<root>/<key>/manifest.json`` + ``<root>/<key>/files/…``.

Publication is a single ``os.rename`` of a fully written staging directory (``<root>/.tmp/…``) onto ``<root>/<key>``,
so a reader sees either nothing or a complete entry. Two writers of the same key: the first rename wins and the
second keeps the published entry. A damaged entry is moved to ``<root>/.trash/`` before its replacement is renamed in.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path, PurePath

SCHEMA = 1
MANIFEST = "manifest.json"
DATA = "data.json"
FILES = "files"
_ROOT: Path | None = None


class Corrupt(Exception):
    """The entry exists but is not intact (unreadable manifest, missing or truncated file)."""


@dataclass(frozen=True)
class Entry:
    key: str
    dir: Path
    data: object
    files: dict = field(default_factory=dict)        # {name: Path}
    metadata: dict | None = None


# ------------------------------------------------------------------------------------------------------------- keys
def _plain(obj):
    """The parameters as plain JSON values, deterministically: a closed set of types, nothing guessed."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"porunga.key: {obj!r} is not a finite number")
        return obj
    if isinstance(obj, PurePath):
        return obj.as_posix()
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                raise TypeError(f"porunga.key: dictionary keys must be strings, not {type(k).__name__} ({k!r})")
            out[k] = _plain(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        items = [_plain(v) for v in obj]
        return sorted(items, key=lambda v: json.dumps(v, sort_keys=True))
    if hasattr(obj, "tolist") and callable(obj.tolist):       # numpy arrays and scalars, without importing numpy
        return _plain(obj.tolist())
    raise TypeError(f"porunga.key: {type(obj).__name__} is not a supported parameter type (use str, numbers, bool, "
                    f"None, lists, dicts with string keys, sets or paths)")


def canonical(parameters) -> str:
    """The canonical JSON text of ``parameters`` (a JSON text, or a JSON-compatible value): sorted object keys, no
    whitespace, UTF-8 characters as they are, numbers as Python writes them (``1`` and ``1.0`` differ). A JSON text is
    parsed first, so its formatting and key order never matter. Specification: docs/identity.md."""
    if isinstance(parameters, (bytes, bytearray)):
        parameters = parameters.decode("utf-8")
    if isinstance(parameters, str):
        parameters = json.loads(parameters, parse_constant=_no_constant)
    return json.dumps(_plain(parameters), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _no_constant(name):
    raise ValueError(f"porunga.key: {name} is not allowed in an identity")


def key(parameters) -> str:
    """A stable key: the SHA-256 (64 lowercase hex characters) of the UTF-8 bytes of ``canonical(parameters)``.
    Identical descriptions give the identical key in any process, whatever their formatting or key order. A plain
    string parameter is read as JSON text — to hash a bare string, wrap it: ``key({"name": "…"})``."""
    return hashlib.sha256(canonical(parameters).encode("utf-8")).hexdigest()


# ------------------------------------------------------------------------------------------------------------ store
def configure(root) -> Path:
    """Set the default root for this process (``None`` falls back to ``$PORUNGA_DIR`` again)."""
    global _ROOT
    _ROOT = Path(root).expanduser().resolve() if root is not None else None
    return _ROOT


def _root(root) -> Path:
    if root is not None:
        return Path(root).expanduser()
    if _ROOT is not None:
        return _ROOT
    env = os.environ.get("PORUNGA_DIR")
    if env:
        return Path(env).expanduser()
    raise ValueError("porunga: no cache directory — pass root=, call porunga.configure(path) or set PORUNGA_DIR")


def _check_key(k: str) -> str:
    if not (isinstance(k, str) and len(k) == 64 and all(c in "0123456789abcdef" for c in k)):
        raise ValueError(f"porunga: {k!r} is not a key (use porunga.key(parameters))")
    return k


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _inspect(d: Path, k: str, verify: bool) -> dict:
    """The manifest of an intact entry; ``Corrupt`` with the reason otherwise."""
    try:
        man = json.loads((d / MANIFEST).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise Corrupt(f"{d}: no {MANIFEST}") from None
    except (OSError, ValueError) as e:
        raise Corrupt(f"{d}: unreadable {MANIFEST} ({e})") from None
    if not isinstance(man, dict) or man.get("schema") != SCHEMA or man.get("key") != k or man.get("status") != "complete":
        raise Corrupt(f"{d}: {MANIFEST} is not a complete entry for this key")
    for name, rec in {DATA: man.get("data"), **{f"{FILES}/{n}": r for n, r in (man.get("files") or {}).items()}}.items():
        p = d / name
        try:
            size = p.stat().st_size
        except OSError:
            raise Corrupt(f"{d}: {name} is missing") from None
        if not isinstance(rec, dict) or size != rec.get("size"):
            raise Corrupt(f"{d}: {name} has {size} bytes, the manifest says {rec.get('size') if isinstance(rec, dict) else '?'}")
        if verify and _sha256(p) != rec.get("sha256"):
            raise Corrupt(f"{d}: {name} does not match its recorded sha256")
    return man


def check(k: str, *, root=None, verify: bool = False) -> bool:
    """Whether a complete, intact entry exists for ``k`` (``verify=True`` also re-hashes every file)."""
    d = _root(root) / _check_key(k)
    if not d.is_dir():
        return False
    try:
        _inspect(d, k, verify)
    except Corrupt:
        return False
    return True


def load(k: str, *, root=None, verify: bool = False) -> Entry:
    """The entry for ``k``: ``KeyError`` when there is none, ``Corrupt`` when it is damaged."""
    d = _root(root) / _check_key(k)
    if not d.is_dir():
        raise KeyError(k)
    man = _inspect(d, k, verify)
    data = json.loads((d / DATA).read_text(encoding="utf-8"))
    files = {n: d / FILES / n for n in (man.get("files") or {})}
    return Entry(key=k, dir=d, data=data, files=files, metadata=man.get("metadata"))


def _sources(files) -> dict:
    if files is None:
        return {}
    if isinstance(files, (str, os.PathLike)):
        base = Path(files)
        if not base.is_dir():
            raise ValueError(f"porunga.save: files={base} is not a directory")
        return {p.relative_to(base).as_posix(): p for p in sorted(base.rglob("*")) if p.is_file()}
    out = {}
    for name, src in dict(files).items():
        name = PurePath(name).as_posix()
        if not name or name.startswith("/") or ".." in PurePath(name).parts:
            raise ValueError(f"porunga.save: file name {name!r} must be a relative path inside the entry")
        out[name] = Path(src)
    return out


def _fsync_dir(d: Path) -> None:
    try:
        fd = os.open(d, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _write(path: Path, text: str) -> dict:
    raw = text.encode("utf-8")
    with open(path, "wb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    return {"size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def save(k: str, data, metadata: dict | None = None, *, files=None, root=None) -> Entry:
    """Publish ``data`` (JSON-serialisable), the optional ``files`` (``{name: path}`` or a directory, copied) and the
    optional ``metadata`` (stored as given) under ``k``, atomically. An intact entry already there is kept and
    returned (the first writer wins); a damaged one is replaced. On any error nothing is published."""
    base = _root(root)
    _check_key(k)
    srcs = _sources(files)
    tmp_root = base / ".tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    tmp = tmp_root / f"{k}.{os.getpid()}.{uuid.uuid4().hex}"
    target = base / k
    try:
        (tmp / FILES).mkdir(parents=True)
        recs = {}
        for name, src in srcs.items():
            dst = tmp / FILES / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            with open(dst, "rb") as f:
                os.fsync(f.fileno())
            recs[name] = {"size": dst.stat().st_size, "sha256": _sha256(dst)}
        data_rec = _write(tmp / DATA, json.dumps(data, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=1))
        man = {"schema": SCHEMA, "key": k, "status": "complete", "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "data": data_rec, "files": recs, "metadata": metadata}
        _write(tmp / MANIFEST, json.dumps(man, sort_keys=True, ensure_ascii=False, indent=1, default=str))
        _fsync_dir(tmp)
        for _ in range(3):
            try:
                os.rename(tmp, target)
                _fsync_dir(base)
                return load(k, root=base)
            except OSError:
                if not target.exists():
                    raise
            try:                                              # someone was first: keep theirs if it is intact
                return load(k, root=base)
            except (KeyError, Corrupt):
                pass
            trash = base / ".trash"
            trash.mkdir(exist_ok=True)
            try:
                os.rename(target, trash / f"{k}.{os.getpid()}.{uuid.uuid4().hex}")
            except FileNotFoundError:
                pass
        raise RuntimeError(f"porunga.save: could not publish {k} (the entry keeps changing under concurrent writers)")
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)


def clean_partial(root=None) -> int:
    """Remove what interrupted or replaced writes left (``.tmp/``, ``.trash/``); returns how many were removed.
    Explicit only: run it when no write is in progress. Published entries are never touched."""
    base = _root(root)
    n = 0
    for sub in (".tmp", ".trash"):
        d = base / sub
        if d.is_dir():
            for p in d.iterdir():
                shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
                n += 1
    return n
