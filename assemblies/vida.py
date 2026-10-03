"""The assembly tree and its file, ``.vida`` (virtual integral design assembly).

An ``Assembly`` is a node: a name, a kind, the parameters that define it, its sub-assemblies, what was computed on it
(``results``: JSON values and numpy arrays) and the files it made (``files``: STEP, STL, meshes, solver cases). A
workflow builds the tree, components fill it, ``save`` writes it to one ``.vida`` file and ``load`` reads it back on
any machine.

A ``.vida`` is a zip archive (standard library, random access, ``unzip -l`` shows it)::

    manifest.json                         format version, root, written_at, git commit, versions, include level
    tree.json                             every node: name, kind, params, meta, key, files listed, children
    nodes/<path>/results.json             JSON results; arrays replaced by references into arrays.npz
    nodes/<path>/arrays.npz               the numpy arrays (no pickle)
    nodes/<path>/files/<name>[/...]       the files of the included levels

``include`` chooses how much goes in: "results" (parameters and results only: kB-MB, the level committed to
``assemblies/data``), "geometry" (+ STEP/STL), "mesh" (+ meshes and FEA results), "cases" (+ whole solver case
directories, to move a product to another machine). Files of levels left out are still listed with their path and
SHA-256.

Reuse: ``key`` is the SHA-256 of a node's kind, parameters and its children's keys. ``reuse(prior)`` copies the results
and files of every node whose key equals the same node's key in ``prior`` (a tree loaded from an earlier ``.vida``).
That is the whole cache rule: when code changed but parameters did not, the engineer forces a new run.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import re
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np

FORMAT = 1
LEVELS = ("results", "geometry", "mesh", "cases")
REPO = Path(__file__).resolve().parents[1]          # paths inside the repository are stored relative to it
_NAME = re.compile(r"^[A-Za-z0-9_.+\-]+$")
_RESERVED = ("__array__", "__float__")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------------------------- plain data
def canonical(obj, where: str = "params"):
    """Plain JSON form of a parameter value, order-independent for dicts; numpy becomes lists, non-finite floats
    strings. Raises ``TypeError`` for anything without a stable plain form (pass ``to_dict()`` of objects)."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(obj) else repr(float(obj))
    if isinstance(obj, dict):
        bad = [k for k in obj if not isinstance(k, str)]
        if bad:
            raise TypeError(f"{where}: dict keys must be strings, got {bad[:3]}")
        return {k: canonical(obj[k], f"{where}.{k}") for k in sorted(obj)}
    if isinstance(obj, (list, tuple)):
        return [canonical(v, f"{where}[{i}]") for i, v in enumerate(obj)]
    if isinstance(obj, np.ndarray):
        return canonical(obj.tolist(), where)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"{where}: {type(obj).__name__} has no plain form; store its to_dict() or plain numbers")


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(canonical(obj), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _encode(obj, arrays: dict, where: str):
    """Results to JSON: arrays go to ``arrays`` (referenced by name), non-finite floats are tagged."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int, np.integer)) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else {"__float__": repr(f)}
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            raise TypeError(f"{where}: object arrays cannot be stored; use numbers or lists")
        name = f"a{len(arrays)}"
        arrays[name] = obj
        return {"__array__": name}
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                raise TypeError(f"{where}: dict keys must be strings, got {k!r}")
            if k in _RESERVED:
                raise TypeError(f"{where}: {k!r} is a reserved key")
            out[k] = _encode(v, arrays, f"{where}.{k}")
        return out
    if isinstance(obj, (list, tuple)):
        return [_encode(v, arrays, f"{where}[{i}]") for i, v in enumerate(obj)]
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"{where}: {type(obj).__name__} cannot be stored in a .vida; store plain values, lists, dicts or "
                    f"numpy arrays (an object's to_dict())")


def _decode(obj, arrays):
    if isinstance(obj, dict):
        if set(obj) == {"__array__"}:
            return arrays[obj["__array__"]]
        if set(obj) == {"__float__"}:
            return float(obj["__float__"])
        return {k: _decode(v, arrays) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode(v, arrays) for v in obj]
    return obj


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_path(path: Path) -> tuple[str, int]:
    """SHA-256 and size of a file, or of a directory (its files' relative paths and hashes)."""
    if path.is_file():
        return _sha256_file(path), path.stat().st_size
    h, size = hashlib.sha256(), 0
    for f in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(f.relative_to(path).as_posix().encode() + b"\0" + _sha256_file(f).encode() + b"\n")
        size += f.stat().st_size
    return h.hexdigest(), size


# --------------------------------------------------------------------------------------------- files
@dataclass
class FileRef:
    """A file or directory an assembly made: where it is on disk, its level, its hash; in a loaded tree also the
    archive member that holds it (when its level was included)."""

    level: str
    path: Path | None
    sha256: str | None = None
    size: int | None = None
    is_dir: bool = False
    member: str | None = None
    archive: Path | None = None

    def to_dict(self) -> dict:
        path, where = (str(self.path), "absolute") if self.path else (None, None)
        if self.path is not None and self.path.is_absolute():
            try:
                path, where = self.path.relative_to(REPO).as_posix(), "repo"
            except ValueError:
                pass
        return {"level": self.level, "path": path, "path_from": where, "sha256": self.sha256, "size": self.size,
                "is_dir": self.is_dir, "member": self.member}


def _path_from(f: dict) -> Path | None:
    if not f.get("path"):
        return None
    return REPO / f["path"] if f.get("path_from") == "repo" else Path(f["path"])


# --------------------------------------------------------------------------------------------- the tree
@dataclass
class Assembly:
    """One node of the assembly tree. Workflows build it, components fill ``results`` and ``files``."""

    name: str
    kind: str
    params: dict = field(default_factory=dict)
    children: list["Assembly"] = field(default_factory=list)
    results: dict = field(default_factory=dict)
    files: dict[str, FileRef] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if not _NAME.match(self.name):
            raise ValueError(f"assembly name {self.name!r}: use letters, digits and _ . + - only")
        self.params = canonical(dict(self.params), f"{self.name}.params")
        names = [c.name for c in self.children]
        if len(set(names)) != len(names):
            raise ValueError(f"{self.name}: two sub-assemblies share a name")

    # -- identity ---------------------------------------------------------------------------
    @property
    def key(self) -> str:
        """SHA-256 of the kind, the parameters and the children's keys: what the node is, not what was computed."""
        return digest({"kind": self.kind, "params": self.params, "children": [[c.name, c.key] for c in self.children]})

    # -- the tree ---------------------------------------------------------------------------
    def add(self, child: "Assembly") -> "Assembly":
        if any(c.name == child.name for c in self.children):
            raise ValueError(f"{self.name} already has a sub-assembly {child.name!r}")
        self.children.append(child)
        return child

    def child(self, path: str) -> "Assembly":
        """A sub-assembly by its path below this node: ``"engine/compressor"``."""
        node = self
        for part in [p for p in path.split("/") if p]:
            for c in node.children:
                if c.name == part:
                    node = c
                    break
            else:
                raise KeyError(f"{node.name} has no sub-assembly {part!r} (it has {[c.name for c in node.children]})")
        return node

    def walk(self, prefix: str = "") -> Iterator[tuple[str, "Assembly"]]:
        """Every node with its path from this root ("" for the root itself), depth first."""
        yield prefix, self
        for c in self.children:
            yield from c.walk(f"{prefix}/{c.name}" if prefix else c.name)

    def copy(self, name: str | None = None) -> "Assembly":
        """A deep copy, renamed (to graft a loaded tree as a sub-assembly: ``aguya.add(microjet.copy("engine"))``)."""
        new = copy.deepcopy(self)
        if name is not None:
            if not _NAME.match(name):
                raise ValueError(f"assembly name {name!r}: use letters, digits and _ . + - only")
            new.name = name
        return new

    # -- results ----------------------------------------------------------------------------
    def record(self, **results) -> "Assembly":
        """Store computed results (JSON values, numpy arrays) and stamp the time; clears an earlier NOT RUN note."""
        for k, v in results.items():
            _encode(v, {}, f"{self.name}.results.{k}")          # fail now, not at save time
        self.results.update(results)
        self.meta["computed_at"] = utc_now()
        self.meta.pop("not_run", None)
        self.meta.pop("reused", None)
        return self

    def not_run(self, reason: str) -> "Assembly":
        """Say why something on this node was not computed (shown by ``table``)."""
        self.meta.setdefault("not_run", [])
        if reason not in self.meta["not_run"]:
            self.meta["not_run"].append(reason)
        return self

    def attach(self, name: str, path: str | Path, level: str = "geometry") -> FileRef:
        """Register a file or directory this node made; ``level`` decides whether ``save`` puts it in the archive."""
        if level not in LEVELS[1:]:
            raise ValueError(f"level must be one of {LEVELS[1:]}")
        if not _NAME.match(name):
            raise ValueError(f"file name {name!r}: use letters, digits and _ . + - only")
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        ref = FileRef(level=level, path=path.resolve(), is_dir=path.is_dir())
        self.files[name] = ref
        return ref

    def file(self, name: str, to: str | Path | None = None) -> Path:
        """The file ``name`` on disk: the recorded path when it still holds the same content, else extracted from the
        archive it was loaded from (to ``to``, or a folder in the temp directory). Raises ``FileNotFoundError`` when
        it is neither (its level was not included and the original is gone or changed)."""
        if name not in self.files:
            raise KeyError(f"{self.name} has no file {name!r} (it has {sorted(self.files)})")
        ref = self.files[name]
        if ref.path is not None and ref.path.exists() and (ref.sha256 is None or _sha256_path(ref.path)[0] == ref.sha256):
            return ref.path
        if ref.member is not None and ref.archive is not None:
            base = Path(to) if to else Path(tempfile.gettempdir()) / "vida" / (
                f"{ref.archive.stem}-{ref.archive.stat().st_size}-{ref.archive.stat().st_mtime_ns}")
            with zipfile.ZipFile(ref.archive) as z:
                members = [m for m in z.namelist() if m == ref.member or m.startswith(ref.member + "/")]
                for m in members:
                    z.extract(m, base)
            return base / ref.member
        raise FileNotFoundError(f"{self.name}: {name} ({ref.level}) is not in the archive and {ref.path} is gone or "
                                f"changed; save with include={ref.level!r} to carry it")

    def forget(self) -> "Assembly":
        """Drop the results and files of this node and its sub-assemblies (they will be computed again)."""
        for _, n in self.walk():
            n.results, n.files = {}, {}
            for k in ("computed_at", "reused", "not_run"):
                n.meta.pop(k, None)
        return self

    def reuse(self, prior: "Assembly | None") -> list[str]:
        """Copy results and files from ``prior`` (an earlier tree of the same assembly) for every node whose key is
        unchanged, which had results there and has none here; returns the paths reused. Nodes are matched by their path
        below the root. Call it again after adding nodes whose parameters needed earlier results."""
        if prior is None:
            return []
        old = dict(prior.walk())
        done = []
        for path, node in self.walk():
            p = old.get(path)
            if node.results or p is None or p.key != node.key or not p.results:
                continue
            node.results = copy.deepcopy(p.results)
            node.files = copy.deepcopy(p.files)
            node.meta["reused"] = {"computed_at": p.meta.get("computed_at") or p.meta.get("reused", {}).get("computed_at"),
                                   "from": prior.meta.get("loaded_from")}
            node.meta.pop("not_run", None)
            done.append(path or self.name)
        return done

    # -- showing it -------------------------------------------------------------------------
    def status(self) -> str:
        if self.meta.get("not_run") and not self.results:
            return "NOT RUN"
        if self.meta.get("reused"):
            return "reused"
        if self.results:
            return "computed" + (" (part NOT RUN)" if self.meta.get("not_run") else "")
        return "-"

    def rows(self) -> list[dict]:
        out = []
        for path, n in self.walk():
            when = n.meta.get("computed_at") or (n.meta.get("reused") or {}).get("computed_at") or ""
            out.append({"assembly": path or n.name, "kind": n.kind, "status": n.status(), "computed": when,
                        "results": ", ".join(sorted(n.results)), "files": ", ".join(sorted(n.files)),
                        "not run": "; ".join(n.meta.get("not_run", [])), "key": n.key[:10]})
        return out

    def table(self):
        """One row per node (pandas DataFrame): path, kind, status (computed / reused / NOT RUN), when, what."""
        import pandas as pd

        return pd.DataFrame(self.rows()).set_index("assembly")

    def __str__(self) -> str:
        lines = []
        for path, n in self.walk():
            depth = path.count("/") + (1 if path else 0)
            extra = f"  NOT RUN: {'; '.join(n.meta['not_run'])}" if n.meta.get("not_run") else ""
            res = f" [{', '.join(sorted(n.results))}]" if n.results else ""
            lines.append(f"{'  ' * depth}{n.name} ({n.kind}) {n.status()}{res}{extra}")
        return "\n".join(lines)

    # -- the file ---------------------------------------------------------------------------
    def save(self, path: str | Path, include: str | tuple = "results") -> Path:
        """Write the tree to ``path`` (a ``.vida``), with the files of the levels up to ``include``."""
        level = _level(include)
        keep = set(LEVELS[: LEVELS.index(level) + 1])
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        written = utc_now()
        with zipfile.ZipFile(tmp, "w") as z:
            def node_dict(n: Assembly, npath: str) -> dict:
                base = f"nodes/{npath}"
                arrays: dict[str, np.ndarray] = {}
                results = _encode(n.results, arrays, f"{npath}.results")
                meta = {k: v for k, v in n.meta.items() if k not in ("loaded_from", "manifest", "key_changed")}
                d = {"name": n.name, "kind": n.kind, "key": n.key, "params": n.params,
                     "meta": canonical(meta, f"{npath}.meta"), "results": None, "arrays": None, "files": {}}
                if n.results:
                    d["results"] = f"{base}/results.json"
                    z.writestr(d["results"], json.dumps(results, separators=(",", ":")), zipfile.ZIP_DEFLATED)
                if arrays:
                    buf = io.BytesIO()
                    np.savez(buf, **arrays)
                    d["arrays"] = f"{base}/arrays.npz"
                    z.writestr(d["arrays"], buf.getvalue(), zipfile.ZIP_STORED)
                for fname, ref in n.files.items():
                    src = ref.path if ref.path is not None and ref.path.exists() else None
                    if src is not None:
                        ref.sha256, ref.size = _sha256_path(src)
                        ref.is_dir = src.is_dir()
                    member = None
                    if ref.level in keep:
                        member = f"{base}/files/{fname}"
                        if src is not None:
                            _write_member(z, src, member)
                        elif ref.member is not None and ref.archive is not None:       # carried over from a loaded tree
                            _copy_member(z, ref.archive, ref.member, member)
                        else:
                            raise FileNotFoundError(f"{npath}: file {fname} ({ref.level}) is gone and not in an archive")
                    d["files"][fname] = dict(ref.to_dict(), member=member)
                d["children"] = [node_dict(c, f"{npath}/{c.name}") for c in n.children]
                return d

            tree = node_dict(self, self.name)
            manifest = {"vida": FORMAT, "root": self.name, "kind": self.kind, "written_at": written, "include": level,
                        "git": _git(), "versions": _versions()}
            z.writestr("manifest.json", json.dumps(manifest, indent=1), zipfile.ZIP_DEFLATED)
            z.writestr("tree.json", json.dumps(tree, indent=1), zipfile.ZIP_DEFLATED)
        os.replace(tmp, path)
        return path


def _level(include) -> str:
    levels = [include] if isinstance(include, str) else list(include)
    bad = [x for x in levels if x not in LEVELS]
    if bad or not levels:
        raise ValueError(f"include must be one of {LEVELS} (or a tuple of them), got {include!r}")
    return max(levels, key=LEVELS.index)


def _write_member(z: zipfile.ZipFile, src: Path, member: str) -> None:
    text = (".json", ".txt", ".log", ".csv", ".inp", ".dat", ".msh", ".step", ".stp", ".stl", ".py", "")
    if src.is_file():
        z.write(src, member, zipfile.ZIP_DEFLATED if src.suffix.lower() in text else zipfile.ZIP_STORED)
        return
    for f in sorted(p for p in src.rglob("*") if p.is_file()):
        z.write(f, f"{member}/{f.relative_to(src).as_posix()}", zipfile.ZIP_DEFLATED)


def _copy_member(z: zipfile.ZipFile, archive: Path, old: str, new: str) -> None:
    with zipfile.ZipFile(archive) as src:
        for m in src.infolist():
            if m.filename == old or m.filename.startswith(old + "/"):
                z.writestr(new + m.filename[len(old):], src.read(m), m.compress_type)


def _git() -> dict:
    here = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=here, capture_output=True, text=True, timeout=10)
        if commit.returncode == 0:
            return {"commit": commit.stdout.strip(), "dirty": bool(dirty.stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        pass
    return {"commit": None, "dirty": None}


def _versions() -> dict:
    from importlib import metadata

    out = {}
    for dist in ("vegeta-cli", "vegeta-core", "numpy", "cadquery", "gmsh"):
        try:
            out[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            pass
    return out


# --------------------------------------------------------------------------------------------- reading
def manifest(path: str | Path) -> dict:
    """The manifest of a ``.vida`` without reading the tree."""
    with zipfile.ZipFile(path) as z:
        return json.loads(z.read("manifest.json"))


def load(path: str | Path) -> Assembly:
    """Read a ``.vida``: the tree with every node's params, meta and results; files stay in the archive until
    ``node.file(name)`` asks for one. The root's ``meta["loaded_from"]`` is the path."""
    path = Path(path).resolve()
    with zipfile.ZipFile(path) as z:
        man = json.loads(z.read("manifest.json"))
        if man.get("vida", 0) > FORMAT:
            raise ValueError(f"{path} is .vida format {man.get('vida')}; this reader knows up to {FORMAT}: update Vegeta")
        tree = json.loads(z.read("tree.json"))

        def node(d: dict) -> Assembly:
            arrays = {}
            if d.get("arrays"):
                with np.load(io.BytesIO(z.read(d["arrays"])), allow_pickle=False) as npz:
                    arrays = {k: npz[k] for k in npz.files}
            results = _decode(json.loads(z.read(d["results"])), arrays) if d.get("results") else {}
            files = {k: FileRef(level=f["level"], path=_path_from(f), sha256=f.get("sha256"),
                                size=f.get("size"), is_dir=f.get("is_dir", False), member=f.get("member"),
                                archive=path if f.get("member") else None)
                     for k, f in d.get("files", {}).items()}
            n = Assembly(d["name"], d["kind"], params=d.get("params", {}), results=results, meta=dict(d.get("meta", {})))
            n.files = files
            n.children = [node(c) for c in d.get("children", [])]
            if n.key != d.get("key"):
                n.meta["key_changed"] = "the key computed now differs from the one written (canonical form changed)"
            return n

        root = node(tree)
    root.meta["loaded_from"] = str(path)
    root.meta["manifest"] = man
    return root
