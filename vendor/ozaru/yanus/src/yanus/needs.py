"""needs — the need types and their checks. A tool is found (and asked its version with a short timeout), a
package is found and not imported, an environment variable is set or not."""
from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class Need:
    name: str
    why: str = ""
    fix: str = ""

    @property
    def label(self) -> str:
        return self.name


@dataclass(frozen=True)
class Tool(Need):
    """A program on the PATH (``ccx``) — or, with ``probe``, anything a callable can find: ``probe() -> version
    string`` (raising or returning ``None`` = not found). ``version_cmd`` is run (2 s timeout) for the version."""
    version_cmd: tuple = ()
    probe: Callable[[], str | None] | None = field(default=None, compare=False)


@dataclass(frozen=True)
class Package(Need):
    """A Python module importable in this interpreter (found, not imported: ``numpy``, ``gmsh``)."""


@dataclass(frozen=True)
class AnyOf(Need):
    """One of several needs is enough: ``AnyOf("section backend", Package("neuralfoil"), Tool("xfoil"))``."""
    options: tuple = ()

    def __init__(self, name: str, *options: Need, why: str = "", fix: str = ""):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "why", why)
        object.__setattr__(self, "fix", fix)
        object.__setattr__(self, "options", tuple(options))

    @property
    def label(self) -> str:
        return f"{self.name} ({' or '.join(o.label for o in self.options)})"


@dataclass(frozen=True)
class Env(Need):
    """An environment variable that must be set (its value is not shown)."""


TIMEOUT = 2.0
_CACHE: dict = {}


@dataclass
class NeedResult:
    need: Need
    ok: bool
    version: str = ""
    detail: str = ""

    @property
    def fix(self) -> str:
        if self.ok:
            return ""
        if self.need.fix:
            return self.need.fix
        if isinstance(self.need, Package):
            return f"pip install the distribution that provides {self.need.name}"
        if isinstance(self.need, Env):
            return f"export {self.need.name}=…"
        return f"install {self.need.name} and put it on the PATH"


def _version_of(cmd: tuple) -> str:
    try:
        p = subprocess.run(list(cmd), capture_output=True, text=True, timeout=TIMEOUT)
    except (OSError, subprocess.SubprocessError):
        return ""
    text = (p.stdout or p.stderr or "").strip().splitlines()
    return text[0][:60] if text else ""


def _package_version(name: str) -> str:
    top = name.split(".")[0]
    for dist in (name.replace(".", "-"), top, top.replace("_", "-")):
        try:
            return importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            continue
    return ""


def check_need(need: Need, *, cache: bool = True) -> NeedResult:
    key = (type(need).__name__, need.name, getattr(need, "version_cmd", ()), getattr(need, "options", ()))
    if cache and key in _CACHE:
        return _CACHE[key]
    if isinstance(need, AnyOf):
        subs = [check_need(o, cache=cache) for o in need.options]
        hit = next((r for r in subs if r.ok), None)
        res = NeedResult(need, hit is not None, f"{hit.need.label} {hit.version}".strip() if hit else "",
                         "" if hit else "none of " + ", ".join(o.label for o in need.options))
    elif isinstance(need, Tool):
        if need.probe is not None:
            try:
                v = need.probe()
                res = NeedResult(need, bool(v), str(v or ""), "" if v else "not found")
            except Exception as exc:
                res = NeedResult(need, False, "", f"not found ({str(exc)[:80]})")
        else:
            exe = shutil.which(need.version_cmd[0] if need.version_cmd else need.name) or shutil.which(need.name)
            res = NeedResult(need, bool(exe), _version_of(need.version_cmd) if exe and need.version_cmd else ("found" if exe else ""),
                             exe or "not on the PATH")
    elif isinstance(need, Package):
        try:
            spec = importlib.util.find_spec(need.name)
        except (ImportError, ValueError) as exc:
            spec, err = None, str(exc)
        else:
            err = ""
        res = NeedResult(need, spec is not None, _package_version(need.name) if spec is not None else "", "" if spec else (err or "not importable"))
    elif isinstance(need, Env):
        res = NeedResult(need, bool(os.environ.get(need.name)), "set" if os.environ.get(need.name) else "", "" if os.environ.get(need.name) else "not set")
    else:
        res = NeedResult(need, False, "", f"unknown need {need!r}")
    if cache:
        _CACHE[key] = res
    return res


def check_needs(needs) -> list[NeedResult]:
    return [check_need(n) for n in needs]


def clear_cache() -> None:
    _CACHE.clear()


def table(results) -> str:
    """The results as plain text: one line per need — ok/MISSING, the label, the version or what is wrong, the fix."""
    rows = [("ok" if r.ok else "MISSING", r.need.label, r.version if r.ok else r.detail, r.fix) for r in results]
    w = [max([len(x[i]) for x in rows] + [0]) for i in range(3)]
    return "\n".join(f"{a:<{w[0]}}  {b:<{w[1]}}  {c:<{w[2]}}" + (f"  → {d}" if d else "") for a, b, c, d in rows)
