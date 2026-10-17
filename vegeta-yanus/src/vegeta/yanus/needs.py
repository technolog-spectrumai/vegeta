"""needs — is a tool, a package, an environment variable there on this machine? (Found, never run beyond a
``--version``-style probe with a short timeout; a package is found, not imported.)"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import shutil
import subprocess
from dataclasses import dataclass

from .contract import Env, Need, Package, Tool

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
            return f"pip install the distribution that provides {self.need.name} (./install_local.sh), then source activate.sh"
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
    key = (type(need).__name__, need.name, getattr(need, "version_cmd", ()))
    if cache and key in _CACHE:
        return _CACHE[key]
    if isinstance(need, Tool):
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
