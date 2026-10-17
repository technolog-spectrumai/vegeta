"""registry — finding and loading the project's flow (``workflows/flow.py``) once per process."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from .contract import Flow, FlowError

ENV = "VEGETA_YANUS_FLOW"
DEFAULT = Path("workflows") / "flow.py"
_LOADED: dict[Path, Flow] = {}


def find_flow_file(path=None, start=None) -> Path:
    """``path``, else ``$VEGETA_YANUS_FLOW``, else ``workflows/flow.py`` in the current directory or a parent."""
    if path:
        return Path(path).resolve()
    if os.environ.get(ENV):
        return Path(os.environ[ENV]).resolve()
    here = Path(start or Path.cwd()).resolve()
    for d in (here, *here.parents):
        if (d / DEFAULT).is_file():
            return d / DEFAULT
    raise FileNotFoundError(f"no flow registry: set {ENV} or run from inside the repository (looked for {DEFAULT} from {here} up)")


def load_flow(path=None, *, reload: bool = False, validate: bool = True) -> Flow:
    """The ``flow`` object of the registry file (cached per path). The file's own module is ``yanus_flow``."""
    p = find_flow_file(path)
    if p in _LOADED and not reload:
        return _LOADED[p]
    spec = importlib.util.spec_from_file_location("yanus_flow", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    flow = getattr(mod, "flow", None)
    if not isinstance(flow, Flow):
        raise FlowError(f"{p} defines no Flow named 'flow'")
    if flow.root is None:
        flow.root = p.parents[1] if p.parent.name == "workflows" else p.parent
    if validate:
        flow.validate()
    _LOADED[p] = flow
    return flow


_CURRENT: Flow | None = None


def use(flow: Flow | None) -> Flow | None:
    """Make ``flow`` the one ``current()`` returns (tests, tools that build their own); ``None`` goes back to the file."""
    global _CURRENT
    _CURRENT = flow
    return flow


def current(flow: Flow | None = None) -> Flow:
    if flow is not None:
        return flow
    if _CURRENT is not None:
        return _CURRENT
    return load_flow()
