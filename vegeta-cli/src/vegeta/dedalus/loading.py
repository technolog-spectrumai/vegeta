"""Load designs from Python files or modules (used by the CLI)."""
from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
from pathlib import Path

from .design import Design


def load_module(target: str):
    """Import ``target`` which is either a path to a ``.py`` file or a dotted module name."""
    path = Path(target)
    if target.endswith(".py") or path.is_file():
        if not path.is_file():
            raise FileNotFoundError(f"no such design file: {target}")
        name = f"_dedalus_user_{abs(hash(str(path.resolve())))}"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        # Compile the file's current text (never a cached .pyc): the code that runs must be exactly the
        # source that source_identity() hashes, even when the file was edited a moment ago.
        exec(compile(path.read_text(), str(path.resolve()), "exec"), module.__dict__)
        return module
    return importlib.import_module(target)


def load_design(spec: str) -> Design:
    """Load ``file.py[:Name]`` or ``package.module[:Name]`` and return a Design instance.

    ``Name`` may be a Design subclass (instantiated without arguments) or a Design instance.
    Without ``Name`` the file must define exactly one design.
    """
    target, _, attr = spec.partition(":")
    if ":" not in spec and len(spec) > 2 and spec[1] == ":":  # Windows drive letter
        target, attr = spec, ""
    module = load_module(target)
    if attr:
        if not hasattr(module, attr):
            raise ValueError(f"{target} has no attribute {attr!r}")
        return _as_design(getattr(module, attr), attr)
    found = {
        name: obj for name, obj in vars(module).items()
        if (isinstance(obj, Design) or (inspect.isclass(obj) and issubclass(obj, Design)
            and obj.__module__ == module.__name__ and obj not in (Design,)))
    }
    if len(found) != 1:
        raise ValueError(
            f"{target} defines {len(found)} designs {sorted(found)}; select one with {target}:<Name>"
        )
    name, obj = next(iter(found.items()))
    return _as_design(obj, name)


def _as_design(obj, name) -> Design:
    if isinstance(obj, Design):
        return obj
    if inspect.isclass(obj) and issubclass(obj, Design):
        return obj()
    raise ValueError(f"{name!r} is not a Design (got {type(obj).__name__})")
