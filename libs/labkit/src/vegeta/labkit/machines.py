"""The registry of exportable machines: what leaves Vegeta for a simulator (Piccolo) as standard files through Yax.

Each lab declares its machines in ``designs/machines.py``::

    from vegeta.labkit import machines
    machines.register("drongo", "drongo_robot:drongo", parts=_drongo_parts)

``register(name, build, parts=None)``: ``build`` is a callable returning a Chiron ``Robot``, or ``"module:function"``
imported lazily from the lab's ``designs/`` (on ``sys.path`` once ``discover`` has run); ``parts(machine)`` returns the
part data a simulator needs and MJCF cannot hold, keyed by kind — ``{"ogma": …}`` (rotor constants, wheel radii),
``{"propulsion": …}`` (propeller maps, ``vegeta.labkit.propulsion``) — merged into the machine's ``data`` (kept in the USD).
``discover(*designs_dirs)`` runs each lab's ``machines.py``; ``tools/export.py`` exports what is registered."""
from __future__ import annotations

import importlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class Machine:
    name: str
    build: object                                    # callable or "module:function"
    parts: Callable | None = None
    lab: str | None = None

    def robot(self):
        if callable(self.build):
            return self.build()
        mod, fn = self.build.split(":")
        return getattr(importlib.import_module(mod), fn)()


_REGISTRY: dict[str, Machine] = {}
_current_lab: list[str | None] = [None]


def register(name: str, build, parts: Callable | None = None) -> Machine:
    if name in _REGISTRY and _REGISTRY[name].lab != _current_lab[0]:
        raise ValueError(f"machine {name!r} is registered by two labs ({_REGISTRY[name].lab}, {_current_lab[0]})")
    m = Machine(name, build, parts, _current_lab[0])
    _REGISTRY[name] = m
    return m


def get(name: str) -> Machine:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"no machine {name!r}; registered: {sorted(_REGISTRY)}") from None


def names() -> list[str]:
    return sorted(_REGISTRY)


def discover(*designs_dirs) -> list[str]:
    """Run each ``<dir>/machines.py`` (the dir put on ``sys.path`` first, as the notebooks do); the names registered."""
    for d in designs_dirs:
        d = Path(d).resolve()
        f = d / "machines.py"
        if not f.exists():
            continue
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
        lab = d.parent.name if d.name == "designs" and d.parent.parent.name == "labs" else d.name
        modname = f"_vegeta_machines_{lab}"
        if modname in sys.modules:
            continue
        _current_lab[0] = lab
        try:
            spec = importlib.util.spec_from_file_location(modname, f)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[modname] = mod
            spec.loader.exec_module(mod)
        finally:
            _current_lab[0] = None
    return names()
