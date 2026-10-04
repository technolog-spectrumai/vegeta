"""Simulation results cached next to the notebook, loaded instead of solving again.

    cache.notebook("08_quadcopter")                          # vegeta.cache (or talos.cache / aeromant.cache): ./08_quadcopter.cache/
    r = model.solve(RUNS / "thrust", cache="frame_thrust")  # solved once; then loaded from 08_quadcopter.cache/frame_thrust.json
    m = base.solve_modes(RUNS / "modal", n_modes=8, cache="frame_modes")
    c = case.run(cache="canopy_15ms")                         # Aeromant: the same

The rule is deliberately simple: **if the entry exists it is loaded and nothing runs; if it does not, the simulation runs
and, when it succeeds, is saved.** Nothing checks whether the inputs or the code changed since: when they do, delete
the entry (``cache.clear("frame_thrust")``, or the file) or the whole folder (``cache.clear()``) and run again.

An entry is ``<folder>/<name>.json`` (the ``Result``: metrics, messages, the commands run) and ``<folder>/<name>/``
(copies of the files later cells read: the FEA ``.frd``/``.dat``/mesh, the CFD force and residual logs). A loaded
``Result`` points its artifacts at those copies, so ``talos.read_frd``, the plots and the fatigue assessment work
from the cache alone; artifacts that are not copied (an Aeromant case directory) keep their original path. Failed
results are never saved, so a failed run is tried again next time.

The folder: ``cache.notebook(name)`` sets ``<name>.cache`` (relative to the working directory, i.e. next to the
notebook); without it the notebook is detected in Jupyter / VS Code when possible, else ``./vegeta.cache``.
``VEGETA_CACHE=off`` (or ``cache.disable()``) runs everything and reads or writes nothing.

The settings live in the process environment (``VEGETA_CACHE_DIR``, ``VEGETA_CACHE``): Talos and Aeromant each carry
an identical copy of this module (the tools import nothing from each other), and one ``notebook()`` call in either, or
in ``vegeta.cache``, sets the folder for both.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import warnings
from pathlib import Path
from typing import Callable, Iterable

SUFFIX = ".cache"
FALLBACK = "vegeta.cache"
DIR_ENV, SWITCH_ENV = "VEGETA_CACHE_DIR", "VEGETA_CACHE"
verbose = True                      # one line when an entry is loaded or saved
_warned_fallback = False
_NAME = re.compile(r"^[A-Za-z0-9_.+\-]+(/[A-Za-z0-9_.+\-]+)*$")


# ------------------------------------------------------------------------------------------------ the folder
def notebook(name: str | Path | None = None) -> Path:
    """Use ``<name>.cache`` as the folder (``"08_quadcopter"``, ``"08_quadcopter.ipynb"`` or a path to the notebook;
    relative paths are relative to the working directory). Without ``name``: the notebook detected in Jupyter or VS
    Code. Returns the folder."""
    if name is None:
        nb = _detect_notebook()
        if nb is None:
            raise RuntimeError("cannot detect the notebook: call cache.notebook('<notebook name>')")
        name = nb
    p = Path(name)
    if p.suffix == ".ipynb":
        p = p.with_suffix("")
    if p.suffix != SUFFIX:
        p = p.with_name(p.name + SUFFIX)
    folder = p if p.is_absolute() else Path.cwd() / p
    os.environ[DIR_ENV] = str(folder)                        # shared with the other Vegeta tools' copy of this module
    return folder


def directory() -> Path:
    """The folder entries go to (see the module docstring for the order)."""
    global _warned_fallback
    if os.environ.get(DIR_ENV):
        return Path(os.environ[DIR_ENV])
    nb = _detect_notebook()
    if nb is not None:
        nb = Path(nb)
        return (nb if nb.is_absolute() else Path.cwd() / nb).with_suffix("").with_name(nb.stem + SUFFIX)
    if not _warned_fallback:
        warnings.warn(f"vegeta.cache: no notebook set, using ./{FALLBACK} (call cache.notebook('<name>'))", stacklevel=3)
        _warned_fallback = True
    return Path.cwd() / FALLBACK


def _detect_notebook() -> str | None:
    p = os.environ.get("JPY_SESSION_NAME", "")                  # Jupyter server: the notebook's path
    if p.endswith(".ipynb"):
        return p
    try:
        ip = get_ipython()                                       # noqa: F821  (defined inside IPython only)
    except NameError:
        return None
    ns = getattr(ip, "user_ns", {}) or {}
    for k in ("__session__", "__vsc_ipynb_file__"):              # ipykernel, VS Code
        v = str(ns.get(k) or "")
        if v.endswith(".ipynb"):
            return v
    return None


def disable() -> None:
    """Run everything; read and write nothing (the same as ``VEGETA_CACHE=off``)."""
    os.environ[SWITCH_ENV] = "off"


def enable() -> None:
    os.environ[SWITCH_ENV] = "on"


def enabled() -> bool:
    return os.environ.get(SWITCH_ENV, "on").lower() not in ("0", "off", "false", "no")


# ------------------------------------------------------------------------------------------------ entries
def path(name: str) -> Path:
    """``<folder>/<name>.json``. Names: letters, digits, ``_ . + -``; ``/`` groups entries in sub-folders."""
    if not _NAME.match(name or "") or any(part in (".", "..") for part in name.split("/")):
        raise ValueError(f"cache name {name!r}: use letters, digits and _ . + - (and / for sub-folders, no . or ..)")
    return directory() / f"{name}.json"


def files_dir(name: str) -> Path:
    """``<folder>/<name>/``: the copied files of an entry."""
    return path(name).with_suffix("")


def exists(name: str) -> bool:
    return path(name).is_file()


def entries() -> list[str]:
    """The names of the entries in the folder."""
    d = directory()
    return sorted(str(p.relative_to(d).with_suffix("")) for p in d.rglob("*.json")) if d.is_dir() else []


def clear(name: str | None = None) -> list[str]:
    """Delete one entry (its ``.json`` and its files) or, without ``name``, the whole folder; returns what went."""
    if name is None:
        d = directory()
        gone = entries()
        if d.is_dir():
            shutil.rmtree(d)
        return gone
    p, f = path(name), files_dir(name)
    gone = [name] if p.is_file() else []
    p.unlink(missing_ok=True)
    if f.is_dir():
        shutil.rmtree(f)
    return gone


def load(name: str, result_type, record_type, *, restore_to: str | Path | None = None, label: str = "vegeta"):
    """The entry as a ``result_type`` (its artifacts pointing at the copied files), or None when there is none.
    ``restore_to``: also copy the files there (e.g. a mesh into the directory the next solve uses) and point the
    artifacts at those copies."""
    if not enabled():
        return None
    p = path(name)
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"cache entry {p} cannot be read ({exc}); delete it to run again") from None
    base = p.parent
    cached = set(d.get("metadata", {}).get("cached_artifacts", []))
    arts = {}
    for k, v in d.get("artifacts", {}).items():
        arts[k] = (base / v) if k in cached else Path(v)
    if restore_to is not None:
        restore_to = Path(restore_to)
        restore_to.mkdir(parents=True, exist_ok=True)
        for k in cached:
            src = arts[k]
            if src.is_file():
                dst = restore_to / src.name
                shutil.copy2(src, dst)
                arts[k] = dst
    res = result_type(kind=d["kind"], status=d.get("status", "success"), metrics=dict(d.get("metrics", {})),
                      artifacts=arts, messages=list(d.get("messages", [])), duration_s=d.get("duration_s", 0.0),
                      execution=[record_type(**c) for c in d.get("execution", [])], metadata=dict(d.get("metadata", {})))
    res.metadata["cached"] = str(p)
    if verbose:
        print(f"{label}: {name!r} loaded from {p} (delete it to run again)")
    return res


def save(name: str, result, keep: Iterable[str] | Callable[[str, Path], bool] | None = None, *, label: str = "vegeta") -> Path:
    """Write ``result`` as the entry ``name``: the artifacts named in ``keep`` (or for which ``keep(key, path)`` is
    true; None: every artifact that is a file) are copied into ``<folder>/<name>/``. Returns the ``.json`` path."""
    p, fdir = path(name), files_dir(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    if fdir.is_dir():
        shutil.rmtree(fdir)
    d = result.to_dict()
    copied = []
    for k, v in result.artifacts.items():
        src = Path(v)
        want = (src.is_file() if keep is None else keep(k, src) if callable(keep) else k in set(keep))
        if want and src.is_file():
            fdir.mkdir(parents=True, exist_ok=True)
            dst = fdir / src.name
            if dst.exists():                                     # two artifacts with one file name
                dst = fdir / f"{k}_{src.name}"
            shutil.copy2(src, dst)
            d["artifacts"][k] = str(dst.relative_to(p.parent))
            copied.append(k)
        else:
            d["artifacts"][k] = str(src.resolve()) if src.exists() else str(src)
    d.setdefault("metadata", {})["cached_artifacts"] = copied
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(d, indent=1, default=str))
    tmp.replace(p)
    if verbose:
        print(f"{label}: {name!r} saved to {p}")
    return p


def cached(name: str | None, compute: Callable, result_type, record_type, *, keep=None, restore_to=None, label: str = "vegeta"):
    """``compute()`` once: the entry ``name`` when it exists, else ``compute()`` saved when it succeeds. ``name`` None
    or the cache disabled: just ``compute()``."""
    if name is None or not enabled():
        return compute()
    hit = load(name, result_type, record_type, restore_to=restore_to, label=label)
    if hit is not None:
        return hit
    res = compute()
    if res.ok:
        save(name, res, keep, label=label)
    return res


def entry_names(cache, defaults: list, what: str = "item") -> list:
    """One entry name (or None) per item of a batch: ``cache`` None/False -> all None; True -> ``defaults``; else the
    sequence given (one per item; names must not repeat)."""
    if cache is None or cache is False:
        return [None] * len(defaults)
    names = list(defaults) if cache is True else list(cache)
    if len(names) != len(defaults):
        raise ValueError(f"cache: one entry name (or None) per {what}")
    given = [n for n in names if n is not None]
    if len(set(given)) != len(given):
        raise ValueError(f"cache: two {what}s share an entry name ({sorted({n for n in given if given.count(n) > 1})})")
    return names

