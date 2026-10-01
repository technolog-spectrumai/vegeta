"""Paired trials: a picklable trial description, a cached parallel runner and helpers for paired designs.

A **Trial** names everything one ChironLab run needs as plain data — the robot and controller *factories* as
``'file.py:function'`` (or ``'module:function'``) strings with their keyword arguments, the terrain spec, the seed,
the course, the target speed, the failure-rule overrides, scheduled disturbances, the lab settings and the metric
options — so it pickles, prints and hashes. Nothing robot-specific lives here; the study's own module builds its
trial list (e.g. ``notebooks/designs/stability_experiments.py``).

    trials = paired_trials(make_trial, {"treatment": ["spring", "spring_damper"], "level": [0.15, 0.25]},
                           paired_seeds(30))
    df = run_trials(trials, processes=4, cache_dir="runs/chiron_cache")      # one row per trial
    ep = rerun_with_log(trials[0])                                          # the full log, for mechanism plots

**Running.** ``run_trials`` simulates every trial not yet in the cache, each in its own forked process
(``processes`` at a time; the factory modules and MuJoCo are imported once in the parent and inherited). The
worker builds the robot, terrain, lab, rules and controller, runs the episode, computes the metrics *in the
worker* (``metrics_fn``, default ``vegeta.chiron.metrics:trial_metrics``) and writes the row to the cache
immediately, so an interrupted sweep loses only the trials in flight. A trial that raises — or whose process
dies — gives a row with ``status='error'`` and the message; it is not cached and is retried next time.

**Cache.** ``<cache_dir>/rows/<trial_id>.json``; ``trial_id`` is a SHA-256 of the trial's canonical JSON spec —
every field that can change the result (factories, keyword arguments, terrain, seed, speeds, course, rules,
disturbances, lab settings, log info, metric function and options, ``version``) — and not the labels (``name``,
``factors``). Re-runs skip cached trials. The cache does not see code changes: bump ``Trial.version`` (or use a new
``cache_dir``) when a factory, a controller or the metrics change.

**Rows** (``pandas.DataFrame``, trials in input order): ``trial_id``, ``name``, the ``factors`` columns, the spec
flattened (``robot_factory``, ``robot_kwargs.<k>``, ``controller_factory``, ``controller_kwargs.<k>``,
``terrain.<k>`` or ``terrain_factory``/``terrain_kwargs.<k>``, ``seed``, ``v_target``, ``course_m``, ``duration``,
``settle``, ``rules.<k>``, ``disturbances`` (JSON), ``lab_kwargs.<k>``, ``info.<k>``, ``metrics_kwargs.<k>``,
``version``, ``metrics_fn``), then ``status``, ``error``, ``cached``, ``log_path``, the outcome (``success``,
``reason``, ``t_end``, ``distance_m``, ``x_end``, ``detail``), the episode's ``robot``/``controller``/``treatment``
names, ``total_mass``, timing (``wall_time_s``, ``simulated_s``, ``realtime_factor``) and every metric.
A metric whose name is already a column with a different value is stored as ``metric.<name>``.

**Factory conventions.** ``robot``: ``fn(**robot_kwargs) -> chiron.Robot``. ``controller``:
``fn(**controller_kwargs) -> controller`` (an object with ``reset(lab, seed)`` and ``__call__(obs) -> Command``,
or any callable); it also receives ``v_target=trial.v_target`` if it has a ``v_target`` parameter that
``controller_kwargs`` does not set, and ``robot=<the built Robot>`` if it has a ``robot`` parameter.
``controller=None`` holds the standing pose. A terrain dict is a ``terrain_from_spec`` spec; when it has no
``seed`` and its kind takes one, the trial's seed is used (paired configurations share both). A terrain string is a
factory ``fn(**terrain_kwargs) -> Terrain | spec dict`` (given ``seed`` likewise). The terrain's bookkeeping keys
(``level``, ``label``, ``difficulty``) are recorded in the log's terrain spec. ``info['treatment']`` defaults to the
robot's ``treatment`` attribute when it has one.

A ``.py`` factory file is imported under its own module name with its directory on ``sys.path`` (so design files
that ``import`` each other plainly share one module object); forked trials inherit what the parent imported.
"""
from __future__ import annotations

import dataclasses
import enum
import hashlib
import importlib
import importlib.util
import inspect
import itertools
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

__all__ = ["Trial", "run_trials", "rerun_with_log", "paired_seeds", "factor_grid", "paired_trials", "pair_up",
           "trials_frame", "load_cached", "load_ref", "trial_id", "DEFAULT_METRICS"]

DEFAULT_METRICS = "vegeta.chiron.metrics:trial_metrics"
CACHE_FORMAT = 1
_SEEDED_KINDS = {"long_bumps", "longitudinal_bumps", "alt_bumps", "alternating_bumps", "steps", "rough"}
_LEVEL_KEYS = ("level", "label", "difficulty")
_OUTCOME_KEYS = ("success", "reason", "t_end", "distance_m", "x_end", "detail")


# ----------------------------------------------------------------------------------------------- references
def load_ref(ref: str | Callable) -> Any:
    """Resolve ``'path/to/file.py:name'`` or ``'package.module:name'`` (``name`` may be dotted) to the object.

    A callable is returned as is. A ``.py`` file is imported as the module named by its stem with its directory
    put first on ``sys.path`` (design files import each other plainly); if that name is taken by a different
    file, a unique name is used instead.
    """
    if callable(ref):
        return ref
    if not isinstance(ref, str) or ":" not in ref:
        raise ValueError(f"expected 'file.py:name' or 'module:name', got {ref!r}")
    target, _, attr = ref.rpartition(":")
    if not target or not attr:
        raise ValueError(f"expected 'file.py:name' or 'module:name', got {ref!r}")
    module = _load_file(target) if target.endswith(".py") else importlib.import_module(target)
    obj = module
    for part in attr.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            raise ValueError(f"{target} has no attribute {attr!r}") from None
    return obj


def _load_file(target: str):
    path = Path(target).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"no such file: {target}")
    folder = str(path.parent)
    if folder not in sys.path:
        sys.path.insert(0, folder)
    name = path.stem
    mod = sys.modules.get(name)
    if mod is not None:
        f = getattr(mod, "__file__", None)
        if f and Path(f).resolve() == path:
            return mod
        name = f"_chiron_user_{path.stem}_{hashlib.sha1(str(path).encode()).hexdigest()[:10]}"
        mod = sys.modules.get(name)
        if mod is not None:
            return mod
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _ref_name(fn) -> str:
    """A stable text name for a metrics function (its 'module:qualname' when importable)."""
    if isinstance(fn, str):
        return fn
    mod, qual = getattr(fn, "__module__", None), getattr(fn, "__qualname__", None)
    if mod and qual and "<" not in qual:
        return f"{mod}:{qual}"
    return f"{mod}:{getattr(fn, '__name__', repr(fn))}"


def _accepts(fn, name: str) -> bool:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in params and params[name].kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                                    inspect.Parameter.KEYWORD_ONLY)


# ----------------------------------------------------------------------------------------------- canonical form
def _canon(obj):
    """Plain JSON-able, order-independent form of a spec value (raises on objects without a stable form)."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else repr(obj)
    if isinstance(obj, Mapping):
        return {str(k): _canon(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_canon(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted((_canon(v) for v in obj), key=lambda v: json.dumps(v, sort_keys=True))
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {"__dataclass__": type(obj).__qualname__,
                **{f.name: _canon(getattr(obj, f.name)) for f in dataclasses.fields(obj)}}
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "tolist") and hasattr(obj, "dtype"):          # numpy scalars and arrays
        return _canon(obj.tolist())
    if hasattr(obj, "spec") and callable(obj.spec):                 # a chiron Terrain
        return _canon(obj.spec())
    if isinstance(obj, enum.Enum):
        return f"{type(obj).__qualname__}.{obj.name}"
    raise TypeError(f"trial spec value of type {type(obj).__name__} has no stable form; use plain data "
                    "(numbers, strings, lists, dicts, dataclasses)")


def _dumps(obj) -> str:
    return json.dumps(_canon(obj), sort_keys=True, separators=(",", ":"), allow_nan=False)


def trial_id(spec: Mapping) -> str:
    """SHA-256 (first 20 hex digits) of a spec's canonical JSON."""
    return hashlib.sha256(_dumps(spec).encode()).hexdigest()[:20]


# ----------------------------------------------------------------------------------------------- the trial
@dataclass
class Trial:
    """One ChironLab run as plain, picklable data (see the module docstring for the factory conventions).

    ``robot`` / ``controller``: factory references; ``robot_kwargs`` / ``controller_kwargs`` their keyword
    arguments. ``terrain``: a terrain spec dict (``{'kind': 'rough', 'rms': 0.027, 'correlation_length': 0.0425,
    'start': 0.3, 'level': 0.15}``; ``level`` is bookkeeping) or a factory reference with ``terrain_kwargs``.
    ``seed``: the trial's seed (terrain randomness unless the spec sets its own, and the controller's ``reset``).
    ``v_target`` [m/s]; ``course_m`` [m]: with a course, ``FailureRules(course_m, v_target=v_target, **rules)``
    decide the outcome; without one, ``duration`` [s] is required and the run has no failure rules.
    ``duration`` [s] caps the walking time (default: the rules' timeout); ``settle`` [s] before walking.
    ``disturbances``: dicts of ``Disturbance`` fields (``body``, ``t_start``, ``duration``, ``impulse`` or
    ``force``, ``direction``). ``lab_kwargs``: ``ChironLab`` keywords (``timestep``, ``control_dt``, ``log_dt``,
    ``course_extent``, ...). ``info``: extra entries for the episode log (``treatment``, ``payload``, ...).
    ``metrics_kwargs``: keyword arguments of the metrics function. ``version``: bump to invalidate cached rows.
    ``name`` and ``factors`` (dict → DataFrame columns) are labels only and do not enter ``key()``.
    """

    robot: str
    controller: str | None = None
    terrain: dict | str = field(default_factory=lambda: {"kind": "flat"})
    seed: int = 0
    robot_kwargs: dict = field(default_factory=dict)
    controller_kwargs: dict = field(default_factory=dict)
    terrain_kwargs: dict = field(default_factory=dict)
    v_target: float | None = None
    course_m: float | None = None
    duration: float | None = None
    settle: float = 0.5
    rules: dict = field(default_factory=dict)
    disturbances: list = field(default_factory=list)
    lab_kwargs: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)
    metrics_kwargs: dict = field(default_factory=dict)
    version: str = ""
    name: str = ""
    factors: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.robot, str):
            raise TypeError("Trial.robot must be a factory reference string ('file.py:function')")
        if self.controller is not None and not isinstance(self.controller, str):
            raise TypeError("Trial.controller must be a factory reference string or None")
        if not isinstance(self.terrain, (dict, str)):
            if hasattr(self.terrain, "spec"):
                self.terrain = dict(self.terrain.spec())
            else:
                raise TypeError("Trial.terrain must be a spec dict, a Terrain or a factory reference")
        dists = []
        for d in self.disturbances:
            if dataclasses.is_dataclass(d):
                d = dataclasses.asdict(d)
            d = dict(d)
            if "direction" in d:
                d["direction"] = [float(v) for v in d["direction"]]
            dists.append(d)
        self.disturbances = dists
        self.seed = int(self.seed)

    # ---- identity
    def spec(self) -> dict:
        """Every field that can change the result (the hashed part), with the terrain seed resolved."""
        d = {f.name: getattr(self, f.name) for f in dataclasses.fields(self) if f.name not in ("name", "factors")}
        d["terrain"] = self.terrain_spec() if isinstance(self.terrain, dict) else self.terrain
        return d

    def key(self, metrics_fn: str | Callable | None = DEFAULT_METRICS) -> str:
        """The cache key: ``trial_id`` of ``spec()`` plus the metrics function's name."""
        spec = self.spec()
        spec["metrics_fn"] = None if metrics_fn is None else _ref_name(metrics_fn)
        return trial_id(spec)

    def replace(self, **changes) -> "Trial":
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}

    @classmethod
    def from_dict(cls, d: Mapping) -> "Trial":
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown Trial fields {sorted(unknown)}")
        return cls(**d)

    # ---- building blocks
    def terrain_spec(self) -> dict:
        """The terrain spec dict with the trial's seed filled in for seeded kinds (dict terrains only)."""
        if not isinstance(self.terrain, dict):
            raise TypeError("terrain_spec() is defined for spec-dict terrains; this trial uses a factory")
        spec = dict(self.terrain)
        params = spec.get("params")
        has_seed = "seed" in spec or (isinstance(params, Mapping) and "seed" in params)
        if spec.get("kind") in _SEEDED_KINDS and not has_seed:
            spec["seed"] = self.seed
        return spec

    def make_terrain(self):
        from .terrain import Terrain, terrain_from_spec

        if isinstance(self.terrain, dict):
            spec = self.terrain_spec()
            return terrain_from_spec(spec), {k: spec[k] for k in _LEVEL_KEYS if k in spec}
        fn = load_ref(self.terrain)
        kw = dict(self.terrain_kwargs)
        if "seed" not in kw and _accepts(fn, "seed"):
            kw["seed"] = self.seed
        out = fn(**kw)
        if isinstance(out, Terrain):
            extra = {k: kw[k] for k in _LEVEL_KEYS if k in kw and kw[k] is not None}
            return out, extra
        if isinstance(out, Mapping):
            spec = dict(out)
            if spec.get("kind") in _SEEDED_KINDS and "seed" not in spec:
                spec["seed"] = self.seed
            extra = {k: spec[k] for k in _LEVEL_KEYS if k in spec}
            extra.update({k: kw[k] for k in _LEVEL_KEYS if k in kw and kw[k] is not None and k not in extra})
            return terrain_from_spec(spec), extra
        raise TypeError(f"terrain factory {self.terrain} returned {type(out).__name__}, not a Terrain or spec dict")

    def make_robot(self):
        from .robot import Robot

        robot = load_ref(self.robot)(**self.robot_kwargs)
        if not isinstance(robot, Robot):
            raise TypeError(f"robot factory {self.robot} returned {type(robot).__name__}, not a chiron.Robot")
        return robot

    def make_controller(self, robot=None):
        if self.controller is None:
            return _Stand()
        fn = load_ref(self.controller)
        kw = dict(self.controller_kwargs)
        if "v_target" not in kw and self.v_target is not None and _accepts(fn, "v_target"):
            kw["v_target"] = self.v_target
        if "robot" not in kw and robot is not None and _accepts(fn, "robot"):
            kw["robot"] = robot
        ctrl = fn(**kw)
        if not callable(ctrl):
            raise TypeError(f"controller factory {self.controller} returned {type(ctrl).__name__}, which is not "
                            "callable (obs -> Command)")
        return ctrl

    def make_rules(self):
        from .lab import FailureRules

        if self.course_m is None:
            if self.rules:
                raise ValueError("Trial.rules needs course_m (failure rules are defined on a course)")
            if self.duration is None:
                raise ValueError("a trial without course_m needs a duration")
            return None
        return FailureRules(course_m=float(self.course_m), v_target=self.v_target, **self.rules)

    def build(self, **lab_overrides):
        """``(lab, controller, rules, info)`` ready for ``lab.run`` (disturbances already scheduled)."""
        from .lab import ChironLab, Disturbance

        robot = self.make_robot()
        terrain, terrain_extra = self.make_terrain()
        rules = self.make_rules()
        lab = ChironLab(robot, terrain, **{**self.lab_kwargs, **lab_overrides})
        for d in self.disturbances:
            lab.add_disturbance(Disturbance(**d))
        controller = self.make_controller(robot)
        info = dict(self.info)
        if "treatment" not in info and isinstance(getattr(robot, "treatment", None), str):
            info["treatment"] = robot.treatment
        if self.v_target is not None:
            info.setdefault("v_target", float(self.v_target))
        if self.course_m is not None:
            info.setdefault("course_m", float(self.course_m))
        terrain_info = dict(terrain_extra)
        if isinstance(info.get("terrain"), Mapping):
            terrain_info.update(info["terrain"])
        if terrain_info:
            info["terrain"] = terrain_info
        return lab, controller, rules, info

    def run(self, log: bool = True, **lab_overrides):
        """Build and run this trial; returns the ``Episode``. ``lab_overrides`` (e.g. ``log_geoms=True``)
        change only the lab construction, not the hashed spec."""
        return self.run_built(*self.build(**lab_overrides), log=log)

    def run_built(self, lab, controller, rules, info, *, log: bool = True):
        """Run on what ``build()`` returned (lets a caller tell a bad setup from a failed run)."""
        return lab.run(controller, duration=self.duration, rules=rules, settle=self.settle, seed=self.seed,
                       log=log, info=info)

    def flat(self) -> dict:
        """The spec as flat columns (see the module docstring)."""
        return _flatten_spec(self)


class _Stand:
    """Holds the robot's standing pose (``Trial.controller=None``)."""

    name = "stand"

    def reset(self, lab, seed):
        self._cmd = lab.nominal_command()

    def __call__(self, obs):
        return self._cmd


def _flatten_spec(trial: Trial) -> dict:
    out: dict[str, Any] = {"robot_factory": trial.robot}
    _flatten_into(out, "robot_kwargs", trial.robot_kwargs)
    out["controller_factory"] = trial.controller
    _flatten_into(out, "controller_kwargs", trial.controller_kwargs)
    if isinstance(trial.terrain, dict):
        _flatten_into(out, "terrain", trial.terrain_spec())
    else:
        out["terrain_factory"] = trial.terrain
        _flatten_into(out, "terrain_kwargs", trial.terrain_kwargs)
    for k in ("seed", "v_target", "course_m", "duration", "settle"):
        out[k] = getattr(trial, k)
    _flatten_into(out, "rules", trial.rules)
    out["disturbances"] = _dumps(trial.disturbances) if trial.disturbances else ""
    _flatten_into(out, "lab_kwargs", trial.lab_kwargs)
    _flatten_into(out, "info", trial.info)
    _flatten_into(out, "metrics_kwargs", trial.metrics_kwargs)
    out["version"] = trial.version
    return out


def _flatten_into(out: dict, prefix: str, d: Mapping) -> None:
    for k, v in d.items():
        key = f"{prefix}.{k}"
        if isinstance(v, Mapping):
            _flatten_into(out, key, v)
        elif v is None or isinstance(v, (bool, int, float, str)):
            out[key] = v
        else:
            try:
                out[key] = _dumps(v)
            except TypeError:
                out[key] = repr(v)


# ----------------------------------------------------------------------------------------------- the worker
def _plain_value(v):
    if hasattr(v, "item") and hasattr(v, "dtype") and getattr(v, "shape", None) == ():
        v = v.item()
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    try:
        return _canon(v)
    except TypeError:
        return repr(v)


def _same(a, b) -> bool:
    if a is None and b is None:
        return True
    try:
        if isinstance(a, float) or isinstance(b, float):
            fa = float("nan") if a is None else float(a)
            fb = float("nan") if b is None else float(b)
            return (math.isnan(fa) and math.isnan(fb)) or fa == fb
    except (TypeError, ValueError):
        return False
    return a == b


def _merge(row: dict, extra: Mapping) -> None:
    for k, v in extra.items():
        v = _plain_value(v)
        if k in row:
            if _same(row[k], v):
                continue
            k = f"metric.{k}"
        row[k] = v


def _execute(trial: Trial, key: str, metrics_fn, metrics_name: str | None, cache_dir: Path | None,
             log_dir: Path | None) -> dict:
    """Run one trial (in a worker): outcome + metrics row; cached on success."""
    rec: dict[str, Any] = {"status": "ok", "error": "", "log_path": ""}
    t0 = time.perf_counter()
    try:
        ep = trial.run(log=True)
    except Exception as exc:  # noqa: BLE001 - a sweep must survive one broken trial
        rec.update(status="error", error=_error_text(exc), wall_time_s=time.perf_counter() - t0)
        return rec
    o, log, meta = ep.outcome, ep.log, ep.meta
    for k in _OUTCOME_KEYS:
        rec[k] = _plain_value(o.get(k))
    for k in ("robot", "controller", "treatment", "total_mass"):
        rec[k] = _plain_value(log.get(k))
    rec["wall_time_s"] = meta.get("wall_time_s")
    rec["simulated_s"] = meta.get("simulated_s")
    rec["realtime_factor"] = meta.get("realtime_factor")
    if log_dir is not None:
        path = ep.save(Path(log_dir) / f"{key}.npz")
        rec["log_path"] = str(path)
    if metrics_fn is not None:
        try:
            m = metrics_fn(log, o, **trial.metrics_kwargs)
            if not isinstance(m, Mapping):
                raise TypeError(f"metrics function {metrics_name} returned {type(m).__name__}, not a dict")
            _merge(rec, m)
        except Exception as exc:  # noqa: BLE001
            rec.update(status="metrics_error", error=_error_text(exc))
    if cache_dir is not None and rec["status"] == "ok":
        _write_cache(Path(cache_dir), key, trial, metrics_name, rec)
    return rec


def _error_text(exc: BaseException) -> str:
    tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
    tail = "".join(tb[-3:]).strip()
    return f"{type(exc).__name__}: {exc}\n{tail}"


def _cache_file(cache_dir: Path, key: str) -> Path:
    return Path(cache_dir) / "rows" / f"{key}.json"


def _write_cache(cache_dir: Path, key: str, trial: Trial, metrics_name, rec: dict) -> None:
    path = _cache_file(cache_dir, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"format": "chiron.trial_row", "version": CACHE_FORMAT, "trial_id": key, "metrics_fn": metrics_name,
           "spec": _canon(trial.spec()), "name": trial.name, "factors": _canon(trial.factors), "row": rec,
           "written": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(doc, default=_json_fallback, allow_nan=True))
    os.replace(tmp, path)


def _json_fallback(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


def _read_cache(cache_dir: Path, key: str) -> dict | None:
    path = _cache_file(cache_dir, key)
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if doc.get("format") != "chiron.trial_row" or doc.get("trial_id") != key:
        return None
    row = doc.get("row")
    return row if isinstance(row, dict) else None


# pool state: set in the parent before the trials fork, inherited by them (only the rows are pickled)
_POOL: dict = {}


def _pool_work(i: int):
    s = _POOL
    trial, key = s["jobs"][i]
    return i, _execute(trial, key, s["metrics_fn"], s["metrics_name"], s["cache_dir"], s["log_dir"])


def _run_pool(n_jobs: int, n_proc: int, done: Callable) -> None:
    """Run job indices 0..n_jobs-1, each in its own forked process (at most ``n_proc`` at a time); ``done(i, rec)``
    as each finishes. A trial whose process dies (segfault, out of memory, ``os._exit``) gets an error row and
    takes nothing else down; a fresh process per trial also returns all its memory."""
    import multiprocessing as mp
    from multiprocessing.connection import wait

    ctx = mp.get_context("fork")
    pending = list(range(n_jobs))
    active: dict = {}
    try:
        while pending or active:
            while pending and len(active) < n_proc:
                i = pending.pop(0)
                recv, send = ctx.Pipe(duplex=False)
                proc = ctx.Process(target=_isolated_work, args=(i, send), daemon=True)
                proc.start()
                send.close()
                active[recv] = (proc, i)
            for conn in wait(list(active)):
                proc, i = active.pop(conn)
                try:
                    rec = conn.recv()
                except (EOFError, OSError):
                    rec = None
                conn.close()
                proc.join()
                if rec is None:
                    rec = {"status": "error", "log_path": "",
                           "error": f"worker process died (exit code {proc.exitcode}) while running this trial"}
                done(i, rec)
    finally:
        for conn, (proc, _) in active.items():
            proc.terminate()
            proc.join()
            conn.close()


def _isolated_work(i: int, conn) -> None:
    try:
        _, rec = _pool_work(i)
    except BaseException as exc:  # noqa: BLE001 - report anything, then exit
        rec = {"status": "error", "log_path": "", "error": _error_text(exc)}
    try:
        conn.send(rec)
    except Exception as exc:  # noqa: BLE001 - e.g. a value that cannot be pickled
        conn.send({"status": "error", "log_path": "", "error": _error_text(exc)})
    conn.close()


def _preload(jobs) -> None:
    """Import what the workers will need once, in the parent, so every forked trial inherits it."""
    refs = set()
    for t, _ in jobs:
        refs.add(t.robot)
        if t.controller:
            refs.add(t.controller)
        if isinstance(t.terrain, str):
            refs.add(t.terrain)
    for ref in sorted(refs):
        try:
            load_ref(ref)
        except Exception:  # noqa: BLE001 - the trial reports it in its own row
            pass
    try:
        import mujoco  # noqa: F401

        from . import lab, metrics  # noqa: F401
    except ImportError:  # pragma: no cover
        pass


# ----------------------------------------------------------------------------------------------- the runner
def run_trials(trials: Sequence[Trial], processes: int | None = 4, cache_dir=None,
               metrics_fn: str | Callable | None = DEFAULT_METRICS, keep_logs: bool = False, *, log_dir=None,
               rerun: bool = False, progress: bool = True, on_error: str = "record", as_frame: bool = True):
    """Run ``trials`` (cached ones are read, not re-run) and return one row per trial, in input order.

    ``processes``: how many trials run at once, each in its own forked process (None = all CPUs; ≤ 1 runs in
    this process). A trial whose process dies (segfault, out of memory) gets an error row; the others go on.
    ``cache_dir``: where rows (and kept logs) go; None = no cache. ``metrics_fn``: ``'module:function'`` or a
    callable ``fn(log, outcome, **trial.metrics_kwargs) -> dict`` evaluated in the workers (None = outcome only).
    ``keep_logs``: save every episode (``.npz``) to ``log_dir`` (default ``<cache_dir>/logs``); the row's
    ``log_path`` points to it. ``rerun``: ignore and overwrite cached rows. ``on_error``: 'record' (a row with
    ``status='error'``) or 'raise'. Returns a ``pandas.DataFrame`` (``as_frame=False``: a list of dicts) whose
    ``attrs`` hold ``factors`` (the factor column names), ``n_run``, ``n_cached``, ``n_failed`` and ``cache_dir``.
    """
    trials = list(trials)
    if on_error not in ("record", "raise"):
        raise ValueError("on_error must be 'record' or 'raise'")
    for t in trials:
        if not isinstance(t, Trial):
            raise TypeError(f"expected Trial objects, got {type(t).__name__}")
    metrics_name = None if metrics_fn is None else _ref_name(metrics_fn)
    fn = None if metrics_fn is None else (load_ref(metrics_fn) if isinstance(metrics_fn, str) else metrics_fn)
    cache = Path(cache_dir) if cache_dir is not None else None
    if keep_logs:
        if log_dir is None and cache is None:
            raise ValueError("keep_logs needs a cache_dir or a log_dir")
        logs = Path(log_dir) if log_dir is not None else cache / "logs"
        logs.mkdir(parents=True, exist_ok=True)
    else:
        logs = None
    keys = [t.key(metrics_fn) for t in trials]
    results: dict[str, dict] = {}
    cached: set[str] = set()
    if cache is not None and not rerun:
        for k in dict.fromkeys(keys):
            row = _read_cache(cache, k)
            if row is not None and (logs is None or (row.get("log_path") and Path(row["log_path"]).is_file())):
                results[k] = row
                cached.add(k)
    jobs, seen = [], set()
    for t, k in zip(trials, keys):
        if k not in results and k not in seen:
            jobs.append((t, k))
            seen.add(k)
    n_proc = (os.cpu_count() or 1) if processes is None else int(processes)
    bar = _progress(len(jobs), progress)
    try:
        if n_proc <= 1 or len(jobs) <= 1:
            for t, k in jobs:
                rec = _execute(t, k, fn, metrics_name, cache, logs)
                _check(rec, t, on_error)
                results[k] = rec
                bar(rec)
        else:
            def done(i, rec):
                t, k = jobs[i]
                _check(rec, t, on_error)
                results[k] = rec
                bar(rec)

            _preload(jobs)
            _POOL.clear()
            _POOL.update(jobs=jobs, metrics_fn=fn, metrics_name=metrics_name, cache_dir=cache, log_dir=logs)
            try:
                _run_pool(len(jobs), n_proc, done)
            finally:
                _POOL.clear()
    finally:
        bar(None)
    rows = []
    for t, k in zip(trials, keys):
        rec = dict(results[k])
        rec["cached"] = k in cached
        rows.append(_compose(t, k, metrics_name, rec))
    factor_names = list(dict.fromkeys(f for t in trials for f in t.factors))
    stats = {"factors": factor_names, "n_run": len(jobs), "n_cached": len(cached),
             "n_failed": sum(1 for r in rows if r.get("status") != "ok"),
             "cache_dir": str(cache) if cache is not None else None}
    if not as_frame:
        return rows
    import pandas as pd

    df = pd.DataFrame(rows)
    df.attrs.update(stats)
    return df


def _check(rec: dict, trial: Trial, on_error: str) -> None:
    if on_error == "raise" and rec.get("status") != "ok":
        raise RuntimeError(f"trial {trial.name or trial.key()} failed: {rec.get('error')}")


def _compose(trial: Trial, key: str, metrics_name, rec: Mapping) -> dict:
    row: dict[str, Any] = {"trial_id": key, "name": trial.name}
    for k, v in trial.factors.items():
        row[str(k)] = _plain_value(v)
    _merge(row, trial.flat())
    row["metrics_fn"] = metrics_name
    for k in ("status", "error", "cached", "log_path"):
        row[k] = rec.get(k, "" if k != "cached" else False)
    _merge(row, {k: v for k, v in rec.items() if k not in ("status", "error", "cached", "log_path")})
    return row


def _progress(total: int, enabled: bool):
    """A callback: rec → advance the bar; None → close."""
    if not enabled or total == 0:
        return lambda rec: None
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover
        return lambda rec: None
    tqdm.monitor_interval = 0          # no monitor thread: trials fork after the bar exists
    bar = tqdm(total=total, desc="chiron trials", unit="trial")
    counts = {"ok": 0, "failed": 0}

    def step(rec):
        if rec is None:
            bar.close()
            return
        counts["ok" if rec.get("status") == "ok" else "failed"] += 1
        bar.set_postfix(errors=counts["failed"], refresh=False)
        bar.update(1)

    return step


# ----------------------------------------------------------------------------------------------- after the run
def rerun_with_log(trial: Trial, *, cache_dir=None, log_dir=None, log_geoms: bool = False, save=None,
                   metrics_fn: str | Callable | None = DEFAULT_METRICS, **lab_overrides):
    """The trial's full ``Episode`` (for mechanism plots and videos).

    A log kept by ``run_trials(keep_logs=True)`` (in ``log_dir`` or ``<cache_dir>/logs``) is loaded when present
    and has what is asked for; otherwise the trial is simulated again — runs are deterministic, so it is the same
    episode. ``log_geoms=True`` also records every geom's pose (``viz``). ``save``: write the episode there.
    """
    from .lab import Episode

    key = trial.key(metrics_fn)
    folder = Path(log_dir) if log_dir is not None else (Path(cache_dir) / "logs" if cache_dir is not None else None)
    ep = None
    if folder is not None and (folder / f"{key}.npz").is_file():
        ep = Episode.load(folder / f"{key}.npz")
        if log_geoms and "geom_pose" not in ep.log:
            ep = None
    if ep is None:
        if log_geoms:
            lab_overrides["log_geoms"] = True
        ep = trial.run(log=True, **lab_overrides)
    if save is not None:
        ep.save(save)
    return ep


def load_cached(cache_dir, as_frame: bool = True):
    """Every row in a cache directory (labels and spec as recorded when it was written)."""
    rows = []
    for path in sorted((Path(cache_dir) / "rows").glob("*.json")):
        try:
            doc = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if doc.get("format") != "chiron.trial_row":
            continue
        try:
            trial = Trial.from_dict({**_uncanon_spec(doc["spec"]), "name": doc.get("name", ""),
                                     "factors": doc.get("factors") or {}})
        except (TypeError, ValueError, KeyError):
            continue
        rec = dict(doc["row"])
        rec["cached"] = True
        rows.append(_compose(trial, doc["trial_id"], doc.get("metrics_fn"), rec))
    if not as_frame:
        return rows
    import pandas as pd

    return pd.DataFrame(rows)


def _uncanon_spec(spec: Mapping) -> dict:
    d = {k: v for k, v in spec.items() if k != "metrics_fn"}
    return d


def trials_frame(trials: Iterable[Trial], metrics_fn: str | Callable | None = DEFAULT_METRICS):
    """The trial list as a DataFrame (no simulation): ``trial_id``, ``name``, factors and the flat spec."""
    import pandas as pd

    name = None if metrics_fn is None else _ref_name(metrics_fn)
    rows = []
    for t in trials:
        row = {"trial_id": t.key(metrics_fn), "name": t.name}
        row.update({str(k): _plain_value(v) for k, v in t.factors.items()})
        _merge(row, t.flat())
        row["metrics_fn"] = name
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------- paired designs
def paired_seeds(n: int, start: int = 0, *, stream: str | None = None) -> list[int]:
    """``n`` seeds that every configuration of a paired design shares: ``start, start + 1, ...``.

    ``stream`` (a name such as 'tuning' or 'confirmation') moves the block to a reproducible range of its own,
    ``start + 1000 · (hash(stream) mod 100003)``, so seed sets of different experiments do not overlap
    (``n`` ≤ 1000 then)."""
    n = int(n)
    if n < 0:
        raise ValueError("n must be non-negative")
    base = int(start)
    if stream is not None:
        if n > 1000:
            raise ValueError("a named stream holds at most 1000 seeds")
        base += 1000 * (int(hashlib.sha256(str(stream).encode()).hexdigest(), 16) % 100003)
    return list(range(base, base + n))


def factor_grid(factors: Mapping[str, Sequence] | None = None, **more) -> list[dict]:
    """The cross product of factor levels as dicts, in the given order (the last factor varies fastest):
    ``factor_grid({'treatment': ['a', 'b'], 'level': [0.1, 0.2]})`` → 4 cells."""
    fac = dict(factors or {})
    fac.update(more)
    names = list(fac)
    for k, v in fac.items():
        if isinstance(v, (str, bytes)) or not isinstance(v, Iterable):
            raise TypeError(f"factor {k!r} needs a sequence of levels")
    return [dict(zip(names, combo)) for combo in itertools.product(*(list(fac[k]) for k in names))]


def paired_trials(make: Callable[..., Trial], factors: Mapping[str, Sequence] | Sequence[Mapping],
                  seeds: Iterable[int]) -> list[Trial]:
    """Every factor cell on every seed: ``make(**cell, seed=seed) -> Trial``.

    ``factors``: a dict of levels (crossed with ``factor_grid``) or an explicit list of cells. Each trial's
    ``factors`` become ``{**cell, 'seed': seed, **(what make set)}``, so rows of one seed pair across cells.
    Ordered seed-major (all cells of the first seed first)."""
    cells = factor_grid(factors) if isinstance(factors, Mapping) else [dict(c) for c in factors]
    out = []
    for seed in seeds:
        for cell in cells:
            t = make(**cell, seed=seed)
            if not isinstance(t, Trial):
                raise TypeError(f"make() returned {type(t).__name__}, not a Trial")
            out.append(t.replace(factors={**cell, "seed": int(seed), **t.factors}))
    return out


def pair_up(df, factor: str, a, b, value: str, on: Sequence[str] | None = None):
    """Align a paired comparison: rows with ``df[factor] == a`` against ``df[factor] == b``, matched on ``on``
    (default: the other factor columns in ``df.attrs['factors']``, else ``['seed']``).

    Returns a DataFrame with the ``on`` columns and ``value`` of each side in columns ``a`` and ``b`` (only
    complete pairs; e.g. ``stats.mcnemar_exact(p[a], p[b])``)."""
    if on is None:
        fac = [f for f in df.attrs.get("factors", []) if f != factor and f in df.columns]
        on = fac or ["seed"]
    on = list(on)
    left = df.loc[df[factor] == a, on + [value]]
    right = df.loc[df[factor] == b, on + [value]]
    for side, name in ((left, a), (right, b)):
        if side.duplicated(on).any():
            raise ValueError(f"rows with {factor} == {name!r} are not unique on {on}")
    merged = left.merge(right, on=on, how="inner", suffixes=("__a", "__b"))
    merged = merged.rename(columns={f"{value}__a": a, f"{value}__b": b})
    return merged.reset_index(drop=True)
