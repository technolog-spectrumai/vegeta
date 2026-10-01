"""Command line interface: ``chiron info|run|metrics|render``. Robots and controllers are Python factories
(``file.py:function`` or ``module:function``); every physical input is an argument or a factory parameter.

Exit codes: 0 ok, 1 a run (or its metrics / video) failed, 2 invalid input (bad reference, unknown or
incomplete terrain, missing course/duration, a factory that rejects its parameters).
"""
from __future__ import annotations

import argparse
import ast
import inspect
import json
import math
import sys
import time
from pathlib import Path

from .experiments import Trial, load_ref
from .result import Result, _jsonable

#: The terrain parameter a ``KIND:LEVEL`` level sets (SI: metres, or degrees for the slope angle).
LEVEL_PARAM = {"long_bumps": "height", "longitudinal_bumps": "height", "alt_bumps": "height",
               "alternating_bumps": "height", "steps": "height", "rough": "rms", "cross_slope": "angle_deg"}

KEY_METRICS = (
    ("progress_m", "progress", "m"), ("achieved_speed_m_s", "achieved speed", "m/s"),
    ("cot_mech", "cost of transport, mechanical", ""), ("cot_el_est", "cost of transport, electrical (est.)", ""),
    ("worst_tilt_p95_deg", "tilt p95, worst body", "deg"), ("worst_tilt_max_deg", "tilt max, worst body", "deg"),
    ("payload_tilt_p95_deg", "tilt p95, payload", "deg"), ("lateral_max_m", "max |y| of the COM", "m"),
    ("heading_rms_deg", "heading RMS", "deg"), ("slip_per_m", "foot slip per metre", "m/m"),
    ("belly_contact_fraction", "belly contact, time fraction", ""),
    ("support_margin_min_m", "support margin, min", "m"), ("support_outside_fraction", "COM outside support", ""),
    ("infeasible_fraction_dynamic", "contact forces infeasible (dynamic)", ""),
    ("duty_mean", "measured duty factor", ""), ("phase_sd_max_cycles", "interlimb phase SD, max", "cycles"),
    ("impulse_Ns", "push impulse", "N s"), ("recovered", "recovered", ""), ("recovery_time_s", "recovery time", "s"),
)


class InvalidInput(ValueError):
    """Bad command-line input (exit code 2)."""


# ----------------------------------------------------------------------------------------------- parsing helpers
def parse_value(text: str):
    """A command-line value: a Python literal (number, bool, None, list, tuple, dict, quoted string) or text."""
    low = text.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def parse_kv(items, what: str = "parameter") -> dict:
    out = {}
    for item in items or []:
        name, sep, value = item.partition("=")
        if not sep or not name.strip():
            raise InvalidInput(f"{what} expects name=value, got {item!r}")
        out[name.strip()] = parse_value(value)
    return out


def parse_terrain(text: str, params: dict) -> dict:
    """``KIND[:LEVEL]`` plus ``-t`` parameters → a terrain spec dict (LEVEL sets ``LEVEL_PARAM[kind]``)."""
    from .terrain import TERRAIN_KINDS

    kind, _, level_s = text.partition(":")
    kind = kind.strip()
    if kind not in TERRAIN_KINDS:
        raise InvalidInput(f"unknown terrain kind {kind!r}; known: {', '.join(sorted(TERRAIN_KINDS))}")
    spec = {"kind": kind}
    level = None
    if level_s.strip():
        try:
            level = float(level_s)
        except ValueError:
            raise InvalidInput(f"terrain level must be a number, got {level_s!r}") from None
    if kind == "flat":
        if level not in (None, 0.0) or params:
            raise InvalidInput("flat terrain takes no level or parameters")
        return spec
    cls = TERRAIN_KINDS[kind]
    sig = inspect.signature(cls.__init__)
    known = [p for p in sig.parameters if p != "self"]
    unknown = sorted(set(params) - set(known))
    if unknown:
        raise InvalidInput(f"terrain {kind} has no parameter(s) {unknown}; known: {known}")
    spec.update(params)
    if level is not None:
        key = LEVEL_PARAM[kind]
        if key in params and params[key] != level:
            raise InvalidInput(f"terrain level {level} conflicts with -t {key}={params[key]}")
        spec[key] = level
        spec["level"] = level
    required = [p.name for p in sig.parameters.values()
                if p.name != "self" and p.default is inspect.Parameter.empty]
    missing = [p for p in required if p not in spec]
    if missing:
        hint = " ".join(f"-t {m}=..." for m in missing)
        raise InvalidInput(f"terrain {kind} needs {', '.join(missing)} (give {kind}:LEVEL for "
                           f"{LEVEL_PARAM[kind]} and {hint}); nothing is assumed")
    return spec


def parse_push(text: str) -> dict:
    """``BODY:IMPULSE:T_START:DURATION[:DX,DY,DZ]`` (N·s, walking s, s; direction default +y)."""
    parts = text.split(":")
    if len(parts) not in (4, 5) or not parts[0]:
        raise InvalidInput(f"--push expects BODY:IMPULSE:T_START:DURATION[:DX,DY,DZ], got {text!r}")
    try:
        d = {"body": parts[0], "impulse": float(parts[1]), "t_start": float(parts[2]), "duration": float(parts[3])}
        if len(parts) == 5:
            vec = [float(v) for v in parts[4].split(",")]
            if len(vec) != 3:
                raise ValueError
            d["direction"] = vec
    except ValueError:
        raise InvalidInput(f"--push expects numbers: BODY:IMPULSE:T_START:DURATION[:DX,DY,DZ], got {text!r}") from None
    return d


def _robot(ref: str, params: dict):
    from .robot import Robot

    factory = _resolve(ref)
    try:
        robot = factory(**params)
    except TypeError as exc:
        raise InvalidInput(f"{ref}: {exc}") from None
    if not isinstance(robot, Robot):
        raise InvalidInput(f"{ref} returned {type(robot).__name__}, not a chiron.Robot")
    return robot


def _resolve(ref: str):
    try:
        obj = load_ref(ref)
    except (ImportError, AttributeError) as exc:
        raise InvalidInput(f"cannot load {ref}: {exc}") from None
    if not callable(obj):
        raise InvalidInput(f"{ref} is not callable")
    return obj


def _fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return "nan" if math.isnan(v) else f"{v:.4g}"
    return str(v)


# ----------------------------------------------------------------------------------------------- info
def cmd_info(args) -> int:
    robot = _robot(args.robot, parse_kv(args.param))
    summary = robot.summary()
    lab_summary, lab_note = None, ""
    try:
        from .lab import ChironLab

        lab_summary = ChironLab(robot).summary()
    except ImportError as exc:
        lab_note = f"standing pose not computed ({exc})"
    if args.json:
        print(json.dumps(_jsonable({"robot": summary, "lab": lab_summary, "messages": [lab_note] if lab_note else []}),
                         indent=2))
        return 0
    s = summary
    print(f"{s['name']}  ({s['source']} robot, {s['total_mass_kg']:.4g} kg)")
    if "link_mass_kg" in s:
        masses = ", ".join(f"{k} {v:.4g}" for k, v in s["link_mass_kg"].items())
        print(f"  links ({s['n_links']}) [kg]: {masses}")
    print(f"  logged bodies: {', '.join(s['logged_bodies']) or '-'}")
    print(f"  feet ({len(s['feet'])}):")
    for f in s["feet"]:
        print(f"    {f['name']:<16} geom {f['geom']}, on {f['body']}; joints {', '.join(f['joints']) or '-'}")
    servos = {}
    if "joints" in s:
        print(f"  joints ({s['n_joints']}: {s['n_actuated']} actuated, {s['n_passive']} passive):")
        print(f"    {'name':<24} {'tag':<11} {'range [deg]':<17} drive")
        for j in s["joints"]:
            rng = (f"{math.degrees(j['range'][0]):.1f} .. {math.degrees(j['range'][1]):.1f}" if j["range"]
                   else "free")
            if j["servo"]:
                drive = "servo"
                servos.setdefault(json.dumps(j["servo"], sort_keys=True), []).append(j["name"])
            else:
                drive = f"passive k {j['stiffness']:g} N·m/rad, c {j['damping']:g} N·m·s/rad"
            print(f"    {j['name']:<24} {j['tag'] or '-':<11} {rng:<17} {drive}")
    else:
        print(f"  actuated joints ({s['n_actuated']}): {', '.join(s['servos'])}")
        for name, sv in s["servos"].items():
            servos.setdefault(json.dumps(sv, sort_keys=True), []).append(name)
    print(f"  actuators ({len(servos)} servo model{'s' if len(servos) != 1 else ''}):")
    for key, names in servos.items():
        sv = json.loads(key)
        print(f"    stall {sv['stall_torque']:g} N·m, rated {sv['rated_torque']:g} N·m, no-load "
              f"{sv['no_load_speed']:.3g} rad/s, {sv['stall_current']:g} A at {sv['voltage']:g} V; kp {sv['kp']:g} "
              f"N·m/rad, kd {sv['kd']:g} N·m·s/rad, armature {sv['armature']:g} kg·m²; {len(names)} joint(s)"
              + (f" — {sv['source']}" if sv.get("source") else ""))
    if lab_summary:
        print(f"  standing on flat ground: root {lab_summary['nominal_base_height_m']:.4g} m, hips "
              f"{lab_summary['nominal_hip_height_m']:.4g} m; MuJoCo nq {lab_summary['nq']}, nv {lab_summary['nv']}")
    elif lab_note:
        print(f"  {lab_note}")
    for k, v in (s.get("sources") or {}).items():
        print(f"  source — {k}: {v}")
    if s.get("notes"):
        print(f"  notes: {s['notes']}")
    return 0


# ----------------------------------------------------------------------------------------------- run
def trial_from_args(args) -> Trial:
    """The ``Trial`` a ``chiron run`` command line describes."""
    tparams = parse_kv(args.terrain_param, "--terrain-param")
    if args.terrain_factory:
        kind, _, level_s = args.terrain.partition(":")
        level = parse_value(level_s) if level_s else None
        terrain, terrain_kwargs = args.terrain_factory, {"kind": kind, "level": level, **tparams}
    else:
        terrain, terrain_kwargs = parse_terrain(args.terrain, tparams), {}
    rules = parse_kv(args.rule, "--rule")
    if args.course is None and args.duration is None:
        raise InvalidInput("give --course (with --v-target: the failure rules decide the outcome) or --duration")
    if args.course is not None and args.v_target is None and "timeout" not in rules:
        raise InvalidInput("--course needs --v-target (the stall rule and the timeout use it) or -r timeout=S")
    if args.course is None and rules:
        raise InvalidInput("--rule needs --course")
    lab = {"timestep": args.timestep, "control_dt": args.control_dt, "log_dt": args.log_dt}
    lab = {k: v for k, v in lab.items() if v is not None}
    lab.update(parse_kv(args.lab, "--lab"))
    if args.video or args.log_geoms:
        lab["log_geoms"] = True
    return Trial(robot=args.robot, controller=args.controller, terrain=terrain, terrain_kwargs=terrain_kwargs,
                 seed=args.seed, robot_kwargs=parse_kv(args.param), controller_kwargs=parse_kv(args.ctrl_param,
                                                                                                "--ctrl-param"),
                 v_target=args.v_target, course_m=args.course, duration=args.duration, settle=args.settle,
                 rules=rules, disturbances=[parse_push(p) for p in args.push or []], lab_kwargs=lab,
                 info=parse_kv(args.info, "--info"), metrics_kwargs=_metric_options(args))


def _metric_options(args) -> dict:
    opts = parse_kv(getattr(args, "metric_option", None), "--metric-option")
    if getattr(args, "payload", None):
        opts["payload"] = args.payload
    if getattr(args, "feasibility_every", None) is not None:
        opts["feasibility_every"] = args.feasibility_every or None
    return opts


def cmd_run(args) -> int:
    trial = trial_from_args(args)
    out = Path(args.out)
    t0 = time.perf_counter()
    try:
        built = trial.build()
    except (TypeError, KeyError, ValueError, FileNotFoundError, ImportError, AttributeError) as exc:
        raise InvalidInput(f"cannot set up the trial: {type(exc).__name__}: {exc}") from None
    lab = built[0]
    res = Result(kind="chiron.run", metadata={"trial": trial.to_dict(), "trial_id": trial.key(),
                                              "lab": lab.summary(), "command": list(sys.argv)})
    out.mkdir(parents=True, exist_ok=True)
    try:
        ep = trial.run_built(*built, log=True)
    except Exception as exc:  # noqa: BLE001 - report a failed run with exit code 1
        res.fail(f"the run failed: {type(exc).__name__}: {exc}")
        res.duration_s = time.perf_counter() - t0
        res.artifacts["summary"] = out / "summary.json"
        res.save_json(out / "summary.json")
        _print_result(res, args, None)
        return 1
    res.artifacts["episode"] = ep.save(out / "episode.npz")
    base = ep.to_result()
    res.metrics.update(base.metrics)
    res.metadata["episode"] = ep.meta
    res.messages += base.messages
    row = None
    if not args.no_metrics:
        try:
            from .metrics import trial_metrics

            row = trial_metrics(ep.log, ep.outcome, **trial.metrics_kwargs)
            res.metrics.update({k: v for k, v in row.items() if k not in res.metrics})
            res.metadata["metrics_options"] = dict(trial.metrics_kwargs)
        except Exception as exc:  # noqa: BLE001
            res.fail(f"metrics failed: {type(exc).__name__}: {exc}")
    if args.video:
        try:
            from .viz import render

            res.artifacts["video"] = render(ep, out / "episode.mp4", speed=args.speed, camera=args.camera)
        except Exception as exc:  # noqa: BLE001
            res.fail(f"video failed: {type(exc).__name__}: {exc} (headless: run under xvfb-run -a with "
                     "PYVISTA_OFF_SCREEN=true)")
    res.duration_s = time.perf_counter() - t0
    res.artifacts["summary"] = out / "summary.json"
    res.save_json(out / "summary.json")
    _print_result(res, args, ep)
    if not res.ok:
        return 1
    if args.require_success and not ep.success:
        return 1
    return 0


def _print_result(res: Result, args, ep) -> None:
    if args.json:
        print(res.to_json())
        return
    if ep is None:
        print(f"{res.kind}: {res.status.upper()}")
        for m in res.messages:
            print(f"  ! {m}")
        return
    log, o = ep.log, ep.outcome
    terrain = log.get("terrain") or {}
    level = f" {terrain['level']:g}" if isinstance(terrain.get("level"), (int, float)) else ""
    who = log.get("robot", "?") + (f" [{log['treatment']}]" if log.get("treatment") else "")
    print(f"{who} / {log.get('controller', '?')} on {terrain.get('kind', '?')}{level}, seed {log.get('seed')}: "
          f"{str(o.get('reason')).upper()} at t = {o.get('t_end', float('nan')):.2f} s, "
          f"distance {o.get('distance_m', float('nan')):.3f} m")
    if o.get("detail"):
        print(f"  {o['detail']}")
    rt = ep.meta.get("realtime_factor")
    print(f"  simulated {ep.meta.get('simulated_s', 0):.2f} s in {ep.meta.get('wall_time_s', 0):.2f} s"
          + (f" ({rt:.1f}x real time)" if rt else ""))
    for key, label, unit in KEY_METRICS:
        if key in res.metrics:
            print(f"  {label:<38} {_fmt(res.metrics[key])} {unit}".rstrip())
    for k, v in res.artifacts.items():
        print(f"  {k:<38} {v}")
    for m in res.messages:
        if m != o.get("detail"):
            print(f"  ! {m}")


# ----------------------------------------------------------------------------------------------- metrics, render
def _load_episode(path: str):
    from .lab import Episode

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"no such file: {path}")
    try:
        return Episode.load(p)
    except (OSError, ValueError, KeyError) as exc:
        raise InvalidInput(f"{path} is not a Chiron episode (.npz): {exc}") from None


def cmd_metrics(args) -> int:
    from .metrics import trial_result

    ep = _load_episode(args.episode)
    opts = _metric_options(args)
    payload = opts.pop("payload", None)
    every = opts.pop("feasibility_every", 10)
    res = trial_result(ep.log, ep.outcome, payload, every, **opts)
    res.artifacts["episode"] = Path(args.episode)
    if args.out:
        res.artifacts["summary"] = Path(args.out)
        res.save_json(args.out)
    if args.json:
        print(json.dumps(res.to_dict()["metrics"], indent=2))
    else:
        print("\n".join(res.summary_lines()))
    return 0


def cmd_render(args) -> int:
    from .viz import render

    ep = _load_episode(args.episode)
    try:
        w, h = (int(v) for v in args.size.lower().split("x"))
    except ValueError:
        raise InvalidInput(f"--size expects WIDTHxHEIGHT, got {args.size!r}") from None
    try:
        path = render(ep, args.out, speed=args.speed, fps=args.fps, camera=args.camera, size=(w, h))
    except Exception as exc:  # noqa: BLE001
        print(f"render failed: {type(exc).__name__}: {exc} (headless: run under xvfb-run -a with "
              "PYVISTA_OFF_SCREEN=true)", file=sys.stderr)
        return 1
    print(path)
    return 0


# ----------------------------------------------------------------------------------------------- parser
def build_parser(prog: str = "chiron") -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=prog, description="Legged-robot dynamics on MuJoCo: ChironLab episodes, "
                                                        "locomotion metrics (SI units)")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("info", help="masses, bodies, feet, joints and actuators of a robot factory")
    p.add_argument("robot", help="file.py:factory or module:factory returning a chiron.Robot")
    p.add_argument("--param", "-p", action="append", metavar="NAME=VALUE", help="robot factory keyword (repeatable)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_info)

    r = sub.add_parser("run", help="run one episode: writes episode.npz and summary.json")
    r.add_argument("robot", help="file.py:factory or module:factory returning a chiron.Robot")
    r.add_argument("--param", "-p", action="append", metavar="NAME=VALUE", help="robot factory keyword (repeatable)")
    r.add_argument("--controller", help="file.py:factory returning a controller (default: hold the standing pose)")
    r.add_argument("--ctrl-param", "-c", action="append", metavar="NAME=VALUE",
                   help="controller factory keyword (repeatable); v_target is passed when the factory takes it")
    r.add_argument("--terrain", default="flat", metavar="KIND[:LEVEL]",
                   help="flat, long_bumps, alt_bumps, cross_slope, steps or rough; LEVEL sets the height / RMS [m] "
                        "or the angle [deg] (default flat)")
    r.add_argument("--terrain-param", "-t", action="append", metavar="NAME=VALUE",
                   help="terrain parameter in SI, e.g. -t spacing=0.17 -t width=0.085 (repeatable)")
    r.add_argument("--terrain-factory", metavar="FILE.py:FN",
                   help="map KIND[:LEVEL] with your own function fn(kind, level, seed=..., **terrain params)")
    r.add_argument("--seed", type=int, default=0, help="trial seed (terrain randomness and the controller)")
    r.add_argument("--v-target", type=float, help="target speed [m/s] (failure rules; passed to the controller)")
    r.add_argument("--course", type=float, help="course length [m]: the protocol's failure rules decide the outcome")
    r.add_argument("--duration", type=float, help="walking time limit [s] (default: the rules' timeout)")
    r.add_argument("--settle", type=float, default=0.5, help="settle time before walking [s] (default 0.5)")
    r.add_argument("--rule", "-r", action="append", metavar="NAME=VALUE",
                   help="FailureRules override, e.g. -r max_tilt_deg=45 -r timeout=10 (repeatable)")
    r.add_argument("--push", action="append", metavar="BODY:IMPULSE:T_START:DURATION[:DX,DY,DZ]",
                   help="impulse [N s] on a body's COM at walking time T_START [s] over DURATION [s], direction "
                        "world (default +y) (repeatable)")
    r.add_argument("--timestep", type=float, help="physics step [s] (default 0.001)")
    r.add_argument("--control-dt", type=float, help="controller period [s] (default 0.001)")
    r.add_argument("--log-dt", type=float, help="log period [s] (default 0.01)")
    r.add_argument("--lab", "-l", action="append", metavar="NAME=VALUE",
                   help="other ChironLab keyword, e.g. -l flat_as_plane=False (repeatable)")
    r.add_argument("--info", action="append", metavar="NAME=VALUE",
                   help="bookkeeping written into the log, e.g. --info treatment=spring (repeatable)")
    r.add_argument("--payload", help="payload body for the metrics (e.g. head)")
    r.add_argument("--feasibility-every", type=int, help="contact-force LP every n-th sample (0 = skip; default 10)")
    r.add_argument("--metric-option", "-m", action="append", metavar="NAME=VALUE",
                   help="trial_metrics keyword, e.g. -m load_fraction=0.02 (repeatable)")
    r.add_argument("--no-metrics", action="store_true", help="skip the metrics")
    r.add_argument("--log-geoms", action="store_true", help="log every geom's pose (for chiron render)")
    r.add_argument("--video", action="store_true", help="also render episode.mp4 (pyvista; headless: xvfb-run -a)")
    r.add_argument("--camera", default="follow", help="video camera: follow, side, front, top, iso")
    r.add_argument("--speed", type=float, default=1.0, help="video playback speed")
    r.add_argument("--require-success", action="store_true",
                   help="exit 1 unless the outcome is success (default: any outcome of a completed run is exit 0)")
    r.add_argument("--out", "-o", default="chiron_run", help="output directory (default chiron_run)")
    r.add_argument("--json", action="store_true", help="print summary.json")
    r.set_defaults(func=cmd_run)

    m = sub.add_parser("metrics", help="locomotion metrics of a saved episode (vegeta.chiron.metrics)")
    m.add_argument("episode", help="episode .npz written by chiron run or Episode.save")
    m.add_argument("--payload", help="payload body (e.g. head)")
    m.add_argument("--feasibility-every", type=int, help="contact-force LP every n-th sample (0 = skip; default 10)")
    m.add_argument("--metric-option", "-m", action="append", metavar="NAME=VALUE", help="trial_metrics keyword")
    m.add_argument("--out", "-o", help="also write the metrics as a Result JSON")
    m.add_argument("--json", action="store_true", help="print the metrics as JSON")
    m.set_defaults(func=cmd_metrics)

    v = sub.add_parser("render", help="render a saved episode to a video (pyvista; headless: xvfb-run -a)")
    v.add_argument("episode", help="episode .npz (record with --log-geoms for the full geometry)")
    v.add_argument("--out", "-o", default="episode.mp4")
    v.add_argument("--camera", default="follow", help="follow, side, front, top, iso")
    v.add_argument("--speed", type=float, default=1.0)
    v.add_argument("--fps", type=int, default=25)
    v.add_argument("--size", default="960x540", help="WIDTHxHEIGHT")
    v.set_defaults(func=cmd_render)
    return ap


def main(argv=None, prog: str = "chiron") -> int:
    args = build_parser(prog).parse_args(argv)
    try:
        return args.func(args)
    except (InvalidInput, ValueError, FileNotFoundError) as exc:
        print(f"{prog}: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
