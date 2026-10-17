"""batch — a batch of chapter runs from one ``batch.yaml``, in a strict order, from the command line.

The desktop interface (or an editor) writes the file; ``wf batch batch.yaml`` runs it — on this machine or on a
cloud box that only has the repository, the solvers and a shell. The steps run one after the other, each as the
chapter's own script with an explicit ``--out`` run directory, so the results land where the file says they will;
the batch keeps its own folder under ``workflows/runs/_batches/<stamp>_<name>/`` with the YAML it ran, ``batch.json``
(every step: state, exit code, run directory, summary file), ``status.json`` and ``run.log`` (the GUI follows it like
a run). The last lines of the terminal say where every output file is.

    name: kestrel night                 # a label; the batch folder and the run directories carry it
    stop_on_failure: true               # a failed step stops the batch (default); false = go on, report at the end
    steps:
      - chapter: K1                     # a chapter code (K1, k2 …) or book/chapter
        variant: 4Q-2
        options: {quality: fine, processors: 8}      # the script's own options, without the dashes
        limits: --max-minutes 120       # anything else the script takes, verbatim
      - chapter: K11
        variant: 4Q-2
        options: {element-size: 2, threads: 4}
        mode: run                       # run (default) | prepare (cases written, no solver)
        cache: true                     # the book's case cache (default true)
        ignore_prerequisites: false

``wf batch --example`` prints this file; ``wf batch file.yaml --dry-run`` prints the commands only; ``--from 3`` starts
at the third step. PyYAML reads the file when it is installed; without it a small reader handles this layout.
"""
from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import library  # noqa: E402
import runkit  # noqa: E402

ROOT = library.ROOT
EXIT_TEXT = {0: "completed", 3: "partial", 1: "failed", 4: "nothing solved", 2: "did not start (prerequisite or option)", 130: "cancelled"}

EXAMPLE = """\
# batch.yaml — a batch of Kestrel chapters, run in this order by:  wf batch batch.yaml
# (write it here or from the Workflows window: Batch → Save batch.yaml; run it on any machine with the solvers)
name: kestrel night
stop_on_failure: true
steps:
  - chapter: K1              # airframe-sweep
    variant: 4Q-2
    options: {quality: fine, processors: 4}
    limits: --max-minutes 180
  - chapter: K2              # airframe-stall
    variant: 4Q-2
    options: {quality: fine, processors: 4}
  - chapter: K3              # airframe-cruise (after K1)
    variant: 4Q-2
    options: {quality: screening, processors: 4}
  - chapter: K11             # frame-and-joint (FEA)
    variant: 4Q-2
    options: {element-size: 3, threads: 2}
"""


# ----------------------------------------------------------------------------------------------- the file
@dataclass
class Step:
    chapter: str
    variant: str = ""
    options: dict = field(default_factory=dict)
    mode: str = "run"                   # run | prepare
    cache: bool = True
    limits: str = ""
    ignore_prerequisites: bool = False

    def as_dict(self) -> dict:
        d = {"chapter": self.chapter}
        if self.variant:
            d["variant"] = self.variant
        if self.options:
            d["options"] = dict(self.options)
        if self.mode != "run":
            d["mode"] = self.mode
        if not self.cache:
            d["cache"] = False
        if self.limits:
            d["limits"] = self.limits
        if self.ignore_prerequisites:
            d["ignore_prerequisites"] = True
        return d


@dataclass
class Batch:
    name: str = "batch"
    steps: list = field(default_factory=list)
    stop_on_failure: bool = True
    notes: str = ""

    def as_dict(self) -> dict:
        d = {"name": self.name, "stop_on_failure": self.stop_on_failure}
        if self.notes:
            d["notes"] = self.notes
        d["steps"] = [s.as_dict() for s in self.steps]
        return d


def _bool(v) -> bool:
    return v if isinstance(v, bool) else str(v).strip().lower() in ("1", "true", "yes", "on")


def from_dict(d: dict) -> Batch:
    if not isinstance(d, dict) or not isinstance(d.get("steps"), list):
        raise ValueError("a batch file is a mapping with a 'steps' list (wf batch --example)")
    steps = []
    for i, s in enumerate(d["steps"], 1):
        if isinstance(s, str):
            s = {"chapter": s}
        if not isinstance(s, dict) or not s.get("chapter"):
            raise ValueError(f"step {i}: needs a 'chapter' (a code such as K1)")
        opts = s.get("options") or {}
        if not isinstance(opts, dict):
            raise ValueError(f"step {i}: 'options' is a mapping such as {{quality: fine}}")
        mode = str(s.get("mode", "run")).strip().lower()
        if mode not in ("run", "prepare"):
            raise ValueError(f"step {i}: mode is 'run' or 'prepare', not {mode!r}")
        steps.append(Step(str(s["chapter"]).strip(), str(s.get("variant", "") or ""), {str(k).lstrip("-"): v for k, v in opts.items()}, mode,
                          _bool(s.get("cache", True)), str(s.get("limits", "") or ""), _bool(s.get("ignore_prerequisites", False))))
    return Batch(str(d.get("name", "batch")), steps, _bool(d.get("stop_on_failure", True)), str(d.get("notes", "") or ""))


def dumps(batch: Batch) -> str:
    """The batch as YAML (a fixed, readable layout — no PyYAML needed to write it)."""
    def scalar(v) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return str(v)
        s = str(v)
        return s if s and not any(c in s for c in ":#{}[],&*!|>'\"%@`\n") and s.strip() == s and s.lower() not in ("true", "false", "yes", "no", "null", "on", "off") else json.dumps(s)
    lines = [f"# batch.yaml — {len(batch.steps)} chapter(s), run in this order by:  wf batch <this file>", f"name: {scalar(batch.name)}",
             f"stop_on_failure: {scalar(batch.stop_on_failure)}"]
    if batch.notes:
        lines.append(f"notes: {scalar(batch.notes)}")
    lines.append("steps:")
    for s in batch.steps:
        d = s.as_dict()
        lines.append(f"  - chapter: {scalar(d['chapter'])}")
        for k, v in d.items():
            if k == "chapter":
                continue
            if k == "options":
                lines.append("    options: {" + ", ".join(f"{kk}: {scalar(vv)}" for kk, vv in v.items()) + "}")
            else:
                lines.append(f"    {k}: {scalar(v)}")
    return "\n".join(lines) + "\n"


def loads(text: str) -> Batch:
    try:
        import yaml
        data = yaml.safe_load(text)
    except ImportError:
        data = _mini_yaml(text)
    return from_dict(data)


def load(path) -> Batch:
    return loads(Path(path).read_text())


def save(batch: Batch, path) -> Path:
    p = Path(path)
    p.write_text(dumps(batch))
    return p


def _mini_yaml(text: str):
    """The subset this module writes: mappings and lists of mappings by indentation, flow mappings ``{a: b}``, scalars."""
    def val(s: str):
        s = s.strip()
        if s.startswith("{") and s.endswith("}"):
            inner = s[1:-1].strip()
            return {k.strip(): val(v) for k, v in (kv.split(":", 1) for kv in _split_commas(inner))} if inner else {}
        if s.startswith("[") and s.endswith("]"):
            return [val(x) for x in _split_commas(s[1:-1])] if s[1:-1].strip() else []
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            return json.loads(s) if s[0] == '"' else s[1:-1]
        low = s.lower()
        if low in ("true", "yes", "on"):
            return True
        if low in ("false", "no", "off"):
            return False
        if low in ("null", "~", ""):
            return None
        try:
            return int(s)
        except ValueError:
            try:
                return float(s)
            except ValueError:
                return s

    lines = []
    for raw in text.splitlines():
        s = raw.split(" #", 1)[0] if not raw.lstrip().startswith("#") else ""
        s = s.rstrip()
        if s.strip():
            lines.append((len(s) - len(s.lstrip()), s.strip()))

    def parse(i: int, indent: int):
        if i >= len(lines):
            return None, i
        if lines[i][1].startswith("- "):
            out = []
            while i < len(lines) and lines[i][0] == indent and lines[i][1].startswith("- "):
                ind, s = lines[i]
                body = s[2:]
                if ":" in body and not body.startswith(("{", "[", '"', "'")):
                    lines[i] = (indent + 2, body)                            # the first key of a list item, as its own line
                    item, i = parse(i, indent + 2)
                else:
                    item, i = val(body), i + 1
                out.append(item)
            return out, i
        out = {}
        while i < len(lines) and lines[i][0] == indent and not lines[i][1].startswith("- "):
            ind, s = lines[i]
            k, _, rest = s.partition(":")
            if rest.strip():
                out[k.strip()] = val(rest)
                i += 1
            else:
                i += 1
                if i < len(lines) and lines[i][0] > indent:
                    out[k.strip()], i = parse(i, lines[i][0])
                else:
                    out[k.strip()] = None
        return out, i

    data, _ = parse(0, lines[0][0] if lines else 0)
    return data


def _split_commas(s: str) -> list[str]:
    out, depth, cur = [], 0, ""
    for c in s:
        if c in "{[":
            depth += 1
        elif c in "}]":
            depth -= 1
        if c == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += c
    if cur.strip():
        out.append(cur)
    return out


# ----------------------------------------------------------------------------------------------- resolving a step
def resolve(step: Step) -> tuple[str, str, dict]:
    return library.by_code(step.chapter)


def argv_for(step: Step, out_dir: Path | None = None) -> list[str]:
    """The step's command (without the interpreter): the script, ``--chapter``, the options, the mode, ``--out``."""
    book, name, info = resolve(step)
    argv = [info["script"], "--chapter", info["campaign"]]
    if step.variant:
        argv += ["--variant", step.variant]
    for k, v in step.options.items():
        if v not in (None, ""):
            argv += [f"--{k}", str(v)]
    if step.mode == "prepare":
        argv.append("--prepare-only")
    if not step.cache:
        argv.append("--no-cache")
    if step.ignore_prerequisites:
        argv.append("--ignore-prerequisites")
    if step.limits.strip():
        argv += shlex.split(step.limits)
    if out_dir is not None:
        argv += ["--out", str(out_dir)]
    return argv


def validate(batch: Batch) -> list[str]:
    """Problems that would stop the batch before it starts: unknown chapters, variants, modes."""
    problems = []
    if not batch.steps:
        problems.append("no steps")
    for i, s in enumerate(batch.steps, 1):
        try:
            book, name, info = resolve(s)
        except KeyError as exc:
            problems.append(f"step {i}: {exc.args[0]}")
            continue
        variants = library.BOOKS[book].get("variants") or ()
        if s.variant and variants and s.variant not in variants:
            problems.append(f"step {i} ({info['code']}): variant {s.variant!r} is not one of {', '.join(variants)}")
        kind = library.KINDS[info["kind"]]
        known = {o[0].lstrip("-") for o in kind.get("options", [])}
        for k in s.options:
            if known and k not in known and k not in ("processors", "threads", "quality", "element-size", "n-modes"):
                problems.append(f"step {i} ({info['code']}): option --{k} is not one the {info['kind']} script lists ({', '.join('--' + o for o in sorted(known))})")
    return problems


# ----------------------------------------------------------------------------------------------- running
def batch_dir(batch: Batch, base: Path | None = None) -> Path:
    base = Path(base) if base else library.RUNS / "_batches"
    return base / f"{runkit.stamp()}_{runkit.slug(batch.name)}"


def run(batch: Batch, *, source: Path | None = None, dry_run: bool = False, start: int = 1, log=print, launch=None, folder: Path | None = None,
        python: str | None = None) -> dict:
    """Run the steps from ``start`` (1-based) in order. ``launch(cmd, cwd, log_file) -> returncode`` replaces the
    subprocess in tests. Returns the report also written to ``<batch dir>/batch.json``; its ``exit_code`` is 0 when
    every step completed, 3 when some did (partial / stopped), 1 when one failed and nothing else finished, 2 for
    a bad file."""
    problems = validate(batch)
    if problems:
        for p in problems:
            log(f"batch: {p}")
        return {"exit_code": 2, "problems": problems, "steps": []}
    py = python or sys.executable
    plan = []
    for i, s in enumerate(batch.steps, 1):
        book, name, info = resolve(s)
        plan.append((i, s, book, name, info))
    if dry_run:
        log(f"batch {batch.name!r}: {len(plan)} step(s), in this order" + (f", starting at {start}" if start > 1 else "") + " (--dry-run: nothing started)")
        for i, s, book, name, info in plan:
            mark = "  " if i >= start else "··"
            log(f"{mark}{i:2d}. {info['code']:4s} {name:22s} {s.variant or '':6s} $ " + shlex.join([".venv/bin/python", *argv_for(s)]) + "  --out <batch run dir>")
        return {"exit_code": 0, "dry_run": True, "steps": [{"n": i, "chapter": info["code"], "variant": s.variant, "command": shlex.join(argv_for(s))} for i, s, _, _, info in plan]}

    bdir = folder or batch_dir(batch)
    bdir.mkdir(parents=True, exist_ok=True)
    (bdir / "batch.yaml").write_text(dumps(batch) if source is None else Path(source).read_text())
    report = {"schema": 1, "name": batch.name, "batch_dir": str(bdir), "source": str(source) if source else None, "started_at": runkit.stamp(),
              "stop_on_failure": batch.stop_on_failure, "host": os.uname().nodename, "steps": []}
    cancel = {"flag": False}
    t_all = time.time()
    rlog = (bdir / "run.log").open("a", buffering=1)

    def say(msg: str) -> None:
        log(msg)
        rlog.write(time.strftime("%H:%M:%S ") + msg + "\n")

    def status(state: str, **kw) -> None:
        done = sum(1 for r in report["steps"] if r["state"] == "completed")
        failed = sum(1 for r in report["steps"] if r["state"] in ("failed", "nothing solved", "did not start (prerequisite or option)"))
        runkit.write_json_atomic(bdir / "status.json", {"schema": 1, "state": state, "pid": os.getpid(), "heartbeat": runkit.stamp(), "batch": batch.name,
                                                      "cases": {"n": len(plan), "done": done + sum(1 for r in report["steps"] if r["state"] == "partial"), "failed": failed,
                                                                "solved": done, "reused": 0}, **kw})
        runkit.write_json_atomic(bdir / "batch.json", report)

    def on_signal(signum, _frame):
        cancel["flag"] = True
        say(f"batch: signal {signum} — the current step is told to stop; the batch ends after it")
        if current.get("proc") is not None:
            try:
                current["proc"].send_signal(signal.SIGINT)
            except OSError:
                pass

    current: dict = {"proc": None}
    old = {s: signal.signal(s, on_signal) for s in (signal.SIGINT, signal.SIGTERM)} if threading_main() else {}
    say(f"batch {batch.name!r}: {len(plan)} step(s), folder {_rel(bdir)}" + (f", starting at step {start}" if start > 1 else ""))
    status("running", step=None)
    try:
        for i, s, book, name, info in plan:
            rec = {"n": i, "chapter": info["code"], "name": name, "variant": s.variant, "state": "skipped" if i < start else "planned", "exit_code": None,
                   "run_dir": None, "summary": None, "seconds": 0.0, "command": ""}
            report["steps"].append(rec)
            if i < start:
                continue
            if cancel["flag"]:
                rec["state"] = "cancelled"
                continue
            prev_failed = any(r["state"] in ("failed", "nothing solved", "did not start (prerequisite or option)", "cancelled") for r in report["steps"][:-1])
            if prev_failed and batch.stop_on_failure:
                rec["state"] = "not started (an earlier step failed)"
                continue
            out_dir = library.RUNS / book / name / f"{bdir.name}_s{i}{('_' + runkit.slug(s.variant)) if s.variant else ''}"
            k = 2
            while out_dir.exists():                                      # the same batch folder run twice (tests, --from)
                out_dir = out_dir.with_name(f"{bdir.name}_s{i}{('_' + runkit.slug(s.variant)) if s.variant else ''}_{k}")
                k += 1
            argv = argv_for(s, out_dir)
            rec["command"] = shlex.join([".venv/bin/python", *argv])
            rec["run_dir"] = str(out_dir)
            ns = library.needs_status(book, name)
            if not ns["ok"] and s.mode != "prepare":
                rec["state"], rec["exit_code"] = "did not start (prerequisite or option)", 2
                say(f"step {i} {info['code']} {name}: needs {', '.join(ns['missing'])} on this machine — not found; " + "; ".join(dict.fromkeys(ns["fixes"])))
                status("running", step=i)
                continue
            if not s.ignore_prerequisites and s.variant:
                missing = library.check_prerequisites(book, name, variant=s.variant)
                if missing:
                    rec["state"], rec["exit_code"] = "did not start (prerequisite or option)", 2
                    say(f"step {i} {info['code']} {name}: prerequisite missing — " + " | ".join(missing))
                    status("running", step=i)
                    continue
            say(f"step {i}/{len(plan)}: {info['code']} {name} {s.variant}\n  $ {rec['command']}")
            rec["state"] = "running"
            status("running", step=i)
            t0 = time.time()
            with (bdir / f"step_{i:02d}_{info['code']}.out").open("ab") as f:
                if launch is not None:
                    rc = launch([py, str(ROOT / argv[0]), *argv[1:]], str(ROOT), f)
                else:
                    proc = subprocess.Popen([py, str(ROOT / argv[0]), *argv[1:]], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                    current["proc"] = proc
                    for line in iter(proc.stdout.readline, b""):
                        f.write(line)
                        f.flush()
                        txt = line.decode(errors="replace").rstrip("\n")
                        if "\r" in txt:                                  # a progress bar: keep the last state only
                            txt = txt.rsplit("\r", 1)[-1]
                        if txt.strip():
                            log("  │ " + txt)
                    rc = proc.wait()
                    current["proc"] = None
            rec["seconds"] = round(time.time() - t0, 1)
            rec["exit_code"] = rc
            rec["state"] = EXIT_TEXT.get(rc, f"exit {rc}")
            if out_dir.is_dir():
                summ = out_dir / "summary.json" if (out_dir / "summary.json").exists() else (out_dir / "summary.partial.json" if (out_dir / "summary.partial.json").exists() else None)
                rec["summary"] = str(summ) if summ else None
            else:
                rec["run_dir"] = None
            say(f"step {i} {info['code']}: {rec['state']} in {rec['seconds'] / 60:.1f} min" + (f" → {_rel(rec['summary'])}" if rec["summary"] else ""))
            status("running", step=i)
    finally:
        for sg, h in old.items():
            signal.signal(sg, h)
    states = [r["state"] for r in report["steps"] if r["state"] != "skipped"]
    n_ok = states.count("completed")
    n_part = states.count("partial")
    n_bad = sum(1 for st in states if st not in ("completed", "partial"))
    code = 0 if n_bad == 0 and n_part == 0 else (3 if (n_ok or n_part) else 1)
    if cancel["flag"]:
        code = 130
    report["exit_code"], report["finished_at"], report["minutes"] = code, runkit.stamp(), round((time.time() - t_all) / 60, 1)
    final = "completed" if code == 0 else ("cancelled" if code == 130 else ("partial" if code == 3 else "failed"))
    status(final, step=None)
    say("")
    say(f"batch {batch.name!r} {final}: {n_ok} of {len(states)} step(s) completed" + (f", {n_part} partial" if n_part else "") + (f", {n_bad} failed or not started" if n_bad else "")
        + f" ({report['minutes']} min)")
    for r in report["steps"]:
        mark = "✔" if r["state"] == "completed" else ("◐" if r["state"] == "partial" else ("·" if r["state"] in ("skipped",) else "✘"))
        where = _rel(r["summary"]) if r["summary"] else (_rel(r["run_dir"]) + "  (no summary)" if r["run_dir"] and Path(r["run_dir"]).is_dir() else "")
        say(f"  {mark} {r['n']:2d}. {r['chapter']:4s} {r['name']:22s} {r['variant'] or '':6s} {r['state']:36s} {where}")
    say(f"results: {_rel(bdir / 'batch.json')}   (every step: its run directory and summary file)")
    say(f"         {_rel(bdir)}/step_NN_*.out hold each script's terminal output; the runs: wf runs")
    rlog.close()
    return report


def threading_main() -> bool:
    import threading
    return threading.current_thread() is threading.main_thread()


def _rel(p) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT.resolve()))
    except (ValueError, TypeError):
        return str(p)
