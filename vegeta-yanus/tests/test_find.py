"""yanus.find: the newest usable run or an explicit one, every refusal with its reason and the command, never a
substitute and never a computation."""
import json
import os
import subprocess

import pytest

from conftest import make_run
from vegeta import yanus

FP = {"v": 1}


def test_latest_compatible_run_is_chosen_and_newer_unusable_ones_are_named(repo):
    make_run(repo.root, "sweep", "r1", created="2026-10-01T00:00:00Z")
    make_run(repo.root, "sweep", "r2", created="2026-10-02T00:00:00Z", design={"v": 2})                 # incompatible
    make_run(repo.root, "sweep", "r3", created="2026-10-03T00:00:00Z", state="failed", summary=False)
    s = yanus.find("D1", variant="A", fingerprint=FP)
    assert s.ok and s.run.run_id == "r1" and s.compatibility == "compatible" and s.summary["counts"]["solved"] == 2
    assert [rid for rid, _ in s.skipped] == ["r3", "r2"]
    assert "failed" in s.skipped[0][1] and "incompatible: v differs" in s.skipped[1][1]
    assert "latest run r1" in s.text() and s.require() == s.summary


def test_chapter_by_code_or_ref_and_variant(repo):
    make_run(repo.root, "sweep", "a1", variant="A")
    make_run(repo.root, "sweep", "b1", variant="B", created="2026-10-18T00:00:00Z")
    assert yanus.find("demo/sweep", variant="A", fingerprint=FP).run.run_id == "a1"
    assert yanus.find("d1", variant="B", fingerprint=FP).run.run_id == "b1"
    assert not yanus.find("D2", variant="A").ok                                    # another chapter's runs are not its
    with pytest.raises(KeyError):
        yanus.find("D9")


def test_incompatible_inputs_are_refused_with_the_reason(repo):
    make_run(repo.root, "sweep", "r1", design={"v": 2})
    s = yanus.find("D1", variant="A", fingerprint=FP)
    assert not s.ok and s.summary is None and "incompatible: v differs" in s.skipped[0][1]
    with pytest.raises(yanus.RunUnusable, match=r"wf D1 --variant A"):
        s.require()


def test_warnings_are_accepted_and_shown(repo):
    make_run(repo.root, "sweep", "r1", design={"v": 3})
    s = yanus.find("D1", variant="A", fingerprint={"v": 1, "warn": 3})
    assert s.ok and s.compatibility == "warnings" and s.compat_reasons == ["sources changed"] and "sources changed" in s.text()


def test_unknown_compatibility_stays_visibly_unverified(repo):
    make_run(repo.root, "sweep", "r1", design={"v": 2})
    s = yanus.find("D1", variant="A")                                   # no fingerprint: not checked, not called compatible
    assert s.ok and s.compatibility == "unverified" and s.compat_reasons == ["no fingerprint given"]
    assert "unverified" in s.text() and s.provenance()["compatibility"] == "unverified"
    repo.books["demo"].compatible = None
    s = yanus.find("D1", variant="A", fingerprint=FP)
    assert s.ok and s.compatibility == "unverified" and "no compatibility check" in s.compat_reasons[0]


def test_quality_must_match_and_be_recorded(repo):
    make_run(repo.root, "sweep", "screen", quality="screening", created="2026-10-03T00:00:00Z")
    make_run(repo.root, "sweep", "noq", quality=None, created="2026-10-02T00:00:00Z")
    make_run(repo.root, "sweep", "fine", quality="fine", created="2026-10-01T00:00:00Z")
    s = yanus.find("D1", variant="A", fingerprint=FP, quality="fine")
    assert s.ok and s.run.run_id == "fine"
    assert dict(s.skipped) == {"screen": "quality screening, fine requested", "noq": "the run records no quality, fine requested"}
    assert yanus.find("D1", variant="A", fingerprint=FP, quality=("screening", "fine")).run.run_id == "screen"
    s = yanus.find("D1", variant="A", fingerprint=FP, quality="extreme")
    assert not s.ok and s.command == "wf D1 --variant A --quality extreme"
    assert yanus.find("D3", variant="A", quality="fine").command == "wf D3 --variant A"     # FEA takes no --quality


@pytest.mark.parametrize("kw, why", [
    (dict(state="failed", summary=False), "run state 'failed'"),
    (dict(state="partial", summary=False), "run state 'partial'"),
    (dict(state="running", summary=False, pid=999999999), "its process is gone"),
    (dict(summary=False), "summary.json is missing"),
    (dict(summary="{not json"), "summary.json is not readable JSON"),
    (dict(schema=7), "schema 7"),
    (dict(workflow="wfea"), "a wfea run, not wcfd"),
    (dict(variant="B"), "a run of B, not A"),
])
def test_failed_and_malformed_runs_are_refused(repo, kw, why):
    d = make_run(repo.root, "sweep", "bad", **kw)
    s = yanus.find("D1", variant="A", fingerprint=FP, run=d)
    assert not s.ok and why in "; ".join(s.reasons)
    assert why in s.message() and "wf D1 --variant A" in s.message()


def test_a_malformed_manifest_is_reported_not_skipped_silently(repo):
    d = make_run(repo.root, "sweep", "broken")
    (d / "manifest.json").write_text("{oops")
    s = yanus.find("D1", variant="A", fingerprint=FP)
    assert not s.ok and s.skipped == [("broken", f"{d}: manifest.json is not readable JSON")]


def test_explicit_run_is_checked_and_never_replaced(repo):
    good = make_run(repo.root, "sweep", "good", created="2026-10-02T00:00:00Z")
    old = make_run(repo.root, "sweep", "old", design={"v": 2}, created="2026-10-01T00:00:00Z")
    s = yanus.find("D1", variant="A", fingerprint=FP, run="workflows/runs/demo/sweep/good")    # relative to the repository
    assert s.ok and s.explicit and s.root == "workflows/runs/demo/sweep/good" and "explicit run good" in s.text()
    s = yanus.find("D1", variant="A", fingerprint=FP, run=old)
    assert not s.ok and s.run.run_id == "old" and s.skipped == []                     # the usable 'good' is NOT chosen instead
    assert "the run " in s.message() and "incompatible" in s.message()
    s = yanus.find("D1", variant="A", run=repo.root / "nowhere")
    assert not s.ok and s.reasons[0].startswith("not found") and s.command == "wf D1 --variant A"
    s = yanus.find("D1", variant="A", run=repo.root / "workflows/runs/demo/sweep/legacy_2026")   # an import's run, gone
    assert not s.ok and s.command.startswith("wf import_old")
    assert good.exists()


def test_mock_runs_are_never_engineering_results(repo, monkeypatch):
    m = make_run(repo.root, "sweep", "mock", mock=True, created="2026-10-03T00:00:00Z")
    make_run(repo.root, "sweep", "dbg", quality="debug", created="2026-10-02T00:00:00Z")
    s = yanus.find("D1", variant="A", fingerprint=FP)
    assert not s.ok and all("made under the mocks" in why for _, why in s.skipped)
    s = yanus.find("D1", variant="A", fingerprint=FP, run=m)
    assert not s.ok and "made under the mocks" in s.reasons[0]
    real = make_run(repo.root, "sweep", "real", mock=False, created="2026-10-01T00:00:00Z")
    assert yanus.find("D1", variant="A", fingerprint=FP).run.root == real
    monkeypatch.setenv("VEGETA_WORKFLOW_RUNS", str(repo.root / "workflows" / "runs-mock"))    # the mocks' isolated folder
    s = yanus.find("D1", variant="A", fingerprint=FP)
    assert not s.ok and "mocks'" in s.reasons[0]


def test_nothing_on_disk(repo):
    s = yanus.find("D1", variant="A", fingerprint=FP)
    assert not s.ok and s.reasons == ["no run under workflows/runs/demo/sweep for A"] and s.skipped == []
    with pytest.raises(yanus.RunUnusable) as exc:
        s.require()
    assert "produce one with:  wf D1 --variant A" in str(exc.value) and "nothing is run from here" in str(exc.value)


def test_provenance_records_the_choice(repo):
    make_run(repo.root, "sweep", "r1")
    p = yanus.find("D1", variant="A", fingerprint=FP, quality="fine").provenance()
    assert p["request"] == {"chapter": "demo/sweep", "code": "D1", "variant": "A", "quality": ["fine"], "run": None, "mode": "latest"}
    assert p["usable"] and p["run"] == "workflows/runs/demo/sweep/r1" and p["run_id"] == "r1" and p["quality"] == "fine"
    assert p["compatibility"] == "compatible" and p["git"] == "abc123" and p["command"] is None
    json.dumps(p)                                                                 # goes into a JSON report as it is
    q = yanus.find("D1", variant="B").provenance()
    assert not q["usable"] and q["run"] is None and q["command"] == "wf D1 --variant B" and q["reasons"]


def test_table(repo):
    make_run(repo.root, "sweep", "r1")
    t = yanus.table({"a": yanus.find("D1", variant="A", fingerprint=FP), "b": yanus.find("D1", variant="B")})
    assert list(t.index) == ["D1 sweep, A", "D1 sweep, B"] and t.loc["D1 sweep, B", "command"] == "wf D1 --variant B"


def test_finding_and_checking_never_start_a_computation(repo, monkeypatch, capsys):
    """Every way a process could start is a trap: find, require, the table, the CLI and a script's entry check pass."""
    def trap(*a, **k):
        raise AssertionError(f"a process was started: {a[:1]}")
    for name in ("Popen", "run", "call", "check_call", "check_output"):
        monkeypatch.setattr(subprocess, name, trap)
    for name in ("system", "popen", "fork", "execv", "execve", "execvp", "spawnv", "posix_spawn"):
        if hasattr(os, name):
            monkeypatch.setattr(os, name, trap)
    d = make_run(repo.root, "sweep", "r1")
    make_run(repo.root, "sweep", "r0", state="running", summary=False, pid=999999999, created="2026-10-18T00:00:00Z")
    before = sorted(p.relative_to(repo.root) for p in repo.root.rglob("*"))
    s = yanus.find("D1", variant="A", fingerprint=FP)
    assert s.ok and s.require()
    assert not yanus.find("D2", variant="A", fingerprint=FP).ok                  # missing: reported, not produced
    yanus.find("D1", variant="A", run=d).provenance()
    yanus.table([s])
    from vegeta.yanus import cli
    assert cli.main(["find", "D1", "--variant", "A"]) == 0 and cli.main(["find", "D2", "--variant", "A"]) == 2
    with pytest.raises(SystemExit):
        yanus.check_chapter("D2", variant="B", expected=FP)                       # its prerequisite D1 has no run for B: exit, nothing started
    assert yanus.check_chapter("D2", variant="A", expected=FP)["missing"] == []
    assert sorted(p.relative_to(repo.root) for p in repo.root.rglob("*")) == before          # nothing written
