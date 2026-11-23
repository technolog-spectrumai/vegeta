"""VEGETA_NO_SOLVERS=1 (the mock-only test runs) refuses every external solver before it starts: CalculiX, OpenFOAM,
PrusaSlicer, XFOIL. A fake ``ccx`` on the PATH proves nothing was spawned: it would leave a marker file."""
import os
import stat

import pytest


@pytest.fixture
def guarded(tmp_path, monkeypatch):
    marker = tmp_path / "SPAWNED"
    bindir = tmp_path / "bin"; bindir.mkdir()
    for name in ("ccx", "simpleFoam", "blockMesh", "prusa-slicer", "xfoil"):
        exe = bindir / name
        exe.write_text(f"#!/bin/sh\necho {name} >> {marker}\n")
        exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("VEGETA_NO_SOLVERS", "1")
    return marker


def test_calculix_is_refused(guarded, tmp_path):
    from vegeta.talos import ccx, _process
    with pytest.raises(RuntimeError, match="solver blocked"):
        ccx.ccx_version("ccx")
    with pytest.raises(RuntimeError, match="solver blocked"):
        ccx.run_ccx(tmp_path, "job")
    with pytest.raises(RuntimeError, match="solver blocked"):
        _process.run_command(["ccx", "-i", "job"], tmp_path)
    assert not guarded.exists()


def test_openfoam_is_neither_found_nor_started(guarded, tmp_path):
    from vegeta.aeromant import _process
    from vegeta.aeromant.environment import OpenFOAMEnvironment
    assert OpenFOAMEnvironment.candidates() == []
    with pytest.raises(RuntimeError):
        OpenFOAMEnvironment.detect()
    with pytest.raises(RuntimeError, match="solver blocked"):
        _process.run_command(["simpleFoam"], tmp_path)
    assert not guarded.exists()


def test_slicer_and_xfoil_are_refused(guarded, tmp_path):
    from vegeta.mellonia import _process
    from vegeta.zephyr import polar
    with pytest.raises(RuntimeError, match="solver blocked"):
        _process.run_command(["prusa-slicer", "--help"], tmp_path)
    assert polar.backends()["xfoil"] is None
    assert not guarded.exists()
