"""The OpenFOAM comparison tooling (benchmarks/openfoam/cube_in_duct.py): force integration, recirculation length,
and — when OpenFOAM is installed — a coarse end-to-end OpenFOAM run."""
import importlib.util
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "openfoam" / "cube_in_duct.py"
spec = importlib.util.spec_from_file_location("cube_in_duct", SCRIPT)
C = importlib.util.module_from_spec(spec)
sys.modules["cube_in_duct"] = C
spec.loader.exec_module(C)

N = 4
SHAPE = (C.LENGTH * N, C.HEIGHT * N, C.WIDTH * N)


def test_uniform_pressure_pushes_down_on_the_top_only():
    """A uniform gauge pressure p loads the five wetted faces: x and z cancel, the top gives −p h² in y."""
    P = np.full(SHAPE, 2.0)
    U = np.zeros((3,) + SHAPE)
    f = C.cube_force(U, P, N, nu=0.1)
    assert np.allclose(f, (0.0, -2.0, 0.0))


def test_uniform_stream_shears_the_cube():
    """u = 1 next to every face: shear ν u /(Δ/2) over the top and the two sides (3 h²) acts along +x; on the
    up- and downstream faces u is normal, which gives no shear."""
    U = np.zeros((3,) + SHAPE)
    U[0] = 1.0
    f = C.cube_force(U, np.zeros(SHAPE), N, nu=0.1)
    assert np.isclose(f[0], 3 * 0.1 / (0.5 / N)) and np.allclose(f[1:], 0.0)


def test_recirculation_length():
    U = np.zeros((3,) + SHAPE)
    U[0] = 1.0
    start = (C.CUBE_X0 + 1) * N
    U[0, start:start + 6, 0, int(1.5 * N)] = -0.5        # nodes 0..5 behind the cube reversed
    U[0, start + 6, 0, int(1.5 * N)] = 0.5                # crossing halfway between nodes 5 and 6
    assert np.isclose(C.recirculation_length(U, N), (5 + 0.5 + 0.5) / N)
    U[0, start, 0, int(1.5 * N)] = 0.1                    # a corner vortex at the base does not change it
    assert np.isclose(C.recirculation_length(U, N), (5 + 0.5 + 0.5) / N)
    U[0, start:start + 6, 0, int(1.5 * N)] = 0.2
    assert C.recirculation_length(U, N) == 0.0


def test_block_mesh_dict_leaves_out_the_cube():
    text = C.block_mesh_dict(N)
    assert text.count("hex (") == 17                      # 3 × 2 × 3 blocks minus the cube
    assert "cube { type wall; faces (" in text


BASHRC = os.environ.get("MANCHA_OPENFOAM_BASHRC", "/usr/share/openfoam/etc/bashrc")


@pytest.mark.slow
@pytest.mark.skipif(not (Path(BASHRC).exists() or shutil.which("simpleFoam")), reason="OpenFOAM not installed")
def test_openfoam_runs_the_case(tmp_path):
    f = C.run_openfoam(N, 40.0, BASHRC, tmp_path, 2000, 1)
    assert f["converged"] and f["cells"] == np.prod(SHAPE) - N ** 3
    assert np.isnan(f["p"]).sum() == N ** 3                # exactly the cube's cells are empty
    assert 2.0 < f["cd"] < 6.0 and abs(f["cz"]) < 1e-6      # symmetric in z
