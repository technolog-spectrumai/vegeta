"""The Triton kernels compile for real NVIDIA targets (sm_80, sm_90) without a GPU: catches code the interpreter
accepts but the compiler rejects, and register spills."""
import inspect
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

triton = pytest.importorskip("triton")
pytest.importorskip("torch")
if os.environ.get("TRITON_INTERPRET", "0") not in ("", "0"):
    pytest.skip("kernels are interpreted functions under TRITON_INTERPRET; nothing to compile",
                allow_module_level=True)
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.compiler import ASTSource  # noqa: E402

from mancha.backends import triton_kernels as K  # noqa: E402

STEP_TYPES = dict(f_ptr="*fp32", out_ptr="*fp32", links_ptr="*i32", faces_ptr="*fp32", nx="i32", ny="i32",
                  nz="i32", N="i32", omega="fp32", rho_ref="fp32", fx="fp32", fy="fp32", fz="fp32")
CONFIGS = {
    "periodic": dict(K0=0, K1=0, K2=0, K3=0, K4=0, K5=0, HAS_SOLID=False, HAS_FACES=False, HAS_PRESSURE=False,
                     HAS_FORCE=False, INT64=False, BLOCK=128),
    "everything": dict(K0=2, K1=1, K2=4, K3=3, K4=1, K5=2, HAS_SOLID=True, HAS_FACES=True, HAS_PRESSURE=True,
                       HAS_FORCE=True, INT64=False, BLOCK=128),
    "int64": dict(K0=2, K1=1, K2=4, K3=4, K4=4, K5=4, HAS_SOLID=True, HAS_FACES=True, HAS_PRESSURE=True,
                  HAS_FORCE=False, INT64=True, BLOCK=256),
}


def _compile(fn, types, consts, arch):
    names = list(inspect.signature(fn.fn).parameters)
    src = ASTSource(fn=fn, signature={n: types.get(n, "constexpr") for n in names},
                    constexprs={(names.index(k),): v for k, v in consts.items()})
    return triton.compile(src, target=GPUTarget("cuda", arch, 32))


@pytest.mark.parametrize("arch", [80, 90])
@pytest.mark.parametrize("config", sorted(CONFIGS))
def test_stream_collide_compiles(config, arch, tmp_path):
    kernel = _compile(K.stream_collide_bgk, STEP_TYPES, CONFIGS[config], arch)
    ptx = kernel.asm["ptx"]
    assert ".entry stream_collide_bgk" in ptx
    ptxas = Path(triton.__file__).parent / "backends" / "nvidia" / "bin" / "ptxas"
    if not ptxas.exists():
        ptxas = shutil.which("ptxas")
    if ptxas:
        (tmp_path / "k.ptx").write_text(ptx)
        target = re.search(r"^\.target\s+(sm_\w+)", ptx, re.M).group(1)    # sm_90 compiles as sm_90a
        info = subprocess.run([str(ptxas), f"-arch={target}", "-v", str(tmp_path / "k.ptx"), "-o", "/dev/null"],
                              capture_output=True, text=True, check=True).stderr
        assert "0 bytes spill stores" in info, info


def test_macroscopic_compiles():
    types = dict(f_ptr="*fp32", links_ptr="*i32", out_ptr="*fp32", N="i32", fx="fp32", fy="fp32", fz="fp32")
    kernel = _compile(K.macroscopic, types, dict(HAS_SOLID=True, INT64=False, BLOCK=128), 80)
    assert ".entry macroscopic" in kernel.asm["ptx"]
