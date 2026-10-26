"""The Triton backend: populations in a PyTorch tensor, one fused kernel launch per time step.

Runs on CUDA. Without CUDA it runs only under Triton's interpreter (``TRITON_INTERPRET=1`` set before
``triton`` is imported), on CPU tensors: slow, meant for testing the kernels on machines without a GPU.
"""
from __future__ import annotations

import os

import numpy as np

from ..boundaries import BoundaryConditions, Kind
from ..collision import BGK, CollisionModel, RegularizedBGK
from ..errors import BackendUnavailableError, InsufficientMemoryError, InvalidParameterError
from ..frames import RotatingFrame
from ..lattice import D3Q19
from ..turbulence import WALE
from .base import Backend

try:
    import torch
    import triton
except ImportError as exc:   # pragma: no cover - depends on the installation
    raise BackendUnavailableError("the triton backend needs PyTorch and Triton (pip install 'mancha[gpu]')") from exc

from . import triton_kernels as K

# collision kind -> (fused kernel, its REGULARIZED switch); TRT/MRT/cumulant add entries here
KERNELS = {BGK.kind: (K.stream_collide, False), RegularizedBGK.kind: (K.stream_collide, True)}

DEFAULT_BLOCK = 128      # one node per thread with 4 warps: 44–80 registers on sm_80, no spills
INT32_LIMIT = 2 ** 31 - 1
MEMORY_HEADROOM = 0.9    # use at most this fraction of the free device memory


def interpreter_active() -> bool:
    return os.environ.get("TRITON_INTERPRET", "0") not in ("", "0")


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if interpreter_active():
        return "cpu"
    raise BackendUnavailableError("the triton backend needs a CUDA GPU (or TRITON_INTERPRET=1 for the slow CPU "
                                  "interpreter); use backend='numpy' otherwise")


def required_bytes(shape: tuple[int, int, int], q: int = 19, with_solid: bool = True, les: bool = False) -> int:
    """Device memory of the solver state: two population buffers, the link mask, the macroscopic buffer and,
    with LES, two velocity buffers."""
    n = int(np.prod(shape))
    return 2 * q * n * 4 + (4 * n if with_solid else 0) + 4 * n * 4 + (2 * 3 * n * 4 if les else 0)


class TritonBackend(Backend):
    name = "triton"

    def __init__(self, lattice: D3Q19, shape: tuple[int, int, int], collision: CollisionModel,
                 dtype: np.dtype = np.float32, rest_density: float = 1.0,
                 force: tuple[float, float, float] | None = None, frame: RotatingFrame | None = None,
                 turbulence: WALE | None = None, block_size: int | None = None, device: str | None = None):
        if np.dtype(dtype) != np.float32:
            raise InvalidParameterError("the triton backend computes in float32 only")
        if collision.kind not in KERNELS:
            raise InvalidParameterError(f"the triton backend has no kernel for collision model {collision.kind!r}")
        super().__init__(lattice, shape, collision, dtype, rest_density, force, frame, turbulence)
        self._kernel, self._regularized = KERNELS[collision.kind]
        self.device = torch.device(device or default_device())
        if self.device.type == "cpu" and not interpreter_active():
            raise BackendUnavailableError("CPU tensors need TRITON_INTERPRET=1")
        self.block = int(block_size or DEFAULT_BLOCK)
        if self.block < 16 or self.block & (self.block - 1):
            raise InvalidParameterError(f"block_size must be a power of two >= 16, got {self.block}")
        self.n = int(np.prod(shape))
        self._int64 = lattice.q * self.n > INT32_LIMIT
        self._check_memory(required_bytes(shape, lattice.q, les=self.turbulence is not None))
        try:
            self.f = torch.empty((lattice.q,) + shape, dtype=torch.float32, device=self.device)
            self._buf = torch.empty_like(self.f)
            self._macro = torch.empty((4,) + shape, dtype=torch.float32, device=self.device)
            if self.turbulence is not None:       # previous / new step velocity for the WALE gradient
                self._u_prev = torch.zeros((3,) + shape, dtype=torch.float32, device=self.device)
                self._u_new = torch.zeros_like(self._u_prev)
            else:
                self._u_prev = self._u_new = torch.zeros(1, dtype=torch.float32, device=self.device)
        except torch.OutOfMemoryError as exc:     # the estimate was optimistic (fragmentation, other processes)
            self.f = self._buf = self._macro = None
            raise InsufficientMemoryError(f"out of device memory allocating a {shape} grid: {exc}") from exc
        self._links = torch.zeros(1, dtype=torch.int32, device=self.device)
        self._faces = torch.zeros(24, dtype=torch.float32, device=self.device)
        self.configure(BoundaryConditions(), None)

    def _check_memory(self, need: int) -> None:
        if self.device.type != "cuda":
            return
        free, total = torch.cuda.mem_get_info(self.device)
        if need > MEMORY_HEADROOM * free:
            raise InsufficientMemoryError(
                f"a {self.shape} grid needs about {need / 2**30:.2f} GiB on {torch.cuda.get_device_name(self.device)}; "
                f"{free / 2**30:.2f} of {total / 2**30:.2f} GiB are free")

    def configure(self, boundaries: BoundaryConditions, solid: np.ndarray | None) -> None:
        boundaries.validate()
        codes, vel, rho = boundaries.table()
        self._codes = tuple(int(k) for k in codes)
        table = np.zeros((6, 4), dtype=np.float32)
        table[:, :3], table[:, 3] = vel, rho
        self._faces = torch.from_numpy(table.reshape(-1)).to(self.device)
        self._has_faces = any(k != Kind.PERIODIC for k in self._codes)
        self._has_pressure = any(k == Kind.PRESSURE for k in self._codes)
        self.solid = None if solid is None or not solid.any() else solid.astype(bool)
        if self.solid is not None:
            self._links = torch.from_numpy(link_mask(self.solid, self.lattice.c)).to(self.device)
        else:
            self._links = torch.zeros(1, dtype=torch.int32, device=self.device)
        self._sync_velocity()

    def set_populations(self, f: np.ndarray) -> None:
        self.f.copy_(torch.from_numpy(np.ascontiguousarray(f, dtype=np.float32)))
        self._sync_velocity()

    def _sync_velocity(self) -> None:
        """LES: the 'previous step' velocity is that of the current state (as the NumPy backend does)."""
        if self.turbulence is not None:
            self._u_prev.copy_(self._macroscopic()[1:])

    def populations(self) -> np.ndarray:
        return self.f.cpu().numpy()

    def _grid(self) -> tuple[int]:
        return (triton.cdiv(self.n, self.block),)

    def _frame_args(self) -> tuple[float, float, float]:
        if self.frame is None:
            return 0.0, 0.0, 0.0
        return float(self.frame.omega), float(self.frame.center[0]), float(self.frame.center[1])

    def step(self, steps: int = 1) -> None:
        fx, fy, fz = (float(v) for v in (self.force if self.force is not None else (0.0, 0.0, 0.0)))
        om, yc, zc = self._frame_args()
        k0, k1, k2, k3, k4, k5 = self._codes
        omega = 1.0 / self.collision.tau
        nu0 = float(self.collision.viscosity)
        cw2 = float(self.turbulence.c_w ** 2) if self.turbulence is not None else 0.0
        nx, ny, nz = self.shape
        grid = self._grid()
        for _ in range(steps):
            self._kernel[grid](
                self.f, self._buf, self._links, self._faces, self._u_prev, self._u_new, nx, ny, nz, self.n, omega,
                nu0, cw2, self.rest_density, fx, fy, fz, om, yc, zc, K0=k0, K1=k1, K2=k2, K3=k3, K4=k4, K5=k5,
                HAS_SOLID=self.solid is not None, HAS_FACES=self._has_faces, HAS_PRESSURE=self._has_pressure,
                HAS_FORCE=self.force is not None or self.frame is not None, HAS_FRAME=self.frame is not None,
                HAS_LES=self.turbulence is not None, REGULARIZED=self._regularized, INT64=self._int64,
                BLOCK=self.block)
            self.f, self._buf = self._buf, self.f
            if self.turbulence is not None:
                self._u_prev, self._u_new = self._u_new, self._u_prev

    def _macroscopic(self) -> torch.Tensor:
        fx, fy, fz = (float(v) for v in (self.force if self.force is not None else (0.0, 0.0, 0.0)))
        om, yc, zc = self._frame_args()
        K.macroscopic[self._grid()](self.f, self._links, self._macro, self.shape[1], self.shape[2], self.n, fx, fy,
                                    fz, om, yc, zc, HAS_SOLID=self.solid is not None,
                                    HAS_FRAME=self.frame is not None, INT64=self._int64, BLOCK=self.block)
        return self._macro

    def density(self) -> np.ndarray:
        return self._macroscopic()[0].cpu().numpy()

    def velocity(self) -> np.ndarray:
        return self._macroscopic()[1:].cpu().numpy()

    def health(self) -> tuple[bool, float]:
        m = self._macroscopic()
        finite = bool(torch.isfinite(self.f).all().item())
        if not finite:
            return False, float("nan")
        return True, float(torch.sqrt((m[1:] * m[1:]).sum(dim=0)).max().item())

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def memory_bytes(self) -> int:
        return required_bytes(self.shape, self.lattice.q, self.solid is not None, self.turbulence is not None)


def link_mask(solid: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Per node: bit ``i`` when the source ``x − c_i`` is solid (periodic wrap; faces override in the kernel),
    bit 19 when the node itself is solid."""
    bits = np.zeros(solid.shape, dtype=np.int32)
    for i, ci in enumerate(c):
        if ci.any():
            bits |= np.roll(solid, shift=tuple(ci), axis=(0, 1, 2)).astype(np.int32) << i
    bits |= solid.astype(np.int32) << K.SOLID_BIT.value
    return np.ascontiguousarray(bits.reshape(-1))
