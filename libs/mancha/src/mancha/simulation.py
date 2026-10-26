"""The public entry point: a D3Q19 simulation on a box, in lattice units, on a chosen backend."""
from __future__ import annotations

import json
import math
import platform
import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .backends import Backend, create_backend
from .backends.numpy_backend import rest_populations
from .boundaries import FACES, BoundaryConditions, Kind, validate_mask
from .collision import BGK, CollisionModel, from_parameters, tau_from_viscosity
from .diagnostics import kinetic_energy, momentum_exchange_force, total_mass
from .errors import InvalidParameterError, NumericalInstabilityError
from .lattice import D3Q19
from .units import CS2, UnitConverter

MACH_WARN = 0.1                  # lattice speed above which compressibility errors (≈ Ma²) exceed ~3 %
SOUND_SPEED = math.sqrt(CS2)     # a fluid node faster than this means the solution has blown up


@dataclass(frozen=True)
class Record:
    """One line of the monitored history (lattice units)."""

    step: int
    mass: float
    kinetic_energy: float
    max_speed: float


class Simulation:
    """A D3Q19 lattice Boltzmann simulation on an ``nx × ny × nz`` box.

    Everything is in lattice units: ``dx = dt = 1``, reference density ``density`` (default 1). Give the
    viscosity either as ``viscosity`` (lattice ``nu``) or ``tau`` (BGK relaxation time), or pass a
    ``collision`` model. ``force`` is a uniform body-force density (Guo forcing). ``units`` (a
    :class:`~mancha.units.UnitConverter`) is optional and only used to report physical quantities. Every face
    starts periodic; change it with :meth:`set_boundary`.

    The state is checked every ``check_interval`` steps (and after every :meth:`run`): non-finite populations or
    a node faster than the lattice speed of sound raise :class:`~mancha.errors.NumericalInstabilityError`.
    """

    def __init__(self, lattice: D3Q19 | None = None, shape: Sequence[int] = (32, 32, 32),
                 viscosity: float | None = None, *, tau: float | None = None,
                 collision: CollisionModel | None = None, backend: str = "numpy", dtype: Any = None,
                 density: float = 1.0, velocity: Sequence[float] = (0.0, 0.0, 0.0),
                 force: Sequence[float] | None = None, units: UnitConverter | None = None,
                 block_size: int | None = None, device: str | None = None,
                 check_interval: int = 100, history_interval: int | None = None):
        self.lattice = lattice or D3Q19()
        if not isinstance(self.lattice, D3Q19):
            raise InvalidParameterError("only the D3Q19 lattice is implemented")
        shape = tuple(int(n) for n in shape)
        if len(shape) != 3 or min(shape) < 1:
            raise InvalidParameterError(f"shape must be three positive sizes, got {shape}")
        self.shape: tuple[int, int, int] = shape  # type: ignore[assignment]
        self.collision = _collision(viscosity, tau, collision)
        if check_interval < 1:
            raise InvalidParameterError("check_interval must be at least 1")
        if not (math.isfinite(density) and density > 0):
            raise InvalidParameterError(f"the reference density must be positive, got {density!r}")
        self.check_interval = int(check_interval)
        self.history_interval = history_interval
        self.reference_density = float(density)
        self.force = None if force is None else tuple(float(x) for x in force)
        if self.force is not None and (len(self.force) != 3 or not all(map(math.isfinite, self.force))):
            raise InvalidParameterError(f"force must be three finite numbers, got {force!r}")
        self.units = units
        self.backend_name = backend
        if dtype is None:
            dtype = np.float64 if backend == "numpy" else np.float32
        self.dtype = np.dtype(dtype)
        kwargs: dict[str, Any] = dict(lattice=self.lattice, shape=self.shape, collision=self.collision,
                                      dtype=self.dtype, rest_density=self.reference_density,
                                      force=self.force)
        if backend == "triton":
            kwargs.update(block_size=block_size, device=device)
        elif block_size is not None or device is not None:
            raise InvalidParameterError("block_size and device apply to the triton backend only")
        self.backend: Backend = create_backend(backend, **kwargs)
        self.boundaries = BoundaryConditions()
        self.solid: np.ndarray | None = None
        self.steps_done = 0
        self.history: list[Record] = []
        self.wall_time = 0.0
        self._dirty = False
        self.initialize(density=self.reference_density, velocity=velocity)

    # -- set-up --------------------------------------------------------------------------------------------------
    def set_boundary(self, face: str, kind: str, **params: Any) -> "Simulation":
        """Set a face condition: ``"periodic"``, ``"wall"`` (``velocity=`` for a moving lid),
        ``"velocity"`` (``velocity=``) or ``"pressure"`` (``density=``). Values in lattice units."""
        self.boundaries.set(face, kind, **params)
        _check_speed(self.boundaries[face].velocity, f"the {face} velocity")
        self._dirty = True
        return self

    def set_solid(self, mask: np.ndarray | None) -> "Simulation":
        """Static solid voxels (``True`` = solid), replacing any previous mask; ``None`` removes them."""
        self.solid = None if mask is None else validate_mask(mask, self.shape)
        if self.solid is not None:
            f = self.backend.populations()
            f[:, self.solid] = (self.lattice.w * self.reference_density)[:, None]
            self.backend.set_populations(f)
        self._dirty = True
        return self

    def initialize(self, density: float | np.ndarray = 1.0, velocity: Sequence[float] | np.ndarray = (0, 0, 0)
                   ) -> "Simulation":
        """Set the populations to the equilibrium of the given density and velocity (scalars or fields
        ``[nx, ny, nz]`` / ``[3, nx, ny, nz]``). Resets the step counter and the history."""
        rho = np.broadcast_to(np.asarray(density, dtype=np.float64), self.shape)
        u = np.asarray(velocity, dtype=np.float64)
        u = np.broadcast_to(u.reshape(3, 1, 1, 1) if u.ndim == 1 else u, (3,) + self.shape)
        if not (np.isfinite(rho).all() and np.isfinite(u).all()) or (rho <= 0).any():
            raise InvalidParameterError("the initial density must be positive and the fields finite")
        _check_speed(float(np.sqrt((u * u).sum(axis=0)).max()), "the initial velocity")
        if self.force is not None:   # the stored state carries ρu + F/2 (Guo), so the reported velocity is u
            u = u + np.asarray(self.force).reshape(3, 1, 1, 1) / (2.0 * rho)
        f = rest_populations(self.lattice, rho, u, self.dtype)
        if self.solid is not None:
            f[:, self.solid] = (self.lattice.w * self.reference_density)[:, None]
        self.backend.set_populations(f)
        self.steps_done = 0
        self.history = []
        return self

    # -- running -------------------------------------------------------------------------------------------------
    def run(self, steps: int, callback: Callable[["Simulation"], None] | None = None,
            callback_interval: int | None = None) -> "Simulation":
        """Advance ``steps`` time steps. ``callback(sim)`` runs every ``callback_interval`` steps."""
        steps = int(steps)
        if steps < 0:
            raise InvalidParameterError("steps must be non-negative")
        self._configure()
        chunk = self.check_interval
        for every in (callback_interval if callback else None, self.history_interval):
            if every:
                chunk = math.gcd(chunk, int(every))
        if self.history_interval and self.steps_done == 0 and not self.history:
            self._record()
        t0 = time.perf_counter()
        done = 0
        while done < steps:
            n = min(chunk, steps - done)
            self.backend.step(n)
            done += n
            self.steps_done += n
            if self.steps_done % self.check_interval == 0 or done == steps:
                self._check()
            if self.history_interval and self.steps_done % self.history_interval == 0:
                self._record()
            if callback and callback_interval and self.steps_done % callback_interval == 0:
                callback(self)
        self.backend.synchronize()
        self.wall_time += time.perf_counter() - t0
        return self

    def _configure(self) -> None:
        if self._dirty:
            self.boundaries.validate()
            self.backend.configure(self.boundaries, self.solid)
            self._dirty = False

    def _check(self) -> None:
        finite, umax = self.backend.health()
        if not finite:
            raise NumericalInstabilityError(f"non-finite populations by step {self.steps_done} "
                                            f"(tau = {getattr(self.collision, 'tau', '?')}); reduce the velocity "
                                            "or raise the viscosity / resolution")
        if umax > SOUND_SPEED:
            raise NumericalInstabilityError(f"max |u| = {umax:.3g} exceeds the lattice speed of sound "
                                            f"{SOUND_SPEED:.3f} at step {self.steps_done}: the solution is unstable")

    def _record(self) -> None:
        rho, u = self.density(), self.velocity()
        speed = np.sqrt((u * u).sum(axis=0))
        self.history.append(Record(self.steps_done, self.mass(rho), self.kinetic_energy(rho, u),
                                   float(speed.max())))

    # -- fields (host NumPy arrays, lattice units) ---------------------------------------------------------------
    def density(self) -> np.ndarray:
        """Density ``[nx, ny, nz]``."""
        return self.backend.density()

    def velocity(self) -> np.ndarray:
        """Velocity ``[3, nx, ny, nz]``; zero in solid voxels."""
        return self.backend.velocity()

    def pressure(self) -> np.ndarray:
        """Gauge pressure ``cs² (ρ − ρ_ref)`` in lattice units."""
        return CS2 * (self.density() - self.reference_density)

    def populations(self) -> np.ndarray:
        return self.backend.populations()

    def mass(self, rho: np.ndarray | None = None) -> float:
        return total_mass(self.density() if rho is None else rho, self.solid)

    def kinetic_energy(self, rho: np.ndarray | None = None, u: np.ndarray | None = None) -> float:
        return kinetic_energy(self.density() if rho is None else rho, self.velocity() if u is None else u,
                              self.solid)

    def solid_force(self) -> np.ndarray:
        """Force on the solid voxels by momentum exchange (lattice units; :meth:`UnitConverter.force` converts)."""
        if self.solid is None:
            raise InvalidParameterError("no solid mask is set")
        periodic = tuple(self.boundaries.faces[2 * a].kind == Kind.PERIODIC for a in range(3))
        return momentum_exchange_force(self.populations(), self.solid, self.lattice.c, periodic)

    @property
    def viscosity(self) -> float:
        return self.collision.viscosity

    @property
    def time(self) -> float | None:
        """Simulated physical time [s] when ``units`` is set."""
        return None if self.units is None else self.units.time(self.steps_done)

    # -- output --------------------------------------------------------------------------------------------------
    def metadata(self) -> dict[str, Any]:
        """Everything needed to reproduce the run (JSON-serialisable)."""
        return {
            "mancha_version": __version__,
            "lattice": self.lattice.name,
            "shape": list(self.shape),
            "collision": self.collision.parameters(),
            "viscosity_lattice": self.collision.viscosity,
            "reference_density": self.reference_density,
            "force": None if self.force is None else list(self.force),
            "boundaries": self.boundaries.to_dict(),
            "solid_voxels": 0 if self.solid is None else int(self.solid.sum()),
            "backend": self.backend_name,
            "dtype": self.dtype.name,
            "steps": self.steps_done,
            "check_interval": self.check_interval,
            "history_interval": self.history_interval,
            "units": None if self.units is None else self.units.to_dict(),
            "wall_time_s": self.wall_time,
            "platform": {"python": platform.python_version(), "numpy": np.__version__, "machine": platform.machine()},
            "units_note": "fields are in lattice units (dx = dt = 1)",
        }

    def save(self, directory: str | Path, vtk: bool = True) -> Path:
        """Write ``density.npy``, ``velocity.npy`` (``[3, nx, ny, nz]``), ``solid.npy`` (if any),
        ``history.json`` (if monitored), ``metadata.json`` and ``fields.vti``."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        rho, u = self.density(), self.velocity()
        np.save(out / "density.npy", rho)
        np.save(out / "velocity.npy", u)
        if self.solid is not None:
            np.save(out / "solid.npy", self.solid)
        if self.history:
            (out / "history.json").write_text(json.dumps([r.__dict__ for r in self.history], indent=1))
        (out / "metadata.json").write_text(json.dumps(self.metadata(), indent=2))
        if vtk:
            from .io.vtk import write_vti

            scalars = {"density": rho, "pressure": CS2 * (rho - self.reference_density)}
            if self.solid is not None:
                scalars["solid"] = self.solid
            write_vti(out / "fields.vti", scalars, {"velocity": u})
        return out

    def checkpoint(self, path: str | Path) -> Path:
        """Save the full state (populations, mask, configuration) to a ``.npz`` file for :meth:`restore`."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {"populations": self.populations()}
        if self.solid is not None:
            arrays["solid"] = self.solid
        meta = self.metadata()
        meta["history"] = [r.__dict__ for r in self.history]
        np.savez_compressed(path, metadata=np.array(json.dumps(meta)), **arrays)
        return path if path.suffix == ".npz" else path.with_name(path.name + ".npz")

    @classmethod
    def restore(cls, path: str | Path, backend: str | None = None, **kwargs: Any) -> "Simulation":
        """Rebuild a simulation from :meth:`checkpoint`; ``backend`` may differ from the one that wrote it."""
        with np.load(path) as data:
            meta = json.loads(str(data["metadata"]))
            f = data["populations"]
            solid = data["solid"] if "solid" in data else None
        name = backend or meta["backend"]
        units = UnitConverter(**meta["units"]) if meta["units"] else None
        dtype = kwargs.pop("dtype", meta["dtype"] if name == meta["backend"] else None)
        sim = cls(shape=meta["shape"], collision=from_parameters(meta["collision"]), backend=name, dtype=dtype,
                  density=meta["reference_density"], force=meta.get("force"), units=units,
                  check_interval=meta["check_interval"],
                  history_interval=meta["history_interval"], **kwargs)
        sim.boundaries = BoundaryConditions.from_dict(meta["boundaries"])
        sim._dirty = True
        if solid is not None:
            sim.set_solid(solid)
        sim.backend.set_populations(f.astype(sim.dtype))
        sim.steps_done = int(meta["steps"])
        sim.history = [Record(**r) for r in meta.get("history", [])]
        return sim

    def __repr__(self) -> str:
        faces = ", ".join(f"{n}={f.kind.name.lower()}" for n, f in zip(FACES, self.boundaries.faces))
        return (f"Simulation({self.lattice.name}, shape={self.shape}, {self.collision!r}, "
                f"backend={self.backend_name!r}, step={self.steps_done}, {faces})")


def _collision(viscosity: float | None, tau: float | None, collision: CollisionModel | None) -> CollisionModel:
    given = sum(x is not None for x in (viscosity, tau, collision))
    if given != 1:
        raise InvalidParameterError("give exactly one of viscosity, tau or collision")
    if collision is not None:
        return collision
    if tau is not None:
        return BGK(tau)
    if not (math.isfinite(viscosity) and viscosity > 0):  # type: ignore[arg-type]
        raise InvalidParameterError(f"the lattice viscosity must be positive, got {viscosity!r}")
    return BGK(tau_from_viscosity(viscosity))  # type: ignore[arg-type]


def _check_speed(speed: Any, what: str) -> None:
    s = float(np.linalg.norm(speed)) if np.ndim(speed) else float(speed)
    if s >= SOUND_SPEED:
        raise InvalidParameterError(f"{what} ({s:.3g}) reaches the lattice speed of sound {SOUND_SPEED:.3f}")
    if s > MACH_WARN:
        warnings.warn(f"{what} is {s:.3g} lattice units (Ma = {s / SOUND_SPEED:.2f}): compressibility errors "
                      "grow as Ma²; keep it below ~0.1", stacklevel=3)
