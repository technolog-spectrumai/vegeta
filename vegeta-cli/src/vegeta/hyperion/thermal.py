"""What a laser beam does to a solid: the materials a beam meets at 1 µm (metals, plastics, brick — absorptivity,
density, heat capacity, conductivity, the melting and the failure points with their latent heats), the closed forms
(a semi-infinite solid's surface temperature, a lumped thin plate, the steady centre rise under a Gaussian spot), and
an explicit **2-D axisymmetric transient conduction solver** for a plate under a beam: the enthalpy method for
melting, cell removal for ablation (a plastic's decomposition, a vapourising surface), convection and radiation at the
faces, Beer–Lambert deposition for a partly transparent material, the transmitted share of a clear one — the time to
surface damage and to burn-through that a laser guard (IEC 60825-4) is judged on. An order of magnitude where the
material data are (brick, char), a few per cent where they are not (aluminium's conduction); the standard asks for a
test on the real panel, and this is the number to design that test around."""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

SIGMA_SB = 5.670374419e-8
T_ROOM = 293.15
EXPOSURE_LONG = 30000.0                                          # IEC 60825-4's longest exposure class (T3): "holds" means holds this long


@dataclass(frozen=True)
class Solid:
    """``absorptivity`` at the beam's wavelength (the share of the incident power that enters), ``reflectivity``
    the Fresnel share a clear material sends back (0 for an opaque one, whose absorptivity already holds its
    reflection), ``absorption_depth`` [m] over which the entering power is deposited (None: at the surface),
    ``T_melt``/``L_melt`` the melting (or softening) point and latent heat, ``T_fail``/``L_fail`` the point at which the
    material leaves (vaporises, decomposes, chars away), ``failure``: ``"melt"`` (a melt-through lets the beam through: metals,
    brick) or ``"ablate"`` (the material must decompose: plastics)."""
    name: str
    rho: float
    c: float
    k: float
    T_melt: float
    L_melt: float
    T_fail: float
    L_fail: float
    absorptivity: float
    emissivity: float = 0.8
    reflectivity: float = 0.0
    absorption_depth: float | None = None
    failure: str = "melt"
    note: str = ""

    @property
    def diffusivity(self) -> float:
        return self.k / (self.rho * self.c)

    def enthalpy_to_melt(self, T0: float = T_ROOM) -> float:
        """J/m³ from T0 to fully molten."""
        return self.rho * (self.c * (self.T_melt - T0) + self.L_melt)

    def enthalpy_to_fail(self, T0: float = T_ROOM) -> float:
        """J/m³ from T0 to gone (vaporised or decomposed)."""
        return self.rho * (self.c * (self.T_fail - T0) + self.L_melt + self.L_fail)

    def transmitted(self, thickness: float) -> float:
        """The share of the incident power that comes out of the back of a slab (two Fresnel faces, Beer–Lambert inside)."""
        if self.absorption_depth is None:
            return 0.0
        return (1.0 - self.reflectivity) ** 2 * math.exp(-thickness / self.absorption_depth)

    def with_(self, **changes) -> "Solid":
        return replace(self, **changes)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("name", "rho", "c", "k", "T_melt", "L_melt", "T_fail", "L_fail", "absorptivity", "emissivity", "reflectivity",
                                               "absorption_depth", "failure", "note")}


SOLIDS: dict[str, Solid] = {}


def _add(s: Solid) -> Solid:
    SOLIDS[s.name.lower()] = s
    return s


MILD_STEEL = _add(Solid("mild steel", 7850, 600, 40, 1810, 2.7e5, 3134, 6.1e6, 0.38, 0.7, note="absorptivity of a bare, as-rolled surface at 1 µm (0.3 polished, 0.5 oxidised); c and k the solid's means to the melt"))
STAINLESS = _add(Solid("stainless 304", 7900, 580, 20, 1700, 2.7e5, 3100, 6.1e6, 0.35, 0.5, note="low conductivity: the weld pool is narrow and deep"))
ALUMINIUM = _add(Solid("aluminium", 2700, 950, 200, 933, 3.97e5, 2743, 1.05e7, 0.12, 0.2, note="bare 5xxx/6xxx at 1 µm: 0.08 polished, 0.2 oxidised or sand-blasted; a high conductor"))
BLACK_PC = _add(Solid("black polycarbonate", 1200, 1250, 0.21, 420, 0.0, 720, 1.5e6, 0.95, 0.9, failure="ablate",
                      note="an opaque carbon-filled panel: softens at T_g ≈ 150 °C, chars and decomposes near 450 °C; L_fail the pyrolysis heat"))
CLEAR_PC = _add(Solid("clear polycarbonate", 1200, 1250, 0.21, 420, 0.0, 720, 1.5e6, 0.92, 0.9, reflectivity=0.05, absorption_depth=0.05, failure="ablate",
                      note="transmits at 1 µm: 5 % Fresnel per face, an absorption depth of ~5 cm (≈ 0.2 /cm) — a 5 mm sheet passes ~80 %"))
LASER_PC = _add(Solid("laser-safety polycarbonate", 1200, 1250, 0.21, 420, 0.0, 720, 1.5e6, 0.92, 0.9, reflectivity=0.05, absorption_depth=0.3e-3, failure="ablate",
                      note="dyed for 1 µm (OD 5+): the power is absorbed within the first few tenths of a millimetre"))
BLACK_PE = _add(Solid("black polyethylene", 950, 2100, 0.42, 400, 2.0e5, 680, 1.2e6, 0.95, 0.9, failure="ablate", note="HDPE, carbon black; melts at 130 °C, decomposes near 400 °C"))
BRICK = _add(Solid("fired clay brick", 1900, 850, 0.8, 1450, 4.0e5, 2500, 5.0e6, 0.75, 0.9, note="a red facing brick: softens and runs from ~1150 °C; absorptivity of a dull red surface at 1 µm"))
# the shielding variants of notebook 47 (Svarog III): the same metals with other surfaces, and wood — handbook-class numbers,
# each an order of magnitude where the surface decides (absorptivity at 1 µm spans 0.08–0.8 on aluminium alone)
OXIDISED_ALUMINIUM = _add(ALUMINIUM.with_(name="aluminium, oxidised", absorptivity=0.25, emissivity=0.3,
                                          note="a weathered or sand-blasted 5xxx sheet at 1 µm: 0.2–0.3 (the bare sheet's 0.12 is the polished end)"))
COATED_ALUMINIUM = _add(ALUMINIUM.with_(name="aluminium, coated", absorptivity=0.7, emissivity=0.9,
                                        note="a painted or dark-anodised sheet: the coating absorbs 0.5–0.8 at 1 µm and chars first; the metal under it conducts as bare aluminium"))
WOOD = _add(Solid("softwood (spruce)", 450, 1600, 0.12, 573, 0.0, 673, 1.0e6, 0.6, 0.9, failure="ablate",
                  note="dry spruce/pine board across the grain: ρ 400–500, k 0.10–0.14 (Wood Handbook); pyrolysis from ~300 °C (T_melt as the char point), "
                       "piloted ignition ≈ 300–400 °C, unpiloted ≈ 400 °C (T_fail); absorptivity 0.5–0.7 at 1 µm (pale wood reflects); L_fail the pyrolysis heat"))
PLYWOOD = _add(WOOD.with_(name="plywood", rho=550, k=0.13, note="birch/spruce plywood: ρ 500–650, the glue lines add little; delaminates and chars as the solid board does"))
FR_WOOD = _add(WOOD.with_(name="fire-retardant wood", rho=500, absorptivity=0.6,
                          note="a retardant-treated board (phosphate/borate): the same heating as untreated wood — the treatment lowers flame spread and raises the piloted-ignition "
                               "flux, not the laser's local heating; its benefit needs the product's test report (EN 13501-1 class), not a conduction model"))


def solid(name: str | Solid) -> Solid:
    if isinstance(name, Solid):
        return name
    key = str(name).lower()
    if key in SOLIDS:
        return SOLIDS[key]
    raise ValueError(f"unknown solid {name!r}; known: {', '.join(s.name for s in SOLIDS.values())}")


# ------------------------------------------------------------------------------------------------ the closed forms
def semi_infinite_surface_rise(solid: Solid, absorbed_irradiance: float, t) -> np.ndarray:
    """A uniform absorbed flux on a semi-infinite solid: ΔT_surface = 2 q √(t / (π ρ c k)) (Carslaw & Jaeger)."""
    t = np.asarray(t, dtype=float)
    return 2.0 * absorbed_irradiance * np.sqrt(t / (math.pi * solid.rho * solid.c * solid.k))


def thin_plate_time_to(solid: Solid, thickness: float, absorbed_irradiance: float, T: float, T0: float = T_ROOM) -> float:
    """A lumped plate under a uniform absorbed flux, no losses: the time to reach ``T`` (through its latent heat when
    T ≥ T_melt): t = ρ d (c ΔT + L) / q — the lower bound a real plate beats by its lateral conduction."""
    L = solid.L_melt if T >= solid.T_melt else 0.0
    return solid.rho * thickness * (solid.c * (T - T0) + L) / absorbed_irradiance


def gaussian_steady_rise(solid: Solid, absorbed_power: float, radius: float) -> float:
    """The steady centre rise of a semi-infinite solid under a Gaussian spot of 1/e² radius w:
    ΔT = P_abs / (√(2π) k w) (Carslaw & Jaeger's Gaussian source with a = w/√2)."""
    return absorbed_power / (math.sqrt(2 * math.pi) * solid.k * radius)


# ------------------------------------------------------------------------------------------------- the beam on it
def irradiance(power: float, radius: float, r, shape: str = "gaussian") -> np.ndarray:
    """W/m² at radius ``r``: a Gaussian of 1/e² ``radius`` (peak 2P/πw²) or a top-hat of that radius (P/πw²)."""
    r = np.asarray(r, dtype=float)
    if shape == "gaussian":
        return 2.0 * power / (math.pi * radius ** 2) * np.exp(-2.0 * (r / radius) ** 2)
    if shape == "tophat":
        return np.where(r <= radius, power / (math.pi * radius ** 2), 0.0)
    raise ValueError("shape is 'gaussian' or 'tophat'")


@dataclass
class HeatingResult:
    """The transient: ``t`` and the centre's top and bottom temperatures, the melt and ablation depths at the centre,
    the hottest point; ``surface_damage_time`` (the centre's face reaches the melting/softening point),
    ``failure_time`` (the melt front or the ablation front reaches the back face at the centre: the beam gets through),
    ``steady`` (the temperatures stopped changing before anything failed — then it never does), the final field."""
    solid: Solid
    thickness: float
    power: float
    radius: float
    shape: str
    t: np.ndarray
    T_top: np.ndarray
    T_bottom: np.ndarray
    T_max: np.ndarray
    melt_depth: np.ndarray
    ablated_depth: np.ndarray
    surface_damage_time: float = math.nan
    failure_time: float = math.nan
    failure_mode: str = ""
    steady: bool = False
    t_end: float = 0.0
    transmitted_fraction: float = 0.0
    T_field: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    r: np.ndarray = field(default_factory=lambda: np.zeros(0))
    z: np.ndarray = field(default_factory=lambda: np.zeros(0))
    note: str = ""

    @property
    def failed(self) -> bool:
        return math.isfinite(self.failure_time)

    def survives(self, exposure_s: float) -> bool:
        """Whether the barrier holds for ``exposure_s`` (a known failure time later than it, or a steady state reached)."""
        if self.failed:
            return self.failure_time > exposure_s
        return self.steady or self.t_end >= exposure_s

    def to_dict(self) -> dict:
        return {"solid": self.solid.name, "thickness": self.thickness, "power": self.power, "radius": self.radius, "shape": self.shape,
                "surface_damage_time": self.surface_damage_time, "failure_time": self.failure_time, "failure_mode": self.failure_mode, "steady": self.steady,
                "t_end": self.t_end, "T_top_end": float(self.T_top[-1]), "T_bottom_end": float(self.T_bottom[-1]), "T_max_end": float(self.T_max[-1]),
                "melt_depth_end": float(self.melt_depth[-1]), "ablated_depth_end": float(self.ablated_depth[-1]), "transmitted_fraction": self.transmitted_fraction,
                "note": self.note}


BACKENDS = ("auto", "numpy", "fenicsx")


def _backend(backend: str) -> str:
    """``"auto"``: FEniCSx when dolfinx is installed (the tested finite-element route), else the numpy solver."""
    if backend not in BACKENDS:
        raise ValueError(f"backend is one of {BACKENDS}")
    if backend == "numpy":
        return "numpy"
    from . import thermal_fenicsx
    if thermal_fenicsx.available():
        return "fenicsx"
    if backend == "fenicsx":
        raise ImportError("backend='fenicsx' asked but dolfinx is not installed (docs/fenics.md)")
    return "numpy"


def steady_plate(solid: Solid | str, thickness: float, power: float, radius: float, shape: str = "gaussian", h: float = 10.0, T0: float = T_ROOM,
                 nr: int = 32, nz: int = 8, r_max: float | None = None, tol: float = 1e-3, max_iter: int = 200_000, backend: str = "auto") -> dict:
    """The steady state of the same plate under the same beam, without phase change: Jacobi iteration of the conduction
    balance on the r–z grid with the convection and the (linearised) radiation at both faces and the far edge at T0.
    Cheap (a fraction of a second), and the answer to "does it ever get there?": ``T_max`` against the melting or
    decomposition point — when it stays below, no transient is needed; when it does not, the transient says when.
    ``backend`` as in ``heat_plate``: FEniCSx (Newton on the nonlinear radiation) when installed, else this Jacobi iteration."""
    if _backend(backend) == "fenicsx":
        from .thermal_fenicsx import steady_plate_fenicsx
        return steady_plate_fenicsx(solid, thickness, power, radius, shape, h, T0, nr, nz, r_max)
    s = solid if isinstance(solid, Solid) else SOLIDS[str(solid).lower()]
    if r_max is None:
        r_max = max(8.0 * radius, 12.0 * thickness, 0.02)
    dr, dz = r_max / nr, thickness / nz
    r_c = (np.arange(nr) + 0.5) * dr
    r_f = np.arange(nr + 1) * dr
    q_in = s.absorptivity * irradiance(power, radius, r_c, shape)
    if s.absorption_depth is not None:
        w_abs = np.exp(-np.arange(nz) * dz / s.absorption_depth) - np.exp(-(np.arange(nz) + 1) * dz / s.absorption_depth)
        src = np.outer(w_abs, q_in) / dz                           # W/m³
    else:
        src = np.zeros((nz, nr))
        src[0] = q_in / dz
    T = np.full((nz, nr), T0, dtype=float)
    k = s.k
    cz = k / dz ** 2                                               # axial coupling per volume (W/m³/K)
    cr_in = k * r_f[:-1] / (r_c * dr ** 2)                         # radial couplings per volume, per column (nr,)
    cr_out = k * r_f[1:] / (r_c * dr ** 2)
    cr_in[0] = 0.0                                                 # the axis is insulated
    cr = cr_in + cr_out
    change = math.inf
    for it in range(max_iter):
        inner = np.hstack([T[:, :1], T[:, :-1]])                   # radial neighbours (the far edge at T0)
        outer = np.hstack([T[:, 1:], np.full((nz, 1), T0)])
        radial = cr_in * inner + cr_out * outer + src              # (nz, nr)
        # the faces lose by convection and radiation, linearised about the current temperature: an h_eff per column
        h_top = h + s.emissivity * SIGMA_SB * (T[0] ** 2 + T0 ** 2) * (T[0] + T0)
        h_bot = h + s.emissivity * SIGMA_SB * (T[-1] ** 2 + T0 ** 2) * (T[-1] + T0)
        Tn = np.empty_like(T)
        if nz == 1:
            Tn[0] = (radial[0] + (h_top + h_bot) * T0 / dz) / (cr + (h_top + h_bot) / dz)
        else:
            Tn[0] = (radial[0] + cz * T[1] + h_top * T0 / dz) / (cr + cz + h_top / dz)
            Tn[-1] = (radial[-1] + cz * T[-2] + h_bot * T0 / dz) / (cr + cz + h_bot / dz)
            if nz > 2:
                Tn[1:-1] = (radial[1:-1] + cz * (T[:-2] + T[2:])) / (cr + 2 * cz)
        change = float(np.abs(Tn - T).max())
        T = Tn
        if change < tol:
            break
    damage_T = s.T_melt if s.failure == "melt" else s.T_fail
    return {"T_max": float(T.max()), "T_top_centre": float(T[0, 0]), "T_bottom_centre": float(T[-1, 0]), "damage_T": damage_T,
            "reaches_damage": bool(T.max() >= damage_T), "iterations": it + 1, "converged": change < tol, "T_field": T, "r": r_c,
            "z": (np.arange(nz) + 0.5) * dz, "transmitted_fraction": s.transmitted(thickness), "backend": "numpy"}


def heat_plate(solid: Solid | str, thickness: float, power: float, radius: float, t_max: float, shape: str = "gaussian", h: float = 10.0,
               T0: float = T_ROOM, nr: int = 32, nz: int = 8, r_max: float | None = None, stop_at_failure: bool = True, steady_tol: float = 0.02,
               record_every: float | None = None, max_steps: int = 4_000_000, backend: str = "auto") -> HeatingResult:
    """A plate of ``thickness`` under a beam of ``power`` and ``radius`` for up to ``t_max`` seconds; ``h`` the convection
    on both faces [W/m²/K] (10 still air; the emissivity's radiation is added), the far edge at ``r_max`` (default: the
    larger of 8 radii and 12 thicknesses) held at ``T0``. Explicit finite differences in r and z (``nr`` × ``nz`` cells),
    the enthalpy method for the latent heats, a cell that reaches the failure enthalpy is removed and the beam falls on
    the next. Stops at failure (``stop_at_failure``), at ``t_max``, or once steady (the hottest point moves less than
    ``steady_tol`` K per second over the last ten seconds and nothing is melting). ``backend``: ``"numpy"`` (the default
    under ``"auto"`` for the transient: it has the stall detection the guard study needs) or ``"fenicsx"`` (the finite-element
    transient of ``thermal_fenicsx``, when dolfinx is installed)."""
    if backend == "fenicsx" and _backend(backend) == "fenicsx":            # "auto" keeps numpy for the transient: its stall detection is not in the FEM one yet
        from .thermal_fenicsx import heat_plate_fenicsx
        return heat_plate_fenicsx(solid, thickness, power, radius, t_max, shape, h, T0, nr, nz, r_max, stop_at_failure, record_every=record_every)
    s = solid if isinstance(solid, Solid) else SOLIDS[str(solid).lower()]
    if r_max is None:
        r_max = max(8.0 * radius, 12.0 * thickness, 0.02)
    dr, dz = r_max / nr, thickness / nz
    r_c = (np.arange(nr) + 0.5) * dr                               # cell centres
    r_f = np.arange(nr + 1) * dr                                   # faces
    z_c = (np.arange(nz) + 0.5) * dz
    vol = 2 * math.pi * r_c * dr * dz                              # cell volumes (a ring), shape (nr,)
    area_r = 2 * math.pi * r_f * dz                                # radial face areas (nr+1,)
    area_z = 2 * math.pi * r_c * dr                                # axial face areas (nr,)
    kappa = s.diffusivity
    dt = 0.4 / (kappa * (1.0 / dr ** 2 + 1.0 / dz ** 2))            # 80 % of the explicit limit 1/(2κ(1/dr² + 1/dz²))
    H = np.zeros((nz, nr))                                         # enthalpy density above T0, J/m³
    alive = np.ones((nz, nr), dtype=bool)
    H_melt0 = s.rho * s.c * (s.T_melt - T0)
    H_melt1 = H_melt0 + s.rho * s.L_melt
    H_fail = s.enthalpy_to_fail(T0)
    q_in = s.absorptivity * irradiance(power, radius, r_c, shape)  # W/m² entering at the top, per column
    if s.absorption_depth is not None:                             # Beer–Lambert deposition weights per cell depth (from the live surface)
        w_abs = np.exp(-np.arange(nz) * dz / s.absorption_depth) - np.exp(-(np.arange(nz) + 1) * dz / s.absorption_depth)
    else:
        w_abs = None
    transmitted = s.transmitted(thickness)

    H_boil = H_melt1 + s.rho * s.c * (s.T_fail - s.T_melt)      # beyond it the cell is boiling / decomposing: a plateau at T_fail until it leaves

    def temperature(Hc):
        T = np.where(Hc < H_melt0, T0 + Hc / (s.rho * s.c), np.where(Hc < H_melt1, s.T_melt, s.T_melt + (Hc - H_melt1) / (s.rho * s.c)))
        return np.minimum(T, s.T_fail)

    # the time step: the explicit limit, and no surface cell may take more than 5 % of its way to failure in one step
    q_peak = float(q_in.max()) if q_in.size else 0.0
    if q_peak > 0:
        dt = min(dt, 0.05 * H_fail * dz / q_peak)
    rec_dt = record_every if record_every else max(dt, t_max / 400.0)
    rows = {"t": [], "T_top": [], "T_bottom": [], "T_max": [], "melt": [], "abl": []}
    t, step, next_rec = 0.0, 0, 0.0
    surface_damage, failure, mode = math.nan, math.nan, ""
    steady = False
    history: list[tuple[float, float, float]] = []                 # (t, T_max, melt + ablated depth at the centre)
    while t < t_max and step < max_steps:
        T = temperature(H)
        Tl = np.where(alive, T, T0)
        kk = np.where(alive, s.k, 0.0)
        # conduction fluxes: axial between rows, radial between columns; a dead cell conducts nothing
        qz = np.zeros((nz + 1, nr))
        kz = 2 * kk[1:] * kk[:-1] / np.maximum(kk[1:] + kk[:-1], 1e-30)
        qz[1:-1] = kz * (Tl[:-1] - Tl[1:]) / dz                     # positive downward (W/m²)
        qr = np.zeros((nz, nr + 1))
        kr = 2 * kk[:, 1:] * kk[:, :-1] / np.maximum(kk[:, 1:] + kk[:, :-1], 1e-30)
        qr[:, 1:-1] = kr * (Tl[:, :-1] - Tl[:, 1:]) / dr               # positive outward
        qr[:, -1] = kk[:, -1] * (Tl[:, -1] - T0) / (0.5 * dr)           # the far edge held at T0
        dH = np.zeros_like(H)
        dH += (qz[:-1] - qz[1:]) * area_z / vol                       # axial net inflow per volume
        dH += (qr[:, :-1] * area_r[:-1] - qr[:, 1:] * area_r[1:]) / vol
        # the exposed faces: the first live cell of each column takes the beam and loses by convection and radiation; the last loses too
        top = np.argmax(alive, axis=0)                               # the live surface index per column
        col_alive = alive.any(axis=0)
        cols = np.where(col_alive)[0]
        i0 = top[cols]
        per_vol = area_z[cols] / vol[cols]                           # 1/dz: a face flux into a cell's volume
        src = np.zeros_like(H)
        if w_abs is None:
            src[i0, cols] += q_in[cols] * per_vol
        else:                                                        # Beer–Lambert from the live surface down
            for j, i in zip(cols, i0):
                src[i:, j] += q_in[j] * w_abs[:nz - i] * (area_z[j] / vol[j])
        T_face = Tl[i0, cols]
        src[i0, cols] -= (h * (T_face - T0) + s.emissivity * SIGMA_SB * (T_face ** 4 - T0 ** 4)) * per_vol
        T_back = Tl[-1, cols]
        src[-1, cols] -= (h * (T_back - T0) + s.emissivity * SIGMA_SB * (T_back ** 4 - T0 ** 4)) * per_vol
        dH += src
        H = H + dt * dH
        H = np.where(alive, np.maximum(H, -s.rho * s.c * (T0 - 0.0)), 0.0)
        # ablation: cells past the failure enthalpy leave
        gone = alive & (H >= H_fail)
        if gone.any():
            alive &= ~gone
            H = np.where(alive, H, 0.0)
        t += dt
        step += 1
        T = temperature(H)
        Tl = np.where(alive, T, T0)
        centre_top = top[0]
        T_top_c = float(Tl[centre_top, 0]) if col_alive[0] else math.nan
        if math.isnan(surface_damage) and col_alive[0] and T_top_c >= s.T_melt - 1e-9:
            surface_damage = t
        melt_depth = float(np.sum((H[:, 0] >= H_melt1) & alive[:, 0]) * dz) if s.L_melt > 0 else float(np.sum((T[:, 0] >= s.T_melt) & alive[:, 0]) * dz)
        abl_depth = float(np.sum(~alive[:, 0]) * dz)
        if math.isnan(failure):
            if s.failure == "melt" and (abl_depth + melt_depth >= thickness - 1e-12 or not col_alive[0]):
                failure, mode = t, "melt-through at the centre"
            elif s.failure == "ablate" and not col_alive[0]:
                failure, mode = t, "ablated through at the centre"
        if t >= next_rec or not math.isnan(failure):
            rows["t"].append(t)
            rows["T_top"].append(T_top_c)
            rows["T_bottom"].append(float(Tl[-1, 0]))
            rows["T_max"].append(float(Tl.max()))
            rows["melt"].append(melt_depth)
            rows["abl"].append(abl_depth)
            next_rec = t + rec_dt
            history.append((t, float(Tl.max()), melt_depth + abl_depth))
            window = max(10.0, 0.05 * t)                            # the last ten seconds, or 5 % of the run
            history = [hh for hh in history if hh[0] >= t - window]
            if len(history) >= 3 and t > 10.0 and not gone.any():
                span = max(history[-1][0] - history[0][0], 1e-9)
                rate = abs(history[-1][1] - history[0][1]) / span
                front = abs(history[-1][2] - history[0][2]) / span   # the melt/ablation front's speed at the centre [m/s]
                if rate < steady_tol and front * EXPOSURE_LONG < 0.1 * thickness:
                    steady = True                                   # the temperatures hold and the front would not cross a tenth of the plate in 8 h
        if (not math.isnan(failure) and stop_at_failure) or steady:
            break
    res = HeatingResult(s, thickness, power, radius, shape, np.array(rows["t"]), np.array(rows["T_top"]), np.array(rows["T_bottom"]), np.array(rows["T_max"]),
                        np.array(rows["melt"]), np.array(rows["abl"]), surface_damage, failure, mode, steady, t, transmitted,
                        np.where(alive, temperature(H), np.nan), r_c, z_c)
    if step >= max_steps:
        res.note = f"stopped at the step limit ({max_steps}) at t = {t:.1f} s"
    return res
