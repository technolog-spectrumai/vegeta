"""Beam combining of complete fibre-laser outputs — three methods on one field model:

- ``IncoherentCombiner``: mutually incoherent beams; their *intensities* add (near field and far field). The power adds,
  the brightness does not: the combined beam's étendue is the union of the channels'.
- ``CoherentCombiner``: one wavelength, mutually coherent channels with set relative phases; their complex *fields*
  add, the intensity is |ΣE|² — interference. In phase and overlapped the on-axis far-field intensity is N² times a
  channel's; tiled side by side it is the central lobe of an array (a filled-aperture or a tiled-aperture combiner).
- ``SpectralCombiner``: channels at distinct wavelengths overlapped on one axis by a dispersive element (a grating or
  dichroics): the intensities add (different colours never interfere) *and* the étendue stays a single channel's up to the
  residual pointing the dispersion leaves — the power and the brightness both scale with N.

A ``Channel`` is a fibre laser's output at the combining plane: a ``GaussianBeam`` (reused as it is: wavelength, waist,
M²), its power, its centre ``(x, y)``, its phase and a loss on the way in. The fields are sampled on a square grid
(``Grid``) at the combining plane (every channel at its waist there) and propagated to the far field by the Fraunhofer
integral as an FFT, scaled so that the integral of the far-field intensity over angle is the power — the
conservation-of-power check every result carries. Beam quality is reported as second-moment widths and the beam
parameter product in both axes, the brightness as power per (area × solid angle), and M² only where it is defined (one
wavelength; the second-moment definition of ISO 11146) — a multi-wavelength beam has a BPP and a brightness, not an M².

Pump-diode combining (the (N+1)×1 pump combiners of ``svarog_industrial``) is a different thing — étendue bookkeeping
of multimode pigtails into a cladding — and is not this module's. Nothing here is about what the beam does to a
target. Pure numpy; the drawings live in ``notebooks/designs/svarog_combining_drawings.py``."""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from .beam import GaussianBeam


# -------------------------------------------------------------------------------------------------------- channels
@dataclass(frozen=True)
class Channel:
    """One fibre laser's output at the combining plane: ``beam`` (a GaussianBeam at its waist: wavelength, w0, M²),
    ``power`` [W] before the combiner, its centre ``x``, ``y`` [m], ``phase`` [rad] (coherent combining), ``loss``
    (the fraction lost on the way in: a splice, a coating, a grating's efficiency), ``name``."""
    beam: GaussianBeam
    power: float
    x: float = 0.0
    y: float = 0.0
    phase: float = 0.0
    loss: float = 0.0
    name: str = ""

    @property
    def wavelength(self) -> float:
        return self.beam.wavelength

    @property
    def w0(self) -> float:
        return self.beam.w0

    @property
    def power_in(self) -> float:
        """What reaches the combining plane."""
        return self.power * (1.0 - self.loss)

    def with_(self, **changes) -> "Channel":
        return replace(self, **changes)

    def field(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """The complex field on the grid [√(W/m²)]: a Gaussian of 1/e² radius w0 carrying ``power_in``, with the phase.
        The channel's M² is not in this field (a scalar field is diffraction-limited): the incoherent and spectral
        combiners widen the channel's far field by M² (``far_intensity``), which is what a multimode beam does; the
        coherent combiner takes M² = 1 only."""
        w = self.w0
        amp = math.sqrt(2.0 * self.power_in / (math.pi * w * w))
        return amp * np.exp(-((X - self.x) ** 2 + (Y - self.y) ** 2) / (w * w)) * np.exp(1j * self.phase)

    def far_intensity(self, grid: "Grid", theta_axis: np.ndarray, tilt: float = 0.0) -> np.ndarray:
        """This channel's own far-field intensity [W/sr] on the angle axis ``theta_axis`` (both axes alike): the Fraunhofer
        field of ``field`` at the channel's wavelength, its angles widened by M² (I(θ) = I_dl(θ/M²)/M⁴: the same power in
        an M² times wider cone), tilted by ``tilt`` [rad] along x, and put onto ``theta_axis`` when the channel's own
        angle grid differs (another wavelength, M² ≠ 1)."""
        X, Y = grid.xy()
        E = self.field(X, Y)
        if tilt:
            E = E * np.exp(1j * 2 * math.pi * tilt * X / self.wavelength)
        I = np.abs(far_field(E, grid, self.wavelength)) ** 2
        m2 = self.beam.m2
        tx, _ = grid.angles(self.wavelength)
        src = tx[0] * m2
        if m2 == 1.0 and np.array_equal(src, theta_axis):
            return I
        return _resample(I, src, theta_axis) / (m2 * m2)

    def to_dict(self) -> dict:
        return {"name": self.name, "wavelength": self.wavelength, "w0": self.w0, "m2": self.beam.m2, "power": self.power, "x": self.x, "y": self.y,
                "phase": self.phase, "loss": self.loss}


def channels(n: int, wavelength: float, power: float, w0: float, pitch: float | None = None, layout: str = "ring", m2: float = 1.0,
             phases=None, losses=None, wavelengths=None, name: str = "ch") -> list[Channel]:
    """``n`` identical channels on a ``layout``: ``"overlap"`` (all at the origin: a filled aperture), ``"ring"`` (on a
    circle of radius ``pitch``, one at the centre when n = 7, 1 alone at the origin), ``"line"`` (along x at ``pitch``),
    ``"hex"`` (a hexagonal close packing at ``pitch``). ``phases``/``losses``/``wavelengths`` per channel when given."""
    pitch = 3.5 * w0 if pitch is None else pitch                                # neighbours' overlap e^{-pitch²/(2w0²)} = 0.2 %
    if n == 1 or layout == "overlap":
        pos = [(0.0, 0.0)] * n
    elif layout == "line":
        pos = [((i - (n - 1) / 2) * pitch, 0.0) for i in range(n)]
    elif layout == "ring":
        if n == 7:
            pos = [(0.0, 0.0)] + [(pitch * math.cos(2 * math.pi * i / 6), pitch * math.sin(2 * math.pi * i / 6)) for i in range(6)]
        else:
            r = pitch / (2 * math.sin(math.pi / n)) if n > 2 else pitch / 2       # neighbours ``pitch`` apart
            pos = [(r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n)) for i in range(n)]
    elif layout == "hex":
        pts = [(0.0, 0.0)]
        ring = 1
        while len(pts) < n:
            for k in range(6 * ring):
                a = 2 * math.pi * k / (6 * ring)
                pts.append((pitch * ring * math.cos(a), pitch * ring * math.sin(a)))
            ring += 1
        pos = pts[:n]
    else:
        raise ValueError("layout is 'overlap', 'ring', 'line' or 'hex'")
    out = []
    for i, (x, y) in enumerate(pos):
        lam = wavelengths[i] if wavelengths is not None else wavelength
        out.append(Channel(GaussianBeam(lam, w0, 0.0, m2, 1.0, f"{name}{i + 1}"), power, x, y, phases[i] if phases is not None else 0.0,
                           losses[i] if losses is not None else 0.0, f"{name}{i + 1}"))
    return out


# ------------------------------------------------------------------------------------------------------------ grid
@dataclass(frozen=True)
class Grid:
    """A square grid of ``n`` × ``n`` points over ``size`` [m] at the combining plane, and its far-field angles.
    ``for_channels`` sizes it: the window holds every channel with ``margin`` waists around (12: the far-field pixel λ/size
    is then ≤ 1/7.6 of a channel's 1/e² half-angle), the sampling resolves the smallest waist."""
    size: float
    n: int = 256

    @property
    def dx(self) -> float:
        return self.size / self.n

    def xy(self):
        x = (np.arange(self.n) - self.n / 2) * self.dx
        return np.meshgrid(x, x, indexing="xy")

    def angles(self, wavelength: float):
        """The far-field angle grid [rad] the FFT of this grid gives at ``wavelength``: θ = λ f, f the spatial frequencies."""
        f = np.fft.fftshift(np.fft.fftfreq(self.n, self.dx))
        th = wavelength * f
        return np.meshgrid(th, th, indexing="xy")

    @classmethod
    def for_channels(cls, chans: list[Channel], n: int = 256, margin: float = 12.0, points_per_waist: float = 6.0) -> "Grid":
        xs = [c.x for c in chans]; ys = [c.y for c in chans]
        w = max(c.w0 for c in chans)
        extent = max(max(xs) - min(xs), max(ys) - min(ys)) + 2 * margin * w
        wmin = min(c.w0 for c in chans)
        n_needed = int(math.ceil(extent / (wmin / points_per_waist)))
        n = max(n, 1 << (n_needed - 1).bit_length()) if n_needed > n else n
        return cls(extent, n)


# ------------------------------------------------------------------------------------------------------- results
@dataclass
class Combined:
    """What a combiner gives: the near-field intensity ``I_near`` [W/m²] on the grid, the far-field intensity ``I_far``
    [W/sr] on the angle grid (``theta_x``, ``theta_y`` [rad]); the power accounting (``power_in`` before losses,
    ``power_out`` after the channels' and the combiner's, ``power_near``/``power_far`` the integrals of the two
    intensities, ``conservation`` their relative difference — Parseval — and ``overlap`` the near field's against ``power_out``, the
    coherent cross term of overlapping channels); the beam metrics: second-moment radii ``sigma_x/y`` [m]
    and ``sigma_tx/ty`` [rad], ``bpp_x/y`` = σ_x σ_θx [m·rad], ``m2_x/y`` (one wavelength only, else nan), ``brightness``
    [W/(m²·sr)] = P / (π·2σ_x·2σ_y · π·2σ_θx·2σ_θy), the on-axis far-field intensity and the ``efficiency`` the method
    defines (see each combiner). ``wavelengths`` the channels'."""
    method: str
    grid: Grid
    I_near: np.ndarray
    I_far: np.ndarray
    theta_x: np.ndarray
    theta_y: np.ndarray
    power_in: float
    power_out: float
    power_near: float
    power_far: float
    sigma_x: float
    sigma_y: float
    sigma_tx: float
    sigma_ty: float
    bpp_x: float
    bpp_y: float
    m2_x: float
    m2_y: float
    brightness: float
    I_far_axis: float
    efficiency: float
    wavelengths: tuple
    note: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def conservation(self) -> float:
        """Parseval: the far-field power against the near-field power, relative — the numerical check (≈ 1e-15)."""
        return abs(self.power_far - self.power_near) / self.power_near if self.power_near > 0 else 0.0

    @property
    def overlap(self) -> float:
        """The near-field power against the channels' power after losses, relative: zero for incoherent sums and for
        tiles that do not overlap; for coherent channels that overlap it is the interference cross term
        2 Σ √(P_i P_j) cos(φ_i − φ_j) e^{−d_ij²/(2 w0²)} — physical, not numerical — the share of power a real combiner
        (a beam splitter, a tiled aperture with no overlap) would have to account for elsewhere."""
        return (self.power_near - self.power_out) / self.power_out if self.power_out > 0 else 0.0

    def cross_section(self, axis: str = "x", far: bool = False):
        """The intensity along the axis through the origin: (coordinate, intensity)."""
        I = self.I_far if far else self.I_near
        n = I.shape[0] // 2
        if far:
            coord = self.theta_x[n] if axis == "x" else self.theta_y[:, n]
        else:
            X, Y = self.grid.xy()
            coord = X[n] if axis == "x" else Y[:, n]
        return coord, (I[n] if axis == "x" else I[:, n])

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("method", "power_in", "power_out", "power_near", "power_far", "sigma_x", "sigma_y", "sigma_tx", "sigma_ty", "bpp_x", "bpp_y",
                                               "m2_x", "m2_y", "brightness", "I_far_axis", "efficiency", "note")} | {"conservation": self.conservation, "overlap": self.overlap,
                                                                                                                      "wavelengths": list(self.wavelengths), "grid_n": self.grid.n, "grid_size": self.grid.size,
                                                                                                                      **{k: v for k, v in self.extra.items() if not hasattr(v, "shape")}}


def far_field(E: np.ndarray, grid: Grid, wavelength: float) -> np.ndarray:
    """The Fraunhofer far field of a near field ``E`` [√(W/m²)] as a complex amplitude over angle [√(W/sr)]:
    E_far(θ) = (dx dy / λ) Σ E e^{−i 2π (x θx + y θy)/λ} — Parseval then gives ∫|E_far|² dθ = ∫|E|² dA."""
    F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(E)))
    return F * (grid.dx * grid.dx / wavelength)


def _moments(I: np.ndarray, X: np.ndarray, Y: np.ndarray, dA: float):
    p = float(I.sum() * dA)
    if p <= 0:
        return p, math.nan, math.nan, math.nan, math.nan
    cx = float((I * X).sum() * dA / p); cy = float((I * Y).sum() * dA / p)
    sx = math.sqrt(max(float((I * (X - cx) ** 2).sum() * dA / p), 0.0)); sy = math.sqrt(max(float((I * (Y - cy) ** 2).sum() * dA / p), 0.0))
    return p, cx, cy, sx, sy


def _assemble(method: str, grid: Grid, I_near, I_far, thx, thy, power_in, power_out, wavelengths, efficiency, note="", extra=None) -> Combined:
    X, Y = grid.xy()
    dA = grid.dx ** 2
    p_near, _, _, sx, sy = _moments(I_near, X, Y, dA)
    dth = float(thx[0, 1] - thx[0, 0])
    p_far, _, _, stx, sty = _moments(I_far, thx, thy, dth * dth)
    lam = wavelengths[0] if len(set(wavelengths)) == 1 else math.nan
    m2x = 4 * math.pi * sx * stx / lam if lam == lam else math.nan
    m2y = 4 * math.pi * sy * sty / lam if lam == lam else math.nan
    bright = power_out / (math.pi * 2 * sx * 2 * sy * math.pi * 2 * stx * 2 * sty) if sx * sy * stx * sty > 0 else math.nan
    n = I_far.shape[0] // 2
    return Combined(method, grid, I_near, I_far, thx, thy, power_in, power_out, p_near, p_far, sx, sy, stx, sty, sx * stx, sy * sty, m2x, m2y, bright,
                    float(I_far[n, n]), efficiency, tuple(wavelengths), note, extra or {})


# ----------------------------------------------------------------------------------------------------- combiners
@dataclass(frozen=True)
class IncoherentCombiner:
    """Mutually incoherent channels (different lasers, no phase control): intensities add. ``throughput`` the combiner's
    own transmission. Efficiency = power out / power in (the losses only — nothing is lost to interference)."""
    throughput: float = 1.0
    name: str = "incoherent"

    def combine(self, chans: list[Channel], grid: Grid | None = None, n: int = 256) -> Combined:
        grid = grid or Grid.for_channels(chans, n)
        X, Y = grid.xy()
        lam0 = chans[0].wavelength
        thx, thy = grid.angles(lam0)
        I_near = np.zeros(X.shape); I_far = np.zeros(X.shape)
        for c in chans:
            I_near += np.abs(c.field(X, Y)) ** 2 * self.throughput
            I_far += c.far_intensity(grid, thx[0]) * self.throughput
        p_in = sum(c.power for c in chans); p_out = sum(c.power_in for c in chans) * self.throughput
        return _assemble(self.name, grid, I_near, I_far, thx, thy, p_in, p_out, [c.wavelength for c in chans], p_out / p_in if p_in else 0.0,
                         "intensities add; the étendue is the union of the channels'")


@dataclass(frozen=True)
class CoherentCombiner:
    """Mutually coherent channels at one wavelength with set phases: fields add. Two apertures:

    - ``"filled"``: the channels overlap on one axis through a lossless N-port combiner (a tree of beam splitters, a
      diffractive element): the output field is Σ E_i /√N — in phase the output carries ΣP, any phase error sends the
      rest out of the other ports (``extra["port_loss"]``); ``power_out`` is what leaves the output port.
    - ``"tiled"``: the channels sit side by side (a tiled aperture): fields add as they are, every tile keeps its power;
      the far field is the single-tile envelope times the array factor, so the on-axis intensity is N² × one tile's but
      the central lobe narrows and side lobes carry the rest.

    ``"auto"`` takes ``"filled"`` when every channel is at the same position, else ``"tiled"``. ``throughput`` the
    combiner's own transmission. Efficiency = the power in the central lobe — the cone a diffraction-limited Gaussian
    beam of the array's near-field size (2σ) would have, half-angle λ/(π·2σ) — against the 1 − e⁻² such a beam puts
    there; 1 for one Gaussian, less for an array (its side lobes) or for phase errors. ``extra["axis_gain"]`` is the
    on-axis far-field intensity against the same channels in phase (1 when in phase, e^{−σ²} under phase noise)."""
    throughput: float = 1.0
    aperture: str = "auto"
    name: str = "coherent"

    def combine(self, chans: list[Channel], grid: Grid | None = None, n: int = 256) -> Combined:
        lams = {c.wavelength for c in chans}
        if len(lams) > 1:
            raise ValueError(f"coherent combining needs one wavelength; got {sorted(lams)}")
        if any(c.beam.m2 != 1.0 for c in chans):
            raise ValueError("coherent combining is modelled for diffraction-limited channels (M² = 1): a multimode beam's modes do not add as one field")
        aperture = self.aperture
        if aperture == "auto":
            aperture = "filled" if all(c.x == chans[0].x and c.y == chans[0].y for c in chans) else "tiled"
        if aperture not in ("filled", "tiled"):
            raise ValueError("aperture is 'auto', 'filled' or 'tiled'")
        grid = grid or Grid.for_channels(chans, n)
        X, Y = grid.xy()
        lam = chans[0].wavelength
        thx, thy = grid.angles(lam)
        norm = math.sqrt(self.throughput) / (math.sqrt(len(chans)) if aperture == "filled" else 1.0)
        E = sum(c.field(X, Y) for c in chans) * norm
        I_near = np.abs(E) ** 2
        I_far = np.abs(far_field(E, grid, lam)) ** 2
        dA = grid.dx ** 2
        p_near = float(I_near.sum() * dA)
        p_in = sum(c.power for c in chans); p_avail = sum(c.power_in for c in chans) * self.throughput
        p_out = p_near if aperture == "filled" else p_avail                 # filled: the other ports take the rest
        # the central lobe: the cone of a diffraction-limited Gaussian of the array's near-field size
        _, _, _, sx, sy = _moments(I_near, X, Y, dA)
        w_eff = 2 * math.sqrt(sx * sy) if sx * sy > 0 else min(c.w0 for c in chans)
        cone_half = lam / (math.pi * w_eff)
        cone = (thx ** 2 + thy ** 2) <= cone_half ** 2
        dth = float(thx[0, 1] - thx[0, 0]); dOm = dth * dth
        in_lobe = float(I_far[cone].sum() * dOm)
        # the reference: a diffraction-limited Gaussian of the same size and power, its cone share on this grid (≈ 1 − e⁻²)
        ref = Channel(GaussianBeam(lam, w_eff), p_avail if p_avail > 0 else 1.0).field(X, Y)
        in_lobe_ref = float((np.abs(far_field(ref, grid, lam)) ** 2)[cone].sum() * dOm)
        eff = in_lobe / in_lobe_ref if in_lobe_ref > 0 and p_avail > 0 else 0.0
        # the in-phase reference with the same geometry and losses
        n0 = I_far.shape[0] // 2
        E_ref = sum(c.with_(phase=0.0).field(X, Y) for c in chans) * norm
        I_ref_axis = float(np.abs(far_field(E_ref, grid, lam))[n0, n0] ** 2)
        single = chans[0].with_(x=0.0, y=0.0, phase=0.0, loss=0.0)
        I_single_axis = float(np.abs(far_field(single.field(X, Y), grid, lam))[n0, n0] ** 2)
        return _assemble(self.name, grid, I_near, I_far, thx, thy, p_in, p_out, [lam] * len(chans), eff,
                         f"fields add ({aperture} aperture); efficiency = power in the array's central lobe against a Gaussian's 1 − e⁻²",
                         {"aperture": aperture, "power_in_lobe": in_lobe, "cone_half_angle": cone_half,
                          "axis_gain": float(I_far[n0, n0]) / I_ref_axis if I_ref_axis > 0 else math.nan,
                          "axis_vs_single": float(I_far[n0, n0]) / I_single_axis if I_single_axis > 0 else math.nan,
                          "port_loss": p_avail - p_out})


@dataclass(frozen=True)
class SpectralCombiner:
    """Channels at distinct wavelengths (at least ``min_separation`` apart) overlapped on one axis by a dispersive
    element: intensities add (no interference between colours) and the output's étendue is a single channel's up to the
    residual pointing ``dispersion`` × (λ − λ_centre) [rad/m] leaves (a grating's angular dispersion times the channel's
    detuning from the design wavelength, or its linewidth). ``efficiency_per_channel`` the element's (a grating's 90–97 %,
    a dichroic's ~99 %); the channels' own ``loss`` is kept separately. Efficiency = power out / power in."""
    min_separation: float = 1e-9
    dispersion: float = 0.0
    efficiency_per_channel: float = 0.95
    name: str = "spectral"

    def combine(self, chans: list[Channel], grid: Grid | None = None, n: int = 256) -> Combined:
        lams = sorted(c.wavelength for c in chans)
        for a, b in zip(lams, lams[1:]):
            if b - a < self.min_separation:
                raise ValueError(f"spectral combining needs wavelengths at least {self.min_separation * 1e9:.2f} nm apart; got {a * 1e9:.3f} and {b * 1e9:.3f} nm")
        lam0 = float(np.mean(lams))
        overlapped = [c.with_(x=0.0, y=0.0, loss=1.0 - (1.0 - c.loss) * self.efficiency_per_channel) for c in chans]
        grid = grid or Grid.for_channels(overlapped, n)
        X, Y = grid.xy()
        thx, thy = grid.angles(lam0)
        I_near = np.zeros(X.shape); I_far = np.zeros(X.shape)
        for c in overlapped:
            I_near += np.abs(c.field(X, Y)) ** 2
            tilt = self.dispersion * (c.wavelength - lam0)                       # the residual pointing of this colour [rad]
            I_far += c.far_intensity(grid, thx[0], tilt)
        p_in = sum(c.power for c in chans); p_out = sum(c.power_in for c in overlapped)
        return _assemble(self.name, grid, I_near, I_far, thx, thy, p_in, p_out, [c.wavelength for c in chans], p_out / p_in if p_in else 0.0,
                         "intensities add on one axis; the étendue is one channel's plus the dispersion's residual pointing",
                         {"residual_pointing_max": max(abs(self.dispersion * (c.wavelength - lam0)) for c in chans), "wavelength_centre": lam0})


def _resample(I: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """A far-field intensity sampled on angle axis ``src`` (both axes alike) put onto ``dst`` (bilinear, zero outside), keeping its integral."""
    def interp_axis(A, s, d, axis):
        out = np.apply_along_axis(lambda row: np.interp(d, s, row, left=0.0, right=0.0), axis, A)
        return out
    J = interp_axis(interp_axis(I, src, dst, 1), src, dst, 0)
    ds, dd = float(src[1] - src[0]), float(dst[1] - dst[0])
    return J * (I.sum() * ds * ds) / (J.sum() * dd * dd) if J.sum() > 0 else J


# ------------------------------------------------------------------------------------------------- references
def equivalent_gaussian(power: float, wavelength: float, w0: float, grid: Grid, m2: float = 1.0) -> Combined:
    """One TEM₀₀ beam of the same total power at the origin, on the same grid — the reference every combined beam is
    held against (its brightness is the most that power can have at that wavelength and waist)."""
    c = Channel(GaussianBeam(wavelength, w0, 0.0, m2), power, name="equivalent")
    return IncoherentCombiner(name="single").combine([c], grid)


def phase_sensitivity(chans: list[Channel], combiner: CoherentCombiner, sigma_phases, trials: int = 20, grid: Grid | None = None, seed: int = 0) -> dict:
    """The coherent combiner's efficiency and on-axis gain against the rms phase error of the channels (Gaussian random
    phases, ``trials`` draws each): the mean and the spread, and the closed form the mean on-axis gain follows,
    1/N + (1 − 1/N) e^{−σ²} (→ e^{−σ²} for many channels)."""
    rng = np.random.default_rng(seed)
    grid = grid or Grid.for_channels(chans)
    sig = np.asarray(list(sigma_phases), dtype=float)
    eff_m, eff_s, gain_m = [], [], []
    for s in sig:
        effs, gains = [], []
        for _ in range(trials if s > 0 else 1):
            ph = rng.normal(0.0, s, len(chans))
            r = combiner.combine([c.with_(phase=c.phase + float(p)) for c, p in zip(chans, ph)], grid)
            effs.append(r.efficiency); gains.append(r.extra["axis_gain"])
        eff_m.append(float(np.mean(effs))); eff_s.append(float(np.std(effs))); gain_m.append(float(np.mean(gains)))
    return {"sigma_phase": sig, "efficiency": np.array(eff_m), "efficiency_std": np.array(eff_s), "axis_gain": np.array(gain_m),
            "closed_form": 1.0 / len(chans) + (1.0 - 1.0 / len(chans)) * np.exp(-sig ** 2)}


def power_table(results: dict) -> dict:
    """{name: the numbers an engineer compares} from {name: Combined}."""
    return {k: {"power in [W]": r.power_in, "power out [W]": r.power_out, "efficiency": r.efficiency, "conservation": r.conservation, "overlap": r.overlap,
                "σx [µm]": r.sigma_x * 1e6, "σθx [mrad]": r.sigma_tx * 1e3, "BPP x [mm·mrad]": r.bpp_x * 1e6, "M² x": r.m2_x,
                "brightness [W/(cm²·sr)]": r.brightness / 1e4, "on-axis far field [W/sr]": r.I_far_axis} for k, r in results.items()}
