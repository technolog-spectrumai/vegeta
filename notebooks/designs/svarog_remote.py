"""svarog_remote — welding at 20 m with many lasers crossing at one point (notebook 46_svarog_remote).

Everything here is closed form or an exact ray trace of a few thousand rays: nothing runs longer than a second.

The pieces: the Gaussian-beam budget (what beam a head must send for a given spot at the standoff), a small
three-dimensional ray tracer (spheres and planes, tilted at will, Snell's law in vector form, optical path along
the ray) for the spherical aberration and the astigmatism of the real singlets and plates, the focal spot at the
target from the traced wavefront (a Fraunhofer sum over the exit pupil: the focus of a lens IS its Fourier plane),
the incoherent sum of N such spots with pointing errors, the crossing geometry of a ring of heads, and Rosenthal's
moving point source for the weld pool.

Sign conventions: light travels +z; a sphere's radius is positive when its centre lies to the right of the vertex
(convex toward the incoming light); tilts are rotations about the x axis (the tangential plane is then y–z, the
sagittal plane x–z). Lengths in metres unless the name says otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

# refractive indices at 1070 nm (Sellmeier, Schott / Malitson)
N_BK7 = 1.5066
N_UVFS = 1.4496


# =============================================================================================== 1. the Gaussian budget
@dataclass(frozen=True)
class Laser:
    name: str
    power_W: float
    wavelength_m: float
    M2: float
    head_diameter_m: float          # 1/e² diameter of the collimated beam leaving the laser's own collimator
    source: str = ""
    verify: str = ""                # what to check on the vendor's page before ordering

    @property
    def bpp_mm_mrad(self) -> float:
        return self.M2 * self.wavelength_m / np.pi * 1e6


# the candidates: the single-mode kilowatt is what the geometry wants; the cheap multimode source and the blue diode
# show what beam quality costs at 20 m. Prices are rough street prices (2026), the specs the vendors' datasheets.
LASERS = {
    "IPG YLR-1000-SM": Laser("IPG YLR-1000-SM", 1000.0, 1.070e-6, 1.1, 5e-3,
                             source="IPG YLR datasheet: single mode, M² < 1.1, HLC-8 fibre connector, 5 mm collimator",
                             verify="collimated diameter of the HLC-8 collimator option; M² at full power"),
    "Raycus RFL-C1000 (50 µm)": Laser("Raycus RFL-C1000 (50 µm)", 1000.0, 1.080e-6, 4.4, 8e-3,
                                      source="multimode 50 µm delivery fibre, BPP ≈ 1.5 mm·mrad (M² ≈ 4.4)",
                                      verify="BPP of the exact model and fibre; 14 µm fibre versions are near single mode"),
    "blue diode 450 nm 200 W": Laser("blue diode 450 nm 200 W", 200.0, 450e-9, 60.0, 10e-3,
                                     source="fibre-coupled diode stack, 100–200 µm fibre, BPP ≈ 8 mm·mrad",
                                     verify="BPP — diode stacks vary by an order of magnitude"),
}


def spot_at(laser: Laser, w_in: float, L: float) -> dict:
    """A beam of 1/e² radius ``w_in`` at the head, focused at the standoff ``L``: the waist, the Rayleigh range and the
    peak power density there (a Gaussian: I0 = 2P / (π w0²)). w0 = M² λ L / (π w_in) is the far-field limit, exact for
    a head that sends the beam converging on L with the waist there (L ≫ z_R)."""
    w0 = laser.M2 * laser.wavelength_m * L / (np.pi * w_in)
    zR = np.pi * w0 ** 2 / (laser.M2 * laser.wavelength_m)
    I0 = 2 * laser.power_W / (np.pi * w0 ** 2)
    return {"w0_mm": w0 * 1e3, "spot_diameter_mm": 2 * w0 * 1e3, "zR_m": zR, "I0_W_cm2": I0 * 1e-4}


def head_radius_for(laser: Laser, w0: float, L: float) -> float:
    """The 1/e² radius the head must send to put a waist ``w0`` at ``L``: w_in = M² λ L / (π w0) = BPP · L / w0."""
    return laser.M2 * laser.wavelength_m * L / (np.pi * w0)


def expander_focus_offset(f2: float, L: float) -> float:
    """A Galilean expander (f1 < 0, f2 > 0) collimates at spacing f1 + f2. Lengthen the spacing by δ (the virtual
    source then sits outside L2's front focal point) and the output converges at L ≈ f2² / δ: δ = f2² / L. The focus
    knob of the head: a few tens of millimetres of travel of one lens."""
    return f2 ** 2 / L


# ================================================================================================ 2. the ray tracer
def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def rot_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


@dataclass
class Surface:
    """A refracting surface: a sphere (centre ``C``, radius ``R``, vertex ``V``) or a plane (``R`` = inf, point ``V``,
    normal ``N``); ``n_after`` is the index behind it, the index before is the previous surface's."""
    V: np.ndarray
    N: np.ndarray                    # the plane's normal, or the sphere's axis at the vertex (for the root choice)
    R: float
    n_after: float
    label: str = ""

    @property
    def C(self) -> np.ndarray:
        return self.V + self.R * self.N

    def tilted(self, angle: float, pivot: np.ndarray) -> "Surface":
        M = rot_x(angle)
        return replace(self, V=pivot + M @ (self.V - pivot), N=M @ self.N)

    def shifted(self, dz: float) -> "Surface":
        return replace(self, V=self.V + np.array([0, 0, dz]))

    def hit(self, P: np.ndarray, D: np.ndarray):
        """Intersection points and outward-facing unit normals for rays (P, D)."""
        if np.isinf(self.R):
            s = ((self.V - P) @ self.N) / (D @ self.N)
            Q = P + s[:, None] * D
            return Q, np.broadcast_to(self.N, Q.shape).copy(), s
        C = self.C
        pc = P - C
        b = np.einsum("ij,ij->i", pc, D)
        c = np.einsum("ij,ij->i", pc, pc) - self.R ** 2
        disc = b ** 2 - c
        ok = disc >= 0
        sq = np.sqrt(np.where(ok, disc, 0.0))
        s1, s2 = -b - sq, -b + sq
        Q1, Q2 = P + s1[:, None] * D, P + s2[:, None] * D
        # the root nearest the vertex region: the one whose axial distance from the vertex is the smaller
        d1 = np.abs((Q1 - self.V) @ self.N)
        d2 = np.abs((Q2 - self.V) @ self.N)
        use1 = d1 <= d2
        Q = np.where(use1[:, None], Q1, Q2)
        s = np.where(use1, s1, s2)
        s = np.where(ok, s, np.nan)
        Nrm = _unit(Q - C) * np.sign(self.R)   # points against the incoming light for a convex-toward-light surface
        return Q, Nrm, s


def _refract(D, Nrm, n1, n2):
    """Snell's law in vector form; rays that would be totally reflected come back as NaN."""
    cos_i = -np.einsum("ij,ij->i", D, Nrm)
    flip = cos_i < 0
    Nrm = np.where(flip[:, None], -Nrm, Nrm)
    cos_i = np.abs(cos_i)
    eta = n1 / n2
    k = 1 - eta ** 2 * (1 - cos_i ** 2)
    k = np.where(k < 0, np.nan, k)
    return eta * D + (eta * cos_i - np.sqrt(k))[:, None] * Nrm


@dataclass
class Train:
    """A sequence of surfaces in air (n = 1 before the first)."""
    surfaces: list = field(default_factory=list)

    def __add__(self, other: "Train") -> "Train":
        return Train(self.surfaces + other.surfaces)

    def trace(self, P: np.ndarray, D: np.ndarray):
        """Rays from points ``P`` with unit directions ``D``: the exit points, directions and the optical path length
        from ``P`` (every ray starts on the same plane wavefront)."""
        n = 1.0
        opl = np.zeros(len(P))
        for S in self.surfaces:
            Q, Nrm, s = S.hit(P, D)
            opl = opl + n * s
            D = _refract(D, Nrm, n, S.n_after)
            P, n = Q, S.n_after
        return P, D, opl


def lens(f: float, D: float, n: float = N_BK7, *, z: float = 0.0, curved_first: bool = True, edge: float = 3e-3,
         label: str = "") -> Train:
    """A plano-convex (f > 0) or plano-concave (f < 0) singlet of focal length ``f`` (EFL = R / (n − 1), exact for a
    plano lens of any thickness) and diameter ``D``; its vertex plane at ``z``. ``curved_first``: the curved face meets
    the light first (right for a positive lens receiving collimated light; for the converging-output positive lens of
    a Galilean expander the curved face should be LAST, toward the collimated side)."""
    R = (n - 1) * f
    sag = abs(R) - np.sqrt(R ** 2 - (D / 2) ** 2)
    tc = edge + sag if f > 0 else edge
    te = edge if f > 0 else edge + sag           # the concave lens is thin at the centre, thick at the edge
    ez = np.array([0.0, 0.0, 1.0])
    if curved_first:
        s1 = Surface(np.array([0, 0, z]), ez, R, n, f"{label} curved")
        s2 = Surface(np.array([0, 0, z + tc]), ez, np.inf, 1.0, f"{label} plane")
    else:
        s1 = Surface(np.array([0, 0, z]), ez, np.inf, n, f"{label} plane")
        s2 = Surface(np.array([0, 0, z + tc]), ez, -R, 1.0, f"{label} curved")
    return Train([s1, s2])


def plate(t: float, n: float = N_UVFS, *, z: float = 0.0, tilt: float = 0.0, label: str = "plate") -> Train:
    """A plane-parallel plate of thickness ``t`` whose first face passes through (0, 0, z), tilted by ``tilt`` (rad)
    about the x axis."""
    N = rot_x(tilt) @ np.array([0.0, 0.0, 1.0])
    V1 = np.array([0.0, 0.0, z])
    return Train([Surface(V1, N, np.inf, n, f"{label} in"), Surface(V1 + t * N, N, np.inf, 1.0, f"{label} out")])


def tilt_train(tr: Train, angle: float, pivot_z: float) -> Train:
    pivot = np.array([0.0, 0.0, pivot_z])
    return Train([S.tilted(angle, pivot) for S in tr.surfaces])


def galilean(f1: float, f2: float, D2: float, *, L: float = np.inf, n1: float = N_BK7, n2: float = N_BK7,
             D1: float | None = None, positive_curved_last: bool = True, D_in: float | None = None) -> tuple[Train, dict]:
    """A Galilean expander, negative lens at z = 0, positive lens at the spacing that puts the paraxial focus of the
    output at ``L`` (``L`` = inf: collimated). The thin-lens spacing f1 + f2 + f2²/L is the start; the exact spacing
    is found on the traced paraxial ray (the principal planes of the thick plano lenses move it by millimetres).
    Returns the train and its layout."""
    D1 = D1 or D2 * abs(f1) / f2 * 1.3
    D_in = D_in or D1 / 1.3

    def build(delta):
        spacing = f1 + f2 + delta
        neg = lens(f1, D1, n1, z=0.0, curved_first=False, label="L1")     # plano toward the laser, concave toward L2
        z2 = neg.surfaces[1].V[2] + spacing                                # L1's curved face is its second surface
        if positive_curved_last:
            tc = lens(f2, D2, n2).surfaces[1].V[2]
            pos = lens(f2, D2, n2, z=z2 - tc, curved_first=False, label="L2")
        else:
            pos = lens(f2, D2, n2, z=z2, curved_first=True, label="L2")
        return neg + pos, spacing

    def z_focus(delta):
        return paraxial_focus(build(delta)[0], D_in)

    delta = 0.0 if np.isinf(L) else f2 ** 2 / L
    if not np.isinf(L):                                                    # secant on 1/z_focus − 1/L
        a, b = 0.5 * delta, 1.5 * delta
        ga, gb = 1 / z_focus(a) - 1 / L, 1 / z_focus(b) - 1 / L
        for _ in range(30):
            c = b - gb * (b - a) / (gb - ga)
            gc = 1 / z_focus(c) - 1 / L
            a, ga, b, gb = b, gb, c, gc
            if abs(gc) < 1e-9:
                break
        delta = b
    tr, spacing = build(delta)
    return tr, {"f1": f1, "f2": f2, "spacing": spacing, "delta": delta, "delta_thin_lens": 0.0 if np.isinf(L) else f2 ** 2 / L,
                "magnification": f2 / abs(f1), "D1": D1, "D2": D2, "D_in": D_in, "z_focus": z_focus(delta)}


def refocused(train: Train, L: float, D_in: float, *, last: int = 2) -> tuple[Train, float]:
    """Slide the last ``last`` surfaces (the output lens: the head's focus knob) along z until the paraxial focus is at
    ``L``; returns the train and the shift (m). What the operator does after a plate goes in or the standoff changes."""
    head, tail = Train(train.surfaces[:-last]), train.surfaces[-last:]

    def z_focus(dz):
        return paraxial_focus(head + Train([S.shifted(dz) for S in tail]), D_in)

    a, b = 0.0, 1e-3
    ga, gb = 1 / z_focus(a) - 1 / L, 1 / z_focus(b) - 1 / L
    for _ in range(40):
        c = b - gb * (b - a) / (gb - ga)
        c = float(np.clip(c, -0.1, 0.1))                 # the knob has 100 mm of travel
        gc = 1 / z_focus(c) - 1 / L
        a, ga, b, gb = b, gb, c, gc
        if abs(gc) < 1e-10:
            break
    return head + Train([S.shifted(b) for S in tail]), float(b)


# ============================================================================= 3. aberrations from the traced rays
def fan(train: Train, heights: np.ndarray, *, z0: float = -0.05, sagittal: bool = False, chief_y: float = 0.0):
    """A meridional (y) or sagittal (x) fan of parallel rays entering along +z at the given heights."""
    h = np.asarray(heights, float)
    P = np.zeros((len(h), 3))
    P[:, 2] = z0
    if sagittal:
        P[:, 0] = h
        P[:, 1] = chief_y
    else:
        P[:, 1] = h
    D = np.tile([0.0, 0.0, 1.0], (len(h), 1))
    return train.trace(P, D)


def paraxial_focus(train: Train, D_in: float, *, h_frac: float = 1e-3) -> float:
    """The z where two rays at ±h_frac·D_in/2 meet in the tangential plane: the paraxial focus even when the train is
    tilted and the chief ray leaves the axis."""
    h = np.array([-h_frac, h_frac]) * D_in / 2
    P, Dd, _ = fan(train, h)
    m1, m2 = Dd[0, 1] / Dd[0, 2], Dd[1, 1] / Dd[1, 2]
    return float((P[1, 1] - P[0, 1] - m2 * P[1, 2] + m1 * P[0, 2]) / (m1 - m2))


def focus_of(P, D, *, axis: int) -> np.ndarray:
    """The z where each exit ray crosses the plane ``axis`` = 0 (0: x = 0, 1: y = 0)."""
    return P[:, 2] - P[:, axis] / D[:, axis] * D[:, 2]


def longitudinal_sa(train: Train, D: float, *, n_rays: int = 9) -> dict:
    """Marginal and zonal foci against the paraxial one, from a meridional fan (positive LSA: the marginal ray focuses
    beyond the paraxial focus — overcorrected)."""
    h = np.linspace(0.02, 1.0, n_rays) * D / 2
    zp = paraxial_focus(train, D)
    P, Dd, _ = fan(train, h)
    z = focus_of(P, Dd, axis=1)
    return {"heights": h, "z_focus": z, "z_paraxial": zp, "LSA": z - zp, "LSA_marginal": z[-1] - zp}


def astigmatism(train: Train, D: float, *, h_frac: float = 0.3) -> dict:
    """Tangential (y fan) and sagittal (x fan) foci of a narrow bundle about the chief ray (the Coddington foci), from
    close zonal rays so spherical aberration stays out. The chief ray is the axial one (field angle = the tilts)."""
    h = np.array([-h_frac, h_frac]) * D / 2
    Pt, Dt, _ = fan(train, h)
    Ps, Ds, _ = fan(train, h, sagittal=True)
    # where the two rays of each pair meet, measured along z
    def meet(P, Dd, axis):
        # two lines P_i + s_i D_i: solve for the meeting z in the (axis, z) plane
        (x1, z1), (x2, z2) = (P[0, axis], P[0, 2]), (P[1, axis], P[1, 2])
        m1, m2 = Dd[0, axis] / Dd[0, 2], Dd[1, axis] / Dd[1, 2]
        return (x2 - x1 - m2 * z2 + m1 * z1) / (m1 - m2)
    zt, zs = meet(Pt, Dt, 1), meet(Ps, Ds, 0)
    return {"z_tangential": zt, "z_sagittal": zs, "astigmatism": zt - zs}


def plate_shifts(t: float, n: float, i: float) -> dict:
    """A plate of thickness ``t`` tilted by ``i`` in a converging beam: the sagittal and tangential focus shifts (along
    the axis), exact for the chief ray (Smith, Modern Optical Engineering §4.4): Δ_S = t (1 − cos i / (n cos i')),
    Δ_T = t (1 − cos³ i / (n cos³ i')), sin i = n sin i'. Their difference, t i² (n² − 1) / n³ for small tilts, is the
    astigmatism the plate adds; at i = 0 both are t (1 − 1/n), the plain focus shift."""
    ip = np.arcsin(np.sin(i) / n)
    dS = t * (1 - np.cos(i) / (n * np.cos(ip)))
    dT = t * (1 - np.cos(i) ** 3 / (n * np.cos(ip) ** 3))
    return {"sagittal": dS, "tangential": dT, "astigmatism": dT - dS, "small_angle": t * i ** 2 * (n ** 2 - 1) / n ** 3}


def plate_tilt_for(ast: float, t: float, n: float) -> float:
    """The tilt (rad) at which a plate of thickness ``t`` adds the longitudinal astigmatism ``ast`` (small-angle
    formula inverted; the ray trace then refines it)."""
    return float(np.sqrt(abs(ast) * n ** 3 / (t * (n ** 2 - 1))))


# ================================================================================= 4. the spot at the target (PSF)
def pupil_wavefront(train: Train, D_in: float, w_in: float, L_target: float, *, D_ap: float = np.inf, N: int = 256,
                    z0: float = -0.05) -> dict:
    """Trace an N×N grid of rays over the entrance beam (extent ``D_in``, Gaussian 1/e² radius ``w_in``) and return the
    exit pupil: its coordinates (the entrance grid times the traced magnification), the amplitude (the Gaussian,
    clipped by every surface a ray misses and by the exit clear aperture ``D_ap``) and the wavefront error W (m)
    against the sphere centred on the design focus (0, 0, ``L_target``) — the optical path from the entrance plane to
    that sphere, minus the chief ray's. Pointing (the mean tilt of W) is removed and reported: an alignment, not a
    blur. ``throughput`` is the fraction of the beam's power that reaches the target."""
    u = (np.arange(N) + 0.5) / N * D_in - D_in / 2
    X, Y = np.meshgrid(u, u)
    P = np.zeros((N * N, 3)); P[:, 0] = X.ravel(); P[:, 1] = Y.ravel(); P[:, 2] = z0
    Dd = np.tile([0.0, 0.0, 1.0], (N * N, 1))
    Pe, De, opl = train.trace(P, Dd)
    ok = np.isfinite(opl) & np.isfinite(Pe).all(axis=1) & (Pe[:, 0] ** 2 + Pe[:, 1] ** 2 <= (D_ap / 2) ** 2)
    # the exit pupil's scale: the traced magnification (least squares over the valid rays)
    M = float(np.sum(Pe[ok, 0] * P[ok, 0] + Pe[ok, 1] * P[ok, 1]) / np.sum(P[ok, 0] ** 2 + P[ok, 1] ** 2))
    Pc, Dc, oplc = train.trace(np.array([[0.0, 0.0, z0]]), np.array([[0.0, 0.0, 1.0]]))
    F = np.array([0.0, 0.0, L_target])
    Rref = np.linalg.norm(F - Pc[0])
    sph = Surface(F - np.array([0, 0, Rref]), np.array([0.0, 0.0, 1.0]), Rref, 1.0)   # vertex on the axis, centre F
    Pe_ok = np.where(ok[:, None], Pe, 0.0); De_ok = np.where(ok[:, None], De, [0.0, 0.0, 1.0])
    _, _, s = sph.hit(Pe_ok, De_ok)
    W = (opl + s) - (oplc[0] + sph.hit(Pc, Dc)[2][0])
    W = np.where(ok, W, np.nan).reshape(N, N)
    inside = ok.reshape(N, N)
    Xe, Ye = X * M, Y * M
    A = np.stack([np.ones(inside.sum()), Xe[inside], Ye[inside]], axis=1)
    coef, *_ = np.linalg.lstsq(A, W[inside], rcond=None)
    W_res = W - (coef[0] + coef[1] * Xe + coef[2] * Ye)
    amp = np.exp(-(X ** 2 + Y ** 2) / w_in ** 2) * inside
    du = u[1] - u[0]
    throughput = float((amp ** 2).sum() * du ** 2 / (np.pi * w_in ** 2 / 2))
    return {"x": u * M, "X": Xe, "Y": Ye, "amp": amp, "W": W_res, "inside": inside, "w_beam": w_in * M, "magnification": M,
            "pointing_rad": (float(coef[1]), float(coef[2])), "throughput": throughput,
            "rms_m": float(np.sqrt(np.nanmean(W_res[inside] ** 2))), "pv_m": float(np.nanmax(W_res[inside]) - np.nanmin(W_res[inside]))}


def psf(pw: dict, lam: float, L_target: float, power_W: float, *, pad: int = 2048, defocus: float = 0.0) -> dict:
    """The intensity at the target plane: the Fraunhofer transform of the pupil (the focus of the head is the Fourier
    plane of its exit pupil); ``defocus`` moves the plane by that much (m) through the quadratic phase. Returns the
    map (W/m²), its grid (m) and the peak (W/cm²), normalised to the power that clears the aperture."""
    X, Y, amp, W = pw["X"], pw["Y"], pw["amp"], np.nan_to_num(pw["W"])
    r2 = X ** 2 + Y ** 2
    phase = 2 * np.pi / lam * (W - defocus * r2 / (2 * L_target ** 2))
    U = np.zeros((pad, pad), complex)
    N = amp.shape[0]
    U[:N, :N] = amp * np.exp(1j * phase)
    F = np.fft.fftshift(np.fft.fft2(U))
    I = np.abs(F) ** 2
    dx_p = pw["x"][1] - pw["x"][0]
    dx_t = lam * L_target / (pad * dx_p)
    P_through = power_W * pw["throughput"]
    I *= P_through / (I.sum() * dx_t ** 2)
    xt = (np.arange(pad) - pad // 2) * dx_t
    # the 86 % (1/e²) radius about the peak: the circle holding 86.5 % of the power
    iy, ix = np.unravel_index(np.argmax(I), I.shape)
    rr = np.sqrt((xt[None, :] - xt[ix]) ** 2 + (xt[:, None] - xt[iy]) ** 2)
    order = np.argsort(rr.ravel())
    cum = np.cumsum(I.ravel()[order]) * dx_t ** 2
    r86 = rr.ravel()[order][np.searchsorted(cum, 0.865 * P_through)]
    return {"I": I, "x": xt, "dx": dx_t, "peak_W_cm2": float(I.max()) * 1e-4, "r86_mm": float(r86) * 1e3,
            "power_through_W": P_through, "peak_xy_mm": (xt[ix] * 1e3, xt[iy] * 1e3)}


def strehl(pw: dict, lam: float) -> float:
    """The Maréchal estimate from the RMS wavefront error over the (amplitude-weighted) pupil."""
    w = pw["amp"] ** 2
    ok = pw["inside"]
    rms = np.sqrt(np.sum(w[ok] * pw["W"][ok] ** 2) / np.sum(w[ok]) - (np.sum(w[ok] * pw["W"][ok]) / np.sum(w[ok])) ** 2)
    return float(np.exp(-(2 * np.pi * rms / lam) ** 2))


def best_focus(pw: dict, lam: float, L_target: float, power_W: float, span: float = 0.5, n: int = 11, **kw) -> dict:
    """The defocus (m) that gives the highest peak at the target, scanned over ± ``span``."""
    best = None
    for dz in np.linspace(-span, span, n):
        r = psf(pw, lam, L_target, power_W, defocus=dz, **kw)
        if best is None or r["peak_W_cm2"] > best["peak_W_cm2"]:
            best = {**r, "defocus_m": float(dz)}
    return best


# ============================================================================ 5. many heads crossing at one point
def ring_layout(n_heads: int, head_size: float, *, gap: float = 0.02) -> dict:
    """N heads of outer size ``head_size`` on the smallest ring that seats them side by side; the incidence angle at a
    target ``L`` away comes from ``crossing``."""
    pitch = head_size + gap
    R = max(pitch / (2 * np.sin(np.pi / n_heads)), 0.0) if n_heads > 1 else 0.0
    if n_heads == 2:
        R = pitch / 2
    return {"n": n_heads, "ring_radius_m": R, "pitch_m": pitch,
            "angles": np.arange(n_heads) * 2 * np.pi / n_heads}


def crossing(ring_radius: float, L: float, spot_radius: float, *, fraction: float = 1 / 3) -> dict:
    """Beams from a ring of radius R cross at L. Move the target along the axis by Δz and the beams separate by
    Δz · R / L; the axial tolerance for a separation of ``fraction`` of the spot radius is Δz = fraction · w0 · L / R.
    The pointing each head must hold for the same separation is fraction · w0 / L."""
    alpha = np.arctan2(ring_radius, L)
    return {"incidence_deg": np.degrees(alpha), "footprint_stretch": 1 / np.cos(alpha),
            "axial_tolerance_m": fraction * spot_radius * L / ring_radius if ring_radius > 0 else np.inf,
            "pointing_tolerance_urad": fraction * spot_radius / L * 1e6}


def sum_of_spots(spot: dict, n_heads: int, pointing_sigma_rad: float, L: float, *, rng=None, trials: int = 50) -> dict:
    """N identical spots (the same ``psf`` map) land with independent Gaussian pointing errors of ``pointing_sigma_rad``
    per axis; the intensities add (different lasers: mutually incoherent). The expected peak over ``trials`` draws and
    the perfectly aligned one."""
    rng = rng or np.random.default_rng(1)
    I, dx = spot["I"], spot["dx"]
    peaks = []
    for _ in range(trials):
        tot = np.zeros_like(I)
        for _h in range(n_heads):
            sx, sy = rng.normal(0, pointing_sigma_rad * L / dx, 2)
            tot += np.roll(np.roll(I, int(round(sy)), axis=0), int(round(sx)), axis=1)
        peaks.append(tot.max())
    peaks = np.array(peaks) * 1e-4
    return {"peak_aligned_W_cm2": n_heads * spot["peak_W_cm2"], "peak_mean_W_cm2": float(peaks.mean()),
            "peak_p10_W_cm2": float(np.percentile(peaks, 10)), "peak_p90_W_cm2": float(np.percentile(peaks, 90))}


# ======================================================================================= 6. the weld pool (Rosenthal)
@dataclass(frozen=True)
class Metal:
    name: str
    k: float            # W/(m K)
    rho: float          # kg/m³
    cp: float           # J/(kg K)
    T_melt: float       # K
    absorptivity: float  # at 1.07 µm, solid, near-normal incidence (rises once a keyhole forms)

    @property
    def kappa(self) -> float:
        return self.k / (self.rho * self.cp)


STEEL = Metal("mild steel", 35.0, 7800.0, 500.0, 1790.0, 0.35)
STAINLESS = Metal("stainless 304", 16.0, 7900.0, 500.0, 1700.0, 0.35)
ALUMINIUM = Metal("aluminium 6061", 170.0, 2700.0, 900.0, 925.0, 0.10)

KEYHOLE_W_CM2 = 1e6     # the usual threshold on steel for a keyhole (10⁶ W/cm²); conduction welding below ~10⁵


def rosenthal(metal: Metal, q_abs: float, v: float, *, T0: float = 293.0, grid_mm: float | None = None, half_mm: float | None = None) -> dict:
    """Rosenthal's moving point source on a semi-infinite body, steady state: T − T0 = q / (2π k R) · exp(−v (R + x) / 2κ)
    with the source moving along −x at speed ``v`` (so the pool trails toward +x). Returns the melt pool's width, length
    and depth from the T = T_melt isotherm (conduction mode only: a point source overestimates the centre temperature
    and knows nothing of the keyhole)."""
    r_melt = q_abs / (2 * np.pi * metal.k * (metal.T_melt - T0))            # the melt radius of the resting source: the grid follows it
    half_mm = half_mm or max(8.0, 4 * r_melt * 1e3)
    grid_mm = grid_mm or half_mm / 160
    s = np.arange(-half_mm, half_mm, grid_mm) * 1e-3
    X, Y = np.meshgrid(s, s)
    R = np.sqrt(X ** 2 + Y ** 2) + 1e-6
    T = T0 + q_abs / (2 * np.pi * metal.k * R) * np.exp(-v * (R + X) / (2 * metal.kappa))
    melt = T >= metal.T_melt
    if not melt.any():
        return {"width_mm": 0.0, "length_mm": 0.0, "depth_mm": 0.0, "T": T, "x_mm": s * 1e3}
    width = (Y[melt].max() - Y[melt].min()) * 1e3
    length = (X[melt].max() - X[melt].min()) * 1e3
    # depth: the same field below the surface along y = 0 (the point source is symmetric in y and z)
    depth = Y[melt][np.abs(X[melt]) < grid_mm * 1e-3].max() * 1e3 if (np.abs(X[melt]) < grid_mm * 1e-3).any() else width / 2
    return {"width_mm": width, "length_mm": length, "depth_mm": depth, "T": T, "x_mm": s * 1e3}


# ====================================================================================== 7. the parts on the market
# Stock optics that fit the design (street catalogue, 2026). ``verify`` names what to check on the vendor's page.
PARTS = [
    dict(role="L2, positive, 2\"", part="Thorlabs LA1727-C", spec="N-BK7 plano-convex Ø50.8 mm f = 750 mm, AR 1050–1700 nm",
         price_eur=90, verify="clear aperture (~46 mm); swap for a UVFS LA4xxx-C at kilowatt CW (thermal lens)"),
    dict(role="L2, positive, 3\"", part="Edmund 75 mm Ø PCX (NIR coat), f 500–750 mm", spec="N-BK7 or UVFS plano-convex Ø75 mm",
         price_eur=250, verify="which focal lengths are stocked in Ø75 mm; UVFS version"),
    dict(role="L1, negative, 1\"", part="Thorlabs LC1258-C", spec="N-BK7 plano-concave Ø25.4 mm f = −75 mm, AR 1050–1700 nm",
         price_eur=60, verify="the exact f (−75 mm) and the C coating; UVFS LC4xxx-C"),
    dict(role="compensator plate", part="Thorlabs WG42012 / WG41050 class", spec="UVFS window Ø50.8 mm, t = 8–12 mm, λ/10, <5 arcsec",
         price_eur=150, verify="thickness stocked in Ø2\" (8 mm, 12 mm) and the C coating"),
    dict(role="compensator plate, large", part="Edmund #54-229", spec="UVFS window Ø150 mm, t = 15 mm, λ/10, <10 arcsec, uncoated",
         price_eur=900, verify="AR coating at 1070 nm (uncoated loses 7 % and returns it to the laser)"),
    dict(role="focusing mirror (alternative)", part="Thorlabs MPD3124 class", spec="Ø3\" off-axis parabola, RFL 12\" (305 mm), protected silver",
         price_eur=1100, verify="no SA on axis, no thermal lens, but a 305 mm RFL means f/4: alignment-critical"),
    dict(role="steering", part="fast steering mirror Ø2\" (PI S-331 / Optotune MR-E-2 class)", spec="±25 mrad, ~1 kHz, µrad resolution",
         price_eur=4000, verify="aperture (the 40–50 mm beam), coating for kW at 1070 nm, resolution < 10 µrad"),
    dict(role="laser", part="IPG YLR-1000-SM", spec="1 kW CW, 1070 nm, M² < 1.1, HLC-8, 5 mm collimator", price_eur=35000,
         verify="collimator options; a 14 µm-fibre Raycus/MAX single-mode source is the cheaper equivalent"),
]


# ===================================================================================== 8. stray beams and shielding
@dataclass(frozen=True)
class Shield:
    """A shield sheet: absorptivity at 1.07 µm, conductivity, density, heat capacity, the temperature rise it may take
    (softening for the plastic, far below melting for the metals) and the sheet thickness. Clear polycarbonate is
    transparent at 1 µm (it transmits ~85 %): only the black, filled grade is a shield at all."""
    name: str
    absorptivity: float
    k: float
    rho: float
    cp: float
    dT_allow: float
    thickness: float
    note: str = ""


SHIELDS = [
    Shield("aluminium sheet 3 mm", 0.10, 170.0, 2700.0, 900.0, 250.0, 3e-3, "reflects 90 %: the reflection must be caught too"),
    Shield("steel sheet 3 mm", 0.35, 35.0, 7800.0, 500.0, 500.0, 3e-3, "absorbs most, conducts little: hot spots"),
    Shield("polycarbonate 5 mm, black", 0.90, 0.2, 1200.0, 1250.0, 100.0, 5e-3, "softens at 150 °C; clear PC is no shield (transparent at 1 µm)"),
]


def stray_radius(w_in: float, L: float, d_past_focus: float, w0: float) -> float:
    """The beam's 1/e² radius ``d`` beyond the focus: it re-expands along the same cone it came in on, w ≈ w0 + w_in·d/L
    (the geometric cone; the diffraction cone is the same beyond z_R)."""
    return w0 + w_in * d_past_focus / L


H_SURFACE = 15.0     # W/(m² K): natural convection (~8) plus radiation from a warm grey sheet (~7), per face


def shield_response(sh: Shield, P: float, w: float, *, thickness: float | None = None, rim: float = 0.5, h: float = H_SURFACE) -> dict:
    """A Gaussian beam of power ``P`` and 1/e² radius ``w`` held on a shield of thickness δ (``thickness`` overrides the
    sheet's own). Two steady states bracket a sheet. The half space, ΔT = A·P / (√(2π) k w) (Carslaw & Jaeger's
    Gaussian source; lateral conduction is all the cooling; right when δ ≫ w). The thin plate (δ ≪ w), cooled by
    radial conduction to a rim ``rim`` away and by convection and radiation ``h`` from both faces: conduction alone
    gives ΔT = A·P / (2π k δ) · (ln(ℓ / w) + 1/2) with ℓ = min(rim, √(k δ / 2h)) the length the heat travels before
    the surface loses it, and the surface loss alone caps the rise at A·I0 / (2h); the smaller of the two is the
    plate's rise (the surface cap wins when the beam is wider than ℓ, as it is far from the focus). The sheet's
    steady rise is the larger of the half-space and plate values; when it passes the allowed rise, the adiabatic
    heating time t = ρ c δ ΔT_allow / (A I0) says how long before it fails."""
    d = thickness or sh.thickness
    I0 = 2 * P / (np.pi * w ** 2)
    dT_half = sh.absorptivity * P / (np.sqrt(2 * np.pi) * sh.k * w)
    ell = min(rim, np.sqrt(sh.k * d / (2 * h)))
    dT_cond = sh.absorptivity * P / (2 * np.pi * sh.k * d) * (np.log(max(ell / w, 1.0)) + 0.5)
    dT_surf = sh.absorptivity * I0 / (2 * h)
    dT_plate = min(dT_cond, dT_surf)
    dT = max(dT_half, dT_plate)
    t_fail = sh.rho * sh.cp * d * sh.dT_allow / (sh.absorptivity * I0)
    return {"I0_W_cm2": I0 * 1e-4, "dT_half_space_K": dT_half, "dT_thin_plate_K": dT_plate, "dT_steady_K": dT, "holds": dT < sh.dT_allow, "t_to_limit_s": t_fail}


def diffuse_hazard_distance(P: float, reflectance: float, limit_W_m2: float) -> float:
    """A Lambertian reflection of ``P`` with ``reflectance``: the distance at which its irradiance ρP cosθ/(π d²) falls
    to ``limit`` (θ = 0, normal to the surface)."""
    return float(np.sqrt(reflectance * P / (np.pi * limit_W_m2)))


# ================================================================ 9. more beams, more power, farther (the scaling)
# The kilowatt-class sources by power: single-mode up to ~2–3 kW; above that the fibre is multimode and the BPP grows.
SOURCES_BY_POWER = {
    1000.0: Laser("single-mode 1 kW (IPG YLR-1000-SM class)", 1000.0, 1.070e-6, 1.1, 5e-3, source="M² < 1.1"),
    2000.0: Laser("single-mode 2 kW (IPG YLS-2000-SM class)", 2000.0, 1.070e-6, 1.3, 5e-3, source="M² ≈ 1.3", verify="M² at full power; some 2 kW SM units are 1.1–1.5"),
    10000.0: Laser("multimode 10 kW (IPG YLS-10000, 100 µm fibre)", 10000.0, 1.070e-6, 7.3, 5e-3, source="BPP ≈ 2.5 mm·mrad (M² ≈ 7.3)",
                   verify="BPP of the delivery fibre: 50 µm → ~2, 100 µm → ~4 mm·mrad"),
}


def gaussian_spot(P: float, w0: float, dx: float, n: int) -> dict:
    """A Gaussian spot of power ``P`` and 1/e² radius ``w0`` on an n×n grid of pitch ``dx`` — the same dict shape as
    ``psf`` gives, for the incoherent sums of beams the ray trace was not run for."""
    x = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(x, x)
    I = 2 * P / (np.pi * w0 ** 2) * np.exp(-2 * (X ** 2 + Y ** 2) / w0 ** 2)
    return {"I": I, "x": x, "dx": dx, "peak_W_cm2": float(I.max()) * 1e-4, "r86_mm": w0 * 1e3, "power_through_W": P}


def scenario(n_beams: int, laser: Laser, L: float, *, w_in: float, efficiency: float, head_size: float = 0.080,
             v: float = 0.01, metal: Metal = STEEL, pointing_sigma: float = 10e-6, rng=None) -> dict:
    """N beams of one source class, the same head (beam radius ``w_in`` at L2), at the standoff ``L``: the spot from
    the Gaussian budget times the traced head's ``efficiency`` (its peak over the ideal Gaussian's, the clipping and
    the residual aberration), the summed power density and its regime, Rosenthal's pool, the ring and its
    tolerances, and the Monte Carlo peak with ``pointing_sigma`` per head."""
    sp = spot_at(laser, w_in, L)
    w0 = sp["w0_mm"] * 1e-3
    peak_each = sp["I0_W_cm2"] * efficiency
    ring = ring_layout(n_beams, head_size)
    cr = crossing(ring["ring_radius_m"], L, w0)
    pool = rosenthal(metal, metal.absorptivity * n_beams * laser.power_W, v)
    g = gaussian_spot(laser.power_W * efficiency, w0, w0 / 12, 512)       # the pitch follows the spot: 12 pixels per w0
    mc = sum_of_spots(g, n_beams, pointing_sigma, L, rng=rng, trials=30)
    total = n_beams * peak_each
    return {"beams": n_beams, "P per beam [kW]": laser.power_W / 1e3, "total [kW]": n_beams * laser.power_W / 1e3, "source": laser.name, "M²": laser.M2, "L [m]": L,
            "w0 [mm]": round(w0 * 1e3, 2), "z_R [m]": round(sp["zR_m"], 2), "peak per beam [W/cm²]": peak_each, "peak, all aligned [W/cm²]": total,
            "regime": "keyhole" if total >= KEYHOLE_W_CM2 else ("conduction" if total >= 1e5 else "heating only"),
            f"peak at σ = {pointing_sigma*1e6:.0f} µrad, mean [W/cm²]": mc["peak_mean_W_cm2"], "regime at σ": "keyhole" if mc["peak_p10_W_cm2"] >= KEYHOLE_W_CM2 else ("conduction" if mc["peak_p10_W_cm2"] >= 1e5 else "heating only"),
            "pool width [mm]": round(pool["width_mm"], 1), "pool depth (conduction) [mm]": round(pool["depth_mm"], 1),
            "ring radius [m]": round(ring["ring_radius_m"], 2), "axial tolerance [mm]": round(cr["axial_tolerance_m"] * 1e3, 1), "pointing tolerance [µrad]": round(cr["pointing_tolerance_urad"], 1),
            "_pool": pool}


# ============================================================== 11. the ablation studies (FEniCSx, book `svarog`)
@dataclass(frozen=True)
class AblationSolid:
    """What the transient enthalpy solver needs of a material: the latent heats of melting and of removal (boiling
    for the metals, decomposition for the plastic), the removal temperature, the emissivity. The thermal numbers are
    the ones ``STEEL`` / ``SHIELDS`` use, so the closed forms and the FEM see the same inputs."""
    name: str
    rho: float
    c: float
    k: float
    T_melt: float
    L_melt: float
    T_fail: float
    L_fail: float
    absorptivity: float
    emissivity: float
    failure: str          # "melt" (a metal: damage is melting, removal is boiling) or "decompose" (the plastic)


ABLATION_SOLIDS = {
    "mild_steel": AblationSolid("mild steel", 7800.0, 500.0, 35.0, 1790.0, 2.7e5, 3130.0, 6.1e6, 0.35, 0.6, "melt"),
    "aluminium": AblationSolid("aluminium", 2700.0, 900.0, 170.0, 933.0, 4.0e5, 2740.0, 1.1e7, 0.10, 0.2, "melt"),
    "black_pc": AblationSolid("black polycarbonate", 1200.0, 1250.0, 0.2, 420.0, 0.0, 720.0, 1.5e6, 0.90, 0.9, "decompose"),
}

ABLATION_VARIANTS = ("20m", "50m", "100m")
# the sets at the target (beams × W per beam, the source class) and the shields (material × thickness, under one
# stray kilowatt beam this far past the focus)
TARGET_SETS = {"2x1kW": (2, 1000.0), "10x1kW": (10, 1000.0), "10x2kW": (10, 2000.0), "20x1kW": (20, 1000.0), "10x10kW": (10, 10000.0)}
SHIELD_THICKNESS = (1e-3, 3e-3, 5e-3)
SHIELD_PAST_FOCUS = (20.0, 50.0, 100.0)
HEAD_W_IN = 0.0208               # the design head's 1/e² beam radius at L2 (notebook 46 §3.1: 2", ×8)
HEAD_EFFICIENCY = 0.64           # its traced peak over the ideal Gaussian's (the clipping and the residual aberration)


def ablation_fingerprint(variant: str) -> dict:
    """The design a run is made for: the standoff, the head and the source classes. A change here makes old runs
    incompatible, as Yanus wants."""
    L = float(variant.rstrip("m"))
    return {"standoff_m": L, "head_w_in_m": HEAD_W_IN, "head_efficiency": HEAD_EFFICIENCY,
            "sources": {str(int(p)): {"M2": las.M2, "wavelength_m": las.wavelength_m} for p, las in SOURCES_BY_POWER.items()},
            "solids": {k: s.name for k, s in ABLATION_SOLIDS.items()}}


def ablation_cases(variant: str, campaign: str) -> dict:
    """The explicit cases of a chapter at the standoff ``variant``. ``target``: the steel target under each set's
    summed beams (the spot from the budget, the power the head passes); ``shield``: each shield sheet under one stray
    kilowatt beam 20, 50 and 100 m past the focus. Every case names the solid, the sheet thickness, the beam's power
    and 1/e² radius, and how long to follow it."""
    L = float(variant.rstrip("m"))
    out = {}
    if campaign == "target":
        for name, (n, P) in TARGET_SETS.items():
            las = SOURCES_BY_POWER[P]
            sp = spot_at(las, HEAD_W_IN, L)
            out[name] = {"solid": "mild_steel", "thickness": 10e-3, "power": n * P * HEAD_EFFICIENCY, "radius": sp["w0_mm"] * 1e-3, "shape": "gaussian",
                         "t_max": 2.0, "dt": 2e-4, "nr": 80, "nz": 40, "note": f"{n} × {P/1e3:g} kW ({las.name}) at {L:g} m"}
    elif campaign == "shield":
        las = SOURCES_BY_POWER[1000.0]
        w0 = spot_at(las, HEAD_W_IN, L)["w0_mm"] * 1e-3
        for key, s in ABLATION_SOLIDS.items():
            for t in SHIELD_THICKNESS:
                for d in SHIELD_PAST_FOCUS:
                    w = stray_radius(HEAD_W_IN, L, d, w0)
                    out[f"{key}_{t*1e3:g}mm_{d:g}m"] = {"solid": key, "thickness": t, "power": 1000.0 * HEAD_EFFICIENCY, "radius": w, "shape": "gaussian",
                                                        "t_max": 120.0 if key == "black_pc" else 600.0, "dt": 0.02 if key == "black_pc" else 0.1, "nr": 60, "nz": 20,
                                                        "note": f"{s.name} {t*1e3:g} mm, one stray kW {d:g} m past the focus of a {L:g} m head"}
    else:
        raise ValueError(f"unknown campaign {campaign!r}")
    return out


# ======================================================================= 12. the moving part: response, dwell, groove
def tracking_budget(w0: float, L: float, v_part: float, a_part: float) -> dict:
    """The spot must stay within w0/3 of the groove. A loop of latency τ that only follows lags v·τ; one that
    predicts the velocity leaves ½·a·τ²; so τ_max = w0 / (3 v) without prediction, √(2 w0 / (3 a)) with it. Also the
    angular rate and acceleration the steering mirror must deliver (half the target's, the mirror doubling angles),
    and the pointing in angle — the same at every distance, since the spot grows with it."""
    e = w0 / 3
    return {"error budget [mm]": e * 1e3, "τ_max, following [ms]": e / v_part * 1e3, "τ_max, velocity predicted [ms]": np.sqrt(2 * e / a_part) * 1e3,
            "mirror rate [mrad/s]": v_part / L * 1e3 / 2, "mirror acceleration [rad/s²]": a_part / L / 2, "pointing budget [µrad]": e / L * 1e6}


def dwell_times(metal: Metal, I_abs: float, w0: float, pool_width: float) -> dict:
    """How long the beams must stay on one spot. The surface melts after t_m = π (k ΔT_m / 2 q)² / κ (the 1-D
    short-time rise ΔT = 2 q √(κ t / π) / k under an absorbed flux q); the pool reaches the spot's own scale after
    w0² / κ and its steady width after (width/2)² / κ. The same t_m is how fast the beams must be blanked when the
    loop loses the groove: the time the beam takes to melt whatever it lands on."""
    dTm = metal.T_melt - 293.0
    t_melt = np.pi * (metal.k * dTm / (2 * I_abs)) ** 2 / metal.kappa
    return {"melt onset [ms]": t_melt * 1e3, "pool at spot scale [ms]": w0 ** 2 / metal.kappa * 1e3, "pool at full width [s]": (pool_width / 2) ** 2 / metal.kappa, "blanking time [ms]": t_melt * 1e3}


def seam_sensor(L: float, D: float, *, lam: float = 0.6e-6, centroid_gain: float = 10.0) -> dict:
    """A camera telescope of aperture ``D`` looking at the groove from ``L``: the diffraction resolution 1.22 λ L / D
    and the position it can centroid a bright edge, a projected laser line or the weld's own glow to (resolution /
    gain)."""
    res = 1.22 * lam * L / D
    return {"resolution [mm]": res * 1e3, "centroid [mm]": res / centroid_gain * 1e3}
