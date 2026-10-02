"""Air propeller on a pod with a pylon — tractor or pusher (notebook 25).

One 10-inch propeller on the motor pod of a small fixed-wing drone, the pod hanging from a pylon (the wing or
a strut). The same pod and pylon in two layouts:

- ``tractor``  the propeller ahead of the pod: it pulls, its slipstream washes over the pod and the pylon; the
  blades see only the weak potential-flow blockage of the pylon's leading edge ahead of them.
- ``pusher``   the propeller behind the pod: it pushes, the blades cut through the pylon's viscous wake once per
  revolution — a load pulse per revolution on every blade, the classic pusher noise and vibration penalty.

Frame (the CFD frame of Aeromant's rotor templates): the propeller axis is +x through the origin, the flow comes
along +x, the pylon rises along +y (so the movie's side view, x against y, shows it), units mm. The propeller hub
spans ``x = ±hub_height/2``; the pod starts ``gap`` behind it (tractor) or ends ``gap`` ahead of it (pusher).

Beside the CAD (``PropPod``): ``outline`` (the pod and pylon as polygons for the particle movies), ``pylon_wake``
(the wake the blades cross, as a Boreas ``WakeField``: Silverstein's airfoil wake for the pusher, a Rankine
leading-edge blockage for the tractor), ``installation_wake`` (that plus the pod's own mean share: its boundary-layer
wake behind, its nose blockage ahead), ``installation_drag`` (the extra pod and pylon drag the running propeller causes:
the thrust deduction), ``pod_friction`` and ``synthesize`` / ``write_wav`` (a propeller's tones and broadband as a
sound you can listen to, all files at one common scale so louder sounds louder).
"""
import math
import wave
from pathlib import Path

import cadquery as cq
import numpy as np
from vegeta.dedalus import Design, Parameter

LAYOUTS = ("tractor", "pusher")


def _naca00(t, n=24):
    """Closed symmetric NACA 4-digit section, unit chord, leading edge at 0 (x, y) points, cut at 99 %."""
    xs = [0.99 * 0.5 * (1 - math.cos(math.pi * i / n)) for i in range(n + 1)]
    yt = [5 * t * (0.2969 * math.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1015 * x ** 4) for x in xs]
    up = list(zip(xs, yt))
    lo = [(x, -y) for x, y in zip(xs, yt)]
    return up[::-1] + lo[1:]


class PropPod(Design):
    """The pod (a body of revolution on the propeller axis) and the pylon (a symmetric aerofoil strut rising along +y
    from the pod's axis), placed for ``layout``. One solid: the standing body of ``rotor_mrf_installed``."""

    parameters = [
        Parameter("layout", "tractor", choices=LAYOUTS, description="propeller ahead of (tractor) or behind (pusher) the pod"),
        Parameter("hub_height", 10.0, "mm", min=2, description="the propeller hub's axial length (centred on x = 0)"),
        Parameter("gap", 3.0, "mm", min=0.5, description="free space between the hub face and the pod"),
        Parameter("pod_diameter", 46.0, "mm", min=10),
        Parameter("pod_length", 240.0, "mm", min=50),
        Parameter("front_length", 50.0, "mm", min=5, description="the pod's rounded front (motor face / nose)"),
        Parameter("rear_length", 80.0, "mm", min=5, description="the pod's tapering rear (tail cone)"),
        Parameter("motor_end_diameter", 22.0, "mm", min=4, description="the pod's end at the propeller (the motor bell)"),
        Parameter("hub_diameter", 20.0, "mm", min=2, description="the propeller hub's diameter (for the flow models; not built)"),
        Parameter("pylon_chord", 90.0, "mm", min=10),
        Parameter("pylon_thickness", 0.12, "", min=0.04, max=0.3, description="thickness / chord (NACA 00xx)"),
        Parameter("pylon_height", 230.0, "mm", min=20, description="from the pod axis up to the domain-side end"),
        Parameter("pylon_gap", 64.0, "mm", min=5, description="from the propeller plane to the pylon's nearer edge"),
    ]

    @staticmethod
    def pod_profile(p):
        """Half profile (x, r) of the pod in mm, from its upstream to its downstream end, in the propeller frame."""
        L, R, re = p["pod_length"], p["pod_diameter"] / 2, p["motor_end_diameter"] / 2
        lf, lr = p["front_length"], p["rear_length"]
        if lf + lr > L:
            raise ValueError("front_length + rear_length must not exceed pod_length")
        s = np.linspace(0, 1, 13)
        if p["layout"] == "tractor":           # motor end at the front (facing the propeller), tail cone behind
            x0 = p["hub_height"] / 2 + p["gap"]
            front = [(x0 + lf * (1 - math.cos(a * math.pi / 2)), re + (R - re) * math.sin(a * math.pi / 2)) for a in s]
            rear = [(x0 + L - lr + lr * a, R - (R - 0.15 * R) * a ** 2) for a in s[1:]]
            return [(x0, 0.0)] + front + rear + [(x0 + L, 0.0)]
        x1 = -p["hub_height"] / 2 - p["gap"]   # pusher: rounded nose upstream, tail cone ending at the motor
        x0 = x1 - L
        nose = [(x0 + lf * (1 - math.cos(a * math.pi / 2)), R * math.sin(a * math.pi / 2)) for a in s]
        rear = [(x1 - lr + lr * a, R - (R - re) * (3 * a ** 2 - 2 * a ** 3)) for a in s[1:]]
        return nose + rear + [(x1, 0.0)]

    @staticmethod
    def pylon_x(p):
        """(leading edge, trailing edge) of the pylon along x, mm."""
        if p["layout"] == "tractor":
            le = p["pylon_gap"]
            return le, le + p["pylon_chord"]
        te = -p["pylon_gap"]
        return te - p["pylon_chord"], te

    def build(self, p):
        prof = self.pod_profile(p)
        pod = cq.Workplane("XY").polyline(prof).close().revolve(360, (0, 0, 0), (1, 0, 0))
        le, _ = self.pylon_x(p)
        c = p["pylon_chord"]
        sec = [(le + c * x, c * y) for x, y in _naca00(p["pylon_thickness"])]
        # the section lies in the x-z plane; extrude along +y from inside the pod to the pylon's end
        pylon = (cq.Workplane("XZ", origin=(0, 0, 0)).polyline([(x, z) for x, z in sec]).close()
                 .extrude(-p["pylon_height"]))
        return pod.union(pylon)


def outline(p, units: float = 0.001) -> dict:
    """The pod and pylon as filled polygons for ``vegeta.aeromant.movie`` (``units`` mm -> m): the side view (x, y)
    and the view along the axis (y, z)."""
    p = dict({q.name: q.default for q in PropPod.parameters}, **p)
    prof = np.array(PropPod.pod_profile(p))
    side_pod = np.vstack([prof, prof[::-1] * [1, -1]])
    le, te = PropPod.pylon_x(p)
    side_pylon = np.array([[le, 0.0], [te, 0.0], [te, p["pylon_height"]], [le, p["pylon_height"]]])
    a = np.linspace(0, 2 * math.pi, 48)
    R = p["pod_diameter"] / 2
    ax_pod = np.column_stack([R * np.cos(a), R * np.sin(a)])
    half_t = p["pylon_thickness"] * p["pylon_chord"] / 2
    ax_pylon = np.array([[0.0, -half_t], [p["pylon_height"], -half_t], [p["pylon_height"], half_t], [0.0, half_t]])
    return {"side": [side_pod * units, side_pylon * units], "axial": [ax_pod * units, ax_pylon * units]}


def pylon_wake(p, radius_mm: float, *, cd: float = 0.012, r_frac=(0.12, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0), n_phi: int = 720):
    """The axial wake fraction ``w(r/R, phi)`` the blades cross (a Boreas ``WakeField``; phi from +y towards +z, so the
    pylon is at phi = 0), from the pylon alone (the pod's own wake is axisymmetric and adds no harmonics).

    - pusher: Silverstein's airfoil wake at ``x = pylon_gap`` behind the trailing edge — centreline deficit
      ``2.42 sqrt(cd) / (x/c + 0.3)``, half width (to half deficit) ``0.68 c sqrt(cd (x/c + 0.15))``, Gaussian across;
    - tractor: the blockage of the pylon's leading edge ``pylon_gap`` ahead of it, as a 2-D Rankine half-body of the
      pylon's thickness: ``u/U = -h / (pi d)`` on the stagnation line (h the half thickness, d the distance), falling
      off as ``d^2 / (d^2 + s^2)`` across (s the lateral distance).

    Only the blade's path above the pod (where the pylon is) sees it: for ``r < pod radius`` nothing, and the pylon
    must reach beyond the tip (``pylon_height >= radius``). ``cd`` is the pylon section's drag coefficient (an input)."""
    p = dict({q.name: q.default for q in PropPod.parameters}, **p)
    from vegeta.boreas.wake import WakeField

    c, gap = p["pylon_chord"], p["pylon_gap"]
    phi = np.arange(n_phi) * 360.0 / n_phi
    r = np.asarray(r_frac, float)
    w = np.zeros((len(r), n_phi))
    for i, rf in enumerate(r):
        rr = rf * radius_mm
        if rr <= p["pod_diameter"] / 2 or rr > p["pylon_height"]:
            continue
        s = rr * np.radians((phi + 180.0) % 360.0 - 180.0)          # lateral distance from the pylon's plane, mm
        if p["layout"] == "pusher":
            xc = gap / c
            w0 = 2.42 * math.sqrt(cd) / (xc + 0.3)
            b = 0.68 * c * math.sqrt(cd * (xc + 0.15))              # half width at half deficit
            w[i] = w0 * np.exp(-math.log(2) * (s / b) ** 2)
        else:
            h = p["pylon_thickness"] * c / 2
            w[i] = h / (math.pi * gap) * gap ** 2 / (gap ** 2 + s ** 2)
    src = (f"pylon wake (Silverstein, cd {cd}, {gap:.0f} mm behind the trailing edge)" if p["layout"] == "pusher"
           else f"pylon leading-edge blockage (Rankine, {gap:.0f} mm ahead)")
    return WakeField(r, phi, w, src)


def _defaults(p):
    return dict({q.name: q.default for q in PropPod.parameters}, **p)


def pod_friction(p, speed: float, nu: float = 1.5e-5):
    """The pod's friction drag area ``Cd A`` [m^2] in a free stream (turbulent flat plate, Prandtl–Schlichting
    ``Cf = 0.455 / (log10 Re_L)^2.58``, times Hoerner's body-of-revolution form factor ``1 + 1.5 (d/L)^1.5 + 7 (d/L)^3``),
    its wetted area per profile segment [m^2] (for ``installation_drag``) and the boundary-layer thickness at the tail [m]
    (``0.37 L Re_L^-0.2``)."""
    p = _defaults(p)
    prof = np.array(PropPod.pod_profile(p)) / 1000.0
    x, r = prof[:, 0], prof[:, 1]
    seg = np.pi * (r[1:] + r[:-1]) * np.hypot(np.diff(x), np.diff(r))          # frustum side areas
    L, d = p["pod_length"] / 1000.0, p["pod_diameter"] / 1000.0
    re = speed * L / nu
    cf = 0.455 / math.log10(re) ** 2.58
    ff = 1 + 1.5 * (d / L) ** 1.5 + 7 * (d / L) ** 3
    return {"cd_area_m2": cf * ff * float(seg.sum()), "cf": cf, "form_factor": ff, "segment_area_m2": seg,
            "segment_x_m": 0.5 * (x[1:] + x[:-1]), "delta_tail_m": 0.37 * L * re ** -0.2}


def _gauss(n):
    t, w = np.polynomial.legendre.leggauss(n)
    return t, w


def disc_solid_angle(x, rho, a, n: int = 48) -> np.ndarray:
    """Solid angle [sr] of a disc of radius ``a`` (in the plane x = 0, centred on the axis) seen from the points (``x``,
    ``rho``) — arrays broadcast together. The angular integral is done exactly (``4 E(m) / ((A - B) sqrt(A + B))``), the radial
    one by Gauss–Legendre after the substitution ``s = rho + |x| tan t`` that removes the near-plane peak. On the axis it is
    ``2 pi (1 - |x| / sqrt(x^2 + a^2))``."""
    from scipy.special import ellipe

    x, rho, a = np.broadcast_arrays(np.abs(np.asarray(x, float)), np.asarray(rho, float), np.asarray(a, float))
    x = np.maximum(x, 1e-12)
    t0, t1 = np.arctan((0.0 - rho) / x), np.arctan((a - rho) / x)
    g, w = _gauss(n)
    t = 0.5 * (t1 - t0)[..., None] * g + 0.5 * (t1 + t0)[..., None]
    s = rho[..., None] + x[..., None] * np.tan(t)
    m = np.clip(4 * rho[..., None] * s / (x[..., None] ** 2 + (rho[..., None] + s) ** 2), 0.0, 1.0)
    f = 4 * s * ellipe(m) / np.sqrt(x[..., None] ** 2 + (rho[..., None] + s) ** 2)
    return 0.5 * (t1 - t0) * np.sum(w * f, axis=-1)


def _body_area(p):
    """The body of revolution the disc sees, pod + propeller hub, as (x [m], r [m]) points: the hub (``hub_diameter``,
    ``hub_height``) bridges the gap to the pod's motor end, so the body is closed (it starts and ends at r = 0)."""
    prof = [(x / 1000.0, r / 1000.0) for x, r in PropPod.pod_profile(p)]
    rh, hh = p["hub_diameter"] / 2000.0, p["hub_height"] / 2000.0
    if p["layout"] == "tractor":                                       # the hub's front face, the hub, then the pod from its motor end
        return [(-hh, 0.0), (-hh, rh), (prof[0][0], rh), *prof[1:]]
    return [*prof[:-1], (prof[-1][0], rh), (hh, rh), (hh, 0.0)]        # pusher: the pod, then the hub behind its motor end


def _line_sources(points, step: float = 0.0005):
    """Slender-body line sources of a closed body of revolution: strengths (per unit free-stream speed) ``dS`` of the
    area ``S = pi r^2`` between densely spaced stations, at the stations' mid-points. They sum to zero for a closed body."""
    xs, ss = [], []
    for (x0, r0), (x1, r1) in zip(points[:-1], points[1:]):
        n = max(1, int(math.ceil(abs(x1 - x0) / step)))
        for k in range(n):
            a, b = k / n, (k + 1) / n
            xa, xb = x0 + (x1 - x0) * a, x0 + (x1 - x0) * b
            ra, rb = r0 + (r1 - r0) * a, r0 + (r1 - r0) * b
            xs.append(0.5 * (xa + xb)); ss.append(math.pi * (rb ** 2 - ra ** 2))
    return np.array(xs), np.array(ss)


def _pylon_sheet(p, nc: int = 28, ns: int = 32):
    """The pylon as a closed thin-strut source sheet in its plane (z = 0): strengths per unit speed ``dT/dx dx dy`` (T the
    NACA 00xx thickness with a closed trailing edge) at chordwise mid-points x and span stations y (from the pod's surface to
    ``pylon_height``), metres. They sum to zero along every chord."""
    le, te = (v / 1000.0 for v in PropPod.pylon_x(p))
    c, t = p["pylon_chord"] / 1000.0, p["pylon_thickness"]
    xi = 0.5 * (1 - np.cos(np.pi * np.arange(nc + 1) / nc))
    T = 2 * 5 * t * c * (0.2969 * np.sqrt(xi) - 0.1260 * xi - 0.3516 * xi ** 2 + 0.2843 * xi ** 3 - 0.1036 * xi ** 4)
    xm = le + c * 0.5 * (xi[1:] + xi[:-1])
    y_edges = np.linspace(p["pod_diameter"] / 2000.0, p["pylon_height"] / 1000.0, ns + 1)
    ym, dy = 0.5 * (y_edges[1:] + y_edges[:-1]), np.diff(y_edges)
    q = np.diff(T)[:, None] * dy[None, :]
    X, Y = np.meshgrid(xm, ym, indexing="ij")
    Tm = 0.5 * (T[1:] + T[:-1])
    dx = c * np.diff(xi)
    return X, Y, q, Tm[:, None] * np.ones_like(Y), dx[:, None] * dy[None, :]


def potential_wake_w(p, r_m: np.ndarray, phi_deg: np.ndarray) -> np.ndarray:
    """The potential-flow (inviscid, closed-body) wake fraction ``w = -u_x / V`` on the propeller plane (x = 0) at radii
    ``r_m`` [m] (rows) and angles ``phi_deg`` (columns; 0 = towards the pylon, +y, rotating towards +z): the pod with the hub
    as slender-body line sources, the pylon as a closed source sheet. Linear in the speed, so independent of it."""
    xs, qs = _line_sources(_body_area(p))
    r = np.asarray(r_m, float)[:, None]
    d = (xs[None, :] ** 2 + r ** 2) ** 1.5
    w_pod = np.sum(qs[None, :] / (4 * np.pi) * xs[None, :] / d, axis=1)            # -u_x / V = q x_s / (4 pi d^3)
    X, Y, q, _, _ = _pylon_sheet(p)
    ph = np.radians(np.asarray(phi_deg, float))
    py, pz = (r * np.cos(ph)[None, :]), (r * np.sin(ph)[None, :])
    dx, dyy, dz = (-X.ravel())[None, None, :], py[..., None] - Y.ravel()[None, None, :], pz[..., None]
    w_pylon = np.sum(q.ravel() / (4 * np.pi) * (-dx) / (dx ** 2 + dyy ** 2 + dz ** 2) ** 1.5, axis=-1)
    return w_pod[:, None] + w_pylon


def installation_wake(p, radius_mm: float, speed: float, *, cd: float = 0.012, nu: float = 1.5e-5,
                      r_frac=(0.06, 0.09, 0.12, 0.16, 0.2, 0.28, 0.35, 0.5, 0.7, 0.85, 1.0), n_phi: int = 720):
    """The axial wake fraction ``w(r/R, phi)`` the blades work in (a Boreas ``WakeField``), both layouts modelled alike:

    - **potential flow** of the closed bodies (``potential_wake_w``): the pod and the propeller hub as slender-body line
      sources ``V dS/dx`` (the nose and the hub's front face slow the air ahead, a closing tail cone slows the air behind it),
      the pylon as a closed source sheet ``V dT/dx`` over chord and span;
    - **viscous** wakes, behind the bodies only (the pusher): the pylon's Silverstein wake (``pylon_wake``) and the pod's
      boundary-layer wake, a Gaussian ``w0 exp(-(r/b)^2)`` of width ``b = r_end + delta_tail`` whose depth comes from the pod's
      friction drag by momentum, ``Cd A / 2 = pi b^2 w0 (1 - w0/2)``.

    ``speed`` sets the Reynolds number of the pod's friction. For ``boreas.wake.effective_inflow`` (the mean) and
    ``boreas.wake.load_harmonics`` (the orders)."""
    from vegeta.boreas.wake import WakeField

    p = _defaults(p)
    r = np.asarray(r_frac, float)
    phi = np.arange(n_phi) * 360.0 / n_phi
    r_m = r * radius_mm / 1000.0
    w = potential_wake_w(p, r_m, phi)
    src = "closed bodies (pod + hub line sources, pylon source sheet)"
    if p["layout"] == "pusher":
        w = w + pylon_wake(p, radius_mm, cd=cd, r_frac=r_frac, n_phi=n_phi).w
        f = pod_friction(p, speed, nu)
        b = p["motor_end_diameter"] / 2000.0 + f["delta_tail_m"]
        k = f["cd_area_m2"] / (2 * math.pi * b ** 2)                    # w0 (1 - w0 / 2) = k
        w0 = 1 - math.sqrt(max(1 - 2 * k, 0.0))
        w = w + (w0 * np.exp(-(r_m / b) ** 2))[:, None]
        src += f" + pylon wake (Silverstein, cd {cd}) + pod boundary-layer wake (w0 {w0:.3f}, b {b * 1000:.0f} mm)"
    return WakeField(r, phi, w, src)


_FIELDS: dict = {}


def _disc_set(R, loading):
    """Nested discs (radius, weight per unit total-pressure jump) of a loaded actuator disc: ``loading`` = (station radii,
    dT/dr) gives the jumps ``dH = dT/dr / (2 pi r)`` per annulus; None is the uniform disc of radius R."""
    if loading is None:
        return np.array([R]), None
    r = np.asarray(loading[0], float)
    dr = np.gradient(r)
    edges_out = r + 0.5 * dr
    return np.concatenate([[r[0] - 0.5 * dr[0]], edges_out]), r


def _field(p, R, radii_key):
    """Influence matrices of the disc's linear pressure field ``p = sign(x) sum_k dH_k Omega_k / (4 pi)`` (nested discs of
    radii ``radii_key``) at the pod's profile segments and the pylon's panels (and at x +- 1 mm for the gradient)."""
    key = (tuple(sorted((k, v) for k, v in p.items())), R, radii_key)
    if key in _FIELDS:
        return _FIELDS[key]
    a = np.asarray(radii_key, float)
    prof = np.array(PropPod.pod_profile(p)) / 1000.0
    x, r = prof[:, 0], prof[:, 1]
    xm, rm = 0.5 * (x[1:] + x[:-1]), 0.5 * (r[1:] + r[:-1])
    G = lambda xx, rr: np.sign(xx)[..., None] * disc_solid_angle(xx[..., None], rr[..., None], a[None, :]) / (4 * np.pi)
    X, Y, _, Tm, dA = _pylon_sheet(p, nc=16, ns=20)
    h = 0.001
    f = {"pod_x": xm, "pod_r": rm, "pod_G": G(xm, rm),
         "annulus": np.pi * (r[:-1] ** 2 - r[1:] ** 2),
         "end_face": None,
         "pyl_x": X, "pyl_y": Y, "pyl_vol": Tm * dA, "pyl_area": dA, "pyl_G": G(X, Y), "pyl_dGdx": (G(X + h, Y) - G(X - h, Y)) / (2 * h)}
    x_face = (p["hub_height"] / 2 + p["gap"]) / 1000.0 * (1 if p["layout"] == "tractor" else -1)
    f["end_face"] = (np.abs(x[:-1] - x_face) < 1e-9) & (np.abs(x[1:] - x_face) < 1e-9)
    _FIELDS[key] = f
    return f


def installation_drag(p, radius_mm: float, speed: float, thrust: float, *, rho: float = 1.225, nu: float = 1.5e-5,
                      cd_pylon: float = 0.012, loading=None) -> dict:
    """The extra drag [N] the running propeller puts on the pod and the pylon (thrust deduction ``t = dD / T``), from the
    disc's pressure field in linear actuator-disc theory: ``p(x, r) = sign(x) sum_k dH_k Omega_k(x, r) / (4 pi)``, nested
    discs whose total-pressure jumps ``dH`` follow the blade loading (``loading = (r, dT/dr)`` of the blade-element solution:
    ``dH = dT/dr / (2 pi r)``, nothing over the hub) or, without it, a uniform disc ``T / A``; ``Omega_k`` the exact solid
    angle of disc k (``disc_solid_angle``). Ahead of the disc the pressure falls towards ``-dH/2``, behind it rises to
    ``+dH/2`` and recovers along the slipstream.

    - pressure on the pod: ``-p n dA`` over its profile (annuli). The flat motor-end face is left out: it faces the propeller
      hub across the small ``gap`` and the hub carries the same pressure the other way (the hub is not in the blade-element
      thrust, so both go together — as in the CFD, where they cancel in ``T - dD``);
    - pressure on the pylon: the force of a thin body in a pressure gradient, ``-sum T dA dp/dx`` over its thickness;
    - friction: every pod segment's and the pylon's friction drag scaled by the local ``(V + u)^2 / V^2`` from Bernoulli,
      ``V^2 - 2 p / rho`` ahead of the disc and ``V^2 + 2 (dH(r) - p) / rho`` behind it (the streamline through the disc at
      about the same radius carries its jump ``dH(r)``: a pod behind the blade root sees little of it).

    The swirl and the blade-passing unsteadiness are left out. Returns the parts and the total; zero thrust gives zero."""
    p = _defaults(p)
    R = radius_mm / 1000.0
    A = math.pi * R ** 2
    V = speed
    if thrust <= 0:
        return {"v_i": 0.0, "pressure_N": 0.0, "pylon_pressure_N": 0.0, "pod_friction_N": 0.0, "pylon_N": 0.0, "total_N": 0.0, "t": 0.0}
    v_i = 0.5 * (-V + math.sqrt(V ** 2 + 2 * thrust / (rho * A)))       # T = 2 rho A (V + v_i) v_i (for reference)
    radii, stations = _disc_set(R, loading)
    if loading is None:
        jumps = np.array([thrust / A])
        H_of_r = lambda rr: np.where(np.asarray(rr) <= R, thrust / A, 0.0)
    else:
        dH = np.maximum(np.asarray(loading[1], float), 0.0) / (2 * np.pi * stations)
        jumps = np.concatenate([[-dH[0]], dH - np.concatenate([dH[1:], [0.0]])])   # hub disc cancels the inner jump
        H_of_r = lambda rr: np.interp(rr, stations, dH, left=0.0, right=0.0)
    f = _field(p, R, tuple(np.round(radii, 9)))
    p_pod = f["pod_G"] @ jumps
    pressure = float(np.sum(np.where(f["end_face"], 0.0, -p_pod * f["annulus"])))
    p_pyl, dpdx = f["pyl_G"] @ jumps, f["pyl_dGdx"] @ jumps
    pylon_pressure = float(-np.sum(f["pyl_vol"] * dpdx))
    friction = pylon = 0.0
    if V > 0:
        q = 0.5 * rho * V ** 2
        fr = pod_friction(p, V, nu)
        H = np.where(f["pod_x"] > 0, H_of_r(f["pod_r"]), 0.0)
        factor = 1 + 2 * (H - p_pod) / (rho * V ** 2)
        friction = float(np.sum(q * fr["cf"] * fr["form_factor"] * fr["segment_area_m2"] * (factor - 1)))
        Hy = np.where(f["pyl_x"] > 0, H_of_r(f["pyl_y"]), 0.0)
        fac_y = 1 + 2 * (Hy - p_pyl) / (rho * V ** 2)                # per panel
        chord = p["pylon_chord"] / 1000.0
        dy = f["pyl_area"].sum(axis=0) / chord                         # span of each strip
        pylon = float(np.sum(q * cd_pylon * chord * dy * (fac_y.mean(axis=0) - 1)))   # each strip's profile drag, chord-averaged factor
    total = pressure + pylon_pressure + friction + pylon
    return {"v_i": v_i, "pressure_N": pressure, "pylon_pressure_N": pylon_pressure, "pod_friction_N": friction, "pylon_N": pylon,
            "total_N": total, "t": total / thrust}


def synthesize(tones, broadband_db: float, *, seconds: float = 3.0, rate: int = 44100, band=(400.0, 6000.0), seed: int = 0,
               p_ref: float = 20e-6) -> np.ndarray:
    """A sound pressure signal [Pa]: every tone ``(frequency_hz, spl_db)`` as a sine of that rms (random phase), plus
    broadband noise of level ``broadband_db`` (dB re ``p_ref``) shaped as band-passed noise between ``band``."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * rate)) / rate
    x = np.zeros_like(t)
    for f, L in tones:
        if f < rate / 2 and np.isfinite(L):
            x += math.sqrt(2) * p_ref * 10 ** (L / 20) * np.sin(2 * math.pi * f * t + rng.uniform(0, 2 * math.pi))
    if np.isfinite(broadband_db):
        n = rng.standard_normal(len(t))
        F = np.fft.rfft(n)
        fr = np.fft.rfftfreq(len(t), 1 / rate)
        F[(fr < band[0]) | (fr > band[1])] = 0
        nb = np.fft.irfft(F, len(t))
        nb *= p_ref * 10 ** (broadband_db / 20) / max(float(np.std(nb)), 1e-30)
        x += nb
    fade = np.minimum(1.0, np.minimum(t, t[-1] - t) / 0.05)          # no clicks at the ends
    return x * fade


def write_wav(path, signal: np.ndarray, full_scale_pa: float, rate: int = 44100) -> Path:
    """16-bit mono WAV; ``full_scale_pa`` is the pressure of a full-scale sample — use one value for every file of a
    comparison, so the louder propeller is louder on playback."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(np.asarray(signal) / full_scale_pa, -1.0, 1.0)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes((pcm * 32767).astype("<i2").tobytes())
    return path
