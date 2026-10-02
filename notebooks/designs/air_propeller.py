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
leading-edge blockage for the tractor) and ``synthesize`` / ``write_wav`` (a propeller's tones and broadband as a
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
