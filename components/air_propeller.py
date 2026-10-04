"""Air propeller on a pod with a pylon — tractor or pusher: the geometry (from ``notebooks/designs/air_propeller.py``,
notebook 25).

One 10-inch propeller on the motor pod of a small fixed-wing drone, the pod hanging from a pylon (the wing or
a strut). The same pod and pylon in two layouts:

- ``tractor``  the propeller ahead of the pod;
- ``pusher``   the propeller behind the pod.

Frame (the CFD frame of Aeromant's rotor templates): the propeller axis is +x through the origin, the flow comes
along +x, the pylon rises along +y, units mm. The propeller hub spans ``x = ±hub_height/2``; the pod starts ``gap``
behind it (tractor) or ends ``gap`` ahead of it (pusher).

Geometry only: ``PropPod`` (with its ``pod_profile`` and ``pylon_x``) and the pylon's NACA 00xx section. The outline
for the particle movies, the wake and installation-drag models, the pod friction and the sound synthesis stay in
``notebooks/designs/air_propeller.py``.
"""
import math

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


__all__ = ["PropPod", "LAYOUTS"]
