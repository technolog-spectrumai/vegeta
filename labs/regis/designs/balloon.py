import math

import cadquery as cq
from vegeta.dedalus import Design, Parameter


class Balloon(Design):
    """A small tethered survey balloon, deliberately simple: a round envelope with a short filling neck, a rigid
    load ring, ``rope_count`` suspension lines and a box gondola for the instruments. Regis shows it as the Wraith.
    x forward, y left, z up; z = 0 the gondola's underside."""

    parameters = [
        Parameter("envelope_diameter", 1800.0, "mm", min=300, description="envelope (sphere) diameter"),
        Parameter("neck_diameter", 260.0, "mm", min=40, description="filling neck under the envelope"),
        Parameter("neck_length", 220.0, "mm", min=20),
        Parameter("ring_diameter", 520.0, "mm", min=80, description="load ring the lines hang from"),
        Parameter("ring_section", 24.0, "mm", min=4, description="load ring tube diameter"),
        Parameter("drop", 700.0, "mm", min=100, description="load ring to gondola top"),
        Parameter("rope_count", 4, "", min=3, max=12, description="suspension lines"),
        Parameter("rope_diameter", 10.0, "mm", min=2),
        Parameter("gondola", 380.0, "mm", min=60, description="gondola width and depth"),
        Parameter("gondola_height", 300.0, "mm", min=40),
    ]

    def build(self, p):
        g, gh = p["gondola"], p["gondola_height"]
        ring_z = gh + p["drop"]
        neck_top = ring_z + p["neck_length"]
        r_env = p["envelope_diameter"] / 2
        envelope = cq.Workplane("XY").sphere(r_env).translate((0, 0, neck_top + r_env * 0.92))
        neck = cq.Workplane("XY").workplane(offset=ring_z).circle(p["neck_diameter"] / 2).extrude(p["neck_length"] + r_env * 0.2)
        ring = cq.Workplane("XY").workplane(offset=ring_z).circle(p["ring_diameter"] / 2) \
            .circle(p["ring_diameter"] / 2 - p["ring_section"]).extrude(p["ring_section"])
        gondola = cq.Workplane("XY").box(g, g, gh, centered=(True, True, False)).edges("|Z").fillet(min(g, gh) * 0.08)
        ropes = None
        n = int(p["rope_count"])
        for k in range(n):
            a = 2 * math.pi * (k + 0.5) / n
            top = cq.Vector(math.cos(a) * (p["ring_diameter"] / 2 - p["ring_section"] / 2),
                            math.sin(a) * (p["ring_diameter"] / 2 - p["ring_section"] / 2), ring_z)
            bot = cq.Vector(math.cos(a) * g * 0.45, math.sin(a) * g * 0.45, gh)
            d = top - bot
            rope = cq.Workplane(cq.Plane(origin=bot, xDir=d.cross(cq.Vector(0, 0, 1)).normalized()
                                         if abs(d.x) + abs(d.y) > 1e-6 else cq.Vector(1, 0, 0), normal=d.normalized())) \
                .circle(p["rope_diameter"] / 2).extrude(d.Length)
            ropes = rope if ropes is None else ropes.union(rope)
        return (cq.Assembly(name="balloon")
                .add(envelope, name="envelope", color=cq.Color(0.82, 0.88, 0.95, 1.0))
                .add(neck, name="neck", color=cq.Color(0.70, 0.76, 0.84, 1.0))
                .add(ring, name="load_ring", color=cq.Color(0.35, 0.37, 0.40, 1.0))
                .add(ropes, name="lines", color=cq.Color(0.55, 0.52, 0.46, 1.0))
                .add(gondola, name="gondola", color=cq.Color(0.42, 0.45, 0.50, 1.0)))
