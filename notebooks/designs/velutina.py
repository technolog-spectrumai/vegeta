"""Velutina — a medical-courier quadrotor with a slim body (notebook 24).

A printed courier for the last mile of a medical delivery: a cylindrical body with an ogive nose that is the
removable **medical capsule** (blood, samples, a defibrillator kit), four short arms near the tail carrying pusher
propellers, four cruciform fins behind them for straight fast flight, and the emergency **parachute** in the tail
cone. It flies nose first at speed (the body is the lifting and streamlined part), hovers for a precise set-down on
a pad, and flies home under power; the parachute is for a failure, not for the normal landing.

Frame: body axis along +X, nose tip at x = 0, tail at x = ``body_length``; the free stream comes along +X (the nose
meets the air first); up is +Z. The arms are at 45°, 135°, 225° and 315° about the axis (an X seen from the front),
the fins at 0°, 90°, 180° and 270° between them. ``part`` selects what is built:

- ``aircraft``   the whole machine, solid (CFD, masses, pictures)
- ``body``       the hollow printed body shell with the arm roots and the fins (FEA of the shell, printing)
- ``arm``        one arm with its motor pod, root face at x = ``arm_x`` plane (arm FEA and modes)
- ``capsule``    the nose capsule, hollow (capsule FEA, printing)
- ``fin``        one fin (printing)

Units mm; all dimensions are inputs with no optimisation inside the CAD.
"""
import math

import cadquery as cq
from vegeta.dedalus import Design, Parameter

ARM_ANGLES_DEG = (45.0, 135.0, 225.0, 315.0)
FIN_ANGLES_DEG = (0.0, 90.0, 180.0, 270.0)


class Velutina(Design):
    parameters = [
        Parameter("part", "aircraft", choices=("aircraft", "body", "arm", "capsule", "fin"), description="what to build"),
        Parameter("body_diameter", 90.0, "mm", min=40),
        Parameter("body_length", 520.0, "mm", min=200, description="nose tip to tail end"),
        Parameter("nose_length", 150.0, "mm", min=40, description="the ogive capsule (nose tip to its base joint)"),
        Parameter("tail_length", 90.0, "mm", min=20, description="tail cone (parachute bay)"),
        Parameter("tail_end_diameter", 40.0, "mm", min=10),
        Parameter("wall", 2.0, "mm", min=0.8, description="printed shell wall (body and capsule)"),
        Parameter("arm_x", 340.0, "mm", min=50, description="arm centreline from the nose tip"),
        Parameter("arm_reach", 165.0, "mm", min=60, description="axis to motor centre"),
        Parameter("arm_width", 20.0, "mm", min=6, description="arm width (along X) at the root"),
        Parameter("arm_thickness", 9.0, "mm", min=3, description="arm thickness (tangential)"),
        Parameter("arm_taper", 0.7, "", min=0.3, max=1.0, description="arm width at the pod / at the root"),
        Parameter("pod_diameter", 30.0, "mm", min=12, description="motor pod"),
        Parameter("pod_length", 40.0, "mm", min=10),
        Parameter("motor_pattern", 16.0, "mm", min=5, description="motor bolt square (M3)"),
        Parameter("motor_hole", 3.2, "mm", min=1),
        Parameter("fin_span", 80.0, "mm", min=20, description="fin height from the body surface"),
        Parameter("fin_root_chord", 100.0, "mm", min=20),
        Parameter("fin_tip_chord", 60.0, "mm", min=10),
        Parameter("fin_thickness", 4.0, "mm", min=1.5),
        Parameter("handle", True, "", description="grab handle on the nose (hand-over mode: a person takes the capsule)"),
        Parameter("handle_diameter", 50.0, "mm", min=20, description="handle loop diameter"),
        Parameter("handle_rod", 8.0, "mm", min=3, description="handle rod diameter"),
        Parameter("angle_of_attack_deg", 0.0, "deg", min=-10, max=90, description="rotate nose-up about Y (CFD)"),
    ]

    # ---------------------------------------------------------------- body of revolution
    def _half_profile(self, p, inset=0.0):
        """Half profile in the XY plane (x along the axis, y the radius), ``inset`` mm inside the skin."""
        D, L, ln, lt = p["body_diameter"], p["body_length"], p["nose_length"], p["tail_length"]
        r = D / 2 - inset
        pts = []
        # tangent ogive nose: y = r * sqrt(1 - ((x - ln) / ln)^2)-like ellipse (smooth, printable)
        for i in range(0, 13):
            a = math.pi / 2 * i / 12
            pts.append((ln - ln * math.cos(a) + inset * (1 - math.sin(a)), r * math.sin(a)))
        x_cyl_end = L - lt
        pts.append((x_cyl_end, r))
        r_end = max(p["tail_end_diameter"] / 2 - inset, 0.0)
        x_end = L - inset
        pts.append((x_end, r_end))
        wp = cq.Workplane("XY").moveTo(pts[0][0], 0.0)
        wp = wp.spline(pts[1:13], includeCurrent=True)
        for q in pts[13:]:
            wp = wp.lineTo(*q)
        wp = wp.lineTo(x_end, 0.0).close()
        return wp

    def _solid_body(self, p):
        return self._half_profile(p).revolve(360, (0, 0, 0), (1, 0, 0))

    def _shell_body(self, p):
        """Hollow body without the nose (the capsule is a separate part): the cavity is open at the capsule joint."""
        ln, w = p["nose_length"], p["wall"]
        outer = self._solid_body(p)
        inner = self._half_profile(p, inset=w).revolve(360, (0, 0, 0), (1, 0, 0))
        shell = outer.cut(inner)
        # cut the nose away: the body starts at the capsule joint (x = nose_length)
        cutter = cq.Workplane("YZ").circle(p["body_diameter"]).extrude(ln)
        shell = shell.cut(cutter)
        # bulkhead at the joint (the capsule seats on it), with a hole for the latch pin
        bulk = (cq.Workplane("YZ").workplane(offset=ln).circle(p["body_diameter"] / 2 - w + 0.01).extrude(2 * w)
                .faces(">X").workplane().hole(12.0))
        return shell.union(bulk)

    def _capsule(self, p):
        """The nose capsule: the ogive, hollow, open at its base, with a seating ring."""
        ln, w = p["nose_length"], p["wall"]
        outer = self._solid_body(p)
        inner = self._half_profile(p, inset=w).revolve(360, (0, 0, 0), (1, 0, 0))
        cap = outer.cut(inner)
        cutter = cq.Workplane("YZ").workplane(offset=ln).circle(p["body_diameter"]).extrude(p["body_length"])
        cap = cap.cut(cutter)
        ring = (cq.Workplane("YZ").workplane(offset=ln - 3 * w).circle(p["body_diameter"] / 2 - w)
                .circle(p["body_diameter"] / 2 - 2.5 * w).extrude(3 * w))
        return cap.union(ring)

    def _handle(self, p):
        """A D-shaped grab loop ahead of the nose tip, in the XZ plane: half a ring plus two legs that run back into
        the ogive (to where the shell is wide enough to hold them)."""
        d, r = p["handle_diameter"], p["handle_rod"] / 2
        ring = cq.Workplane("XZ").circle(d / 2).circle(d / 2 - 2 * r).extrude(r, both=True)
        half = ring.intersect(cq.Workplane("XY").box(d, 4 * r, 2 * d).translate((-d / 2, 0, 0)))   # the half at x < 0
        ln, R = p["nose_length"], p["body_diameter"] / 2
        z = d / 2 - r
        # the ogive radius reaches z + rod at x_leg: ellipse (x/ln - 1)^2 + (y/R)^2 = 1
        x_leg = ln * (1 - math.sqrt(max(0.0, 1 - ((z + 2 * r) / R) ** 2))) + 2 * r
        legs = cq.Workplane("YZ").workplane(offset=-r).pushPoints([(0, z), (0, -z)]).circle(r).extrude(x_leg + r)
        return half.union(legs)

    # ---------------------------------------------------------------- arms, pods, fins
    def _arm(self, p, angle_deg, with_root=True):
        """One arm along the radial direction at ``angle_deg`` (0° = +Y, 90° = +Z) with its motor pod; the pod's
        axis is along X and the propeller sits behind it (pusher). Built along +Y then rotated about X."""
        R, w, t, k = p["arm_reach"], p["arm_width"], p["arm_thickness"], p["arm_taper"]
        r0 = p["body_diameter"] / 2 - (3.0 if with_root else 0.0)      # the root sinks into the body skin
        x0 = p["arm_x"]
        # planform in the XY plane (x along the body, y radial), extruded in Z (the thickness)
        arm = (cq.Workplane("XY")
               .polyline([(x0 - w / 2, r0), (x0 + w / 2, r0), (x0 + w * k / 2, R), (x0 - w * k / 2, R)]).close()
               .extrude(t).translate((0, 0, -t / 2)))
        pd, pl = p["pod_diameter"], p["pod_length"]
        pod = (cq.Workplane("YZ").workplane(offset=x0 - pl / 2).center(R, 0.0).circle(pd / 2).extrude(pl)
               .faces("<X").edges().fillet(pd * 0.25))
        m, h = p["motor_pattern"] / 2, p["motor_hole"] / 2
        holes = (cq.Workplane("YZ").workplane(offset=x0 + pl / 2 - 10.0)
                 .pushPoints([(R + m, m), (R - m, m), (R + m, -m), (R - m, -m)]).circle(h).extrude(20.0))
        arm = arm.union(pod).cut(holes)
        return arm.rotate((0, 0, 0), (1, 0, 0), angle_deg)

    def _fin(self, p, angle_deg):
        """A tapered flat fin at the tail, trailing edge at the body end; built along +Y then rotated about X."""
        L, lt = p["body_length"], p["tail_length"]
        cr, ct, s, t = p["fin_root_chord"], p["fin_tip_chord"], p["fin_span"], p["fin_thickness"]
        r0 = p["tail_end_diameter"] / 2 - 2.0                                  # root sunk into the tail cone
        xte = L - 2.0
        fin = (cq.Workplane("XY")
               .polyline([(xte - cr, r0), (xte, r0), (xte, p["body_diameter"] / 2 + s), (xte - ct, p["body_diameter"] / 2 + s)])
               .close().extrude(t).translate((0, 0, -t / 2)))
        return fin.rotate((0, 0, 0), (1, 0, 0), angle_deg)

    # ---------------------------------------------------------------- assembly
    def build(self, p):
        if p["nose_length"] + p["tail_length"] >= p["body_length"]:
            raise ValueError("nose_length + tail_length must be smaller than body_length")
        if p["arm_x"] <= p["nose_length"] + p["arm_width"] or p["arm_x"] >= p["body_length"] - p["tail_length"]:
            raise ValueError("arm_x must lie on the cylindrical part of the body")
        if p["arm_reach"] <= p["body_diameter"] / 2 + p["pod_diameter"]:
            raise ValueError("arm_reach too short for the pod")
        part = p["part"]
        if part == "arm":
            return self._arm(p, 90.0, with_root=False)          # radial along +Z, root face at the body surface
        if part == "capsule":
            cap = self._capsule(p)
            return cap.union(self._handle(p)) if p["handle"] else cap
        if part == "fin":
            return self._fin(p, 90.0)
        if part == "body":
            body = self._shell_body(p)
            for a in FIN_ANGLES_DEG:
                body = body.union(self._fin(p, a))
            # arm root sockets: a short stub of each arm, so the arm loads can be applied to the shell
            for a in ARM_ANGLES_DEG:
                stub = self._arm(p, a)
                box = cq.Workplane("XY").box(p["arm_width"] + 4, p["body_diameter"] + 24, p["body_diameter"] + 24).translate((p["arm_x"], 0, 0))
                body = body.union(stub.intersect(box))
            return body
        aircraft = self._solid_body(p)
        if p["handle"]:
            aircraft = aircraft.union(self._handle(p))
        for a in ARM_ANGLES_DEG:
            aircraft = aircraft.union(self._arm(p, a))
        for a in FIN_ANGLES_DEG:
            aircraft = aircraft.union(self._fin(p, a))
        if p["angle_of_attack_deg"]:
            # right-hand rotation about +Y moves +x toward -z: the tail (large x) goes down, the nose comes up
            aircraft = aircraft.rotate((p["body_length"] / 2, 0, 0), (p["body_length"] / 2, 1, 0), p["angle_of_attack_deg"])
        return aircraft
