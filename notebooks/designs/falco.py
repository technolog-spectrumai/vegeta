"""FALCO — the tractor mountain aircraft: NISUS+'s wing, pack and missions on a tractor fuselage with a 20-inch propeller
in the nose and a conventional tail on one roll-wrapped carbon tube (notebook 35).

Why this shape (notebook 34, the frame study): the large propeller the mountains want for efficiency does not fit
between NISUS+'s booms — moved apart for a 20" pusher the booms overload the wing's spar and cut crow's span — and a
single tail tube is 213 g lighter, three times stiffer in bending and puts nothing into the wing's spars. So FALCO is
**NISUS+ with the fuselage and the tail frame replaced** (``class Falco(NisusPlus)``: the flapped three-piece wing with
crow, the spar joiner, the bays' logic, the trays, the skid and the marking are inherited); the fuselage is MERLIN's
tractor body (``merlin.Merlin.fuselage_profile``'s recipe: a spinner, the motor's cowl, a cylinder, a tail cone) in
NISUS+'s section rows, the tail a stabiliser across the tube and one dorsal fin at the tube's end fitting.

**The landing** (the decision of notebook 35): a belly landing with the propeller **parked** horizontal by the ESC's
brake before the flare — a vertical 20" blade reaches 100 mm below the keel skid (``layout['prop_ground_margin']`` < 0:
the number that says why), the spinner clears it by 130 mm (``prop_ground_margin_parked``). No landing gear.

Frame and conventions as ``nisus`` (wing root leading edge at the origin, x aft, y right, z up; mm; the fuselage's
axis at ``pod_axis_z`` with the wing on top). The NISUS+ parameter names are kept where the meaning is the same
(``pod_width`` = the fuselage diameter, ``nose_length`` = the spinner tip ahead of the wing's leading edge, ``boom_z``
= the tail tube's axis, ``boom_x1`` = the tail's trailing edge) so NISUS+'s modules serve FALCO through their
``design=`` seams.

``part``: ``aircraft``, ``wing``, ``flap``, ``aileron``, ``spar``, ``rear_spar``, ``spar_joiner``, ``pod`` (the fuselage
shell), ``nose`` (the spinner and the motor cowl), ``tail_tube``, ``tail_socket`` (the keel insert), ``tail_fitting``,
``tail`` (stabiliser + fin), ``motor_mount`` (the firewall), ``tray``, ``battery_tray``, ``skid``; FEA parts
``spar_fea``, ``rear_spar_fea``, ``spar_joiner_fea``, ``tail_tube_fea`` (three pieces), ``tail_socket_fea``,
``motor_mount_fea``.
"""
from __future__ import annotations

import math

import cadquery as cq
import numpy as np
from vegeta.dedalus import Parameter

import nisus
import nisus_plus
from fixed_wing import FixedWing
from nisus import naca4
from nisus_plus import NisusPlus

NAME = "Falco-Zero (autonomous mountain survey, a tractor with a parked-propeller landing, Jetson Orin onboard)"
PARTS = ("aircraft", "wing", "flap", "aileron", "spar", "rear_spar", "spar_joiner", "pod", "nose", "tail_tube", "tail_socket", "tail_fitting", "tail",
         "motor_mount", "tray", "battery_tray", "skid")
FEA_PARTS = ("spar_fea", "rear_spar_fea", "spar_joiner_fea", "tail_tube_fea", "tail_socket_fea", "motor_mount_fea")
PER_CALL = nisus_plus.PER_CALL
_DROP = {"boom_y", "boom_od", "boom_id", "boom_length", "boom_x0", "fitting_width", "fitting_ring_wall", "fitting_web", "socket_od", "motor_z", "part"}


def _np(name, default=None, **kw):
    """NISUS+'s parameter ``name`` with a new default where FALCO differs."""
    p = next(q for q in NisusPlus.parameters if q.name == name)
    f = dict(name=p.name, default=p.default, units=p.units, min=p.min, max=p.max, description=p.description, choices=p.choices)
    if default is not None:
        f["default"] = default
    f.update(kw)
    return Parameter(**f)


_CHANGED = {
    "nose_length": 330.0, "pod_length": 830.0, "pod_width": 130.0, "pod_height": 130.0, "nose_cone_length": 150.0,
    "flap_y0": 160.0, "rear_spar_od": 12.0, "rear_spar_id": 10.0, "rear_spar_half_length": 220.0,
    "fin_height": 360.0, "fin_ventral": 0.0, "fin_chord": 270.0, "boom_z": -65.0,
    "motor_diameter": 56.0, "motor_length": 58.0, "motor_downthrust_deg": 2.0, "mount_thickness": 6.0,
    "motor_bolt_a": 30.0, "motor_bolt_b": 30.0, "prop_diameter": 508.0, "prop_gap": 8.0,
    "skid_depth": 40.0, "skid_x0": -120.0, "skid_x1": 200.0, "marking": "FALCO-109",
}
_DESCRIPTIONS = {
    "nose_length": "the spinner's tip ahead of the wing's leading edge (the motor, its cowl, the ESC, the chin sensors and the Orin sit between); 330 mm balances the motor in the nose against the pack under the wing at 28 % MAC",
    "pod_length": "the fuselage from the spinner's tip to the tail cone's end (the tail tube's socket)",
    "pod_width": "the fuselage's diameter (a body of revolution: pod_height is the same)",
    "pod_height": "= pod_width (a round fuselage)",
    "nose_cone_length": "the nose piece: the spinner and the motor cowl, to the joint behind the firewall",
    "flap_y0": "flap inboard end: just outboard of the fuselage's side and the saddle's root rib (no boom fitting in the way)",
    "rear_spar_od": "a light rear tube: the saddle's rear bolt and the wing's torsion only (no booms' moment to carry)",
    "rear_spar_half_length": "through the saddle to the root ribs",
    "fin_height": "one dorsal fin on the tail tube's end fitting: on the centre line, clear of the stabiliser's tips, 82 % of NISUS+'s two fins' area",
    "fin_ventral": "none: a belly landing on the tube's end bumper",
    "fin_chord": "the fin's chord (deeper than the stabiliser's: a stiff root on the end fitting)",
    "boom_z": "the tail tube's axis: the fuselage's axis (the tail cone ends on it)",
    "motor_diameter": "motor bell (T-Motor AT5220 class, 52 mm stator)",
    "motor_length": "the bell's length (the can and the mount)",
    "motor_downthrust_deg": "a tractor's down-thrust (2° assumed: the thrust line below the wing's drag)",
    "prop_diameter": "20 inch (APC 20x15E, fixed: it brakes, regenerates and parks)",
    "prop_gap": "hub face to the motor bell",
    "skid_depth": "the wear keel: the parked spinner sits above the belly, the keel keeps it clear of the meadow in a nose-down slide",
    "marking": "the aircraft's name, raised on both sides of the fuselage (printed with it)",
}


class Falco(NisusPlus):
    """FALCO: NISUS+ with a tractor fuselage, a 20-inch propeller in the nose and a conventional tail on one tube."""

    parameters = [Parameter("part", "aircraft", choices=PARTS + FEA_PARTS, description="what to build")] + [
        _np(q.name, _CHANGED.get(q.name), **({"description": _DESCRIPTIONS[q.name]} if q.name in _DESCRIPTIONS else {}))
        for q in NisusPlus.parameters if q.name not in _DROP] + [
        Parameter("spinner_length", 32.0, "mm", min=0, description="the spinner ahead of the propeller's hub (a quarter ellipse)"),
        Parameter("hub_diameter", 44.0, "mm", min=10, description="the propeller's hub and the spinner's base (an APC 20-inch hub with its adapter)"),
        Parameter("hub_height", 14.0, "mm", min=4, description="the hub's thickness along the axis"),
        Parameter("fairing_length", 120.0, "mm", min=30, description="the cowl from the motor bell's radius to the fuselage's (a sine)"),
        Parameter("tail_cone", 200.0, "mm", min=50, description="the tail cone's length to the tube's socket"),
        Parameter("tail_x_end", 1000.0, "mm", min=600, description="the tail's trailing edge (NISUS+'s arm kept)"),
        Parameter("tail_tube_od", 30.0, "mm", min=10, description="the tail tube: roll-wrapped carbon (torsion: the frame study's finding)"),
        Parameter("tail_tube_id", 28.0, "mm", min=0),
        Parameter("tail_socket_length", 200.0, "mm", min=80, description="the tube bonded in the keel's PETG socket, from the saddle to the tail cone's end"),
        Parameter("stab_spar_od", 12.0, "mm", min=4, description="the stabiliser's carbon spar across the tube (a cantilever each side: 12/10)"),
        Parameter("stab_spar_id", 10.0, "mm", min=0),
        Parameter("park_clearance_min", 60.0, "mm", min=0, description="the parked propeller's least clearance to the ground line at touchdown the design asks for"),
    ]

    # ------------------------------------------------------------------------------------------- numbers
    @staticmethod
    def layout(p) -> dict:
        """NISUS+'s keys recomputed for the tractor: the propeller at the nose, the tail on the tube, the parked and the
        blade-down ground clearances, the transport pieces."""
        c0, c1, b = p["root_chord"], p["tip_chord"], p["span"]
        b2, lam = b / 2, c1 / c0
        yc = p["pod_width"] / 2 + 5.0
        S = b * (c0 + c1) / 2 * 1e-6
        mac = (2 / 3) * c0 * (1 + lam + lam ** 2) / (1 + lam)
        y_mac = b2 * (1 + 2 * lam) / (3 * (1 + lam))
        le_sweep = (c0 - c1) / 4
        x_mac_le = le_sweep * y_mac / b2
        x_ac = x_mac_le + 0.25 * mac
        D = p["pod_width"]
        z_axis = -D / 2
        x_nose = -p["nose_length"]                                       # the spinner's tip
        prop_x = x_nose + p["spinner_length"] + p["hub_height"] / 2       # the propeller's plane
        front_x = prop_x + p["hub_height"] / 2 + p["prop_gap"]           # the motor bell's front face
        x_m1 = front_x + p["motor_length"]                               # the bell's rear: the firewall
        x_fairing_end = front_x + p["fairing_length"]
        x_pod_end = p["pod_length"] - p["nose_length"]
        R = p["prop_diameter"] / 2
        zt = p["boom_z"]                                                 # the tube's axis
        x_end = p["tail_x_end"]
        tail_le, tail_te = x_end - p["tail_chord"], x_end
        z_stab = zt + p["tail_tube_od"] / 2 + 6.0                        # the stabiliser on the tube's saddle
        x_ac_t = tail_le + 0.25 * p["tail_chord"]
        fin_le = x_end - p["fin_chord"]
        x_ac_v = fin_le + 0.25 * p["fin_chord"]
        S_h = p["tail_span"] * p["tail_chord"] * 1e-6
        S_v1 = (p["fin_height"] + p["fin_ventral"]) * p["fin_chord"] * 1e-6
        l_t, l_v = x_ac_t - x_ac, x_ac_v - x_ac
        y_a0 = yc + (1 - p["aileron_span_frac"]) * (b2 - yc)
        c_a0 = c0 + (c1 - c0) * (y_a0 / b2)
        S_ail1 = p["aileron_chord_frac"] * (c_a0 + c1) / 2 * (b2 - y_a0) * 1e-6
        # the ground: at rest on the keel's rear corner and the tube's end bumper; the propeller blade down and parked
        belly = -p["pod_height"]
        skid_low = belly - p["skid_depth"]
        z_stub = zt - p["tail_tube_od"] / 2 - 12.0                       # the bumper under the tube's end fitting
        x_stub = x_end - 25.0
        slope = (z_stub - skid_low) / (x_stub - p["skid_x1"])
        z_line_prop = skid_low + (prop_x - p["skid_x1"]) * slope        # the ground line under the propeller's plane
        rest_pitch = math.degrees(math.atan2(z_stub - skid_low, x_stub - p["skid_x1"]))
        prop_low = z_axis - R                                            # a blade straight down
        parked_low = z_axis - p["hub_diameter"] / 2                      # parked horizontal: the hub and the spinner
        x_tube0 = x_pod_end - p["tail_socket_length"]
        pieces = {"centre section": 2 * p["wing_joint_y"], "outer panel": b2 - p["wing_joint_y"], "fuselage": x_pod_end - x_nose,
                  "tail tube with the tail": x_end - x_tube0, "tail (stabiliser)": p["tail_span"]}
        y0, y1 = p["flap_y0"], p["flap_y1"]

        def chord(y):
            return c0 + (c1 - c0) * max(0.0, (abs(y) - yc) / (b2 - yc))

        S_flap = p["flap_chord_frac"] * 0.5 * (chord(y0) + chord(y1)) * (y1 - y0) * 1e-6
        return {"S_ref": S, "AR": b ** 2 * 1e-6 / S, "mac": mac, "y_mac": y_mac, "x_mac_le": x_mac_le, "x_ac": x_ac,
                "yc": yc, "b2": b2, "taper": lam, "le_sweep_tip": le_sweep,
                "x_nose": x_nose, "x_pod_end": x_pod_end, "pod_axis_z": z_axis, "x_nose_joint": x_m1 + 12.0, "x_fairing_end": x_fairing_end,
                "x_motor0": front_x, "x_motor1": x_m1, "x_firewall": x_m1, "prop_x": prop_x, "prop_R": R, "motor_z": z_axis, "front_x": front_x,
                "boom_x1": x_end, "tail_le": tail_le, "tail_te": tail_te, "fin_le": fin_le, "boom_z": zt, "tail_z": z_stab, "fin_root_z": z_stab,
                "n_fins": 1, "x_tube0": x_tube0, "tube_free_length": tail_le - x_pod_end, "tube_length": x_end - x_tube0,
                "x_ac_tail": x_ac_t, "x_ac_fin": x_ac_v, "l_t": l_t, "l_v": l_v,
                "S_h": S_h, "S_v_each": S_v1, "S_v": S_v1, "V_h": S_h * l_t / (S * mac), "V_v": S_v1 * l_v / (S * b),
                "S_elevator": p["elevator_frac"] * S_h, "S_rudder_each": p["rudder_frac"] * S_v1,
                "S_aileron_each": S_ail1, "y_aileron0": y_a0,
                "overall_length": tail_te - x_nose, "height_over_skid": z_stab + p["fin_height"] - skid_low,
                "prop_clearance_boom": 1e9, "prop_clearance_wing_te": 1e9,
                "prop_lowest_z": prop_low, "skid_lowest_z": skid_low, "prop_ground_margin": prop_low - skid_low,
                "prop_ground_margin_resting": prop_low - z_line_prop, "prop_ground_margin_parked": parked_low - skid_low,
                "prop_ground_margin_parked_resting": parked_low - z_line_prop, "resting_pitch_deg": rest_pitch, "tail_bumper_z": z_stub,
                "prop_tip_z_top": z_axis + R, "pod_top_z": 0.0, "spar_x_root": p["spar_x_frac"] * c0,
                "S_flap_each": S_flap, "y_flap0": y0, "y_flap1": y1, "flap_span_share": 2 * (y1 - y0) / p["span"],
                "flapped_area_share": 2 * 0.5 * (chord(y0) + chord(y1)) * (y1 - y0) * 1e-6 / S,
                "y_joint": p["wing_joint_y"], "chord_joint": chord(p["wing_joint_y"]), "pieces_mm": pieces, "longest_piece_mm": max(pieces.values())}

    @staticmethod
    def pod_profile(p, n=14) -> np.ndarray:
        """MERLIN's tractor profile in NISUS+'s section rows (x, a, b, zc), a = b (a body of revolution about
        ``pod_axis_z``): the spinner (a quarter ellipse to the hub's radius), the hub, the step to the motor bell, the
        cowl (a sine from the bell's radius to the fuselage's over ``fairing_length``), the cylinder, the tail cone (a
        smoothstep down to the tube socket's radius) ending at ``x_pod_end``."""
        L = Falco.layout(p)
        R, z0 = p["pod_width"] / 2, L["pod_axis_z"]
        rh, rm = p["hub_diameter"] / 2, p["motor_diameter"] / 2
        s = np.linspace(0, 1, n)
        x_nose, prop_x, front_x = L["x_nose"], L["prop_x"], L["front_x"]
        pts = [(x_nose + p["spinner_length"] * (1 - math.cos(a)), max(rh * math.sin(a), 2.0)) for a in s * math.pi / 2]
        pts += [(prop_x + p["hub_height"] / 2, rh), (front_x - 0.01, rh), (front_x, rm)]
        pts += [(front_x + p["fairing_length"] * math.sin(a), rm + (R - rm) * math.sin(a)) for a in s[1:] * math.pi / 2]
        x_c1 = L["x_pod_end"] - p["tail_cone"]
        re = p["tail_tube_od"] / 2 + 4.0
        pts += [(x_c1, R)]
        pts += [(x_c1 + p["tail_cone"] * t, R - (R - re) * (3 * t ** 2 - 2 * t ** 3)) for t in s[1:]]
        rows = [(x, r, r, z0) for x, r in pts]
        out = [rows[0]] + [r for i, r in enumerate(rows[1:], 1) if r[0] - rows[i - 1][0] > 1e-6]
        return np.array(out)

    @staticmethod
    def bays(p) -> dict:
        """Nose to tail: the motor bay (the bell inside its cowl, the firewall behind it), the ESC behind the firewall
        with the cooling intake, the camera, the lidar and the pitot in a chin under it (the propeller ahead of the
        lens: computer vision lives with it), the Orin, the pack under the wing (it slides to set the CG against the
        motor in the nose), the flight controller tray above the pack, the tail tube's socket in the keel."""
        L = Falco.layout(p)
        w = p["pod_width"] - 2 * p["pod_wall"]
        h = p["pod_height"] - 2 * p["pod_wall"]
        xf = L["x_firewall"]
        return {"motor bay (the bell in its cowl, the firewall behind)": (L["front_x"], xf, p["motor_diameter"], p["motor_diameter"]),
                "ESC and intake (behind the firewall)": (xf + 4.0, xf + 60.0, w * 0.7, h * 0.5),
                "camera, lidar, pitot (chin)": (xf + 4.0, xf + 100.0, w * 0.8, h * 0.4),
                "computer bay (the Orin)": (xf + 60.0, xf + 170.0, w, h),
                "battery bay": (xf + 170.0, L["x_tube0"], w, h),
                "flight controller tray (above the pack)": (0.0, 130.0, w, h),
                "tail tube socket (the keel)": (L["x_tube0"], L["x_pod_end"], p["tail_tube_od"] + 8.0, p["tail_tube_od"] + 8.0)}

    # ------------------------------------------------------------------------------------------- shapes
    def _tail_tube(self, p, pieces=False):
        """The tail tube from the socket's start to the tail's end; ``pieces``: in three pieces fused without cleaning
        (in the socket, the free span, under the end fitting) for the FEA."""
        L = self.layout(p)
        cuts = [L["x_tube0"], L["x_pod_end"], L["tail_le"], p["tail_x_end"]] if pieces else [L["x_tube0"], p["tail_x_end"]]
        parts = []
        for a, b in zip(cuts[:-1], cuts[1:]):
            t = cq.Solid.makeCylinder(p["tail_tube_od"] / 2, b - a, cq.Vector(a, 0, p["boom_z"]), cq.Vector(1, 0, 0))
            if p["tail_tube_id"] > 0:
                t = t.cut(cq.Solid.makeCylinder(p["tail_tube_id"] / 2, b - a + 0.2, cq.Vector(a - 0.1, 0, p["boom_z"]), cq.Vector(1, 0, 0)))
            parts.append(t)
        return cq.Workplane("XY").add(parts[0].fuse(*parts[1:]) if len(parts) > 1 else parts[0])

    def _tail_socket(self, p, bore=True, pieces=False):
        """The PETG insert in the keel that holds the tube: a block ``tail_socket_length`` long around the tube from the
        saddle to the tail cone's end, its top flat under the wing's rear bolt. ``pieces``: the bore cut in three
        segments (the front and the exit 25 mm long, the middle) kept as separate faces for the FEA's bearing regions."""
        L = self.layout(p)
        w = p["tail_tube_od"] + 8.0
        x0, Ls = L["x_tube0"], p["tail_socket_length"]
        blk = cq.Workplane("XY").box(Ls, w, w, centered=(False, True, True)).translate((x0, 0, p["boom_z"]))
        if not bore:
            return blk
        r = p["tail_tube_od"] / 2 + 0.1
        if not pieces:
            b = cq.Workplane("YZ").workplane(offset=x0 - 1).center(0, p["boom_z"]).circle(r).extrude(Ls + 2)
            return blk.cut(b)
        seg = 25.0
        out = blk
        for xa, xb in ((x0 - 1.0, x0 + seg), (x0 + seg, x0 + Ls - seg), (x0 + Ls - seg, x0 + Ls + 1.0)):
            out = out.cut(cq.Workplane("YZ").workplane(offset=xa).center(0, p["boom_z"]).circle(r).extrude(xb - xa), clean=False)
        return out

    def _tail_fitting(self, p, side=0):
        """The printed sleeve on the tube's end that carries the stabiliser's spar (a saddle at 30 % of its chord) and
        the fin (a post), with the tail bumper under it."""
        L = self.layout(p)
        x0, x1 = L["tail_le"], p["tail_x_end"]
        r = p["tail_tube_od"] / 2 + 2.0
        sleeve = cq.Workplane("YZ").workplane(offset=x0).center(0, p["boom_z"]).circle(r).extrude(x1 - x0)
        saddle = cq.Workplane("XY").box(36.0, 2 * r + 6.0, 10.0, centered=(True, True, False)).translate((x0 + 0.3 * p["tail_chord"], 0, p["boom_z"] + r - 4.0))
        post = cq.Workplane("XY").box(0.5 * p["fin_chord"], 6.0, 30.0, centered=(True, True, False)).translate((L["fin_le"] + 0.45 * p["fin_chord"], 0, L["tail_z"] - 2.0))
        bumper = cq.Workplane("XY").box(40.0, 12.0, 12.0, centered=(True, True, False)).translate((x1 - 25.0, 0, p["boom_z"] - r - 10.0))
        fit = sleeve.union(saddle).union(post).union(bumper)
        bore = cq.Workplane("YZ").workplane(offset=x0 - 1).center(0, p["boom_z"]).circle(p["tail_tube_od"] / 2 + 0.1).extrude(x1 - x0 + 2)
        return fit.cut(bore)

    def _stab(self, p):
        L = self.layout(p)
        up, lo = naca4(0.0, 0.4, p["tail_thickness"])
        s = p["tail_span"] / 2
        stab = self._section(cq.Workplane("XZ").workplane(offset=-s), p["tail_chord"], L["tail_le"], L["tail_z"], up, lo).extrude(2 * s)
        if p["tail_incidence_deg"]:
            x = L["tail_le"] + 0.25 * p["tail_chord"]
            stab = stab.rotate((x, 0, L["tail_z"]), (x, 1, L["tail_z"]), p["tail_incidence_deg"])
        return stab

    def _fin(self, p, side=0):
        """One dorsal fin on the stabiliser's centre (``side`` ignored: NISUS+'s two fins become one)."""
        L = self.layout(p)
        up, lo = naca4(0.0, 0.4, p["tail_thickness"])
        return self._section(cq.Workplane("XY").workplane(offset=L["tail_z"]), p["fin_chord"], L["fin_le"], 0.0, up, lo).extrude(p["fin_height"])

    def _motor_mount(self, p, tilt=True):
        """The firewall: a ``mount_thickness`` PETG disc behind the motor bell at ``x_firewall`` with the motor's cross
        bolt pattern and a centre hole, a 15 mm skirt bonded inside the cowl."""
        L = self.layout(p)
        prof = self.pod_profile(p)
        xf = L["x_firewall"]
        r_loc = float(np.interp(xf, prof[:, 0], prof[:, 1])) - p["pod_wall"] - 0.3
        plate = cq.Workplane("YZ").workplane(offset=xf).center(0, L["motor_z"]).circle(r_loc).extrude(p["mount_thickness"])
        skirt = cq.Workplane("YZ").workplane(offset=xf + p["mount_thickness"]).center(0, L["motor_z"]).circle(r_loc + 0.3).circle(r_loc - 2.5).extrude(15.0)
        mount = plate.union(skirt)
        a, b = p["motor_bolt_a"] / 2, p["motor_bolt_b"] / 2
        holes = (cq.Workplane("YZ").workplane(offset=xf - 1).center(0, L["motor_z"]).pushPoints([(a, 0), (-a, 0), (0, b), (0, -b)])
                 .circle(p["motor_bolt_d"] / 2).extrude(p["mount_thickness"] + 2))
        centre = cq.Workplane("YZ").workplane(offset=xf - 1).center(0, L["motor_z"]).circle(8.0).extrude(p["mount_thickness"] + 20)
        return mount.cut(holes).cut(centre)

    def _tilt(self, shape, p):
        return shape

    def _motor(self, p):
        L = self.layout(p)
        can = cq.Workplane("YZ").workplane(offset=L["x_motor0"]).center(0, L["motor_z"]).circle(p["motor_diameter"] / 2).extrude(p["motor_length"])
        shaft = cq.Workplane("YZ").workplane(offset=L["prop_x"] - p["hub_height"] / 2).center(0, L["motor_z"]).circle(4.0).extrude(p["hub_height"] + p["prop_gap"] + 1)
        return can.union(shaft)

    def _prop_disc(self, p):
        L = self.layout(p)
        return cq.Workplane("YZ").workplane(offset=L["prop_x"] - 0.5).center(0, L["motor_z"]).circle(L["prop_R"]).extrude(1.0)

    def _prop_parked(self, p):
        """The two-blade propeller parked horizontal (a thin plate across the hub) for the drawings."""
        L = self.layout(p)
        return cq.Workplane("YZ").workplane(offset=L["prop_x"] - 1.5).center(0, L["motor_z"]).rect(2 * L["prop_R"], 28.0).extrude(3.0)

    def _skid(self, p):
        """NISUS+'s keel under the fuselage's lower line, deeper."""
        return NisusPlus._skid(self, p)

    def build(self, p):
        if p["tip_chord"] >= p["root_chord"]:
            raise ValueError("tip_chord must be smaller than root_chord")
        L = self.layout(p)
        if p["flap_y1"] >= p["wing_joint_y"] or p["flap_y0"] <= L["yc"] + 20.0:
            raise ValueError("the flap must lie between the fuselage's side (plus the root rib) and the panel joint")
        if L["y_aileron0"] <= p["wing_joint_y"]:
            raise ValueError("the aileron must start outboard of the panel joint")
        if p["joiner_od"] >= p["spar_id"]:
            raise ValueError("the joiner must slide inside the main spar")
        if L["prop_ground_margin_parked"] < p["park_clearance_min"]:
            raise ValueError(f"the parked propeller's spinner is {L['prop_ground_margin_parked']:.0f} mm above the keel's bottom (< {p['park_clearance_min']:g}): deepen the skid")
        if L["x_pod_end"] < p["root_chord"] + 20.0:
            raise ValueError("the tail cone must end behind the wing's trailing edge (lengthen pod_length)")
        if L["x_tube0"] < L["spar_x_root"] + 40.0:
            raise ValueError("the tail tube's socket must start behind the main spar's saddle (shorten tail_socket_length)")
        part = p["part"]
        if part == "wing":
            return self._wing_solid(p)
        if part == "flap":
            return self._flap_part(p, "flap")
        if part == "aileron":
            return self._flap_part(p, "aileron")
        if part == "spar":
            return self._spar(p)
        if part == "rear_spar":
            return self._rear_spar(p)
        if part == "spar_joiner":
            return self._joiner(p)
        if part == "spar_joiner_fea":
            return self._joiner(p, pieces=True)
        if part == "spar_fea":
            return self._half_tube(p, p["spar_x_frac"], p["spar_od"], p["spar_id"], p["spar_half_length"], 8)
        if part == "rear_spar_fea":
            return self._half_tube(p, p["rear_spar_x_frac"], p["rear_spar_od"], p["rear_spar_id"], p["rear_spar_half_length"], 3)
        if part == "pod":
            return self._pod_shell(p)
        if part == "nose":
            return self._pod_shell(p, x_to=L["x_nose_joint"])
        if part == "tail_tube":
            return self._tail_tube(p)
        if part == "tail_tube_fea":
            return self._tail_tube(p, pieces=True)
        if part == "tail_socket":
            return self._tail_socket(p)
        if part == "tail_socket_fea":
            return self._tail_socket(p, pieces=True)
        if part == "tail_fitting":
            return self._tail_fitting(p)
        if part == "tail":
            return self._stab(p).union(self._fin(p))
        if part in ("motor_mount", "motor_mount_fea"):
            return self._motor_mount(p)
        if part == "tray":
            return self._tray(p)
        if part == "battery_tray":
            return self._battery_tray(p)
        if part == "skid":
            return self._skid(p)
        aircraft = (self._pod_solid(p).union(self._wing_solid(p)).union(self._tail_tube(p)).union(self._tail_socket(p, bore=False))
                    .union(self._tail_fitting(p)).union(self._stab(p)).union(self._fin(p)).union(self._motor_mount(p)).union(self._motor(p)))
        skid = self._skid(p)
        if skid is not None:
            aircraft = aircraft.union(skid)
        if p["show_prop"]:
            aircraft = aircraft.union(self._prop_disc(p))
        if p["angle_of_attack_deg"]:
            aircraft = aircraft.rotate((0, 0, 0), (0, 1, 0), p["angle_of_attack_deg"])
        return aircraft

    # ------------------------------------------------------------------------------------------- the lattice's quads
    def planform(self, p=None) -> dict:
        """NISUS's planform quads with FALCO's tail: the stabiliser's halves at the tube's saddle height, one fin at y = 0."""
        p = resolve(p)
        L = self.layout(p)
        c0, c1, yc, b2 = p["root_chord"], p["tip_chord"], L["yc"], L["b2"]
        dx, dz = L["le_sweep_tip"], (b2 - yc) * math.tan(math.radians(p["dihedral_deg"]))
        centre = np.array([[0, 0, 0], [0, -yc, 0], [c0, -yc, 0], [c0, 0, 0]], float)
        panel = np.array([[0, -yc, 0], [dx, -b2, dz], [dx + c1, -b2, dz], [c0, -yc, 0]], float)
        mirror = np.array([1, -1, 1])
        wing = [centre, panel, centre * mirror, panel * mirror]
        ts, tc, zt = p["tail_span"] / 2, p["tail_chord"], L["tail_z"]
        xt = L["tail_le"]
        tail_l = np.array([[xt, 0, zt], [xt, -ts, zt], [xt + tc, -ts, zt], [xt + tc, 0, zt]], float)
        fin = np.array([[L["fin_le"], 0, zt], [L["fin_le"], 0, zt + p["fin_height"]], [p["tail_x_end"], 0, zt + p["fin_height"]], [p["tail_x_end"], 0, zt]], float)
        return {"wing": [q / 1000 for q in wing], "tail": [tail_l / 1000, tail_l * mirror / 1000], "fins": [fin / 1000], "n_fins": 1,
                "S_ref": L["S_ref"], "c_ref": L["mac"] / 1000, "b_ref": p["span"] / 1000,
                "wing_incidence_deg": p["wing_incidence_deg"], "tail_incidence_deg": p["tail_incidence_deg"],
                "x_ac_wing": L["x_ac"] / 1000, "l_t": L["l_t"] / 1000, "l_v": L["l_v"] / 1000, "S_h": L["S_h"], "S_v": L["S_v"],
                "V_h": L["V_h"], "V_v": L["V_v"], "z_tail": zt / 1000}

    # ------------------------------------------------------------------------------------------- drag
    def wetted_areas(self, p=None) -> dict:
        p = resolve(p)
        L = self.layout(p)
        c0, c1 = p["root_chord"], p["tip_chord"]
        S = L["S_ref"]
        S_exposed = S - p["pod_width"] * c0 * 1e-6
        prof = self.pod_profile(p)
        fus = nisus._frusta_ellipse(prof) * 1e-6
        tube = math.pi * p["tail_tube_od"] * 1e-3 * L["tube_free_length"] * 1e-3
        tail = 2 * L["S_h"] * 1.02
        fins = 2 * L["S_v"] * 1.02
        return {"wing": 2 * 1.02 * S_exposed, "pod": fus, "booms": tube, "tail": tail, "fins": fins,
                "wing_mac": L["mac"] / 1000, "pod_length": (prof[-1, 0] - prof[0, 0]) / 1000, "pod_diameter": p["pod_width"] / 1000,
                "boom_length": L["tube_free_length"] / 1000, "boom_diameter": p["tail_tube_od"] / 1000, "tail_chord": p["tail_chord"] / 1000,
                "fin_chord": p["fin_chord"] / 1000, "planform": S, "aspect_ratio": L["AR"], "span": p["span"] / 1000,
                "thickness": p["thickness"], "tail_t_c": p["tail_thickness"]}

    def drag_buildup(self, p=None, speed: float = 20.0, nu: float = 1.5e-5) -> dict:
        """NISUS's build-up (Raymer) on FALCO: the fuselage as a body of revolution (``1 + 60/f³ + f/400``), the tail
        tube likewise, the wing and the tail plates, plus 10 % interference and the extra drag area; the tractor's
        wash over the fuselage and the wing's root is not in it (ASSUMED absent; the CFD's job)."""
        a = self.wetted_areas(p)

        def cf(length):
            re = max(speed * length / nu, 1e4)
            return 0.455 / math.log10(re) ** 2.58

        t, tt = a["thickness"], a["tail_t_c"]
        f_pod = a["pod_length"] / a["pod_diameter"]
        f_tube = a["boom_length"] / a["boom_diameter"]
        parts = {"wing": cf(a["wing_mac"]) * (1 + 2 * t + 60 * t ** 4) * a["wing"],
                 "pod": cf(a["pod_length"]) * (1 + 60 / f_pod ** 3 + f_pod / 400) * a["pod"],
                 "booms": cf(a["boom_length"]) * (1 + 60 / f_tube ** 3 + f_tube / 400) * a["booms"],
                 "tail": cf(a["tail_chord"]) * (1 + 2 * tt + 60 * tt ** 4) * a["tail"],
                 "fins": cf(a["fin_chord"]) * (1 + 2 * tt + 60 * tt ** 4) * a["fins"]}
        total = 1.10 * sum(parts.values()) + nisus.EXTRA_CD_AREA_M2
        return {"parts_m2": parts, "cd_area_m2": total, "cd0": total / a["planform"], "planform": a["planform"], "aspect_ratio": a["aspect_ratio"],
                "note": "flat-plate build-up; the propeller's wash over the fuselage and the wing root (a tractor) is not in it"}


# --------------------------------------------------------------------------------------------------- helpers
def overrides(p=None) -> dict:
    return {k: v for k, v in dict(p or {}).items() if k not in PER_CALL}


def resolve(p=None, **kw) -> dict:
    return nisus.resolve(p, Falco(), **kw)


def planform(p=None) -> dict:
    return Falco().planform(p)


def planform_split(p=None) -> dict:
    return nisus_plus.planform_split(p, Falco())


def wetted_areas(p=None) -> dict:
    return Falco().wetted_areas(p)


def drag_buildup(p=None, speed: float = 20.0, nu: float = 1.5e-5) -> dict:
    return Falco().drag_buildup(p, speed, nu)


def exploded_parts(p=None, spread=1.0) -> dict:
    """The parts moved apart for the exploded view: the wing in three pieces with its spars and joiners, the fuselage
    and the nose cowl, the tail tube, its socket and its end fitting, the stabiliser and the fin, the firewall, the
    motor and the propeller disc, the trays and the skid."""
    d = Falco()
    p = resolve(p)
    L = Falco.layout(p)
    s = 1.4 * spread
    parts = {
        "wing": d._wing_solid(p).translate((0, 0, 120 * s)),
        "spar": d._spar(p).translate((0, 0, 190 * s)),
        "rear spar": d._rear_spar(p).translate((0, 0, 170 * s)),
        "flap R": d._flap_part(p, "flap").translate((90 * s, 40 * s, 120 * s)),
        "aileron R": d._flap_part(p, "aileron").translate((90 * s, 160 * s, 120 * s)),
        "spar joiner R": d._joiner(p).translate((0, 60 * s, 220 * s)),
        "fuselage": d._pod_shell(p, x_from=L["x_nose_joint"]),
        "nose: spinner and cowl": d._pod_shell(p, x_to=L["x_nose_joint"]).translate((-90 * s, 0, 0)),
        "firewall": d._motor_mount(p).translate((-60 * s, 0, -70 * s)),
        "motor": d._motor(p).translate((-100 * s, 0, -70 * s)),
        "propeller disc": d._prop_disc(p).translate((-150 * s, 0, -70 * s)),
        "tail tube": d._tail_tube(p).translate((0, 0, -60 * s)),
        "tail socket": d._tail_socket(p).translate((0, 0, -120 * s)),
        "tail fitting": d._tail_fitting(p).translate((60 * s, 0, -40 * s)),
        "stabiliser": d._stab(p).translate((120 * s, 0, 0)),
        "fin": d._fin(p).translate((120 * s, 0, 60 * s)),
        "electronics tray": d._tray(p).translate((0, 0, -120 * s)),
        "battery tray": d._battery_tray(p).translate((0, 0, -150 * s)),
    }
    skid = d._skid(p)
    if skid is not None:
        parts["belly skid"] = skid.translate((0, 0, -80 * s))
    return parts


def outline(p=None) -> dict:
    """NISUS's outline keys (``top``, ``side``, ``front``, ``hinges``, ``bays``) on FALCO: the propeller at the nose, the
    tube as the one 'boom', one fin on the centre line, the parked propeller's plate in the side view."""
    d = Falco()
    p = resolve(p)
    L = Falco.layout(p)
    pl = d.planform(p)
    prof = Falco.pod_profile(p)
    x, a, b, zc = prof.T
    pod_top = np.vstack([np.c_[x, a], np.c_[x[::-1], -a[::-1]]]) / 1000
    pod_side = np.vstack([np.c_[x, zc + b], np.c_[x[::-1], (zc - b)[::-1]]]) / 1000
    c0 = p["root_chord"]
    up, lo = naca4(p["camber"], p["camber_pos"], p["thickness"])
    ai = math.radians(p["wing_incidence_deg"])
    xq = 0.25 * c0
    sec = []
    for xx, yy in up + lo[::-1]:
        X, Z = xx * c0, yy * c0
        sec.append((xq + (X - xq) * math.cos(ai) + Z * math.sin(ai), -(X - xq) * math.sin(ai) + Z * math.cos(ai)))
    wing_sec = np.array(sec) / 1000
    r = p["tail_tube_od"] / 2
    zt, x0, x1 = p["boom_z"], L["x_tube0"], p["tail_x_end"]
    tube_top = [np.array([[x0, -r], [x1, -r], [x1, r], [x0, r]]) / 1000]
    tube_side = np.array([[x0, zt - r], [x1, zt - r], [x1, zt + r], [x0, zt + r]]) / 1000
    zs = L["tail_z"]
    fin_side = np.array([[L["fin_le"], zs], [x1, zs], [x1, zs + p["fin_height"]], [L["fin_le"], zs + p["fin_height"]]]) / 1000
    tail_side = np.array([[L["tail_le"], zs - 0.5 * p["tail_thickness"] * p["tail_chord"]], [L["tail_te"], zs], [L["tail_le"], zs + 0.5 * p["tail_thickness"] * p["tail_chord"]]]) / 1000
    zm, rm = L["motor_z"], p["motor_diameter"] / 2
    motor_side = np.array([[L["x_motor0"], zm - rm], [L["x_motor1"], zm - rm], [L["x_motor1"], zm + rm], [L["x_motor0"], zm + rm]]) / 1000
    skid_side = np.array([[p["skid_x0"], -p["pod_height"]], [p["skid_x1"], -p["pod_height"]], [p["skid_x1"], L["skid_lowest_z"]], [p["skid_x0"], L["skid_lowest_z"]]]) / 1000
    th = np.linspace(0, 2 * math.pi, 64)
    i_max = int(np.argmax(a))
    pod_front = np.c_[a[i_max] * np.cos(th), zc[i_max] + b[i_max] * np.sin(th)] / 1000
    yc, b2 = L["yc"], L["b2"]
    dz = (b2 - yc) * math.tan(math.radians(p["dihedral_deg"]))
    tmax = p["thickness"] * c0
    wing_front = np.array([[-b2, dz], [-yc, 0], [yc, 0], [b2, dz], [b2, dz + tmax * p["tip_chord"] / c0], [yc, tmax], [-yc, tmax], [-b2, dz + tmax * p["tip_chord"] / c0]]) / 1000
    tube_front = [np.c_[r * np.cos(th), zt + r * np.sin(th)] / 1000]
    fins_front = [np.array([[-5, zs], [5, zs], [5, zs + p["fin_height"]], [-5, zs + p["fin_height"]]]) / 1000]
    stab_front = np.array([[-p["tail_span"] / 2, zs - 4], [p["tail_span"] / 2, zs - 4], [p["tail_span"] / 2, zs + 4], [-p["tail_span"] / 2, zs + 4]]) / 1000
    prop_front = np.c_[L["prop_R"] * np.cos(th), zm + L["prop_R"] * np.sin(th)] / 1000
    skid_front = np.array([[-3, -p["pod_height"]], [3, -p["pod_height"]], [3, L["skid_lowest_z"]], [-3, L["skid_lowest_z"]]]) / 1000
    ya0 = L["y_aileron0"]
    hinges_top = []
    for s in (1, -1):
        f0 = (ya0 - yc) / (b2 - yc)
        c_a0 = c0 + (p["tip_chord"] - c0) * f0
        x_h0 = L["le_sweep_tip"] * f0 + (1 - p["aileron_chord_frac"]) * c_a0
        x_h1 = L["le_sweep_tip"] + (1 - p["aileron_chord_frac"]) * p["tip_chord"]
        hinges_top.append(np.array([[x_h0, s * ya0], [x_h1, s * b2]]) / 1000)
        xa, _ = d._hinge(p, p["flap_y0"], 1 - p["flap_chord_frac"])
        xb, _ = d._hinge(p, p["flap_y1"], 1 - p["flap_chord_frac"])
        hinges_top.append(np.array([[xa, s * p["flap_y0"]], [xb, s * p["flap_y1"]]]) / 1000)
    hinges_top.append(np.array([[L["tail_le"] + (1 - p["elevator_frac"]) * p["tail_chord"], -p["tail_span"] / 2],
                                [L["tail_le"] + (1 - p["elevator_frac"]) * p["tail_chord"], p["tail_span"] / 2]]) / 1000)
    hinge_side = np.array([[L["fin_le"] + (1 - p["rudder_frac"]) * p["fin_chord"], zs], [L["fin_le"] + (1 - p["rudder_frac"]) * p["fin_chord"], zs + p["fin_height"]]]) / 1000
    joints = []
    for s in (1, -1):
        xj = L["le_sweep_tip"] * (p["wing_joint_y"] - yc) / (b2 - yc)
        joints.append(np.array([[xj, s * p["wing_joint_y"]], [xj + L["chord_joint"], s * p["wing_joint_y"]]]) / 1000)
    bays = {k: (v[0] / 1000, v[1] / 1000, v[2] / 1000, v[3] / 1000) for k, v in Falco.bays(p).items()}
    return {"top": {"pod": pod_top, "wing": [q[:, :2] for q in pl["wing"]], "tail": [q[:, :2] for q in pl["tail"]], "booms": tube_top,
                    "fins": [np.array([[L["fin_le"], -4], [x1, -4], [x1, 4], [L["fin_le"], 4]]) / 1000],
                    "motor": np.array([[L["x_motor0"], -rm], [L["x_motor1"], -rm], [L["x_motor1"], rm], [L["x_motor0"], rm]]) / 1000,
                    "prop": np.array([[L["prop_x"], -L["prop_R"]], [L["prop_x"], L["prop_R"]]]) / 1000, "hinges": hinges_top, "joints": joints},
            "side": {"pod": pod_side, "wing_section": wing_sec, "boom": tube_side, "fin": fin_side, "tail": tail_side, "motor": motor_side, "skid": skid_side,
                     "prop": np.array([[L["prop_x"], zm - L["prop_R"]], [L["prop_x"], zm + L["prop_R"]]]) / 1000,
                     "prop_parked": np.array([[L["prop_x"] - 2, zm - p["hub_diameter"] / 2], [L["prop_x"] + 2, zm - p["hub_diameter"] / 2],
                                              [L["prop_x"] + 2, zm + p["hub_diameter"] / 2], [L["prop_x"] - 2, zm + p["hub_diameter"] / 2]]) / 1000,
                     "hinges": [hinge_side]},
            "front": {"pod": pod_front, "wing": wing_front, "booms": tube_front, "fins": fins_front, "stab": stab_front, "prop": prop_front, "skid": skid_front},
            "bays": bays, "layout": L, "pod_axis_z": L["pod_axis_z"] / 1000, "prop_x": L["prop_x"] / 1000}


def transport_check(p=None, backpack_mm: float = 1150.0) -> dict:
    p = resolve(p)
    L = Falco.layout(p)
    return {"pieces_mm": L["pieces_mm"], "longest_mm": L["longest_piece_mm"], "backpack_mm": backpack_mm, "fits": L["longest_piece_mm"] <= backpack_mm,
            "note": "the centre section, two outer panels on the joiner, the fuselage with the motor and the spinner, the tail tube with the tail "
                    "(it pulls out of the keel socket), the propeller off"}


__all__ = ["Falco", "NAME", "PARTS", "FEA_PARTS", "PER_CALL", "overrides", "resolve", "planform", "planform_split", "wetted_areas", "drag_buildup",
           "exploded_parts", "outline", "transport_check"]
