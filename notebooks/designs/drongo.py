import math
from dataclasses import replace

import cadquery as cq
from vegeta.dedalus import Parameter

from quad_frame import QuadFrame


def _frame_params():
    """Notebook 08's QuadFrame parameters (the X-frame Drongo flies on), unchanged."""
    return [replace(p) for p in QuadFrame.parameters]


class Drongo(QuadFrame):
    """Drongo: notebook 08's printed X-frame quadcopter (``QuadFrame``: 250 mm wheelbase, 5-inch propellers) with a
    landing gear of two skids and a pincer under the centre plate, for carrying groceries.

    The pincer is a parallel gripper: one servo in a housing under the plate turns a pinion between two racks, so the
    two jaws slide apart and together along x on a rail; each jaw is a finger with a rubber (TPU) pad at its lower
    end. The pads sit at ``grip_height`` above the ground when Drongo stands on its skids: it lands over an item,
    closes the jaws on its middle and takes it along; landed again, the item stands on the ground (the place
    variant). x forward (between two arms), y left, z up; z = 0 the frame's underside (QuadFrame's), the ground at
    z = −``skid_height`` when landed.

    ``part``: ``drongo`` (everything, with the bought parts — motors, propeller discs, battery — drawn as blocks),
    ``frame`` (QuadFrame alone), ``gear`` (both skids with their struts), ``gripper`` (servo housing and rail),
    ``jaw`` (one finger with its rack carriage, the pad bore cut out), ``pad`` (one TPU pad). Single parts are in the
    assembly's frame."""

    parameters = [Parameter("part", "drongo", choices=("drongo", "frame", "gear", "gripper", "jaw", "pad"),
                            description="what to build")] + _frame_params() + [
        Parameter("skid_height", 120.0, "mm", min=40, description="frame underside above the ground, landed"),
        Parameter("skid_y", 75.0, "mm", min=20, description="skids either side of the centre line"),
        Parameter("skid_length", 180.0, "mm", min=40, description="skid tube length (along x)"),
        Parameter("skid_diameter", 8.0, "mm", min=3, description="skid tube"),
        Parameter("strut_x", 50.0, "mm", min=5, description="struts ahead of / behind the centre"),
        Parameter("strut_y_top", 22.0, "mm", min=5, description="struts' upper ends either side of the centre line"),
        Parameter("strut_diameter", 6.0, "mm", min=2),
        Parameter("housing_x", 44.0, "mm", min=10, description="servo housing under the plate"),
        Parameter("housing_y", 34.0, "mm", min=10),
        Parameter("housing_z", 16.0, "mm", min=5),
        Parameter("rail_length", 190.0, "mm", min=40, description="jaw rail along x"),
        Parameter("rail_width", 12.0, "mm", min=4),
        Parameter("rail_height", 8.0, "mm", min=2),
        Parameter("carriage_x", 16.0, "mm", min=4, description="rack carriage riding under the rail"),
        Parameter("carriage_y", 20.0, "mm", min=4),
        Parameter("carriage_z", 8.0, "mm", min=2),
        Parameter("finger_thickness", 5.0, "mm", min=2, description="finger plate (x)"),
        Parameter("finger_width", 16.0, "mm", min=4, description="finger plate (y)"),
        Parameter("pad_thickness", 6.0, "mm", min=1, description="TPU pad (x)"),
        Parameter("pad_width", 40.0, "mm", min=5, description="TPU pad (y)"),
        Parameter("pad_height", 30.0, "mm", min=5, description="TPU pad (z)"),
        Parameter("grip_height", 36.5, "mm", min=5, description="pad centres above the ground, landed (the items' middle)"),
        Parameter("closed_gap", 30.0, "mm", min=0, description="pad-to-pad gap with the jaws closed"),
        Parameter("jaw_travel", 60.0, "mm", min=5, description="each jaw's travel from closed to fully open"),
        Parameter("jaw_opening", 40.0, "mm", min=0, description="each jaw's opening from closed (as drawn)"),
        Parameter("max_item_height", 75.0, "mm", min=10, description="the tallest item it lands over"),
        Parameter("show_bought", True, "", description="draw motors, propeller discs and battery in the assembly"),
    ]

    # ---- derived geometry shared by the notebook and the Chiron robot
    @staticmethod
    def pad_center_z(p):
        """Pad centre height in the frame [mm] (negative: below the frame's underside)."""
        return p["grip_height"] - p["skid_height"]

    @staticmethod
    def carriage_bottom_z(p):
        return -(p["housing_z"] + p["rail_height"] + p["carriage_z"])

    @staticmethod
    def gap(p, opening=None):
        """Pad-to-pad gap [mm] with each jaw opened by ``opening`` (default: as drawn)."""
        return p["closed_gap"] + 2 * (p["jaw_opening"] if opening is None else opening)

    # ---- parts
    def _gear(self, p):
        sh, sy, L = p["skid_height"], p["skid_y"], p["skid_length"]
        r, rs = p["skid_diameter"] / 2, p["strut_diameter"] / 2
        gear = None
        for s in (1, -1):
            skid = cq.Workplane("YZ").center(s * sy, -sh + r).circle(r).extrude(L / 2, both=True)
            gear = skid if gear is None else gear.union(skid)
            for sx in (p["strut_x"], -p["strut_x"]):
                a = cq.Vector(sx, s * p["strut_y_top"], 0.0)
                b = cq.Vector(sx, s * sy, -sh + r)
                strut = cq.Solid.makeCylinder(rs, (b - a).Length, a, b - a)
                gear = gear.union(cq.Workplane("XY").add(strut))
        return gear

    def _gripper(self, p):
        hx, hy, hz = p["housing_x"], p["housing_y"], p["housing_z"]
        housing = cq.Workplane("XY").box(hx, hy, hz).translate((0, 0, -hz / 2))
        rail = cq.Workplane("XY").box(p["rail_length"], p["rail_width"], p["rail_height"]) \
            .translate((0, 0, -hz - p["rail_height"] / 2))
        return housing.union(rail)

    def _jaw(self, p, side=1, opening=None):
        """One jaw (``side`` +1 front, −1 rear): carriage under the rail, finger down to the pad. The pad's inner face
        is at x = side × gap / 2."""
        x_face = side * self.gap(p, opening) / 2
        t_pad, t_f = p["pad_thickness"], p["finger_thickness"]
        x_f = x_face + side * (t_pad + t_f / 2)
        zc = self.carriage_bottom_z(p)
        z_pad = self.pad_center_z(p)
        carriage = cq.Workplane("XY").box(p["carriage_x"], p["carriage_y"], p["carriage_z"]) \
            .translate((x_face + side * (t_pad + p["carriage_x"] / 2), 0, zc + p["carriage_z"] / 2))
        z_top, z_bot = zc, z_pad - p["pad_height"] / 2
        finger = cq.Workplane("XY").box(t_f, p["finger_width"], z_top - z_bot).translate((x_f, 0, (z_top + z_bot) / 2))
        return carriage.union(finger)

    def _pad(self, p, side=1, opening=None):
        x_face = side * self.gap(p, opening) / 2
        return cq.Workplane("XY").box(p["pad_thickness"], p["pad_width"], p["pad_height"]) \
            .translate((x_face + side * p["pad_thickness"] / 2, 0, self.pad_center_z(p)))

    def _bought(self, p):
        """Motors, propeller discs (5 inch), battery and flight stack as blocks (not printed; for the picture)."""
        R, h = p["wheelbase"] / 2, p["arm_height"]
        out = cq.Workplane("XY").box(75, 35, 30).translate((0, 0, p["plate_thickness"] + 15))     # 4S 1500 mAh
        for k in range(4):
            a = math.radians(45 + 90 * k)
            cx, cy = R * math.cos(a), R * math.sin(a)
            out = out.union(cq.Workplane("XY").center(cx, cy).circle(14).extrude(16).translate((0, 0, h)))
            out = out.union(cq.Workplane("XY").center(cx, cy).circle(63.5).extrude(1.0).translate((0, 0, h + 20)))
        return out

    def build(self, p):
        part = p["part"]
        clearance = p["skid_height"] + self.carriage_bottom_z(p) - p["max_item_height"]
        if clearance < 5.0:
            raise ValueError(f"the rail carriages clear the tallest item by {clearance:.1f} mm (< 5 mm): raise skid_height")
        if self.pad_center_z(p) + p["pad_height"] / 2 > self.carriage_bottom_z(p):
            raise ValueError("the pads reach above the carriages: lower grip_height or raise skid_height")
        if p["jaw_opening"] > p["jaw_travel"]:
            raise ValueError("jaw_opening exceeds jaw_travel")
        if p["closed_gap"] / 2 + p["jaw_travel"] + p["pad_thickness"] + p["carriage_x"] / 2 > p["rail_length"] / 2:
            raise ValueError("fully open, the carriages run off the rail: lengthen rail_length or shorten jaw_travel")
        if p["skid_y"] - p["skid_diameter"] / 2 < p["pad_width"] / 2 + 5.0:
            raise ValueError("the skids are inside the pads' width")
        if part == "gear":
            return self._gear(p)
        if part == "gripper":
            return self._gripper(p)
        if part == "jaw":
            return self._jaw(p)
        if part == "pad":
            return self._pad(p)
        frame = super().build(p)
        if part == "frame":
            return frame
        body = frame.union(self._gear(p)).union(self._gripper(p))
        for side in (1, -1):
            body = body.union(self._jaw(p, side)).union(self._pad(p, side))
        if p["show_bought"]:
            body = body.union(self._bought(p))
        return body
