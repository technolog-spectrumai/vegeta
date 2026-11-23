"""The pinhole camera: intrinsics, the mount on the airframe, projection, back-projection, size and blur.

Default: the Sony IMX219 module (Raspberry Pi Camera v2) on Nisus-Zero, 3280 x 2464 pixels of 1.12 µm with the
stock 3.04 mm lens (62.2° x 48.8°), datasheet values. It sits in the nose looking ``tilt_deg`` below the body x axis.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["PinholeCamera", "IMX219", "IMX219_6MM", "mount_rotation", "camera_pose"]


@dataclass(frozen=True)
class PinholeCamera:
    width: int = 3280
    height: int = 2464
    pixel_um: float = 1.12
    focal_mm: float = 3.04
    exposure_s: float = 1.0 / 2000          # daylight, shutter priority (assumption)
    tilt_deg: float = 15.0                  # optical axis below the body x axis
    offset_b: tuple = (0.33, 0.0, -0.03)    # camera centre in the body frame [m]
    name: str = "IMX219 + 3.04 mm (stock lens)"

    @property
    def fx(self) -> float:
        return self.focal_mm * 1000.0 / self.pixel_um

    @property
    def cx(self) -> float:
        return 0.5 * self.width

    @property
    def cy(self) -> float:
        return 0.5 * self.height

    @property
    def hfov_deg(self) -> float:
        return math.degrees(2 * math.atan(0.5 * self.width / self.fx))

    @property
    def vfov_deg(self) -> float:
        return math.degrees(2 * math.atan(0.5 * self.height / self.fx))

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fx, self.cy], [0, 0, 1.0]])

    def pixels_across(self, size_m, range_m):
        """How many pixels an object of ``size_m`` spans at ``range_m`` (small-angle)."""
        return np.asarray(size_m) * self.fx / np.maximum(np.asarray(range_m, float), 1e-3)

    def range_for_pixels(self, size_m, pixels):
        return size_m * self.fx / pixels

    def project(self, pts_w, R_wc, p_wc):
        """World points (N, 3) → pixels (N, 2) and depth along the axis (N,) (negative: behind the camera)."""
        pc = (np.atleast_2d(pts_w) - p_wc) @ R_wc            # world → camera: R_wc^T (p - c)
        z = pc[:, 2]
        zs = np.where(np.abs(z) < 1e-6, 1e-6, z)
        uv = np.c_[self.fx * pc[:, 0] / zs + self.cx, self.fx * pc[:, 1] / zs + self.cy]
        return uv, z

    def ray(self, uv, R_wc):
        """Unit rays in the world frame through pixels (N, 2)."""
        uv = np.atleast_2d(uv)
        d = np.c_[(uv[:, 0] - self.cx) / self.fx, (uv[:, 1] - self.cy) / self.fx, np.ones(len(uv))]
        d = d @ R_wc.T
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def in_frame(self, uv, z, margin=0.0):
        m = margin * self.width
        return (z > 0.5) & (uv[:, 0] >= -m) & (uv[:, 0] < self.width + m) & (uv[:, 1] >= -m) & (uv[:, 1] < self.height + m)

    def blur_px(self, rel_vel_w, rel_pos_w, omega_b, R_wc):
        """Motion blur during the exposure [px]: the line of sight's rotation from the relative motion plus the
        aircraft's own rotation (the stronger of the two components, added)."""
        r = np.linalg.norm(rel_pos_w)
        u = rel_pos_w / max(r, 1e-6)
        v_perp = rel_vel_w - (rel_vel_w @ u) * u
        los_rate = np.linalg.norm(v_perp) / max(r, 1e-3)
        body_rate = float(np.linalg.norm(omega_b))
        return (los_rate + body_rate) * self.exposure_s * self.fx


IMX219 = PinholeCamera()
IMX219_6MM = PinholeCamera(focal_mm=6.0, exposure_s=1.0 / 4000, name="IMX219 + 6 mm M12 lens")      # ~34° x 26°: twice the reach of the stock lens


def mount_rotation(tilt_deg: float) -> np.ndarray:
    """Camera → body rotation for a camera looking ``tilt_deg`` below the body x axis (OpenCV axes)."""
    s, c = math.sin(math.radians(tilt_deg)), math.cos(math.radians(tilt_deg))
    x_c = np.array([0.0, -1.0, 0.0])                 # image right = body right (−y)
    z_c = np.array([c, 0.0, -s])                     # optical axis: forward, tilted down
    y_c = np.cross(z_c, x_c)                         # image down
    return np.c_[x_c, y_c, z_c]


def camera_pose(cam: PinholeCamera, R_wb: np.ndarray, p_wb: np.ndarray):
    """The camera's rotation (camera → world) and centre for the aircraft's attitude and position."""
    R_wc = R_wb @ mount_rotation(cam.tilt_deg)
    return R_wc, np.asarray(p_wb, float) + R_wb @ np.asarray(cam.offset_b, float)
