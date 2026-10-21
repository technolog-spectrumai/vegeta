"""Multi-bird tracking from a moving camera: a world-frame IMM filter per bird and ByteTrack-style association.

Measurement. A detection is a bearing (the ray through the box centre, precise) and a range from the apparent size
(``size_prior_m`` x fx / box width: uncertain, the species and the wing's attitude are unknown). The filter uses it in
the camera's coordinates at the exposure: the two normalised image coordinates and the logarithm of the range
(``range_sigma_frac`` is its 1-σ), an EKF measurement. (Fusing it as a 3-D point with a long covariance along the ray
is unstable from a moving camera: every ray passes through the camera, so a wrong association pulls the estimate
towards it, where every later ray gates. In log range, a point near the camera is many σ from a bird 200 m out.)
The size-based range errors are correlated in time (the species, the wing's attitude), so the filter would grow
overconfident in the range and then mispredict the parallax of the aircraft's own motion; a floor on the
uncertainty along the line of sight (``range_floor_frac``, ``los_vel_floor``) keeps it honest.
The camera's pose at the exposure comes with the detections, so the aircraft's own motion is removed exactly
(ego-motion compensation by navigation instead of image registration).

Filter. Each track runs an IMM (interacting multiple models) over a constant-velocity model (straight flight) and a
coordinated-turn model (circling, a thermal, a break) on the state [x, y, z, vx, vy, vz, ω] (ω: turn rate in the
horizontal plane; the CV model keeps it at zero). The CT model is an EKF.

Association (Zhang et al. 2022, ByteTrack). Detections split by score: the high ones are matched first to every
track (confirmed, tentative, lost) with the Hungarian algorithm on the Mahalanobis distance of the 3-D measurement,
gated by χ²(3); then the low-score ones (a small, blurred, far bird) to the confirmed tracks still unmatched, which
keeps those tracks alive without starting tracks from noise. Unmatched high detections start tentative tracks; a
track is confirmed after ``n_confirm`` hits; it turns 'lost' (coasting on its prediction, out of view behind the
fixed camera or missed) after ``lost_after_s`` without an update and is deleted after ``delete_after_s``.

Re-acquisition (after Cao et al. 2023, OC-SORT's observation-centric re-update): when a lost track is matched again,
its velocity is reset from the last observation to the new one (a virtual straight trajectory over the gap) and its
covariance re-inflated, instead of trusting a velocity that coasted through the gap.
Appearance re-identification (DeepSORT, BoT-SORT) is left out: birds of a species look alike and it costs GPU time.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .bus import Node
from .camera import PinholeCamera
from .messages import Detections, TrackEstimate, Tracks

__all__ = ["TrackerConfig", "IMM", "BirdTracker", "measurement_from_box"]

CHI2_3_999 = 16.27


@dataclass(frozen=True)
class TrackerConfig:
    size_prior_m: float = 0.9           # wingspan assumed for the range (a mid-size bird: crow, gull)
    range_sigma_frac: float = 0.40      # 1-σ of ln(range) from the size (wing attitude, species)
    pixel_sigma: float = 2.0            # px, plus 5 % of the box width
    high_score: float = 0.5
    low_score: float = 0.1
    init_score: float = 0.6
    n_confirm: int = 3
    lost_after_s: float = 1.5
    delete_after_s: float = 25.0
    delete_tentative_s: float = 2.0
    q_cv: float = 4.0                   # m/s² white acceleration, straight flight (birds flap and weave)
    q_ct: float = 6.0                   # m/s², turning
    q_omega: float = 0.3                # rad/s² turn-rate random walk
    p_switch: float = 0.03              # IMM model switch probability per second
    gate: float = CHI2_3_999
    reacquire_gap_s: float = 0.6
    range_floor_frac: float = 0.25      # the range error from size is correlated (species, wing attitude): never trust it more
    los_vel_floor: float = 3.0          # m/s, along the line of sight (unobservable from bearings)
    v_max: float = 28.0                 # no bird of interest flies faster (a stooping falcon aside)
    v_reset_sigma_max: float = 6.0      # reset the velocity from two observations only when that is this precise


def measurement_from_box(cam: PinholeCamera, box, R_wc, p_wc, cfg: TrackerConfig):
    """3-D point and covariance from a detection box."""
    cx, cy = 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])
    w = max(box[2] - box[0], 1.0)
    u = cam.ray(np.array([[cx, cy]]), R_wc)[0]
    rng = cfg.size_prior_m * cam.fx / w
    z = p_wc + rng * u
    s_ang = (cfg.pixel_sigma + 0.05 * w) / cam.fx
    s_cross, s_along = rng * s_ang, cfg.range_sigma_frac * rng
    # an orthonormal frame around the ray
    a = np.array([0.0, 0.0, 1.0]) if abs(u[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(u, a); e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    E = np.c_[e1, e2, u]
    R = E @ np.diag([s_cross ** 2, s_cross ** 2, s_along ** 2]) @ E.T
    return z, R, rng


H = np.c_[np.eye(3), np.zeros((3, 4))]


@dataclass(frozen=True)
class CamMeasurement:
    """A detection in the camera's coordinates: y = (x/z, y/z, ln r) with its noise, the pose, and the 3-D point."""
    y: np.ndarray
    Rm: np.ndarray
    R_wc: np.ndarray
    p_wc: np.ndarray
    point: np.ndarray
    point_cov: np.ndarray
    rng: float


def cam_measurement(cam: PinholeCamera, box, R_wc, p_wc, cfg) -> CamMeasurement:
    cx, cy = 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])
    w = max(box[2] - box[0], 1.0)
    rng = cfg.size_prior_m * cam.fx / w
    s_img = (cfg.pixel_sigma + 0.05 * w) / cam.fx
    y = np.array([(cx - cam.cx) / cam.fx, (cy - cam.cy) / cam.fx, math.log(rng)])
    Rm = np.diag([s_img ** 2, s_img ** 2, cfg.range_sigma_frac ** 2])
    z, R3, _ = measurement_from_box(cam, box, R_wc, p_wc, cfg)
    return CamMeasurement(y, Rm, R_wc, p_wc, z, R3, rng)


def _h(x, m: CamMeasurement):
    """Predicted measurement and its Jacobian (3 x 7) for state ``x``; None when the point is behind the camera."""
    pc = m.R_wc.T @ (x[:3] - m.p_wc)
    a, b, c = pc
    if c < 0.5:
        return None, None
    r2 = float(pc @ pc)
    y = np.array([a / c, b / c, 0.5 * math.log(r2)])
    J = np.array([[1 / c, 0, -a / c ** 2], [0, 1 / c, -b / c ** 2], pc / r2])
    Hx = np.zeros((3, 7)); Hx[:, :3] = J @ m.R_wc.T
    return y, Hx


def _f_cv(x, dt):
    y = x.copy()
    y[:3] += x[3:6] * dt
    y[6] = 0.0
    F = np.eye(7); F[:3, 3:6] = np.eye(3) * dt; F[6, 6] = 0.0
    return y, F


def _f_ct(x, dt):
    px, py, pz, vx, vy, vz, w = x
    y = x.copy()
    F = np.eye(7)
    if abs(w) < 1e-4:
        y[0], y[1] = px + vx * dt, py + vy * dt
        F[0, 3] = F[1, 4] = dt
        F[0, 6] = -0.5 * vy * dt ** 2
        F[1, 6] = 0.5 * vx * dt ** 2
        F[3, 6] = -vy * dt
        F[4, 6] = vx * dt
    else:
        s, c = math.sin(w * dt), math.cos(w * dt)
        y[0] = px + (vx * s - vy * (1 - c)) / w
        y[1] = py + (vx * (1 - c) + vy * s) / w
        y[3] = vx * c - vy * s
        y[4] = vx * s + vy * c
        F[0, 3], F[0, 4] = s / w, -(1 - c) / w
        F[1, 3], F[1, 4] = (1 - c) / w, s / w
        F[3, 3], F[3, 4] = c, -s
        F[4, 3], F[4, 4] = s, c
        F[0, 6] = (vx * (dt * c * w - s) - vy * (dt * s * w - (1 - c))) / w ** 2
        F[1, 6] = (vx * (dt * s * w - (1 - c)) + vy * (dt * c * w - s)) / w ** 2
        F[3, 6] = -dt * (vx * s + vy * c)
        F[4, 6] = dt * (vx * c - vy * s)
    y[2] = pz + vz * dt
    F[2, 5] = dt
    return y, F


def _Q(q, dt, q_w=0.0):
    G = np.zeros((7, 3)); G[:3] = np.eye(3) * 0.5 * dt ** 2; G[3:6] = np.eye(3) * dt
    Q = G @ G.T * q ** 2
    Q[6, 6] = (q_w * dt) ** 2 + 1e-9
    return Q


class IMM:
    """Two-model IMM (CV, CT) on [x, y, z, vx, vy, vz, ω]."""

    def __init__(self, z, R, cfg: TrackerConfig, v0=None):
        x = np.zeros(7); x[:3] = z
        if v0 is not None:
            x[3:6] = v0
        P = np.zeros((7, 7)); P[:3, :3] = R
        P[3:6, 3:6] = np.eye(3) * (12.0 ** 2 if v0 is None else 4.0 ** 2)
        P[5, 5] = 3.0 ** 2
        P[6, 6] = 0.3 ** 2
        Pcv = P.copy(); Pcv[6, 6] = 1e-6
        self.x = [x.copy(), x.copy()]
        self.P = [Pcv, P.copy()]
        self.mu = np.array([0.7, 0.3])
        self.cfg = cfg

    def predict(self, dt):
        if dt <= 0:
            return
        cfg = self.cfg
        p = min(cfg.p_switch * dt, 0.5)
        T = np.array([[1 - p, p], [p, 1 - p]])
        cbar = T.T @ self.mu
        mix = (T * self.mu[:, None]) / np.maximum(cbar[None, :], 1e-12)      # mix[i, j] = P(i | j)
        xs, Ps = [], []
        for j in range(2):
            x0 = sum(mix[i, j] * self.x[i] for i in range(2))
            P0 = sum(mix[i, j] * (self.P[i] + np.outer(self.x[i] - x0, self.x[i] - x0)) for i in range(2))
            xs.append(x0); Ps.append(P0)
        x, F = _f_cv(xs[0], dt)
        P = F @ Ps[0] @ F.T + _Q(cfg.q_cv, dt); P[6, 6] = 1e-6
        x1, F1 = _f_ct(xs[1], dt)
        P1 = F1 @ Ps[1] @ F1.T + _Q(cfg.q_ct, dt, cfg.q_omega)
        self.x, self.P, self.mu = [x, x1], [P, P1], cbar

    def combined(self):
        x = self.mu[0] * self.x[0] + self.mu[1] * self.x[1]
        P = sum(self.mu[i] * (self.P[i] + np.outer(self.x[i] - x, self.x[i] - x)) for i in range(2))
        return x, P

    def innovation(self, m: CamMeasurement):
        x, P = self.combined()
        yp, Hx = _h(x, m)
        if yp is None:
            return None, None, math.inf
        S = Hx @ P @ Hx.T + m.Rm
        nu = m.y - yp
        return nu, S, float(nu @ np.linalg.solve(S, nu))

    def update(self, m: CamMeasurement):
        lik = np.zeros(2)
        for i in range(2):
            yp, Hx = _h(self.x[i], m)
            if yp is None:
                lik[i] = 1e-300
                continue
            S = Hx @ self.P[i] @ Hx.T + m.Rm
            nu = m.y - yp
            K = self.P[i] @ Hx.T @ np.linalg.inv(S)
            self.x[i] = self.x[i] + K @ nu
            self.P[i] = (np.eye(7) - K @ Hx) @ self.P[i]
            self.P[i] = 0.5 * (self.P[i] + self.P[i].T)
            d2 = float(nu @ np.linalg.solve(S, nu))
            lik[i] = math.exp(-0.5 * min(d2, 50.0)) / math.sqrt(max(np.linalg.det(2 * math.pi * S), 1e-300))
        mu = self.mu * lik
        self.mu = mu / mu.sum() if mu.sum() > 0 else np.array([0.5, 0.5])
        self.mu = np.clip(self.mu, 1e-3, 1 - 1e-3); self.mu /= self.mu.sum()

    def reset_velocity(self, v, sigma=4.0):
        for i in range(2):
            self.x[i][3:6] = v
            self.P[i][3:6, :] = 0.0; self.P[i][:, 3:6] = 0.0
            self.P[i][3:6, 3:6] = np.eye(3) * sigma ** 2


class _Track:
    def __init__(self, tid, t, z, R, box, score, cfg):
        self.id, self.cfg = tid, cfg
        self.f = IMM(z, R, cfg)
        self.t = t                      # filter time
        self.t_birth = t
        self.t_update = t
        self.hits = 1
        self.status = "tentative"
        self.box, self.score = box, score
        self.last_z = z

    def predict_to(self, t):
        self.f.predict(t - self.t)
        self.t = max(self.t, t)

    def estimate(self, t_now) -> TrackEstimate:
        x, P = self.f.combined()
        dt = max(t_now - self.t, 0.0)
        pos = x[:3] + x[3:6] * dt
        return TrackEstimate(self.id, self.status, pos, x[3:6].copy(), P[:6, :6].copy(), float(x[6]), float(self.f.mu[1]),
                             self.hits, t_now - self.t_birth, t_now - self.t_update, self.box, self.score)


class BirdTracker(Node):
    """The tracking node: ``detections`` in, ``tracks`` out (every detection message, and at its own rate)."""
    name = "tracker"
    subscriptions = ("detections",)

    def __init__(self, cam: PinholeCamera, cfg: TrackerConfig = TrackerConfig()):
        self.cam, self.cfg = cam, cfg
        self.tracks: list = []
        self.next_id = 1
        self.stats = {"updates": 0, "births": 0, "deaths": 0, "reacquired": 0}

    # -- one detection message
    def process(self, det: Detections):
        cfg, cam = self.cfg, self.cam
        t = det.t_capture
        for k in self.tracks:
            k.predict_to(t)
        meas = [(d, cam_measurement(cam, d.box, det.R_wc, det.p_wc, cfg)) for d in det.items]
        high = [m for m in meas if m[0].score >= cfg.high_score]
        low = [m for m in meas if cfg.low_score <= m[0].score < cfg.high_score]
        pool = list(self.tracks)
        m1, u_tracks, u_high = self._match(pool, high)
        for k, m in m1:
            self._update(k, m, t)
        conf = [k for k in u_tracks if k.status == "confirmed"]
        m2, _, _ = self._match(conf, low)
        for k, m in m2:
            self._update(k, m, t)
        for m in u_high:
            if m[0].score >= cfg.init_score:
                self.tracks.append(_Track(self.next_id, t, m[1].point, m[1].point_cov, m[0].box, m[0].score, cfg))
                self.next_id += 1
                self.stats["births"] += 1

    def _visible(self, k, det):
        x, _ = k.f.combined()
        uv, z = self.cam.project(x[:3][None], det.R_wc, det.p_wc)
        if not self.cam.in_frame(uv, z)[0]:
            return False
        if det.roi is not None:
            x0, y0, x1, y1 = det.roi
            return x0 <= uv[0, 0] <= x1 and y0 <= uv[0, 1] <= y1
        return True

    def _match(self, tracks, meas):
        if not tracks or not meas:
            return [], list(tracks), list(meas)
        C = np.full((len(tracks), len(meas)), 1e6)
        for i, k in enumerate(tracks):
            for j, (d, cm) in enumerate(meas):
                _, _, d2 = k.f.innovation(cm)
                if d2 <= self.cfg.gate:
                    C[i, j] = d2 + (0.0 if k.status == "confirmed" else 1.0)
        r, c = linear_sum_assignment(C)
        pairs = [(tracks[i], meas[j]) for i, j in zip(r, c) if C[i, j] < 1e5]
        mt = {id(p[0]) for p in pairs}
        mm = {id(p[1]) for p in pairs}
        return pairs, [k for k in tracks if id(k) not in mt], [m for m in meas if id(m) not in mm]

    def _update(self, k: _Track, m, t):
        d, cm = m
        z, rng = cm.point, cm.rng
        gap = t - k.t_update
        cfg = self.cfg
        if k.status == "lost" or gap > cfg.reacquire_gap_s:               # observation-centric re-update
            s_v = cfg.range_sigma_frac * rng / max(gap, 1e-3)
            if s_v < cfg.v_reset_sigma_max:
                v = (z - k.last_z) / gap
                n = np.linalg.norm(v)
                k.f.reset_velocity(v if n <= cfg.v_max else v * cfg.v_max / n, sigma=max(3.0, s_v))
            else:
                for i in range(2):                                         # too imprecise: only re-open the velocity
                    k.f.P[i][3:6, 3:6] += np.eye(3) * 4.0 ** 2
            self.stats["reacquired"] += k.status == "lost"
            if k.status == "lost":
                k.status = "confirmed"
        k.f.update(cm)
        x, _ = k.f.combined()
        los = x[:3] - cm.p_wc
        r = float(np.linalg.norm(los))
        u = los / max(r, 1e-6)
        uu = np.outer(u, u)
        for i in range(2):                                                 # floors along the line of sight
            P = k.f.P[i]
            va = float(u @ P[:3, :3] @ u)
            fl = (cfg.range_floor_frac * r) ** 2
            if va < fl:
                P[:3, :3] += (fl - va) * uu
            vv = float(u @ P[3:6, 3:6] @ u)
            if vv < cfg.los_vel_floor ** 2:
                P[3:6, 3:6] += (cfg.los_vel_floor ** 2 - vv) * uu
        for i in range(2):                                                 # physical limit on the speed
            v = k.f.x[i][3:6]
            n = np.linalg.norm(v)
            if n > cfg.v_max:
                k.f.x[i][3:6] = v * cfg.v_max / n
        k.hits += 1
        k.t_update = t
        k.box, k.score, k.last_z = d.box, d.score, z
        if k.status == "tentative" and k.hits >= self.cfg.n_confirm:
            k.status = "confirmed"
        self.stats["updates"] += 1

    def prune(self, t):
        keep = []
        for k in self.tracks:
            age = t - k.t_update
            if (k.status == "tentative" and age > self.cfg.delete_tentative_s) or age > self.cfg.delete_after_s:
                self.stats["deaths"] += 1
                continue
            if k.status == "confirmed" and age > self.cfg.lost_after_s:       # coasting (out of view, or missed): lost
                k.status = "lost"
            keep.append(k)
        self.tracks = keep

    def snapshot(self, t) -> Tracks:
        return Tracks(t, tuple(k.estimate(t) for k in self.tracks if k.status != "tentative" or k.hits >= 2))

    def step(self, t, bus):
        for det in bus.take("detections", self.name):
            self.process(det)
        self.prune(t)
        bus.publish("tracks", self.snapshot(t))
