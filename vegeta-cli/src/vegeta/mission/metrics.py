"""Tracking scores against the truth (simulation or labelled flight video): CLEAR-MOT and the identity metrics.

At each evaluation time the ground truth is the set of birds the camera could see (in the frame, ``min_px`` or more
across); the hypotheses are the confirmed tracks whose estimate lies in the frame. They are matched by the angle
between the lines of sight (the range is too uncertain to match on), Hungarian, gated at ``gate_deg``.
MOTA = 1 − (misses + false tracks + identity switches) / truth instances (Bernardin & Stiefelhagen 2008);
IDF1 = 2 IDTP / (2 IDTP + IDFP + IDFN) with the one-to-one identity mapping that maximises the matches (Ristani et
al. 2016).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .camera import PinholeCamera

__all__ = ["Sample", "evaluate"]


@dataclass
class Sample:
    t: float
    truth_ids: np.ndarray
    truth_pos: np.ndarray        # (N, 3)
    truth_size: np.ndarray       # (N,) wingspan [m]
    R_wc: np.ndarray
    p_wc: np.ndarray
    track_ids: np.ndarray
    track_pos: np.ndarray        # (M, 3)


def _bearings(P, p):
    d = np.atleast_2d(P) - p
    return d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-6)


def evaluate(samples, cam: PinholeCamera, *, gate_deg: float = 5.0, min_px: float = 6.0) -> dict:
    gt_total = fn = fp = idsw = tp = 0
    last = {}                                    # truth id -> track id
    pairs = defaultdict(int)
    gt_count, hyp_count = defaultdict(int), defaultdict(int)
    for s in samples:
        if len(s.truth_ids):
            uv, z = cam.project(s.truth_pos, s.R_wc, s.p_wc)
            px = cam.pixels_across(s.truth_size, np.linalg.norm(s.truth_pos - s.p_wc, axis=1))
            vis = cam.in_frame(uv, z) & (px >= min_px)
        else:
            vis = np.zeros(0, bool)
        g_ids, g_pos = s.truth_ids[vis], s.truth_pos[vis]
        if len(s.track_ids):
            uv, z = cam.project(s.track_pos, s.R_wc, s.p_wc)
            hv = cam.in_frame(uv, z)
        else:
            hv = np.zeros(0, bool)
        h_ids, h_pos = s.track_ids[hv], s.track_pos[hv]
        gt_total += len(g_ids)
        for g in g_ids:
            gt_count[int(g)] += 1
        for h in h_ids:
            hyp_count[int(h)] += 1
        matched = []
        if len(g_ids) and len(h_ids):
            cosang = _bearings(g_pos, s.p_wc) @ _bearings(h_pos, s.p_wc).T
            ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
            C = np.where(ang <= gate_deg, ang, 1e6)
            r, c = linear_sum_assignment(C)
            matched = [(int(g_ids[i]), int(h_ids[j])) for i, j in zip(r, c) if C[i, j] < 1e5]
        tp += len(matched)
        fn += len(g_ids) - len(matched)
        fp += len(h_ids) - len(matched)
        for g, h in matched:
            if g in last and last[g] != h:
                idsw += 1
            last[g] = h
            pairs[(g, h)] += 1
    # identity metrics
    gs, hs = sorted(gt_count), sorted(hyp_count)
    idtp = 0
    if gs and hs:
        M = np.zeros((len(gs), len(hs)))
        for (g, h), n in pairs.items():
            M[gs.index(g), hs.index(h)] = n
        r, c = linear_sum_assignment(-M)
        idtp = int(M[r, c].sum())
    n_gt, n_hyp = sum(gt_count.values()), sum(hyp_count.values())
    idfn, idfp = n_gt - idtp, n_hyp - idtp
    return {"truth instances": gt_total, "matches": tp, "misses": fn, "false tracks": fp, "identity switches": idsw,
            "MOTA": 1 - (fn + fp + idsw) / gt_total if gt_total else float("nan"),
            "IDF1": 2 * idtp / (2 * idtp + idfp + idfn) if (idtp + idfp + idfn) else float("nan"),
            "recall": tp / gt_total if gt_total else float("nan"),
            "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
            "birds seen": len(gt_count), "track ids": len(hyp_count)}
