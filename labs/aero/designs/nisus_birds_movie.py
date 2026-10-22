"""The bird-photography mission as a video: what the camera and the mission software see, and where everyone is.

Layout (1280 x 720):
- left, large: the onboard camera, a SYNTHETIC render from the logged camera pose (sky, ground, the birds drawn at
  their true size and position; top left, the full-resolution crop around the target, the bird at the size a photo
  would show it), with the mission software's view on top: the latest simulated YOLOX boxes (green:
  high score, yellow: low score; the crop of the ROI pass dashed), the tracks (id; the target with the red reticle),
  a white flash when the shutter fires. Below it, the status lines.
- right, top: a chase view (Chiron's pyvista renderer) with the birds marked;
- right, middle: the map: home, the search star, the birds (orange when fleeing), the tracks with their 2-σ
  ellipses, the target line, the aircraft's track, the photos (stars);
- right, bottom: the photo log (the best photos so far, judged from the truth).
The speed changes with the action: slow during passes and photos, fast on the way out and back.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from vegeta.mission.sim import SPECIES

import nisus_robot as nr
import nisus_scenario as nsc

__all__ = ["render_movie", "frame_times"]

W, H = 1280, 720
ONB = (860, 484)
FONT = cv2.FONT_HERSHEY_SIMPLEX
SKY_TOP, SKY_HOR, GROUND_NEAR, GROUND_FAR = np.array([120, 160, 215]), np.array([200, 220, 240]), np.array([80, 120, 60]), np.array([150, 170, 140])


def _rec_index(rec_t, t):
    return int(np.clip(np.searchsorted(rec_t, t), 0, len(rec_t) - 1))


def frame_times(ep, fps=20, slow=2.5, mid=12.0, fast=30.0):
    """Frame times: ``slow`` x real time in passes, breakoffs and around photos, ``mid`` while hunting, ``fast``
    before and after (launch to the bird area, the return)."""
    rec = ep.log["birds"]
    rt = np.array([r["t"] for r in rec])
    photos = np.array([p["t"] for p in ep.log["photos"]]) if ep.log["photos"] else np.zeros(0)
    t_end = (ep.log.get("touchdown") or rt[-1]) + 5.0
    t, out = 0.0, []
    while t < min(t_end, rt[-1]):
        r = rec[_rec_index(rt, t)]
        action = r["mode"] in ("pass", "breakoff") or (len(photos) and np.min(np.abs(photos - t)) < 2.0)
        chasing = r["mode"] in ("intercept", "reposition")
        speed = slow if action else (0.5 * mid if chasing else (mid if r.get("active") else fast))
        out.append(t)
        t += speed / fps
    return out


def _sky_ground(R_wc, p_wc, cam, size):
    """The background: rays of a coarse grid coloured by elevation (sky) or by the distance to the ground hit."""
    w, h = size
    gw, gh = w // 2, h // 2
    u = np.linspace(0, cam.width, gw)
    v = np.linspace(0, cam.height, gh)
    U, V = np.meshgrid(u, v)
    d = np.stack([(U - cam.cx) / cam.fx, (V - cam.cy) / cam.fx, np.ones_like(U)], -1) @ R_wc.T
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    el = d[..., 2]
    img = np.zeros((gh, gw, 3))
    sky = el > 0
    a = np.clip(el / 0.6, 0, 1)[..., None]
    img = np.where(sky[..., None], (1 - a) * SKY_HOR + a * SKY_TOP, 0)
    dist = np.where(~sky, p_wc[2] / np.maximum(-el, 1e-3), 0)
    b = np.clip(dist / 1500.0, 0, 1)[..., None]
    gx = p_wc[0] + d[..., 0] * dist
    gy = p_wc[1] + d[..., 1] * dist
    check = ((np.floor(gx / 40) + np.floor(gy / 40)) % 2)[..., None] * 8 * (1 - b)
    img = np.where(~sky[..., None], (1 - b) * GROUND_NEAR + b * GROUND_FAR + check, img)
    return cv2.resize(img.astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR)


def _bird_glyph(img, c, span, colour, phase):
    """A flying bird: two wings (a flattened M) of ``span`` pixels."""
    x, y = c
    s = max(span, 3.0)
    flap = 0.25 * s * math.sin(phase)
    pts = np.array([[x - s / 2, y - flap], [x - s / 4, y - 0.12 * s], [x, y + 0.05 * s], [x + s / 4, y - 0.12 * s], [x + s / 2, y - flap]], np.int32)
    cv2.polylines(img, [pts], False, colour, max(1, int(s / 14)), cv2.LINE_AA)
    cv2.circle(img, (int(x), int(y + 0.04 * s)), max(1, int(s / 16)), colour, -1, cv2.LINE_AA)


def _dashed_rect(img, p0, p1, colour, dash=8):
    x0, y0 = p0; x1, y1 = p1
    for (a, b) in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        n = max(int(math.hypot(b[0] - a[0], b[1] - a[1]) / dash), 1)
        for k in range(0, n, 2):
            q0 = (int(a[0] + (b[0] - a[0]) * k / n), int(a[1] + (b[1] - a[1]) * k / n))
            q1 = (int(a[0] + (b[0] - a[0]) * min(k + 1, n) / n), int(a[1] + (b[1] - a[1]) * min(k + 1, n) / n))
            cv2.line(img, q0, q1, colour, 1, cv2.LINE_AA)


def _zoom(img, r, cam, species, sizes, t, bg, crop=360, size=230):
    """Top left: the full-resolution frame around the target (``crop`` px of the 3280 px frame), so the bird is seen
    at the size the photo would have it; the 150 px "good" width is marked."""
    if r["target"] is None:
        return
    k = next((k for k in r["tracks"] if k[0] == r["target"]), None)
    if k is None:
        return
    uv, z = cam.project(k[2][None], r["R_wc"], r["p_wc"])
    if z[0] < 1.0 or not cam.in_frame(uv, z)[0]:
        return
    cx, cy = uv[0]
    s = size / crop
    pane = np.empty((size, size, 3), np.uint8)
    sx = bg.shape[1] / cam.width
    pane[:] = bg[int(np.clip(cy * sx, 0, bg.shape[0] - 1)), int(np.clip(cx * sx, 0, bg.shape[1] - 1))]
    bu, bz = cam.project(r["birds"], r["R_wc"], r["p_wc"])
    for i in np.argsort(-bz):
        if bz[i] < 1.0:
            continue
        x, y = (bu[i, 0] - cx) * s + size / 2, (bu[i, 1] - cy) * s + size / 2
        span = cam.fx * sizes[i] / bz[i] * 0.8 * s
        if -span < x < size + span and -span < y < size + span:
            _bird_glyph(pane, (x, y), span, tuple(int(c) for c in SPECIES[species[i]].colour), 7.0 * t + i)
    g = 150 * s
    cv2.line(pane, (int(size / 2 - g / 2), size - 12), (int(size / 2 + g / 2), size - 12), (255, 255, 255), 2)
    cv2.putText(pane, "150 px", (int(size / 2 - 20), size - 16), FONT, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(pane, (0, 0), (size - 1, size - 1), (255, 70, 60), 2)
    cv2.putText(pane, f"target #{r['target']}: full-res crop", (5, 14), FONT, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    img[24:24 + size, 4:4 + size] = pane


def _onboard(r, cam, species, sizes, photo_flash, t):
    w, h = ONB
    sx = w / cam.width
    img = _sky_ground(r["R_wc"], r["p_wc"], cam, ONB)
    bg = img.copy()
    uv, z = cam.project(r["birds"], r["R_wc"], r["p_wc"])
    for i in np.argsort(-z):
        if z[i] < 1.0:
            continue
        span = cam.fx * sizes[i] / z[i] * 0.8 * sx
        x, y = uv[i] * sx
        if -50 < x < w + 50 and -50 < y < h + 50:
            col = SPECIES[species[i]].colour
            _bird_glyph(img, (x, y), span, tuple(int(c) for c in col[::-1][::-1]), 7.0 * t + i)
    if r["dets"] is not None:
        source, roi, boxes, t_cap = r["dets"]
        if roi is not None:
            _dashed_rect(img, (roi[0] * sx, roi[1] * sx), (roi[2] * sx, roi[3] * sx), (90, 230, 255))
            cv2.putText(img, "ROI crop", (int(roi[0] * sx) + 3, int(roi[1] * sx) + 12), FONT, 0.38, (90, 230, 255), 1, cv2.LINE_AA)
        for box, sc in boxes:
            col = (80, 230, 90) if sc >= 0.5 else (240, 220, 60)
            cv2.rectangle(img, (int(box[0] * sx), int(box[1] * sx)), (int(box[2] * sx) + 1, int(box[3] * sx) + 1), col, 1)
            cv2.putText(img, f"{sc:.2f}", (int(box[0] * sx), int(box[1] * sx) - 2), FONT, 0.32, col, 1, cv2.LINE_AA)
    if r["tracks"]:
        P = np.array([k[2] for k in r["tracks"]])
        uv, z = cam.project(P, r["R_wc"], r["p_wc"])
        for (tid, status, pos, cov, box), (x, y), zz in zip(r["tracks"], uv * sx, z):
            if zz < 1.0 or not (0 <= x < w and 0 <= y < h):
                continue
            is_t = tid == r["target"]
            col = (255, 70, 60) if is_t else ((255, 255, 255) if status == "confirmed" else (170, 170, 170))
            if is_t:
                cv2.circle(img, (int(x), int(y)), 22, col, 2, cv2.LINE_AA)
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    cv2.line(img, (int(x + 16 * dx), int(y + 16 * dy)), (int(x + 30 * dx), int(y + 30 * dy)), col, 2, cv2.LINE_AA)
            cv2.putText(img, f"#{tid}" + ("" if status == "confirmed" else f" {status}"), (int(x) + 8, int(y) - 8), FONT, 0.42, col, 1, cv2.LINE_AA)
    _zoom(img, r, cam, species, sizes, t, bg)
    if photo_flash is not None:
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (255, 255, 255), 8)
        good = "GOOD" if photo_flash["good_true"] else "not good"
        cv2.putText(img, f"PHOTO  bird {photo_flash['bird']} {photo_flash.get('species', '')}: {photo_flash['px_true']:.0f} px, "
                         f"{photo_flash['range_true']:.0f} m, blur {photo_flash['blur_true']:.1f} px  ({good})",
                    (12, h - 14), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (w, 20), (25, 30, 40), -1)
    cv2.putText(img, f"onboard camera: SYNTHETIC render ({cam.name}, {cam.hfov_deg:.0f} x {cam.vfov_deg:.0f} deg)  |  boxes: simulated YOLOX (not run)",
                (6, 14), FONT, 0.4, (240, 240, 240), 1, cv2.LINE_AA)
    return img


def _map(ep, r, i_rec, rec, centre, size=(420, 280)):
    w, h = size
    img = np.full((h, w, 3), (58, 72, 52), np.uint8)
    pos = r["nav_pos"][:2]
    c = 0.5 * (pos + centre)
    span = max(420.0, 2.4 * float(np.linalg.norm(pos - centre)) + 120)

    def px(xy):
        return (int(w / 2 + (xy[0] - c[0]) / span * w), int(h / 2 - (xy[1] - c[1]) / span * w))

    cfg = ep.birds.cfg.guidance
    star = [np.asarray(cfg.search_centre) + cfg.search_radius * np.array([math.cos(math.radians(90 + 144 * k)), math.sin(math.radians(90 + 144 * k))]) for k in range(6)]
    for a, b in zip(star[:-1], star[1:]):
        cv2.line(img, px(a), px(b), (95, 110, 90), 1, cv2.LINE_AA)
    trail = np.array([q["nav_pos"][:2] for q in rec[max(0, i_rec - 600):i_rec + 1]])
    for a, b in zip(trail[:-1], trail[1:]):
        cv2.line(img, px(a), px(b), (255, 230, 120), 1, cv2.LINE_AA)
    for p in ep.log["photos"]:
        if p["t"] <= r["t"]:
            j = _rec_index(np.array([q["t"] for q in rec]), p["t"])
            q = px(rec[j]["nav_pos"][:2])
            cv2.drawMarker(img, q, (255, 255, 255) if p["good_true"] else (160, 160, 160), cv2.MARKER_STAR, 9, 1)
    for tid, status, tpos, cov, box in r["tracks"]:
        q = px(tpos[:2])
        if status != "confirmed":
            cv2.drawMarker(img, q, (130, 130, 130), cv2.MARKER_TILTED_CROSS, 6, 1)
            continue
        ev, evec = np.linalg.eigh(cov[:2, :2])
        ax = tuple(int(max(2 * math.sqrt(max(e, 0)) / span * w, 1)) for e in ev[::-1])
        ang = math.degrees(math.atan2(evec[1, 1], evec[0, 1]))
        col = (255, 70, 60) if tid == r["target"] else ((230, 230, 230) if status == "confirmed" else (150, 150, 150))
        cv2.ellipse(img, q, ax, -ang, 0, 360, col, 1, cv2.LINE_AA)
        cv2.putText(img, f"{tid}", (q[0] + 4, q[1] - 4), FONT, 0.33, col, 1, cv2.LINE_AA)
        if tid == r["target"]:
            cv2.line(img, px(pos), q, col, 1, cv2.LINE_AA)
    for b, af in zip(r["birds"], r["bird_afraid"]):
        cv2.circle(img, px(b[:2]), 3, (255, 160, 40) if af else (40, 40, 40), -1, cv2.LINE_AA)
    cv2.circle(img, px((0, 0)), 4, (240, 240, 240), -1)
    cv2.putText(img, "home", (px((0, 0))[0] + 6, px((0, 0))[1] + 4), FONT, 0.38, (240, 240, 240), 1, cv2.LINE_AA)
    v = r["nav_vel"][:2]
    q = px(pos)
    if np.linalg.norm(v) > 1:
        e = (int(q[0] + 12 * v[0] / np.linalg.norm(v)), int(q[1] - 12 * v[1] / np.linalg.norm(v)))
        cv2.arrowedLine(img, q, e, (80, 200, 255), 2, tipLength=0.4)
    cv2.circle(img, q, 4, (80, 200, 255), -1)
    cv2.rectangle(img, (0, 0), (w, 18), (25, 30, 40), -1)
    cv2.putText(img, f"map {span:.0f} m across: birds (orange: fleeing), tracks 2-sigma, photos *", (6, 13), FONT, 0.38, (240, 240, 240), 1, cv2.LINE_AA)
    return img


def _photo_log(ep, t, size=(420, 174)):
    w, h = size
    img = np.full((h, w, 3), (30, 34, 44), np.uint8)
    ph = [p for p in ep.log["photos"] if p["t"] <= t]
    good = [p for p in ph if p["good_true"]]
    birds = sorted({p["bird"] for p in good})
    cv2.putText(img, f"photos {len(ph)}   good {len(good)} (truth)   birds photographed well: {len(birds)}", (8, 18), FONT, 0.45, (255, 230, 140), 1, cv2.LINE_AA)
    best = sorted(ph, key=lambda p: -p["px_true"])[:5]
    y = 42
    for p in best:
        col = (180, 255, 180) if p["good_true"] else (200, 200, 200)
        cv2.putText(img, f"t {p['t']:5.1f} s  bird {p['bird']} {str(p.get('species', ''))[:17]:17s} {p['px_true']:4.0f} px {p['range_true']:4.0f} m blur {p['blur_true']:.1f}",
                    (8, y), FONT, 0.38, col, 1, cv2.LINE_AA)
        y += 24
    cv2.putText(img, "good: >= 150 px across, blur <= 2 px, near the centre", (8, h - 8), FONT, 0.36, (150, 150, 150), 1, cv2.LINE_AA)
    return img


def _status(ep, r, t, ts_row, phase, size=(860, 206)):
    w, h = size
    img = np.full((h, w, 3), (22, 26, 34), np.uint8)
    hl = r["health"]
    tgt = r["target"]
    est = ""
    if tgt is not None:
        k = next((k for k in r["tracks"] if k[0] == tgt), None)
        if k is not None:
            est = f"estimated range {np.linalg.norm(k[2] - r['nav_pos']):.0f} m"
    d_true = np.linalg.norm(r["birds"] - r["nav_pos"], axis=1).min() if len(r["birds"]) else float("nan")
    lines = [(f"t = {t:6.1f} s   autopilot phase: {phase}   mission: {r['mode'] or '-'}   {r['note']}", (255, 225, 120)),
             (f"target: {'#' + str(tgt) if tgt is not None else 'none'}   {est}   done: {list(r['done'])}   nearest bird (truth) {d_true:5.1f} m", (210, 230, 255)),
             (f"airspeed {ts_row['V']:5.1f} m/s   height {ts_row['h']:5.1f} m   bank {ts_row['roll_deg']:+5.1f} deg   throttle {100 * ts_row['throttle']:3.0f} %   "
              f"power {ts_row['P_el_W']:4.0f} W", (210, 230, 255)),
             ((f"{hl[2]}, {hl[3]}: GPU load {100 * hl[0]:.0f} % (ASSUMED budget), detections {hl[1]:.1f} Hz" if hl else "Jetson: mission software idle"), (190, 255, 190)),
             (f"energy used {ts_row['E_used_Wh']:5.2f} Wh of {ep.log['energy']['E_available_wh']:.1f}   return margin {ts_row.get('margin_wh', float('nan')):+5.1f} Wh", (190, 255, 190))]
    events = [e for e in ep.log["events"] if 0 <= t - e[0] < 5 and e[1] != "energy"]
    if events:
        lines.append((f"> {events[-1][1]}: {events[-1][2][:110]}", (255, 200, 90)))
    y = 26
    for text, col in lines:
        cv2.putText(img, text, (10, y), FONT, 0.47, col, 1, cv2.LINE_AA)
        y += 29
    return img


def _chase_cams(ep, times):
    log = ep.log
    t = np.asarray(log["t"])
    com = np.asarray(log["body_pos"])[:, 0]
    vel = np.asarray(log["body_linvel"])[:, 0]
    out, idx = [], []
    head = np.array([1.0, 0.0, 0.0])
    for tt in times:
        i = int(np.clip(np.searchsorted(t, tt), 0, len(t) - 1))
        c, v = com[i], vel[i]
        if np.linalg.norm(v[:2]) > 2.0:
            d = v / np.linalg.norm(v)
            head = head + 0.3 * (d - head); head /= np.linalg.norm(head)
        out.append((tuple(c - 9.0 * head + np.array([0, 0, 2.5])), tuple(c + 10.0 * head), (0, 0, 1)))
        idx.append(i)
    return out, idx


def _slice_log(log, idx):
    """The log at the frame samples only (Chiron's renderer draws every sample it is given)."""
    n = len(log["t"])

    def series(v):
        return not isinstance(v, (str, dict)) and np.ndim(v) >= 1 and len(v) == n

    out = {}
    for k, v in log.items():
        if k in ("birds", "photos", "near", "events", "mission"):
            out[k] = v
        elif isinstance(v, dict):
            out[k] = {kk: (np.asarray(vv)[idx] if series(vv) else vv) for kk, vv in v.items()}
        elif series(v):
            out[k] = np.asarray(v)[idx]
        else:
            out[k] = v
    return out


def _project_view(pts, cam_pose, size, fov_deg=30.0):
    pos, foc, up = (np.asarray(x, float) for x in cam_pose)
    f = foc - pos; f /= np.linalg.norm(f)
    rgt = np.cross(f, up); rgt /= np.linalg.norm(rgt)
    u = np.cross(rgt, f)
    d = np.atleast_2d(pts) - pos
    z = d @ f
    fpx = 0.5 * size[1] / math.tan(math.radians(fov_deg / 2))
    x = size[0] / 2 + fpx * (d @ rgt) / np.maximum(z, 1e-3)
    y = size[1] / 2 - fpx * (d @ u) / np.maximum(z, 1e-3)
    return np.c_[x, y], z


def render_movie(ep, path, *, fps=20, slow=2.5, mid=12.0, fast=30.0, chase=True, max_frames=None, progress=False):
    """The MP4 (see the module); returns the path."""
    from vegeta.aeromant._watermark import watermark
    from vegeta.chiron import viz

    rec = ep.log["birds"]
    rt = np.array([r["t"] for r in rec])
    times = frame_times(ep, fps, slow, mid, fast)
    if max_frames:
        times = times[:max_frames]
    cam = ep.birds.cfg.camera
    truth0 = ep.birds.loop.birds
    species, sizes = list(truth0.truth().species), truth0.size
    centre = np.asarray(ep.birds.cfg.guidance.search_centre, float)
    ts = nsc.timeseries(ep)
    ta = ts["t"].to_numpy()
    starts = [(tt, name) for tt, name, note in ep.log["mission"] if note == "start"]
    chase_imgs = None
    cams, idx = _chase_cams(ep, times)
    if chase:
        sub = _slice_log(ep.log, idx)
        it = iter(cams)
        extent = 1200.0
        chase_imgs = viz.frames(sub, camera=lambda com: next(it), size=(420, 236), every=1, show_time=False,
                                ground=(-extent, extent, -extent, extent), scenery_range=3000.0, ground_color="#6f9a52", background="#bcd7f0")
    photos = ep.log["photos"]
    ph_t = np.array([p["t"] for p in photos]) if photos else np.zeros(0)
    title = f"{ep.bird_scenario.label}   (MuJoCo/Chiron 6-DOF + vegeta.mission; birds and detections simulated)"
    frames = []
    for n, t in enumerate(times):
        r = rec[_rec_index(rt, t)]
        flash = None
        if len(ph_t):
            j = int(np.argmin(np.abs(ph_t - t)))
            if abs(ph_t[j] - t) < 0.12:
                flash = photos[j]
        canvas = np.zeros((H, W, 3), np.uint8)
        canvas[:30] = (25, 30, 40)
        cv2.putText(canvas, title, (10, 20), FONT, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
        canvas[30:30 + ONB[1], :ONB[0]] = _onboard(r, cam, species, sizes, flash, t)
        row = ts.iloc[min(int(np.searchsorted(ta, t)), len(ts) - 1)]
        phase = next((name for tt, name in reversed(starts) if tt <= t + 1e-9), "prelaunch")
        canvas[514:720, :860] = _status(ep, r, t, row, phase)
        if chase_imgs is not None:
            ci = np.ascontiguousarray(chase_imgs[n])
            uv, z = _project_view(r["birds"], cams[n], (420, 236))
            for (x, y), zz in zip(uv, z):
                if zz > 1 and 0 <= x < 420 and 0 <= y < 236:
                    cv2.circle(ci, (int(x), int(y)), 4, (255, 120, 40), 1, cv2.LINE_AA)
            cv2.rectangle(ci, (0, 0), (420, 18), (25, 30, 40), -1)
            cv2.putText(ci, "chase view (birds circled)", (6, 13), FONT, 0.38, (240, 240, 240), 1, cv2.LINE_AA)
            canvas[30:266, 860:1280] = ci
        canvas[266:546, 860:1280] = _map(ep, r, _rec_index(rt, t), rec, centre)
        canvas[546:720, 860:1280] = _photo_log(ep, t)
        frames.append(canvas)
        if progress and n % 200 == 0:
            print(f"frame {n}/{len(times)}", flush=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (W, H))
    try:
        for img in frames:
            vw.write(watermark(cv2.cvtColor(img, cv2.COLOR_RGB2BGR)))
    finally:
        vw.release()
    return path
