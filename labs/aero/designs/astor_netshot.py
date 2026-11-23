"""ASTOR-TWIN fires GAVILAN's net launcher at a quadcopter: the sortie with one launcher and three nets (2026-11-22).

**The pass.** ASTOR comes up behind the quadcopter, along its track and a little above it, at 1.3 x its stall speed,
wings spread. The launch tube lies under the forward belly, 15 deg below the body axis, on a mount that pans +-35 deg and tilts the tube
between level and 15 deg down (150 deg/s): the mount, not the airframe, does the last pointing. (Pointing the nose with
bank was tried: this heavy, lightly yaw-damped airframe swung +-20 deg and never came on.) When the tracker's prediction puts the net on
the quadcopter, between 4.5 and 9.5 m ahead, it fires; ASTOR flies on over the net. A miss is followed by a full
turn back behind the quadcopter and the next net (three in all; the feed and the gas refill take seconds, the turn
~40 s).

**The net** is not simulated cord by cord here. It is the launcher's own test, reduced to three numbers
(``gavilan_flight.NET``: a 150 g cartridge at 40 m/s, 10 m in 0.51 s, 2.95 m open; from a fixed launcher at a
hovering 1.2 kg quadcopter it took 22 of 24 between 4 and 10 m and 1 of 12 beyond):

- its centre flies as a point with a drag that reproduces "10 m in 0.51 s" (in still air; here the drag acts on its
  speed through the air, which is ASTOR's speed more: it goes less far — the main untested step);
- it opens in proportion to the distance flown, fully at 6 m;
- it takes the quadcopter if the quadcopter is inside 80 % of its half span when it passes, having flown 4-10 m, with
  the test's 22 in 24.

A quadcopter taken falls with the net (a simple fall with drag). ASTOR is flown by ``astor_v_flight`` ("twin").
"""
from __future__ import annotations

import math

import numpy as np

import arigo1_model as am
import arigo1_mujoco as amj
import astor_catch as AC
import astor_hunt as H
import astor_netdrop as ND
import astor_talon as TA
import astor_v_flight as VF

NETS = 3
NET_KG, MUZZLE_M_S, OPEN_SPAN_M, FULL_OPEN_M = 0.150, 40.0, 2.95, 6.0
NET_K = 0.127                      # 1/m: dv/dt = -k v|v| brings 40 m/s to 10 m in 0.51 s (the launcher's test)
RANGE_M, P_TAKE, INSIDE = (4.0, 10.0), 22.0 / 24.0, 0.8
FIRE_M = (4.5, 9.5)                # fire only when the meeting is predicted inside this
DEPRESSION_DEG = 15.0              # the tube's rest tilt below the body axis; the mount moves it between this and 0
BREECH_BODY, TUBE_M = np.array([-0.082, 0.0, -0.068]), 0.390             # the trunnion from the CG (x forward, z up) [m]; breech to muzzle
MUZZLE_BODY = BREECH_BODY + TUBE_M * np.array([math.cos(math.radians(15.0)), 0.0, -math.sin(math.radians(15.0))])     # at rest
MEAN_TRACK_S, CLOSE_BANK_DEG = 8.0, 20.0   # the mean track: a fit over one weave period (4 s still swung with it); the bank allowed inside 70 m
AIM_TILT_DEG = 11.0                 # the approach is flown for a shot with the tube here: travel left both ways (0-15 deg)
TRAIL_M = 40.0                     # the approach aims this far behind the quadcopter on its track, drawn in as the range closes
PAN_DEG, TILT_DEG, SLEW_DEG_S = 35.0, (0.0, 15.0), 150.0   # pan each way; the tube's tilt below the body axis, from level to its rest     # the launcher's mount: travel each way and slew rate (a servo pan-tilt, HAND)
AIM_M, AIM_K = 90.0, 1.5           # within this range the nose is turned onto the meeting point (35 m left 2 s: too late), at 1.5 /s of the bearing error
FIRE_INSIDE = 0.6                  # fire when the predicted miss is within this share of the net's half span
RELOAD_S, HUNT_S = 5.0, 170.0
NETTED_CDA = 0.14                  # m2: the quadcopter tumbling in the net (gavilan_net_fall's bandit 0.04 + the net bunched round it, HAND)


WEAVE_WINDOW_S, WEAVE_PERIODS_S = 12.0, np.arange(4.0, 16.5, 0.5)       # the tracker's memory for the weave fit; the periods it tries
# A JINKING quadcopter (a new heading every 2 s once ASTOR is near) has no mean track to fly along: when the quick
# tracker's heading leaves the 8 s mean by more than JINK_DEG inside JINK_RANGE_M, ASTOR stops flying the mean line
# wings level and chases the quadcopter itself: proportional navigation on its present track, the flaps down, the
# pass slower (1.3 x the flaps-down stall), the bank free up to JINK_BANK_DEG. The mount still does the last pointing.
JINK_DEG, JINK_RANGE_M, JINK_BANK_DEG, JINK_HOLD_S = 25.0, 140.0, 62.0, 4.0
# the chase is flown faster than a plain pass: the turn rate is lift-limited there and grows with speed (0.45 rad/s at
# 25 m/s, 0.67 at 30); and inside JINK_LEVEL_M ASTOR rolls level: banked, the quadcopter sits on the inside of the
# turn, ABOVE the wings' plane, where the tube under the belly (0-15 deg down) cannot point (first tries: 46-51 deg of bank at the pass, no shot)
JINK_SPEED, JINK_LEVEL_M, JINK_LEVEL_BANK_DEG = 1.5, 28.0, 12.0
JINK_KZ, JINK_VZ = 1.6, 8.0          # the height loop in the chase: stiffer (banked turns left ASTOR 3-6 m off the quadcopter's height)
AWAY_BANK_DEG, AWAY_SPEED = 50.0, 1.55                                  # the turn back after a pass: bank, and speed as a multiple of the stall


def weave_fit(buf, t_now, latency):
    """The quadcopter's track as a straight mean line plus one sinusoid across it (a weave), fitted to the tracker's
    samples of the last WEAVE_WINDOW_S: returns predict(t) -> (position, velocity) in the horizontal plane, the mean
    velocity, and whether a weave was found (else the straight line alone). A straight line fitted over one weave
    period is biased by the weave (up to ~5 m here): that put ASTOR 10 m to the side of weaving quadcopters."""
    old = [x for x in buf if x[0] <= t_now - latency]
    if len(old) < 40:
        return None
    ts = np.array([x[0] for x in old])
    P = np.array([x[1][:2] for x in old])
    t0 = ts[-1]
    tt = ts - t0
    A1 = np.stack([np.ones_like(tt), tt], axis=1)
    c1, r1, *_ = np.linalg.lstsq(A1, P, rcond=None)
    rss1 = float(((A1 @ c1 - P) ** 2).sum())
    best = None
    if ts[-1] - ts[0] >= 5.0:
        for T in WEAVE_PERIODS_S:
            if T > 1.6 * (ts[-1] - ts[0]):
                break
            w = 2 * math.pi / T
            A = np.stack([np.ones_like(tt), tt, np.sin(w * tt), np.cos(w * tt)], axis=1)
            c, *_ = np.linalg.lstsq(A, P, rcond=None)
            rss = float(((A @ c - P) ** 2).sum())
            if best is None or rss < best[0]:
                best = (rss, w, c)
    if best is not None and best[0] < 0.4 * rss1 and float(np.hypot(*np.linalg.norm(best[2][2:], axis=1))) > 1.0:
        _, w, c = best

        def predict(t):
            x = t - t0
            return (c[0] + c[1] * x + c[2] * math.sin(w * x) + c[3] * math.cos(w * x), c[1] + w * (c[2] * math.cos(w * x) - c[3] * math.sin(w * x)))
        return predict, c[1], True
    return (lambda t: (c1[0] + c1[1] * (t - t0), c1[1])), c1[1], False


def half_span(flown) -> float:
    return 0.5 * OPEN_SPAN_M * min(1.0, max(flown, 0.0) / FULL_OPEN_M)


class FlyingNet:
    def __init__(self, p, v, wind):
        self.p, self.v, self.wind, self.flown, self.t = np.array(p, float), np.array(v, float), np.asarray(wind, float), 0.0, 0.0
        self.u0 = self.v / np.linalg.norm(self.v)
        self.result = None

    def step(self, dt):
        va = self.v - self.wind
        self.v = self.v + (-NET_K * float(np.linalg.norm(va)) * va + np.array([0.0, 0.0, -am.G])) * dt
        self.p = self.p + self.v * dt
        self.flown += float(np.linalg.norm(self.v)) * dt * 0.0
        self.t += dt


def predict(muzzle, v0, wind, b, vb, horizon=0.5, dt=0.004):
    """Where a net fired now would pass the quadcopter (flown straight on): the miss across, the distance flown."""
    n = FlyingNet(muzzle, v0, wind)
    p0 = n.p.copy()
    prev = None
    for k in range(int(horizon / dt)):
        n.step(dt)
        q = np.asarray(b) + np.asarray(vb) * n.t
        rel = q - n.p
        vr = n.v - np.asarray(vb)
        ahead = float(rel @ vr)
        if prev is not None and prev > 0.0 >= ahead:
            u = vr / max(float(np.linalg.norm(vr)), 1e-6)
            perp = rel - (rel @ u) * u
            return float(np.linalg.norm(perp)), float(np.linalg.norm(n.p - p0)), n.t
        prev = ahead
    return None


def netshot(seed=0, kind=None, latency_s=0.10, noise_m=0.3, record=None, sim=None, nets=NETS, height=120.0) -> dict:
    """One sortie of ASTOR-TWIN with the launcher against one quadcopter (``kind`` None: drawn from astor_hunt's
    population)."""
    rng = np.random.default_rng(seed)
    pick = lambda pairs: pairs[int(np.searchsorted(np.cumsum([w for _, w in pairs]), rng.random()))][0]
    kind = kind or pick(H.POPULATION["behaviour"])
    q_kg = float(rng.uniform(*H.POPULATION["mass_kg"]))
    sim = sim or VF.SimV("twin", bandit=False)
    m, d = sim.m, sim.d
    Vs = VF.stall(VF.get("twin", VF.SWEEPS[0]))
    bandit = H.Quad(kind, rng, height=height)
    sim.start(1.4 * Vs, bandit.p[2] + 6.0)
    sim.sweep_cmd = VF.SWEEPS[0]                                         # the pass is flown spread
    ap = amj.Autopilot(sim)
    cam = ND.Camera(latency_s, noise_m, seed, window_s=0.5)
    dep = math.radians(DEPRESSION_DEG)
    left, shots, net, phase, t_phase, t_shot = nets, [], None, "attack", 0.0, -1e9
    alpha_f, was_ahead, outcome, netted, vq = 0.0, None, None, None, None
    mount = [0.0, 0.0]                                                   # the launcher's pan and tilt from its rest [deg]
    cam.v_last = np.zeros(3)
    slow = ND.Camera(latency_s, noise_m, seed + 1000, window_s=MEAN_TRACK_S)
    wide = ND.Camera(latency_s, noise_m, seed + 2000, window_s=WEAVE_WINDOW_S)      # the same sightings kept longer, for the weave fit
    weave_used, chase_until, chase_used, jinker = 0, -1.0, 0, False
    Vs_flap = VF.stall(VF.with_flap(VF.get("twin", VF.SWEEPS[0]), 1.0))
    z_min = 1e9
    t0 = d.time
    while outcome is None:
        t = d.time - t0
        s = sim.state()
        R = d.xmat[sim.bid].reshape(3, 3)
        el0, az0 = dep + math.radians(mount[1]), math.radians(mount[0])
        muzzle = np.asarray(s["pos"]) + R @ (BREECH_BODY + TUBE_M * np.array([math.cos(el0) * math.cos(az0), math.cos(el0) * math.sin(az0), -math.sin(el0)]))
        # the launcher on its mount: pan and tilt follow the direction that puts the net on the quadcopter (the net
        # leaves with ASTOR's velocity plus 40 m/s along the tube), within the mount's travel and slew rate
        los = np.asarray(bandit.p if False else (cam.buf[-1][1] if cam.buf else bandit.p)) - muzzle
        l_hat = los / max(float(np.linalg.norm(los)), 1e-6)
        w = np.asarray(s["v_world"]) - np.asarray(getattr(cam, "v_last", np.zeros(3)))
        lw = float(l_hat @ w)
        sp = lw + math.sqrt(max(lw * lw - float(w @ w) + MUZZLE_M_S ** 2, 0.0))
        want = R.T @ ((sp * l_hat - w) / MUZZLE_M_S + np.array([0.0, 0.0, 0.5 * am.G * 0.2 ** 2 / max(float(np.linalg.norm(los)), 3.0)]))
        pan_c = max(-PAN_DEG, min(PAN_DEG, math.degrees(math.atan2(want[1], want[0]))))
        tilt_c = max(TILT_DEG[0] - DEPRESSION_DEG, min(TILT_DEG[1] - DEPRESSION_DEG, -math.degrees(math.atan2(want[2], math.hypot(want[0], want[1]))) - DEPRESSION_DEG))
        mount[0] += max(-SLEW_DEG_S * 0.01, min(SLEW_DEG_S * 0.01, pan_c - mount[0]))
        mount[1] += max(-SLEW_DEG_S * 0.01, min(SLEW_DEG_S * 0.01, tilt_c - mount[1]))
        el, az = dep + math.radians(mount[1]), math.radians(mount[0])
        tube = R @ np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), -math.sin(el)])
        cam.sig = min(noise_m, TA.ANGLE_NOISE * math.dist(bandit.p, s["pos"]))
        b, vb = cam.see(t, bandit.p, bandit.v) if netted is None else (np.asarray(bandit.p), np.zeros(3))
        cam.v_last = np.asarray(vb)
        # the approach follows the quadcopter's MEAN track (an 8 s fit): ASTOR at 30 m/s cannot turn with a weave or a
        # jink (their 30 m radius would take 3 g), and banking to try cost it its height. It flies the mean line wings
        # level and the launcher's mount takes the quadcopter as it crosses.
        bm, vbm = slow.see(t, bandit.p, bandit.v) if netted is None else (b, vb)
        wf = None
        if netted is None:
            wide.sig = cam.sig
            wide.see(t, bandit.p, bandit.v)
            wf = weave_fit(wide.buf, t, latency_s)
        sb = float(np.linalg.norm(np.asarray(vb)[:2]))
        if netted is not None:                                           # the job is done: climb away
            ap(phi_c=0.0, theta_c=math.radians(8.0), throttle=0.8)
            if t - netted > 6.0:
                outcome = "netted"
        elif phase == "attack":
            # level above the quadcopter by what puts the tube's line on it at 7 m
            h = min(2.2, max(0.5, 7.0 * math.sin(math.radians(AIM_TILT_DEG) - s["theta"]) + 0.12))     # for a shot with the tube mid-travel
            # come in along the quadcopter's own track: the aim is a point behind it on that track, drawn in to it as the
            # range closes (a collision course from the side carries a lead angle the launcher's mount cannot follow)
            d_h = math.hypot(bm[0] - s["pos"][0], bm[1] - s["pos"][1])
            sbm = max(float(np.linalg.norm(np.asarray(vbm)[:2])), 0.5)
            ub = np.array([vbm[0], vbm[1], 0.0]) / sbm
            aim = np.asarray(bm) - ub * max(0.0, min(TRAIL_M, 0.5 * (d_h - 25.0))) + np.array([0.0, 0.0, h])
            aim[2] = float(b[2]) + h
            r = aim - s["pos"]
            rng_h = math.hypot(r[0], r[1])
            aim_v = vbm
            if wf is not None and wf[2]:
                # a weave was found: fly to where the quadcopter WILL be when ASTOR gets there (the mean line plus the
                # weave at the arrival time), coming in along the mean track; that point barely moves as the range closes
                d_q = math.hypot(b[0] - s["pos"][0], b[1] - s["pos"][1])
                vm = np.array([wf[1][0], wf[1][1], 0.0])
                um = vm / max(float(np.linalg.norm(vm)), 0.5)
                t_go = d_q / max(s["V"] - float(np.linalg.norm(vm)), 5.0)
                pp, _ = wf[0](t + t_go)
                trail = max(0.0, min(TRAIL_M, 0.5 * (d_q - 25.0)))
                aim = np.array([pp[0], pp[1], float(b[2]) + h]) - um * trail
                aim_v = np.zeros(3) if trail <= 0.0 else um * 0.5 * max(s["V"] - float(np.linalg.norm(vm)), 5.0)
                r = aim - s["pos"]
                rng_h = math.hypot(r[0], r[1])
                weave_used += 1
            # a jinking quadcopter: its present heading has left the mean track
            vq2, vm2 = np.asarray(vb)[:2], np.asarray(vbm)[:2]
            cosd = float(vq2 @ vm2) / max(float(np.linalg.norm(vq2) * np.linalg.norm(vm2)), 1e-6)
            d_now = math.hypot(b[0] - s["pos"][0], b[1] - s["pos"][1])
            if not (wf is not None and wf[2]) and d_now < JINK_RANGE_M and cosd < math.cos(math.radians(JINK_DEG)):
                chase_until = t + JINK_HOLD_S
            jinker = jinker or t < chase_until                          # once seen to jink, it is chased on every pass
            chasing = jinker and d_now < JINK_RANGE_M
            sim.flap_cmd = 1.0 if jinker and d_now < 250.0 else 0.0      # the flaps go down early: their trim change is over before the chase
            if chasing:
                aim = np.array([b[0], b[1], float(b[2]) + h])
                aim_v = np.asarray(vb)
                r = aim - s["pos"]
                rng_h = math.hypot(r[0], r[1])
                chase_used += 1
            phi, gam = amj.pn_command(sim, aim, aim_v)
            n_av = 0.5 * am.RHO * s["V"] ** 2 * sim.C["S"] * sim.C["CL_MAX"] / (sim.C["mass"] * am.G)
            phi_max = math.acos(min(1.0, 1.0 / max(0.85 * n_av, 1.0001)))
            V_c = max((JINK_SPEED * Vs_flap) if chasing else (TA.PASS_SPEED * Vs), sb + min(14.0, 4.0 + 0.2 * rng_h))
            if chasing:
                n_av = 0.5 * am.RHO * s["V"] ** 2 * sim.C["S"] * sim.C["CL_MAX"] / (sim.C["mass"] * am.G)      # with the flaps as they now are
                phi_max = min(math.acos(min(1.0, 1.0 / max(0.85 * n_av, 1.0001))), math.radians(JINK_BANK_DEG if rng_h > JINK_LEVEL_M else JINK_LEVEL_BANK_DEG))
            if rng_h < TA.TERMINAL_M:
                alpha_f += (float(sim.info.get("alpha", 0.0)) - alpha_f) * 0.01 / 0.5
                kz, vzm = (JINK_KZ, JINK_VZ) if chasing else (AC.TERMINAL_KZ, TA.TERMINAL_VZ)
                vz = max(-vzm, min(vzm, kz * float(r[2])))
                gam = max(-0.3, min(0.3, vz / max(s["V"], 10.0) + alpha_f))
            else:
                alpha_f = float(sim.info.get("alpha", 0.0))
            thr = max(0.0, min(1.0, sim.trim["throttle"] + 0.15 * (V_c - s["V"]) + 0.06 * float(r[2])))
            if rng_h < 70.0 and not chasing:
                phi_max = min(phi_max, math.radians(CLOSE_BANK_DEG))          # close in: nearly wings level, the mount does the pointing
            ap(phi_c=max(-phi_max, min(phi_max, phi)), theta_c=gam, V_c=V_c, throttle=thr)
            rel = np.asarray(b) - muzzle
            dist = float(np.linalg.norm(rel))
            if net is None and left > 0 and t - t_shot > RELOAD_S and dist < 14.0 and float(rel @ tube) > 0.0:
                v0 = np.asarray(s["v_world"]) + MUZZLE_M_S * tube
                pr = predict(muzzle, v0, sim.wind, b, vb)
                if pr is not None and FIRE_M[0] <= pr[1] <= FIRE_M[1] and pr[0] < FIRE_INSIDE * half_span(pr[1]):
                    net = FlyingNet(muzzle, v0, sim.wind)
                    net.p0, net.was_ahead = muzzle.copy(), None
                    left -= 1
                    t_shot = t
                    m.body_mass[sim.bid] -= NET_KG
                    shots.append({"t": round(t, 2), "range_m": round(dist, 1), "predicted_miss_m": round(pr[0], 2), "predicted_flight_m": round(pr[1], 1),
                                  "astor_above_m": round(float(s["pos"][2] - bandit.p[2]), 2), "mount_pan_tilt_deg": [round(mount[0], 1), round(mount[1], 1)], "closing_m_s": round(float(np.linalg.norm(np.asarray(s["v_world"]) - np.asarray(bandit.v))), 1)})
            ahead = float((np.asarray(bandit.p) - np.asarray(s["pos"]))[:2] @ np.asarray(s["v_world"])[:2])
            if was_ahead is not None and was_ahead > 0.0 >= ahead and net is None:          # passed it: round again
                phase, t_phase = "away", t
            was_ahead = ahead
        else:
            los = np.asarray(b) - np.asarray(s["pos"])
            vh = np.asarray(s["v_world"])
            ang = math.degrees(math.acos(max(-1.0, min(1.0, float(los[:2] @ vh[:2]) / max(float(np.linalg.norm(los[:2]) * np.linalg.norm(vh[:2])), 1e-6)))))
            same_way = float(vh[:2] @ np.asarray(vb)[:2]) > 0.0
            # the turn back: a held speed and a steep bank (at 0.7 throttle and 40 deg it ran to 80 m/s and took 55 s and 1.7 km to come round)
            sim.flap_cmd = 0.0
            Va = AWAY_SPEED * Vs
            ap(phi_c=math.radians(AWAY_BANK_DEG), theta_c=math.radians(max(-6.0, min(8.0, 2.0 * (float(los[2]) + 6.0)))) + float(sim.info.get("alpha", 0.0)),
               V_c=Va, throttle=max(0.0, min(1.0, sim.trim["throttle"] * 2.0 + 0.15 * (Va - s["V"]))))
            if left <= 0 and t - t_phase > 2.0:
                outcome = "no_catch"
            elif t - t_phase > 6.0 and ang < 20.0 and same_way and float(np.linalg.norm(los[:2])) > 60.0:
                phase, was_ahead = "attack", None
        if t > HUNT_S and outcome is None:
            outcome = "no_catch"
        if s["pos"][2] < 1.5:
            outcome = "lost"
            break
        sim.step(5)
        z_min = min(z_min, float(sim.state()["pos"][2]))
        # the quadcopter, and the net in flight
        if netted is None:
            bandit.step(0.01, t, hunter=tuple(s["pos"]))
        else:                                                             # taken: it falls in the net
            vq = vq + (np.array([0.0, 0.0, -am.G]) - 0.5 * am.RHO * NETTED_CDA / (q_kg + NET_KG) * float(np.linalg.norm(vq)) * vq) * 0.01
            bandit.p = list(np.maximum(np.asarray(bandit.p) + vq * 0.01, [-1e9, -1e9, 0.05]))
            bandit.v = tuple(vq)
        if net is not None:
            for _ in range(5):
                net.step(0.002)
            flown = float(np.linalg.norm(net.p - net.p0))
            rel = np.asarray(bandit.p) - net.p
            vr = net.v - np.asarray(bandit.v)
            a_ = float(rel @ vr)
            if net.was_ahead is not None and net.was_ahead > 0.0 >= a_:
                u = vr / max(float(np.linalg.norm(vr)), 1e-6)
                miss = float(np.linalg.norm(rel - (rel @ u) * u))
                inside = miss <= INSIDE * half_span(flown) and RANGE_M[0] <= flown <= RANGE_M[1]
                took = bool(inside and rng.random() < P_TAKE)
                shots[-1].update({"miss_m": round(miss, 2), "flown_m": round(flown, 1), "half_span_m": round(half_span(flown), 2), "inside": bool(inside), "took": took})
                if took:
                    netted, vq = t, (q_kg * np.asarray(bandit.v) + NET_KG * net.v) / (q_kg + NET_KG)
                net.done = True
            net.was_ahead = a_
            if getattr(net, "done", False) or net.t > 1.0:
                if "took" not in shots[-1]:
                    shots[-1].update({"miss_m": None, "inside": False, "took": False, "flown_m": round(flown, 1)})
                gone = net
                net = None
                if netted is None:
                    phase, t_phase = "away", t
        if record is not None:
            record(sim, bandit, net, t, netted)
    s = sim.state()
    if outcome != "lost" and (abs(math.degrees(s["phi"])) > 70 or s["V"] < 0.9 * Vs):
        outcome = "lost"
    return {"seed": seed, "outcome": outcome, "win": outcome == "netted", "astor_lost": outcome == "lost",
            "quadcopter": {"behaviour": kind, "mass_kg": round(q_kg, 2), "speed_m_s": round(bandit.speed, 1)},
            "nets_fired": len(shots), "shots": shots, "weave_fit_used_s": round(weave_used * 0.01, 1), "chase_used_s": round(chase_used * 0.01, 1), "t_netted_s": None if netted is None else round(netted, 1),
            "astor": {"mass_kg": round(float(sim.C["mass"]), 2), "stall_m_s": round(Vs, 1), "lowest_m": round(z_min, 1), "duration_s": round(d.time - t0, 1)}}
