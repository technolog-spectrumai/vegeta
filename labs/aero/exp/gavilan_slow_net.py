"""The light net (net_drone --light) thrown slower: 15, 20 and 30 m/s, four shots per case, a quadcopter 4 m up, hovering
or crossing at 5 m/s, from a fixed launcher. Needs net_drone.py (dev_gavilan_net) beside it.    python gavilan_slow_net.py out.json"""
import json, sys, time, numpy as np, net_drone as nd
out = {}
for v0 in (15.0, 20.0, 30.0):
    nd.configure(light=True, drone="quad"); nd.V0 = v0
    cal = nd.calibrate(nd.Sim())
    for vd in (0.0, 5.0):
        for R in (3.0, 4.0, 5.0):
            sim = nd.Sim(); rng = np.random.default_rng(5); res = []
            for k in range(4):
                r = nd.shot(sim, cal, np.array([R, 0.0, 4.0]), np.array([0.0, vd, 0.0]), rng)
                res.append(bool(r["captured"]))
            out[f"{v0:g} m/s, bandit {vd:g} m/s, {R:g} m"] = f"{sum(res)}/{len(res)}"
            print(f"net {v0:g} m/s, bandit {vd:g} m/s crossing, range {R:g} m: captured {sum(res)}/{len(res)}", flush=True)
            json.dump(out, open(sys.argv[1], "w"), indent=1)
