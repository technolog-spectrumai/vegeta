"""The mixed-fleet civilian response scenario (``docs/pekari_pumba.md`` §6): two Pekari-2M, one Pumba and
(optionally) a survey drone answer an incident at an industrial hall — deterministic, on ``fleet.Fleet``.

The site (40 x 30 m, 0.2 m cells): a yard with the base station and a staging area; a hall (26 x 22 m) with a
3 m main door, racking, a collapsed doorway into the plant room (a debris pile only Pumba can clear, after the
operator's approval) and a 1 m crawlway under a 0.55 m pipe rack into the control room (only Pekari-2M folded
fits). The control room is out of radio range of the base: whoever goes in loses the link and must come back.

The operator's tasks: thermal inspection of the valves in the control room (confined), first aid and water for a
casualty in the hall (25 kg), a 120 kg pump to the plant room (heavy transport, after the debris), closing a
valve in the plant room with two arms, and an aerial survey of the roof and yard (the drone; dropped without
one). Events: Pekari-2M "P2" has a drive fault at 40 s (``FAULT_T``), the operator presses the emergency stop at
240 s and resets it at 270 s (``ESTOP``).

    import fleet_scenario as fs
    res = fs.run()                     # dict: tasks, coverage, agents, events
    fs.table(res)
"""
from __future__ import annotations

import numpy as np

import fleet as fl

RES = 0.2
SIZE = (40.0, 30.0)
BASE = (3.0, 15.0)
STAGING_KIT = (6.0, 8.0)
STAGING_PUMP = (7.0, 22.0)
CASUALTY = (20.0, 17.0)
VALVES_CONTROL = (34.0, 8.5)
VALVE_PLANT = (35.5, 23.0)
PUMP_DEST = (33.0, 19.0)
ROOF_SURVEY = (25.0, 28.5)
FAULT_T = 40.0                 # P2's drive fault
FAULT_T_TIMON = 15.0           # with Timon: P2's drive fault early, in the hall, where Timon can reach it
BATTERY_T = 60.0               # with Timon: P1's battery runs low (a cell fault: down to 15 %)
TIMON_HOME = (5.0, 10.0)
ESTOP = (240.0, 270.0)         # the operator's emergency stop and reset


def _box(a, x0, x1, y0, y1, v):
    i0, i1 = int(round(x0 / RES)), int(round(x1 / RES))
    j0, j1 = int(round(y0 / RES)), int(round(y1 / RES))
    a[i0:i1, j0:j1] = v


def site() -> fl.GridMap:
    """The ground truth: walls, doors, racking, the debris pile (id 1), the pipe rack's headroom, indoor cells."""
    nx, ny = int(SIZE[0] / RES), int(SIZE[1] / RES)
    occ = np.zeros((nx, ny), np.int8)
    indoor = np.zeros((nx, ny), bool)
    debris = np.zeros((nx, ny), np.int16)
    ceiling = np.full((nx, ny), np.inf)
    _box(indoor, 12.0, 38.0, 4.0, 26.0, True)
    w = RES                                                       # one-cell walls
    for x0, x1, y0, y1 in ((12.0, 38.0, 4.0, 4.0 + w), (12.0, 38.0, 26.0 - w, 26.0),     # south, north
                           (12.0, 12.0 + w, 4.0, 26.0), (38.0 - w, 38.0, 4.0, 26.0),     # west, east
                           (28.0, 28.0 + w, 4.0, 26.0),                                   # the partition
                           (28.0, 38.0, 13.0, 13.0 + w)):                                 # control | plant room
        _box(occ, x0, x1, y0, y1, fl.OCC)
    _box(occ, 12.0, 12.0 + w, 13.0, 16.0, fl.FREE)                 # main door, 3 m
    _box(occ, 28.0, 28.0 + w, 18.0, 20.0, fl.FREE)                 # plant-room door, 2 m ...
    _box(occ, 26.6, 29.4, 17.6, 20.4, fl.OCC)                      # ... blocked by the collapse
    _box(debris, 26.6, 29.4, 17.6, 20.4, 1)
    _box(occ, 28.0, 28.0 + w, 7.0, 8.0, fl.FREE)                   # the crawlway, 1 m wide ...
    _box(ceiling, 26.0, 30.0, 6.4, 8.6, 0.55)                      # ... under a 0.55 m pipe rack
    for y in (6.4, 10.0, 21.6, 23.8):                              # racking in the hall
        _box(occ, 15.0, 25.0, y, y + 0.4, fl.OCC)
    return fl.GridMap(occ, RES, indoor, debris, ceiling)


def region(m: fl.GridMap) -> np.ndarray:
    """What the fleet explores: the hall and its rooms (the yard is the drone's survey task)."""
    return m.indoor.copy()


def agents(truth: fl.GridMap, *, with_drone: bool = True, third_party: dict | None = None,
           with_timon: bool = False) -> list:
    import pekari_2m_robot as p2
    import pumba_robot as pu

    p2m = fl.profile_from_platform(p2.PEKARI_2M)
    pum = fl.profile_from_platform(pu.PUMBA)
    out = [fl.Agent("P1", p2m, (4.0, 12.0), (4.0, 12.0), truth.copy_unknown()),
           fl.Agent("P2", p2m, (4.0, 18.0), (4.0, 18.0), truth.copy_unknown()),
           fl.Agent("PUMBA", pum, (6.0, 15.0), (6.0, 15.0), truth.copy_unknown())]
    if with_drone:
        out.append(fl.Agent("UAV", fl.drone_profile(), BASE, BASE, truth.copy_unknown()))
    if with_timon:
        import timon_robot as tr

        out.append(fl.Agent("TIMON", fl.profile_from_platform(tr.TIMON), TIMON_HOME, TIMON_HOME, truth.copy_unknown()))
    if third_party is not None:
        prof = fl.third_party_profile(third_party["name"], third_party)
        out.append(fl.Agent("TP", prof, (4.0, 22.0), (4.0, 22.0), truth.copy_unknown()))
    return out


def operator_tasks(with_drone: bool = True) -> list:
    T = fl.Task
    out = [T("inspect_control_room", "inspect", VALVES_CONTROL, tags=frozenset({"inspect", "thermal", "confined"}),
             duration_s=60.0, priority=2, reach_m=0.6),
           T("first_aid", "transport", STAGING_KIT, dest=CASUALTY, tags=frozenset({"transport"}), payload_kg=25.0,
             duration_s=30.0, priority=3, reach_m=0.6),
           T("pump_to_plant_room", "transport", STAGING_PUMP, dest=PUMP_DEST, tags=frozenset({"heavy_transport"}),
             payload_kg=120.0, duration_s=90.0, priority=2, after=("clear_debris1",), reach_m=0.8),
           T("close_plant_valve", "manipulate", VALVE_PLANT, tags=frozenset({"manipulate", "dual_arm"}), lift_kg=4.0,
             duration_s=45.0, priority=2, after=("clear_debris1",), reach_m=0.7)]
    if with_drone:
        out.append(T("aerial_survey", "map_aerial", ROOF_SURVEY, tags=frozenset({"aerial"}), duration_s=120.0, priority=1))
    return out


def build(*, with_drone: bool = True, fault: bool = True, estop: bool = True, approve_at: float = 60.0,
          with_timon: bool = False,
          third_party: dict | None = None) -> fl.Fleet:
    truth = site()
    sup = fl.Supervisor()
    for tk in operator_tasks(with_drone):
        sup.add_task(0.0, tk)
    sup.approve(approve_at, "clear_debris1")              # the operator approves the debris removal (pre-approval
    if estop:                                             # if it is not reported yet: applied when it is)
        sup.estop(ESTOP[0])
        sup.reset(ESTOP[1])
    faults = [(FAULT_T_TIMON if with_timon else FAULT_T, "P2", "fail")] if fault else []
    if with_timon:
        faults.append((BATTERY_T, "P1", "battery"))
    return fl.Fleet(truth, agents(truth, with_drone=with_drone, third_party=third_party, with_timon=with_timon),
                    fl.Comms(BASE, range_m=40.0), sup,
                    faults=faults, explore_region=region(truth))


def run(t_end: float = 2400.0, **kw) -> dict:
    return build(**kw).run(t_end)


def table(res: dict):
    """The tasks as a DataFrame: kind, status, who did it, when."""
    import pandas as pd

    rows = [{"task": k, **{f: v[f] for f in ("kind", "status", "agent", "done_t")}, "reassigned":
             sum(1 for h in v["history"] if h[1] == "released")} for k, v in res["tasks"].items()]
    return pd.DataFrame(rows).set_index("task")


def plot(fleet: fl.Fleet, path=None):
    """The shared map at the end with the agents, the operator's task sites and the base (matplotlib)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = fleet.coord.map
    img = np.where(m.occ == fl.UNKNOWN, 0.75, np.where(m.occ == fl.OCC, 0.1, 1.0)).T
    fig, ax = plt.subplots(figsize=(9, 6.8))
    ax.imshow(img, origin="lower", cmap="gray", vmin=0, vmax=1, extent=(0, SIZE[0], 0, SIZE[1]))
    low = np.isfinite(fleet.truth.ceiling)
    ax.contour(np.arange(m.shape[0]) * RES + RES / 2, np.arange(m.shape[1]) * RES + RES / 2, low.T, levels=[0.5],
               colors="tab:purple", linewidths=1)
    colors = {"P1": "tab:orange", "P2": "tab:red", "PUMBA": "goldenrod", "UAV": "tab:blue", "TP": "tab:green",
              "TIMON": "tab:cyan"}
    for a in fleet.agents:
        ax.plot(*a.pos, "o", ms=9, color=colors.get(a.id, "k"), mec="k", label=f"{a.id} ({a.profile.kind}, {a.state})")
    for name, p in (("base", BASE), ("kit", STAGING_KIT), ("pump", STAGING_PUMP), ("casualty", CASUALTY),
                    ("valves", VALVES_CONTROL), ("plant valve", VALVE_PLANT)):
        ax.plot(*p, "kx")
        ax.annotate(name, p, textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Shared map at t = {fleet.t:.0f} s (grey unknown, purple: 0.55 m pipe rack)")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
    return fig
