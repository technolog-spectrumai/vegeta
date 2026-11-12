"""Heterogeneous fleet coordination for the Pekari family (``docs/pekari_pumba.md`` §5) — deterministic baseline.

Nothing in the repository coordinated several robots before this module (no WOTAN, MIMIR, Wagenburg or other
fleet layer exists here); it is new, small and deterministic on purpose: the same inputs give the same allocation,
the same events and the same result, so a scenario is a regression test. Learned multi-agent policies are not used;
they would be judged against this baseline.

The pieces, each a plain class:

* ``Agent``: one robot — its capability profile (``profile_from_platform`` for Pekari-2M and Pumba from
  ``pekari_family.capabilities``; ``drone_profile`` for a future aerial drone; ``third_party_profile`` for another
  maker's robot mapped onto the same tags), its state, its own map and its battery.
* ``Task``: what the operator or the coordinator wants done (``explore``, ``inspect``, ``transport``,
  ``clear_debris``, ``manipulate``, ``map_aerial``), with the capability tags, payload, lift and approval it needs.
* ``GridMap``: the site as cells (truth and each agent's knowledge): free / occupied / unknown, indoor or outdoor,
  debris ids; width-aware path planning (a robot passes where the free width ≥ its width + ``CLEARANCE``).
* ``Comms``: the mesh — a link exists when the distance plus a per-wall penalty is within range; agents relay;
  the base station is the coordinator's link. A separate safety radio carries the emergency stop (assumed
  independent of the mesh, as industrial radio e-stops are).
* ``Coordinator``: capability-based allocation (a sequential single-item auction: the cheapest eligible
  (agent, task) pair first, ties broken by ids), the shared map (cell-wise merge of what connected agents saw),
  frontier exploration tasks, debris tasks when debris blocks a route, dynamic reassignment (failure, lost link,
  blocked path), communication-loss recovery (the agent finishes, then returns to its last connected point; the
  coordinator releases its task after ``T_RELEASE``; duplicates are reconciled on reconnection), and the
  ``Supervisor``'s commands (emergency stop, hold, release, approvals, operator tasks) applied before anything else.
* ``Fleet``: the discrete-time simulation of all of it (1 s ticks) with an event log.

Units SI, positions (x, y) [m] in the site frame.
"""
from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

__all__ = ["CLEARANCE", "T_RETURN", "T_RELEASE", "GridMap", "Profile", "profile_from_platform", "drone_profile",
           "third_party_profile", "Task", "Agent", "Comms", "Supervisor", "Coordinator", "Fleet"]

CLEARANCE = 0.05          # m: a robot passes where the free width is at least its width plus this
T_RETURN = 60.0           # s without a link before an agent heads back to its last connected point
T_RELEASE = 90.0          # s without a heartbeat before the coordinator releases the agent's task
MIN_FRONTIER_M = 0.6      # m: a shorter frontier cluster is noise (cells a ray missed on a wall)
#: The swappable battery module of each robot kind (what a servicing arm lifts) [kg].
MODULE_KG = {"Pekari-2M": 9.0, "Pumba": 26.0, "Calydon": 22.0, "Perseus": 16.0, "Timon": 26.0}
SWAP_S = 180.0            # a battery module swap on site: dock, open, out, in, close [s]
HOOKUP_S = 90.0           # hooking a failed robot up for towing [s]
LOW_BATTERY = 0.20        # below this state of charge an agent takes no new task and goes home

FREE, OCC, UNKNOWN = 0, 1, -1


# ----------------------------------------------------------------------------------------------- the map
class GridMap:
    """A site grid: ``occ`` (cells × cells) FREE / OCC / UNKNOWN, ``indoor`` (bool), ``debris`` (0 or a pile id),
    ``res`` [m/cell], the origin at cell (0, 0)'s corner. ``free_width`` is the passage width at each cell [m]."""

    def __init__(self, occ: np.ndarray, res: float, indoor: np.ndarray | None = None, debris: np.ndarray | None = None,
                 ceiling: np.ndarray | None = None):
        self.occ = np.asarray(occ, dtype=np.int8)
        self.ceiling = np.full(self.occ.shape, np.inf) if ceiling is None else np.asarray(ceiling, float)
        self.res = float(res)
        self.indoor = np.zeros(self.occ.shape, bool) if indoor is None else np.asarray(indoor, bool)
        self.debris = np.zeros(self.occ.shape, np.int16) if debris is None else np.asarray(debris, np.int16)
        self.stamp = np.full(self.occ.shape, -np.inf)       # when each cell was last seen (knowledge maps)
        self._width = None

    @property
    def shape(self):
        return self.occ.shape

    def cell(self, xy) -> tuple:
        i = int(math.floor(xy[0] / self.res))
        j = int(math.floor(xy[1] / self.res))
        return (min(max(i, 0), self.shape[0] - 1), min(max(j, 0), self.shape[1] - 1))

    def xy(self, c) -> tuple:
        return ((c[0] + 0.5) * self.res, (c[1] + 0.5) * self.res)

    def copy_unknown(self) -> "GridMap":
        """A map of the same site with every cell unknown (an agent's or the coordinator's knowledge)."""
        return GridMap(np.full(self.shape, UNKNOWN, np.int8), self.res, self.indoor.copy(), ceiling=self.ceiling.copy())

    def invalidate(self):
        self._width = None

    def free_width(self) -> np.ndarray:
        """Passage width [m] at each cell: (2 d − 1) × res, d the distance in cells from the cell's centre to the
        nearest occupied cell's (Euclidean with SciPy, else 4-connected BFS); the map's border is a wall; unknown
        cells count as free (optimistic planning). A gap of n cells reads n × res at its centre."""
        if self._width is None:
            try:
                from scipy.ndimage import distance_transform_edt
            except ImportError:
                distance_transform_edt = None
            if distance_transform_edt is not None:
                padded = np.pad(self.occ != OCC, 1, constant_values=False)
                d = distance_transform_edt(padded)[1:-1, 1:-1]
                self._width = (2 * d - 1) * self.res
                return self._width
            occ = self.occ == OCC
            d = np.full(self.shape, 10 ** 6, np.int32)
            q = deque()
            for c in zip(*np.nonzero(occ)):
                d[c] = 0
                q.append(c)
            nx, ny = self.shape
            if not q:
                d[:] = 10 ** 3
            while q:
                i, j = q.popleft()
                for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                    if 0 <= a < nx and 0 <= b < ny and d[a, b] > d[i, j] + 1:
                        d[a, b] = d[i, j] + 1
                        q.append((a, b))
            # the map's border is a wall too
            ii, jj = np.indices(self.shape)
            border = np.minimum(np.minimum(ii + 1, nx - ii), np.minimum(jj + 1, ny - jj))
            d = np.minimum(d, border)
            self._width = (2 * d - 1) * self.res
        return self._width

    def passable(self, width: float, *, flies: bool = False, height: float = 0.0) -> np.ndarray:
        """Cells a robot of ``width`` and ``height`` can occupy: free (or unknown) with enough free width and
        headroom (``ceiling``: pipe racks, low ducts); a flyer: outdoor cells (it flies over obstacles and debris,
        never indoors)."""
        if flies:
            return ~self.indoor
        return (self.occ != OCC) & (self.free_width() >= width + CLEARANCE - 1e-9) & (self.ceiling >= height + CLEARANCE)

    def path(self, start, goal, width: float, *, flies: bool = False, goal_tolerance: float = 0.0, height: float = 0.0):
        """The shortest 8-connected path [(x, y)] from ``start`` to ``goal`` for a robot of ``width`` (A*), or
        None. ``goal_tolerance`` [m]: stop at any passable cell this close to the goal (to reach a task's object
        that sits in an obstacle, e.g. debris or a valve on a wall)."""
        ok = self.passable(width, flies=flies, height=height)
        s, g = self.cell(start), self.cell(goal)
        if not ok[s]:                     # the robot stands where the map now says it does not fit (it learnt a
            ok = ok.copy()                # wall next to it): let it back out over the free cells under it
            r = int(math.ceil((width / 2 + CLEARANCE) / self.res)) + 1
            i0, i1 = max(s[0] - r, 0), min(s[0] + r + 1, self.shape[0])
            j0, j1 = max(s[1] - r, 0), min(s[1] + r + 1, self.shape[1])
            ok[i0:i1, j0:j1] |= self.occ[i0:i1, j0:j1] != OCC
        tol = max(goal_tolerance / self.res, 0.0)
        nx, ny = self.shape

        def h(c):
            return math.hypot(c[0] - g[0], c[1] - g[1])

        def at_goal(c):
            return c == g or (tol > 0 and h(c) <= tol)

        openq = [(h(s), 0.0, s)]
        came = {s: None}
        cost = {s: 0.0}
        steps = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                 (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2)))
        end = None
        while openq:
            _, c0, c = heapq.heappop(openq)
            if c0 > cost.get(c, math.inf) + 1e-12:
                continue
            if at_goal(c):
                end = c
                break
            for di, dj, w in steps:
                a, b = c[0] + di, c[1] + dj
                if not (0 <= a < nx and 0 <= b < ny) or not ok[a, b]:
                    continue
                if di and dj and not (ok[c[0] + di, c[1]] and ok[c[0], c[1] + dj]):
                    continue              # no corner cutting
                nc = c0 + w
                if nc < cost.get((a, b), math.inf) - 1e-12:
                    cost[(a, b)] = nc
                    came[(a, b)] = c
                    heapq.heappush(openq, (nc + h((a, b)), nc, (a, b)))
        if end is None:
            return None
        cells = []
        c = end
        while c is not None:
            cells.append(c)
            c = came[c]
        return [self.xy(c) for c in reversed(cells)]

    def frontier(self) -> np.ndarray:
        """Known-free cells next to unknown cells (bool)."""
        free = self.occ == FREE
        unk = self.occ == UNKNOWN
        nb = np.zeros_like(unk)
        nb[1:, :] |= unk[:-1, :]
        nb[:-1, :] |= unk[1:, :]
        nb[:, 1:] |= unk[:, :-1]
        nb[:, :-1] |= unk[:, 1:]
        return free & nb


def _rays(radius_cells: int, n_rays: int = 90):
    """Cell offsets along ``n_rays`` rays out to ``radius_cells`` (n_rays × steps × 2)."""
    steps = np.arange(1, radius_cells + 1)
    ang = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    off = np.stack([np.outer(np.cos(ang), steps), np.outer(np.sin(ang), steps)], axis=-1)
    return np.rint(off).astype(int)


def sense(truth: GridMap, known: GridMap, xy, radius: float, rays=None, t: float = 0.0) -> int:
    """Reveal into ``known`` what a range sensor at ``xy`` sees of ``truth`` at time ``t`` (rays stop at the first
    occupied cell, which is revealed); returns the number of newly known cells."""
    c = np.array(truth.cell(xy))
    rays = _rays(max(1, int(radius / truth.res))) if rays is None else rays
    cells = c + rays
    nx, ny = truth.shape
    inside = (cells[..., 0] >= 0) & (cells[..., 0] < nx) & (cells[..., 1] >= 0) & (cells[..., 1] < ny)
    ci = np.clip(cells[..., 0], 0, nx - 1)
    cj = np.clip(cells[..., 1], 0, ny - 1)
    occ = (truth.occ[ci, cj] == OCC) | ~inside
    blocked_before = np.cumsum(occ, axis=1) - occ            # occupied cells strictly before this step
    seen = inside & (blocked_before == 0)
    si, sj = ci[seen], cj[seen]
    before = int((known.occ[si, sj] == UNKNOWN).sum())
    known.occ[si, sj] = truth.occ[si, sj]
    known.debris[si, sj] = truth.debris[si, sj]
    known.stamp[si, sj] = t
    known.occ[c[0], c[1]] = truth.occ[c[0], c[1]]
    known.stamp[c[0], c[1]] = t
    known.invalidate()
    return before


# ----------------------------------------------------------------------------------------------- profiles
@dataclass(frozen=True)
class Profile:
    """What an agent can do: ``tags`` (capabilities: ground, aerial, map, inspect, thermal, manipulate,
    transport, heavy_lift, heavy_transport, dual_arm, confined), payload and arm lift [kg], width [m], speed
    [m/s], endurance [h], sensor range [m], whether it flies, the power it draws working [W] (battery model)."""

    kind: str
    tags: frozenset
    payload_kg: float
    lift_kg: float
    width_m: float
    height_m: float
    v_m_s: float
    endurance_h: float
    sensor_range_m: float
    flies: bool = False
    source: str = ""


def profile_from_platform(pl) -> Profile:
    """A Pekari-family platform's profile from ``pekari_family.capabilities`` (one source of truth)."""
    import pekari_family as pf

    cap = pf.capabilities(pl)
    rng = 12.0 if any(s.startswith("lidar") for s in pl.sensors) else 6.0
    return Profile(kind=pl.name, tags=frozenset(cap["tags"]), payload_kg=cap["payload_kg"], lift_kg=cap["arm_lift_kg"],
                   width_m=cap["width_m"], height_m=cap["height_folded_m"], v_m_s=min(cap["v_max_m_s"], 1.2), endurance_h=cap["endurance_h"],
                   sensor_range_m=rng, source="pekari_family.capabilities")


def drone_profile(name: str = "survey quadcopter (future)") -> Profile:
    """A future aerial drone: maps and inspects from above, outdoors only, carries nothing (assumed class values:
    a 25-minute survey quadcopter at 4 m/s with a 20 m downward camera footprint)."""
    return Profile(kind=name, tags=frozenset({"aerial", "map", "inspect", "thermal"}), payload_kg=0.0, lift_kg=0.0,
                   width_m=0.0, height_m=0.0, v_m_s=4.0, endurance_h=25 / 60, sensor_range_m=20.0, flies=True,
                   source="assumed class values (no drone in the land lab yet)")


#: How another maker's capability names map onto the fleet's tags (extend per robot).
THIRD_PARTY_TAGS = {"navigation": "ground", "slam": "map", "camera": "inspect", "ir_camera": "thermal",
                    "arm": "manipulate", "cargo": "transport", "narrow": "confined"}


def third_party_profile(name: str, their_caps: dict) -> Profile:
    """A third-party robot described in its own words (``their_caps``: ``capabilities`` list in its vocabulary,
    ``payload_kg``, ``lift_kg``, ``width_m``, ``height_m``, ``speed_m_s``, ``endurance_h``, ``sensor_range_m``) mapped through
    ``THIRD_PARTY_TAGS``; unknown capability names are dropped (never invented)."""
    tags = frozenset(THIRD_PARTY_TAGS[c] for c in their_caps.get("capabilities", ()) if c in THIRD_PARTY_TAGS)
    return Profile(kind=name, tags=tags, payload_kg=float(their_caps.get("payload_kg", 0.0)),
                   lift_kg=float(their_caps.get("lift_kg", 0.0)), width_m=float(their_caps["width_m"]),
                   height_m=float(their_caps.get("height_m", 0.5)),
                   v_m_s=float(their_caps.get("speed_m_s", 0.5)), endurance_h=float(their_caps.get("endurance_h", 1.0)),
                   sensor_range_m=float(their_caps.get("sensor_range_m", 5.0)), source=f"third party: {name}")


# ----------------------------------------------------------------------------------------------- tasks, agents
@dataclass
class Task:
    """A unit of work at ``pos`` (and ``dest`` for transport): it needs ``tags`` (all), ``payload_kg`` on the deck,
    ``lift_kg`` in an arm, ``duration_s`` of work on site; ``requires_approval`` waits for the supervisor;
    ``after`` lists task ids that must be done first. ``reach_m``: how close the robot must get (an object on a
    wall or in debris is reached by the arm)."""

    id: str
    kind: str
    pos: tuple
    tags: frozenset = frozenset()
    payload_kg: float = 0.0
    lift_kg: float = 0.0
    duration_s: float = 10.0
    dest: tuple | None = None
    priority: int = 1
    requires_approval: bool = False
    after: tuple = ()
    reach_m: float = 0.0
    debris_id: int = 0
    subject: str | None = None    # the agent a service task is for (swap_battery, recover)
    status: str = "open"          # open, awaiting_approval, assigned, active, done, cancelled
    agent: str | None = None
    approved: bool = False
    done_t: float | None = None
    history: list = field(default_factory=list)


@dataclass
class Agent:
    id: str
    profile: Profile
    pos: tuple
    home: tuple
    known: GridMap
    soc: float = 1.0                       # state of charge
    state: str = "idle"                    # idle, moving, working, returning, holding, estopped, failed
    task: str | None = None
    path: list = field(default_factory=list)
    work_left: float = 0.0
    phase: str = ""                        # transport: 'pickup' then 'deliver'
    last_link_t: float = 0.0
    last_link_pos: tuple = (0.0, 0.0)
    linked: bool = True
    done_offline: list = field(default_factory=list)   # tasks finished out of contact, reported on reconnection
    estop: bool = False
    hold: bool = False
    odometer_m: float = 0.0
    events: list = field(default_factory=list)


# ----------------------------------------------------------------------------------------------- comms
@dataclass
class Comms:
    """The mesh: base station at ``base`` [m]; a link between two nodes when distance + ``wall_penalty_m`` per wall
    cell on the line ≤ ``range_m``; agents relay. The safety radio (e-stop) reaches ``safety_range_m`` regardless."""

    base: tuple
    range_m: float = 30.0
    wall_penalty_m: float = 6.0
    safety_range_m: float = 300.0

    def link(self, truth: GridMap, a, b) -> bool:
        d = math.dist(a, b)
        if d > self.range_m:
            return False
        n = max(2, int(d / truth.res) + 1)
        walls = 0
        prev = None
        for k in range(n + 1):
            c = truth.cell((a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n))
            if c != prev and truth.occ[c] == OCC:
                walls += 1
            prev = c
        return d + self.wall_penalty_m * walls <= self.range_m

    def connected(self, truth: GridMap, agents) -> set:
        """Ids of the agents with a mesh route to the base (BFS over links, relays through agents)."""
        live = [a for a in agents if a.state not in ("failed", "recovered")]
        reached, frontier = set(), [("base", self.base)]
        while frontier:
            _, p = frontier.pop()
            for a in live:
                if a.id not in reached and self.link(truth, p, a.pos):
                    reached.add(a.id)
                    frontier.append((a.id, a.pos))
        return reached


# ----------------------------------------------------------------------------------------------- supervisor
class Supervisor:
    """The human operator: commands queued with a time and applied by the coordinator before anything else.

    ``estop(t, agent=None)`` (None = all; latched), ``reset(t, agent=None)``, ``hold(t, agent)`` / ``release(t,
    agent)``, ``approve(t, task_id)``, ``add_task(t, task)``, ``cancel(t, task_id)``. ``auto_approve``: approve on
    request (a scenario without an operator in the loop)."""

    def __init__(self, auto_approve: bool = False):
        self.queue = []
        self.auto_approve = auto_approve
        self.log = []

    def _q(self, t, kind, **kw):
        self.queue.append((float(t), len(self.queue), kind, kw))
        self.queue.sort(key=lambda e: (e[0], e[1]))
        return self

    def estop(self, t, agent=None):
        return self._q(t, "estop", agent=agent)

    def reset(self, t, agent=None):
        return self._q(t, "reset", agent=agent)

    def hold(self, t, agent):
        return self._q(t, "hold", agent=agent)

    def release(self, t, agent):
        return self._q(t, "release", agent=agent)

    def approve(self, t, task_id):
        return self._q(t, "approve", task=task_id)

    def add_task(self, t, task: Task):
        return self._q(t, "add_task", task=task)

    def cancel(self, t, task_id):
        return self._q(t, "cancel", task=task_id)

    def due(self, t):
        out = [e for e in self.queue if e[0] <= t + 1e-9]
        self.queue = [e for e in self.queue if e[0] > t + 1e-9]
        return out


# ----------------------------------------------------------------------------------------------- coordinator
class Coordinator:
    """Allocation, the shared map, exploration, reassignment and recovery (see the module docstring)."""

    def __init__(self, truth: GridMap, comms: Comms, *, explore_limit: int = 4, explore_region=None):
        self.map = truth.copy_unknown()
        self.comms = comms
        self.tasks: dict = {}
        self.events = []
        self.explore_limit = explore_limit
        self.explore_region = explore_region      # bool mask: frontiers outside it are not explored
        self._n_explore = 0
        self._failed_pairs = set()               # (agent, task) that could not be planned: not offered again
        self._known_debris = set()
        self.pre_approved = set()
        self.unreachable = np.zeros(truth.shape, bool)      # frontier given up: no agent fits there

    # ---- bookkeeping
    def log(self, t, event, **kw):
        self.events.append((round(float(t), 3), event, kw))

    def add(self, task: Task, t=0.0):
        if task.id in self.pre_approved:
            task.approved = True
        if task.requires_approval and not task.approved:
            task.status = "awaiting_approval"
        self.tasks[task.id] = task
        self.log(t, "task_added", task=task.id, kind=task.kind)

    # ---- map
    def merge(self, agent: Agent):
        """Cell-wise merge of the agent's knowledge into the shared map and back: the newer observation of a cell
        wins (``stamp``), so cleared debris and opened passages propagate and stale views do not come back."""
        km, sm = agent.known, self.map
        newer = km.stamp > sm.stamp
        if newer.any():
            changed = newer & (sm.occ != km.occ)
            sm.occ[newer], sm.debris[newer], sm.stamp[newer] = km.occ[newer], km.debris[newer], km.stamp[newer]
            if changed.any():
                sm.invalidate()
        back = sm.stamp > km.stamp
        if back.any():
            km.occ[back], km.debris[back], km.stamp[back] = sm.occ[back], sm.debris[back], sm.stamp[back]
            km.invalidate()

    # ---- eligibility and cost
    def eligible(self, a: Agent, task: Task) -> bool:
        p = a.profile
        if not task.tags <= p.tags:
            return False
        if task.payload_kg > p.payload_kg + 1e-9 or task.lift_kg > p.lift_kg + 1e-9:
            return False
        if a.soc < LOW_BATTERY or (a.id, task.id) in self._failed_pairs:
            return False
        return True

    def plan(self, a: Agent, task: Task):
        """The agent's path to the task (and on to its destination for a transport), on the shared map."""
        m = self.map
        p1 = m.path(a.pos, task.pos, a.profile.width_m, flies=a.profile.flies, height=a.profile.height_m, goal_tolerance=task.reach_m)
        if p1 is None:
            return None, math.inf
        length = sum(math.dist(p1[i], p1[i + 1]) for i in range(len(p1) - 1))
        if task.dest is not None:
            p2 = m.path(p1[-1], task.dest, a.profile.width_m, flies=a.profile.flies, height=a.profile.height_m, goal_tolerance=task.reach_m)
            if p2 is None:
                return None, math.inf
            length += sum(math.dist(p2[i], p2[i + 1]) for i in range(len(p2) - 1))
        return p1, length

    def cost(self, a: Agent, task: Task, length: float) -> float:
        """Time to finish [s] (travel at the agent's speed plus the work), over the task's priority."""
        return (length / max(a.profile.v_m_s, 1e-3) + task.duration_s) / max(task.priority, 1)

    def ready(self, task: Task) -> bool:
        return task.status == "open" and all(d in self.tasks and self.tasks[d].status == "done" for d in task.after)

    # ---- allocation: sequential single-item auction
    def allocate(self, t, agents, connected):
        free = [a for a in agents if a.id in connected and a.state == "idle" and not a.estop and not a.hold]
        open_tasks = [tk for tk in self.tasks.values() if self.ready(tk)]
        bids = []
        for a in free:
            for tk in open_tasks:
                if not self.eligible(a, tk):
                    continue
                if math.dist(a.pos, tk.pos) / max(a.profile.v_m_s, 1e-3) > 3600 * a.soc * a.profile.endurance_h * 0.5:
                    continue                                          # out of range on the battery left
                bids.append((tk, a))
        # cost needs a path: evaluate the cheapest straight-line candidates first, plan lazily
        scored = []
        for tk, a in bids:
            est = self.cost(a, tk, math.dist(a.pos, tk.pos))
            scored.append((est, -tk.priority, a.id, tk.id))
        scored.sort()
        taken_a, taken_t = set(), set()
        planned = {}
        for est, _, aid, tid in scored:
            if aid in taken_a or tid in taken_t:
                continue
            a = next(x for x in agents if x.id == aid)
            tk = self.tasks[tid]
            path, length = self.plan(a, tk)
            if path is None:
                self._failed_pairs.add((aid, tid))
                self.log(t, "no_path", agent=aid, task=tid)
                continue
            planned[(aid, tid)] = (path, length)
            taken_a.add(aid)
            taken_t.add(tid)
            self.assign(t, a, tk, path)
        # a frontier no live agent can reach is given up (and kept out of the frontier), not retried for ever
        live = [a for a in agents if a.state not in ("failed", "recovered") and a.id in connected]
        for tk in open_tasks:
            if tk.kind not in ("explore", "swap_battery", "recover") or tk.status != "open":
                continue
            able = [a for a in live if tk.tags <= a.profile.tags]
            if able and all((a.id, tk.id) in self._failed_pairs for a in able):
                tk.status = "cancelled"
                if tk.kind != "explore":
                    self.log(t, "unreachable", task=tk.id, pos=tuple(round(float(v), 2) for v in tk.pos),
                             why="no servicing agent has a path to it")
                    continue
                c = self.map.cell(tk.pos)
                r = int(round(1.0 / self.map.res))
                self.unreachable[max(c[0] - r, 0):c[0] + r + 1, max(c[1] - r, 0):c[1] + r + 1] = True
                self.log(t, "unreachable", task=tk.id, pos=tuple(round(float(v), 2) for v in tk.pos))

    def assign(self, t, a: Agent, tk: Task, path):
        tk.status, tk.agent = "assigned", a.id
        tk.history.append((t, "assigned", a.id))
        a.task, a.state, a.path = tk.id, "moving", list(path[1:])
        a.phase = "pickup" if tk.dest is not None else "go"
        self.log(t, "assigned", agent=a.id, task=tk.id, kind=tk.kind)

    def release(self, t, tk: Task, why: str):
        if tk.status in ("done", "cancelled"):
            return
        tk.history.append((t, "released", tk.agent, why))
        self.log(t, "released", task=tk.id, agent=tk.agent, why=why)
        tk.status, tk.agent = "open", None

    # ---- exploration and debris
    def explore_tasks(self, t, agents):
        """Frontier clusters (8-connected) of the shared map → explore tasks for ``map`` agents, at most
        ``explore_limit`` open at once; a frontier cell within 2 m of an existing explore task is covered."""
        fr = self.map.frontier() & ~self.unreachable
        if self.explore_region is not None:
            fr &= self.explore_region
        live = [tk for tk in self.tasks.values() if tk.kind == "explore" and tk.status not in ("done", "cancelled")]
        for tk in live:                                             # a frontier that vanished is done
            c = self.map.cell(tk.pos)
            i0, i1 = max(c[0] - 3, 0), c[0] + 4
            j0, j1 = max(c[1] - 3, 0), c[1] + 4
            if not fr[i0:i1, j0:j1].any() and tk.status == "open":
                tk.status, tk.done_t = "done", t
                self.log(t, "explored", task=tk.id)
        live = [tk for tk in self.tasks.values() if tk.kind == "explore" and tk.status not in ("done", "cancelled")]
        if len(live) >= self.explore_limit or not fr.any():
            return
        for comp in self.frontier_clusters(fr):
            if len(live) >= self.explore_limit:
                break
            ci = sum(c[0] for c in comp) / len(comp)
            cj = sum(c[1] for c in comp) / len(comp)
            best = min(comp, key=lambda c: ((c[0] - ci) ** 2 + (c[1] - cj) ** 2, c))
            pos = self.map.xy(best)
            if any(math.dist(pos, tk.pos) < 2.0 for tk in live):
                continue
            self._n_explore += 1
            tk = Task(f"explore{self._n_explore}", "explore", pos, tags=frozenset({"map"}), duration_s=2.0, priority=1)
            self.add(tk, t)
            live.append(tk)

    def frontier_clusters(self, fr=None) -> list:
        """The frontier's 8-connected clusters worth exploring (at least ``MIN_FRONTIER_M`` long: a few cells a
        ray missed on a wall are not), largest first."""
        if fr is None:
            fr = self.map.frontier() & ~self.unreachable
            if self.explore_region is not None:
                fr &= self.explore_region
        seen = np.zeros_like(fr)
        clusters = []
        idx = list(zip(*np.nonzero(fr)))
        for c in idx:
            if seen[c]:
                continue
            comp, q = [], deque([c])
            seen[c] = True
            while q:
                i, j = q.popleft()
                comp.append((i, j))
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        a_, b_ = i + di, j + dj
                        if 0 <= a_ < fr.shape[0] and 0 <= b_ < fr.shape[1] and fr[a_, b_] and not seen[a_, b_]:
                            seen[a_, b_] = True
                            q.append((a_, b_))
            if len(comp) * self.map.res >= MIN_FRONTIER_M:
                clusters.append(comp)
        clusters.sort(key=lambda cm: (-len(cm), cm[0]))
        return clusters

    def debris_tasks(self, t):
        """A clear_debris task per newly seen debris pile (needs ``heavy_lift``: the grab clears it piece by piece, ≤ 30 kg
        a piece; the operator approves)."""
        ids = set(int(v) for v in np.unique(self.map.debris) if v > 0)
        for d in sorted(ids - self._known_debris):
            cells = np.argwhere(self.map.debris == d)
            c = tuple(cells[len(cells) // 2])
            tk = Task(f"clear_debris{d}", "clear_debris", self.map.xy(c), tags=frozenset({"heavy_lift"}), lift_kg=30.0,
                      duration_s=60.0 * len(cells) * self.map.res ** 2 / 0.25, priority=3, requires_approval=True,
                      reach_m=1.0, debris_id=d)
            self.add(tk, t)
            self._known_debris.add(d)

    def service_tasks(self, t, agents):
        """With a ``service`` agent in the fleet: a ``swap_battery`` task for an agent below ``LOW_BATTERY`` (it waits
        where it is), and with a ``tow`` agent a ``recover`` task for a failed one (towed to its home); once each."""
        can_service = any("service" in a.profile.tags and a.state != "failed" for a in agents)
        can_tow = any("tow" in a.profile.tags and a.state != "failed" for a in agents)
        def live(kind, aid):
            return [tk for tk in self.tasks.values() if tk.kind == kind and tk.subject == aid
                    and tk.status not in ("done", "cancelled")]

        def last(kind, aid):
            mine = [tk for tk in self.tasks.values() if tk.kind == kind and tk.subject == aid]
            return mine[-1] if mine else None

        for a in agents:
            if a.profile.flies or "service" in a.profile.tags:
                continue
            open_swap = live("swap_battery", a.id)
            for tk in open_swap:                                  # the swap goes where the robot is
                if tk.status == "open":
                    tk.pos = a.pos
            prev = last("swap_battery", a.id)
            moved = prev is None or (prev.status == "cancelled" and math.dist(prev.pos, a.pos) > 2.0) or \
                (prev.status == "done" and a.soc < LOW_BATTERY)
            if can_service and a.state not in ("failed", "recovered") and a.soc < LOW_BATTERY and not open_swap and moved:
                n = sum(1 for tk in self.tasks.values() if tk.kind == "swap_battery" and tk.subject == a.id)
                self.add(Task(f"swap_battery_{a.id}" + (f"_{n + 1}" if n else ""), "swap_battery", a.pos,
                              tags=frozenset({"service"}), lift_kg=MODULE_KG.get(a.profile.kind, 26.0),
                              duration_s=SWAP_S, priority=4, reach_m=1.3, subject=a.id), t)
            if can_tow and a.state == "failed" and last("recover", a.id) is None:
                self.add(Task(f"recover_{a.id}", "recover", a.pos, tags=frozenset({"tow"}), duration_s=2 * HOOKUP_S,
                              dest=a.home, priority=3, reach_m=1.5, subject=a.id), t)

    def unblock(self, frontiers: bool = False):
        """After the map changed every failed pair may be possible again; ``frontiers``: a route opened (debris
        cleared): the frontiers given up are tried again too."""
        self._failed_pairs.clear()
        if frontiers:
            self.unreachable[:] = False


# ----------------------------------------------------------------------------------------------- the simulation
class Fleet:
    """The fleet on a site: ``truth`` map, the agents, the coordinator, the supervisor and scheduled faults.

    ``faults``: [(t, agent, kind)] with kind ``fail`` (the agent stops for good: a drive fault) or ``battery``
    (its state of charge drops to 15 %). ``run(t_end)`` steps 1 s ticks and returns ``result()``."""

    def __init__(self, truth: GridMap, agents, comms: Comms, supervisor: Supervisor | None = None, *, faults=(),
                 explore_region=None, dt: float = 1.0):
        self.truth = truth
        self.agents = list(agents)
        self.comms = comms
        self.sup = supervisor or Supervisor()
        self.coord = Coordinator(truth, comms, explore_region=explore_region)
        self.faults = sorted(faults, key=lambda f: (f[0], f[1]))
        self.dt = dt
        self.t = 0.0
        self._rays = {}
        for a in self.agents:
            a.last_link_pos = a.pos
        self.connected = set()

    def agent(self, aid) -> Agent:
        return next(a for a in self.agents if a.id == aid)

    # ---- supervisor commands first
    def _supervise(self):
        c = self.coord
        for _, _, kind, kw in self.sup.due(self.t):
            ids = [a.id for a in self.agents] if kw.get("agent") is None and kind in ("estop", "reset") else [kw.get("agent")]
            if kind == "estop":
                for aid in ids:                                    # the safety radio: no mesh needed
                    a = self.agent(aid)
                    if math.dist(a.pos, self.comms.base) <= self.comms.safety_range_m:
                        a.estop = True
                        a.events.append((self.t, "estop"))
                c.log(self.t, "estop", agents=ids)
            elif kind == "reset":
                for aid in ids:
                    self.agent(aid).estop = False
                c.log(self.t, "estop_reset", agents=ids)
            elif kind == "hold":
                self.agent(kw["agent"]).hold = True
                c.log(self.t, "hold", agent=kw["agent"])
            elif kind == "release":
                self.agent(kw["agent"]).hold = False
                c.log(self.t, "release_hold", agent=kw["agent"])
            elif kind == "approve":
                tk = c.tasks.get(kw["task"])
                if tk is not None:
                    tk.approved = True
                    if tk.status == "awaiting_approval":
                        tk.status = "open"
                    c.log(self.t, "approved", task=tk.id)
                else:                                              # approved before it is reported
                    c.pre_approved.add(kw["task"])
                    c.log(self.t, "pre_approved", task=kw["task"])
            elif kind == "add_task":
                c.add(kw["task"], self.t)
            elif kind == "cancel":
                tk = c.tasks.get(kw["task"])
                if tk is not None and tk.status != "done":
                    if tk.agent:
                        self._drop(self.agent(tk.agent))
                    tk.status = "cancelled"
                    c.log(self.t, "cancelled", task=tk.id)
        if self.sup.auto_approve:
            for tk in c.tasks.values():
                if tk.status == "awaiting_approval":
                    tk.approved, tk.status = True, "open"
                    c.log(self.t, "approved", task=tk.id, by="auto")

    def _drop(self, a: Agent):
        a.task, a.path, a.state, a.work_left, a.phase = None, [], "idle", 0.0, ""

    # ---- faults
    def _faults(self):
        while self.faults and self.faults[0][0] <= self.t + 1e-9:
            _, aid, kind = self.faults.pop(0)
            a = self.agent(aid)
            if kind == "fail":
                a.state = "failed"
                self.coord.log(self.t, "agent_failed", agent=aid)
            elif kind == "battery":
                a.soc = min(a.soc, 0.15)
                self.coord.log(self.t, "battery_low", agent=aid)

    # ---- comms, heartbeats, recovery
    def _comms(self):
        c = self.coord
        self.connected = self.comms.connected(self.truth, self.agents)
        for a in self.agents:
            if a.state in ("failed", "recovered"):
                if a.task and self.t - a.last_link_t >= T_RELEASE and c.tasks[a.task].status not in ("done", "cancelled"):
                    c.release(self.t, c.tasks[a.task], "agent silent (failed)")
                    a.task = None
                continue
            if a.id in self.connected:
                if not a.linked:
                    c.log(self.t, "reconnected", agent=a.id, offline_s=round(self.t - a.last_link_t, 1))
                    self._reconcile(a)
                a.linked, a.last_link_t, a.last_link_pos = True, self.t, a.pos
                c.merge(a)
                for tk in c.tasks.values():                       # it dropped a task out of contact: free it
                    if tk.agent == a.id and tk.status in ("assigned", "active") and a.task != tk.id:
                        c.release(self.t, tk, "dropped by the agent")
            else:
                if a.linked:
                    c.log(self.t, "link_lost", agent=a.id)
                a.linked = False
                silent = self.t - a.last_link_t
                if a.task and silent >= T_RELEASE and c.tasks[a.task].agent == a.id \
                        and c.tasks[a.task].status not in ("done", "cancelled"):
                    c.release(self.t, c.tasks[a.task], "no heartbeat")
                if silent >= T_RETURN and a.state == "idle" and not a.estop and not a.hold:
                    if math.dist(a.pos, a.last_link_pos) > 0.5:          # first the last point it had a link,
                        goal, why = a.last_link_pos, "last link point"
                    else:                                              # then home, next to the base station
                        goal, why = a.home, "home"
                    if math.dist(a.pos, goal) > 0.5 and self._go(a, goal, "returning"):
                        c.log(self.t, "return_to_comms", agent=a.id, to=why)

    def _reconcile(self, a: Agent):
        """On reconnection: report work done offline (and cancel duplicates the coordinator handed out), and drop a
        task the coordinator released meanwhile."""
        c = self.coord
        for tid in a.done_offline:
            tk = c.tasks[tid]
            if tk.status != "done":
                if tk.agent and tk.agent != a.id:
                    other = self.agent(tk.agent)
                    if other.task == tid:
                        self._drop(other)
                        c.log(self.t, "duplicate_cancelled", task=tid, agent=other.id)
                tk.status, tk.done_t, tk.agent = "done", self.t, a.id
                c.log(self.t, "done_reported", task=tid, agent=a.id)
        a.done_offline = []
        if a.task and (c.tasks[a.task].agent != a.id or c.tasks[a.task].status in ("done", "cancelled")):
            c.log(self.t, "task_superseded", agent=a.id, task=a.task)
            self._drop(a)
        if a.state == "returning":
            a.state, a.path = "idle", []

    def _go(self, a: Agent, goal, state: str):
        """Head for ``goal`` on the agent's own map (all it has out of contact)."""
        p = a.known.path(a.pos, goal, a.profile.width_m, flies=a.profile.flies, height=a.profile.height_m)
        if p:
            a.path, a.state = list(p[1:]), state
        return bool(p)

    # ---- motion, sensing, work
    def _sense(self, a: Agent):
        r = a.profile.sensor_range_m
        if r not in self._rays:
            self._rays[r] = _rays(max(1, int(r / self.truth.res)))
        if a.profile.flies:                                       # a camera footprint from above: outdoor cells
            c = np.array(self.truth.cell(a.pos))
            off = self._rays[r].reshape(-1, 2)
            cells = c + off
            nx, ny = self.truth.shape
            ok = (cells[:, 0] >= 0) & (cells[:, 0] < nx) & (cells[:, 1] >= 0) & (cells[:, 1] < ny)
            cells = cells[ok]
            out = ~self.truth.indoor[cells[:, 0], cells[:, 1]]
            cells = cells[out]
            a.known.occ[cells[:, 0], cells[:, 1]] = self.truth.occ[cells[:, 0], cells[:, 1]]
            a.known.debris[cells[:, 0], cells[:, 1]] = self.truth.debris[cells[:, 0], cells[:, 1]]
            a.known.stamp[cells[:, 0], cells[:, 1]] = self.t
            a.known.invalidate()
        else:
            sense(self.truth, a.known, a.pos, r, self._rays[r], self.t)

    def _towing(self, a: Agent):
        tk = self.coord.tasks.get(a.task) if a.task else None
        if tk is not None and tk.kind == "recover" and a.phase == "deliver" and tk.subject:
            return self.agent(tk.subject)
        return None

    def _move(self, a: Agent):
        step = a.profile.v_m_s * self.dt * (0.5 if self._towing(a) else 1.0)     # half speed with a robot in tow
        moved = 0.0
        ok = None if a.profile.flies else a.known.passable(a.profile.width_m, height=a.profile.height_m)
        here = self.truth.cell(a.pos)
        while a.path and step > 1e-9:
            nxt = a.path[0]
            c = self.truth.cell(nxt)
            if not a.profile.flies and self.truth.occ[c] == OCC:     # the plan ran into something: replan
                a.known.occ[c] = OCC
                a.known.debris[c] = self.truth.debris[c]
                a.known.stamp[c] = self.t
                a.known.invalidate()
                return "blocked"
            if ok is not None and not ok[c] and c != here and ok[here]:
                return "blocked"                                      # what it saw since planning: it does not fit
            d = math.dist(a.pos, nxt)
            if d <= step:
                a.pos = nxt
                a.path.pop(0)
                step -= d
                moved += d
            else:
                a.pos = (a.pos[0] + (nxt[0] - a.pos[0]) * step / d, a.pos[1] + (nxt[1] - a.pos[1]) * step / d)
                moved += step
                step = 0.0
        a.odometer_m += moved
        towed = self._towing(a)
        if towed is not None:
            towed.pos = a.pos                                      # on the hitch behind
        return "arrived" if not a.path else "moving"

    def _drain(self, a: Agent, working: bool):
        if a.state in ("failed", "recovered"):
            return
        rate = 1.0 / (a.profile.endurance_h * 3600.0)             # the profile's average mission power
        a.soc = max(0.0, a.soc - rate * self.dt * (1.0 if working or a.state == "moving" else 0.3))

    def _step_agent(self, a: Agent):
        c = self.coord
        if a.state in ("failed", "recovered"):
            return
        if a.estop:
            self._drain(a, False)
            return
        if a.hold:
            self._drain(a, False)
            return
        self._sense(a)
        tk = c.tasks.get(a.task) if a.task else None
        if a.state in ("moving", "returning"):
            r = self._move(a)
            if r == "blocked":
                if tk is not None:
                    path, _ = c.plan(a, tk) if a.linked else (a.known.path(a.pos, tk.pos, a.profile.width_m,
                                                                           goal_tolerance=tk.reach_m, height=a.profile.height_m), 0.0)
                    if path is None:
                        c.log(self.t, "blocked", agent=a.id, task=tk.id)
                        if a.linked:
                            c.release(self.t, tk, "path blocked")
                            c._failed_pairs.add((a.id, tk.id))
                        self._drop(a)
                    else:
                        a.path = list(path[1:])
                else:
                    a.path = []
            elif r == "arrived":
                if a.state == "returning":
                    a.state = "idle"
                elif tk is not None:
                    if tk.dest is not None and a.phase == "pickup":
                        a.state, a.work_left = "working", tk.duration_s / 2
                    else:
                        a.state, a.work_left = "working", tk.duration_s if tk.dest is None else tk.duration_s / 2
                    if tk.status == "assigned":
                        tk.status = "active"
            self._drain(a, False)
            return
        if a.state == "working" and tk is not None:
            a.work_left -= self.dt
            self._drain(a, True)
            if a.work_left <= 1e-9:
                if tk.dest is not None and a.phase == "pickup":
                    a.phase = "deliver"
                    p = (c.map if a.linked else a.known).path(a.pos, tk.dest, a.profile.width_m, flies=a.profile.flies, height=a.profile.height_m,
                                                             goal_tolerance=tk.reach_m)
                    if p is None:
                        c.log(self.t, "blocked", agent=a.id, task=tk.id)
                        self._drop(a)
                        return
                    a.path, a.state = list(p[1:]), "moving"
                    return
                self._finish(a, tk)
            return
        swap = any(tk.kind == "swap_battery" and tk.subject == a.id and tk.status not in ("done", "cancelled")
                   for tk in c.tasks.values())
        if a.state == "idle" and swap:
            self._drain(a, False)                                  # wait where it is for the servicing rover
            return
        if a.state == "idle" and a.soc < LOW_BATTERY and math.dist(a.pos, a.home) > 0.5:
            self._go(a, a.home, "returning")
            c.log(self.t, "going_home", agent=a.id, soc=round(a.soc, 2))
        self._drain(a, False)

    def _finish(self, a: Agent, tk: Task):
        c = self.coord
        if tk.kind == "swap_battery" and tk.subject:
            sub = self.agent(tk.subject)
            sub.soc = 1.0
            c.log(self.t, "battery_swapped", agent=sub.id, by=a.id)
        if tk.kind == "recover" and tk.subject:
            sub = self.agent(tk.subject)
            sub.pos = a.pos
            sub.state = "recovered"
            c.log(self.t, "recovered", agent=sub.id, by=a.id, at=tuple(round(v, 1) for v in a.pos))
        if tk.kind == "clear_debris" and tk.debris_id:
            hit = self.truth.debris == tk.debris_id
            self.truth.occ[hit] = FREE
            self.truth.debris[hit] = 0
            self.truth.invalidate()
            for k in (a.known, c.map):
                k.occ[hit] = FREE
                k.debris[hit] = 0
                k.stamp[hit] = self.t
                k.invalidate()
            c.unblock(frontiers=True)
        if a.linked:
            tk.status, tk.done_t = "done", self.t
            c.log(self.t, "done", task=tk.id, agent=a.id)
        else:
            a.done_offline.append(tk.id)
            c.log(self.t, "done_offline", task=tk.id, agent=a.id)
        a.task, a.state, a.path, a.phase = None, "idle", [], ""

    # ---- one tick
    def step(self):
        self._supervise()
        self._faults()
        self._comms()
        c = self.coord
        if int(self.t) % 30 == 0:
            c.unblock()                                         # the map has grown: retry what had no path
        c.debris_tasks(self.t)
        c.service_tasks(self.t, self.agents)
        c.explore_tasks(self.t, self.agents)
        c.allocate(self.t, self.agents, self.connected)
        for a in sorted(self.agents, key=lambda x: x.id):
            self._step_agent(a)
        self.t += self.dt

    def finished(self) -> bool:
        """Every task done or cancelled and no frontier left to explore (in the region)."""
        c = self.coord
        if any(tk.status not in ("done", "cancelled") for tk in c.tasks.values()):
            return False
        return not c.frontier_clusters()

    def run(self, t_end: float = 3600.0, stop_when_done: bool = True) -> dict:
        while self.t < t_end:
            self.step()
            if stop_when_done and self.t > 5 and self.finished():
                break
        return self.result()

    def result(self) -> dict:
        c = self.coord
        known = c.map.occ != UNKNOWN
        region = c.explore_region if c.explore_region is not None else np.ones_like(known)
        return {"t_end": self.t,
                "tasks": {tk.id: {"kind": tk.kind, "status": tk.status, "agent": tk.agent, "done_t": tk.done_t,
                                  "history": list(tk.history)} for tk in c.tasks.values()},
                "coverage": float((known & region).sum() / max(region.sum(), 1)),
                "agents": {a.id: {"kind": a.profile.kind, "state": a.state, "pos": a.pos, "soc": a.soc,
                                  "odometer_m": a.odometer_m} for a in self.agents},
                "events": list(c.events)}
