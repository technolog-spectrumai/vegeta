# Science of the races — tuning the race autopilot

Moved from Vegeta's `docs/science.md` §9 when the racing quad hatched (2026-10-20); unchanged text.

## 9. Tuning the race autopilot: CMA-ES (default) or PSO + pattern search

`./tune_autopilot.sh` → `races/tune.py` (the driver) → `races/optimizers.py` (the strategies) → `races/laps.py`
(the objective).

- **Structure (strategy pattern).** The driver knows no algorithm. It builds a strategy, calls `step()` until the
  strategy is done, times each phase, and checkpoints `state()` after every step.
  - `--cma` (the default) and `--pso` select the class. Each strategy's options are the fields of its `Config`
    dataclass, and the driver turns them into command-line options.
  - A new method is one class with `step`, `best`, `state` and `restore`, added to `STRATEGIES`.
  - The fixed grids of the first rounds (`--quick`, `--round2`) and the `--method` flag were removed in this
    refactor. The last commit that has them is `67a682e` (branch `dev_cma`).

- **Problem.** Minimise a lap score f(x) over the controller's knobs x:

  | knob | meaning |
  |---|---|
  | `safety` | share of the corner speed flown |
  | `lookahead_s` | carrot distance in time |
  | `omega_pos` | position-loop bandwidth |
  | `omega_att` | attitude-loop bandwidth |
  | `tilt_max_deg` | tilt limit |
  | `v_cap` | speed cap |

  Each knob is scaled to [0, 1] between physical bounds. f is a black box: MuJoCo laps, deterministic, with no
  gradient.
- **Objective: robust over training scenarios** (`--train-k`, default 6). Every candidate is flown on the nominal lap
  and on K training scenarios, one per perturbation family of the validation (settings ±2 %, start, mass, thrust,
  wind, combined; `races.validate.training`).
  - The scenarios use seed 1, never the held-out seed of `./test_autopilot.sh`.
  - Every candidate flies the same scenarios (common random numbers), so a comparison compares settings, not luck.
  - f is the mean of the worst ⌈(K+1)/3⌉ lap scores, a tail average (CVaR-like).
  - It was introduced because the single-lap objective overfitted: on held-out scenarios three courses scored clean
    rates of 0.29, 0.02 and 0.29 (see the validation below).
  - `--train-k 0` restores the single nominal lap.
  - One evaluation (the budget unit, `--max-evals`) is one candidate, which is K+1 laps.
- **Score of one lap:**
  - a finished lap scores lap time + 2 s per collision;
  - a DNF scores 1000 − 20 × zones passed, so more of the course is better before any lap finishes;
  - a crash of the simulation scores 2000.
- **CMA-ES (`--cma`, the default).** Hansen & Ostermeier (2001, *Evol. Comput.* 9(2)); Hansen (2016), "The CMA
  evolution strategy: a tutorial", arXiv:1604.00772. It uses the PyPI package `cma`, installed by `install_local.sh`.
  - Each generation samples λ candidates (default 4 + ⌊3 ln 6⌋ = 9) from N(m, σ²C). The mean moves to the
    weighted best μ, and σ and the covariance C adapt to the ranking.
  - Only ranks matter, so the jumps of DNF scores do no harm. The population averages over the rugged robust
    score, which is why it suits this problem better than a fine local polish.
  - The box is handled by `cma`'s boundary transformation.
  - The start (m₀) is the stored settings, with σ₀ = 0.2 of each range.
  - It stops on `cma`'s own criteria:
    - `--tolfun` 0.1: recent scores span less than this;
    - `--tolx` 0.001: steps below this share of each range;
    - stagnation;
    - `--generations` (40);
    - `--max-evals` (300).
  - The stop is reported as "a local result, not a proven global optimum".
  - Reproducible: each generation's normal draws come from a generator seeded by (`--seed`, generation). `cma` would
    otherwise use NumPy's global generator, which a saved state does not carry. The state is `cma`'s object, pickled
    into the checkpoint, and its version is part of the fingerprint.
  - Checked: a run interrupted after its first generation and resumed flew the same 36 laps as an uninterrupted one.
  - **Racing** (`--race`, on by default; after Maron & Moore 1997 and Birattari et al.'s F-race, 2002). It uses the
    robust score's own bound.
    1. Every candidate flies the first ⌈(K+1)/3⌉ scenarios, the nominal lap first. The worst m laps of a superset are
       never better than those of a subset, so the mean of that part is a lower bound of the full score.
    2. In the order of these bounds, the remaining scenarios are flown in parallel batches, while T, the μ-th best
       full score, is tracked.
    3. A candidate whose bound is ≥ T cannot reach the best μ, and neither can any after it, so their remaining laps
       are not flown.

    The update then uses only the best μ: the plain (μ/μ_w, λ) form, because `cma`'s "active" covariance update
    would also weight the worst candidates. Those μ have exact scores. Every other candidate reports max(its bound,
    the μ-th score) + 10⁻⁶, which does not depend on the batches. So racing changes the laps flown, never a decision:
    `-j` and racing give the same run.
    - Checked on a stand-in objective: the same CMA-ES mean in every generation with racing as without, and fewer
      laps.
    - On MuJoCo (industrial, K = 6, λ = 9): 16–20 of 63 laps saved per generation.
    - The stop tests (`tolfun`, …) see the reported values, bounds included.
    - `--no-race` flies everything and keeps the active update.
  - CMA-ES samples around its start without flying it, so the driver flies the start once at the end. If nothing beat
    it, the result is the start: "no improvement found over the starting settings".
- **PSO, then pattern search (`--pso`).** This is the earlier search, kept intact; the details follow.
- **Pattern-search polish.** Generalized pattern search (GPS: Torczon 1997, *SIAM J. Optim.* 7(1); Audet & Dennis 2003, *SIAM J.
  Optim.* 13(3)). At each iteration:
  1. Poll the 2n points x ± Δ·eᵢ (a positive spanning set of directions) and take the first that lowers f.
  2. On success, move there and expand the mesh: Δ ← min(2Δ, 0.25).
  3. On failure, x is a mesh local optimum: refine, Δ ← Δ/2.
- **Convergence criterion:** stop when Δ < Δ_min (default 1/64 of every knob's range).
  - Torczon's theorem: for a continuously differentiable f, the mesh local optima of the refining subsequence
    converge to a stationary point.
  - For a Lipschitz f (Audet & Dennis), the limit is Clarke-stationary in the polled directions.
  - Our f is **piecewise**: a lap that turns into a DNF jumps. The honest guarantee is therefore a **mesh local
    optimum at resolution Δ_min**. It is not a global optimum, and a different start can find a different one.
  - `--max-evals` is a safety budget only, and is reported if it is what stopped the search. `--patience` is an
    optional heuristic early stop, off by default.
- **Local minima: a global stage first.** Pattern search only refines from its start. With `--pso`, a
  **particle swarm** searches the whole box first.
  - It uses Kennedy & Eberhart's PSO (1995) in Clerc & Kennedy's constriction form (2002, *IEEE Trans. Evol. Comput.*
    6(1)): v ← χ[v + c₁r₁(p − x) + c₂r₂(g − x)], with χ = 0.7298 and c₁ = c₂ = 1.49618. These are the parameters for
    which the particles' trajectories provably converge.
  - Velocity is limited to 0.2 per step.
  - There are 12 particles: one at the stored settings, the rest spread over the box by a Latin hypercube with a
    fixed seed, so a run can be repeated.
  - The swarm stops after 4 iterations without improvement, once the swarm's radius is below 0.05, or at its share of
    the evaluation budget. `--pso-iterations` (default 15) is a **hard maximum**: reaching it always ends the swarm, even if
    it is still improving (the log says so), and the polish starts from the best particle found.
  - Pattern search then polishes the swarm's best to Δ_min. Its evaluations are reserved: `--polish-evals` (default 120)
    of `--max-evals` are kept for it whatever the swarm does.
  - PSO gives no proof of global optimality: no method does with a budget of laps on a black box. What it gives is a
    sample of the whole space, so a lone basin is unlikely to be missed. `--pso-iterations 0` skips the swarm and
    polishes the stored settings; `--polish-evals 0` is the swarm alone.
- **Parallel and reproducible.** `-j N` flies a batch (a swarm generation, a complete poll) on N worker processes.
  The batch is fixed before it flies and its results are read in a fixed order (the poll stays opportunistic in the
  axis order), so `-j 1` and `-j N` give identical laps. Checked: 33 laps identical, 1.9 min against 0.7 min on 4
  cores. The swarm's random numbers come from a seeded generator (`--seed`), whose state is saved.
- **Cutting slow laps, only in the PSO polish and only with `--train-k 0`.** Under the robust objective a lap's score
  enters a tail average, where a cut could change a comparison, so nothing is cut. A polishing lap stops at 1.25 × the best finished lap known when its batch
  starts + 5 s. A cut lap scores as a DNF, and the polish only compares with that incumbent, which it could not have
  beaten anyway, so the decisions are unchanged. The swarm never cuts: each particle compares with its own best, which
  a cut could change.
- **Checkpoints and resume.** After every swarm and polishing iteration the whole state is written atomically
  (temporary file, fsync, rename) to `races/runs/tune/<course>.ckpt.json`. That covers:
  - the phase, the swarm's positions, velocities and personal and global bests, the generator's state and counters;
  - the polish's point, Δ and counters;
  - every lap flown, and the time spent.

  A rerun with the same options resumes exactly where it stopped. Checked: interrupted once during the swarm and once
  during the polish, the resumed runs flew the same laps as an uninterrupted one. A finished course is skipped; other
  options are refused (`--fresh` starts over).
- **Results and wording.** After every course, `best.json`, `races/autopilot.json` and the CSV of every lap are written,
  with the time spent: swarm, polish, total, and wall time per lap. A search that ends no better than its start says
  "no improvement found over the starting settings". It is never reported as proof of an optimum.
- **Bookkeeping.** MuJoCo is deterministic, so every setting is flown once per scenario and remembered. The search
  starts from the course's stored settings (`races/autopilot.json`). An entry records its objective:
  - under the same objective, only a better score replaces it;
  - a result of another objective replaces it, because the scores do not compare.
- **Results with the earlier single-lap objective and PSO** (2026-10-20, all four courses, `-j 12`, about 12 min in
  total). Every course
  converged at Δ = 1/128 < Δ_min = 1/64: a mesh local optimum per course, not a proven global optimum.

  | course | before | after | collisions | laps | time |
  |---|---|---|---|---|---|
  | hangar | DNF (13 of 18 zones) | 60.1 s | 1 (+5 s) | 215 | 5.9 min |
  | industrial | 32.0 s | 29.4 s | 0 | 153 | 1.7 min |
  | farm | 36.7 s | 31.4 s | 0 | 150 | 1.7 min |
  | quarry | 49.7 s | 47.4 s | 0 | 205 | 3.1 min |

  The hangar's first finished lap is a slow, careful setting (safety 0.31, tilt limited to 50°, speed capped at
  9.8 m/s) that none of the earlier grids had tried.
  A second search of the hangar with another seed (`--seed 1 --fresh`: a new swarm, starting from these settings) found
  no improvement and converged back to the same point in 107 laps (1.6 min). This checks robustness; it does not prove
  a global optimum.
- **The hangar's knife edge, and what fixed it** (2026-10-20). That 60.1 s point failed under small changes:
  ±0.01 of safety, or a little more omega_pos, gave a DNF. Flying the laps showed two faults, neither a tuning problem:
  - **The drop through the stair opening** (zones 12–13). The quad arrived over the opening at about twice the
    schedule's speed, and the reference jumped below it. The controller (section 5) then asked for almost no thrust,
    so all four rotors sat at idle. At idle the mixer has no differential thrust left, so it makes no roll or pitch
    torque: the geometric controller of Lee et al. assumes rotors that are not saturated. The attitude drifted to 74°
    and the quad fell into the gable and stair corner.
    The fix is a drop guard in the autopilot (`races.path.Drops`), a fixed rule rather than a tuning knob. It applies
    where the line turns into a descent steeper than 45°:
    1. stop on the top point;
    2. wait over it until within 0.5 m and below 0.8 m/s horizontally;
    3. descend at no more than 2 m/s, with the reference at most 1 m below.
    The mock's point-mass flyer has no attitude to lose and is unchanged.
  - **The window** (zones 15–16). The tuned lap's one collision was the wall's pier beside the window, not the stair.
    The gate before the window sat diagonal to the 1.6 m opening, so the quad cut the corner. It now sits square in
    front of the window, which makes the course 165 m long.
  - **The tuned settings now:** 43.8 s with 0 collisions. In 12 laps at changes of up to 0.3 in safety, 1 in
    omega_pos and 10° of tilt, 10 laps finish clean.
    The 2 failures are at safety 0.5 and 0.6, on the slab edge of the east-hole climb (zones 8–9). This is the known
    weak spot at higher speeds.
    The other three courses have no drop and fly the same laps as before.
  - **A re-tune from there** (`-j 4 --fresh`: 245 laps, 10.4 min) converged to a mesh local optimum at 24.0 s with
    0 collisions, but that point is a knife edge again. With ±1 % on each knob only 3 of 12 laps finish, and the rest
    fail all over the course.
    `races/autopilot.json` keeps the robust 43.8 s settings. The score has no term for robustness: one lap per setting
    rewards the edge of the feasible region. Scoring the worst of a few perturbed laps would fix that, at a few times
    the cost.

- **Validation on held-out scenarios** (`./test_autopilot.sh`, `races/validate.py`). This is validation in the
  machine-learning sense. The tuner fits each course's settings on one condition, the nominal lap. The validation flies
  those settings on conditions the tuner never saw, drawn from a fixed seed (20261020). The tuner never perturbs the
  world, and its swarm seeds are 0 and 1.
  - **The scenario families:** the settings ±2 % (the tolerance of the tuned point), the start moved by ±0.5 m, the
    mass and inertia ±5 % (the controller keeps believing the nominal quad), the thrust ±5 %, a steady wind up to
    2 m/s, and all of these at once. Each family has its own random stream.
  - **The nominal lap** is the training score. It must reproduce `autopilot.json`.
  - **Metrics per family:**
    - the clean rate (finished with no collision);
    - the finish rate;
    - the median and p90 lap;
    - the gap: the median finished lap minus the nominal lap.
  - **Grades on the clean rate:** pass at 0.8 or more, weak at 0.5 or more, otherwise fail. This is a convention, not a
    requirement.
  - **The report** is `races/runs/validate/report.json` (one line per family), with every lap in
    `races/runs/validate/<course>.csv`. The exit code is 0 only when every course passes.
  - **First result** (2026-10-20, 8 laps per family, 49 laps per course, 8.1 min with `-j 4`):

    | course | held-out clean rate | verdict | weakest family |
    |---|---|---|---|
    | hangar | 1.00 | pass (every family, gaps ≤ 0.04 s) | none |
    | industrial | 0.29 | fail | combined (0.00); wind 0.12 |
    | farm | 0.02 | fail | everything ≤ 0.12, gaps of 19–42 s |
    | quarry | 0.29 | fail | thrust ±5 % (0.00); wind 0.12 |

    This is overfitting. Three courses were tuned to the edge of their nominal lap: industrial picks up a collision in
    a 0.1 m/s tailwind, and the farm loses its clean lap when its settings move by 2 %. The hangar's settings come
    from the slow 60 s search re-flown with the drop guard, so they sit well inside the region where laps finish, and
    they generalise.
    The cure is a robust tuning objective: score each setting by its result over a few held-out-style perturbations
    (a training set of scenarios), and keep this seed for validation only.
    The farm's nominal lap here (31.74 s) does not reproduce its stored 31.37 s, which the report flags as
    `"reproduced": false`.

- **Training history** (`./viz_autopilot_train_hist.sh [--animate]`; `races/trainviz.py`). It reads a course's
  checkpoint, or its CSV, and writes to `races/runs/tune/train_hist/`.
  - The laps are grouped into candidates per generation and scored as the tuner scored them. Candidates that racing
    dropped are drawn hollow: their score is a lower bound.
  - **Loss history:** every candidate, the best and median of each step, and the best so far, on a log scale (a DNF
    scores near 1000, a lap 30–60).
  - **The search plane:** six settings cannot be drawn as one cloud of dots, and a plot per pair of settings hides
    how they move together. So the candidates' normalised settings go through a principal component analysis (SVD)
    over the whole run, and are drawn on PC1 × PC2 with their share of the variance.
    - CMA-ES's sampling distribution is drawn on the same plane: the 2σ ellipse of N(m, σ²C) projected (Wᵀσ²CW),
      with the mean's path, as in the classic CMA-ES pictures.
    - The checkpoint now records each generation's mean, σ and C.
    - A loadings panel says which settings make each axis. The plane is a projection: points close on it may differ
      in the four directions it omits.
  - **The settings per step:** each setting in its own units, with the best so far and the CMA-ES mean (mapped into
    the box), for reading off the values.
  - `--animate`: a GIF of the plane, generation by generation.

- **Videos of the tuned autopilot** (`./gen_pilot_videos.sh`; `races.sim.render`, `races.video`). Each course is
  flown in MuJoCo with its settings from `races/autopilot.json` and rendered from the scene's cameras (default
  `chase`, `onboard`, `overview`).
  - The quad is replayed with its **recorded attitude**: the quaternion in the history. Earlier renders showed a level
    quad facing along its velocity, which hides the banking.
  - Frames are picked **by time**, so `--quality lap` plays the whole lap in real time at 30 fps and 960×540
    (`--speed` 2 is twice as fast).
  - A HUD shows the course, the camera, the label, time, speed, altitude, zones and collisions.
  - `--held-out` also flies the worst lap of every family the latest `./test_autopilot.sh` did not pass. The scenario
    is rebuilt from `races.validate` and its seed, and the index records whether the lap reproduces the validation run
    (checked on industrial: 6 of 6 do). So a failure can be watched, not just counted.
  - Without an OpenGL backend that really renders, the matplotlib replay is written instead, labelled. The backend is
    probed by rendering one frame in a subprocess: EGL, then OSMesa, then GLFW.
  - Output: `races/runs/videos/<course>/<stamp>_mujoco/` (the run export with provenance, one mp4 per camera) and
    `races/runs/videos/index.json`.
  - **Cameras:**
    - `fpv` is the quad's on-board camera, tilted 30° up and moving with the recorded attitude.
    - `chase` is a third-person camera 2.2 m behind the quad along its heading. The heading is smoothed with a 0.6 s
      lag. The camera is pulled in front of any colliding geometry by a MuJoCo ray cast from the quad.
    - `overview` looks down on the whole course.
    - The files are named `<course>_fpv.mp4`, `<course>_chase.mp4` and `<course>_overview.mp4`.
  - **Logo and side panel.** The Vegeta logo is in the corner. To the right (`races.video_panel`, `--no-panel`) is
    the autopilot panel:
    - the six settings flown as bars, each normalised to its search range, with the value above and the name below;
    - the mode now: racing, hover, a missed gate, or the drop guard's approach / align / descend;
    - 10 s strip charts of commanded against measured speed, tilt (with its limit) and thrust (in weights, with the
      drive limit, the limited commands shaded), and the tracking error with the altitude.

    `run_mujoco` records these signals in `result["pilot"]` (`races.sim.PILOT_COLUMNS`), and the run export writes
    them as `autopilot.csv`. The flight itself is unchanged: the same lap times.
  - **Fixed on the way:**
    - MuJoCo's default offscreen framebuffer (640×480) is enlarged to the frame size. Before, every render above that
      size failed, including the notebooks' `fine` preset.
    - MuJoCo scales the clip planes by the model's extent, and the scene's `znear` 0.05 made a near plane 3 m (hangar)
      to 10 m (industrial) away. Everything closer was cut: the quad in the chase view and the ground below the camera
      (the black lower half). Renders now use a 2 cm near plane.
