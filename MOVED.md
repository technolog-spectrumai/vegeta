# Moved out of Vegeta

Vegeta conceives and designs machines; a machine whose design is frozen hatches and develops in Piccolo
(ARCHITECTURE.md). These paths left Vegeta. **Their history stays in this repository's git** (`git log -- <old path>`)
and was carried into the new home with `git filter-repo` (`git log --full-history` there reaches back).

## 2026-10-20 — Chiron becomes its own repository

| Vegeta path | Now |
|---|---|
| `vegeta-cli/src/vegeta/chiron`, `vegeta-cli/tests/chiron`, `docs/chiron.md` | [technolog-spectrumai/chiron](https://github.com/technolog-spectrumai/chiron) (`import chiron`) — retired 2026-10-21: `vegeta.chiron` is the one Chiron. |

## 2026-10-21 — The races come home

User decision: Piccolo is ROS-centric full simulation (Gazebo/ROS); the MuJoCo races stay in Vegeta. `races/`, its
golden test and `docs/science_races.md` came back with their Piccolo history (`git filter-repo` + merge). They fly
`ogma.rotors` again (the local copy `races/rotors.py` is gone) and `designs/data/mini_raptor_drive_6S_5x4p3.json`
(the frozen copy is gone). The table below is history.

## 2026-10-20 — The racing quad hatches (Piccolo `dev_fusion_dance`) — reverted 2026-10-21

| Vegeta path | Piccolo path |
|---|---|
| `races/**` | `races/**` |
| `tune_autopilot.{py,sh}`, `test_autopilot.{py,sh}`, `gen_pilot_videos.{py,sh}`, `viz_autopilot_train_hist.{py,sh}` | `races/scripts/` |
| `notebooks/races/01_hangar … 04_quarry.ipynb`, `race_design_factors.ipynb` | `races/notebooks/` |
| `side_labs/races/race.py` (side-lab book R) | `races/lab/race.py` (its own run directories) |
| `docs/science.md` §9 | `docs/science_races.md` |
| `designs/drongo_controller.py` — **copied**, Drongo has not hatched | `races/flight.py` (Ref, Flight, quat_to_R) |
| `designs/data/mini_raptor_drive_6S_5x4p3.json`, `notebooks/logo.png` — **copied** | `races/data/`, `races/assets/` (`races/data/PROVENANCE.md`) |

The race laps' design factors come back as data: Piccolo writes `races/data/design_factors/`, hand it over to
`designs/data/races/`. The Merlin race (`notebooks/races/26b_*`, `27_*`) and the mini raptors (`designs/exp/`) are
designs and stay here.

## 2026-10-20 — Standard-part physics moves to Ogma

| Vegeta path | Ogma (`import ogma as og`) |
|---|---|
| `designs/gears.py` | `og.gears` |
| `designs/terramechanics.py` | `og.soil` |
| `designs/tracks.py` | `og.tracks` |
| `designs/cutting.py` | `og.cutting` |
| `designs/actuators.py` | `og.actuators` |

`import actuators as act` is now `from ogma import actuators as act` (likewise `from ogma import soil as tm`). Ogma's
tests prove the moved modules return exactly what these files returned. Ogma lives in Vegeta at `ogma/` (it came home from its own repo on 2026-10-21).
