# Kestrel-4Q-1 / 4Q-2 vs X-wing X-A / X-B

Comparison drawn from notebook 41 (`41_xwing`, exported 2026-10-17). X-A has the
biplane's wing area (0.28 m²), X-B twice that (0.56 m²). Kestrel figures are the
ones notebook 41 carries over; their per-case transition tables are not in it.

## Summary

The X-wing is not better than Kestrel-4Q-2: 4Q-2 wins on drag, efficiency, top
speed and pitch/roll control. Against 4Q-1 the picture is mixed: X-A has less
drag and needs less power at 16 m/s, and both X variants roll far better (4Q-1
has no elevons), but X-A stalls higher and is less efficient. Only X-B improves
stall speed and turning, and only because it carries twice the wing area.

| | 4Q-1 | 4Q-2 | X-A | X-B |
|---|---|---|---|---|
| Stall speed [m/s] | 10.9 | 10.8 | **12.6** ✗ | **9.7** ✓ |
| Margin to stall at cruise (section) | 2.79 | 3.02 | 2.12 | 3.58 |
| Top speed [m/s] | 30.1 | **33.5** | 32.5 | 30.5 |
| Best L/D | 7.3 | **12.7** | 6.7 | 7.2 |
| Power at 16 m/s [W] | 111 | **62** | 90 | 110 |
| Best endurance [min] | 40 | **57** | 29 | 33 |
| Drag at 16 m/s [N] | 3.34 | **1.43** | 2.40 | 3.28 |
| CdA [cm²] | 186 | **72** | 96 | 170 |
| Induced drag at 16 m/s [N] | 0.41 | 0.40 | **0.89** ✗ | 0.62 |
| Pitch acceleration at 16 m/s [rad/s²] | 14.8 | **49.3** | 16.1 | 15.2 |
| Steady roll rate [deg/s] | 24 | **401** | 352 | 190 |
| Turn radius [m] | 13.7 | 13.4 | 20.6 ✗ | **10.3** |
| Transitions passed (standard entry) | 11/12 | 8/12 | **12/12** | 10/12 |

## Stall

- X-A is worse than both Kestrels: its panels stall at 12 m/s, so it cannot
  cruise there.
- X-B has the lowest stall speed of the four, bought with twice the wing area.
- Slow back-transition entries are the X-wing's weak spot (see Transitions).

### Why a lower stall speed is better

Stall speed is the slowest the aircraft can fly on its wings alone; below it the
rotors must carry the weight.

1. **Endurance.** The least-power speed sits just above stall, so a lower stall
   speed allows slower, cheaper cruise. X-A cannot use 12 m/s, which pushes its
   best endurance point up to 15 m/s.
2. **Back-transitions.** Decelerating to hover passes through stall; the higher
   it is, the earlier the wing quits and the more height is lost before the
   rotors catch the aircraft. X-A loses 4.3 m and 6.1 m entering at 13 and
   11 m/s.
3. **Gust and manoeuvre margin.** More margin above stall means room for gusts,
   turns and pull-ups. X-A has the least (2.1 vs 2.8–3.6), hence its load
   factor of 1.6 and 20.6 m turn radius.
4. **Wind tolerance.** A slow-flying wing can hold position into a headwind
   without falling back to power-hungry hover.

The catch: lowering stall speed by adding wing area costs drag at speed. X-B's
doubled area brings stall to 9.7 m/s but raises drag at 16 m/s to 3.28 N
(X-A: 2.40 N) and lowers top speed. 4Q-2 shows the better route: its cambered
NACA 23112 section has cl max 1.15 (vs 1.06 for the 0012) and trims with almost
no elevon (−0.36°), keeping stall at 10.8 m/s on the small wing without the
drag penalty.

## Efficiency and drag

- At 45° the X panels lose lift slope: CLα is 1.88 vs 4.08 for the biplane
  (about 46 %), and the projected span shrinks to 0.785 m (X-A) vs 1.0 m.
- That doubles the induced-drag factor (k 0.145 vs 0.060). X-A has to fly at
  10.8° angle of attack at 16 m/s and needs −9.3° of elevon to trim (cl lost to
  the elevon −0.53). So its best L/D is below 4Q-1's despite lower parasite drag.
- X-B's extra area adds the parasite drag back, leaving it roughly level with
  4Q-1.
- Endurance is worse for both X variants: 29 and 33 min against 40 and 57 min.

## Manoeuvrability

- Roll: both X variants are far better than 4Q-1 (no elevons) but below 4Q-2.
- Pitch: about the same as 4Q-1, roughly a third of 4Q-2's.
- Turning: X-A is poor (load factor 1.6); X-B is the tightest of the four.
- Hover: both X variants have less pitch and heading authority than either
  Kestrel (pitch 67 / 51 rad/s² vs 95 / 88).

## Transitions

On the standard test X-A is the most robust (12/12), but entry speed on the
back-transition changes the result.

### Forward (vertical to horizontal)

| | X-A | X-B |
|---|---|---|
| Passed | 6/6 | 6/6 |
| Worst height loss | 1.33 m | 1.66 m |
| Rotor thrust at fast (1.5 s) transition | 100 % (saturated) | 100 % (saturated) |
| Least pitch margin | 0.81 | **0.45** (tight) |
| Peak elevon | 4.8° | 13.7° |

Both get through, but at the 1.5 s transition the rotors are at full thrust
with no reserve. X-B is the more strained: its bigger wing needs much more
elevon and leaves half the pitch margin.

### Back (horizontal to vertical)

| Entry speed | X-A | X-B | 4Q-2 |
|---|---|---|---|
| 16 m/s (standard) | 6/6, loses up to 1.7 m | 4/6, fails roll hold at 1.5 s | fails some* |
| 13 m/s | **fails**, loses 4.3 m | passes, loses 1.2 m | **6/6** |
| 11 m/s | **fails**, loses 6.1 m | **fails**, loses 3.9 m | — |

\* Notebook 41 records 4Q-2 as "8 of 12 (6 of 6 back with the 13 m/s entry)",
so its standard-entry failures are back-transitions.

- **X-A** handles roll easily (at most 64 % of its roll authority), but its high
  stall speed means back-transitions only work when entered fast.
- **X-B** has the opposite problem: the big wing balloons up to 3.1 m, and at
  the 1.5 s rate it needs 105 % of the available roll authority.
- **4Q-2** passes every back-transition entered at 13 m/s, so its failures look
  like an entry-speed issue, not a design limit. If its forward transitions all
  pass (8 of 12 implies so but notebook 41 does not show it), 4Q-2 with a
  13 m/s entry would be 12/12 too.
- **4Q-1** passes 11/12 but rolls at only 24 deg/s, so it has little roll
  margin in gusts.

### Ranking

1. **4Q-2 with a 13 m/s back-transition entry**: likely best, pending its
   forward-transition table.
2. **X-A**: clean on the standard test, but only if back-transitions are always
   entered at about 16 m/s.
3. **4Q-1**: passes, with almost no roll authority.
4. **X-B**: fails roll hold on fast back-transitions and has the smallest pitch
   margin going forward.

## Where the X-wing wins

- **Standing on the tips:** tip-over angle about 60° vs 39° for 4Q-1; it takes a
  13–16 m/s wind to knock it over vs about 9 m/s.
- **Transitions:** X-A passes all 12 at the standard entry speed.
- **No bracing.**

## Caveats

- Neither X variant fits the capsule in line: 462 mm needed, 430 mm of pod.
- Notebook 41's summary and drag tables disagree slightly on power at 16 m/s
  (4Q-2: 59 vs 62 W; X-A: 88 vs 90 W). No ranking changes.
- First place in transitions needs the per-case transition tables for 4Q-1 and
  4Q-2 from the earlier Kestrel notebooks.

## Conclusion

On these numbers Kestrel-4Q-2 is still the better aircraft. The X-wing is worth
pursuing only if standing stability on the ground or transition robustness
matters more than efficiency.
