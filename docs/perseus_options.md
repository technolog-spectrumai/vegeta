# Perseus: other manipulators and a second hand — investigation

Perseus's chassis stays as it is: 1100 × 520 mm hull, 150 mm tracks, two 1.5 kW drives, 1.9 kWh, 181 kg. On it,
seven configurations are judged by one set of rules (`labs/land/designs/perseus_options.py`;
`python3 labs/land/sims/perseus_options.py --docs` rewrites the table):

**The rules**
- **The cut zone:** branches 1.2–2.2 m ahead of the hull centre and 0.8–2.2 m up, on the centre line, where
  Perseus's saw works.
- **The saw arm:** the stall safety factor cutting outstretched, with the cutting loads × 1.5 (Perseus's own
  measure).
- **A grab on the cutting arm:** the smaller of what its drives hold (peak ÷ 1.5) and 75 % of the tipping load
  (straight ahead or sideways).
- **A second hand:** the share of the saw's cuts it can hold 0.3 m beyond the cut, what it holds there (median),
  the largest green-spruce piece that is (1.3 m long), and the tip margin with both arms out and the hand holding
  that piece.
- **Driving:** the 20° climb at peak, the steepest grade held continuously, ground pressure and driving
  endurance, all on gravel, the same as in [`land_comparison.md`](land_comparison.md).
- **Roof space:** two pedestals fit side by side on the 520 mm roof only with the boom moved 0.13 m left and the
  hand 0.15 m right.

None of it is measured. The models' masses, actuators and prices are class values.

<!-- table:begin -->
| | boom saw | boom grab | arm6 saw | boom saw + light hand | boom saw + arm6 grab | boom saw + long arm6 grab | arm6 saw + arm6 grab |
|---|---|---|---|---|---|---|---|
| mass [kg] | 181 | 179 | 177 | 188 | 227 | 261 | 223 |
| added to Perseus [kg] | +0 | -2 | -4 | +7 | +46 | +80 | +43 |
| 20° grade (peak, traction) | yes | yes | yes | yes | yes | yes | yes |
| steepest grade held continuously [°] | 28 | 28 | 28 | 28 | 23 | 19 | 23 |
| ground pressure [kPa] | 6.6 | 6.6 | 6.5 | 6.9 | 8.3 | 9.6 | 8.2 |
| driving endurance [h] | 9.1 | 9.1 | 9.2 | 8.8 | 7.8 | 7.0 | 7.8 |
| saw: stall SF cutting outstretched (loads x1.5) | 1.35 | — | 1.28 | 1.35 | 1.35 | 1.35 | 1.28 |
| cut zone the cutter reaches | 70% | 70% | 58% | 70% | 70% | 70% | 56% |
| lift with the grab on the same arm (min of drives, 75 % tipping) | — | 1.4 m: 22 kg, 1.8 m: 22 kg, 2.1 m: 17 kg | 1 m: 34 kg, 1.4 m: 20 kg | — | — | — | — |
| second hand | — | — | — | Pekari-2M light arm | Pumba main arm | Calydon heavy arm | Pumba main arm |
|   its mass [kg] | 0.0 | 0.0 | 0.0 | 7.1 | 46.5 | 80.4 | 46.5 |
|   cut zone it reaches | — | — | — | 0% | 26% | 57% | 26% |
|   of the saw's cuts it can hold (0.3 m beyond the cut) | — | — | — | 0% | 12% | 46% | 16% |
|   holds there, median (peak / 1.5) [kg] | — | — | — | 0.0 | 20.1 | 41.2 | 20.1 |
|   largest branch piece it holds (1.3 m long) [m] | — | — | — | 0.00 | 0.15 | 0.20 | 0.15 |
| tip margin, arms out at the cut [m] | 0.34 | 0.34 | 0.31 | 0.34 | 0.08 | -0.16 | 0.07 |
|   tip angle [°] | 36 | 36 | 36 | 35 | 8 | -15 | 7 |
| second hand's modules (class prices) | — | — | — | 4.9–10.7 k | 16.9–33.0 k | 25.4–48.0 k | 16.9–33.0 k |
<!-- table:end -->

## Findings

**1. Different manipulators: yes, by changing the head, not the boom.**
- **Perseus's cylinder boom** reaches more of the cut zone than any joint-module arm the chassis can carry (70 %
  against 58 % for Pumba's 6-DOF arm), and keeps the widest tip margin.
- **A quick coupler at its wrist** turns it into a multi-tool boom, like Pumba's main arm (`pekari_arms.TOOLS`):
  - the saw, as now
  - a two-jaw grab: 22 kg out to 1.8 m, 17 kg at 2.1 m
  - other heads (hedge cutter, mulcher, probe) by the same rule
- **The wrist decides the grab's rating.** The cylinders have 4–7× margin; the 150 N m wrist module holds 22 kg.
  The 400 N m module (+3.3 kg) raises the drives' rating to 60 kg, but then the machine tips first. That leaves
  36 kg at 1.4 m, 22 kg at 1.8 m and 15 kg at 2.1 m (75 % of tipping). So the stronger wrist pays off only close in.
- **The 6-DOF joint arm with the saw** is 4 kg lighter and more dexterous: wrist yaw and roll let it cut at any
  angle, and it changes tools on the same coupler. It reaches less (58 %), and its saw margin is a little thinner
  (1.28 against 1.35). It suits work close to the machine, not high branches.

**2. A second hand: not on this chassis, for holding the cut piece.** Physics leaves no good middle ground:
- **A light hand** (Pekari-2M's arm, +7 kg) costs little but reaches none of the cut zone: 0.97 m from the roof
  ends short of where the saw works.
- **A Pumba-size hand** (+46 kg) can hold only 12–16 % of the saw's cuts (pieces up to 0.15 m across, about
  20 kg). With both arms out, Perseus's tip margin falls to 7–8 cm (7–8°): the next gust, slope or snag tips it.
  The machine also loses 5° of continuous climb and over an hour of driving.
- **A long hand** (Calydon's arm, +80 kg) holds 46 % of the cuts, up to 0.20 m pieces, but **tips Perseus over**
  (−0.16 m).
- **Why it matters:**
  - A cut branch falls: onto the machine, a line, or people.
  - A branch propped or cut from below pinches the bar: `ogma.cutting.kerf_opens` gives "pinches" for every
    case above.
  - The forestry answer without a second hand is the undercut, then the top cut: the piece falls clear, away
    from the machine. Perseus's mission already does it.
- **Where a second hand does pay off** is close to the machine, not at the cut: picking cut pieces off the ground
  and loading them on a deck or into a chipper. That needs a deck and a wider stance: a different machine.

**3. Recommendation for Perseus:**
- keep the chassis and the cylinder boom
- add a quick coupler at the wrist (saw and grab heads)
- consider the 400 N m wrist only if grab work close in (≤ 1.4 m) matters
- no second hand

A two-handed machine for this work wants Pumba's base: deck, width, two front stabilizers. Or it wants a machine
designed around the second hand: Timon.
