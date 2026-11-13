# vegeta-bionic

Shared primitives for human–robot limbs: a kinematic hand, glove sensor models, a simulated data glove, calibration and
orientation estimation, the glove-to-host link, safety primitives. They serve glove teleoperation of robot pincers
(next: retargeting, haptics, the teleop loop), a puppet arm for teaching by demonstration, and later assistive
exoskeletons. The plan and the decisions are in `../../docs/bionic.md`; the risks in `RISK.md`.

```bash
pip install -e libs/vegeta-bionic[test]
python -m pytest -q libs/vegeta-bionic/tests
```

```python
from vegeta.bionic import Calibration, SimGlove, motions, estimate

glove = SimGlove(seed=1)                                          # a right-hand glove, its units' errors fixed by the seed
routine = motions.calibration_routine()                            # hold flat, fist, pinch, spread
cal = Calibration.fit(glove.run(routine)[0], routine.windows, glove.spec)
cal.with_magnetometer(glove.run(motions.wrist_tour(20.0))[0])     # wave the hand: the magnetometer's offset
samples, truth = glove.run(motions.grasp_cycles())
poses = estimate.estimate_hand(samples, cal)                       # joints + the wrist's orientation
```

Stage 1 numbers (the tests pin them): sensed joints under 2° rms and 3.5° peak after calibration (each unit's mounting
scatter sets that); unsensed DIPs
inferred within ~5–6° rms (tendon coupling); the pinch aperture within ~2 mm rms; the wrist's orientation within ~3° once
the magnetometer is calibrated (5–10° without).

Depends on numpy only; MuJoCo scenes that use it live in a lab.
