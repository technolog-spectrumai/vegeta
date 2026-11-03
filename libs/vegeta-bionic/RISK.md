# vegeta-bionic — risk management file (stage 1)

ISO 14971-style, kept with the code it controls (`docs/bionic.md`: medical-device rules from day one). Stage 1 is
software only: a simulated glove driving nothing. The hazards below are the ones its outputs will carry into stage 2
(teleoperation of robot pincers) and later into worn devices; each has a control and how it is verified.

| # | hazard | cause | harm (when it reaches hardware) | control | verified by |
|---|---|---|---|---|---|
| H1 | wrong joint angle | sensor offset/gain scatter, cross-talk, drift | the pincer closes or opens when the hand did not | per-unit calibration from held poses (`Calibration.fit`); residuals reported per sensor | `test_calibration_recovers_sensed_joints` |
| H2 | calibration from bad poses | the user does not hold the poses, or a pose does not move a joint | systematic error on every command | the fit refuses poses that do not move a joint; pose errors of ±5° bounded in test | `test_calibration_refuses_static_joint`, `test_calibration_with_pose_errors` |
| H3 | wrong wrist orientation | gyro bias, magnetometer hard iron | the arm (stage 6) points the wrong way | Mahony filter with bias estimation; hard-iron sphere fit from a wrist sweep | `test_wrist_orientation_after_magnetometer_calibration` |
| H4 | stale or lost glove data | link latency, jitter, loss | the robot acts on an old hand pose | sequence numbers, stale messages dropped (`InProcessLink`); `Watchdog` for the receiver | `test_link_*`, `test_watchdog` |
| H5 | malformed message | version mismatch, corruption | garbage commands | magic and version checked on decode | `test_decode_rejects_foreign_messages` |
| H6 | command steps | estimation noise, a glitch | a sudden pincer motion | `Limits` (range and rate) on every command (stage 2 applies them) | `test_limits` |

Open for stage 2: the pincer force and speed limits, the clutch's failure modes (a stuck clutch must freeze, never
release a grip unexpectedly), and the watchdog's action on the robot side.
