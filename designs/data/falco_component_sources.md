# FALCO — component sources (market research of 2026-10-10)

Method: web search on 2026-10-10. Only the search-index snippets of the cited pages could be read: no page was opened,
so every row's **availability is "not confirmed"** and the figures are the snippets' (or the manufacturer's sheet as a
retailer copied it). Items not researched are marked **assumed** in `falco_systems.COMPONENTS` and here. What FALCO
shares with NISUS+ (the ESC class, the servos, the flight controller, the sensors, the Orin, the telemetry) is in
`nisus_plus_component_sources.md` and `nisus_component_sources.md`; this file holds what the tractor changes.

## 1. Motor

| motor | KV | mass | limits | published point | URL | availability |
|---|---|---|---|---|---|---|
| **T-Motor AT5220-A KV220, long shaft** (selected) | 220 | 450 g incl. cable; Φ60 x 112.3 mm; 24N28P; 36 mΩ; idle 1.8 A at 10 V | 6-12S; 70 A / 2700 W for 180 s | APC 18x8 at 12S, 100 %: 43.33 V, 48.86 A, 2116.80 W, 8141 rpm, 2.037 N m, 8185 gf, 3.87 g/W; the table from 40 % (44.30 V, 4.27 A, 189 W, 3585 rpm, 1469 gf) to 100 % | https://store.tmotor.com/product/at5220-a-fixed-wing-motor.html ; https://store.tmotor.com/de/product/at5220-a-fixed-wing-motor.html (the rows to 100 %) ; https://cdn.robotshop.com/media/T/Tmo/RB-Tmo-447/pdf/ligpower_at5220a_20_25cc_vtol_fixed_wing_aircraft_datasheet.pdf | not confirmed (getfpv lists it as discontinued; racedayquads, ligpower list it) |
| T-Motor AT5220-A KV380, long shaft | 380 | 465 g; 11 mΩ; idle 4.0 A at 10 V | 6S; 100 A / 2200 W for 180 s | no thrust table in the snippets | https://www.ligpower.com/product/at5220-a--kv380-fixed-wing-motor.html | not confirmed |

T-Motor's note on the table: bench tests, "for reference only, comparison with that of other motor types is not
recommended". Two powers in the published rows (65 %: 611.06 W, 80 %: 1092.62 W) repeat the row above them — typos in
the source; `falco_systems.PUBLISHED_STATIC` replaces them by V x I. The model fitted to the 100 % point
(`falco_systems.motor_model`) gives R = 0.130 Ω (ESC and wires included, as NISUS+'s fit of the AT4125 gave 0.068 Ω)
and kt (I − I0) = 2.043 N m against the published 2.037 (the datasheet's I0 of 1.8 A is consistent with the table);
the BEMT's 18x8 torque at the 70 % row checks within 6 %.

**Why the pack changes to 8S3P** (`falco_systems.drive_options`): on the 6S pack the KV220 has the volts for ~3900 rpm
on the 20x13 only (~0.5 kW shaft, static T/W < 0.7). The KV380 on 6S would run the 20x13 past the ESC's 80 A and sits
at that limit when held (the fallback that keeps the 6S4P is the KV380 with an 18x10, full throttle at the ESC's
limit). The same 24 Molicel P45B cells re-connected 8S3P (the same 264 x 76 x 47 mm box, 388 Wh, 28.8 V nominal) give
the frame study's 20x13 point (~4950 rpm, ~1.05 kW shaft, 59 N static) at 49 A within every limit. Every 8S-fed
device is rated to 36 V (the ESC's 3-8S rating; the FC's 6-36 V input; a 36 V buck and BEC: assumed).

## 2. Propeller

| propeller | mass | URL | availability |
|---|---|---|---|
| **APC 20x15E thin electric** (selected; fixed: it brakes, regenerates and parks) | 117.93 g (Motion RC, LP20015E); 118 g taken | https://motionrc.com/collections/2-blade-propellers/products/apc-20x15-thin-electric-propeller-lp20015e | not confirmed |
| APC 20x13E thin electric (the frame study's size) | 117.08 g (Motion RC, the reverse LP20013EP) / 117.1 g (a French listing of the LP20013E; leguide.com) | https://www.motionrc.eu/collections/apc-propellers/products/apc-20x13-thin-electric-propeller-reverse-lp20013ep ; https://www.leguide.com/gtin/00686661200258 | not confirmed |

The map uses the frame study's generic planform scaled to 20 inches (`frame_study.propeller`); the real blade's chord
and twist are not in the model (todo: the APC geometry file). The pitch is the propeller trade's
(`falco_flight.propeller_trade`): on the AT5220 at 8S the 20x13 cruises near its zero-thrust advance ratio (the
blades at their zero-lift angle: profile drag, 67 % efficiency, more cruise power than NISUS+'s 15x8), the 20x15
cruises at 73 % and climbs 15 % faster for the same static current class. A folding propeller is not an option:
braked, its blades fold back and recover nothing; parked, a two-blade fixed propeller lies along the wing.

## 3. Parking the propeller

| item | what | URL | availability |
|---|---|---|---|
| ESC: AM32 80 A 3-8S class | the active brake (`damped light` regenerative braking: Hobbywing 65A AM32 listing), bidirectional mode (the nudge in reverse), brake-on-stop | https://www.rotorama.com/product/pilotix-esc-4in1-pilotix-80a-am32-8s-cz (a 4-in-1 board: a single-motor unit of the class is assumed) | not confirmed |
| position sensor | a latching Hall sensor on the firewall and a magnet on the spinner's backplate: one pulse per turn says where the blades are | assumed (5 g) | — |
| software | ArduPilot has no propeller-stowing function: the forum thread "using magnetic position sensor on motor to stow the propeller" (2022) suggests a position-sensing ESC (a German maker, ~500 EUR) or a VESC in HFI mode; the parking routine here is a companion-computer script on the Orin (brake to a stop, read the sensor, nudge to the horizontal mark) | https://discuss.ardupilot.org/t/using-magnetic-position-sensor-on-motor-to-stow-the-propeller/79437 ; https://discuss.ardupilot.org/t/can-we-make-motors-to-stop-everytime-at-particular-angle/61099 | — |

A pusher operator's report on the same forum ("landing in FBWA mode throttle will not cut": the propeller still turning
at touchdown broke it) is the failure the parking routine prevents; FALCO's scenarios report whether the propeller was
parked at touchdown.

## 4. Changed from NISUS+

| item | FALCO | basis |
|---|---|---|
| spinner 44 mm with the hub's magnet carrier | 30 g | assumed |
| wiring | 65 g: the battery leads longer (the pack under the wing, the ESC behind the firewall), the motor leads short | estimate |
| elevator and rudder servos | at the tail tube's end fitting (short linkages); their leads through the 28 mm tube | assumed |
| servo BEC, Orin buck | rated to 36 V (8S) | assumed |
| pitot, lidar, camera | in the chin under the propeller's hub, ahead of the wing; the propeller's wash raises the pitot's reading at power (to calibrate) | assumed |
| GNSS | on the tail cone's top, away from the motor | assumed |
