# FALCO — component sources (market research of 2026-10-10)

Method: web search on 2026-10-10. Only the search-index snippets of the cited pages could be read: no page was opened,
so every row's **availability is "not confirmed"** and the figures are the snippets' (or the manufacturer's sheet as a
retailer copied it). Items not researched are marked **assumed** in `falco_systems.COMPONENTS` and here. The avionics
shared with NISUS (GNSS, the Jetson options) are in `nisus_component_sources.md`.

## 1. Motor

| motor | KV | mass | limits | published point | URL | availability |
|---|---|---|---|---|---|---|
| **T-Motor AT4125 KV540** (selected) | 540 | 355 g incl. cable | 6S; 85 A / 2000 W for 180 s; 446 W continuous (Chinese page) | APC 15x8, 100 %: 21.36 V, 72.77 A, 1554.08 W, 8879 rpm, 5537 gf, 3.56 g/W; 40 %: 22.38 V, 11.21 A, 4943 rpm, 1674 gf | https://store.tmotor.com/goods.php?id=827 ; https://cdn.robotshop.com/media/T/Tmo/RB-Tmo-442/pdf/ligpower_at4125_vtol_fixed_wing_aircraft_long_shaf_datasheet.pdf | not confirmed |
| SunnySky X4120 V3 KV430 | 430 | not found | 85 A / 2100 W (30 s) | none in the snippets | https://www.readymaderc.com/products/details/86776-sunnysky-x-series-v3-x4120-430kv-brushless-motor | not confirmed |
| SunnySky X4120 V3 KV480 | 480 | not found | 6S, 97 A (30 s) | none | https://www.readymaderc.com/products/details/86775-sunnysky-x-series-v3-x4120-480kv-brushless-motor | not confirmed |
| T-Motor AT4130 KV450 | 450 | not found | 6-12S | the KV450 rows were cut off | https://rcdrone.top/pl/products/tmotor-at4130 | not confirmed |

T-Motor's note on the table: measured on the 2017 test platform, "for reference only, not comparable horizontally"
with older data. The model fitted to the full-throttle point (`falco_systems.motor_model`) gives R = 0.068 Ω (ESC and
wires included) and checks itself against the 40 % point.

## 2. Propeller

| propeller | mass | URL | availability |
|---|---|---|---|
| APC 15x8E thin electric (fixed; it brakes and regenerates) | 47.9 g (Motion RC, black LPB15080E) / ~44 g (Gator-RC, LP15080E): 46 g taken | https://www.motionrc.com/products/apc-15x8-thin-electric-propeller-black-lpb15080e ; https://www.gator-rc.com/products/15x8e-propeller-electric-apc | not confirmed |

A folding propeller is not an option: braked, its blades fold back and it recovers nothing.

## 3. ESC

| ESC | rating | note | URL | availability |
|---|---|---|---|---|
| AM32 class, 80 A, 3-8S (a single-motor unit **assumed**; the listing is a 4-in-1 board) | 80 A continuous / 100 A peak | AM32's "damped light" mode does regenerative braking ("very fast motor retardation", HobbyWing 65A AM32 listing). **No vendor documents the energy returned to the pack**: a bench measurement (todo 5o.3) | https://www.rotorama.com/product/pilotix-esc-4in1-pilotix-80a-am32-8s-cz | not confirmed |

## 4. Cells and packs

| cell | capacity | current | charge | resistance | mass | URL | availability |
|---|---|---|---|---|---|---|---|
| Molicel INR21700-P45B | 4500 mAh typ (4300 min; one shop 4350) | 45 A | 4.5 A standard / 13.5 A max (Molicel sheet; one shop says 9 A) | AC 7 mΩ, DC 15 mΩ at 10 A (Battery Junction) | 70 g | https://www.imrbatteries.com/content/molicel_p45b.pdf ; https://batteryjunction.com/molicel-inr21700-p45b | not confirmed |
| Molicel INR21700-P50B | 5000 mAh | 60 A | 2C assumed | as P45B assumed | 70 g assumed | https://akkuteile.de/en/lithium-ionen-battery/size-21700/molicel/molicel-inr21700-p50b-5000mah-60a-3-6-3-7v-li-ion-battery_100638_3507 | not confirmed |

Packs (`falco_systems.PACKS`): 6S3P / 6S4P / 6S5P of P45B and 6S3P of P50B, assembled (nickel strip, wrap, balance
lead, XT90: +8 % mass, assumed). The open-circuit-voltage curve, the resistance's growth in the cold and the capacity's
loss are **assumed** (generic NMC); the charge limit below 5 °C is an assumed margin over the usual 0 °C datasheet
limit (lithium plating).

## 5. Not researched (assumed in `falco_systems.COMPONENTS`)

Servos (six 22 g metal-gear wing servos, 6 kg·cm), the servo BEC, the flight controller (Matek H743-WING class), the
pitot-static airspeed sensor, the lidar (TF02-Pro class), the ELRS 900 MHz receiver, the RFD900x telemetry, the
IMX477 camera, the LTE stick, the 12 V buck for the Orin, the wiring. The Jetson Orin Nano Super's figures are NISUS's
(`nisus_systems.JETSON_OPTIONS`, `nisus_component_sources.md` §7).
