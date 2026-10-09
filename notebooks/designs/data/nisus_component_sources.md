# NISUS — component source table (market research, 2026-10-09)

How this table was made: a web search on 2026-10-09 (extended search, result snippets of the cited pages). **Direct
page fetches were blocked from the research sandbox**, so every value is what the search index showed for the cited
page on that day, not a reading of the live page. "Availability confirmed" is **no** everywhere unless the snippet
itself said in stock; nothing below was bought or seen in a shop. Re-verify each line on the live page before
ordering. `nisus_systems.COMPONENTS` carries the subset the design uses, with the same URLs and these caveats.

## 1. Motors (2212–2216 / 2814 class, 9–10 inch propeller, 3S/4S)

| Motor | KV | Mass [g] | Max current / power (published) | Price | URL | Availability confirmed |
|---|---|---|---|---|---|---|
| SunnySky X2216 V3 (X2216-III) short shaft | 880 / 950 / 1100 / **1250** / 1400 | 67.5 / 69 / 69 / **69** / 71 | 1250KV: 40 A / 30 s; "max continuous 600 W" (V3 reseller text); older X2216-II listings: 392 W / 28 A | US$18.99 (sunnyskyusa) / US$24.99 (Team-Legit) / £16.11 (Unmanned Tech, X2216-II) | https://sunnyskyusa.com/products/sunnysky-x2216 ; http://en.rcsunnysky.com/x-fixedwingseries/44.html ; https://team-legit.com/sunnysky-x2216-v3-brushless-motor-1250kv.html ; https://www.3dxr.co.uk/fixed-wing-c27/fixed-wing-motors-c39/sunnysky-x2216-iii-v3-short-shaft-1250kv-p3681 | snippets say "in stock" at Team-Legit / RMRC / Unmanned Tech (not opened) |
| T-Motor AT2814 long shaft | 900 / 1050 / 1200 | 107 (with cable) | 50 A peak (180 s); 700 W (180 s); idle 1.5 A at 10 V | not stated | https://store.tmotor.com/product/at2814-long-shaft-fixed-wing-motor.html | unknown |
| T-Motor AT2312 long shaft | 1150 / 1400 | 60 (with cable) | 25 A peak; 350 W / 180 s; 2–4S; recommended AUW 400–900 g | not stated | https://store.tmotor.com/product/at2312-long-shaft-fixed-wing-motor.html | unknown |
| Emax GT2215/09 | 1180 | 70 | 24–26 A (sources disagree); 280–288 W | €34.50 (AlphaSport RC) | https://www.alphasportrc.com/index.php?main_page=product_info&products_id=6756&language=en ; https://www.amainhobbies.com/emax-gt2215-09-1180kv-brushless-motor-emx-mt-0409/p694266 | unknown |
| Cobra C-2217/16 | 1180 | 73 | 24 A continuous; 265 W continuous (3S) | not stated | https://www.cobramotorsusa.com/motors-2217-16.html | unknown |
| HobbyKing FC 28-22 | 1200 | 39 (43.5 measured with mount) | 14.5 A on 10x5 | not stated | https://hobbyking.com/en_us/fc-28-22-brushless-outrunner-1200kv.html | unknown |

Published propeller test data found:
- **SunnySky X2216 KV1250, APC 9x6 (9060), 11.1 V** (retailer copies of SunnySky's table): ~1 A / 11 W → 100 gf; ~3.7 A / 41 W → 300 gf; ~9.6 A / 107 W → 600 gf; **~20.8 A / 231 W → 1040 gf** (load temperature 42 °C). The earlier claim (20.8 A, 230.9 W, ~10.2 N) is consistent with these copies; the official sheet on sunnyskyusa.com was not opened; the rpm at the 1040 g point is not stated. Sources: https://techrolk.com/shop/product/sunnysky-x2216-brushless-motors-short-shaft-version-1250kv/ ; https://www.3dxr.co.uk/fixed-wing-c27/fixed-wing-motors-c39/sunnysky-x2216-iii-v3-short-shaft-1250kv-p3681
- **T-Motor AT2814 KV1050, ~15 V (4S)**: APC 10x5.5 at 65 % throttle 14.99 V 17.82 A 267 W 8876 rpm 1347 g (5.04 g/W); 100 % 14.62 V 39.83 A 582 W 11297 rpm 2274 g (3.90 g/W). Source: the T-Motor store page above.
- **Emax GT2215/09 1180KV, 3S**: 10x4.7 → ~8300 rpm, 26 A (≤60 s), 1250 g; 10x6 → ~8850 rpm, 24 A, 1140 g (power not stated; old listing).
- **HobbyKing FC 28-22 1200KV, 3S**: 10x5 → 7100 rpm, 14.5 A, 710 g.

## 2. Propellers (APC thin electric)

| Propeller | Mass [g] | Price | Shop / URL | Availability confirmed |
|---|---|---|---|---|
| APC 9x6E (LP09060E) | not stated | 14.50 zł (abc-rc.pl); 15.00 zł (modelarnia.pl); 14.19 zł (modele.sklep.pl) | https://abc-rc.pl/product-pol-1259-Smiglo-APC-9x6E-LP09060E.html ; https://sklep.modelarnia.pl/p520,smiglo-9x6-e-apc.html ; https://modele.sklep.pl/smiglo-apc-9x6-e-thin-electric-p-14259.html ; https://www.apcprop.com/product/9x6e/ | unknown |
| APC 9x6EP (pusher) | not stated | not stated | http://www.modele.sklep.pl/pl/Katalog/SMIGLA-KOLPAKI-PIASTY/Smigla-do-napedow-elektrycznych/APC-9x6-EP-Thin-Electric-Pusher.html | unknown |
| APC 10x5E | 20.13 (rc4max) | 15.50 zł (ef3m.pl); 16.08 zł (modele.sklep.pl); 19.69 zł (rc4max) | https://www.ef3m.pl/pl/p/Smiglo-APC-10x5E/974 ; https://www.rc4max.com/smiglo-apc-10x5e-zgodnie-z-ruchem-wskazowek-zegara | unknown |
| APC 10x5EP (pusher) | not stated | 26.49 zł (rc-team.pl, in a "discontinued" category) | https://rc-team.pl/scorpio-wycofane/157570-apc-smiglo-10x5e-elektro-pchajace-re-lp10050ep.html | likely not available |
| APC 10x7E | ~20 (0.71 oz, APC) | US$3.64–3.69 (US shops); no PL price found | https://www.apcprop.com/product/10x7e/ | unknown |
| APC 9x4.5E | not stated | US$3.29 (Innov8tive) | https://innov8tivedesigns.com/apc-9x4-5e-propeller.html | unknown |

APC performance data: index https://www.apcprop.com/technical-information/performance-data/ (PER3_9x6E.dat, PER3_10x5E.dat; analytical, not test-stand; the files could not be opened, so no values are quoted). RPM limit rule for E-series: 145 000 / D(in) → ~16 100 rpm for 9 inch (https://www.apcprop.com/wp-content/uploads/2022/03/APC-Propeller-RPM-Limits-rev5.pdf). Measured alternative: UIUC propeller database volume 1 (APC thin electric 10x5E, 10x7E static data): https://m-selig.ae.illinois.edu/props/volume-1/propDB-volume-1.html

## 3. ESCs (30–40 A, 3S–4S)

| ESC | Firmware | BEC | Rating | Mass [g] | Price | URL | Availability confirmed |
|---|---|---|---|---|---|---|---|
| Hobbywing Skywalker 40A V2 (HW80060022) | Hobbywing 32-bit | 5 V / 5 A switching | 40 A cont., 55–60 A burst, 3–4S | 36 | €19.90 incl. VAT (Robitronic DE, "available in 14 days"); US$19.99 | https://shop.robitronic.com/en/hobbywing-skywalker-v2-40a-hw80060022 ; https://www.readymaderc.com/products/details/skywalker-40a-airplane-esc-bec | not immediately in stock (Robitronic) |
| ZTW Beatles G2 40A SBEC | ZTW 32-bit | 5/6 V / 4 A | 40 A, 2–4S | 37 | US$15.69–19.99; £22.99 (J Perkins UK) | https://ztwesc.com/products/ztw-beatles-g2-40a-2-4s-sbec-for-rc-rc-airplane-fixed-wing ; https://www.jperkins.com/products/ZTW3040211 | unknown |
| Generic AM32 40A 2–4S with 5 V / 2 A BEC | AM32 | 5 V / 2 A | 40 A, 2–4S | 13.7–14.5 | not stated | https://totsrc.com/products/40amp-esc-40a-esc-4s-esc-for-rc-airplane-fpv-wing-rc-helicopter-w-bec-5v2a | unknown |
| Racerstar PGA40 (no BEC) | BLHeli_32 | none | 40 A, 2–6S | 6.2 (bare) | not stated | https://www.getfpv.com/racerstar-pga40-blheli-32-40a-2-6s-esc.html | unknown |
| Flycolor Francy 2 40A | BLHeli_32 | 5 V and 12 V / 2 A (listing unclear) | 40 A, 3–6S | 14 | not stated | https://www.getfpv.com/electronics/electronic-speed-controllers-esc/single-esc/flycolor-francy-blheli-32-3-6s-dshot-esc-w-led-20a-30a-40a-50a.html | unknown |

Vendor note (unverified): BLHeli_32 development has ended; AM32 is recommended for new purchases.

## 4. Batteries

| Pack | Brand | V nom | mAh | Wh | Mass [g] | Dimensions [mm] | C / current | Price | URL | Availability confirmed |
|---|---|---|---|---|---|---|---|---|---|---|
| 3S 2200 45C G-Tech | Gens Ace | 11.1 | 2200 | 24.4 | 189–190 (182 for the non-G-Tech 45C) | 105×34×23 | 45C | €24.99 (gensace.de) | https://gensace.de/collections/3s-lipo-battery ; https://wetronic.nl/gens-ace-GEA223S45X6GT-g-tech-lipo-3s-2200mah-xt60_nl | unknown |
| 3S 2200 45C | Tattu | 11.1 | 2200 | 24.4 | 190 | ~107×36×24 | 45C | not stated | https://www.motionew.com/shop/power-solution/battery-lipo-3s-2200mah-45c-tattu/ | unknown |
| 3S 3000 30C | Turnigy | 11.1 | 3000 | 33.3 | 269 | 136×43×19 | 30C | not stated | https://hobbyking.com (Turnigy 3000mAh 3S 30C XT60) | unknown |
| 3S 3000 30C nano-tech | Turnigy | 11.1 | 3000 | 33.3 | 215 | 135×44×17 | 30C | not stated | https://hobbyking.com/turnigy-nano-tech-3000mah-3s-30c-lipo-pack-wxt60.html | unknown |
| 3S 3000 120C LiHV | GNB | 11.4 | 3000 | 34.2 | 213 | 109×34×30 | 120C | not stated | https://www.gaoneng.shop/products/gaoneng-gnb-lihv-3s-11.4v-3000mah-120c-xt60-lipo-battery | unknown |
| 3S 3300 60C G-Tech | Gens Ace | 11.1 | 3300 | 36.6 | 267 (±20) | 136.5×42.6×21.6 | 60C | not stated | https://genstattu.com/gens-ace-g-tech-3300mah-3s-11-1v-60c-lipo-battery-pack-with-xt60-plug/ | unknown |
| 3S 3300 30C | Tattu | 11.1 | 3300 | 36.6 | 248 | 141×44×21 | 30C | US$47.00 | https://www.motionew.com/shop/power-solution/battery-lipo-3s-3300mah-30c-tattu/ | unknown |
| 3S 3300 90C | GNB | 11.1 | 3300 | 36.6 | 231 ±5 | 135×44×19 | 90C | bulk only | (Indian distributor) | unknown |
| 4S 2200 40C | Turnigy | 14.8 | 2200 | 32.6 | 216 | 106×35×29 | 40C | not stated | https://hobbyking.com/turnigy-2200mah-4s-40c-lipo-pack-w-xt60.html | unknown |
| 4S 2200 30C Shorty | Turnigy | 14.8 | 2200 | 32.6 | 181 | 79×32×35 | 30C | not stated | https://hobbyking.com/en_us/turnigy-2200mah-4s-14-8v-30c-shorty.html | unknown |
| Li-ion 3S2P Molicel P42A (assembled) | Nexus Battery Systems (US) | 10.8 | 8400 | 90.7 | 450 | 70×48×67 | 45 A/cell cont. | US$102 | https://nexusbatterysystems.com/products/3s2p-42a | unknown (US shop) |
| Molicel P42A cell | Molicel | 3.6 | 4200 | 15.1 | ≤70 (datasheet) | 21700 | 45 A cont. | US$5.99–8.99 | https://www.molicel.com/product/ ; https://www.18650batterystore.com/products/molicel-p42a | — |

No assembled EU Li-ion 3S2P pack with a published mass was found.

## 5. Flight controller, GNSS, RC, telemetry

**Holybro Kakute F405-Wing Mini** — https://holybro.com/products/kakute-f405-wing-mini ; docs https://docs.holybro.com/autopilot/kakute-f405-wing-mini/overview ; firmware https://docs.holybro.com/autopilot/kakute-f405-wing-mini/supported-firmware
- STM32F405RGT6, ICM-42688P, SPL06 baro, 128 Mbit blackbox, 5 UARTs, 7 PWM outputs; **17 g with the USB extension board**; 25×30×20 mm; 20×20 mm M2 mounting; input 2–8S.
- BEC: 5.3 V 3 A continuous / 4.8 A peak (FC, RX, GPS, camera, telemetry); servo rail 5.3 V default or 7.2 V by jumper, 3 A cont. / 4.8 A peak; 3.3 V 150 mA. **No 9 V or 12 V rail.**
- Current sensor 110 A continuous / 132 A peak; ArduPilot BATT_AMP_PERVLT 40 A/V, BATT_VOLT_MULT 11.
- ArduPilot: Holybro names the target **KakuteF405-Wing** (Plane 4.5.2 or later; one page says 4.5.6); INAV 7.1.2+, Betaflight 4.5.2+. The ardupilot.org board page could not be opened.
- EU price: €84.10 "in stock" (snippet) https://openelab.io/products/holybro-kakute-f405-wing-mini ; rc-innovations.es €49.99 but "not available for sale"; https://www.flyingtech.co.uk/product/holybro-kakute-f405-wing-mini-wing-flight-controller/ (price not in snippet); https://www.mybotshop.de/Holybro-Kakute-F405-Wing-Mini_1 ; Holybro store US$77.98.

| Module | Mass [g] | Supply | Current | URL | Availability confirmed |
|---|---|---|---|---|---|
| Matek M10Q-5883 GNSS + compass | ~8 | 4–9 V | 13 mA (snippet) | https://www.unmannedtechshop.co.uk/products/matek-m10q-5883-gps-module ; https://www.rotorama.com/product/matek-m10-5883-gps-s-kompasem | unknown |
| Holybro Micro M10 GPS | 14 (no case) / 16 | 5 V | < 200 mA (max) | https://holybro.com/products/micro-m10-gps | unknown |
| RadioMaster RP1 V2 ELRS 2.4 GHz | 2.2 (with antenna) | 5 V | not stated | https://radiomasterrc.com/products/rp1-expresslrs-2-4ghz-nano-receiver | unknown |
| RadioMaster RP3 V2 ELRS 2.4 GHz diversity | 4.6 | 5 V | not stated; telemetry TX 100 mW | https://radiomasterrc.com/products/rp3-expresslrs-2-4ghz-nano-receiver | unknown |
| Happymodel ES900RX ELRS 900 MHz | 0.6 (no antenna) | 5 V | ~100 mA | https://www.getfpv.com/happymodel-expresslrs-es900rx-receiver-module.html | unknown |
| Holybro SiK Telemetry Radio V3 433 MHz 100 mW | 3.6 (7.4 with antenna) | 5 V | TX 100 mA at 20 dBm, RX 25 mA | https://holybro.com/products/sik-telemetry-radio-v3 ; https://docs.holybro.com/radio/sik-telemetry-radio-v3 | unknown |

## 6. Camera and video link

| Item | Mass [g] | Supply | Current / power | URL | Availability confirmed |
|---|---|---|---|---|---|
| Caddx Ratel 2 (analog 1200TVL) | 5.9 (5.5–6.6) | 5–40 V | ≤ 200 mA at 12 V (third-party) | https://en.caddxfpv.com.cn/products/ratel-2.html ; https://www.fpv24.com/en/caddx/caddx-ratel-2-analoge-fpv-kamera-1200tvl-schwarz | unknown |
| RunCam Phoenix 2 (analog 1000TVL) | 9 | 5–36 V | 200 mA at 5 V / 85 mA at 12 V (~1 W) | https://shop.runcam.com/runcam-phoenix-2/ ; https://www.drone-fpv-racer.com/en/runcam-phoenix-2-fpv-camera-5998.html | unknown |
| RushFPV Tank Solo 5.8 GHz VTX | 12 | 7–36 V | 25/400/800/1000+ mW steps; current at 400 mW not stated; 5 V/1 A aux output | https://rushfpv.net/products/tank-solo-vtx ; https://yourfpv.co.uk/product/rushfpv-tank-solo-vtx/ | unknown |
| Rush Tank Ultimate Mini (reference) | — | — | 185 mA at 200 mW, 380 mA at 800 mW (supply voltage not stated) | (getfpv snippet) | — |
| Walksnail Avatar HD Kit V2 (digital) | ~27 (VTX 17.6, camera 7.2, antenna 2) | 6–25.2 V | power not stated; CE < 14 dBm | https://www.caddxfpv.com/products/walksnail-avatar-hd-kit-v2 ; https://www.rotorama.com/product/walksnail-walksnail-avatar-hd-micro-kit-v2-32gb | unknown |
| DJI O3 Air Unit (digital) | 39.5 | 7.4–26.4 V | ~15–16 W (1.6 A at 10 V) | https://oscarliang.com/dji-o3-air-unit-fpv-goggles-2/ ; https://www.rotorama.com/product/dji-o3-air-unit | unknown |

## 7. Companion computer (Jetson) and camera

- **Jetson Nano 4 GB Developer Kit (B01): end of life.** NVIDIA's lifecycle page lists the developer kit as EOL; the Nano production module stays available through January 2027. https://forums.developer.nvidia.com/t/jetson-nano-developer-kit-eol/276730 ; https://developer.nvidia.com/embedded/lifecycle . Final software: **JetPack 4.6.6 (L4T 32.7.x, Ubuntu 18.04)** https://developer.nvidia.com/jetpack-sdk-466 . Power modes 10 W (default) and 5 W; micro-USB 5 V / 2 A, barrel jack 5 V / 4 A https://jetsonhacks.com/2019/04/10/jetson-nano-use-more-power/ . Module mass not stated; dev kit ~249 g per an Amazon listing (possibly with packaging; unverified). DFRobot marks it discontinued https://www.dfrobot.com/product-1909.html .
- **Jetson Orin Nano Super Developer Kit (945-13766-0005-000)**: US$249 MSRP https://developer.nvidia.com/buy-jetson ; EU €519 incl. VAT **out of stock** (welectron.com) https://www.welectron.com/NVIDIA-Jetson-Orin-Nano-Super-Dev-Kit_1 ; €355 ex-VAT https://www.siliconhighwaydirect.com/product-p/945-13766-0005-000.htm ; 5 329 SEK incl. VAT https://www.electrokit.com/en/nvidia-jetson-orin-nano-developer-kit ; £199 ex-VAT https://uk.rs-online.com/web/p/processor-development-tools/2647384 . 100×79×21 mm (Waveshare) https://www.waveshare.com/jetson-orin-nano.htm ; **mass not stated**. Power modes 7 W, 15 W, and 25 W / MAXN SUPER with JetPack 6.2 https://developer.nvidia.com/blog/nvidia-jetpack-6-2-brings-super-mode-to-nvidia-jetson-orin-nano-and-jetson-orin-nx-modules . JetPack 6.x (Ubuntu 22.04, CUDA 12): JetPack 4.6 software must be retargeted.
- **Raspberry Pi Camera Module 2 (IMX219, CSI)**: ~3 g, 25×23×9 mm; current not stated; Orin Nano needs the 15→22 pin adapter or a native 22-pin IMX219 module. https://www.elektor.com/raspberry-pi-camera-module-v2 ; https://developer.ridgerun.com/wiki/index.php/NVIDIA_Jetson_Orin_Nano/Camera_Sensors_Support/IMX219 ; https://www.arducam.com/arducam-imx219-pdaf-cdaf-autofocus-camera-module-with-case-for-raspberry-pi-nvidiar-jetson-orin-series.html

## 8. Servos

| Servo | Torque 4.8 / 6 V [kg·cm] | Speed 4.8 / 6 V [s/60°] | Mass [g] | Stall current | Price | URL | Availability confirmed |
|---|---|---|---|---|---|---|---|
| Emax ES08MA II (analog, metal gear) | 1.6 / 2.0 (some 1.5 / 1.8) | 0.12 / 0.10 | 12 | not stated (working 200 mA) | US$9.49 (RMRC, discontinued listing) | https://www.readymaderc.com/products/details/emax-es08ma-ii-metal-gear-servo ; https://www.getfpv.com/emax-es08ma-ii-12g-mini-metal-gear-analog-servo-for-rc-model.html | unknown |
| Emax ES09MD (digital, MG) | 2.3 / 2.6 | 0.10 / 0.08 | 14.8 | not stated | £15.39 (4-max UK); US$12.99 | https://www.4-max.co.uk/servo-emax-ES09MD.html ; https://servodatabase.com/servo/emax/es09md | unknown |
| KST X08 V6 (digital, coreless, MG) | 2.2 at 6 V / 2.8 at 8.4 V | 0.15 at 6 V / 0.09 at 8.4 V | 8–8.9 | not stated | not stated | https://www.hyperflight.co.uk/products.asp?code=KST-X08 | unknown |
| Savox SH-0255MG → SH-0255MGP | 3.1 / 3.9 | 0.16 / 0.13 | 15.8 | not stated | US$34.99–41.99 | https://savox-servo.com/en/product/SH-0255MGplus/ | SH-0255MG discontinued |

## 9. Materials

| Material | Spec | Price | URL | Availability confirmed |
|---|---|---|---|---|
| Carbon tube 10/8 mm × 1000 | — | 17.50 zł | https://www.sklep.rc-lipol.pl/pl/p/Rurka-weglowa-10,08,0-x-1000-mm/405 | **out of stock** (snippet "brak towaru") |
| Carbon tube 10/8 × 1000 | — | 19.90 zł | https://modele.sklep.pl/rurka-weglowa-o-10-08-0-x-1000-mm-p-18643.html | stock shown 0 |
| Carbon tube 10/8 × 1000 | — | 25.49 zł | https://www.sklep.modelmaking.pl/product-pol-1770-Rura-weglowa-10-8-x-1000-mm.html | unknown |
| Carbon tube 10/8 × 1000, pultruded | fibre tensile strength 1280 MPa stated | 50.56 zł | https://carboncenter-sklep.pl/rura-z-wlokna-weglowego-10x8x1000-mm-pultruzja-p-346.html | unknown |
| Carbon tube 10/8 × 1000 | — | €6.19 | https://modelemax.pl/en/carbon-tubes/366-carbon-tube-10-8-1000mm.html | snippet "in stock – 32 pc" |
| Easy Composites 10 mm (8 mm) pultruded tube | tensile modulus (lengthways) 28–40 GPa (EC datasheet) | €12.60/m ex-VAT | https://www.easycomposites.eu/10mm-pultruded-carbon-fibre-tube ; https://media.easycomposites.eu/datasheets/EC-TDS-Carbon-Fibre-Pultrusions.pdf | unknown |
| Easy Composites 10 mm (8 mm) roll-wrapped | — | €12.65/m ex-VAT | https://www.easycomposites.eu/10mm-roll-wrapped-carbon-fibre-tube-metric | unknown |
| Carbon tube 8/6 × 1000 | — | 22.99 zł (rc.susco.pl "available"); 27.50 zł (rc-lipol, out of stock) | https://rc.susco.pl/26977-rurka-weglowa-o-8060-x-1000-mm | susco: available (snippet) |
| Carbon tube 6/4 × 1000 | — | 21.00 zł (modelarzsklep.pl) | https://www.modelarzsklep.pl | unknown |
| Balsa 1.5×100×1000 mm | — | €2.59 incl. VAT (Höllein, prices as of 04/2026) | https://hoelleinshop.com/en/products/balsabrett-1-5mm-x-100mm-x-1000mm | unknown |
| colorFabb LW-PLA 750 g | — | €26.49–32.05 | https://colorfabb.com/lw-pla-black | snippet "in stock" |
| XPS (Styrodur 300) | per pack | 146–240 zł per pack | https://www.hurtowniastyropianu.pl/styropian/styropian/styrodur-xps/xps-300/ ; https://realbud.com/pl/c/Styrodur-XPS/2466 | unknown |
| Pultruded CFRP modulus (other vendors) | 70–150 GPa quoted (T300 70–90, T800 130–150) | — | https://carbonfiber.flexcompositeeng.com/resources/pultruded-carbon-fiber-tube-mechanical-properties-typical-values-ranges.html | — |

Note the spread of the stated carbon-tube modulus (28–40 GPa Easy Composites vs 70–150 GPa elsewhere): the design's
boom and spar checks use the datasheet of the tube actually bought; `nisus.CARBON_TUBE` carries the assumption and the
notebook shows the sensitivity.
