# ARIGO products: short descriptions

The ARIGO family is a set of compact electric fixed-wing interceptors with a nose camera and PN-guided autonomous chase.
All numbers are from the stability run `arigo_stability_20261120-113605` (commit 239caeff, default pack). KPI scores are 0–100 and relative to the 13 configurations in that run.

Mass budgets are hand estimates (±20 %), and the CFD is screening RANS.

> The CG fixes for ARIGO-0 and ARIGO-20 (commit a6758e6) are not in this run yet.

## In the catalogue

### ARIGO-1 hybrid: recommended all-rounder
- **Layout:** pusher, with the motor behind the wing and the camera alone in the nose (clean view). Hybrid wing.
- **Key numbers:**

  | Mass | Max speed | Stall | Climb | Endurance | Range | Tightest turn radius |
  |---|---|---|---|---|---|---|
  | 1.76 kg | 52.5 m/s | 12.4 m/s | 21.6 m/s | 75 min | 82 km | 16.3 m |

- **Launch:** hand launch, 2.2 s to safe flight.
- **KPIs:** general **64.9 (#1)**, urban **75.8 (#1)**, open field 57.6 (#1).
- **Pack and propeller:** Li-ion 3S2P P42A, 10×12 two-blade propeller.

### ARIGO-1 zero: long patrol
- **Layout:** same fuselage as ARIGO-1 hybrid, with the zero wing.
- **Key numbers:**

  | Mass | Max speed | Stall | Endurance | Range |
  |---|---|---|---|---|
  | 1.76 kg | 51.7 m/s | 11.9 m/s | **78 min (longest)** | 80 km |

- **Launch:** hand launch, 2.1 s.
- **KPIs:** general 61.4 (#2), urban 71.9.

### ARIGO-0 hybrid: low-cost nose tractor
- **Layout:** conventional nose tractor. The camera sits low and looks through the propeller disc.
  - 88 % of the image is behind the disc.
  - About 9 % is blocked per frame.
  - About 2 % remains after two-frame fusion (the spinner).
- **Key numbers:**

  | Mass | Max speed | Stall | Endurance | Range |
  |---|---|---|---|---|
  | 1.78 kg | 50.6 m/s | 11.9 m/s | 68 min | 70 km |

- **Launch:** hand launch, 2.2 s.
- **KPIs:** general 56.2, urban 72.1, open field 43.7. It is 5–10 points behind ARIGO-1.
- **Why choose it:** cost and simplicity.

### ARIGO-20: ducted-fan dash interceptor
- **Layout:** two 70 mm EDFs at the rear and an H tail.
- **Key numbers:**

  | Mass | Max speed | Climb | Stall | Endurance | Range |
  |---|---|---|---|---|---|
  | 2.16 kg | **66 m/s (fastest)** | **28 m/s (best)** | 15.4 m/s | 30 min | 40 km |

- **Launch:** too fast to throw by hand, so it needs a catapult (7.7 s to safe flight).
- **KPIs:** general 45.7, open field 57.1 (#2), urban 33.8 (poor).
- **Role:** a specialist for open ground.

## Discarded

### ARIGO-2: twin wing tractors
- About 80 g heavier than ARIGO-1 and no better: general 49.2.
- With one motor out, the minimum control speed V_mc is about 24 m/s, far above stall. A motor failure at launch means a departure.

### ARIGO-0 chin strut
- The camera sits on a strut ahead of the propeller: drag plus a vibrating mount.
- General 44.0 against 56.2 for the low camera.

### Goshawk wing
- The weakest wing on every airframe.
- ARIGO-1 goshawk: general 47.0, endurance 68 min, stall 14.8 m/s.

### Tilt-rotor and tail-sitter VTOL
- Rejected early. The mechanism or the extra lift motors weigh about as much as the pack, and the mission doesn't need a precise landing.
