# vegeta-fulgur — risks

Kept from the first commit; every result the toolkit returns carries the supply current and the spark margin.

| risk | what the toolkit does about it |
|---|---|
| **High voltage** (15–40 kV DC): lethal at the supply's current, painful at a corona's; charged electrodes and output capacitors stay charged after switch-off | `HVSupply` reports `c_out`'s stored energy and the output current; the lab's build notes require a bleeder, a ballast resistor and a discharge stick; no design is published without its `spark_margin` |
| **Fire**: a spark on a supply without fold-back melts and ignites resin frames (the video's 4B) | `supply.spark_energy` against `resin_melt_energy`; the presets name a current limit and trip time |
| **Ozone and NOx**: a corona in air makes O₃ (~ 1–10 mg per Wh) — indoor tests need ventilation | the chemistry module (`chem`, later) reports the rate; until then the README's note: outdoors or under extraction |
| **Humidity**: flashover along wet resin at voltages that were safe dry | the surface criterion is part of every margin; Warsaw's winter is a design condition |
| **RF noise and EMI** from the corona and from sparks: a flight controller next to it fails | noted for the demonstrator's electronics placement; not modelled |
| **Over-claiming**: EHD thrust per watt is 2–20 mN/W; a demonstrator that cannot lift its supply is not a vehicle | `docs/plasma.md`: the demonstrator decision is made from the scaling study's thrust per kilogram, honestly |
| **Model risk**: the 1-D model has one fitted factor and estimated geometry | every estimated dimension is listed in `RingThruster.estimated`; the 2-D solve (P3) replaces the factor |
