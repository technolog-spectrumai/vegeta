# Svarog — the physics of the device

Svarog is a series of laser designs that leads to one industrial device: a 1 kW fibre laser for welding metal, with
its guard. Each notebook is a step; this page says what physics each step stands on and what it found, with the Hyperion
module that does the computing in brackets. The methods are in [hyperion_physics.md](hyperion_physics.md); how each
notebook computes is in [svarog_solvers.md](svarog_solvers.md).

## 46 — the rod (Svarog): why end-pumped crystals do not scale

A diode-end-pumped Nd:YAG rod is the textbook laser: a four-level medium, a pump focused into one end, a two-mirror
cavity `[gain, resonator]`. Three things decide it:

1. **The pump spot is set by the fibre's brightness**, not by the lens: a 200 µm-core, NA 0.22 diode fibre has M² ≈ 85, so
   its image diverges inside the rod within a millimetre and the *effective* pump radius over the absorption length is
   never below ~280 µm; a 400 µm spot (magnification 4) is the better design because the thermal lens weakens with w_p².
2. **The thermal lens is the cavity's strongest element**: 1/f = η_h P_abs (dn/dT)/(π κ w_p²) gives f ≈ 136 mm at 20 W
   `[Innocenti]`, which closes the textbook plano–concave cavity at 19.5 W and shrinks every stable mode.
3. **Magni's rule**: a cavity that stays stable from cold to full pump holds a TEM₀₀ mode of only w² = 2λ/(π ΔD) in the
   rod — a quarter of the pump inside the mode here; the zone-II cavity (L > R_oc), unstable cold and opened by the lens
   above 13 W, holds two thirds `[Magni]`. The output (11 W at 20 W) hardly depends on any of it; the beam quality does.

The rod is where the physics of pumping, heat and modes is learnt; it is not the industrial source.

## 47 — the fibre laser (Svarog-fiber): the gain medium is the guided path, the gratings are the resonator

In a Yb-doped double-clad fibre the pump is trapped in the inner cladding and crosses the core along metres of fibre
(the (a/b)² overlap), the signal is the fibre's own LP₀₁ mode; the fibre alone amplifies, and only the two fibre Bragg gratings — the
feedback and the boundary conditions — make it an oscillator (an amplifier is the same medium with a seed and no
feedback): no thermal lens,
no alignment, the heat spread over metres `[fibre]`. The rate equations along the fibre `[Kelson & Hardy]` give: a
threshold of a fifth of a watt, a slope of 0.81 (the quantum limit λ_p/λ_s = 0.91 times the coupling), 16 W from 20 W of
976 nm pump in 9 m of 10/125 fibre, and the length as a cost choice once ~90 % of the pump is absorbed. One amplifier
stage (20/125, counter-pumped, 3.5 m) takes it to 59 W on 50 W of pump. The limits that matter are not the thermal lens's:
the coil's bend loss `[Marcuse 1976]`, stimulated Raman scattering `[Smith]`, the core's temperature per cooling
`[Brown & Hoffman]`, the few-mode fibre's higher modes, the gratings' temperature.

## 48 — the industrial laser (Svarog-industrial): a kilowatt at the workpiece

The same physics scaled: a 30/250 Yb fibre power stage seeded by notebook 47's 59 W and counter-pumped by six 220 W
976 nm modules through a (6+1)×1 combiner (the pigtails' étendue must fit the cladding's: it does, with six ports) gives
1 kW on 1.08 kW of pump in 10 m (12.3 dB gain, 93 % on the pump). The heat: ~105 W in the active fibre, 38 W/m at the
pump end — 40 K core rise on a water-cooled spool, 3000 K in still air, so the spool is water-cooled; the unabsorbed pump
and the splice losses (~50 W) into a cladding-mode stripper; 0.3 dB in the 20 m process fibre; a couple of watts in each
lens of the head. The delivery: a 50 µm, NA 0.12 process fibre has a beam parameter product of 3 mm·mrad (M² ≈ 9), so a
100/200 mm welding head gives a 100 µm spot with a 60 mrad cone and a 0.8 mm Rayleigh length — 12 MW/cm² on the work.

The welds `[welding]`: steel's keyhole opens at ~40 W in that spot and aluminium's at ~500 W (reflective, conductive),
so both weld in keyhole mode at 1 kW; the energy-balance penetration is 2.3 mm in steel and 2.0 mm in aluminium at
1 m/min, with full penetration of 1 mm sheet at 4 m/min, 3 mm at 0.6 m/min, 6 mm steel at 0.17 m/min; the process
window per thickness is the fastest full-penetration speed and the speed with 20 % of depth in hand `[Swift-Hook &
Gick, Steen & Mazumder]`.

## 49 — the guard (Svarog-guard): a stray kilowatt at 20 m

A laser guard is judged on the worst beam it can meet for a stated time `[IEC 60825-4]`. The beam cases at 20 m from a
1 kW welding head: the *raw collimated beam* (escaping before the focusing optic, or re-collimated by a reflector —
17 mm radius, 110 W/cm², the worst), the beam *diverging past the focus* (1.2 m radius, 0.02 W/cm², harmless to any
material), and a keyhole back-reflection (a fifth of the power, diverging). The barriers: black and clear and laser-grade
polycarbonate at 5 and 10 mm, a 65 mm fired-clay brick, aluminium at 1.5 and 3 mm, mild steel at 1.5 and 3 mm.

The physics decides most of them without a simulation `[thermal]`: a clear polycarbonate sheet transmits ~80 % at
1 µm — it is not a barrier at all; aluminium at 110 W/cm² absorbs 12 % and conducts it away, its steady state a few
hundred kelvin below melting — it holds for ever; 3 mm steel likewise. The black and the dyed polycarbonates absorb at the
surface, char and are eaten through — 5 mm in 16 s, 10 mm in 59 s: they hold T1 (10 s), not T2 (100 s). Brick melts at
its surface within a minute, then radiates like a furnace wall (2 MW/m² at 2000 K against 0.8 MW/m² absorbed) and its
melt front stalls a few millimetres in — it holds, provided the melt stays put. 1.5 mm steel is the one real
simulation: it reaches ~1250 K and creeps; the steady solve says it never melts.

The stray beam is the reason the laser's enclosure exists; the guard's structure (the panel's own stiffness, its mounts)
is a sheet-metal question, not a laser one, and is skipped on purpose.

## References

The numbered sources are in [hyperion_physics.md](hyperion_physics.md#references). The device-level ones:

- Datasheets the designs are read from: nLight / Coherent (Nufern) Yb double-clad fibres (10/125 LMA, 20/125, 30/250: core
  NA, cladding absorption at 976 and 915 nm, coating), 976 nm wavelength-stabilised pump modules (200/220 µm, NA 0.22
  pigtails), (6+1)×1 pump combiners, fibre Bragg grating pairs, 50 µm process fibres and welding heads with 100 mm
  collimators and 200 mm focusing lenses.
- D. J. Richardson, J. Nilsson, W. A. Clarkson, "High power fiber lasers: current status and future perspectives",
  J. Opt. Soc. Am. B 27, B63 (2010) — the fibre MOPA architecture and its limits.
- C. Jauregui, J. Limpert, A. Tünnermann, "High-power fibre lasers", Nature Photonics 7, 861 (2013).
- W. M. Steen, J. Mazumder, *Laser Material Processing*, 4th ed., Springer (2010) — keyhole welding, process windows.
- IEC 60825-4, *Safety of laser products — Part 4: Laser guards* — the exposure classes T1/T2/T3 and the test method.
- R. Hoffmann et al., "Laser guards for high-power lasers: protection times of passive guards", J. Laser Appl. — the
  measured burn-through of steel and aluminium panels the calculation is compared with when a test is run.
