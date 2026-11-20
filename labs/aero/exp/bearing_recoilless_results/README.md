# Recoilless long cannon for 8 mm steel balls (Pekari)

Pneumatic, recoilless (rear venturi), 100 m effective range. Sizing: `bearing_recoilless.py`; CAD:
`bearing_recoilless_cad.py` (STEP here: `bearing_recoilless.step`); tests: `bearing_target.py`, `bearing_wind.py`,
`bearing_speed_dist.py`; game CFD: `game_cfd/cfd_8mm.sh`.

| item | value |
|---|---|
| ball | 8 mm G100 stainless (440C), 2.08 g (+0.05 g felt wad) |
| air | 300 bar HPA bottle (0.8 L) -> regulator -> 0.5 L steel plenum, 100 bar rated, set to **26.3 bar g** |
| valve | 1/2" hammer valve, 12 mm orifice, Cd 0.75, opens in 1.5 ms, side inlet into the chamber |
| tube | **1.0 m**, 8.00 mm H8 honed bore, 16 mm OD, gravity magazine into a breech port |
| rear venturi | chamber 24 mm x 50 mm, **throat 6.91 mm**, 2:1 cone (exit 9.8 mm), 7.5 deg half angle |
| muzzle speed | **280 m/s** (Mach 0.82), 82 J |
| recoil | **0.000 N s** net over the shot (a plain gun would be 1.08 N s at 21.4 bar) |
| air per shot | 2.57 g (about 93 shots per 0.8 L bottle) |
| back-blast | jet 526 m/s, faster than 15 m/s to 2.1 m behind |
| scatter (assumed) | 0.06 deg direction, 0.5 % speed (precision tube, rigid mount) |

At 100 m (calm): 1.08 deg elevation, 0.52 s flight, 141 m/s, energy 20.5 J (sd 0.17); 16/20 hits on a 300 mm disc.
Energy (Gaussian fit, 2000 shots): 0 m 81.5 ± 0.81 J, 50 m 39.6 ± 0.33 J, 100 m 20.5 ± 0.17 J (`energy_gauss.yaml`).
Wind at 100 m: ~16 cm drift per 1 m/s of crosswind; 2 m/s uncorrected -> 0/20; 5 m/s corrected -> 16/20,
with ±2 m/s gusts -> 6/20; 10 m/s head/tail wind -> 15-16/20 (`../bearing_wind_results/`).
