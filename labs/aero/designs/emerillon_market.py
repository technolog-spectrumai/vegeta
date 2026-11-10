"""ÉMERILLON — the market: what the parts cost and what the shops say about them (notebook 37).

Every bought part of the speed quad as one ``Part`` row: the mass, the electrical numbers the flight model needs, the
price a shop listed and where. The prices were read from shop pages through a web search on ``PRICED_ON`` — search
snippets, not order confirmations: several listings were out of stock, some were sale prices, and a shop's page may
have changed since. ``kind`` says what a row's numbers are:

- ``sourced``  — price and the headline numbers (mass, current, power) come from a shop or maker listing (``url``);
- ``partly``   — the price is sourced, some electrical numbers are assumptions (named in ``note``);
- ``assumed``  — no listing found: an estimate, to be replaced before anything is ordered.

``seen`` keeps the range of prices found for the same part, lowest to highest: the spread between shops is often
larger than the difference between two parts, which matters when the question is speed against price. Prices are in
US dollars because most of the listings are; the few euro prices are converted with ``USD_PER_EUR`` (an assumed
rate — one constant to edit).

``bom(config)`` is the bill of materials of one configuration (``emerillon_flight.Config``), ``price(config)`` its
total. Not in the price: the printer, the radio, goggles, a charger, shipping, taxes, and the parts a crash eats.
"""
from __future__ import annotations

from dataclasses import dataclass, field

PRICED_ON = "2026-11-10"
USD_PER_EUR = 1.08                 # assumed exchange rate (edit it; three rows depend on it)


@dataclass(frozen=True)
class Part:
    key: str
    name: str
    group: str                     # motor / propeller / esc / battery / fc / radio / video / gps / filament / hardware
    price_usd: float               # the price the trade uses (per piece, or per ``unit``)
    mass_g: float = 0.0
    seen: tuple = ()               # (lowest, highest) listed price [USD] across the shops found
    shop: str = ""
    url: str = ""
    kind: str = "sourced"
    note: str = ""
    unit: str = "piece"
    data: dict = field(default_factory=dict, hash=False, compare=False)

    def __getattr__(self, name):                       # the electrical numbers read as attributes: motor.kv
        try:
            return self.__dict__["data"][name]
        except KeyError:
            raise AttributeError(name) from None


def _eur(x):
    return round(x * USD_PER_EUR, 2)


# ----------------------------------------------------------------------------------------------------------- motors
# kv [rpm/V], r_ohm (winding, phase to phase), i0_a (no-load), max_a (burst), max_cells, d_mm x h_mm (the can)
MOTORS = [
    Part("eco2807-1300", "EMAX ECO II 2807 1300KV", "motor", 19.49, 54.0, (19.49, 24.99), "EMAX USA",
         "https://shop.emax-usa.com/products/eco-ii-2807-brushless-motor-1300kv-1500kv-1700kv", "partly",
         "54 g and 3-6S from the listings; resistance, no-load current and the 45 A burst are assumptions (7-inch class)",
         data=dict(kv=1300.0, r_ohm=0.062, i0_a=1.3, max_a=45.0, max_cells=6, d_mm=33.5, h_mm=20.0)),
    Part("f100-2810-1100", "T-Motor F100 2810 1100KV", "motor", 44.99, 67.0, (35.89, 59.76), "Lumenier",
         "https://www.lumenier.com/products/t-motor-f100-2810-lr-cinematic-motor-1100kv-1350kv", "partly",
         "67 g with cable, 45.6 A peak (60 s), 6S from the listings; resistance and no-load current assumed",
         data=dict(kv=1100.0, r_ohm=0.055, i0_a=1.2, max_a=45.6, max_cells=6, d_mm=34.0, h_mm=23.0)),
    Part("v3115-900", "T-Motor Velox V3115 900KV", "motor", 49.90, 113.1, (49.90, 62.99), "Pyrodrone",
         "https://pyrodrone.com/products/t-motor-velox-v3115-cinematic-fpv-drone-motor-900kv", "partly",
         "113.1 g with cable and 1894 W for 10 s from the listings (75 A at 6S); 45 mOhm from one shop's sheet "
         "(Rotorvillage, which also gives 51 A / 1219 W for 60 s); no-load current assumed; flown here on 6S only",
         data=dict(kv=900.0, r_ohm=0.045, i0_a=1.5, max_a=75.0, max_cells=6, d_mm=37.5, h_mm=27.0)),
    Part("v3120-700", "T-Motor Velox V3120 700KV", "motor", 56.90, 135.0, (52.89, 56.90), "T-Hobby",
         "https://www.t-hobby.com/products/tmotor-velox-v3120-brushless-cinematic-motor-for-fpv-drones", "partly",
         "the price and '3-12S' from the listings (the shops disagree on the KV list: 450/500/700); mass, resistance, "
         "no-load current and the 80 A burst are assumptions",
         data=dict(kv=700.0, r_ohm=0.040, i0_a=1.4, max_a=80.0, max_cells=12, d_mm=37.5, h_mm=32.0)),
    Part("v3120-900", "T-Motor 3120 900KV (special wind)", "motor", 85.0, 135.0, (), "none found",
         "https://www.t-hobby.com/products/tmotor-velox-v3120-brushless-cinematic-motor-for-fpv-drones", "assumed",
         "the record holder's motor as the press describes it (four T-Motor 3120 at 900 KV, the winding raised from "
         "800 KV); no shop lists a 3120 at 900 KV, so the price is the 700 KV's plus a guess for a custom wind, and "
         "the resistance is the 700 KV's scaled by (700/900)^2; 110 A burst assumed",
         data=dict(kv=900.0, r_ohm=0.024, i0_a=2.0, max_a=110.0, max_cells=12, d_mm=37.5, h_mm=32.0)),
]

# ------------------------------------------------------------------------------------------------------- propellers
# d_in x p_in, blades, rpm_limit_k: the maker's (or the shops') speed limit as rpm = k / diameter[in]
PROPELLERS = [
    Part("hq7x4x3", "HQProp DP 7x4x3 (polycarbonate)", "propeller", 10.39 / 4, 9.0, (10.39 / 4, 3.99), "rcdrone.top",
         "https://rcdrone.top/products/hqprop-dp7x4x3-light-grey-2cw-2ccw-poly-carbonate", "partly",
         "$10.39 for a set of four; the freestyle propeller the cheap motors are sold for; mass and a speed limit are "
         "assumptions (HQProp publishes none)",
         data=dict(d_in=7.0, p_in=4.0, blades=3, rpm_limit_k=190000.0, chord=(0.085, 0.115, 0.055), tc_tip=0.07)),
    Part("apc6x6e", "APC 6x6E thin electric", "propeller", 2.59, 6.0, (2.55, 2.59), "APC",
         "https://www.apcprop.com/product/6x6e/", "partly",
         "price from APC; the shops quote 145,000/D rpm for thin electric propellers (one quotes 190,000/D for this "
         "one, as a Speed 400 type); mass assumed",
         data=dict(d_in=6.0, p_in=6.0, blades=2, rpm_limit_k=145000.0, chord=(0.075, 0.105, 0.045), tc_tip=0.06)),
    Part("apc7x7e", "APC 7x7E thin electric", "propeller", 3.00, 9.0, (), "BuddyRC",
         "https://www.buddyrc.com/products/apc-b7x7e-thin-electric-propeller-1", "assumed",
         "the listing exists, its price did not come through: about the 6x6E's and the 7x10's; 145,000/D rpm",
         data=dict(d_in=7.0, p_in=7.0, blades=2, rpm_limit_k=145000.0, chord=(0.075, 0.105, 0.045), tc_tip=0.06)),
    Part("apc7x10", "APC 7x10 sport", "propeller", 2.78, 14.0, (2.78, 2.78), "APC",
         "https://www.apcprop.com/product/7x10/", "partly",
         "price from APC; a glow-engine sport propeller (thicker, heavier), 190,000/D rpm; mass assumed. No 7x10E or "
         "7x11E listing was found",
         data=dict(d_in=7.0, p_in=10.0, blades=2, rpm_limit_k=190000.0, chord=(0.08, 0.11, 0.05), tc_tip=0.08)),
    Part("own7x13", "own propeller 7x13 (machined or moulded, proof-spun)", "propeller", 15.0, 12.0, (), "", "", "assumed",
         "no shop sells this pitch at 7 inches: the lab's own blade (prop_lab) in carbon or PA6-CF, each one spun on "
         "the bench above its flight rpm before it flies; the price is a guess at material and machine time, and "
         "there is no maker's limit — the 190,000/D here is only the mark the proof spin has to pass",
         data=dict(d_in=7.0, p_in=13.0, blades=2, rpm_limit_k=190000.0, chord=(0.08, 0.11, 0.05), tc_tip=0.05)),
    Part("own7x16", "own propeller 7x16 (machined or moulded, proof-spun)", "propeller", 15.0, 12.0, (), "", "", "assumed",
         "as the 7x13, with more pitch still",
         data=dict(d_in=7.0, p_in=16.0, blades=2, rpm_limit_k=190000.0, chord=(0.08, 0.11, 0.05), tc_tip=0.05)),
]

# ------------------------------------------------------------------------------------------------------------- ESCs
# max_a per motor, max_cells, count: 1 = a four-in-one board, 4 = one per motor; with_fc: the stack brings its FC
ESCS = [
    Part("speedybee-f405v4-55", "SpeedyBee F405 V4 stack (FC + BLS 55A four-in-one)", "esc", 69.99, 23.4, (69.99, 86.49),
         "Pyrodrone", "https://pyrodrone.com/products/speedybee-f405-v4-stack-w-55a-3-6s-bls-4in1-esc-30x30mm", "partly",
         "the flight controller comes with it; one shop marks it discontinued for a newer version; mass assumed",
         data=dict(max_a=55.0, max_cells=6, count=1, with_fc=True)),
    Part("xrotor-g2-65", "Hobbywing XRotor G2 65A four-in-one", "esc", 76.99, 17.0, (76.99, 119.99), "HobbyKing",
         "https://hobbyking.com/en_us/hobbywing-xrotor-g2-65a-esc-4in1-3-6s-32-bit-bec-for-fpv.html", "partly",
         "3-6S; mass assumed", data=dict(max_a=65.0, max_cells=6, count=1, with_fc=False)),
    Part("apd-80f3", "APD 80F3[X] 80A (one per motor)", "esc", 45.99, 10.0, (45.99, 97.90), "Wires Computing",
         "https://store.wirescomputing.com/", "partly",
         "8S, 80 A; out of stock at the lowest price, $97.90 at NeuMotors; mass assumed",
         data=dict(max_a=80.0, max_cells=8, count=4, with_fc=False)),
    Part("apd-120f3", "APD 120F3[X] 120A 12S (one per motor)", "esc", 99.00, 25.0, (99.00, 163.90), "Wires Computing",
         "https://store.wirescomputing.com/products/apd-120f3-x-12s-50v-120a", "partly",
         "12S (50 V), 120 A; $109.99 at GetFPV (out of stock), $163.90 at NeuMotors; mass assumed",
         data=dict(max_a=120.0, max_cells=12, count=4, with_fc=False)),
]

# -------------------------------------------------------------------------------------------------------- batteries
# every pack is 6S; the aircraft carries n_series x n_parallel of them. size_mm = (length, width, height)
BATTERIES = [
    Part("cnhl-1300-100c", "CNHL Black 6S 1300 mAh 100C", "battery", 29.99, 230.0, (29.99, 51.98), "RaceDayQuads",
         "https://www.racedayquads.com/products/cnhl-black-series-22-2v-6s-1300mah-100c-lipo-battery-xt60",
         note="230 g, 76x47x35 mm from the listings", data=dict(cells=6, ah=1.3, c_rating=100.0, size_mm=(76.0, 47.0, 35.0))),
    Part("tattu-r5-1400-150c", "Tattu R-Line V5 6S 1400 mAh 150C", "battery", 38.79, 222.0, (38.79, 55.49), "YohoRC",
         "https://yohorc.com/product/tattu-r-line-version-5-0-6s-1400mah-lipo-battery-pack/", "partly",
         "222 g from the listings; EUR 35.75 at Drone-FPV-Racer; the size is the R-Line V6 1480's",
         data=dict(cells=6, ah=1.4, c_rating=150.0, size_mm=(76.0, 38.0, 36.0))),
    Part("tattu-r6-1700-160c", "Tattu R-Line V6 6S 1700 mAh 160C", "battery", 46.99, 246.0, (46.99, 53.0), "GenStattu",
         "https://genstattu.com/tattu-r-line-version-6-0-1700mah-6s-160c-22-2v-with-xt60-lipo-battery",
         note="246 g (+-20), 78.5x38.4x40.9 mm from the listings",
         data=dict(cells=6, ah=1.7, c_rating=160.0, size_mm=(78.5, 38.4, 40.9))),
    Part("cnhl-2000-100c", "CNHL Black 6S 2000 mAh 100C", "battery", 50.99, 346.0, (26.99, 56.99), "Pyrodrone",
         "https://pyrodrone.com/collections/batteries/products/cnhl-black-series-2000mah-22-2v-6s-100c-lipo-battery-xt60", "partly",
         "346 g from the listings; $26.99 direct from CNHL, $56.99 at GetFPV — the widest spread in this table; size assumed",
         data=dict(cells=6, ah=2.0, c_rating=100.0, size_mm=(100.0, 45.0, 40.0))),
]

# ------------------------------------------------------------------------------------------- what every build carries
FIXED = [
    Part("matek-h743-slim", "Matek H743-SLIM V3 flight controller", "fc", 96.00, 8.0, (78.00, 139.39), "WREKD",
         "https://wrekd.com/products/matek-h743-slim-flight-controller-30x30", "partly",
         "bought when the ESC does not bring a flight controller; most listings out of stock; mass assumed"),
    Part("rp1-v2", "RadioMaster RP1 V2 ExpressLRS receiver", "radio", 18.99, 1.5, (18.99, 25.0), "RadioMaster",
         "https://www.radiomasterrc.com/products/rp1-expresslrs-2-4ghz-nano-receiver", "partly", "mass assumed"),
    Part("matek-m10q", "Matek M10Q-5883 GPS", "gps", _eur(25.0), 7.0, (), "Unmanned Tech",
         "https://www.unmannedtechshop.co.uk/products/matek-m10q-5883-gps-module", "partly",
         "the speed is measured by GPS, so it is not optional; EUR 25 as NISUS's parts list has it (not re-checked)"),
    Part("runcam-phoenix2", "RunCam Phoenix 2 camera", "video", _eur(28.0), 9.0, (), "RunCam",
         "https://shop.runcam.com/runcam-phoenix-2/", "partly", "EUR 28 as NISUS's parts list has it (not re-checked)"),
    Part("rush-tank-solo", "Rush Tank Solo video transmitter", "video", _eur(40.0), 9.0, (), "Rush FPV",
         "https://rushfpv.net/products/tank-solo-vtx", "partly", "EUR 40 as NISUS's parts list has it (not re-checked)"),
    Part("hardware", "wires, XT60/XT90, screws, heat-set inserts, antenna", "hardware", 25.0, 45.0, (), "", "", "assumed",
         "an allowance"),
]

# price per kilogram; density [kg/m3], modulus [GPa] and strength [MPa] are typical datasheet values (assumed)
FILAMENTS = [
    Part("petg-hf", "Bambu PETG HF", "filament", 22.19, unit="kg", seen=(18.20, 22.19), shop="3DPrintingUSA",
         url="https://3dprintingusa.com/collections/bambu-lab-cf", kind="partly",
         note="the nose, the skins, the tail cone", data=dict(density=1270.0, e_gpa=2.1, strength_mpa=45.0)),
    Part("pa6-cf", "Bambu PA6-CF", "filament", 85.98, unit="kg", seen=(85.98, 85.98), shop="Fabreeko",
         url="https://www.fabreeko.com/products/bambu-pa6-cf", kind="partly",
         note="$42.99 for 0.5 kg (a sale price); the arms and the motor mounts, which get warm",
         data=dict(density=1090.0, e_gpa=5.5, strength_mpa=100.0)),
    Part("tpu", "TPU 95A", "filament", 40.0, unit="kg", kind="assumed", note="the tail bumpers and the camera boot",
         data=dict(density=1210.0, e_gpa=0.03, strength_mpa=30.0)),
]

PRINT_WASTE = 1.6                  # filament bought per gram flying: supports, purge, the prints that failed (assumed)
SPARE_PROPELLER_SETS = 1           # one spare set of four in the price: the first hard landing takes a set
CATALOGUE = {p.key: p for p in MOTORS + PROPELLERS + ESCS + BATTERIES + FIXED + FILAMENTS}


def part(key: str) -> Part:
    return CATALOGUE[key]


def esc_for(cells: int, amps: float) -> Part | None:
    """The cheapest ESC set that takes ``cells`` in series and ``amps`` per motor (None when the catalogue has none)."""
    ok = [e for e in ESCS if e.max_cells >= cells and e.max_a >= amps]
    return min(ok, key=lambda e: e.price_usd * e.count + (0.0 if e.with_fc else part("matek-h743-slim").price_usd), default=None)


def _at(p: Part, which: str) -> float:
    """A part's price as the trade uses it (``used``), or the lowest / highest listing found (``low`` / ``high``)."""
    if which == "used" or not p.seen:
        return p.price_usd
    return p.seen[0] if which == "low" else p.seen[1]


def bom(config, printed_g: dict | None = None, which: str = "used") -> list[dict]:
    """The bill of materials of one configuration: rows of part, quantity, unit price, line price, mass, kind, shop.
    ``which``: 'used', 'low' or 'high' (the spread between shops). ``printed_g`` is the flying mass of each filament ({'petg-hf': g, 'pa6-cf': g, 'tpu': g}); without it the printed
    parts are priced from ``emerillon_flight.printed_masses(config)``."""
    if printed_g is None:
        import emerillon_flight as ef
        printed_g = ef.printed_masses(config)
    esc = part(config.esc)
    lines = [(part(config.motor), 4), (part(config.propeller), 4 * (1 + SPARE_PROPELLER_SETS)), (esc, esc.count),
             (part(config.battery), config.n_series * config.n_parallel)]
    lines += [(p, 1) for p in FIXED if not (p.group == "fc" and esc.with_fc)]
    rows = [{"part": p.name, "group": p.group, "qty": q, "unit [USD]": round(_at(p, which), 2), "line [USD]": round(_at(p, which) * q, 2),
             "mass [g]": round(p.mass_g * (q if p.group != "propeller" else 4), 1), "kind": p.kind, "shop": p.shop, "url": p.url}
            for p, q in lines]
    for key, grams in printed_g.items():
        f = part(key)
        bought = grams * PRINT_WASTE / 1000.0
        rows.append({"part": f"{f.name} ({grams:.0f} g flying, x{PRINT_WASTE} bought)", "group": "filament", "qty": round(bought, 3),
                     "unit [USD]": f.price_usd, "line [USD]": round(bought * f.price_usd, 2), "mass [g]": round(grams, 1),
                     "kind": f.kind, "shop": f.shop, "url": f.url})
    return rows


def price(config, printed_g: dict | None = None, which: str = "used") -> float:
    """The total of ``bom`` [USD]; ``which`` = 'low' / 'high' prices every part at the cheapest / dearest listing found."""
    return round(sum(r["line [USD]"] for r in bom(config, printed_g, which)), 2)


def price_table(parts=None) -> list[dict]:
    """The catalogue as rows for a table: what was found, where, and the spread between shops."""
    rows = []
    for p in (parts or list(CATALOGUE.values())):
        lo, hi = (p.seen + (None, None))[:2] if p.seen else (None, None)
        rows.append({"group": p.group, "part": p.name, "price [USD]": round(p.price_usd, 2), "per": p.unit,
                     "lowest seen": None if lo is None else round(lo, 2), "highest seen": None if hi is None else round(hi, 2),
                     "mass [g]": p.mass_g or None, "kind": p.kind, "shop": p.shop, "note": p.note})
    return rows
