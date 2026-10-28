"""Drako — what one delivery costs in Poland, against a car and a delivery van (Warsaw example).

Mission: 10 kg out at 40 km/h, back empty at 60 km/h, 20 % of the usable battery kept as margin on top of the 15 %
reserve in ``Drako.sizing``. The flight power is the hover power over the mission factor (the same model as the
endurance). The radius that leaves that margin is ``max_radius``; the example trip is Warsaw centre (Palace of Culture)
to Wilanów, about 9.5 km straight line, about 12 km by road.

All prices are 2026 estimates in PLN, net of VAT unless said otherwise (``PRICES``); change them and re-run.
"""
from __future__ import annotations

import math

from drako import Drako

PRICES = {
    "electricity_business_PLN_kWh": 0.95,   # C11 tariff, energy + distribution, net (estimate)
    "electricity_home_PLN_kWh": 1.10,       # G11, gross (estimate)
    "petrol_PLN_l": 6.20,                   # Pb95 (estimate)
    "diesel_PLN_l": 6.30,                   # ON (estimate)
    "battery_pack_PLN": 25000.0,            # 12S 60 Ah semi-solid pack with BMS (estimate)
    "battery_cycles": 1000,                 # to 80 % capacity (estimate)
    "drako_maintenance_PLN_h": 15.0,        # propellers, motors' bearings, inspection (estimate)
    "car_running_PLN_km": 0.60,             # tyres, service, wear; no depreciation, no driver
    "van_running_PLN_km": 0.80,
}
CO2 = {"grid_kg_kWh": 0.62, "petrol_kg_l": 2.31, "diesel_kg_l": 2.68}          # PL grid mix 2025-26, tank-to-wheel
KWH_PER_L = {"petrol": 8.9, "diesel": 9.9}
CHARGER_EFF = 0.92
MARGIN = 0.20

TRIP = {"name": "Warsaw: Palace of Culture -> Wilanów", "air_km": 9.5, "road_km": 12.0}
VEHICLES = {  # urban consumption per 100 km
    "car, petrol (Pb95)": ("petrol", 7.5, "car_running_PLN_km"),
    "car, electric": ("grid", 17.0, "car_running_PLN_km"),
    "delivery van 3.5 t, diesel": ("diesel", 11.0, "van_running_PLN_km"),
}
VAN_PARCELS_PER_ROUTE, VAN_ROUTE_KM = 120, 150          # a courier's day: the van's share per parcel


def _params():
    return {q.name: q.default for q in Drako.parameters}


def flight_power(p=None, mission_factor=0.70):
    s = Drako.sizing(p or _params())
    return {"loaded_W": s["hover_power_design_W"] / mission_factor, "empty_W": s["hover_power_empty_W"] / mission_factor,
            "usable_Wh": s["battery_Wh"] * 0.85}


def trip_energy_Wh(d_km, p=None, v_out=40.0, v_back=60.0):
    f = flight_power(p)
    return f["loaded_W"] * d_km / v_out + f["empty_W"] * d_km / v_back


def max_radius(p=None, margin=MARGIN):
    """Radius [km] of an out-loaded, back-empty trip that keeps ``margin`` of the usable energy."""
    f = flight_power(p)
    return (1 - margin) * f["usable_Wh"] / trip_energy_Wh(1.0, p)


def drako_trip(d_km, p=None):
    E = trip_energy_Wh(d_km, p) / 1000                                # kWh from the battery
    grid = E / CHARGER_EFF
    usable = flight_power(p)["usable_Wh"] / 1000
    hours = d_km / 40 + d_km / 60
    energy_cost = grid * PRICES["electricity_business_PLN_kWh"]
    wear = PRICES["battery_pack_PLN"] / PRICES["battery_cycles"] * E / usable
    return {"mode": "Drako (10 kg out, empty back)", "distance_km": 2 * d_km, "energy_kWh": grid,
            "energy_PLN": energy_cost, "wear_PLN": wear + PRICES["drako_maintenance_PLN_h"] * hours,
            "CO2_kg": grid * CO2["grid_kg_kWh"], "time_min": hours * 60}


def vehicle_trip(name, road_km):
    fuel, per100, run = VEHICLES[name]
    km = 2 * road_km
    q = per100 * km / 100
    if fuel == "grid":
        kwh, cost, co2 = q / CHARGER_EFF, q / CHARGER_EFF * PRICES["electricity_home_PLN_kWh"], q / CHARGER_EFF * CO2["grid_kg_kWh"]
    else:
        kwh, cost, co2 = q * KWH_PER_L[fuel], q * PRICES[f"{fuel}_PLN_l"], q * CO2[f"{fuel}_kg_l"]
    return {"mode": name + " (dedicated round trip)", "distance_km": km, "energy_kWh": kwh, "energy_PLN": cost,
            "wear_PLN": PRICES[run] * km, "CO2_kg": co2, "time_min": km / 25 * 60}       # 25 km/h urban average


def van_shared():
    per = VAN_ROUTE_KM / VAN_PARCELS_PER_ROUTE
    r = vehicle_trip("delivery van 3.5 t, diesel", per / 2)
    r.update(mode=f"delivery van, one of {VAN_PARCELS_PER_ROUTE} parcels on a {VAN_ROUTE_KM} km route", time_min=float("nan"))
    return r


def comparison(p=None, trip=TRIP):
    import pandas as pd
    rows = [drako_trip(trip["air_km"], p)] + [vehicle_trip(n, trip["road_km"]) for n in VEHICLES] + [van_shared()]
    df = pd.DataFrame(rows).set_index("mode")
    df["total_PLN"] = df["energy_PLN"] + df["wear_PLN"]
    return df.round(2)


def plot(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.5))
    labels = [m.split(" (")[0].replace(", one of", "\nshare of 1/") for m in df.index]
    y = range(len(df))
    axs[0].barh(y, df["energy_PLN"], label="energy", color="#2e7d32")
    axs[0].barh(y, df["wear_PLN"], left=df["energy_PLN"], label="battery wear / running", color="#9e9e9e")
    axs[0].set_title("cost per delivery [PLN, net]"); axs[0].legend()
    axs[1].barh(y, df["energy_kWh"], color="#1565c0"); axs[1].set_title("energy per delivery [kWh, at the grid / tank]")
    axs[2].barh(y, df["CO2_kg"], color="#6d4c41"); axs[2].set_title("CO2 per delivery [kg]")
    for a in axs:
        a.set_yticks(list(y)); a.set_yticklabels(labels if a is axs[0] else [""] * len(df)); a.invert_yaxis()
    for i, (e, w) in enumerate(zip(df["energy_PLN"], df["wear_PLN"])):
        axs[0].text(e + w, i, f" {e + w:.2f}", va="center")
    for i, v in enumerate(df["energy_kWh"]):
        axs[1].text(v, i, f" {v:.2f}", va="center")
    for i, v in enumerate(df["CO2_kg"]):
        axs[2].text(v, i, f" {v:.2f}", va="center")
    fig.suptitle(f"{TRIP['name']} and back ({TRIP['air_km']} km by air, {TRIP['road_km']} km by road), 10 kg; "
                 f"Drako's max radius with 20 % margin: {max_radius(p=None):.1f} km")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    return path


if __name__ == "__main__":
    import sys
    print(f"max radius, 10 kg out / empty back, 20 % margin: {max_radius():.1f} km")
    df = comparison()
    print(df.to_string())
    if len(sys.argv) > 1:
        print(plot(df, sys.argv[1]))
