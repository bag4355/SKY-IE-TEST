"""
smartphone_data_prep.py
──────────────────────────────────────────────────────────────────────────────
Reads every raw CSV / SQLite file shipped with `smartphone_data_v22`, performs
all deterministic preprocessing *once*, and exposes **pure‑Python containers**
(dictionaries, DataFrames, lists) used by the optimisation model.

This module is **import‑time only** – no heavy computation inside the MILP
file itself, keeping the model‑build phase fast and deterministic.

(c) OpenAI o3 — 2025‑07‑18
"""
from __future__ import annotations
import os, math, sqlite3, datetime as dt, itertools, warnings
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy  as np
from haversine import haversine

from smartphone_config_utils import (
    BASE_DIR, daterange, truck_leadtime, eu_zone_pair,
    TRUCK_USD_KM, TRUCK_CO2_KM, MODES_FCWH, MODES_WHCT
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ════════════════════ RAW TABLE LOAD ═══════════════════════════════════════
def _load_csv(name: str, **kw):
    path = Path(BASE_DIR) / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, **kw)

site      = _load_csv("site_candidates.csv")
site_cost = _load_csv("site_init_cost.csv")
cap_week  = _load_csv("factory_capacity.csv")
lab_req   = _load_csv("labour_requirement.csv")
lab_pol   = _load_csv("labour_policy.csv")
prod_cost = _load_csv("prod_cost_excl_labour.csv")
inv_cost  = _load_csv("inv_cost.csv").set_index("sku")["inv_cost_per_day"].to_dict()
short_cost= _load_csv("short_cost.csv").set_index("sku")["short_cost_per_unit"].to_dict()
carbon_f  = _load_csv("carbon_factor_prod.csv").set_index("factory")["kg_CO2_per_unit"].to_dict()
sku_meta  = _load_csv("sku_meta.csv", parse_dates=["launch_date"])
weather   = _load_csv("weather.csv", parse_dates=["date"])
oil_price = _load_csv("oil_price.csv", parse_dates=["date"])
currency  = _load_csv("currency.csv", parse_dates=["Date"]).rename(columns={"Date":"date"})
holiday   = _load_csv("holiday_lookup.csv", parse_dates=["date"])
machine_fail = _load_csv("machine_failure_log.csv",
                         parse_dates=["start_date","end_date"])

# demand SQLs
def load_demand(db_name: str, table: str) -> pd.DataFrame:
    with sqlite3.connect(Path(BASE_DIR) / db_name) as conn:
        df = pd.read_sql(f"SELECT * FROM {table}", conn, parse_dates=["date"])
    return df

d_train = load_demand("demand_train.db", "demand_train")
d_eval  = load_demand("demand_eval.db",  "demand_eval")
d_test  = load_demand("demand_test.db",  "demand_test")
demand_full = pd.concat([d_train, d_eval, d_test], ignore_index=True)

# ════════════════════ DERIVED MASTER SETS ══════════════════════════════════
FACTORIES = site.query("site_type=='factory'")['site_id'].tolist()
WAREHOUSES= site.query("site_type=='warehouse'")['site_id'].tolist()
CITIES    = site['city'].unique().tolist()
SKUS      = lab_req['sku'].tolist()

# geo helpers
site_coord = site.set_index("site_id")[["lat","lon"]].to_dict("index")
city_coord = site.drop_duplicates("city").set_index("city")[["lat","lon"]].to_dict("index")
iso_site   = site.set_index("site_id")["country"].to_dict()
iso_city   = site.drop_duplicates("city").set_index("city")["country"].to_dict()

# life‑time (week buckets)
life_weeks = {r.sku: math.ceil(r.life_days / 7) for _, r in sku_meta.iterrows()}

# ════════════════════ DISTANCE / LEADTIME GRIDS ════════════════════════════
edges_FC_WH: list[tuple[str,str,str]] = []
LT_FC_WH, COST_FC_WH, CO2_FC_WH, BORDER_FC_WH = {}, {}, {}, {}

for f, h in itertools.product(FACTORIES, WAREHOUSES):
    isoF, isoH = iso_site[f], iso_site[h]
    dkm = haversine(
        (site_coord[f]["lat"], site_coord[f]["lon"]),
        (site_coord[h]["lat"], site_coord[h]["lon"])
    )
    for mode in MODES_FCWH:
        # mode‑permission logic
        if isoF == isoH and mode != "TRUCK":
            continue
        if isoF != isoH and mode == "TRUCK" and not eu_zone_pair(isoF, isoH):
            continue
        # EU zone cross‑border allows all three modes
        edges_FC_WH.append((f, h, mode))
        lt = truck_leadtime(dkm)
        COST_FC_WH[f, h, mode] = dkm * TRUCK_USD_KM
        CO2_FC_WH [f, h, mode] = dkm * TRUCK_CO2_KM
        LT_FC_WH  [f, h, mode] = lt if mode == "TRUCK" else math.ceil(
            lt * _load_csv("transport_mode_meta.csv"
                ).set_index("mode").loc[mode, "leadtime_factor"] - 1e-9)
    # border fee
    BORDER_FC_WH[f, h] = 0 if isoF == isoH or eu_zone_pair(isoF, isoH) else 4_000

edges_WH_CT: list[tuple[str,str]] = []
LT_WH_CT, COST_WH_CT, CO2_WH_CT = {}, {}, {}
for w, c in itertools.product(WAREHOUSES, CITIES):
    if iso_site[w] != iso_city[c]:
        continue
    dkm = haversine(
        (site_coord[w]["lat"], site_coord[w]["lon"]),
        (city_coord[c]["lat"], city_coord[c]["lon"])
    )
    edges_WH_CT.append((w, c))
    LT_WH_CT[w, c]   = truck_leadtime(dkm)
    COST_WH_CT[w, c] = dkm * TRUCK_USD_KM
    CO2_WH_CT[w, c]  = dkm * TRUCK_CO2_KM

# ════════════════════ DEMAND DICT (date, sku, city) → int ══════════════════
demand_full["qty"] = demand_full["demand"].astype(int)
DEMAND_DICT = {
    (row.date.date(), row.sku, row.city): row.qty
    for row in demand_full.itertuples(index=False)
}

# ════════════════════ WEATHER / OIL / FX PREP ══════════════════════════════
BAD_WEATHER_DATES = set(
    weather.query("rain_mm >= 45.7 or snow_mm >= 3.85 "
                  "or wind_speed_max >= 13.46 or cloud_cover >= 100")["date"].dt.date
)

# weekly oil surge
oil_price["week"] = oil_price["date"].dt.to_period("W-MON")
HIGH_OIL_WEEKS = set(
    oil_price.groupby("week")["brent_usd"].mean().pct_change().loc[lambda s: s > .05].index
)

# currency forward‑fill
currency.sort_values("date", inplace=True)
currency.ffill(inplace=True)
CUR2ISO = {
    "USD": ["USA"], "EUR": ["DEU", "FRA"], "KRW": ["KOR"], "JPY": ["JPN"],
    "GBP": ["GBR"], "CAD": ["CAN"], "AUD": ["AUS"], "BRL": ["BRA"], "ZAR": ["ZAF"]
}
FX_RATE: dict[tuple[dt.date, str], float] = {}
for cur, iso_list in CUR2ISO.items():
    col = next(c for c in currency.columns if c.startswith(cur))
    for d, v in currency[["date", col]].itertuples(index=False):
        for iso in iso_list:
            FX_RATE[(d.date(), iso)] = v

# ════════════════════ MACHINE FAILURE LOOK‑UP ═══════════════════════════════
FAIL_LOOKUP: dict[tuple[str, dt.date], bool] = defaultdict(bool)
for row in machine_fail.itertuples(index=False):
    rng = daterange(row.start_date.date(), row.end_date.date())
    for d in rng:
        FAIL_LOOKUP[(row.factory, d)] = True

# ════════════════════ PUBLIC EXPORTS ═══════════════════════════════════════
__all__ = [
    # raw frames
    "site", "site_cost", "cap_week", "lab_req", "lab_pol", "prod_cost",
    "inv_cost", "short_cost", "carbon_f", "sku_meta", "weather",
    "oil_price", "holiday",
    # master sets
    "FACTORIES", "WAREHOUSES", "CITIES", "SKUS",
    "life_weeks",
    # geo dicts
    "site_coord", "city_coord", "iso_site", "iso_city",
    # distance / leadtime / cost dicts
    "edges_FC_WH", "LT_FC_WH", "COST_FC_WH", "CO2_FC_WH", "BORDER_FC_WH",
    "edges_WH_CT", "LT_WH_CT", "COST_WH_CT", "CO2_WH_CT",
    # time‑series derived
    "BAD_WEATHER_DATES", "HIGH_OIL_WEEKS",
    "FX_RATE", "FAIL_LOOKUP",
    # demand
    "DEMAND_DICT"
]
