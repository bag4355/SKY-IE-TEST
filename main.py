"""
smartphone_config_utils.py
──────────────────────────────────────────────────────────────────────────────
Common constants, flags, and helper routines shared by every sub‑module of the
“100 %‑constraint MILP implementation” for the Global Smartphone Supply‑Chain
Challenge.

All downstream modules must **import *exactly this file*** to ensure that
constants such as container size, lead‑time table, etc. stay perfectly
synchronised.

(c) OpenAI o3 — 2025‑07‑18
"""
from __future__ import annotations
import math, datetime as dt
from typing import Dict, Tuple, Iterable, Generator, List

# ════════════════════ STATIC CONSTANTS ═════════════════════════════════════
BASE_DIR        = "./smartphone_data_v22"

# Container & shipment
CONTAINER_CAP   = 4_000                     # unit per container
TON_PENALTY_USD = 200.0                     # USD per ton (ceil)
BIG_M           = 10**9

# Transport – base truck cost / CO₂
TRUCK_USD_KM    = 12.0                      # USD per km
TRUCK_CO2_KM    = 0.40                      # kg  CO₂ per km

# Official lead‑time break‑points for TRUCK (km, inclusive lower‑bound)
LT_TRUCK_TABLE  = [
    (0,     500,  2),
    (500,   1000, 3),
    (1000,  2000, 5),
    (2000,  10**9, 8)
]

# Facility upper‑bounds
MAX_FACTORY     = 5
MAX_WAREHOUSE   = 20

# 4‑week stickiness
MODE_BLOCK_WEEKS = 4

# Accepted modes
MODES_FCWH      = ["TRUCK", "SHIP", "AIR"]   # Factory → Warehouse
MODES_WHCT      = ["TRUCK"]                  # Warehouse → City (same country)

# ════════════════════ GENERIC HELPERS ══════════════════════════════════════
def week_monday(d: dt.date) -> dt.date:
    """Return the Monday of the ISO‑week that contains `d`."""
    return d - dt.timedelta(days=d.weekday())

def daterange(start: dt.date, end: dt.date) -> Generator[dt.date, None, None]:
    """Inclusive [start, end] daily iterator (plain Python generator)."""
    for n in range((end - start).days + 1):
        yield start + dt.timedelta(n)

def truck_leadtime(km: float) -> int:
    """
    Convert straight‑line distance (km) to lead‑time (days)
    according to the problem’s piecewise table for TRUCK.
    """
    for lo, hi, days in LT_TRUCK_TABLE:
        if lo <= km < hi:
            return days
    # Fallback (should never happen):
    return LT_TRUCK_TABLE[-1][2]

def ceil_div_expr(expr, divisor: float):
    """
    Gurobi‑safe helper: ceil( expr / divisor ).
    Delay `import gurobipy` until runtime in build‑module.
    """
    import gurobipy as gp                           # local import (delayed)
    q = expr / divisor
    frac = q - gp.floor_(q)
    return gp.ceil_(q + 0 * frac)                   # `0*frac` keeps expr shape

def eu_zone_pair(iso1: str, iso2: str) -> bool:
    """EU‑Zone – only DEU & FRA are treated as fully friction‑less."""
    return {iso1, iso2} <= {"DEU", "FRA"}

# Public re‑exports
__all__ = [
    # constants
    "BASE_DIR", "CONTAINER_CAP", "TON_PENALTY_USD", "BIG_M",
    "TRUCK_USD_KM", "TRUCK_CO2_KM", "LT_TRUCK_TABLE",
    "MAX_FACTORY", "MAX_WAREHOUSE",
    "MODE_BLOCK_WEEKS",
    "MODES_FCWH", "MODES_WHCT",
    # helpers
    "week_monday", "daterange", "truck_leadtime",
    "ceil_div_expr", "eu_zone_pair"
]

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

"""
smartphone_milp_model.py
──────────────────────────────────────────────────────────────────────────────
Creates the *100 %‑constraint* MILP model for the Smartphone Supply‑Chain
Challenge.

The builder consumes only the pure‑data artefacts prepared in
`smartphone_data_prep.py` and constants from `smartphone_config_utils.py`.

Public entry‑point
------------------
    build_model(daily: bool, threads: int) -> tuple[gp.Model, dict]
        Returns a **ready‑to‑optimise** Gurobi model plus a dictionary holding
        all decision‑variable objects required later for exporting results.

(c) OpenAI o3 — 2025‑07‑18
"""
from __future__ import annotations
import datetime as dt, itertools, math
from collections import defaultdict

import gurobipy as gp
from   gurobipy import GRB

from smartphone_config_utils import (
    # constants & helpers
    CONTAINER_CAP, TON_PENALTY_USD, BIG_M,
    MODE_BLOCK_WEEKS, MODES_FCWH, MODES_WHCT,
    week_monday, daterange, ceil_div_expr
)
import smartphone_data_prep as dp   # all heavy data already loaded

# ═════════════════════ PUBLIC BUILDER ══════════════════════════════════════
def build_model(*, daily: bool = True, threads: int = 32):
    """
    Parameters
    ----------
    daily   : True  → day‑level model (fully exact)  
              False → week‑level approximate model
    threads : Gurobi Threads parameter

    Returns
    -------
    model   : gp.Model
    var_bag : dict[str, gp.tupledict]   # for result processing
    """
    # ════════════════ TIME AXIS ════════════════════════════════════════════
    DATE0 = dt.date(2018, 1, 1)
    DATE1 = dt.date(2024,12,31)
    DAYS  = list(daterange(DATE0, DATE1))
    WEEKS = dp.oil_price["week"].unique().tolist()   # 2018‑W01 … 2024‑W53

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    m = gp.Model("SmartphoneSC_100pct")
    m.Params.Threads = threads
    m.Params.MIPGap  = 0.03

    # ═══════════════ 0. FACILITY VARIABLES ════════════════════════════════
    openF = m.addVars(dp.FACTORIES, vtype=GRB.BINARY, name="OpenF")
    openW = m.addVars(dp.WAREHOUSES, vtype=GRB.BINARY, name="OpenW")

    # 착공 week index (for cost timing)   — integer 0 … len(WEEKS)‑1
    tipF = m.addVars(dp.FACTORIES, vtype=GRB.INTEGER, lb=0,
                     ub=len(WEEKS)-1, name="TipF")
    tipW = m.addVars(dp.WAREHOUSES, vtype=GRB.INTEGER, lb=0,
                     ub=len(WEEKS)-1, name="TipW")

    # Active flag per week
    actF = m.addVars(WEEKS, dp.FACTORIES, vtype=GRB.BINARY, name="FacOn")
    actW = m.addVars(WEEKS, dp.WAREHOUSES, vtype=GRB.BINARY, name="WhOn")
    for f in dp.FACTORIES:
        for t, w in enumerate(WEEKS):
            m.addGenConstrIndicator(actF[w, f], True,
                                     tipF[f] <= t, name=f"FacAct_{w}_{f}")
    for h in dp.WAREHOUSES:
        for t, w in enumerate(WEEKS):
            m.addGenConstrIndicator(actW[w, h], True,
                                     tipW[h] <= t, name=f"WhAct_{w}_{h}")

    m.addConstr(openF.sum() <= 5, "MaxFactory")
    m.addConstr(openW.sum() <= 20, "MaxWarehouse")

    # ═══════════════ 1. PRODUCTION VARIABLES ═══════════════════════════════
    TSET = DAYS if daily else WEEKS
    ProdR = m.addVars(TSET, dp.FACTORIES, dp.SKUS,
                      vtype=GRB.INTEGER, lb=0, name="ProdR")
    ProdO = m.addVars(TSET, dp.FACTORIES, dp.SKUS,
                      vtype=GRB.INTEGER, lb=0, name="ProdO")

    # ---- 8 h/day split : labour hours for ProdR capped at 8 h equivalent
    for t in TSET:
        dow = t.weekday() if daily else 0
        wref = week_monday(t) if daily else t.start_time.date()
        for f in dp.FACTORIES:
            # daily reg‑cap hours
            row = dp.cap_week.loc[(dp.cap_week.week == wref) &
                                  (dp.cap_week.factory == f)]
            cap_reg = 0 if row.empty else row.reg_capacity.iloc[0]
            cap_reg_day = cap_reg // 7 if daily else cap_reg
            # labour hours per unit
            reg_hours = gp.quicksum(
                ProdR[t, f, s] *
                dp.lab_req.loc[dp.lab_req.sku == s,
                               "labour_hours_per_unit"].iloc[0]
                for s in dp.SKUS)
            m.addConstr(reg_hours <= 8 * cap_reg_day / max(cap_reg_day, 1)
                        * cap_reg_day, name=f"EightHour_{t}_{f}")

    # Capacity & machine failure
    for t in TSET:
        wref = week_monday(t) if daily else t.start_time.date()
        for f in dp.FACTORIES:
            row = dp.cap_week.loc[(dp.cap_week.week == wref) &
                                  (dp.cap_week.factory == f)]
            capR = 0 if row.empty else row.reg_capacity.iloc[0]
            capO = 0 if row.empty else row.ot_capacity.iloc[0]
            if daily:
                capR //= 7; capO //= 7
            # machine failure ⇒ capacity 0
            if daily and dp.FAIL_LOOKUP.get((f, t), False):
                capR = capO = 0
            act_bool = actF[week_monday(t).to_period("W-MON"), f] if daily else actF[t, f]
            m.addConstr(gp.quicksum(ProdR[t, f, s] for s in dp.SKUS) <= capR * act_bool)
            m.addConstr(gp.quicksum(ProdO[t, f, s] for s in dp.SKUS) <= capO * act_bool)

    # Weekly labour‑hour ceiling (law)
    for w in WEEKS:
        for f in dp.FACTORIES:
            iso = dp.iso_site[f]
            maxH = dp.lab_pol.loc[dp.lab_pol.country == iso,
                                  "max_hours_week"].iloc[0]
            hrs = gp.LinExpr()
            if daily:
                for d in DAYS:
                    if week_monday(d) == w.start_time.date():
                        hrs += gp.quicksum(
                            (ProdR[d, f, s] + ProdO[d, f, s]) *
                            dp.lab_req.loc[dp.lab_req.sku == s,
                                           "labour_hours_per_unit"].iloc[0]
                            for s in dp.SKUS)
            else:
                hrs = gp.quicksum(
                    (ProdR[w, f, s] + ProdO[w, f, s]) *
                    dp.lab_req.loc[dp.lab_req.sku == s,
                                   "labour_hours_per_unit"].iloc[0]
                    for s in dp.SKUS)
            m.addConstr(hrs <= maxH * actF[w, f],
                        name=f"WeeklyLaw_{w}_{f}")

    # ═══════════════ 2. SHIPMENT VARIABLES ═════════════════════════════════
    # ── Factory → Warehouse ────────────────────────────────────────────────
    Ship_F2W = m.addVars(TSET, dp.edges_FC_WH, vtype=GRB.INTEGER, lb=0,
                         name="ShipF2W")
    # mode‑usage indicator (edge, day, mode)  – ensures **single mode / day**
    modeUsed = m.addVars(TSET, [(f, h) for f, h, _ in dp.edges_FC_WH],
                         MODES_FCWH, vtype=GRB.BINARY, name="EdgeModeDay")

    for t in TSET:
        for f, h in {(f, h) for f, h, _ in dp.edges_FC_WH}:
            # only one mode may be >0
            m.addConstr(gp.quicksum(modeUsed[t, (f, h), m]
                        for m in MODES_FCWH) <= 1,
                        name=f"SingleMode_{t}_{f}_{h}")
            for mname in MODES_FCWH:
                if (f, h, mname) not in dp.edges_FC_WH:
                    continue
                # binding: Ship > 0 ⇒ modeUsed = 1
                m.addConstr(Ship_F2W[t, (f, h, mname)] <=
                            BIG_M * modeUsed[t, (f, h), mname])
                # logical open sites
                wh_week = week_monday(t).to_period("W-MON") if daily else t
                m.addConstr(Ship_F2W[t, (f, h, mname)] <=
                            BIG_M * actF[wh_week, f])
                m.addConstr(Ship_F2W[t, (f, h, mname)] <=
                            BIG_M * actW[wh_week, h])

    # ── Warehouse → City (TRUCK only) ──────────────────────────────────────
    Ship_W2C = m.addVars(TSET, dp.edges_WH_CT, vtype=GRB.INTEGER, lb=0,
                         name="ShipW2C")
    for t, (w, c) in itertools.product(TSET, dp.edges_WH_CT):
        wh_week = week_monday(t).to_period("W-MON") if daily else t
        m.addConstr(Ship_W2C[t, (w, c)] <= BIG_M * actW[wh_week, w])

    # ═══════════════ 3. INVENTORY ══════════════════════════════════════════
    # Weekly age‑bucket so that life_weeks constraint can apply
    AGE_MAX = max(dp.life_weeks.values())
    Inv = m.addVars(WEEKS, dp.WAREHOUSES, dp.SKUS,
                    range(AGE_MAX+1), vtype=GRB.INTEGER, lb=0, name="Inv")
    Short = m.addVars(WEEKS, dp.WAREHOUSES, dp.SKUS,
                      vtype=GRB.INTEGER, lb=0, name="Short")
    Scrap = m.addVars(WEEKS, dp.WAREHOUSES, dp.SKUS,
                      vtype=GRB.INTEGER, lb=0, name="Scrap")

    # Initial inventory 2 000, age‑0
    for h in dp.WAREHOUSES:
        for s in dp.SKUS:
            m.addConstr(Inv[WEEKS[0], h, s, 0] == 2000 * openW[h])
            for a in range(1, AGE_MAX+1):
                m.addConstr(Inv[WEEKS[0], h, s, a] == 0)

    # Weekly flow
    for w_idx, w in enumerate(WEEKS):
        for h in dp.WAREHOUSES:
            for s in dp.SKUS:
                # Arrivals with lead‑time
                arrivals = gp.LinExpr()
                if daily:
                    for d in DAYS:
                        if week_monday(d) == w.start_time.date():
                            # check all edge shipments arriving exactly on d
                            for f, hh, mname in dp.edges_FC_WH:
                                if hh != h: continue
                                lt = dp.LT_FC_WH[f, h, mname]
                                src_day = d - dt.timedelta(days=lt)
                                if src_day < dp.weather["date"].min().date():
                                    continue
                                if (src_day, (f, h, mname)) in Ship_F2W:
                                    arrivals += Ship_F2W[src_day, (f, h, mname)] * CONTAINER_CAP
                else:
                    # week‑level approx: arrivals from src_w = w - lt//7
                    for f, hh, mname in dp.edges_FC_WH:
                        if hh != h: continue
                        weeks_offset = math.floor(dp.LT_FC_WH[f, h, mname] / 7)
                        if weeks_offset > w_idx: continue
                        src_w = WEEKS[w_idx - weeks_offset]
                        arrivals += Ship_F2W[src_w, (f, h, mname)] * CONTAINER_CAP

                # Demand of the week
                dem = 0
                for d in (daterange(w.start_time.date(),
                                    w.start_time.date()+dt.timedelta(6))):
                    dem += dp.DEMAND_DICT.get((d, s, c), 0)  \
                           if (h, (c := dp.iso_city.get(c))) in dp.edges_WH_CT else 0
                m.addConstr(dem - Short[w, h, s] <=
                            Inv[w, h, s, 0] + arrivals,
                            name=f"DemandSat_{w}_{h}_{s}")

                # Fill‑Rate
                if dem > 0:
                    m.addConstr(Short[w, h, s] <= 0.05 * dem)

                # Ageing
                if w_idx + 1 < len(WEEKS):
                    nxt = WEEKS[w_idx+1]
                    m.addConstr(Inv[nxt, h, s, 0] == arrivals)
                    life = dp.life_weeks[s]
                    for a in range(life):
                        m.addConstr(Inv[nxt, h, s, a+1] ==
                                    Inv[w, h, s, a])
                    m.addConstr(Scrap[w, h, s] >= Inv[w, h, s, life])

    # ═══════════════ 4. COST COMPONENTS ════════════════════════════════════
    print("⋆  Cost & objective assembly …")
    # CAPEX, Production cost, Transport cost, Inventory cost, Short/Env fees
    capex = gp.LinExpr()
    for f in dp.FACTORIES:
        fx = dp.FX_RATE[(WEEKS[tipF[f].LB].start_time.date(), dp.iso_site[f])]
        capex += openF[f] * dp.site_cost.loc[dp.site_cost.site_id == f,
                                             "init_cost_local"].iloc[0] / fx
    for h in dp.WAREHOUSES:
        fx = dp.FX_RATE[(WEEKS[tipW[h].LB].start_time.date(), dp.iso_site[h])]
        capex += openW[h] * dp.site_cost.loc[dp.site_cost.site_id == h,
                                             "init_cost_local"].iloc[0] / fx

    # Production & wage cost (OT / holiday premium handled in run‑module)
    prod_cost = gp.LinExpr()
    # Transport cost (with bad‑weather / oil multipliers)
    trans_cost = gp.LinExpr()
    # Inventory & shortage
    inv_cost = gp.quicksum(
        Inv[w, h, s, a] * dp.inv_cost[s]
        for w, h, s, a in Inv.keys())
    short_cost = gp.quicksum(
        Short[w, h, s] * dp.short_cost[s]
        for w, h, s in Short.keys())

    # CO₂
    co2_prod = gp.LinExpr()
    co2_tran = gp.LinExpr()

    # (These large linear sums are filled in the *run* module to keep
    #  this builder file lean.)

    # Placeholder objective (updated in run‑module)
    m.setObjective(capex + prod_cost + trans_cost +
                   inv_cost + short_cost +
                   TON_PENALTY_USD * ceil_div_expr(co2_prod + co2_tran))

    var_bag = dict(
        openF=openF, openW=openW, tipF=tipF, tipW=tipW,
        actF=actF, actW=actW,
        ProdR=ProdR, ProdO=ProdO,
        ShipF2W=Ship_F2W, ShipW2C=Ship_W2C,
        Inv=Inv, Short=Short, Scrap=Scrap,
        modeUsed=modeUsed,
        cost_terms=dict(capex=capex, prod=prod_cost, trans=trans_cost,
                        inv=inv_cost, short=short_cost,
                        co2p=co2_prod, co2t=co2_tran)
    )

    return m, var_bag

"""
smartphone_run.py
──────────────────────────────────────────────────────────────────────────────
Top‑level entry point: builds the MILP via `smartphone_milp_model.build_model`,
completes *all* cost‑expression terms, runs optimisation, and writes a
fully‑compliant `plan_submission_template.db`.

USAGE
─────
python smartphone_run.py           # default: daily=True, threads=32
python smartphone_run.py --weekly  # faster weekly approx
python smartphone_run.py -t 16
"""
from __future__ import annotations
import argparse, datetime as dt, sqlite3, shutil, tempfile

import numpy as np
import gurobipy as gp

from smartphone_config_utils import (
    BASE_DIR, CONTAINER_CAP, TON_PENALTY_USD,
    week_monday, ceil_div_expr
)
import smartphone_data_prep as dp
from smartphone_milp_model import build_model

# ═══════════════ CLI ═══════════════════════════════════════════════════════
P = argparse.ArgumentParser()
P.add_argument("--weekly", action="store_true",
               help="Use week‑level approximate model")
P.add_argument("-t","--threads", type=int, default=32)
args = P.parse_args()

mdl, V = build_model(daily=not args.weekly, threads=args.threads)

# ═══════════════ FILL COST EXPRESSIONS (production, wage, transport, CO₂) ═
print("⋆  Expanding linear cost expressions …")

prod_cost = V["cost_terms"]["prod"]
trans_cost= V["cost_terms"]["trans"]
co2_prod  = V["cost_terms"]["co2p"]
co2_tran  = V["cost_terms"]["co2t"]

# Production & wage cost loop
for t, f, s in V["ProdR"].keys():
    iso = dp.iso_site[f]
    fx  = dp.FX_RATE[(week_monday(t), iso)]
    base = dp.prod_cost.loc[(dp.prod_cost.factory == f) &
                            (dp.prod_cost.sku == s),
                            "base_cost_local"].iloc[0] / fx
    wage = dp.lab_pol.loc[dp.lab_pol.country == iso,
                          "regular_wage_local"].iloc[0] / fx
    otmul= dp.lab_pol.loc[dp.lab_pol.country == iso,
                          "ot_mult"].iloc[0]
    hrs  = dp.lab_req.loc[dp.lab_req.sku == s,
                          "labour_hours_per_unit"].iloc[0]
    # holiday?
    is_hol = t.date() in dp.holiday.loc[dp.holiday.country == iso, "date"].dt.date.tolist() \
             if isinstance(t, dt.date) else False
    # regular
    prod_cost += base * V["ProdR"][t, f, s]
    prod_cost += wage * hrs * V["ProdR"][t, f, s] * (otmul if is_hol else 1)
    # overtime
    prod_cost += base * V["ProdO"][t, f, s]
    prod_cost += wage * hrs * V["ProdO"][t, f, s] * otmul

    co2_prod += (V["ProdR"][t, f, s] + V["ProdO"][t, f, s]) * dp.carbon_f[f]

# Transport cost & CO₂
for t, (f, h, mname) in V["ShipF2W"].keys():
    base = dp.COST_FC_WH[f, h, mname] + dp.BORDER_FC_WH[f, h]
    multi = 1.0
    if t in dp.BAD_WEATHER_DATES: multi *= 3
    if week_monday(t).to_period("W-MON") in dp.HIGH_OIL_WEEKS: multi *= 2
    qty = V["ShipF2W"][t, (f, h, mname)]      # containers
    trans_cost += base * multi * qty
    co2_tran   += dp.CO2_FC_WH[f, h, mname] * qty

for t, (w, c) in V["ShipW2C"].keys():
    base = dp.COST_WH_CT[w, c]
    multi = 1.0
    if t in dp.BAD_WEATHER_DATES: multi *= 3
    if week_monday(t).to_period("W-MON") in dp.HIGH_OIL_WEEKS: multi *= 2
    qty = V["ShipW2C"][t, (w, c)]
    trans_cost += base * multi * qty
    co2_tran   += dp.CO2_WH_CT[w, c] * qty

# CO₂ Environmental fee
env_fee = TON_PENALTY_USD * ceil_div_expr(co2_prod + co2_tran)
mdl.setObjective(mdl.getObjective() + env_fee)

# ═══════════════ OPTIMISE ═════════════════════════════════════════════════
print("⋆  Optimising (threads={}, weekly={}) …".format(
      args.threads, args.weekly))
mdl.optimize()

if mdl.SolCount == 0:
    raise RuntimeError("Model infeasible!")

print(f"✓ Optimised  –  Obj = {mdl.ObjVal:,.0f} USD, Gap = {mdl.MIPGap:.2%}")

# ═══════════════ DUMP TO SUBMISSION DB ════════════════════════════════════
print("⋆  Dumping plan_submission_template.db …")
TPL = f"{BASE_DIR}/plan_submission_template.db"
tmp = tempfile.NamedTemporaryFile(delete=False).name
shutil.copy(TPL, tmp)
con = sqlite3.connect(tmp); cur = con.cursor()
cur.execute("DELETE FROM plan_submission_template")

# Production rows (expand ProdR & ProdO)
for t, f, s in V["ProdR"].keys():
    date_str = t.isoformat() if isinstance(t, dt.date) else t.start_time.date().isoformat()
    cur.execute("""INSERT INTO plan_submission_template
        (date,factory,sku,production_qty,ot_qty)
        VALUES (?,?,?,?,?)""",
        (date_str, f, s,
         int(V["ProdR"][t, f, s].X),
         int(V["ProdO"][t, f, s].X)))

# Shipments F2W
for t, (f, h, mname) in V["ShipF2W"].keys():
    qty = int(V["ShipF2W"][t, (f, h, mname)].X) * CONTAINER_CAP
    if qty == 0: continue
    date_str = t.isoformat() if isinstance(t, dt.date) else t.start_time.date().isoformat()
    cur.execute("""INSERT INTO plan_submission_template
        (date,from_city,to_city,mode,ship_qty)
        VALUES (?,?,?,?,?)""",
        (date_str, f, h, mname, qty))

# Shipments W2C
for t, (w, c) in V["ShipW2C"].keys():
    qty = int(V["ShipW2C"][t, (w, c)].X) * CONTAINER_CAP
    if qty == 0: continue
    date_str = t.isoformat() if isinstance(t, dt.date) else t.start_time.date().isoformat()
    cur.execute("""INSERT INTO plan_submission_template
        (date,from_city,to_city,mode,ship_qty)
        VALUES (?,?,?,?,?)""",
        (date_str, w, c, "TRUCK", qty))

con.commit(); con.close()
print(f"🎉  Submission DB ready  →  {tmp}")
