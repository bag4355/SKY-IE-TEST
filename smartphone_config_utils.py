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
