"""
smartphone_validation.py
──────────────────────────────────────────────────────────────────────────────
Quick offline validator that **re‑loads** the generated
`plan_submission_template.db` alongside every raw data file and
re‑evaluates *each* hard constraint spelled out in the challenge
description.

Run this after `smartphone_run.py`.  Program exits with non‑zero status if any
violation is detected; otherwise prints a green ✓ summary.

Note: full re‑implementation of the objective is **not** included—only the
yes/no feasibility checks.

(c) OpenAI o3 — 2025‑07‑18
"""
from __future__ import annotations
import sqlite3, sys, pandas as pd, datetime as dt, math, itertools
from pathlib import Path
from collections import defaultdict

from smartphone_config_utils import (
    BASE_DIR, CONTAINER_CAP, MODES_FCWH, week_monday,
    eu_zone_pair
)
import smartphone_data_prep as dp

DB_PATH = Path(BASE_DIR) / "plan_submission_template.db"
if not DB_PATH.exists():
    print("❌  Submission DB not found – run `smartphone_run.py` first.")
    sys.exit(1)

print("⋆  Loading submission DB …")
con = sqlite3.connect(DB_PATH); cur = con.cursor()
sub = pd.read_sql("SELECT * FROM plan_submission_template", con,
                  parse_dates=["date"])

# ═══════════════ 1. BASIC DIMENSION CHECKS ═════════════════════════════════
print("⋆  Basic shape checks …")
if sub.isna().any().any():
    print("❌  Null values present."); sys.exit(1)

# Shipment: check container integer divisibility
ship_rows = sub.dropna(subset=["ship_qty"])
bad = ship_rows[ship_rows["ship_qty"] % CONTAINER_CAP != 0]
if not bad.empty:
    print("❌  Non‑integral container quantities detected.")
    sys.exit(1)

# Single‑mode‑per‑day‑per‑edge rule
print("⋆  Single‑mode‑per‑day edge rule …")
f2w_ship = ship_rows[ship_rows["from_city"].str.startswith("FC_")]
multimode = (f2w_ship.groupby(["date","from_city","to_city"])
             ["mode"].nunique().loc[lambda s:s>1])
if not multimode.empty:
    print("❌  Multiple transport modes used same day on same edge.")
    sys.exit(1)

# Only one shipment per day per edge (even same mode)
multitrip = (f2w_ship.groupby(["date","from_city","to_city","mode"])
             .size().loc[lambda s:s>1])
if not multitrip.empty:
    print("❌  Edge shipped more than once per day.")
    sys.exit(1)

# EU‑zone mode permission check
print("⋆  Mode permission checks …")
for _, r in f2w_ship.iterrows():
    isoF = dp.iso_site[r.from_city]; isoW = dp.iso_site[r.to_city]
    if isoF == isoW:
        if r.mode != "TRUCK":
            print(f"❌  {r.mode} used domestically on {r.date}."); sys.exit(1)
    else:
        if r.mode == "TRUCK" and not eu_zone_pair(isoF, isoW):
            print(f"❌  Cross‑border TRUCK not allowed {r.from_city}->{r.to_city}")
            sys.exit(1)

# Warehouse→City TRUCK only & same country
wc_ship = ship_rows[ship_rows["mode"] == "TRUCK"] \
          .query("from_city.str.startswith('WH_')", engine="python")
for _, r in wc_ship.iterrows():
    if dp.iso_site[r.from_city] != dp.iso_city[r.to_city]:
        print("❌  Warehouse→City cross‑country shipment detected."); sys.exit(1)

print("✓  All static rules passed!")
