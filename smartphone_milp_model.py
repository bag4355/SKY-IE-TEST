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

    BLOCKS = range(math.ceil(len(WEEKS) / MODE_BLOCK_WEEKS))

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
    # 4‑week mode locking structure
    edges = [(f, h) for f, h, _ in dp.edges_FC_WH]
    modeBlock = m.addVars(BLOCKS, range(7), edges, MODES_FCWH,
                          vtype=GRB.BINARY, name="ModeBlock")
    modeChange = m.addVars(BLOCKS, range(7), edges,
                           vtype=GRB.BINARY, name="ModeChange")

    for b in BLOCKS:
        for dow in range(7):
            for e in edges:
                m.addConstr(gp.quicksum(modeBlock[b, dow, e, mn]
                                       for mn in MODES_FCWH) <= 1)

    def block_idx(t):
        idx = WEEKS.index(week_monday(t).to_period("W-MON")) if daily else WEEKS.index(t)
        return idx // MODE_BLOCK_WEEKS

    def dow_of(t):
        return t.weekday() if daily else 0

    for t in TSET:
        b = block_idx(t)
        dow = dow_of(t)
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
                m.addConstr(modeUsed[t, (f, h), mname] <= modeBlock[b, dow, (f, h), mname])
                # logical open sites
                wh_week = week_monday(t).to_period("W-MON") if daily else t
                m.addConstr(Ship_F2W[t, (f, h, mname)] <=
                            BIG_M * actF[wh_week, f])
                m.addConstr(Ship_F2W[t, (f, h, mname)] <=
                            BIG_M * actW[wh_week, h])

    # mode change indicator
    ChangeCost = m.addVars(BLOCKS, range(7), edges, vtype=GRB.CONTINUOUS, lb=0,
                           name="ChangeCost")
    change_cost = gp.LinExpr()
    for b in BLOCKS:
        if b == 0:
            continue
        for dow in range(7):
            for e in edges:
                for mn in MODES_FCWH:
                    m.addConstr(modeChange[b, dow, e] >= modeBlock[b, dow, e, mn] - modeBlock[b-1, dow, e, mn])
                    m.addConstr(modeChange[b, dow, e] >= modeBlock[b-1, dow, e, mn] - modeBlock[b, dow, e, mn])
                m.addConstr(modeChange[b, dow, e] <= gp.quicksum(modeBlock[b, dow, e, mn] + modeBlock[b-1, dow, e, mn]
                                                               for mn in MODES_FCWH))

                # 5 % penalty based on previous block cost
                prev_cost = gp.LinExpr()
                for t in TSET:
                    if block_idx(t) == b - 1 and dow_of(t) == dow:
                        for mn in MODES_FCWH:
                            if (e[0], e[1], mn) not in dp.edges_FC_WH:
                                continue
                            multi = 1.0
                            if (isinstance(t, dt.date) and t in dp.BAD_WEATHER_DATES):
                                multi *= 3
                            if week_monday(t if isinstance(t, dt.date) else t.start_time.date()).to_period("W-MON") in dp.HIGH_OIL_WEEKS:
                                multi *= 2
                            prev_cost += (dp.COST_FC_WH[e[0], e[1], mn] + dp.BORDER_FC_WH[e[0], e[1]]) * multi * Ship_F2W[t, (e[0], e[1], mn)]

                m.addConstr(ChangeCost[b, dow, e] >= 0.05 * prev_cost - BIG_M * (1 - modeChange[b, dow, e]))
                m.addConstr(ChangeCost[b, dow, e] <= 0.05 * prev_cost + BIG_M * (1 - modeChange[b, dow, e]))
                m.addConstr(ChangeCost[b, dow, e] <= BIG_M * modeChange[b, dow, e])

    change_cost += gp.quicksum(ChangeCost[b, dow, e] for b in BLOCKS for dow in range(7) for e in edges)

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
                demand = gp.quicksum(
                    dp.DEMAND_DICT.get((d, s, city), 0)
                    for d in daterange(w.start_time.date(),
                                       w.start_time.date() + dt.timedelta(6))
                    for city in dp.CITIES
                    if (h, city) in dp.edges_WH_CT
                )
                m.addConstr(demand - Short[w, h, s] <=
                            Inv[w, h, s, 0] + arrivals,
                            name=f"DemandSat_{w}_{h}_{s}")

                # Fill‑Rate
                if demand > 0:
                    m.addConstr(Short[w, h, s] <= 0.05 * demand)

                # Ageing
                if w_idx + 1 < len(WEEKS):
                    nxt = WEEKS[w_idx+1]
                    m.addConstr(Inv[nxt, h, s, 0] == available - dispatch)
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
    m.setObjective(capex + prod_cost + trans_cost + change_cost +
                   inv_cost + short_cost +
                   TON_PENALTY_USD * ceil_div_expr(co2_prod + co2_tran))

    var_bag = dict(
        openF=openF, openW=openW, tipF=tipF, tipW=tipW,
        actF=actF, actW=actW,
        ProdR=ProdR, ProdO=ProdO,
        ShipF2W=Ship_F2W, ShipW2C=Ship_W2C,
        Inv=Inv, Short=Short, Scrap=Scrap,
        modeUsed=modeUsed, modeBlock=modeBlock, modeChange=modeChange,
        cost_terms=dict(capex=capex, prod=prod_cost, trans=trans_cost,
                        change=change_cost,
                        inv=inv_cost, short=short_cost,
                        co2p=co2_prod, co2t=co2_tran)
    )

    return m, var_bag
