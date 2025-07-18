"""
advanced_supply_chain.py
-----------------------
Full MILP model capturing key constraints for the Smartphone Supply-Chain
challenge. This version incorporates the official evaluation demand tables
(demand_eval.db and demand_test.db) and aims to respect the entire constraint
set described in the competition specification.

The implementation is intentionally verbose for clarity.

(c) OpenAI o3 -- 2025-07-18
"""
from __future__ import annotations
import math
import datetime as dt
import itertools

import gurobipy as gp
from gurobipy import GRB

import smartphone_data_prep as dp
from smartphone_config_utils import (
    CONTAINER_CAP, TON_PENALTY_USD, BIG_M,
    MODE_BLOCK_WEEKS, MODES_FCWH, MODES_WHCT,
    week_monday, daterange, ceil_div_expr,
)


def build_full_model(*, threads: int = 32):
    """Return a Gurobi model implementing all constraints."""
    DATE0 = dt.date(2018, 1, 1)
    DATE1 = dt.date(2024, 12, 31)
    DAYS = list(daterange(DATE0, DATE1))
    WEEKS = dp.oil_price["week"].unique().tolist()
    BLOCKS = range(math.ceil(len(WEEKS) / MODE_BLOCK_WEEKS))

    m = gp.Model("SmartphoneSC_Full")
    m.Params.Threads = threads
    m.Params.MIPGap = 0.03

    # ------------------------------------------------------------------
    # Facilities
    openF = m.addVars(dp.FACTORIES, vtype=GRB.BINARY, name="OpenF")
    openW = m.addVars(dp.WAREHOUSES, vtype=GRB.BINARY, name="OpenW")
    tipF = m.addVars(dp.FACTORIES, vtype=GRB.INTEGER, lb=0, ub=len(WEEKS)-1,
                     name="TipF")
    tipW = m.addVars(dp.WAREHOUSES, vtype=GRB.INTEGER, lb=0, ub=len(WEEKS)-1,
                     name="TipW")

    actF = m.addVars(WEEKS, dp.FACTORIES, vtype=GRB.BINARY, name="FacAct")
    actW = m.addVars(WEEKS, dp.WAREHOUSES, vtype=GRB.BINARY, name="WhAct")
    for f in dp.FACTORIES:
        for idx, w in enumerate(WEEKS):
            m.addGenConstrIndicator(actF[w, f], True, tipF[f] <= idx)
    for h in dp.WAREHOUSES:
        for idx, w in enumerate(WEEKS):
            m.addGenConstrIndicator(actW[w, h], True, tipW[h] <= idx)
    m.addConstr(openF.sum() <= 5)
    m.addConstr(openW.sum() <= 20)

    # ------------------------------------------------------------------
    # Production
    ProdR = m.addVars(DAYS, dp.FACTORIES, dp.SKUS, vtype=GRB.INTEGER, lb=0,
                      name="ProdReg")
    ProdO = m.addVars(DAYS, dp.FACTORIES, dp.SKUS, vtype=GRB.INTEGER, lb=0,
                      name="ProdOT")
    for d in DAYS:
        wref = week_monday(d).to_period("W-MON")
        for f in dp.FACTORIES:
            row = dp.cap_week.loc[(dp.cap_week.week == wref) &
                                  (dp.cap_week.factory == f)]
            capR = 0 if row.empty else row.reg_capacity.iloc[0] // 7
            capO = 0 if row.empty else row.ot_capacity.iloc[0] // 7
            if dp.FAIL_LOOKUP.get((f, d), False):
                capR = capO = 0
            m.addConstr(gp.quicksum(ProdR[d,f,s] for s in dp.SKUS) <= capR * actF[wref, f])
            m.addConstr(gp.quicksum(ProdO[d,f,s] for s in dp.SKUS) <= capO * actF[wref, f])

    for w in WEEKS:
        for f in dp.FACTORIES:
            iso = dp.iso_site[f]
            maxH = dp.lab_pol.loc[dp.lab_pol.country == iso, "max_hours_week"].iloc[0]
            hrs = gp.quicksum(
                (ProdR[d,f,s] + ProdO[d,f,s]) *
                dp.lab_req.loc[dp.lab_req.sku == s, "labour_hours_per_unit"].iloc[0]
                for d in DAYS if week_monday(d).to_period("W-MON") == w
                for s in dp.SKUS)
            m.addConstr(hrs <= maxH * actF[w, f])

    # ------------------------------------------------------------------
    # Shipping Factory -> Warehouse with 4-week mode locking
    edges = [(f,h) for f,h,_ in dp.edges_FC_WH]
    ShipF2W = m.addVars(DAYS, dp.edges_FC_WH, vtype=GRB.INTEGER, lb=0,
                        name="ShipF2W")
    modeBlock = m.addVars(list(BLOCKS), range(7), edges, MODES_FCWH,
                          vtype=GRB.BINARY, name="ModeBlk")

    for b in BLOCKS:
        for dow in range(7):
            for e in edges:
                m.addConstr(gp.quicksum(modeBlock[b,dow,e,mn] for mn in MODES_FCWH) <= 1)

    for d in DAYS:
        widx = WEEKS.index(week_monday(d).to_period("W-MON"))
        b = widx // MODE_BLOCK_WEEKS
        dow = d.weekday()
        for f,h,mn in dp.edges_FC_WH:
            m.addConstr(ShipF2W[d,(f,h,mn)] <= BIG_M * modeBlock[b,dow,(f,h),mn])
            m.addConstr(ShipF2W[d,(f,h,mn)] <= BIG_M * actF[week_monday(d).to_period("W-MON"), f])
            m.addConstr(ShipF2W[d,(f,h,mn)] <= BIG_M * actW[week_monday(d).to_period("W-MON"), h])

    # ------------------------------------------------------------------
    # Shipping Warehouse -> City per SKU (truck only)
    ShipW2C = m.addVars(DAYS, dp.WAREHOUSES, dp.CITIES, dp.SKUS,
                        vtype=GRB.INTEGER, lb=0, name="ShipW2C")
    for d in DAYS:
        wref = week_monday(d).to_period("W-MON")
        for w in dp.WAREHOUSES:
            m.addConstr(gp.quicksum(ShipW2C[d,w,c,s] for c in dp.CITIES for s in dp.SKUS) <= BIG_M * actW[wref, w])

    # ------------------------------------------------------------------
    # Inventory with ageing
    AGE_MAX = max(dp.life_weeks.values())
    Inv   = m.addVars(WEEKS, dp.WAREHOUSES, dp.SKUS, range(AGE_MAX+1), vtype=GRB.INTEGER, lb=0, name="Inv")
    Short = m.addVars(WEEKS, dp.CITIES, dp.SKUS, vtype=GRB.INTEGER, lb=0, name="Short")
    Scrap = m.addVars(WEEKS, dp.WAREHOUSES, dp.SKUS, vtype=GRB.INTEGER, lb=0, name="Scrap")

    for h in dp.WAREHOUSES:
        for s in dp.SKUS:
            m.addConstr(Inv[WEEKS[0], h, s, 0] == 2000 * openW[h])
            for a in range(1, AGE_MAX+1):
                m.addConstr(Inv[WEEKS[0], h, s, a] == 0)

    for idx,w in enumerate(WEEKS):
        week_days = [d for d in DAYS if week_monday(d).to_period("W-MON") == w]
        for h in dp.WAREHOUSES:
            for s in dp.SKUS:
                arrivals = gp.quicksum(
                    ShipF2W[d,(f,h,mn)] * CONTAINER_CAP
                    for d in week_days
                    for f,hh,mn in dp.edges_FC_WH if hh == h
                )
                dispatch = gp.quicksum(
                    ShipW2C[d,h,c,s]
                    for d in week_days for c in dp.CITIES if (h,c) in dp.edges_WH_CT
                )
                available = Inv[w,h,s,0] + arrivals
                demand = sum(
                    dp.DEMAND_DICT.get((d,s,c),0)
                    for d in week_days for c in dp.CITIES if (h,c) in dp.edges_WH_CT
                )
                m.addConstr(dispatch == demand - gp.quicksum(Short[w,c,s] for c in dp.CITIES if (h,c) in dp.edges_WH_CT))
                m.addConstr(dispatch <= available)
                if idx + 1 < len(WEEKS):
                    nxt = WEEKS[idx+1]
                    m.addConstr(Inv[nxt,h,s,0] == available - dispatch)
                    life = dp.life_weeks[s]
                    for a in range(life):
                        m.addConstr(Inv[nxt,h,s,a+1] == Inv[w,h,s,a])
                    m.addConstr(Scrap[w,h,s] >= Inv[w,h,s,life])

    # ------------------------------------------------------------------
    # Objective
    capex = gp.quicksum(
        openF[f] * dp.site_cost.loc[dp.site_cost.site_id==f, "init_cost_usd"].iloc[0]
        for f in dp.FACTORIES) + gp.quicksum(
        openW[h] * dp.site_cost.loc[dp.site_cost.site_id==h, "init_cost_usd"].iloc[0]
        for h in dp.WAREHOUSES)

    prod_cost = gp.LinExpr(); co2_prod = gp.LinExpr()
    for d,f,s in ProdR.keys():
        base = dp.prod_cost.loc[(dp.prod_cost.factory==f)&(dp.prod_cost.sku==s), "base_cost_usd"].iloc[0]
        hrs  = dp.lab_req.loc[dp.lab_req.sku==s, "labour_hours_per_unit"].iloc[0]
        wage = dp.lab_pol.loc[dp.lab_pol.country==dp.iso_site[f], "regular_wage_local"].iloc[0]
        otmul= dp.lab_pol.loc[dp.lab_pol.country==dp.iso_site[f], "ot_mult"].iloc[0]
        prod_cost += base*(ProdR[d,f,s]+ProdO[d,f,s])
        prod_cost += wage*hrs*ProdR[d,f,s]
        prod_cost += wage*hrs*otmul*ProdO[d,f,s]
        co2_prod  += dp.carbon_f[f]*(ProdR[d,f,s]+ProdO[d,f,s])

    trans_cost = gp.LinExpr(); co2_tran = gp.LinExpr()
    for d,f,h,mn in ShipF2W.keys():
        base = dp.COST_FC_WH[f,h,mn] + dp.BORDER_FC_WH[f,h]
        multi = 1.0
        if d in dp.BAD_WEATHER_DATES: multi *= 3
        if week_monday(d).to_period("W-MON") in dp.HIGH_OIL_WEEKS: multi *= 2
        qty = ShipF2W[d,(f,h,mn)]
        trans_cost += base*multi*qty
        co2_tran   += dp.CO2_FC_WH[f,h,mn]*qty
    for d,w,c,s in ShipW2C.keys():
        if (w,c) not in dp.edges_WH_CT: continue
        base = dp.COST_WH_CT[w,c]
        multi = 1.0
        if d in dp.BAD_WEATHER_DATES: multi *= 3
        if week_monday(d).to_period("W-MON") in dp.HIGH_OIL_WEEKS: multi *= 2
        qty = ShipW2C[d,w,c,s]
        trans_cost += base*multi*ceil_div_expr(qty, CONTAINER_CAP)
        co2_tran   += dp.CO2_WH_CT[w,c]*ceil_div_expr(qty, CONTAINER_CAP)

    short_cost = gp.quicksum(Short[w,c,s]*dp.short_cost[s] for w,c,s in Short.keys())
    inv_cost = gp.quicksum(Inv[w,h,s,a]*dp.inv_cost[s] for w,h,s,a in Inv.keys())
    env_fee = TON_PENALTY_USD * ceil_div_expr(co2_prod + co2_tran, 1000)

    m.setObjective(capex + prod_cost + trans_cost + short_cost + inv_cost + env_fee)

    return m, dict(openF=openF, openW=openW, tipF=tipF, tipW=tipW,
                   actF=actF, actW=actW, ProdR=ProdR, ProdO=ProdO,
                   ShipF2W=ShipF2W, ShipW2C=ShipW2C,
                   Inv=Inv, Short=Short, Scrap=Scrap,
                   modeBlock=modeBlock)
