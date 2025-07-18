"""
smartphone_plot_utils.py
──────────────────────────────────────────────────────────────────────────────
Optional helper functions for generating quick Matplotlib plots—e.g.
visualising total weekly CO₂, capacity utilisation, or the site‑opening
schedule.  **Not required** for submission scoring, but handy when analysing
feasibility gaps.

Usage Example
-------------
>>> import smartphone_plot_utils as spu
>>> spu.plot_weekly_co2("my_run_log.json")

(c) OpenAI o3 — 2025‑07‑18
"""
from __future__ import annotations
import json, datetime as dt, matplotlib.pyplot as plt
from pathlib import Path

def _dateify(s):                 # helper for json str→date
    return dt.datetime.strptime(s, "%Y-%m-%d").date()

def plot_weekly_co2(json_log: str | Path):
    """
    Expects a tiny JSON file emitted by your own logging (date→kg_CO2).
    """
    path = Path(json_log)
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text())
    weeks, kg = zip(*sorted((dt.datetime.strptime(k,"%Y-%m-%d").date(), v)
                            for k,v in data.items()))
    plt.figure(figsize=(9,4))
    plt.plot(weeks, [v/1000 for v in kg])
    plt.ylabel("ton CO₂ / week")
    plt.title("Weekly Total CO₂ Footprint")
    plt.tight_layout()
    plt.show()

__all__ = ["plot_weekly_co2"]
