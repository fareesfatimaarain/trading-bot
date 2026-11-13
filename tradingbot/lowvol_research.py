"""Low-volatility long-short with a hard stop on every short, tested on 2014-2018 and 2019-2026.

Weekly: long the 5 calmest coins, short the 5 wildest (60-day volatility of daily returns), half the
capital per side. A short is bought back if price reaches `stop` above its entry (or at the open if it
gaps past). Stop level fixed before testing at the user's request: 20%.
"""
import sys
import warnings

import numpy as np
import pandas as pd

from . import data, early_history
from .dl_research import LS_COST_PER_SIDE, build_samples
from .ml_research import HOLD_DAYS, summarize

K = 5
STOP = 0.20


def _short_leg(df: pd.DataFrame, i: int, stop: float | None, flip: float | None = None
               ) -> tuple[float, bool, bool, float]:
    """Slot return for a short entered at the open of bar i: (return incl. funding, stopped?, flipped?, extra fills).

    With `flip`, after the stop the coin is bought if it reaches `flip` above the short entry the same week,
    and held to the weekly exit. Returns compound through the slot's capital.
    """
    entry = df["open"].iat[i]
    exit_i = i + HOLD_DAYS
    if exit_i >= len(df):
        return np.nan, False, False, 0
    fund = df["funding"] if "funding" in df else pd.Series(0.0, index=df.index)
    if stop is None:
        return -(df["open"].iat[exit_i] / entry - 1) + fund.iloc[i:exit_i].sum(), False, False, 0
    stop_level = entry * (1 + stop)
    for j in range(i, exit_i):
        if max(df["open"].iat[j], df["high"].iat[j]) < stop_level:
            continue
        cover = max(df["open"].iat[j], stop_level)
        short_ret = -(cover / entry - 1) + fund.iloc[i:j].sum()
        if flip is None:
            return short_ret, True, False, 1
        flip_level = entry * (1 + flip)
        for k in range(j, exit_i):
            if max(df["open"].iat[k], df["high"].iat[k]) >= flip_level:
                buy = max(df["open"].iat[k], flip_level)
                long_ret = df["open"].iat[exit_i] / buy - 1 - fund.iloc[k:exit_i].sum()
                return (1 + short_ret) * (1 + long_ret) - 1, True, True, 3
        return short_ret, True, False, 1
    return -(df["open"].iat[exit_i] / entry - 1) + fund.iloc[i:exit_i].sum(), False, False, 0


def backtest(universe: dict[str, pd.DataFrame], stop: float | None,
             flip: float | None = None) -> tuple[pd.Series, dict]:
    x, meta = build_samples(universe)
    meta["pred"] = -(x[:, 0, -60:] / 10).std(1)
    pos = {s: pd.Series(np.arange(len(df)), index=df.index) for s, df in universe.items()}
    dates = sorted(meta["date"].unique())[::HOLD_DAYS]
    prev_w: dict[str, float] = {}
    out, stops, flips, worst = {}, 0, 0, 0.0
    for d in dates:
        day = meta[meta["date"] == d].dropna(subset=["fwd"])
        w: dict[str, float] = {}
        gross, extra_cost = 0.0, 0.0
        if len(day) >= 2 * K:
            for _, row in day.nlargest(K, "pred").iterrows():
                w[row["symbol"]] = 0.5 / K
                gross += 0.5 / K * (row["fwd"] - row["fund"])
            for _, row in day.nsmallest(K, "pred").iterrows():
                sym = row["symbol"]
                df = universe[sym]
                i = pos[sym].get(d)
                if i is None or i + 1 >= len(df):
                    continue
                ret, stopped, flipped, extra_fills = _short_leg(df, i + 1, stop, flip)
                if np.isnan(ret):
                    continue
                gross += 0.5 / K * ret
                worst = min(worst, ret)
                stops += stopped
                flips += flipped
                extra_cost += 0.5 / K * LS_COST_PER_SIDE * extra_fills  # cover, and buy + sell if flipped
                if not stopped:
                    w[sym] = -0.5 / K
        turnover = sum(abs(w.get(s, 0) - prev_w.get(s, 0)) for s in set(w) | set(prev_w))
        out[d] = gross - turnover * LS_COST_PER_SIDE - extra_cost
        prev_w = w  # stopped shorts are already closed, so they are absent here
    return pd.Series(out), {"stops": stops, "flips": flips, "worst_short_week": worst}


def main() -> None:
    warnings.filterwarnings("ignore")
    eras = [
        ("2014-2018 CoinMarketCap", early_history.load(), None),
        ("2019-2026 Binance spot", data.load_universe(refresh=False, market="spot"), None),
        ("2021-2026 Binance futures", data.load_universe(refresh=False, market="futures"), "2021-01-01"),
    ]
    if "--quick" in sys.argv:
        eras = eras[:1]
    global STOP
    if "--stop" in sys.argv:
        STOP = float(sys.argv[sys.argv.index("--stop") + 1])
    flip = float(sys.argv[sys.argv.index("--flip") + 1]) if "--flip" in sys.argv else None
    variants = [("no stop", None, None), (f"stop +{STOP:.0%}", STOP, None)]
    if flip:
        variants.append((f"stop {STOP:.0%}/flip {flip:.0%}", STOP, flip))
    for label, universe, start in eras:
        for name, stop, fl in variants:
            r, info = backtest(universe, stop, fl)
            if start:
                r = r[r.index >= start]
            yearly = (1 + r).groupby(r.index.year).prod() - 1
            m = summarize(r)
            print(f"{label:<26} {name:<18} CAGR {m['cagr']:>7.1%}  Sharpe {m['sharpe']:>5.2f}  "
                  f"maxDD {m['max_dd']:>7.1%}  stops {info['stops']:>4}  flips {info['flips']:>4}  "
                  f"worst slot {info['worst_short_week']:>6.0%}  "
                  + " ".join(f"{y}:{v:+.0%}" for y, v in yearly.items()), flush=True)


if __name__ == "__main__":
    main()
