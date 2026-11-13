"""Account-level view of confidence_research trades: real futures funding, compounding, position limits."""
import numpy as np
import pandas as pd

from . import config


def funding_per_hour(symbol: str) -> pd.Series:
    path = config.DATA_DIR / f"fund_{symbol}.csv"
    raw = pd.read_csv(path, index_col=0, parse_dates=[0])["fundingRate"]
    return raw.groupby(raw.index.floor("D")).sum() / 24  # average per hour of each UTC day


def with_funding(trades: pd.DataFrame) -> pd.DataFrame:
    trades = trades.copy()
    paid = []
    cache = {}
    for row in trades.itertuples():
        f = cache.setdefault(row.symbol, funding_per_hour(row.symbol))
        rate = f.get(row.time.floor("D"), 0.0)
        side = 1 if row.side == "long" else -1
        paid.append(side * rate * row.hours)  # longs pay positive funding, shorts receive it
    trades["funding"] = paid
    trades["net_after_funding"] = trades["net"] - trades["funding"]
    return trades


def account(trades: pd.DataFrame, size: float, max_open: int) -> pd.Series:
    """Compounding equity: each trade uses `size` of equity at entry; skip when `max_open` are already open."""
    t = trades.sort_values("time")
    equity, open_, curve = 1.0, [], {}
    for row in t.itertuples():
        for o in sorted([o for o in open_ if o[0] <= row.time]):
            equity += o[1] * o[2]
            open_.remove(o)
            curve[o[0]] = equity
        if len(open_) >= max_open:
            continue
        open_.append((row.time + pd.Timedelta(hours=row.hours), equity * size, row.net_after_funding))
    for o in sorted(open_):
        equity += o[1] * o[2]
        curve[o[0]] = equity
    return pd.Series(curve).sort_index()


def main() -> None:
    trades = pd.read_csv(config.RESULTS_DIR / "confidence_trades.csv", parse_dates=["time"])
    trades = with_funding(trades)
    for conf, g in trades.groupby("confidence"):
        conc = max(((g["time"] <= t) & (g["time"] + pd.to_timedelta(g["hours"], unit="h") > t)).sum()
                   for t in g["time"].sample(min(len(g), 2000), random_state=0))
        print(f"\ntop {conf:.1%} confidence: {len(g)} trades, avg before funding {g['net'].mean():+.3%}, "
              f"after funding {g['net_after_funding'].mean():+.3%}, average hold {g['hours'].mean():.0f}h, "
              f"most open at once ~{conc}")
        for size, max_open in ((0.10, 10), (0.25, 4), (0.50, 2)):
            eq = account(g, size, max_open)
            yearly = eq.groupby(eq.index.year).last()
            yearly = yearly / yearly.shift(1).fillna(1.0) - 1
            years = (eq.index[-1] - eq.index[0]).days / 365
            dd = (eq / eq.cummax().clip(lower=1.0) - 1).min()
            print(f"   {size:.0%} per trade, max {max_open} open: {eq.iloc[-1] ** (1 / years) - 1:+.1%}/yr, "
                  f"worst drop {dd:.0%} | " + " ".join(f"{y}:{v:+.0%}" for y, v in yearly.items()))


if __name__ == "__main__":
    main()
