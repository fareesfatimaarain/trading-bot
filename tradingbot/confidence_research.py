"""Confidence-scored entries with planned exits (triple-barrier labels), walk-forward on hourly data.

For every coin-hour the model learns whether a long would hit +2x daily volatility before -1x within 48h
(and a second model the same for shorts). It trades only when its probability is in the top X% of what
it produced on training data. Exit at target, stop (stop first if both in one hour) or after 48h.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from . import config
from .intraday_research import COINS, features, load

TP_MULT, SL_MULT, MAX_HOURS = 2.0, 1.0, 48
COST = 0.002  # round trip, market orders: 0.1% per side incl. slippage
CONFIDENCE = [0.10, 0.05, 0.02, 0.01, 0.005]
TEST_YEARS = range(2022, 2027)


def barrier_outcomes(df: pd.DataFrame, side: int) -> pd.DataFrame:
    """For each hour: did a trade entered at the close hit target first? and its gross return."""
    c, h, lo = df["close"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy()
    sigma = (np.log(df["close"]).diff().rolling(24).std() * np.sqrt(24)).to_numpy()  # daily vol
    n = len(c)
    tp = c * (1 + side * TP_MULT * sigma)
    sl = c * (1 - side * SL_MULT * sigma)
    first_tp = np.full(n, MAX_HOURS + 1)
    first_sl = np.full(n, MAX_HOURS + 1)
    for k in range(MAX_HOURS, 0, -1):  # walk backwards so the earliest hit wins
        hk = np.append(h[k:], np.full(k, np.nan))
        lk = np.append(lo[k:], np.full(k, np.nan))
        hit_tp = (hk >= tp) if side == 1 else (lk <= tp)
        hit_sl = (lk <= sl) if side == 1 else (hk >= sl)
        first_tp[hit_tp] = k
        first_sl[hit_sl] = k
    end_close = np.append(c[MAX_HOURS:], np.full(MAX_HOURS, np.nan))
    timeout_ret = side * (end_close / c - 1)
    win = first_tp < first_sl  # a tie (same hour) counts as the stop
    gross = np.where(win, TP_MULT * sigma, np.where(first_sl <= MAX_HOURS, -SL_MULT * sigma, timeout_ret))
    hours = np.minimum(np.minimum(first_tp, first_sl), MAX_HOURS)
    valid = np.isfinite(sigma) & (np.arange(n) < n - MAX_HOURS)
    return pd.DataFrame({"win": win.astype(int), "gross": gross, "hours": hours, "valid": valid}, index=df.index)


def build() -> pd.DataFrame:
    data = {s: load(s) for s in COINS}
    frames = []
    for sym, df in data.items():
        f = features(df, data["BTCUSDT"])
        long, short = barrier_outcomes(df, 1), barrier_outcomes(df, -1)
        f["long_win"], f["long_gross"], f["long_hours"] = long["win"], long["gross"], long["hours"]
        f["short_win"], f["short_gross"], f["short_hours"] = short["win"], short["gross"], short["hours"]
        f = f[long["valid"]]
        f["symbol"] = sym
        frames.append(f)
    panel = pd.concat(frames).rename_axis("time").reset_index()
    panel["xs_rank_24h"] = panel.groupby("time")["ret_24h"].rank(pct=True)
    return panel.replace([np.inf, -np.inf], np.nan).dropna()


def simulate(test: pd.DataFrame, side: str, threshold: float) -> pd.DataFrame:
    """Take signals above threshold, one open trade per coin at a time."""
    trades = []
    for sym, g in test.sort_values("time").groupby("symbol"):
        busy_until = pd.Timestamp.min
        for row in g[g[f"p_{side}"] >= threshold].itertuples():
            if row.time < busy_until:
                continue
            hours = getattr(row, f"{side}_hours")
            trades.append({"time": row.time, "symbol": sym, "side": side, "hours": int(hours),
                           "net": getattr(row, f"{side}_gross") - COST, "win": getattr(row, f"{side}_win")})
            busy_until = row.time + pd.Timedelta(hours=int(hours))
    return pd.DataFrame(trades)


def main() -> None:
    warnings.filterwarnings("ignore")
    panel = build()
    feats = [c for c in panel.columns if c not in ("time", "symbol") and not c.startswith(("long_", "short_"))]
    base = {s: panel[f"{s}_win"].mean() for s in ("long", "short")}
    print(f"{len(panel):,} coin-hours. Random entry hits the target first: long {base['long']:.1%}, "
          f"short {base['short']:.1%}. Break-even win rate after costs is roughly 35%.\n")
    all_trades, calib = [], []
    for year in TEST_YEARS:
        start, end = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year + 1}-01-01")
        train = panel[panel["time"] < start - pd.Timedelta(hours=MAX_HOURS)]
        test = panel[(panel["time"] >= start) & (panel["time"] < end)].copy()
        if test.empty:
            continue
        for side in ("long", "short"):
            model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                                   min_samples_leaf=500, random_state=0)
            model.fit(train[feats], train[f"{side}_win"])
            p_train = model.predict_proba(train[feats].tail(200_000))[:, 1]
            test[f"p_{side}"] = model.predict_proba(test[feats])[:, 1]
            decile = pd.qcut(test[f"p_{side}"].rank(method="first"), 10, labels=False)
            calib.append(test.groupby(decile)[f"{side}_win"].mean().rename(f"{year} {side}"))
            for conf in CONFIDENCE:
                thr = np.quantile(p_train, 1 - conf)
                t = simulate(test, side, thr)
                if len(t):
                    t["confidence"], t["year"] = conf, year
                    all_trades.append(t)
    trades = pd.concat(all_trades)
    trades.to_csv(config.RESULTS_DIR / "confidence_trades.csv", index=False)

    print("Does higher confidence mean more wins? Actual target-first rate by confidence decile "
          "(1 = least confident, 10 = most), test years:")
    cal = pd.concat(calib, axis=1).mean(axis=1)
    print("  " + "  ".join(f"{i + 1}:{v:.0%}" for i, v in cal.items()))
    print("\nTrading only the most confident signals (long and short combined), 10% of the account per trade:")
    rows = []
    for conf, g in trades.groupby("confidence", sort=False):
        per_year = g.groupby("year").agg(trades=("net", "size"), win=("win", "mean"), avg=("net", "mean"),
                                         account=("net", lambda r: 0.10 * r.sum()))
        rows.append((conf, g, per_year))
    for conf, g, per_year in sorted(rows, key=lambda r: -r[0]):
        print(f"\n  top {conf:.1%} confidence: {len(g)} trades, won {g['win'].mean():.1%}, avg {g['net'].mean():+.2%} "
              f"per trade after costs")
        print("   " + "  ".join(f"{y}: {r.trades:>4} trades, won {r.win:.0%}, account {r.account:+.0%}"
                                for y, r in per_year.iterrows()))


if __name__ == "__main__":
    main()
