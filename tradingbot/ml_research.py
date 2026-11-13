"""Walk-forward machine-learning test: can a model trained on past charts pick coins well enough?

For each test year Y, a gradient-boosted tree model is trained only on data that ended before Y
(with a gap so no training label overlaps Y), then trades Y blind. Settings are fixed up front.
"""
import math

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from . import config, data
from .strategies import add_liquidity_rank

HOLD_DAYS = 7
TOP_K = 5
TOP_N_UNIVERSE = 50
COST_PER_SIDE = 0.0015  # 0.1% fee + 0.05% slippage
TEST_YEARS = range(2021, 2027)
PERIODS_PER_YEAR = 365 / HOLD_DAYS


def coin_features(df: pd.DataFrame, btc: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    r = c.pct_change()
    f = pd.DataFrame(index=df.index)
    for n in (1, 3, 7, 14, 30, 60, 90):
        f[f"ret_{n}"] = c.pct_change(n)
    f["vol_14"] = r.rolling(14).std()
    f["vol_60"] = r.rolling(60).std()
    f["dist_hi55"] = c / df["high"].rolling(55).max() - 1
    f["dist_lo55"] = c / df["low"].rolling(55).min() - 1
    for n in (20, 50, 200):
        f[f"dist_sma{n}"] = c / c.rolling(n).mean() - 1
    f["volume_surge"] = df["quote_volume"] / df["quote_volume"].rolling(30).median()
    f["range_pct"] = (df["high"] - df["low"]) / c
    b = btc["close"].reindex(df.index)
    f["btc_ret_7"] = b.pct_change(7)
    f["btc_ret_30"] = b.pct_change(30)
    f["btc_above_200"] = (b > b.rolling(200).mean()).astype(float)
    # buy at the next open, sell at the open HOLD_DAYS later
    f["fwd"] = df["open"].shift(-(HOLD_DAYS + 1)) / df["open"].shift(-1) - 1
    return f


def build_panel(universe: dict[str, pd.DataFrame]) -> pd.DataFrame:
    btc = universe[config.BENCHMARK]
    prepared = {s: df.copy() for s, df in universe.items()}
    add_liquidity_rank(prepared)
    frames = []
    for sym, df in prepared.items():
        f = coin_features(df, btc)
        f["liq_rank"] = df["liq_rank"]
        f["symbol"] = sym
        frames.append(f)
    panel = pd.concat(frames).rename_axis("date").reset_index()
    panel = panel[panel["liq_rank"] <= TOP_N_UNIVERSE]
    for col in ("ret_7", "ret_30", "ret_90"):
        panel[f"xs_rank_{col}"] = panel.groupby("date")[col].rank(pct=True)
    return panel


FEATURES = [
    "ret_1", "ret_3", "ret_7", "ret_14", "ret_30", "ret_60", "ret_90", "vol_14", "vol_60",
    "dist_hi55", "dist_lo55", "dist_sma20", "dist_sma50", "dist_sma200", "volume_surge", "range_pct",
    "btc_ret_7", "btc_ret_30", "btc_above_200", "xs_rank_ret_7", "xs_rank_ret_30", "xs_rank_ret_90",
]


def new_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                         min_samples_leaf=100, random_state=0)


def trade(rows: pd.DataFrame) -> pd.Series:
    """Weekly top-K long-only portfolio from rows with 'pred' and 'fwd'; returns net weekly returns."""
    dates = sorted(rows["date"].unique())[::HOLD_DAYS]
    prev_w: dict[str, float] = {}
    out = {}
    for d in dates:
        day = rows[rows["date"] == d].dropna(subset=["fwd"])
        picks = day[day["pred"] > 0].nlargest(TOP_K, "pred")
        w = {s: 1 / TOP_K for s in picks["symbol"]}
        turnover = sum(abs(w.get(s, 0) - prev_w.get(s, 0)) for s in set(w) | set(prev_w))
        gross = sum(w[s] * f for s, f in zip(picks["symbol"], picks["fwd"]))
        out[d] = gross - turnover * COST_PER_SIDE
        prev_w = w
    return pd.Series(out)


def summarize(rets: pd.Series) -> dict:
    if rets.empty:
        return {"cagr": float("nan"), "sharpe": float("nan"), "max_dd": float("nan")}
    eq = (1 + rets).cumprod()
    years = len(rets) / PERIODS_PER_YEAR
    return {
        "cagr": eq.iloc[-1] ** (1 / years) - 1,
        "sharpe": rets.mean() / rets.std() * math.sqrt(PERIODS_PER_YEAR) if rets.std() > 0 else float("nan"),
        "max_dd": (eq / eq.cummax() - 1).min(),
    }


def rank_ic(rows: pd.DataFrame) -> float:
    """Average per-day rank correlation between predicted and actual 7-day returns (0 = no skill)."""
    def one(g):
        return g["pred"].rank().corr(g["fwd"].rank()) if len(g) > 5 else np.nan
    return rows.dropna(subset=["fwd"]).groupby("date")[["pred", "fwd"]].apply(one).mean()


def main() -> None:
    universe = data.load_universe(refresh=False, market="spot")
    panel = build_panel(universe).dropna(subset=FEATURES)
    print(f"{len(panel):,} coin-days of training examples, {panel['symbol'].nunique()} coins\n")

    btc = panel[panel["symbol"] == config.BENCHMARK][["date", "fwd"]].set_index("date")["fwd"]
    print(f"{'year':<6}{'model OOS':>11}{'model in-sample':>17}{'BTC hold':>10}"
          f"{'IC OOS':>8}{'IC in-sample':>14}")
    oos_all, ins_all, btc_all = [], [], []
    for year in TEST_YEARS:
        start = pd.Timestamp(f"{year}-01-01")
        cutoff = start - pd.Timedelta(days=HOLD_DAYS + 1)
        train = panel[(panel["date"] < cutoff)].dropna(subset=["fwd"])
        test = panel[(panel["date"] >= start) & (panel["date"] < pd.Timestamp(f"{year + 1}-01-01"))].copy()
        if test.empty:
            continue
        model = new_model().fit(train[FEATURES], train["fwd"].clip(-0.5, 1.0))
        test["pred"] = model.predict(test[FEATURES])
        last_train_year = train[train["date"] >= cutoff - pd.Timedelta(days=365)].copy()
        last_train_year["pred"] = model.predict(last_train_year[FEATURES])

        oos = trade(test)
        ins = trade(last_train_year)
        b = btc.reindex(oos.index).fillna(0)
        oos_all.append(oos), ins_all.append(ins), btc_all.append(b)
        print(f"{year:<6}{(1 + oos).prod() - 1:>11.1%}{(1 + ins).prod() - 1:>17.1%}{(1 + b).prod() - 1:>10.1%}"
              f"{rank_ic(test):>8.3f}{rank_ic(last_train_year):>14.3f}")

    print()
    for name, parts in (("model, unseen years", oos_all), ("model, years it trained on", ins_all),
                        ("BTC buy and hold", btc_all)):
        m = summarize(pd.concat(parts))
        print(f"{name:<28} CAGR {m['cagr']:>7.1%}  Sharpe {m['sharpe']:>5.2f}  maxDD {m['max_dd']:>7.1%}")


if __name__ == "__main__":
    main()
