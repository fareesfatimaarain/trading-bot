"""Can a model call the next hourly move well enough to trade every up and down? Walk-forward test.

Hourly Binance bars for 10 liquid coins with order-flow (taker buy share). Gradient-boosted trees and a
logistic baseline predict whether the next 1h / 4h return is up. Each year is predicted by a model trained
only on earlier years. Trading: long when P(up) is high, short when low, hold for the horizon, with
taker (market order) costs and with best-case maker (limit order) costs.
"""
import time
import warnings

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config

COINS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]
CACHE = config.DATA_DIR / "hourly"
START = pd.Timestamp("2020-01-01")
TEST_YEARS = range(2022, 2027)
COSTS = {"market orders (0.1%/side)": 0.001, "limit orders, best case (0.02%/side)": 0.0002}


def load(symbol: str) -> pd.DataFrame:
    path = CACHE / f"{symbol}_1h.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=[0])
    rows, start = [], int(START.timestamp() * 1000)
    while True:
        r = requests.get("https://data-api.binance.vision/api/v3/klines",
                         params={"symbol": symbol, "interval": "1h", "startTime": start, "limit": 1000}, timeout=30)
        r.raise_for_status()
        batch = r.json()
        rows += batch
        if len(batch) < 1000:
            break
        start = batch[-1][0] + 3_600_000
        time.sleep(0.15)
    df = pd.DataFrame([[b[0], b[1], b[2], b[3], b[4], b[7], b[10]] for b in rows],
                      columns=["t", "open", "high", "low", "close", "quote_volume", "taker_buy_quote"]).astype(float)
    df.index = pd.to_datetime(df.pop("t"), unit="ms")
    df = df.iloc[:-1]  # drop the forming bar
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df


def features(df: pd.DataFrame, btc: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    lr = np.log(c).diff()
    f = pd.DataFrame(index=df.index)
    for n in (1, 2, 3, 6, 12, 24, 72, 168):
        f[f"ret_{n}h"] = np.log(c / c.shift(n))
    f["vol_24h"] = lr.rolling(24).std()
    f["vol_168h"] = lr.rolling(168).std()
    f["vol_ratio"] = f["vol_24h"] / f["vol_168h"]
    qv = df["quote_volume"]
    f["volume_z"] = (np.log(qv) - np.log(qv).rolling(168).mean()) / np.log(qv).rolling(168).std()
    buy = df["taker_buy_quote"] / qv
    f["taker_buy_share"] = buy
    f["taker_buy_share_6h"] = buy.rolling(6).mean()
    f["flow_imbalance_24h"] = ((2 * buy - 1) * qv).rolling(24).sum() / qv.rolling(24).sum()
    f["dist_high_24h"] = c / df["high"].rolling(24).max() - 1
    f["dist_low_24h"] = c / df["low"].rolling(24).min() - 1
    f["dist_high_168h"] = c / df["high"].rolling(168).max() - 1
    f["range_pos"] = (c - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)
    f["hour"] = df.index.hour
    f["weekday"] = df.index.weekday
    b = np.log(btc["close"].reindex(df.index))
    for n in (1, 6, 24):
        f[f"btc_ret_{n}h"] = b - b.shift(n)
    return f


FEATURES = None  # filled in build()


def build() -> pd.DataFrame:
    global FEATURES
    data = {s: load(s) for s in COINS}
    frames = []
    for sym, df in data.items():
        f = features(df, data["BTCUSDT"])
        for h in (1, 4):
            f[f"fwd_{h}h"] = np.log(df["close"].shift(-h) / df["close"])
        f["symbol"] = sym
        frames.append(f)
    panel = pd.concat(frames).rename_axis("time").reset_index()
    panel["xs_rank_24h"] = panel.groupby("time")["ret_24h"].rank(pct=True)
    FEATURES = [c for c in panel.columns if c not in ("time", "symbol", "fwd_1h", "fwd_4h")]
    return panel.replace([np.inf, -np.inf], np.nan).dropna()


def evaluate(panel: pd.DataFrame, horizon: int) -> None:
    target = f"fwd_{horizon}h"
    avg_move = panel[target].abs().mean()
    print(f"\n=== Next {horizon}h: average absolute move {avg_move:.2%} ===")
    for cost_name, cost in COSTS.items():
        rt = 2 * cost
        breakeven = 0.5 + rt / (2 * avg_move)
        print(f"  break-even accuracy with {cost_name}: {breakeven:.1%} of calls right (if every move were average)")
    models = {"gradient-boosted trees": lambda: HistGradientBoostingClassifier(
                  max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=500, random_state=0),
              "logistic baseline": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))}
    for name, make in models.items():
        print(f"\n  {name}:")
        print(f"  {'year':<6}{'accuracy':>9}{'AUC':>7}{'train acc':>10}  " +
              "  ".join(f"{k.split(' (')[0]:>16}" for k in COSTS) + "   (return per year, trading the confident 20%)")
        for year in TEST_YEARS:
            start = pd.Timestamp(f"{year}-01-01")
            train = panel[panel["time"] < start - pd.Timedelta(hours=horizon)]
            test = panel[(panel["time"] >= start) & (panel["time"] < pd.Timestamp(f"{year + 1}-01-01"))]
            if test.empty:
                continue
            y_train = (train[target] > 0).astype(int)
            model = make().fit(train[FEATURES], y_train)
            p = model.predict_proba(test[FEATURES])[:, 1]
            p_train = model.predict_proba(train[FEATURES].tail(200_000))[:, 1]
            y = (test[target] > 0).astype(int).to_numpy()
            acc = ((p > 0.5) == y).mean()
            train_acc = ((p_train > 0.5) == (train[target].tail(200_000) > 0)).mean()
            auc = roc_auc_score(y, p)
            # trade only the most confident 20% of calls (10% most bullish long, 10% most bearish short),
            # non-overlapping: one decision per coin every `horizon` hours, equal weight across coins
            t = test.assign(p=p)
            t = t[t["time"].dt.hour % horizon == 0]
            lo, hi = np.quantile(p_train, [0.10, 0.90])  # cut-offs from training data, not the test year
            side = np.where(t["p"].to_numpy() >= hi, 1, np.where(t["p"].to_numpy() <= lo, -1, 0))
            gross = side * (np.exp(t[target].to_numpy()) - 1)
            traded = side != 0
            yearly = []
            for cost in COSTS.values():
                net = pd.Series(gross - traded * 2 * cost, index=t["time"].to_numpy()).groupby(level=0).mean()
                yearly.append(np.prod(1 + net.to_numpy()) ** (365 * 24 / horizon / max(len(net), 1)) - 1)
            hit = (np.sign(gross[traded]) > 0).mean() if traded.any() else np.nan
            print(f"  {year:<6}{acc:>9.1%}{auc:>7.3f}{train_acc:>10.1%}  " +
                  "  ".join(f"{v:>+16.0%}" for v in yearly) +
                  f"   | confident calls right {hit:.1%}, avg move caught {gross[traded].mean():+.3%} per trade")


def main() -> None:
    warnings.filterwarnings("ignore")
    panel = build()
    print(f"{len(panel):,} coin-hours, {panel['symbol'].nunique()} coins, "
          f"{panel['time'].min().date()} .. {panel['time'].max().date()}, {len(FEATURES)} features")
    for h in (1, 4):
        evaluate(panel, h)


if __name__ == "__main__":
    main()
