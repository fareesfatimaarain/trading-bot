"""Paper trading for the confidence model (same features, labels, thresholds and costs as confidence_research).

Hourly: update prices for the 10 coins, score each newly closed hour with the long and short models, open a
trade when a score is in the top 1% of the model's training scores (one trade per coin, max 4 open, 25% of
equity each), and manage exits: target +2x daily volatility, stop -1x (stop first if both in one hour), or
48 hours. Models retrain every 30 days on all history. Starts at the hour the account is created.
"""
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingClassifier

from . import config, data
from .confidence_research import COST, MAX_HOURS, SL_MULT, TP_MULT, build
from .intraday_research import CACHE, COINS, features, load

NAME = "confidence_model"
MODEL_DIR = config.PAPER_DIR / "confidence_model"
CONFIDENCE = 0.01
SIZE = 0.25
MAX_OPEN = 4
RETRAIN_DAYS = 30
log = logging.getLogger("paper")


def update_hourly() -> dict[str, pd.DataFrame]:
    out = {}
    for sym in COINS:
        df = load(sym)
        start = int(df.index[-1].timestamp() * 1000) + 3_600_000
        rows = []
        while True:
            r = requests.get("https://data-api.binance.vision/api/v3/klines",
                             params={"symbol": sym, "interval": "1h", "startTime": start, "limit": 1000}, timeout=30)
            r.raise_for_status()
            batch = r.json()
            rows += batch
            if len(batch) < 1000:
                break
            start = batch[-1][0] + 3_600_000
            time.sleep(0.15)
        now_ms = time.time() * 1000
        rows = [b for b in rows if b[6] < now_ms]  # completed bars only
        if rows:
            new = pd.DataFrame([[b[0], b[1], b[2], b[3], b[4], b[7], b[10]] for b in rows],
                               columns=["t", "open", "high", "low", "close", "quote_volume", "taker_buy_quote"]
                               ).astype(float)
            new.index = pd.to_datetime(new.pop("t"), unit="ms")
            df = pd.concat([df, new])
            df = df[~df.index.duplicated(keep="last")].sort_index()
            df.to_csv(CACHE / f"{sym}_1h.csv")
        out[sym] = df
    return out


def train() -> dict:
    panel = build()
    feats = [c for c in panel.columns if c not in ("time", "symbol") and not c.startswith(("long_", "short_"))]
    bundle = {"features": feats, "trained_at": str(pd.Timestamp.now(tz="UTC").tz_localize(None)),
              "trained_through": str(panel["time"].max())}
    for side in ("long", "short"):
        model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                               min_samples_leaf=500, random_state=0)
        model.fit(panel[feats], panel[f"{side}_win"])
        p_train = model.predict_proba(panel[feats].tail(200_000))[:, 1]
        bundle[side] = model
        bundle[f"{side}_threshold"] = float(np.quantile(p_train, 1 - CONFIDENCE))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_DIR / "models.joblib")
    log.info("[%s] models retrained on data through %s; thresholds long %.3f short %.3f", NAME,
             bundle["trained_through"], bundle["long_threshold"], bundle["short_threshold"])
    return bundle


def load_models() -> dict:
    path = MODEL_DIR / "models.joblib"
    if path.exists():
        bundle = joblib.load(path)
        age = pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timestamp(bundle["trained_at"])
        if age < pd.Timedelta(days=RETRAIN_DAYS):
            return bundle
    return train()


def scored_hours(prices: dict[str, pd.DataFrame], bundle: dict, after: pd.Timestamp) -> pd.DataFrame:
    frames = []
    for sym, df in prices.items():
        f = features(df, prices["BTCUSDT"])
        f["sigma"] = np.log(df["close"]).diff().rolling(24).std() * np.sqrt(24)
        f["close"] = df["close"]
        f["symbol"] = sym
        frames.append(f[f.index > after - pd.Timedelta(hours=1)])
    panel = pd.concat(frames).rename_axis("time").reset_index()
    panel["xs_rank_24h"] = panel.groupby("time")["ret_24h"].rank(pct=True)
    panel = panel[panel["time"] > after].replace([np.inf, -np.inf], np.nan).dropna(subset=bundle["features"])
    for side in ("long", "short"):
        panel[f"p_{side}"] = bundle[side].predict_proba(panel[bundle["features"]])[:, 1] if len(panel) else []
    return panel


def check_exit(pos: dict, bar: pd.Series, hour: pd.Timestamp) -> tuple[float, str] | None:
    """Exit price and reason if the position closes during `bar` (the hour starting at `hour`)."""
    side = pos["side"]
    hit_tp = bar["high"] >= pos["target"] if side == 1 else bar["low"] <= pos["target"]
    hit_sl = bar["low"] <= pos["stop"] if side == 1 else bar["high"] >= pos["stop"]
    if hit_sl:
        return pos["stop"], "stop"
    if hit_tp:
        return pos["target"], "target"
    if hour + pd.Timedelta(hours=1) >= pd.Timestamp(pos["deadline"]):
        return bar["close"], "48h limit"
    return None


def _funding(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> float:
    try:
        r = requests.get("https://fapi.binance.com/fapi/v1/fundingRate", timeout=30,
                         params={"symbol": symbol, "startTime": int(start.timestamp() * 1000),
                                 "endTime": int(end.timestamp() * 1000), "limit": 1000})
        r.raise_for_status()
        return sum(float(f["fundingRate"]) for f in r.json())
    except requests.RequestException:
        return 0.0


class ConfidenceAccount:
    def __init__(self, capital: float, created: str):
        self.start_capital = self.cash = capital
        self.created = self.last_hour = created
        self.positions: list[dict] = []
        self.trades: list[dict] = []
        self.skipped = 0

    def equity(self) -> float:
        return self.cash + sum(p["notional"] * (1 + p["side"] * (p["last"] / p["entry"] - 1)) for p in self.positions)

    def to_dict(self) -> dict:
        return {"kind": "confidence", **{k: getattr(self, k) for k in (
            "start_capital", "cash", "created", "last_hour", "positions", "trades", "skipped")}}

    @classmethod
    def from_dict(cls, d: dict) -> "ConfidenceAccount":
        a = cls(d["start_capital"], d["created"])
        for k in ("cash", "last_hour", "positions", "trades", "skipped"):
            setattr(a, k, d[k])
        return a

    def close(self, pos: dict, px: float, reason: str, when: pd.Timestamp) -> None:
        funding = pos["side"] * _funding(pos["symbol"], pd.Timestamp(pos["entry_time"]), when)
        ret = pos["side"] * (px / pos["entry"] - 1) - COST - funding
        self.cash += pos["notional"] * (1 + ret)
        self.positions.remove(pos)
        self.trades.append({**{k: pos[k] for k in ("symbol", "side", "entry_time", "entry", "notional", "confidence")},
                            "exit_time": str(when), "exit": px, "reason": reason, "funding": funding,
                            "ret": ret, "pnl": pos["notional"] * ret})
        log.info("[%s] closed %s %s @ %.6g (%s): %+.2f%% = $%+.2f", NAME, "LONG" if pos["side"] == 1 else "SHORT",
                 pos["symbol"], px, reason, ret * 100, pos["notional"] * ret)

    def open(self, row, side: int, filters: dict) -> None:
        notional = self.equity() * SIZE
        rule = filters.get(row.symbol, {})
        step = rule.get("step", 0.0)
        qty = notional / row.close
        if step:
            qty = np.floor(qty / step + 1e-9) * step
        if qty * row.close < rule.get("min_notional", 5.0) or qty <= 0:
            self.skipped += 1
            log.info("[%s] %s signal skipped: order below Binance minimum", NAME, row.symbol)
            return
        notional = qty * row.close
        self.cash -= notional
        when = pd.Timestamp(row.time) + pd.Timedelta(hours=1)  # signal at the bar's close
        pos = {"symbol": row.symbol, "side": side, "entry_time": str(when), "entry": row.close,
               "notional": notional, "last": row.close, "confidence": float(getattr(row, "p_long" if side == 1 else "p_short")),
               "target": row.close * (1 + side * TP_MULT * row.sigma),
               "stop": row.close * (1 - side * SL_MULT * row.sigma),
               "deadline": str(when + pd.Timedelta(hours=MAX_HOURS))}
        self.positions.append(pos)
        log.info("[%s] opened %s %s $%.2f @ %.6g, target %.6g, stop %.6g (confidence %.1f%%)", NAME,
                 "LONG" if side == 1 else "SHORT", row.symbol, notional, row.close, pos["target"], pos["stop"],
                 pos["confidence"] * 100)


def run_all() -> None:
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)  # log(0) on zero-volume hours; handled downstream
    state_path = config.PAPER_DIR / f"{NAME}_state.json"
    prices = update_hourly()
    last_closed = min(df.index[-1] for df in prices.values())
    if state_path.exists():
        acct = ConfidenceAccount.from_dict(json.loads(state_path.read_text()))
    else:
        acct = ConfidenceAccount(config.PAPER_CAPITAL, str(last_closed))
        log.info("[%s] new paper account with $%.2f; trading from the hour after %s UTC", NAME,
                 config.PAPER_CAPITAL, last_closed)
    bundle = load_models()
    filters = data.exchange_filters("futures")
    scored = scored_hours(prices, bundle, pd.Timestamp(acct.last_hour))
    for hour in sorted(scored["time"].unique()):
        hour = pd.Timestamp(hour)
        for pos in list(acct.positions):
            df = prices[pos["symbol"]]
            if hour in df.index and hour >= pd.Timestamp(pos["entry_time"]):
                bar = df.loc[hour]
                pos["last"] = bar["close"]
                hit = check_exit(pos, bar, hour)
                if hit:
                    acct.close(pos, hit[0], hit[1], hour + pd.Timedelta(hours=1))
        held = {p["symbol"] for p in acct.positions}
        for row in scored[scored["time"] == hour].itertuples():
            if row.symbol in held or len(acct.positions) >= MAX_OPEN:
                continue
            margin_long = row.p_long - bundle["long_threshold"]
            margin_short = row.p_short - bundle["short_threshold"]
            if max(margin_long, margin_short) < 0:
                continue
            acct.open(row, 1 if margin_long >= margin_short else -1, filters)
            held.add(row.symbol)
        acct.last_hour = str(hour)
    late = (pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timestamp(acct.last_hour)) / pd.Timedelta(hours=1)
    if late > 3:
        log.warning("[%s] latest processed hour is %.0fh old; data may be lagging", NAME, late)
    tmp = state_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(acct.to_dict(), indent=1))
    tmp.replace(state_path)
    log.info("[%s] run ok: through %s UTC, equity $%.2f, %d open", NAME, acct.last_hour, acct.equity(),
             len(acct.positions))
