import numpy as np
import pandas as pd

from . import config


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n).mean()


def rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + gain / loss)


def add_btc_regime(prepared: dict[str, pd.DataFrame], data: dict[str, pd.DataFrame]) -> None:
    btc = data[config.BENCHMARK]["close"]
    regime = btc > btc.rolling(200).mean()
    for df in prepared.values():
        df["regime"] = regime.reindex(df.index, fill_value=False)


def add_liquidity_rank(prepared: dict[str, pd.DataFrame], min_age_days: int = 365) -> None:
    """Rank coins each day by 30-day median quote volume (1 = most liquid), using only data known then.

    Coins with less than `min_age_days` of history are unranked, so new listings are excluded.
    """
    medians = {}
    for sym, df in prepared.items():
        age = pd.Series(range(len(df)), index=df.index)
        medians[sym] = df["quote_volume"].rolling(30).median().where(age >= min_age_days)
    rank = pd.DataFrame(medians).rank(axis=1, ascending=False)
    for sym, df in prepared.items():
        df["liq_rank"] = rank[sym].reindex(df.index)


class Strategy:
    """Indicators must only use data up to and including the current bar."""

    name = "base"
    stop_atr = 2.0
    max_hold_days: int | None = None
    overrides: dict = {}

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def prepare_universe(self, data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        prepared = {sym: self.prepare(df) for sym, df in data.items()}
        add_liquidity_rank(prepared)
        return prepared

    def entry_signal(self, bar: dict) -> bool:
        raise NotImplementedError

    def exit_signal(self, bar: dict) -> bool:
        raise NotImplementedError

    def short_entry_signal(self, bar: dict) -> bool:
        return False

    def short_exit_signal(self, bar: dict) -> bool:
        return False


class TurtleTrend(Strategy):
    """Turtle System 2: 55-day breakout in, 20-day opposite extreme out, 2 ATR stop.

    Shorts (futures only) mirror the long rules.
    """

    name = "turtle"

    def __init__(self, entry_days: int = 55, exit_days: int = 20, stop_atr: float = 2.0):
        self.entry_days, self.exit_days, self.stop_atr = entry_days, exit_days, stop_atr

    def prepare(self, df):
        out = df.copy()
        out["atr"] = atr(out, 20)
        out["entry_level"] = out["high"].rolling(self.entry_days).max().shift(1)
        out["exit_level"] = out["low"].rolling(self.exit_days).min().shift(1)
        out["short_entry_level"] = out["low"].rolling(self.entry_days).min().shift(1)
        out["short_exit_level"] = out["high"].rolling(self.exit_days).max().shift(1)
        out["score"] = out["close"].pct_change(90)
        return out

    def entry_signal(self, bar):
        return bar["close"] > bar["entry_level"]

    def exit_signal(self, bar):
        return bar["close"] < bar["exit_level"]

    def short_entry_signal(self, bar):
        return bar["close"] < bar["short_entry_level"]

    def short_exit_signal(self, bar):
        return bar["close"] > bar["short_exit_level"]


class MomentumBreakout(Strategy):
    """Minervini trend template + relative strength + volatility contraction breakout on volume."""

    name = "momentum"
    stop_atr = 1.5

    def prepare(self, df):
        out = df.copy()
        close = out["close"]
        out["atr"] = atr(out, 20)
        for n in (20, 50, 150, 200):
            out[f"sma{n}"] = close.rolling(n).mean()
        out["sma200_rising"] = out["sma200"] > out["sma200"].shift(20)
        out["hi252"] = out["high"].rolling(252).max()
        out["breakout_level"] = out["high"].rolling(20).max().shift(1)
        out["vol50"] = out["quote_volume"].rolling(50).mean().shift(1)
        out["contraction"] = (atr(out, 10) / atr(out, 50)).shift(1)
        out["score"] = close.pct_change(90)
        return out

    def prepare_universe(self, data):
        prepared = super().prepare_universe(data)
        rank = pd.DataFrame({sym: df["score"] for sym, df in prepared.items()}).rank(axis=1, pct=True)
        for sym, df in prepared.items():
            df["rs_rank"] = rank[sym].reindex(df.index)
        add_btc_regime(prepared, data)
        return prepared

    def entry_signal(self, b):
        return bool(
            b["regime"]
            and b["rs_rank"] >= 0.7
            and b["close"] > b["sma50"] > b["sma150"] > b["sma200"]
            and b["sma200_rising"]
            and b["close"] >= 0.75 * b["hi252"]
            and b["contraction"] < 0.85
            and b["close"] > b["breakout_level"]
            and b["quote_volume"] > 1.5 * b["vol50"]
        )

    def exit_signal(self, b):
        return b["close"] < b["sma20"]


class DipBuyer(Strategy):
    """Connors-style pullback: RSI(2) washout in a coin above its 200-day SMA while BTC is too.

    Exits on a close above the 5-day SMA or after 10 days; 3 ATR catastrophe stop.
    Risk per trade is 3% because a 1% risk would rarely clear the $5 minimum order.
    """

    name = "dip"
    stop_atr = 3.0
    max_hold_days = 10
    overrides = {"risk_per_trade": 0.03, "max_open_risk": 0.15, "max_positions": 5, "max_position_frac": 0.3}

    def prepare(self, df):
        out = df.copy()
        out["atr"] = atr(out, 20)
        out["sma5"] = out["close"].rolling(5).mean()
        out["sma200"] = out["close"].rolling(200).mean()
        out["rsi2"] = rsi(out["close"], 2)
        out["score"] = -out["rsi2"]
        return out

    def prepare_universe(self, data):
        prepared = super().prepare_universe(data)
        add_btc_regime(prepared, data)
        return prepared

    def entry_signal(self, b):
        return bool(b["regime"] and b["close"] > b["sma200"] and b["rsi2"] < 10)

    def exit_signal(self, b):
        return b["close"] > b["sma5"]


def donchian_ensemble_signal(close: pd.Series, lookbacks) -> pd.Series:
    """Fraction of lookbacks currently long (0..1), per Zarattini, Pagani & Barbon (2025).

    For each n: go long on a close above the prior n-day highest close; the trailing stop is
    the highest midpoint of the n-day close channel seen since entry; exit on a close below it.
    """
    c = close.to_numpy()
    total = np.zeros(len(c))
    for n in lookbacks:
        upper = close.rolling(n).max().shift(1).to_numpy()
        mid = ((close.rolling(n).max() + close.rolling(n).min()) / 2).to_numpy()
        long, stop = False, -np.inf
        for i in range(len(c)):
            if long and c[i] < stop:
                long = False
            if long:
                stop = max(stop, mid[i])
            elif c[i] > upper[i]:
                long, stop = True, mid[i]
            total[i] += long
    return pd.Series(total / len(lookbacks), index=close.index)


class DonchianEnsemble(Strategy):
    """Volatility-targeted Donchian ensemble across nine lookbacks, long only, rebalanced daily.

    Weight per coin = signal x (target_vol / 90-day vol) / universe_top_n, total capped at max_gross.
    """

    name = "donchian"
    lookbacks = (5, 10, 20, 30, 60, 90, 150, 250, 360)
    overrides = {"universe_top_n": 20, "target_vol": 0.25, "max_gross": 1.0, "rebalance_band": 0.25}

    def prepare(self, df):
        out = df.copy()
        out["signal"] = donchian_ensemble_signal(out["close"], self.lookbacks)
        out["vol90"] = out["close"].pct_change().rolling(90).std() * np.sqrt(365)
        return out

    def target_weights(self, bars: dict[str, dict], s: dict) -> dict[str, float]:
        n = s["universe_top_n"]
        weights = {}
        for sym, b in bars.items():
            if not b.get("liq_rank", np.nan) <= n or not b["signal"] > 0 or not b["vol90"] > 0:
                continue
            weights[sym] = b["signal"] * min(s["target_vol"] / b["vol90"], 1.0) / n
        total = sum(weights.values())
        if total > s["max_gross"]:
            weights = {k: v * s["max_gross"] / total for k, v in weights.items()}
        return weights


ALL = {s.name: s for s in (TurtleTrend(), MomentumBreakout(), DipBuyer(), DonchianEnsemble())}
