"""What happens to coins right after they list on Binance spot? (2021 onward, delisted coins included)

Entry is the close of the first minute: the opening print itself goes to bots. Returns are measured from
that entry at fixed horizons, plus "sell when price falls X% from its high since entry" (trailing exit).
"""
import json
import time

import numpy as np
import pandas as pd
import requests

from . import config
from .data import STABLE

KLINES = "https://data-api.binance.vision/api/v3/klines"
CACHE = config.DATA_DIR / "listings"
START = pd.Timestamp("2021-01-01")
ROUND_TRIP_COST = 0.005  # 0.1% fee each way plus wide launch spreads
MEMES = {"DOGE", "SHIB", "PEPE", "FLOKI", "BONK", "WIF", "BOME", "MEME", "PEOPLE", "TURBO", "NEIRO", "PNUT", "ACT",
         "DOGS", "TRUMP", "MUBARAK", "PENGU", "1000SATS", "1000CHEEMS", "CHEEMS", "BROCCOLI714", "TST", "1MBABYDOGE",
         "BABYDOGE", "MOODENG", "NOT", "HMSTR", "GOAT", "POPCAT", "BANANAS31", "SPX", "FARTCOIN", "PUMP", "WLFI"}


def _get(params: dict) -> list:
    for attempt in range(5):
        try:
            r = requests.get(KLINES, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
    return []


def _frame(rows: list) -> pd.DataFrame:
    df = pd.DataFrame([r[:5] for r in rows], columns=["t", "open", "high", "low", "close"]).astype(float)
    df["t"] = pd.to_datetime(df["t"], unit="ms")
    return df.set_index("t")


def fetch(symbol: str) -> dict | None:
    path = CACHE / f"{symbol}.json"
    if path.exists():
        return json.loads(path.read_text())
    first = _get({"symbol": symbol, "interval": "1d", "startTime": 0, "limit": 1})
    if not first:
        return None
    t0 = first[0][0]
    out = {"symbol": symbol, "listed_ms": t0, "fetched": pd.Timestamp.now().isoformat(),
           "minutes": [], "hours": [], "days": []}
    if pd.Timestamp(t0, unit="ms") >= START:
        out["minutes"] = _get({"symbol": symbol, "interval": "1m", "startTime": t0, "limit": 1000})
        if len(out["minutes"]) == 1000:
            out["minutes"] += _get({"symbol": symbol, "interval": "1m", "startTime": out["minutes"][-1][0] + 60_000,
                                    "limit": 440})
        out["hours"] = _get({"symbol": symbol, "interval": "1h", "startTime": t0, "limit": 720})
        out["days"] = _get({"symbol": symbol, "interval": "1d", "startTime": t0, "limit": 365})
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return out


def analyse(raw: dict) -> dict | None:
    if not raw["minutes"] or len(raw["minutes"]) < 60:
        return None
    m, h, d = _frame(raw["minutes"]), _frame(raw["hours"]), _frame(raw["days"])
    listed = m.index[0]
    p0, entry = m["open"].iat[0], m["close"].iat[0]
    base = raw["symbol"][:-4]
    row = {"symbol": raw["symbol"], "listed": listed, "meme": base in MEMES,
           "first_minute_spike": m["high"].iat[0] / p0 - 1, "entry_vs_first_print": entry / p0 - 1}

    def ret_at(frame, delta):
        later = frame[frame.index >= listed + delta]
        px = later["close"].iat[0] if len(later) else frame["close"].iat[-1]  # delisted/no data: last price
        return px / entry - 1

    for name, frame, delta in (("5m", m, "5min"), ("15m", m, "15min"), ("1h", m, "1h"), ("4h", m, "4h"),
                               ("24h", m, "1439min"), ("7d", h, "7D"), ("30d", h, "719h"),
                               ("90d", d, "90D"), ("180d", d, "180D"), ("365d", d, "363D")):  # daily bars start at 00:00, so stay inside the window
        too_recent = listed + pd.Timedelta(delta) > pd.Timestamp(raw.get("fetched", pd.Timestamp.now()))
        beyond_fetch = frame.index[-1] < listed + pd.Timedelta(delta) and len(frame) >= {
            "m": 1440, "h": 720, "d": 365}["m" if frame is m else "h" if frame is h else "d"]
        # otherwise a series ending early means the coin was delisted: use its last price
        row[name] = np.nan if too_recent or beyond_fetch else ret_at(frame, pd.Timedelta(delta))
    peak_t = h["high"].idxmax()
    row["peak_vs_entry_30d"] = h["high"].max() / entry - 1
    row["hours_to_peak_30d"] = (peak_t - listed) / pd.Timedelta(hours=1)
    row["minutes_to_peak_24h"] = (m["high"].idxmax() - listed) / pd.Timedelta(minutes=1)
    row["low_after_peak_30d"] = h.loc[peak_t:, "low"].min() / h["high"].max() - 1
    after = m["close"].to_numpy()[1:]
    for trail in (0.05, 0.10, 0.20, 0.30):
        high, exit_px = entry, after[-1] if len(after) else entry
        for px in after:
            high = max(high, px)
            if px <= high * (1 - trail):
                exit_px = px
                break
        row[f"trail_{int(trail * 100)}"] = exit_px / entry - 1
    return row


def main() -> None:
    info = requests.get("https://data-api.binance.vision/api/v3/exchangeInfo", timeout=60).json()
    symbols = [s["symbol"] for s in info["symbols"] if s["quoteAsset"] == "USDT"
               and s["baseAsset"] not in STABLE and not s["baseAsset"].endswith(("UP", "DOWN", "BULL", "BEAR"))]
    rows = []
    for i, sym in enumerate(symbols, 1):
        raw = fetch(sym)
        if raw:
            r = analyse(raw)
            if r:
                rows.append(r)
        if i % 100 == 0:
            print(f"  {i}/{len(symbols)} symbols checked, {len(rows)} listings since 2021", flush=True)
    analyse_cache_and_report(rows)


def analyse_cache_and_report(rows: list | None = None) -> None:
    if rows is None:
        rows = [r for p in sorted(CACHE.glob("*.json")) if (r := analyse(json.loads(p.read_text())))]
    df = pd.DataFrame(rows)
    df = df[df["listed"] <= pd.Timestamp.now() - pd.Timedelta(days=31)]
    df.to_csv(config.RESULTS_DIR / "listings.csv", index=False)
    report(df)


def report(df: pd.DataFrame) -> None:
    pct = lambda v: f"{v:+.1%}"
    print(f"\n{len(df)} Binance spot listings since 2021 ({df['meme'].sum()} memecoins); entry = close of minute 1\n")
    print(f"first-minute spike above the very first print: median {df['first_minute_spike'].median():+.0%}; "
          f"entry (minute-1 close) vs first print: median {df['entry_vs_first_print'].median():+.0%}")
    t = df["minutes_to_peak_24h"]
    print(f"highest price of day 1 came in minute 1: {(t < 1).mean():.0%}, within 5 min: {(t < 5).mean():.0%}, "
          f"within 1 h: {(t < 60).mean():.0%}")
    print(f"30-day peak above entry: median {df['peak_vs_entry_30d'].median():+.0%}; peak in the first 24h: "
          f"{(df['hours_to_peak_30d'] < 24).mean():.0%}; fall from that peak within 30 days: median "
          f"{df['low_after_peak_30d'].median():.0%}\n")
    cols = ["5m", "15m", "1h", "4h", "24h", "7d", "30d", "90d", "180d", "365d"]
    for label, sub in (("all listings", df), ("memecoins", df[df["meme"]]), ("others", df[~df["meme"]])):
        print(f"Buy at minute-1 close, sell after... ({label}, n={len(sub)})")
        tbl = pd.DataFrame({
            "median": sub[cols].median(), "average": sub[cols].mean(),
            "% up": (sub[cols] > 0).mean(), "avg after 0.5% costs": sub[cols].mean() - ROUND_TRIP_COST})
        print(tbl.T.to_string(float_format=lambda v: f"{v:+.1%}"))
        print()
    trail = [c for c in df.columns if c.startswith("trail_")]
    print("Sell when price falls X% from its high since entry (within day 1, else sell at 24h):")
    tbl = pd.DataFrame({"median": df[trail].median(), "average": df[trail].mean(), "% up": (df[trail] > 0).mean(),
                        "avg after costs": df[trail].mean() - ROUND_TRIP_COST})
    tbl.index = [f"trail {c.split('_')[1]}%" for c in trail]
    print(tbl.T.to_string(float_format=lambda v: f"{v:+.1%}"))
    by_year = df.groupby(df["listed"].dt.year)[["1h", "24h", "30d", "365d"]].median()
    print("\nMedian return by listing year:\n" + by_year.to_string(float_format=pct))


if __name__ == "__main__":
    import sys
    analyse_cache_and_report() if "--report" in sys.argv else main()
