import json
import logging
import time

import pandas as pd
import requests

from . import config

SPOT_KLINES = "https://data-api.binance.vision/api/v3/klines"
FAPI = "https://fapi.binance.com/fapi/v1"
DAY_MS = 86_400_000
COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_base", "taker_quote", "ignore",
]
STABLE = {"USDC", "FDUSD", "TUSD", "DAI", "USDP", "BUSD", "EUR", "USD1", "USDE", "PYUSD", "AEUR", "XUSD",
          "EURI", "BFUSD", "RLUSD", "PAXG", "WBTC", "WBETH", "BNSOL", "USDS"}
log = logging.getLogger(__name__)


def _fetch_klines(url: str, symbol: str, start_ms: int, limit: int) -> list:
    rows = []
    while True:
        resp = requests.get(url, params={"symbol": symbol, "interval": "1d", "startTime": start_ms, "limit": limit},
                            timeout=30)
        resp.raise_for_status()
        batch = resp.json()
        rows.extend(batch)
        if len(batch) < limit:
            return rows
        start_ms = batch[-1][0] + DAY_MS
        time.sleep(0.2)


def _fetch_funding(symbol: str, start_ms: int) -> list:
    rows = []
    while True:
        resp = requests.get(f"{FAPI}/fundingRate", params={"symbol": symbol, "startTime": start_ms, "limit": 1000},
                            timeout=30)
        resp.raise_for_status()
        batch = resp.json()
        rows.extend(batch)
        time.sleep(0.7)  # fundingRate is limited to 500 requests per 5 minutes
        if len(batch) < 1000:
            return rows
        start_ms = batch[-1]["fundingTime"] + 1


def _to_frame(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLUMNS)
    now_ms = int(time.time() * 1000)
    df = df[df["close_time"] < now_ms]  # drop the still-forming daily bar
    df["date"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.set_index("date")[["open", "high", "low", "close", "volume", "quote_volume"]]
    return df.astype(float)


def _last_complete_day() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - pd.Timedelta(days=1)


def _update_csv(path, fetch, start_default_ms: int, last_ts_ms,
                fresh_from: pd.Timedelta = pd.Timedelta(0)) -> pd.DataFrame | None:
    cached = pd.read_csv(path, index_col=0, parse_dates=[0]) if path.exists() else None
    if cached is not None and len(cached) and cached.index[-1] >= _last_complete_day() + fresh_from:
        return cached
    start_ms = last_ts_ms(cached) if cached is not None and len(cached) else start_default_ms
    try:
        fresh = fetch(start_ms)
    except requests.RequestException as exc:
        log.warning("download failed for %s: %s", path.name, exc)
        return cached
    if fresh is None or fresh.empty:
        return cached
    df = fresh if cached is None else pd.concat([cached, fresh])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index_label="date")
    return df


def load(symbol: str, refresh: bool = True, market: str = "spot") -> pd.DataFrame | None:
    prefix = "" if market == "spot" else "fut_"
    path = config.DATA_DIR / f"{prefix}{symbol}_1d.csv"
    if not refresh:
        df = pd.read_csv(path, index_col=0, parse_dates=[0]) if path.exists() else None
    else:
        url, limit = (SPOT_KLINES, 1000) if market == "spot" else (f"{FAPI}/klines", 1500)
        df = _update_csv(
            path,
            lambda start: (lambda rows: _to_frame(rows) if rows else None)(_fetch_klines(url, symbol, start, limit)),
            int(pd.Timestamp(config.HISTORY_START).timestamp() * 1000),
            lambda c: int(c.index[-1].timestamp() * 1000) + DAY_MS,
        )
    if df is None or market == "spot":
        return df
    funding = load_funding(symbol, refresh)
    df = df.copy()
    df["funding"] = funding.reindex(df.index, fill_value=0.0) if funding is not None else 0.0
    return df


def load_funding(symbol: str, refresh: bool = True) -> pd.Series | None:
    """Sum of the day's funding rates (positive = longs pay shorts), indexed by UTC date."""
    path = config.DATA_DIR / f"fund_{symbol}.csv"

    def fetch(start):
        rows = _fetch_funding(symbol, start)
        if not rows:
            return None
        f = pd.DataFrame(rows)
        f["date"] = pd.to_datetime(f["fundingTime"], unit="ms")
        return f.set_index("date")[["fundingRate"]].astype(float)

    if refresh:
        raw = _update_csv(path, fetch, int(pd.Timestamp(config.HISTORY_START).timestamp() * 1000),
                          lambda c: int(c.index[-1].timestamp() * 1000) + 1,
                          fresh_from=pd.Timedelta(hours=16))  # yesterday's last funding is at 16:00 UTC
    else:
        raw = pd.read_csv(path, index_col=0, parse_dates=[0]) if path.exists() else None
    if raw is None or raw.empty:
        return None
    return raw["fundingRate"].groupby(raw.index.normalize()).sum()


def pool() -> list[str]:
    path = config.DATA_DIR / "pool.json"
    return json.loads(path.read_text()) if path.exists() else list(config.UNIVERSE)


def build_pool(min_history_days: int = 365) -> list[str]:
    """Every USDT coin trading on both spot and perps with a year of history (no stablecoins)."""
    info = requests.get("https://data-api.binance.vision/api/v3/exchangeInfo", timeout=60).json()
    spot = {s["symbol"] for s in info["symbols"] if s["status"] == "TRADING" and s["quoteAsset"] == "USDT"}
    fut = requests.get(f"{FAPI}/exchangeInfo", timeout=30).json()
    perps = {s["symbol"] for s in fut["symbols"]
             if s["contractType"] == "PERPETUAL" and s["status"] == "TRADING" and s["quoteAsset"] == "USDT"}
    out = []
    for sym in sorted(spot & perps):
        base = sym[:-4]
        if not sym.isascii() or base in STABLE or base.endswith(("UP", "DOWN", "BULL", "BEAR")):
            continue
        k = requests.get(SPOT_KLINES, params={"symbol": sym, "interval": "1d", "limit": min_history_days + 1},
                         timeout=30).json()
        if len(k) > min_history_days:
            out.append(sym)
        time.sleep(0.05)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    (config.DATA_DIR / "pool.json").write_text(json.dumps(out))
    return out


def save_exchange_filters() -> dict:
    """Per-symbol minimum order value and quantity step for market orders, from Binance exchangeInfo."""

    def parse(symbols, notional_keys):
        out = {}
        for s in symbols:
            f = {x["filterType"]: x for x in s["filters"]}
            notional = next((float(f[k][field]) for k, field in notional_keys if k in f), None)
            lot = f.get("MARKET_LOT_SIZE") or f.get("LOT_SIZE")
            step = float(lot["stepSize"]) if lot else 0.0
            if step == 0.0 and "LOT_SIZE" in f:
                step = float(f["LOT_SIZE"]["stepSize"])
            out[s["symbol"]] = {"min_notional": notional or 5.0, "step": step}
        return out

    spot = requests.get("https://data-api.binance.vision/api/v3/exchangeInfo", timeout=60).json()
    fut = requests.get(f"{FAPI}/exchangeInfo", timeout=30).json()
    filters = {
        "spot": parse(spot["symbols"], [("NOTIONAL", "minNotional"), ("MIN_NOTIONAL", "minNotional")]),
        "futures": parse(fut["symbols"], [("MIN_NOTIONAL", "notional")]),
    }
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    (config.DATA_DIR / "filters.json").write_text(json.dumps(filters))
    return filters


def exchange_filters(market: str) -> dict:
    path = config.DATA_DIR / "filters.json"
    return json.loads(path.read_text()).get(market, {}) if path.exists() else {}


def load_universe(symbols=None, refresh: bool = True, market: str = "spot") -> dict[str, pd.DataFrame]:
    out = {}
    for symbol in symbols or pool():
        df = load(symbol, refresh, market)
        if df is not None and len(df):
            out[symbol] = df
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    save_exchange_filters()
    syms = build_pool()
    print(f"pool: {len(syms)} symbols")
    for i, sym in enumerate(syms, 1):
        load(sym, True, "spot")
        load(sym, True, "futures")
        if i % 25 == 0:
            print(f"{i}/{len(syms)} downloaded", flush=True)
    print("done")
