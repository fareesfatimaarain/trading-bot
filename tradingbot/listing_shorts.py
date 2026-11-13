"""Shorting newly listed coins on Binance USDT-M futures, with stop-losses against squeezes.

Only genuinely new coins: the perp listed within 30 days of the coin's Binance spot listing, or the coin
has no spot market. Entry: short `delay` after the perp's first minute. Exit after `hold`, or at the stop
(price +stop% above entry; fills at the worse of stop and bar open, plus 0.5% extra slippage), or at the
last price if the contract was delisted. Shorts collect funding when it is positive and pay when negative.
Each short is fully collateralised (1x), so a trade can lose at most 100%.
"""
import itertools
import json
import time

import numpy as np
import pandas as pd
import requests

from . import config

FAPI = "https://fapi.binance.com/fapi/v1"
CACHE = config.DATA_DIR / "listing_perps"
SPOT_CACHE = config.DATA_DIR / "listings"
START = pd.Timestamp("2021-01-01")
FEE = 0.0005
STOP_SLIP = 0.005
DELAYS = {"1 min": 1, "1 hour": 60, "1 day": 1440}  # minutes after the first trade
HOLDS = {"1 min": 1, "1 hour": 60, "1 day": 1440, "1 week": 7 * 1440, "1 month": 30 * 1440}
STOPS = (None, 0.15, 0.30, 0.50)
SPLIT = pd.Timestamp("2024-01-01")  # choose on listings before, check on listings after


def _get(path: str, params: dict) -> list:
    for attempt in range(5):
        try:
            r = requests.get(f"{FAPI}/{path}", params=params, timeout=30)
            if r.status_code == 400:
                return []
            r.raise_for_status()
            time.sleep(0.25)
            return r.json()
        except requests.RequestException:
            time.sleep(3 * (attempt + 1))
    return []


def fetch(symbol: str) -> dict | None:
    path = CACHE / f"{symbol}.json"
    if path.exists():
        return json.loads(path.read_text())
    first = _get("klines", {"symbol": symbol, "interval": "1d", "startTime": 0, "limit": 1})
    out = {"symbol": symbol, "listed_ms": first[0][0] if first else None, "minutes": [], "hours": [], "funding": []}
    if first and pd.Timestamp(first[0][0], unit="ms") >= START:
        t0 = first[0][0]
        out["minutes"] = _get("klines", {"symbol": symbol, "interval": "1m", "startTime": t0, "limit": 1500})
        if len(out["minutes"]) == 1500:
            out["minutes"] += _get("klines", {"symbol": symbol, "interval": "1m",
                                              "startTime": out["minutes"][-1][0] + 60_000, "limit": 1500})
        out["hours"] = _get("klines", {"symbol": symbol, "interval": "1h", "startTime": t0, "limit": 1500})
        out["funding"] = _get("fundingRate", {"symbol": symbol, "startTime": t0, "limit": 1000})
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return out


def _spot_listing(symbol: str) -> pd.Timestamp | None:
    base = symbol[:-4]
    for prefix in ("1000000", "10000", "1000", "1M"):
        if base.startswith(prefix) and len(base) > len(prefix):
            base = base[len(prefix):]
            break
    path = SPOT_CACHE / f"{base}USDT.json"
    if not path.exists():
        return None
    return pd.Timestamp(json.loads(path.read_text())["listed_ms"], unit="ms")


def _frame(rows: list) -> pd.DataFrame:
    df = pd.DataFrame([r[:5] for r in rows], columns=["t", "open", "high", "low", "close"]).astype(float)
    df["t"] = pd.to_datetime(df["t"], unit="ms")
    return df.set_index("t")


def prepare(raw: dict) -> dict | None:
    if not raw.get("minutes") or len(raw["minutes"]) < 120:
        return None
    listed = pd.Timestamp(raw["listed_ms"], unit="ms")
    spot = _spot_listing(raw["symbol"])
    if spot is not None and spot < listed - pd.Timedelta(days=30):
        return None  # an established coin getting a new perp, not a new listing
    minutes, hours = _frame(raw["minutes"]), _frame(raw["hours"])
    # one path: minute bars for the first ~2 days, hourly bars after that
    path = pd.concat([minutes, hours[hours.index >= minutes.index[-1] + pd.Timedelta(minutes=1)]])
    funding = pd.Series({pd.Timestamp(f["fundingTime"], unit="ms"): float(f["fundingRate"]) for f in raw["funding"]},
                        dtype=float).sort_index()
    return {"symbol": raw["symbol"], "listed": minutes.index[0], "path": path, "funding": funding,
            "complete": len(raw["hours"]) == 1500}


def short_trade(coin: dict, delay_min: int, hold_min: int, stop: float | None) -> float | None:
    path = coin["path"]
    t_entry = coin["listed"] + pd.Timedelta(minutes=delay_min)
    before = path[path.index < t_entry]
    if before.empty or before.index[-1] < t_entry - pd.Timedelta(minutes=2):
        return None
    entry = before["close"].iat[-1]
    t_exit = t_entry + pd.Timedelta(minutes=hold_min)
    window = path[(path.index >= t_entry) & (path.index < t_exit)]
    if path.index[-1] < t_exit - pd.Timedelta(hours=1) and coin["complete"]:
        return None  # horizon beyond the data we fetched
    entry_cost = FEE + (0.0025 if delay_min <= 60 else 0.001)
    exit_px, exit_t, extra = None, None, 0.0
    if stop is not None and len(window):
        level = entry * (1 + stop)
        hit = window[window["high"] >= level]
        if len(hit):
            bar = hit.iloc[0]
            exit_px, exit_t, extra = max(bar["open"], level), hit.index[0], STOP_SLIP
    if exit_px is None:
        if len(window) == 0:
            return None
        exit_px, exit_t = window["close"].iat[-1], window.index[-1]  # also the delisting price if it ended early
    f = coin["funding"]
    funding = f[(f.index > t_entry) & (f.index <= exit_t)].sum()
    ret = -(exit_px / entry - 1) + funding - entry_cost - (FEE + 0.001 + extra)
    return max(ret, -1.0)


def portfolio(trades: pd.DataFrame, fraction: float = 0.10) -> dict:
    """Chronological compounding: each short uses `fraction` of equity at entry (overlaps allowed)."""
    if trades.empty:
        return {"cagr": np.nan, "max_dd": np.nan}
    t = trades.sort_values("entry")
    events = sorted([(row.entry, 0, i) for i, row in t.iterrows()] + [(row.exit, 1, i) for i, row in t.iterrows()],
                    key=lambda e: (e[0], -e[1]))
    equity, alloc, curve = 1.0, {}, []
    for when, kind, i in events:
        if kind == 0:
            alloc[i] = equity * fraction
        else:
            equity += alloc.pop(i) * t.at[i, "ret"]
            curve.append((when, equity))
    s = pd.Series([e for _, e in curve], index=[w for w, _ in curve])
    years = max((t["exit"].max() - t["entry"].min()) / pd.Timedelta(days=365), 0.1)
    return {"cagr": equity ** (1 / years) - 1 if equity > 0 else -1.0,
            "max_dd": (s / s.cummax().clip(lower=1.0) - 1).min()}


def main() -> None:
    fut = requests.get(f"{FAPI}/exchangeInfo", timeout=60).json()["symbols"]
    symbols = {s["symbol"] for s in fut if s["quoteAsset"] == "USDT" and s["contractType"] == "PERPETUAL"}
    symbols |= {p.stem for p in SPOT_CACHE.glob("*.json")}  # catches perps delisted before today's list
    coins = []
    for i, sym in enumerate(sorted(symbols), 1):
        raw = fetch(sym)
        coin = prepare(raw) if raw else None
        if coin:
            coins.append(coin)
        if i % 100 == 0:
            print(f"  {i}/{len(symbols)} symbols checked, {len(coins)} new-coin perps", flush=True)
    print(f"\n{len(coins)} new-coin perpetual listings since 2021", flush=True)

    rows = []
    for (dname, dmin), (hname, hmin), stop in itertools.product(DELAYS.items(), HOLDS.items(), STOPS):
        for c in coins:
            r = short_trade(c, dmin, hmin, stop)
            if r is not None:
                entry = c["listed"] + pd.Timedelta(minutes=dmin)
                rows.append({"delay": dname, "hold": hname, "stop": stop, "symbol": c["symbol"],
                             "entry": entry, "exit": entry + pd.Timedelta(minutes=hmin), "ret": r})
    trades = pd.DataFrame(rows)
    trades.to_csv(config.RESULTS_DIR / "listing_shorts_trades.csv", index=False)
    report(trades)


def report(trades: pd.DataFrame) -> None:
    trades["stop_label"] = trades["stop"].map(lambda s: "none" if pd.isna(s) else f"+{s:.0%}")
    out = []
    for (d, h, s), g in trades.groupby(["delay", "hold", "stop_label"], sort=False):
        early, late = g[g["entry"] < SPLIT], g[g["entry"] >= SPLIT]
        pf = portfolio(g)
        out.append({"enter after": d, "hold": h, "stop": s, "trades": len(g), "avg": g["ret"].mean(),
                    "median": g["ret"].median(), "win %": (g["ret"] > 0).mean(), "worst": g["ret"].min(),
                    "avg 2021-23": early["ret"].mean(), "avg 2024-26": late["ret"].mean(),
                    "yearly (10%/trade)": pf["cagr"], "max drop": pf["max_dd"]})
    df = pd.DataFrame(out)
    df.to_csv(config.RESULTS_DIR / "listing_shorts_summary.csv", index=False)
    pct = ["avg", "median", "win %", "worst", "avg 2021-23", "avg 2024-26", "yearly (10%/trade)", "max drop"]
    view = df.copy()
    for c in pct:
        view[c] = view[c].map(lambda v: f"{v:+.1%}" if pd.notna(v) else "")
    pd.set_option("display.width", 250)
    print(view.to_string(index=False))
    best = df.sort_values("avg 2021-23", ascending=False).iloc[0]
    print(f"\nBest on 2021-23 listings: enter after {best['enter after']}, hold {best['hold']}, stop {best['stop']}: "
          f"avg {best['avg 2021-23']:+.1%} per trade -> on 2024-26 listings {best['avg 2024-26']:+.1%} per trade")
    by_year = trades.assign(year=trades["entry"].dt.year).query("delay == '1 day' and hold == '1 month'")
    print("\nEnter after 1 day, hold 1 month, average per trade by listing year:")
    print(by_year.pivot_table(index="year", columns="stop_label", values="ret", aggfunc="mean")
          .to_string(float_format=lambda v: f"{v:+.1%}"))


if __name__ == "__main__":
    main()
