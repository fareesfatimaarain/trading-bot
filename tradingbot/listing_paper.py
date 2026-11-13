"""Paper trading for the new-listing short strategy (same rules and costs as listing_shorts.py).

Each run: find USDT perps that launched recently, skip coins with an older spot market, and short each one
at the close of the minute exactly 24h after its first trade (only if that moment is after the account was
created). Positions are 10% of equity, at most 10 open; they close after the hold period, at the stop, or at
the last price if the contract is delisted. Funding is collected or paid as on Binance.
"""
import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests

from . import config
from .listing_shorts import FEE, STOP_SLIP

FAPI = "https://fapi.binance.com/fapi/v1"
SPOT = "https://data-api.binance.vision/api/v3"
ACCOUNTS = {
    "listing_short_1m": {"hold_days": 30, "stop": None},
    "listing_short_1w": {"hold_days": 7, "stop": 0.30},
}
FRACTION = 0.10
MAX_OPEN = 10
ENTRY_DELAY = pd.Timedelta(days=1)
ENTRY_COST = FEE + 0.001
EXIT_COST = FEE + 0.001
log = logging.getLogger("paper")


def _get(url: str, params: dict | None = None):
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 400:
                return []
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"Binance request failed: {url} {params}")


def _first_kline_ms(url: str, symbol: str) -> int | None:
    k = _get(url, {"symbol": symbol, "interval": "1m", "startTime": 0, "limit": 1})
    return k[0][0] if k else None


def _is_new_coin(symbol: str, perp_t0: int) -> bool:
    base = symbol[:-4]
    for prefix in ("1000000", "10000", "1000", "1M"):
        if base.startswith(prefix) and len(base) > len(prefix):
            base = base[len(prefix):]
            break
    spot_t0 = _first_kline_ms(f"{SPOT}/klines", f"{base}USDT")
    return spot_t0 is None or spot_t0 >= perp_t0 - 30 * 86_400_000


def _min_close_at(symbol: str, when: pd.Timestamp) -> float | None:
    """Close of the 1-minute bar that ends at `when`."""
    start = int((when - pd.Timedelta(minutes=1)).timestamp() * 1000)
    k = _get(f"{FAPI}/klines", {"symbol": symbol, "interval": "1m", "startTime": start, "limit": 1})
    return float(k[0][4]) if k and abs(k[0][0] - start) < 60_000 else None


class ListingAccount:
    def __init__(self, name: str, hold_days: int, stop: float | None, capital: float, created: str):
        self.name, self.hold_days, self.stop = name, hold_days, stop
        self.start_capital = self.cash = capital
        self.created = created
        self.positions: list[dict] = []
        self.trades: list[dict] = []
        self.seen: list[str] = []
        self.skipped: list[dict] = []

    def equity(self) -> float:
        return self.cash + sum(p["notional"] * p["unrealised"] for p in self.positions)

    def to_dict(self) -> dict:
        return {"kind": "listing", **{k: getattr(self, k) for k in (
            "name", "hold_days", "stop", "start_capital", "cash", "created", "positions", "trades", "seen", "skipped")}}

    @classmethod
    def from_dict(cls, d: dict) -> "ListingAccount":
        a = cls(d["name"], d["hold_days"], d["stop"], d["start_capital"], d["created"])
        for k in ("cash", "positions", "trades", "seen", "skipped"):
            setattr(a, k, d[k])
        return a

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=1))
        tmp.replace(path)


def _position_return(p: dict, path: pd.DataFrame, funding: pd.Series, now: pd.Timestamp, stop: float | None):
    """(return so far, closed?, exit price, reason) using the backtest's formulas."""
    entry_t, exit_t = pd.Timestamp(p["entry_time"]), pd.Timestamp(p["exit_time"])
    if funding.empty:
        funding = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    window = path[(path.index >= entry_t) & (path.index < min(exit_t, now))]
    received = funding[(funding.index > entry_t) & (funding.index <= min(exit_t, now))].sum()
    if stop is not None and len(window):
        hit = window[window["high"] >= p["entry"] * (1 + stop)]
        if len(hit):
            px = max(hit["open"].iat[0], p["entry"] * (1 + stop))
            received = funding[(funding.index > entry_t) & (funding.index <= hit.index[0])].sum()
            return -(px / p["entry"] - 1) + received - ENTRY_COST - EXIT_COST - STOP_SLIP, True, px, "stop"
    if len(window) == 0:
        return -ENTRY_COST, False, p["entry"], ""
    px = window["close"].iat[-1]
    closed = now >= exit_t
    ret = -(px / p["entry"] - 1) + received - ENTRY_COST - (EXIT_COST if closed else 0.0)
    return max(ret, -1.0), closed, px, "hold period over" if closed else ""


def run(accounts: dict[str, ListingAccount]) -> None:
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    info = _get(f"{FAPI}/exchangeInfo")["symbols"]
    # crypto coins only: stocks/commodities are TRADIFI_PERPETUAL, and index baskets have underlyingType INDEX
    live = {s["symbol"]: s for s in info if s["quoteAsset"] == "USDT" and s["contractType"] == "PERPETUAL"
            and s.get("underlyingType", "COIN") == "COIN"}
    min_notional = {s["symbol"]: float(next((f["notional"] for f in s["filters"] if f["filterType"] == "MIN_NOTIONAL"),
                                            5)) for s in live.values()}

    # 1. update / close open positions
    for acct in accounts.values():
        for p in list(acct.positions):
            sym = p["symbol"]
            start = int(pd.Timestamp(p["entry_time"]).timestamp() * 1000)
            bars = _get(f"{FAPI}/klines", {"symbol": sym, "interval": "1h", "startTime": start, "limit": 1500})
            path = pd.DataFrame([b[:5] for b in bars], columns=["t", "open", "high", "low", "close"]).astype(float)
            path.index = pd.to_datetime(path.pop("t"), unit="ms")
            fund = _get(f"{FAPI}/fundingRate", {"symbol": sym, "startTime": start, "limit": 1000})
            funding = pd.Series({pd.Timestamp(f["fundingTime"], unit="ms"): float(f["fundingRate"]) for f in fund},
                                dtype=float)
            delisted = sym not in live or live[sym]["status"] != "TRADING"
            ret, closed, px, reason = _position_return(p, path, funding, now, acct.stop)
            if delisted and not closed:
                closed, reason = True, "delisted"
            p["unrealised"], p["last"] = ret, px
            if closed:
                acct.positions.remove(p)
                acct.cash += p["notional"] * (1 + ret)
                acct.trades.append({**{k: p[k] for k in ("symbol", "entry_time", "entry", "notional")},
                                    "exit_time": str(min(now, pd.Timestamp(p["exit_time"]))), "exit": px,
                                    "ret": ret, "pnl": p["notional"] * ret, "reason": reason})
                log.info("[%s] closed SHORT %s @ %.6g (%s): %+.1f%% = $%+.2f", acct.name, sym, px, reason,
                         ret * 100, p["notional"] * ret)

    # 2. open shorts on new listings whose 24h mark has passed
    recent = [s for s in live.values() if s["status"] == "TRADING"
              and s["onboardDate"] >= (now - pd.Timedelta(days=10)).timestamp() * 1000]
    for s in recent:
        sym = s["symbol"]
        if all(sym in a.seen for a in accounts.values()):
            continue
        t0 = _first_kline_ms(f"{FAPI}/klines", sym)
        if t0 is None:
            continue
        entry_t = pd.Timestamp(t0, unit="ms") + ENTRY_DELAY
        if entry_t > now:
            continue  # not 24h old yet; check again next run
        new_coin = _is_new_coin(sym, t0)
        entry_px = _min_close_at(sym, entry_t) if new_coin else None
        for acct in accounts.values():
            if sym in acct.seen:
                continue
            acct.seen.append(sym)
            if entry_t < pd.Timestamp(acct.created):
                continue  # listed before this account existed: no backfilling
            if not new_coin:
                log.info("[%s] %s skipped: coin already had a spot market for over 30 days", acct.name, sym)
                continue
            notional = acct.equity() * FRACTION
            why = ("10 shorts already open" if len(acct.positions) >= MAX_OPEN else
                   "no price at the 24h mark" if entry_px is None else
                   f"below Binance minimum ${min_notional.get(sym, 5):g}" if notional < min_notional.get(sym, 5)
                   else None)
            if why:
                acct.skipped.append({"symbol": sym, "time": str(entry_t), "why": why})
                log.info("[%s] %s not shorted: %s", acct.name, sym, why)
                continue
            acct.cash -= notional
            acct.positions.append({"symbol": sym, "entry_time": str(entry_t), "entry": entry_px,
                                   "exit_time": str(entry_t + pd.Timedelta(days=acct.hold_days)),
                                   "notional": notional, "unrealised": -ENTRY_COST, "last": entry_px})
            late = (now - entry_t) / pd.Timedelta(hours=1)
            log.info("[%s] opened SHORT %s $%.2f @ %.6g (24h after listing)%s", acct.name, sym, notional, entry_px,
                     f"; processed {late:.0f}h late" if late > 2 else "")


def run_all() -> None:
    paths = {name: config.PAPER_DIR / f"{name}_state.json" for name in ACCOUNTS}
    accounts = {}
    for name, cfg in ACCOUNTS.items():
        if paths[name].exists():
            accounts[name] = ListingAccount.from_dict(json.loads(paths[name].read_text()))
        else:
            created = str(pd.Timestamp.now(tz="UTC").tz_localize(None).floor("min"))
            accounts[name] = ListingAccount(name, cfg["hold_days"], cfg["stop"], config.PAPER_CAPITAL, created)
            log.info("[%s] new paper account with $%.2f; shorts listings from %s on", name, config.PAPER_CAPITAL, created)
    run(accounts)
    for name, acct in accounts.items():
        acct.save(paths[name])
        log.info("[%s] run ok: equity $%.2f, %d open shorts", name, acct.equity(), len(acct.positions))
