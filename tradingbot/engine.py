import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .strategies import Strategy


@dataclass
class Position:
    symbol: str
    qty: float
    entry_price: float
    entry_fee: float
    stop: float
    risk_per_unit: float
    entry_date: str
    last_price: float
    side: int = 1
    funding_paid: float = 0.0
    initial_qty: float = 0.0
    realized_pnl: float = 0.0  # from partial exits, net of their fees
    partial_date: str | None = None


class Panel:
    """Prepared per-symbol frames stored as date x symbol arrays, so a day's bars are cheap to build."""

    def __init__(self, prepared: dict[str, pd.DataFrame]):
        self.symbols = list(prepared)
        self.col = {s: j for j, s in enumerate(self.symbols)}
        index = pd.DatetimeIndex(sorted(set().union(*(df.index for df in prepared.values()))))
        self.dates = [ts.strftime("%Y-%m-%d") for ts in index]
        columns = sorted(set().union(*(df.columns for df in prepared.values())))
        self.data = {
            c: pd.DataFrame({s: (df[c] if c in df else np.nan) for s, df in prepared.items()})
            .reindex(index)[self.symbols].to_numpy(dtype=float)
            for c in columns
        }

    def bars(self, i: int, watch: set[str], top_n: int | None) -> dict[str, dict]:
        has = ~np.isnan(self.data["close"][i])
        if top_n is None or "liq_rank" not in self.data:
            mask = has.copy()
        else:
            mask = has & (np.nan_to_num(self.data["liq_rank"][i], nan=np.inf) <= top_n)
            for sym in watch:
                j = self.col.get(sym)
                if j is not None and has[j]:
                    mask[j] = True
        return {self.symbols[j]: {c: arr[i, j] for c, arr in self.data.items()} for j in np.flatnonzero(mask)}


def run_days(account, panel: Panel, after: str | None = None, until: str | None = None,
             before_step=None, after_step=None) -> None:
    """Step through panel dates in (after, until]. Stops early if before_step returns False."""
    top_n = account.settings.get("universe_top_n")
    for i, date in enumerate(panel.dates):
        if (after and date <= after) or (until and date > until):
            continue
        bars = panel.bars(i, account.watch(), top_n)
        if before_step and before_step(date, bars) is False:
            return
        account.step(date, bars)
        if after_step:
            after_step(date)


def _eligible(bar: dict, top_n: int | None) -> bool:
    return top_n is None or bar.get("liq_rank", math.nan) <= top_n


def _order_rules(settings: dict, symbol: str) -> tuple[float, float]:
    """(minimum order value, quantity step) for a symbol; step 0 means no rounding."""
    f = settings.get("filters", {}).get(symbol, {})
    return f.get("min_notional", settings["min_notional"]), f.get("step", 0.0)


def _round_down(qty: float, step: float) -> float:
    if step <= 0:
        return qty
    return math.floor(qty / step + 1e-9) * step


class Portfolio:
    """Daily-bar portfolio for spot (long only) or USDT-M futures (long and short).

    Each step: pending exits and entries fill at the open, then stops are checked
    against the bar's range, then funding is charged and close-based signals are queued.
    """

    def __init__(self, strategy: Strategy, capital: float, settings: dict):
        self.strategy = strategy
        self.settings = settings
        self.start_capital = capital
        self.cash = capital
        self.positions: dict[str, Position] = {}
        self.pending_entries: list[dict] = []
        self.pending_exits: list[str] = []
        self.trades: list[dict] = []
        self.equity_curve: list[list] = []
        self.peak = capital
        self.halted = False
        self.last_date: str | None = None
        self.skipped_min_notional = 0
        self.below_min_orders = 0

    @property
    def futures(self) -> bool:
        return self.settings.get("market") == "futures"

    def watch(self) -> set[str]:
        return set(self.positions) | set(self.pending_exits) | {o["symbol"] for o in self.pending_entries}

    def equity(self, prices: dict[str, float] | None = None) -> float:
        prices = prices or {}
        if self.futures:
            return self.cash + sum(p.side * p.qty * (prices.get(s, p.last_price) - p.entry_price)
                                   for s, p in self.positions.items())
        return self.cash + sum(p.qty * prices.get(s, p.last_price) for s, p in self.positions.items())

    def _fill(self, price: float, side: int, opening: bool) -> float:
        buying = (side == 1) == opening
        return price * (1 + self.settings["slippage"]) if buying else price * (1 - self.settings["slippage"])

    def step(self, date: str, bars: dict[str, dict]) -> None:
        s = self.settings

        for sym in list(self.pending_exits):
            if sym in bars and sym in self.positions:
                pos = self.positions[sym]
                self._close(sym, self._fill(bars[sym]["open"], pos.side, False), date, "exit_signal")
            if sym not in self.positions:
                self.pending_exits.remove(sym)

        equity_at_open = self.equity({sym: b["open"] for sym, b in bars.items()})
        for order in self.pending_entries:
            sym = order["symbol"]
            if sym not in bars or sym in self.positions:
                continue
            if len(self.positions) >= s["max_positions"]:
                break
            side = order.get("side", 1)
            self._open(order, self._fill(bars[sym]["open"], side, True), equity_at_open, date)
        self.pending_entries = []

        take_r = s.get("partial_take_r")
        for sym, pos in list(self.positions.items()):
            bar = bars.get(sym)
            if bar is None:
                continue
            target = pos.entry_price + pos.side * take_r * pos.risk_per_unit if take_r else None
            can_take = target is not None and pos.partial_date is None
            gapped_past_target = can_take and pos.side * (bar["open"] - target) >= 0
            if gapped_past_target:
                self._take_partial(pos, bar["open"], date)
            if pos.side == 1 and bar["low"] <= pos.stop:
                self._close(sym, self._fill(min(bar["open"], pos.stop), 1, False), date, "stop")
            elif pos.side == -1 and bar["high"] >= pos.stop:
                self._close(sym, self._fill(max(bar["open"], pos.stop), -1, False), date, "stop")
            elif can_take and not gapped_past_target and pos.side * ((bar["high"] if pos.side == 1 else bar["low"])
                                                                     - target) >= 0:
                # Daily bars can't tell whether the stop or target came first; the stop is checked first.
                self._take_partial(pos, target, date)

        max_hold = self.strategy.max_hold_days
        for sym, pos in self.positions.items():
            bar = bars.get(sym)
            if bar is None:
                continue
            pos.last_price = bar["close"]
            if self.futures:
                funding = bar.get("funding", 0.0)
                cost = pos.side * pos.qty * bar["close"] * (0.0 if math.isnan(funding) else funding)
                self.cash -= cost
                pos.funding_paid += cost
            expired = max_hold is not None and (
                pd.Timestamp(date) - pd.Timestamp(pos.entry_date)).days >= max_hold
            signal = self.strategy.exit_signal(bar) if pos.side == 1 else self.strategy.short_exit_signal(bar)
            if (expired or signal) and sym not in self.pending_exits:
                self.pending_exits.append(sym)

        equity = self.equity()
        self.peak = max(self.peak, equity)
        limit = s.get("kill_switch_drawdown")
        if limit and equity < self.peak * (1 - limit):
            self.halted = True
        self.equity_curve.append([date, equity])

        if not self.halted:
            top_n = s.get("universe_top_n")
            allow_short = self.futures and s.get("allow_short", False)
            candidates = []
            for sym, bar in bars.items():
                if sym in self.positions or not _eligible(bar, top_n):
                    continue
                if self.strategy.entry_signal(bar):
                    side = 1
                elif allow_short and self.strategy.short_entry_signal(bar):
                    side = -1
                else:
                    continue
                candidates.append({"symbol": sym, "atr": bar["atr"], "score": bar["score"], "side": side})
            candidates.sort(key=lambda c: 0 if math.isnan(c["score"]) else -c["side"] * c["score"])
            self.pending_entries = candidates[: max(s["max_positions"] - len(self.positions), 0)]

        self.last_date = date

    def _open(self, order: dict, price: float, equity: float, date: str) -> None:
        s = self.settings
        side = order.get("side", 1)
        rpu = self.strategy.stop_atr * order["atr"]
        if not rpu > 0 or (side == 1 and rpu >= price):
            return
        open_risk = sum(p.risk_per_unit * p.qty for p in self.positions.values())
        risk_budget = min(s["risk_per_trade"] * equity, s["max_open_risk"] * equity - open_risk)
        if risk_budget <= 0:
            return
        caps = [risk_budget / rpu * price, s["max_position_frac"] * equity]
        if self.futures:
            gross = sum(p.qty * p.last_price for p in self.positions.values())
            caps.append(s.get("max_gross_leverage", 1.0) * equity - gross)
        else:
            caps.append(self.cash / (1 + s["fee"]))
        notional = min(caps)
        if notional <= 0:
            return
        minimum, step = _order_rules(s, order["symbol"])
        qty = _round_down(notional / price, step)
        notional = qty * price
        if notional < minimum or qty <= 0:
            if s.get("enforce_min_notional", True) or qty <= 0:
                self.skipped_min_notional += 1
                return
            self.below_min_orders += 1
        fee = notional * s["fee"]
        self.cash -= fee if self.futures else notional + fee
        self.positions[order["symbol"]] = Position(
            order["symbol"], qty, price, fee, price - side * rpu, rpu, date, price, side, initial_qty=qty
        )

    def _take_partial(self, pos: Position, price: float, date: str) -> None:
        """Sell `partial_frac` of the position at `price` (a resting limit, so slippage still applies)."""
        s = self.settings
        minimum, step = _order_rules(s, pos.symbol)
        qty = _round_down(pos.qty * s.get("partial_frac", 0.5), step)
        fill = self._fill(price, pos.side, False)
        pos.partial_date = date
        if qty <= 0 or qty >= pos.qty or (qty * fill < minimum and s.get("enforce_min_notional", True)):
            pos.partial_date = f"skipped {date}"
            return
        fee = qty * fill * s["fee"]
        gross = pos.side * qty * (fill - pos.entry_price)
        self.cash += (gross if self.futures else qty * fill) - fee
        pos.realized_pnl += gross - fee
        pos.qty -= qty

    def _close(self, sym: str, price: float, date: str, reason: str) -> None:
        pos = self.positions.pop(sym)
        fee = pos.qty * price * self.settings["fee"]
        if self.futures:
            gross = pos.side * pos.qty * (price - pos.entry_price)
            self.cash += gross - fee
            pnl = gross - fee - pos.entry_fee - pos.funding_paid
        else:
            self.cash += pos.qty * price - fee
            pnl = pos.qty * (price - pos.entry_price) - fee - pos.entry_fee
        pnl += pos.realized_pnl
        initial = pos.initial_qty or pos.qty
        self.trades.append({
            "symbol": sym,
            "side": pos.side,
            "entry_date": pos.entry_date,
            "exit_date": date,
            "entry_price": pos.entry_price,
            "exit_price": price,
            "qty": initial,
            "partial_date": pos.partial_date,
            "funding": pos.funding_paid,
            "pnl": pnl,
            "r": pnl / (pos.risk_per_unit * initial),
            "reason": reason,
        })

    def to_dict(self) -> dict:
        return {
            "kind": "portfolio",
            "strategy": self.strategy.name,
            "settings": self.settings,
            "start_capital": self.start_capital,
            "cash": self.cash,
            "positions": {s: asdict(p) for s, p in self.positions.items()},
            "pending_entries": self.pending_entries,
            "pending_exits": self.pending_exits,
            "trades": self.trades,
            "equity_curve": self.equity_curve,
            "peak": self.peak,
            "halted": self.halted,
            "last_date": self.last_date,
            "skipped_min_notional": self.skipped_min_notional,
            "below_min_orders": self.below_min_orders,
        }

    @classmethod
    def from_dict(cls, d: dict, strategy: Strategy) -> "Portfolio":
        p = cls(strategy, d["start_capital"], d["settings"])
        p.cash = d["cash"]
        p.positions = {s: Position(**v) for s, v in d["positions"].items()}
        for pos in p.positions.values():
            if not pos.initial_qty:
                pos.initial_qty = pos.qty
        p.pending_entries = d["pending_entries"]
        p.pending_exits = d["pending_exits"]
        p.trades = d["trades"]
        p.equity_curve = d["equity_curve"]
        p.peak = d["peak"]
        p.halted = d["halted"]
        p.last_date = d["last_date"]
        p.skipped_min_notional = d["skipped_min_notional"]
        p.below_min_orders = d.get("below_min_orders", 0)
        return p

    def save(self, path: Path) -> None:
        _save(self.to_dict(), path)


class WeightPortfolio:
    """Spot portfolio that rebalances to the strategy's target weights at each open.

    Targets come from the previous close. A holding is left alone when its change is
    smaller than `rebalance_band` of its current size; sells execute before buys.
    """

    def __init__(self, strategy, capital: float, settings: dict):
        self.strategy = strategy
        self.settings = settings
        self.start_capital = capital
        self.cash = capital
        self.holdings: dict[str, dict] = {}  # symbol -> {"qty", "cost", "last_price"}
        self.targets: dict[str, float] = {}
        self.trades: list[dict] = []
        self.equity_curve: list[list] = []
        self.peak = capital
        self.halted = False
        self.last_date: str | None = None
        self.skipped_min_notional = 0
        self.below_min_orders = 0

    def watch(self) -> set[str]:
        return set(self.holdings) | set(self.targets)

    def equity(self, prices: dict[str, float] | None = None) -> float:
        prices = prices or {}
        return self.cash + sum(h["qty"] * prices.get(s, h["last_price"]) for s, h in self.holdings.items())

    def step(self, date: str, bars: dict[str, dict]) -> None:
        s = self.settings
        equity = self.equity({sym: b["open"] for sym, b in bars.items()})
        orders = []
        for sym in sorted(set(self.holdings) | set(self.targets)):
            bar = bars.get(sym)
            if bar is None:
                continue
            held = self.holdings.get(sym, {}).get("qty", 0.0) * bar["open"]
            target = self.targets.get(sym, 0.0) * equity
            delta = target - held
            if delta == 0 or (target > 0 and held > 0 and abs(delta) < s["rebalance_band"] * held):
                continue
            minimum, _ = _order_rules(s, sym)
            if abs(delta) < minimum:
                if s.get("enforce_min_notional", True):
                    self.skipped_min_notional += 1
                    continue
                self.below_min_orders += 1
            orders.append((sym, delta, bar["open"]))

        for sym, delta, open_px in sorted(orders, key=lambda o: o[1]):
            if delta < 0:
                held_qty = self.holdings[sym]["qty"]
                qty = held_qty if not self.targets.get(sym) else min(-delta / open_px, held_qty)
                self._sell(sym, qty, open_px, date)
            else:
                self._buy(sym, delta, open_px, date)

        for sym, h in self.holdings.items():
            if sym in bars:
                h["last_price"] = bars[sym]["close"]
        equity = self.equity()
        self.peak = max(self.peak, equity)
        limit = s.get("kill_switch_drawdown")
        if limit and equity < self.peak * (1 - limit):
            self.halted = True
        self.equity_curve.append([date, equity])
        self.targets = {} if self.halted else self.strategy.target_weights(bars, s)
        self.last_date = date

    def _buy(self, sym: str, notional: float, open_px: float, date: str) -> None:
        s = self.settings
        price = open_px * (1 + s["slippage"])
        minimum, step = _order_rules(s, sym)
        qty = _round_down(min(notional, self.cash / (1 + s["fee"])) / price, step)
        notional = qty * price
        if qty <= 0 or (notional < minimum and s.get("enforce_min_notional", True)):
            return
        fee = notional * s["fee"]
        self.cash -= notional + fee
        h = self.holdings.setdefault(sym, {"qty": 0.0, "cost": 0.0, "last_price": price})
        h["qty"] += notional / price
        h["cost"] += notional + fee
        self.trades.append({"date": date, "symbol": sym, "side": "buy", "price": price,
                            "notional": notional, "fee": fee, "pnl": 0.0})

    def _sell(self, sym: str, qty: float, open_px: float, date: str) -> None:
        s = self.settings
        h = self.holdings[sym]
        if h["qty"] - qty < 1e-12 * max(h["qty"], 1.0):
            qty = h["qty"]
        else:
            qty = _round_down(qty, _order_rules(s, sym)[1])
            if qty <= 0:
                return
        price = open_px * (1 - s["slippage"])
        proceeds = qty * price
        fee = proceeds * s["fee"]
        basis = h["cost"] * qty / h["qty"]
        self.cash += proceeds - fee
        h["qty"] -= qty
        h["cost"] -= basis
        if h["qty"] <= 0:
            del self.holdings[sym]
        self.trades.append({"date": date, "symbol": sym, "side": "sell", "price": price,
                            "notional": proceeds, "fee": fee, "pnl": proceeds - fee - basis})

    def to_dict(self) -> dict:
        return {
            "kind": "weights",
            "strategy": self.strategy.name,
            "settings": self.settings,
            "start_capital": self.start_capital,
            "cash": self.cash,
            "holdings": self.holdings,
            "targets": self.targets,
            "trades": self.trades,
            "equity_curve": self.equity_curve,
            "peak": self.peak,
            "halted": self.halted,
            "last_date": self.last_date,
            "skipped_min_notional": self.skipped_min_notional,
            "below_min_orders": self.below_min_orders,
        }

    @classmethod
    def from_dict(cls, d: dict, strategy) -> "WeightPortfolio":
        p = cls(strategy, d["start_capital"], d["settings"])
        for key in ("cash", "holdings", "targets", "trades", "equity_curve", "peak", "halted", "last_date",
                    "skipped_min_notional", "below_min_orders"):
            setattr(p, key, d[key])
        return p

    def save(self, path: Path) -> None:
        _save(self.to_dict(), path)


def _save(d: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1, default=float))
    tmp.replace(path)


def load_account(path: Path, strategy):
    d = json.loads(path.read_text())
    return (WeightPortfolio if d.get("kind") == "weights" else Portfolio).from_dict(d, strategy)


def new_account(strategy, capital: float, settings: dict):
    return (WeightPortfolio if hasattr(strategy, "target_weights") else Portfolio)(strategy, capital, settings)
