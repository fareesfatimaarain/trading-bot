import numpy as np
import pandas as pd
import pytest

from tradingbot import config
from tradingbot.engine import Portfolio
from tradingbot.strategies import DipBuyer, DonchianEnsemble, MomentumBreakout, Strategy, TurtleTrend


class Scripted(Strategy):
    name = "scripted"
    stop_atr = 2.0

    def entry_signal(self, bar):
        return bar.get("enter", False)

    def exit_signal(self, bar):
        return bar.get("exit", False)


def bar(o, h, l, c, **flags):
    return {"open": o, "high": h, "low": l, "close": c, "atr": 1.0, "score": 0.0, **flags}


def settings(**over):
    s = config.default_settings()
    s.update(over)
    return s


def test_round_trip_costs_fees_and_slippage():
    s = settings(risk_per_trade=1.0, max_open_risk=1.0, max_position_frac=0.5)
    p = Portfolio(Scripted(), 1000.0, s)
    p.step("d1", {"X": bar(100, 100, 100, 100, enter=True)})
    p.step("d2", {"X": bar(100, 100, 100, 100, exit=True)})
    p.step("d3", {"X": bar(100, 100, 100, 100)})
    assert not p.positions
    (t,) = p.trades
    notional = 500.0
    buy_px, sell_px = 100 * 1.0005, 100 * 0.9995
    qty = notional / buy_px
    expected_cash = 1000 - notional * 1.001 + qty * sell_px * 0.999
    assert p.cash == pytest.approx(expected_cash)
    assert t["pnl"] == pytest.approx(expected_cash - 1000)


def test_stop_fills_at_open_when_price_gaps_below_stop():
    p = Portfolio(Scripted(), 1000.0, settings())
    p.step("d1", {"X": bar(100, 100, 100, 100, enter=True)})
    p.step("d2", {"X": bar(100, 101, 99, 100)})
    stop = p.positions["X"].stop
    p.step("d3", {"X": bar(90, 91, 85, 88)})
    (t,) = p.trades
    assert t["reason"] == "stop"
    assert t["exit_price"] == pytest.approx(90 * 0.9995)
    assert stop > 90


def test_stop_fills_at_stop_when_touched_intraday():
    p = Portfolio(Scripted(), 1000.0, settings())
    p.step("d1", {"X": bar(100, 100, 100, 100, enter=True)})
    p.step("d2", {"X": bar(100, 101, 95, 99)})
    (t,) = p.trades
    assert t["reason"] == "stop"
    assert t["exit_price"] == pytest.approx((100 * 1.0005 - 2.0) * 0.9995)


def test_risk_per_trade_sets_position_size():
    p = Portfolio(Scripted(), 1000.0, settings(max_position_frac=1.0))
    p.step("d1", {"X": bar(100, 100, 100, 100, enter=True)})
    p.step("d2", {"X": bar(100, 100, 100, 100)})
    pos = p.positions["X"]
    assert pos.risk_per_unit * pos.qty == pytest.approx(10.0)  # 1% of 1000


def test_orders_below_minimum_notional_are_skipped():
    p = Portfolio(Scripted(), 72.0, settings())
    p.step("d1", {"X": {**bar(100, 100, 100, 100, enter=True), "atr": 40.0}})
    p.step("d2", {"X": {**bar(100, 100, 100, 100), "atr": 40.0}})
    assert not p.positions
    assert p.skipped_min_notional == 1


def test_never_spends_more_cash_than_available():
    s = settings(risk_per_trade=1.0, max_open_risk=1.0, max_position_frac=1.0)
    p = Portfolio(Scripted(), 100.0, s)
    p.step("d1", {"X": bar(100, 100, 100, 100, enter=True), "Y": bar(50, 50, 50, 50, enter=True)})
    p.step("d2", {"X": bar(100, 100, 100, 100), "Y": bar(50, 50, 50, 50)})
    assert p.cash >= -1e-9


def test_kill_switch_halts_new_entries():
    p = Portfolio(Scripted(), 1000.0, settings(kill_switch_drawdown=0.01, max_position_frac=1.0))
    p.step("d1", {"X": bar(100, 100, 100, 100, enter=True)})
    p.step("d2", {"X": bar(100, 100, 99, 99)})
    p.step("d3", {"X": bar(80, 80, 70, 75), "Y": bar(10, 10, 10, 10, enter=True)})
    assert p.halted
    assert p.pending_entries == []


def test_state_round_trip():
    p = Portfolio(Scripted(), 1000.0, settings())
    p.step("d1", {"X": bar(100, 100, 100, 100, enter=True)})
    p.step("d2", {"X": bar(100, 101, 99, 100)})
    q = Portfolio.from_dict(p.to_dict(), Scripted())
    assert q.to_dict() == p.to_dict()


def _random_ohlcv(n=600, seed=1):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.03, n)))
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "open": close * (1 + rng.normal(0, 0.005, n)),
        "high": close * 1.02, "low": close * 0.98, "close": close,
        "volume": rng.uniform(1, 2, n), "quote_volume": rng.uniform(1e6, 3e6, n),
    }, index=idx)


def test_max_hold_days_forces_exit():
    class Timed(Scripted):
        max_hold_days = 2

    p = Portfolio(Timed(), 1000.0, settings())
    p.step("2026-01-01", {"X": bar(100, 100, 100, 100, enter=True)})
    p.step("2026-01-02", {"X": bar(100, 101, 99, 100)})
    p.step("2026-01-03", {"X": bar(100, 101, 99, 100)})
    assert p.pending_exits == []
    p.step("2026-01-04", {"X": bar(100, 101, 99, 100)})
    assert p.pending_exits == ["X"]
    p.step("2026-01-05", {"X": bar(100, 101, 99, 100)})
    assert not p.positions and p.trades[0]["reason"] == "exit_signal"


@pytest.mark.parametrize("strategy", [TurtleTrend(), MomentumBreakout(), DipBuyer(), DonchianEnsemble()])
def test_indicators_do_not_look_ahead(strategy):
    universe = {config.BENCHMARK: _random_ohlcv(seed=1), "ALTUSDT": _random_ohlcv(seed=2)}
    cut = 450
    base = strategy.prepare_universe(universe)
    altered = {s: df.copy() for s, df in universe.items()}
    for df in altered.values():
        df.iloc[cut + 1:] = df.iloc[cut + 1:] * 3.0
    changed = strategy.prepare_universe(altered)
    for sym in universe:
        pd.testing.assert_frame_equal(base[sym].iloc[: cut + 1], changed[sym].iloc[: cut + 1])
