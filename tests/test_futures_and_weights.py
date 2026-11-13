import numpy as np
import pandas as pd
import pytest

from tradingbot import config
from tradingbot.engine import Panel, Portfolio, WeightPortfolio, load_account
from tradingbot.strategies import DonchianEnsemble, Strategy, add_liquidity_rank, donchian_ensemble_signal


class Scripted(Strategy):
    name = "scripted"
    stop_atr = 2.0

    def entry_signal(self, bar):
        return bar.get("enter", False)

    def exit_signal(self, bar):
        return bar.get("exit", False)

    def short_entry_signal(self, bar):
        return bar.get("short", False)

    def short_exit_signal(self, bar):
        return bar.get("cover", False)


def bar(o, h, l, c, **extra):
    return {"open": o, "high": h, "low": l, "close": c, "atr": 1.0, "score": 0.0, **extra}


def fut_settings(**over):
    s = {**config.default_settings(), "market": "futures", "fee": 0.0005, "allow_short": True,
         "max_gross_leverage": 2.0, "max_position_frac": 0.5}
    s.update(over)
    return s


def test_short_profits_when_price_falls():
    p = Portfolio(Scripted(), 1000.0, fut_settings())
    p.step("2026-01-01", {"X": bar(100, 100, 100, 100, short=True)})
    p.step("2026-01-02", {"X": bar(100, 100, 100, 100)})
    pos = p.positions["X"]
    assert pos.side == -1 and pos.stop == pytest.approx(100 * 0.9995 + 2.0)
    p.step("2026-01-03", {"X": bar(99, 99, 98.5, 99, cover=True)})
    p.step("2026-01-04", {"X": bar(98, 98, 98, 98)})
    (t,) = p.trades
    entry, exit_ = 100 * 0.9995, 98 * 1.0005
    expected = pos.qty * (entry - exit_) - pos.qty * entry * 0.0005 - pos.qty * exit_ * 0.0005
    assert t["pnl"] == pytest.approx(expected)
    assert p.cash == pytest.approx(1000 + expected)


def test_short_stop_triggers_on_high():
    p = Portfolio(Scripted(), 1000.0, fut_settings())
    p.step("2026-01-01", {"X": bar(100, 100, 100, 100, short=True)})
    p.step("2026-01-02", {"X": bar(100, 103, 99, 101)})
    (t,) = p.trades
    assert t["reason"] == "stop"
    assert t["exit_price"] == pytest.approx((100 * 0.9995 + 2.0) * 1.0005)


def test_shorts_disabled_on_spot():
    s = {**config.default_settings(), "allow_short": True}
    p = Portfolio(Scripted(), 1000.0, s)
    p.step("2026-01-01", {"X": bar(100, 100, 100, 100, short=True)})
    assert p.pending_entries == []


def test_funding_paid_by_longs_received_by_shorts():
    for flag, side in (("enter", 1), ("short", -1)):
        p = Portfolio(Scripted(), 1000.0, fut_settings())
        p.step("2026-01-01", {"X": bar(100, 100, 100, 100, **{flag: True})})
        p.step("2026-01-02", {"X": bar(100, 100, 100, 100, funding=0.001)})
        pos = p.positions["X"]
        assert pos.funding_paid == pytest.approx(side * pos.qty * 100 * 0.001)
        assert p.equity() == pytest.approx(1000 - pos.entry_fee - pos.funding_paid
                                           + side * pos.qty * (100 - pos.entry_price))


def test_futures_gross_leverage_cap():
    p = Portfolio(Scripted(), 1000.0, fut_settings(risk_per_trade=1.0, max_open_risk=10.0,
                                                   max_position_frac=10.0, max_gross_leverage=1.5))
    p.step("2026-01-01", {"X": bar(100, 100, 100, 100, enter=True), "Y": bar(100, 100, 100, 100, short=True)})
    p.step("2026-01-02", {"X": bar(100, 100, 100, 100), "Y": bar(100, 100, 100, 100)})
    gross = sum(pos.qty * pos.entry_price for pos in p.positions.values())
    assert gross == pytest.approx(1500, rel=1e-3)


def test_ineligible_coins_are_not_entered():
    p = Portfolio(Scripted(), 1000.0, {**config.default_settings(), "universe_top_n": 2})
    p.step("2026-01-01", {"X": bar(100, 100, 100, 100, enter=True, liq_rank=3.0),
                          "Y": bar(100, 100, 100, 100, enter=True, liq_rank=1.0),
                          "Z": bar(100, 100, 100, 100, enter=True, liq_rank=np.nan)})
    assert [o["symbol"] for o in p.pending_entries] == ["Y"]


def test_min_notional_counted_not_enforced_when_disabled():
    p = Portfolio(Scripted(), 72.0, {**config.default_settings(), "enforce_min_notional": False})
    p.step("2026-01-01", {"X": {**bar(100, 100, 100, 100, enter=True), "atr": 40.0}})
    p.step("2026-01-02", {"X": {**bar(100, 100, 100, 100), "atr": 40.0}})
    assert "X" in p.positions and p.below_min_orders == 1


def test_quantity_rounds_down_to_exchange_step():
    s = {**config.default_settings(), "max_position_frac": 1.0,
         "filters": {"X": {"min_notional": 5.0, "step": 0.3}}}
    p = Portfolio(Scripted(), 1000.0, s)
    p.step("2026-01-01", {"X": bar(100, 100, 100, 100, enter=True)})
    p.step("2026-01-02", {"X": bar(100, 100, 100, 100)})
    qty = p.positions["X"].qty
    assert qty == pytest.approx(4.8)  # $10 risk / $2 stop = 5 units, rounded down to a multiple of 0.3
    assert p.cash == pytest.approx(1000 - 4.8 * 100.05 * 1.001)


def test_per_symbol_minimum_blocks_order():
    s = {**config.default_settings(), "max_position_frac": 1.0,
         "filters": {"BTC": {"min_notional": 50.0, "step": 0.0}, "ALT": {"min_notional": 5.0, "step": 0.0}}}
    p = Portfolio(Scripted(), 72.0, s)
    p.step("2026-01-01", {"BTC": bar(100, 100, 100, 100, enter=True), "ALT": bar(100, 100, 100, 100, enter=True)})
    p.step("2026-01-02", {"BTC": bar(100, 100, 100, 100), "ALT": bar(100, 100, 100, 100)})
    assert set(p.positions) == {"ALT"}  # $36 order clears ALT's $5 minimum but not BTC's $50
    assert p.skipped_min_notional == 1


def test_weight_portfolio_skips_orders_below_minimum():
    class Fixed:
        name = "fixed"

        def target_weights(self, bars, s):
            return {"X": 0.03}

    p = WeightPortfolio(Fixed(), 72.0, {**config.default_settings(), "rebalance_band": 0.25})
    p.step("2026-01-01", {"X": bar(100, 100, 100, 100)})
    p.step("2026-01-02", {"X": bar(100, 100, 100, 100)})
    assert p.holdings == {} and p.skipped_min_notional == 1  # 3% of $72 = $2.16 < $5


def _partial_settings(**over):
    return {**config.default_settings(), "max_position_frac": 1.0, "partial_take_r": 2.0, "partial_frac": 0.5,
            "slippage": 0.0, "fee": 0.0, **over}


def _opened(settings, market_bar=bar(100, 100, 100, 100)):
    p = Portfolio(Scripted(), 1000.0, settings)
    p.step("2026-01-01", {"X": bar(100, 100, 100, 100, enter=True)})
    p.step("2026-01-02", {"X": market_bar})
    return p  # long 5 units at 100, stop 98, 2R target 104


def test_partial_take_sells_half_at_target_and_counts_as_win_when_rest_stops_out():
    p = _opened(_partial_settings())
    p.step("2026-01-03", {"X": bar(101, 105, 100, 103)})
    pos = p.positions["X"]
    assert pos.qty == pytest.approx(2.5) and pos.realized_pnl == pytest.approx(2.5 * 4)
    p.step("2026-01-04", {"X": bar(99, 99, 97, 97)})
    (t,) = p.trades
    assert t["pnl"] == pytest.approx(2.5 * 4 - 2.5 * 2)  # +10 on the half, -5 on the rest
    assert t["r"] == pytest.approx(0.5) and t["qty"] == pytest.approx(5)
    assert p.cash == pytest.approx(1000 + 5)


def test_partial_fills_at_open_when_price_gaps_past_target():
    p = _opened(_partial_settings())
    p.step("2026-01-03", {"X": bar(110, 111, 109, 110)})
    assert p.positions["X"].realized_pnl == pytest.approx(2.5 * 10)


def test_stop_wins_when_bar_hits_both_stop_and_target():
    p = _opened(_partial_settings())
    p.step("2026-01-03", {"X": bar(100, 105, 97, 100)})
    (t,) = p.trades
    assert t["reason"] == "stop" and t["partial_date"] is None and t["r"] == pytest.approx(-1.0)


def test_partial_take_only_once_and_short_side_mirrors():
    s = _partial_settings(market="futures", allow_short=True)
    p = Portfolio(Scripted(), 1000.0, s)
    p.step("2026-01-01", {"X": bar(100, 100, 100, 100, short=True)})
    p.step("2026-01-02", {"X": bar(100, 100, 100, 100)})  # short 5 at 100, stop 102, target 96
    p.step("2026-01-03", {"X": bar(99, 100, 95, 96)})
    p.step("2026-01-04", {"X": bar(96, 96, 90, 92)})
    pos = p.positions["X"]
    assert pos.qty == pytest.approx(2.5) and pos.realized_pnl == pytest.approx(2.5 * 4)


def _frame(closes, volume=1e6, start="2024-01-01"):
    idx = pd.date_range(start, periods=len(closes), freq="D")
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1.0, "quote_volume": volume})


def test_liquidity_rank_requires_age_and_orders_by_volume():
    old_big = _frame([1.0] * 400, volume=5e6)
    old_small = _frame([1.0] * 400, volume=1e6)
    young = _frame([1.0] * 100, volume=9e9, start="2024-12-04")
    prepared = {"A": old_big, "B": old_small, "C": young}
    add_liquidity_rank(prepared)
    last = old_big.index[-1]
    assert prepared["A"].loc[last, "liq_rank"] == 1
    assert prepared["B"].loc[last, "liq_rank"] == 2
    assert np.isnan(prepared["C"].loc[last, "liq_rank"])


def test_panel_keeps_held_symbols_even_when_ineligible():
    prepared = {"A": _frame([1, 2, 3]), "B": _frame([1, 2, 3])}
    prepared["A"]["liq_rank"] = 1.0
    prepared["B"]["liq_rank"] = 99.0
    panel = Panel(prepared)
    assert set(panel.bars(2, set(), top_n=10)) == {"A"}
    assert set(panel.bars(2, {"B"}, top_n=10)) == {"A", "B"}


def test_donchian_signal_enters_on_breakout_and_trails_out():
    closes = pd.Series([10, 10, 10, 11, 12, 13, 9, 9], dtype=float)
    sig = donchian_ensemble_signal(closes, (3,))
    assert list(sig) == [0, 0, 0, 1, 1, 1, 0, 0]


def test_weight_portfolio_rebalances_and_exits():
    class Fixed:
        name = "fixed"
        weights = {"X": 0.5}

        def target_weights(self, bars, s):
            return dict(self.weights)

    strat = Fixed()
    s = {**config.default_settings(), "rebalance_band": 0.25}
    p = WeightPortfolio(strat, 1000.0, s)
    p.step("2026-01-01", {"X": bar(100, 100, 100, 100)})
    p.step("2026-01-02", {"X": bar(100, 100, 100, 100)})
    assert p.holdings["X"]["qty"] * 100 == pytest.approx(500 / 1.0005 * 1.0005, rel=1e-3)
    strat.weights = {"X": 0.55}  # within the 25% band: no trade
    p.step("2026-01-03", {"X": bar(100, 100, 100, 100)})
    n = len(p.trades)
    p.step("2026-01-04", {"X": bar(100, 100, 100, 100)})
    assert len(p.trades) == n
    strat.weights = {}
    p.step("2026-01-05", {"X": bar(100, 100, 100, 100)})
    p.step("2026-01-06", {"X": bar(100, 100, 100, 100)})
    assert p.holdings == {}
    assert p.cash == pytest.approx(p.equity())
    assert p.cash < 1000  # paid fees and slippage


def test_weight_portfolio_state_round_trip(tmp_path):
    strat = DonchianEnsemble()
    p = WeightPortfolio(strat, 100.0, {**config.default_settings(), **strat.overrides})
    p.holdings = {"X": {"qty": 1.0, "cost": 50.0, "last_price": 60.0}}
    p.save(tmp_path / "w.json")
    q = load_account(tmp_path / "w.json", strat)
    assert isinstance(q, WeightPortfolio) and q.to_dict() == p.to_dict()
