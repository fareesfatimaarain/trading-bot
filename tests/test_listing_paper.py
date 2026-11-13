import pandas as pd
import pytest

from tradingbot.listing_paper import ENTRY_COST, EXIT_COST, STOP_SLIP, _position_return

T0 = pd.Timestamp("2026-01-02")


def _path(rows):
    idx = pd.date_range(T0, periods=len(rows), freq="h")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx, dtype=float)


def _pos(hours=5):
    return {"entry_time": str(T0), "exit_time": str(T0 + pd.Timedelta(hours=hours)), "entry": 100.0}


NO_FUNDING = pd.Series(dtype=float)


def test_closes_at_hold_end_with_profit():
    path = _path([(100, 101, 95, 96), (96, 97, 90, 92), (92, 93, 88, 90), (90, 91, 85, 88), (88, 89, 80, 85)])
    ret, closed, px, reason = _position_return(_pos(5), path, NO_FUNDING, T0 + pd.Timedelta(hours=6), stop=None)
    assert closed and px == 85 and reason == "hold period over"
    assert ret == pytest.approx(0.15 - ENTRY_COST - EXIT_COST)


def test_open_position_marked_without_exit_cost():
    path = _path([(100, 101, 95, 96), (96, 97, 90, 92)])
    ret, closed, px, _ = _position_return(_pos(5), path, NO_FUNDING, T0 + pd.Timedelta(hours=2), stop=None)
    assert not closed and px == 92 and ret == pytest.approx(0.08 - ENTRY_COST)


def test_stop_fills_at_stop_level_plus_slippage():
    path = _path([(100, 105, 99, 104), (104, 140, 103, 120), (120, 121, 60, 61)])
    ret, closed, px, reason = _position_return(_pos(5), path, NO_FUNDING, T0 + pd.Timedelta(hours=3), stop=0.30)
    assert closed and reason == "stop" and px == 130
    assert ret == pytest.approx(-0.30 - ENTRY_COST - EXIT_COST - STOP_SLIP)


def test_funding_collected_by_short_and_loss_capped_at_total():
    path = _path([(100, 100, 100, 100), (100, 300, 100, 300)])
    funding = pd.Series({T0 + pd.Timedelta(minutes=30): 0.01})
    ret, closed, _, _ = _position_return(_pos(2), path, funding, T0 + pd.Timedelta(hours=3), stop=None)
    assert closed and ret == -1.0
    path = _path([(100, 100, 100, 100), (100, 100, 100, 100)])
    ret, _, _, _ = _position_return(_pos(2), path, funding, T0 + pd.Timedelta(hours=3), stop=None)
    assert ret == pytest.approx(0.01 - ENTRY_COST - EXIT_COST)
