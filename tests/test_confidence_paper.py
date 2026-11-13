import pandas as pd

from tradingbot.confidence_paper import check_exit

H = pd.Timestamp("2026-01-01 10:00")


def _pos(side):
    return {"side": side, "target": 110.0 if side == 1 else 90.0, "stop": 95.0 if side == 1 else 105.0,
            "deadline": str(H + pd.Timedelta(hours=48))}


def _bar(o, h, l, c):
    return pd.Series({"open": o, "high": h, "low": l, "close": c})


def test_long_hits_target():
    assert check_exit(_pos(1), _bar(100, 111, 99, 108), H) == (110.0, "target")


def test_stop_wins_when_both_hit_in_one_hour():
    assert check_exit(_pos(1), _bar(100, 112, 94, 100), H) == (95.0, "stop")
    assert check_exit(_pos(-1), _bar(100, 106, 89, 100), H) == (105.0, "stop")


def test_short_hits_target():
    assert check_exit(_pos(-1), _bar(100, 101, 89, 92), H) == (90.0, "target")


def test_closes_at_48h_limit_otherwise_stays_open():
    assert check_exit(_pos(1), _bar(100, 101, 99, 100), H) is None
    last = H + pd.Timedelta(hours=47)
    assert check_exit(_pos(1), _bar(100, 101, 99, 100.5), last) == (100.5, "48h limit")
