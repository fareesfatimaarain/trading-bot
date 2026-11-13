from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
PAPER_DIR = ROOT / "paper"

UNIVERSE = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
    "SOLUSDT", "DOGEUSDT", "TRXUSDT", "LTCUSDT", "LINKUSDT",
    "DOTUSDT", "AVAXUSDT", "BCHUSDT", "XLMUSDT", "ATOMUSDT",
    "ETCUSDT", "FILUSDT", "NEARUSDT", "UNIUSDT", "AAVEUSDT",
]
BENCHMARK = "BTCUSDT"

# Paper/backtest accounts. Universe = the N most liquid coins each day (point in time) from data/pool.json.
ACCOUNTS = {
    # partial_take_r=None: half-at-+2R was tried and dropped on 2026-10-07 because it lowered total return.
    "turtle_spot": {"strategy": "turtle", "market": "spot",
                    "settings": {"universe_top_n": 50, "partial_take_r": None}},
    "turtle_futures": {"strategy": "turtle", "market": "futures",
                       "settings": {"universe_top_n": 50, "allow_short": True, "fee": 0.0005,
                                    "max_gross_leverage": 2.0, "partial_take_r": None}},
    "donchian_spot": {"strategy": "donchian", "market": "spot", "settings": {}},
}

HISTORY_START = "2018-01-01"
BACKTEST_START = "2019-01-01"
OOS_START = "2024-01-01"

USD_PKR = 277.0
START_CAPITAL = round(20_000 / USD_PKR, 2)
PAPER_CAPITAL = 1000.0  # user's choice: the Donchian ensemble can't clear $5 minimums at $72


def default_settings(kill_switch: bool = False) -> dict:
    return {
        "market": "spot",
        "fee": 0.001,  # Binance spot taker without BNB discount; futures accounts override to 0.05%
        "slippage": 0.0005,
        "risk_per_trade": 0.01,
        "max_open_risk": 0.10,
        "max_positions": 8,
        "max_position_frac": 0.25,
        "min_notional": 5.0,  # Binance spot minimum order value in USDT
        "kill_switch_drawdown": 0.35 if kill_switch else None,
    }
