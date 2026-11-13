# Crypto Trading Bot & Strategy Research

A Python backtesting engine, research suite and paper-trading bot for Binance spot and USDT-M futures. It tests trend-following, momentum, listing-event and machine-learning strategies across 900+ coins. Every result is walk-forward or held-out, so a rule is always judged on data it never saw.

> Research and paper trading only. Nothing here is financial advice, and no strategy in this repo is presented as profitable live.

## Highlights

- **Realistic backtesting engine** (`engine.py`): daily bars, fees and slippage, Binance minimum order sizes and quantity steps, futures funding, position and risk limits, and a drawdown kill switch. The coin universe is chosen point in time from liquidity ranks, so there is no survivorship bias.
- **Strategies** (`strategies.py`): Turtle trend-following, momentum breakout, dip buying and a volatility-targeted Donchian ensemble, for spot and futures (long/short, up to 2x gross leverage).
- **Paper trading** (`paper.py`): runs every hour against live Binance data. It keeps separate accounts per strategy, saves state crash-safely and retries if the connection drops.
- **Data pipeline** (`data.py`): downloads daily bars, futures funding history and exchange order rules for every listed USDT pair, including coins that were later delisted.
- **40 unit tests** covering the engine, futures, sizing and both paper traders.

## Research — what held up and what didn't

Each study lives in its own module, and its output is committed under `results/`.

| Study | Method | Finding |
|---|---|---|
| Rule search (`search.py`) | 81 entry/exit variants, ranked on 2015–18 and 2019–22, judged on a held-out 2023–26 | All 81 were profitable in both ranking eras. The variant picked by that ranking made **-0.5%/yr** held out, with a 52% drawdown. In-sample ranking did not carry forward. |
| 1D-CNN on raw charts (`dl_research.py`) | Walk-forward; settings tuned on a validation year, never the test year | **+229%/yr on years it trained on and -26%/yr on unseen years**, a textbook case of overfitting. BTC buy-and-hold made +19.5%/yr. |
| Next-hour direction (`intraday_research.py`) | Gradient-boosted trees vs a logistic baseline, with order-flow features | About 51–53% accuracy. Too small an edge to beat trading costs with market orders. |
| Confidence-scored entries (`confidence_research.py`) | Triple-barrier labels, trading only the top X% of model confidence, with real funding costs | Roughly break-even before costs and negative after funding at most settings. |
| Shorting new listings (`listing_shorts.py`) | Binance perps for genuinely new coins, with stops against short squeezes | Strong in 2021–22 (about +20% per trade). The edge mostly faded from 2023, at roughly +1–6% per trade. |

Also included: a walk-forward gradient-boosted coin picker (`ml_research.py`), a low-volatility long/short with hard stops tested on 2014–18 and 2019–26 (`lowvol_research.py`), and a price-trigger threshold search (`threshold_search.py`).

The main takeaway is the testing method itself. Careful walk-forward evaluation rejected most ideas that looked excellent in-sample.

## Running it

Requires Python 3.12.

```bash
python -m venv .venv && .venv/Scripts/activate      # or source .venv/bin/activate
pip install -r requirements.txt

python -m tradingbot.data        # download market data into data/ (~900 MB, public Binance endpoints, no API key)
python -m pytest                 # run the test suite
python -m tradingbot.backtest    # backtest the configured accounts
python -m tradingbot.search      # any research module runs the same way
python -m tradingbot.paper       # one paper-trading pass (schedule it hourly)
```

Accounts, the coin universe and costs are set in `tradingbot/config.py`. The 2013–2018 stress tests also need the Kaggle dataset "Every Cryptocurrency Daily Market Price" saved at `data/early/crypto-markets.csv`.

## Layout

```
tradingbot/
  engine.py, backtest.py, strategies.py, report.py   backtesting core
  data.py, early_history.py                          data download and loading
  paper.py, listing_paper.py, confidence_paper.py    hourly paper traders
  *_research.py, search.py, threshold_search.py      research studies
tests/                                               pytest suite
results/                                             research outputs (logs, equity curves, trades)
```
