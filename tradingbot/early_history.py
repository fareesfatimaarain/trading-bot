"""2013-2018 daily prices for 2,071 coins (CoinMarketCap via Kaggle 'Every Cryptocurrency Daily Market Price').

Used only to stress-test rules on years before Binance existed. Shorting most altcoins was not possible
then, so results show how an effect behaved, not trades anyone could have made.
"""
import numpy as np
import pandas as pd

from . import config

PATH = config.DATA_DIR / "early" / "crypto-markets.csv"
PEGGED_SYMBOLS = {"USDT", "TUSD", "USDC", "DAI", "PAX", "GUSD", "BITUSD", "USNBT", "BITCNY", "BITEUR", "BITGOLD",
                  "BITSILVER", "EURS", "DGX", "NUSD", "SUSD", "USDS", "CNHT", "EBASE", "XAUR", "GOLD", "WBTC"}


def load() -> dict[str, pd.DataFrame]:
    raw = pd.read_csv(PATH, parse_dates=["date"])
    raw = raw[(raw["close"] > 0) & ~raw["symbol"].isin(PEGGED_SYMBOLS)]
    out = {}
    for slug, g in raw.groupby("slug"):
        df = g.set_index("date").sort_index()[["open", "high", "low", "close", "volume"]].astype(float)
        df = df[~df.index.duplicated()]
        r = np.log(df["close"]).diff()
        if len(df) > 60 and r.rolling(60).std().median() < 0.01:  # anything else pegged to a fixed price
            continue
        df["quote_volume"] = df["volume"]  # CoinMarketCap volume is already in USD
        key = config.BENCHMARK if slug == "bitcoin" else slug
        out[key] = df
    return out
