"""Systematic search over entry/exit rule variants, chosen on two eras and confirmed on a held-out one.

Era A: 2015-2018 CoinMarketCap data (incl. coins that died). Era B: 2019-2022 Binance spot.
Era C (held out): 2023-2026 Binance spot. Variants are ranked by their worse CAGR across A and B;
C is reported but never used for ranking. All variants are long-only spot, $10k, fees included.
"""
import itertools
import warnings
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from . import backtest, config, data, early_history
from .engine import Panel
from .strategies import DonchianEnsemble, Strategy, add_btc_regime, add_liquidity_rank, atr

ERA_A = ("2015-01-01", "2019-01-01")
ERA_B = ("2019-01-01", "2023-01-01")
ERA_C = ("2023-01-01", None)
CAPITAL = 10_000.0
ERA_A_SLIPPAGE = 0.0025


class Breakout(Strategy):
    def __init__(self, entry: int, exit_: int, stop_atr: float, regime: bool):
        self.entry, self.exit_, self.stop_atr, self.regime = entry, exit_, stop_atr, regime
        self.name = f"breakout {entry}/{exit_} stop{stop_atr:g}{' +btc200' if regime else ''}"

    def prepare(self, df):
        out = df.copy()
        out["atr"] = atr(out, 20)
        out["entry_level"] = out["high"].rolling(self.entry).max().shift(1)
        out["exit_level"] = out["low"].rolling(self.exit_).min().shift(1)
        out["score"] = out["close"].pct_change(90)
        return out

    def prepare_universe(self, data):
        prepared = super().prepare_universe(data)
        if self.regime:
            add_btc_regime(prepared, data)
        return prepared

    def entry_signal(self, b):
        return b["close"] > b["entry_level"] and (not self.regime or bool(b["regime"]))

    def exit_signal(self, b):
        return b["close"] < b["exit_level"]


class MACross(Strategy):
    def __init__(self, fast: int, slow: int, stop_atr: float, regime: bool):
        self.fast, self.slow, self.stop_atr, self.regime = fast, slow, stop_atr, regime
        self.name = f"ma cross {fast}/{slow} stop{stop_atr:g}{' +btc200' if regime else ''}"

    def prepare(self, df):
        out = df.copy()
        out["atr"] = atr(out, 20)
        out["ma_fast"] = out["close"].rolling(self.fast).mean()
        out["ma_slow"] = out["close"].rolling(self.slow).mean()
        out["score"] = out["close"].pct_change(90)
        return out

    def prepare_universe(self, data):
        prepared = super().prepare_universe(data)
        if self.regime:
            add_btc_regime(prepared, data)
        return prepared

    def entry_signal(self, b):
        return b["ma_fast"] > b["ma_slow"] and (not self.regime or bool(b["regime"]))

    def exit_signal(self, b):
        return b["ma_fast"] < b["ma_slow"]


def variants() -> list[tuple[str, dict]]:
    """(kind, params) for every variant; fixed before any results are seen."""
    out = []
    pairs = [(20, 10), (55, 10), (55, 20), (100, 10), (100, 20), (100, 50)]
    for (e, x), stop, regime, top in itertools.product(pairs, (2.0, 4.0), (False, True), (20, 50)):
        out.append(("breakout", {"entry": e, "exit_": x, "stop_atr": stop, "regime": regime, "top": top}))
    for (f, s), stop, regime, top in itertools.product([(10, 50), (20, 100), (50, 200)], (2.0, 4.0),
                                                       (False, True), (20, 50)):
        out.append(("ma", {"fast": f, "slow": s, "stop_atr": stop, "regime": regime, "top": top}))
    for tv, top in itertools.product((0.25, 0.5, 1.0), (10, 20, 50)):
        out.append(("donchian", {"target_vol": tv, "top": top}))
    return out


def build(kind: str, p: dict):
    if kind == "breakout":
        s = Breakout(p["entry"], p["exit_"], p["stop_atr"], p["regime"])
    elif kind == "ma":
        s = MACross(p["fast"], p["slow"], p["stop_atr"], p["regime"])
    else:
        s = DonchianEnsemble()
        s.name = f"donchian ensemble vol{p['target_vol']:g}"
        return s, backtest.strategy_settings(s, universe_top_n=p["top"], target_vol=p["target_vol"])
    # risk scales with stop width so position size is the same; the stop only changes when trades end
    settings = backtest.strategy_settings(s, universe_top_n=p["top"], risk_per_trade=0.005 * p["stop_atr"],
                                          max_open_risk=0.05 * p["stop_atr"])
    return s, settings


_UNIVERSES: dict = {}


def _init() -> None:
    warnings.filterwarnings("ignore")
    early = early_history.load()
    ranked = {s: df.copy() for s, df in early.items()}
    add_liquidity_rank(ranked)
    keep = [s for s, df in ranked.items() if df["liq_rank"].min() <= 100] + [config.BENCHMARK]
    _UNIVERSES["A"] = {s: early[s] for s in set(keep)}
    _UNIVERSES["BC"] = data.load_universe(refresh=False, market="spot")


def run_variant(args: tuple[str, dict]) -> dict:
    kind, p = args
    strategy, settings = build(kind, p)
    row = {"variant": f"{strategy.name} top{p['top']}", "kind": kind, **p}
    for era_key, segments, extra in (("A", [("A", ERA_A)], {"slippage": ERA_A_SLIPPAGE}),
                                     ("BC", [("B", ERA_B), ("C", ERA_C)], {})):
        universe = _UNIVERSES[era_key]
        s = {**settings, **extra}
        acct = backtest.run(strategy, universe, segments[0][1][0], s, CAPITAL,
                            Panel(strategy.prepare_universe(universe)))
        eq = backtest.equity_series(acct)
        trades = acct.trades if acct.trades and "r" in acct.trades[0] else None
        for label, (start, end) in segments:
            seg = eq[(eq.index >= start) & ((eq.index < end) if end else True)]
            if seg.empty:
                continue
            m = backtest.metrics(eq, trades, start, end)
            row.update({f"{label}_cagr": m["cagr"], f"{label}_sharpe": m["sharpe"], f"{label}_dd": m["max_dd"],
                        f"{label}_trades": m.get("trades")})
    return row


def main(limit: int | None = None) -> None:
    todo = variants()[:limit] if limit else variants()
    print(f"testing {len(todo)} variants on 3 eras...", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=4, initializer=_init) as pool:
        for i, row in enumerate(pool.map(run_variant, todo), 1):
            rows.append(row)
            if i % 10 == 0:
                print(f"  {i}/{len(todo)} done", flush=True)
    df = pd.DataFrame(rows)
    df["worst_AB_cagr"] = df[["A_cagr", "B_cagr"]].min(axis=1)
    df = df.sort_values("worst_AB_cagr", ascending=False).reset_index(drop=True)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(config.RESULTS_DIR / "search_results.csv", index=False)

    pd.set_option("display.width", 250)
    cols = ["variant", "A_cagr", "A_dd", "B_cagr", "B_dd", "worst_AB_cagr", "C_cagr", "C_sharpe", "C_dd"]
    fmt = df[cols].copy()
    for c in cols[1:]:
        fmt[c] = fmt[c].map(lambda v: f"{v:.1%}" if "sharpe" not in c else f"{v:.2f}")
    print("\nTop 15 by worse-era return (A = 2015-18, B = 2019-22; C = 2023-26 held out, not used to rank):")
    print(fmt.head(15).to_string())
    print("\nBottom 5:")
    print(fmt.tail(5).to_string())
    print(f"\nVariants positive in both A and B: {(df['worst_AB_cagr'] > 0).sum()} of {len(df)}")
    best = df.iloc[0]
    print(f"\nChosen (best worse-era): {best['variant']}  ->  held-out 2023-26: CAGR {best['C_cagr']:.1%}, "
          f"Sharpe {best['C_sharpe']:.2f}, maxDD {best['C_dd']:.1%}")
    top10 = df.head(10)
    print(f"Top 10 held-out average CAGR {top10['C_cagr'].mean():.1%} vs all variants {df['C_cagr'].mean():.1%}")


if __name__ == "__main__":
    import sys
    main(int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None)
