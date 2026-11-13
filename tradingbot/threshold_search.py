"""Search over price-move triggers with fixed take-profit / stop-loss exits.

Entry: a coin's return over `window` days crosses +/-`move`. Momentum = long after a rise, short after
a drop; reversal = the opposite. Exit: take profit, stop loss (checked on daily high/low, stop first if
both are hit the same day, gaps fill at the open) or after MAX_HOLD days. Up to 10 positions of 10% each.
Ranked by the worse CAGR of era A (2015-18) and B (2019-22); era C (2023-26) is held out.
"""
import itertools
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from . import config, data, early_history
from .strategies import add_liquidity_rank

# CoinMarketCap highs/lows (era A) aggregate every exchange and carry 2-3x more >10% wicks than Binance,
# which distorts stop/target fills, so A is shown for reference and ranking uses the two Binance halves.
ERAS = {"A": ("2015-01-01", "2019-01-01"), "B1": ("2019-01-01", "2021-01-01"), "B2": ("2021-01-01", "2023-01-01"),
        "C": ("2023-01-01", "2100-01-01")}
TOP_N = 50
SLOTS = 10
MAX_HOLD = 30
FEE = 0.001
SLIPPAGE = {"A": 0.0025, "B1": 0.0005, "B2": 0.0005, "C": 0.0005}
TARGET_FIRST = False  # when a day touches both stop and target: False = stop first (conservative)


def variants() -> list[dict]:
    return [{"move": m, "window": w, "mode": mode, "tp": tp, "sl": sl}
            for m, w, mode, tp, sl in itertools.product((0.05, 0.10, 0.20), (1, 7), ("momentum", "reversal"),
                                                        (0.10, 0.15, 0.20), (0.05, 0.10, 0.20))]


def label(p: dict) -> str:
    return (f"{p['mode']:<8} move {p['move']:.0%}/{p['window']}d  TP {p['tp']:.0%} SL {p['sl']:.0%}")


def _arrays(universe: dict[str, pd.DataFrame]) -> dict:
    frames = {s: df.copy() for s, df in universe.items()}
    add_liquidity_rank(frames)
    idx = pd.DatetimeIndex(sorted(set().union(*(df.index for df in frames.values()))))
    syms = list(frames)
    out = {"dates": idx}
    for col in ("open", "high", "low", "close", "liq_rank"):
        out[col] = pd.DataFrame({s: frames[s][col] for s in syms}).reindex(idx)[syms].to_numpy(float)
    return out


_DATA: dict = {}


def _init() -> None:
    warnings.filterwarnings("ignore")
    early = early_history.load()
    ranked = {s: df.copy() for s, df in early.items()}
    add_liquidity_rank(ranked)
    keep = {s for s, df in ranked.items() if df["liq_rank"].min() <= 100} | {config.BENCHMARK}
    _DATA["A"] = _arrays({s: early[s] for s in keep})
    _DATA["BC"] = _arrays(data.load_universe(refresh=False, market="spot"))


def simulate(a: dict, start: str, end: str, p: dict, slip: float) -> pd.Series:
    O, H, L, C, R = a["open"], a["high"], a["low"], a["close"], a["liq_rank"]
    dates = a["dates"]
    with np.errstate(all="ignore"):
        move = C / np.roll(C, p["window"], axis=0) - 1
    move[: p["window"]] = np.nan
    cost = FEE + slip
    cash, positions, pending, curve = 1.0, [], [], {}
    t0, t1 = dates.searchsorted(pd.Timestamp(start)), dates.searchsorted(pd.Timestamp(end))
    for t in range(t0, min(t1, len(dates))):
        equity_prev = cash + sum(pos["alloc"] * (1 + pos["side"] * (pos["last"] / pos["entry"] - 1))
                                 for pos in positions)
        for j, side in pending:
            if len(positions) >= SLOTS or np.isnan(O[t, j]) or any(pos["j"] == j for pos in positions):
                continue
            alloc = min(equity_prev / SLOTS, cash)
            if alloc <= 0:
                break
            cash -= alloc
            fee = alloc * cost
            positions.append({"j": j, "side": side, "entry": O[t, j], "alloc": alloc - fee, "day": t,
                              "last": O[t, j]})
        pending = []
        still = []
        for pos in positions:
            j, side, entry = pos["j"], pos["side"], pos["entry"]
            o, h, lo, c = O[t, j], H[t, j], L[t, j], C[t, j]
            if np.isnan(c):
                still.append(pos)
                continue
            stop = entry * (1 - side * p["sl"])
            take = entry * (1 + side * p["tp"])
            exit_px = None
            hit_stop = (lo <= stop) if side == 1 else (h >= stop)
            hit_take = (h >= take) if side == 1 else (lo <= take)
            gap_stop = (o <= stop) if side == 1 else (o >= stop)
            gap_take = (o >= take) if side == 1 else (o <= take)
            if gap_stop or gap_take:
                exit_px = o
            elif hit_stop and hit_take:
                exit_px = take if TARGET_FIRST else stop
            elif hit_stop:
                exit_px = stop
            elif hit_take:
                exit_px = take
            if exit_px is None and t - pos["day"] >= MAX_HOLD:
                exit_px = c
            if exit_px is None:
                pos["last"] = c
                still.append(pos)
                continue
            value = pos["alloc"] * (1 + side * (exit_px / entry - 1))
            cash += value * (1 - cost)
        positions = still
        equity = cash + sum(pos["alloc"] * (1 + pos["side"] * (pos["last"] / pos["entry"] - 1)) for pos in positions)
        curve[dates[t]] = equity
        free = SLOTS - len(positions)
        if free <= 0:
            continue
        eligible = np.nan_to_num(R[t], nan=np.inf) <= TOP_N
        mv = np.where(eligible, move[t], np.nan)
        held = {pos["j"] for pos in positions}
        rise, drop = np.nan_to_num(mv, nan=0) >= p["move"], np.nan_to_num(mv, nan=0) <= -p["move"]
        long_on, short_on = (rise, drop) if p["mode"] == "momentum" else (drop, rise)
        cands = [(abs(mv[j]), j, 1) for j in np.flatnonzero(long_on) if j not in held]
        cands += [(abs(mv[j]), j, -1) for j in np.flatnonzero(short_on) if j not in held]
        pending = [(j, side) for _, j, side in sorted(cands, reverse=True)[:free]]
    return pd.Series(curve)


def stats(curve: pd.Series) -> dict:
    if len(curve) < 30:
        return {"cagr": np.nan, "sharpe": np.nan, "dd": np.nan}
    r = curve.pct_change().dropna()
    years = len(curve) / 365
    return {"cagr": curve.iloc[-1] ** (1 / years) - 1 if curve.iloc[-1] > 0 else -1.0,
            "sharpe": r.mean() / r.std() * np.sqrt(365) if r.std() > 0 else np.nan,
            "dd": (curve / curve.cummax() - 1).min()}


def run_variant(p: dict) -> dict:
    row = {"variant": label(p), **p}
    for era, (start, end) in ERAS.items():
        arrays = _DATA["A"] if era == "A" else _DATA["BC"]
        s = stats(simulate(arrays, start, end, p, SLIPPAGE[era]))
        row.update({f"{era}_{k}": v for k, v in s.items()})
    return row


def main() -> None:
    todo = variants()
    print(f"testing {len(todo)} trigger/exit variants on 3 eras...", flush=True)
    with ProcessPoolExecutor(max_workers=4, initializer=_init) as pool:
        rows = list(pool.map(run_variant, todo))
    df = pd.DataFrame(rows)
    df["worst_B"] = df[["B1_cagr", "B2_cagr"]].min(axis=1)
    df = df.sort_values("worst_B", ascending=False).reset_index(drop=True)
    df.to_csv(config.RESULTS_DIR / "threshold_search.csv", index=False)

    cols = ["variant", "A_cagr", "B1_cagr", "B1_dd", "B2_cagr", "B2_dd", "worst_B", "C_cagr", "C_sharpe", "C_dd"]
    view = df[cols].copy()
    for c in cols[1:]:
        view[c] = view[c].map(lambda v: f"{v:.2f}" if "sharpe" in c else f"{v:.1%}")
    pd.set_option("display.width", 250)
    print("\nTop 15 by worse of B1 (2019-20) and B2 (2021-22); A (2015-18) for reference; C (2023-26) held out:")
    print(view.head(15).to_string())
    print("\nWorst 5:")
    print(view.tail(5).to_string())
    print(f"\nPositive in both B1 and B2: {(df['worst_B'] > 0).sum()} of {len(df)};  positive in C: "
          f"{(df['C_cagr'] > 0).sum()} of {len(df)};  positive in A: {(df['A_cagr'] > 0).sum()}")
    print(f"Rank correlation, selection (worse of B1,B2) vs held-out C: {df['worst_B'].rank().corr(df['C_cagr'].rank()):.2f}")
    print(f"Chosen: {df.iloc[0]['variant']} -> held-out C CAGR {df.iloc[0]['C_cagr']:.1%}, "
          f"maxDD {df.iloc[0]['C_dd']:.1%};  top-10 average C {df.head(10)['C_cagr'].mean():.1%}, "
          f"all variants {df['C_cagr'].mean():.1%}")
    for col in ("mode", "move", "window", "tp", "sl"):
        g = df.groupby(col)[["A_cagr", "B1_cagr", "B2_cagr", "C_cagr"]].mean().map(lambda v: f"{v:.1%}")
        print(f"\nAverage by {col}:\n{g.to_string()}")


if __name__ == "__main__":
    main()
