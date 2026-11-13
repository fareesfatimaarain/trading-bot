"""Walk-forward deep-learning test: a 1D convolutional network reads raw 64-day charts.

Same protocol, trading rule and costs as ml_research. For each test year the network sees only
earlier data; its settings are chosen on the last training year (validation), never the test year.
"""
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from . import config, data
from .ml_research import HOLD_DAYS, TEST_YEARS, TOP_N_UNIVERSE, rank_ic, summarize, trade
from .strategies import add_liquidity_rank

WINDOW = 64
CONFIGS = [{"channels": 32, "dropout": 0.3}, {"channels": 64, "dropout": 0.5}]
EXTRA_SEEDS = (1, 2)
MAX_EPOCHS = 20
PATIENCE = 3
BATCH = 512


def build_samples(universe: dict[str, pd.DataFrame]) -> tuple[np.ndarray, pd.DataFrame]:
    prepared = {s: df.copy() for s, df in universe.items()}
    add_liquidity_rank(prepared)
    btc_close = universe[config.BENCHMARK]["close"]
    xs, metas = [], []
    for sym, df in prepared.items():
        c = df["close"]
        channels = np.stack([
            np.log(c).diff() * 10,
            (df["high"] / c - 1) * 10,
            (df["low"] / c - 1) * 10,
            np.log(df["quote_volume"] / df["quote_volume"].rolling(30).median()),
            np.log(btc_close.reindex(df.index)).diff() * 10,
        ]).astype(np.float32)
        channels = np.clip(np.nan_to_num(channels), -5, 5)
        fwd = (df["open"].shift(-(HOLD_DAYS + 1)) / df["open"].shift(-1) - 1).to_numpy()
        # funding paid by a long over the holding days (days t+1..t+HOLD_DAYS); 0 for spot
        fund = (df["funding"].rolling(HOLD_DAYS).sum().shift(-HOLD_DAYS).to_numpy()
                if "funding" in df else np.zeros(len(df)))
        rank = df["liq_rank"].to_numpy()
        idx = [i for i in range(WINDOW + 30, len(df)) if rank[i] <= TOP_N_UNIVERSE]
        if not idx:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(channels, WINDOW, axis=1)  # (5, T-W+1, W)
        xs.append(windows[:, [i - WINDOW + 1 for i in idx], :].transpose(1, 0, 2).copy())
        metas.append(pd.DataFrame({"date": df.index[idx], "symbol": sym, "fwd": fwd[idx], "fund": fund[idx]}))
    return np.concatenate(xs), pd.concat(metas, ignore_index=True)


class ChartCNN(nn.Module):
    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(5, channels, 5, padding=2), nn.ReLU(),
            nn.Conv1d(channels, channels, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(channels, channels * 2, 5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(channels * 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _target(rows: pd.DataFrame, relative: bool) -> np.ndarray:
    """Long-only learns the raw 7-day return; long-short learns return after funding minus that day's average."""
    y = rows["fwd"]
    if relative:
        y = (rows["fwd"] - rows["fund"]).groupby(rows["date"]).transform(lambda s: s - s.mean())
    return (np.clip(y.to_numpy(), -0.5, 1.0) * 10).astype(np.float32)


LS_COST_PER_SIDE = 0.001  # futures taker 0.05% + 0.05% slippage


def trade_long_short(rows: pd.DataFrame, k: int = 5) -> pd.Series:
    """Weekly: long the top k, short the bottom k, half the capital per side; funding and costs included."""
    dates = sorted(rows["date"].unique())[::HOLD_DAYS]
    prev_w: dict[str, float] = {}
    out = {}
    for d in dates:
        day = rows[rows["date"] == d].dropna(subset=["fwd"])
        w: dict[str, float] = {}
        if len(day) >= 2 * k:
            w.update({s: 0.5 / k for s in day.nlargest(k, "pred")["symbol"]})
            w.update({s: -0.5 / k for s in day.nsmallest(k, "pred")["symbol"]})
        net = day.set_index("symbol")
        gross = sum(wt * (net.at[s, "fwd"] - net.at[s, "fund"]) for s, wt in w.items())
        turnover = sum(abs(w.get(s, 0) - prev_w.get(s, 0)) for s in set(w) | set(prev_w))
        out[d] = gross - turnover * LS_COST_PER_SIDE
        prev_w = w
    return pd.Series(out)


def predict(model: nn.Module, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(x[i:i + 4096])).numpy() for i in range(0, len(x), 4096)])


def fit(x_fit, y_fit, x_val, y_val, cfg: dict, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ChartCNN(**cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = nn.SmoothL1Loss()
    best, best_state, stale = np.inf, None, 0
    xv, yv = torch.from_numpy(x_val), torch.from_numpy(y_val)
    for _ in range(MAX_EPOCHS):
        model.train()
        order = rng.permutation(len(x_fit))
        for i in range(0, len(order), BATCH):
            b = order[i:i + BATCH]
            opt.zero_grad()
            loss_fn(model(torch.from_numpy(x_fit[b])), torch.from_numpy(y_fit[b])).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(xv), yv).item()
        if val < best - 1e-5:
            best, best_state, stale = val, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model


def main() -> None:
    t0 = time.time()
    long_short = "longshort" in sys.argv[1:]
    universe = data.load_universe(refresh=False, market="futures" if long_short else "spot")
    x, meta = build_samples(universe)
    print(f"{'LONG-SHORT on futures' if long_short else 'LONG-ONLY on spot'}: {len(meta):,} chart windows "
          f"({WINDOW} days x 5 channels), {meta['symbol'].nunique()} coins", flush=True)
    trade_fn = trade_long_short if long_short else trade

    btc = meta[meta["symbol"] == config.BENCHMARK].set_index("date")["fwd"]
    print(f"{'year':<6}{'model OOS':>11}{'model in-sample':>17}{'BTC hold':>10}{'IC OOS':>8}{'IC in-sample':>14}"
          f"  chosen settings", flush=True)
    oos_all, ins_all, btc_all = [], [], []
    for year in TEST_YEARS:
        start = pd.Timestamp(f"{year}-01-01")
        gap = pd.Timedelta(days=HOLD_DAYS + 1)
        cutoff = start - gap
        val_start = cutoff - pd.Timedelta(days=365)
        has_fwd = meta["fwd"].notna()
        fit_m = (meta["date"] < val_start - gap) & has_fwd
        val_m = (meta["date"] >= val_start) & (meta["date"] < cutoff) & has_fwd
        test_m = (meta["date"] >= start) & (meta["date"] < pd.Timestamp(f"{year + 1}-01-01"))
        if not test_m.any():
            continue
        ins_m = (meta["date"] >= val_start - pd.Timedelta(days=365)) & (meta["date"] < val_start - gap)
        if fit_m.sum() < 5000 or not ins_m.any():
            print(f"{year:<6}skipped: only {fit_m.sum():,} training windows before it", flush=True)
            continue
        x_fit, y_fit = x[fit_m.to_numpy()], _target(meta[fit_m], long_short)
        x_val, y_val = x[val_m.to_numpy()], _target(meta[val_m], long_short)

        scored = []
        for cfg in CONFIGS:
            model = fit(x_fit, y_fit, x_val, y_val, cfg, seed=0)
            val_rows = meta[val_m].copy()
            val_rows["pred"] = predict(model, x_val)
            scored.append((rank_ic(val_rows), cfg, model))
        _, cfg, first = max(scored, key=lambda t: t[0])
        ensemble = [first] + [fit(x_fit, y_fit, x_val, y_val, cfg, seed=s) for s in EXTRA_SEEDS]

        def ens_predict(arr):
            return np.mean([predict(m, arr) for m in ensemble], axis=0)

        test = meta[test_m].copy()
        test["pred"] = ens_predict(x[test_m.to_numpy()])
        ins = meta[ins_m].copy()
        ins["pred"] = ens_predict(x[ins_m.to_numpy()])

        oos_r, ins_r = trade_fn(test), trade_fn(ins)
        b = btc.reindex(oos_r.index).fillna(0)
        oos_all.append(oos_r), ins_all.append(ins_r), btc_all.append(b)
        print(f"{year:<6}{(1 + oos_r).prod() - 1:>11.1%}{(1 + ins_r).prod() - 1:>17.1%}{(1 + b).prod() - 1:>10.1%}"
              f"{rank_ic(test):>8.3f}{rank_ic(ins):>14.3f}  {cfg}  ({time.time() - t0:.0f}s)", flush=True)

    print()
    for name, parts in (("deep model, unseen years", oos_all), ("deep model, years it trained on", ins_all),
                        ("BTC buy and hold", btc_all)):
        m = summarize(pd.concat(parts))
        print(f"{name:<32} CAGR {m['cagr']:>7.1%}  Sharpe {m['sharpe']:>5.2f}  maxDD {m['max_dd']:>7.1%}")


if __name__ == "__main__":
    main()
