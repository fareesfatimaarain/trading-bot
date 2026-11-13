import csv
import math
from datetime import datetime

import pandas as pd

from . import config, data
from .engine import Panel, new_account, run_days
from .strategies import ALL, Strategy


def strategy_settings(strategy: Strategy, kill_switch: bool = False, **extra) -> dict:
    return {**config.default_settings(kill_switch), **strategy.overrides, **extra}


def account_settings(name: str, kill_switch: bool = False, **extra) -> dict:
    acct = config.ACCOUNTS[name]
    s = strategy_settings(ALL[acct["strategy"]], kill_switch, market=acct["market"], **{**acct["settings"], **extra})
    filters = data.exchange_filters(acct["market"])
    s["filters"] = {sym: filters[sym] for sym in data.pool() if sym in filters}
    return s


def run(strategy: Strategy, universe: dict[str, pd.DataFrame], start: str, settings: dict,
        capital: float = config.START_CAPITAL, panel: Panel | None = None):
    account = new_account(strategy, capital, settings)
    run_days(account, panel or Panel(strategy.prepare_universe(universe)),
             after=(pd.Timestamp(start) - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    return account


def buy_and_hold(df: pd.DataFrame, start: str, settings: dict, capital: float = config.START_CAPITAL) -> pd.Series:
    df = df[df.index >= start]
    cost = 1 + settings["fee"] + settings["slippage"]
    qty = capital / (df["open"].iloc[0] * cost)
    return qty * df["close"]


def metrics(equity: pd.Series, trades: list[dict] | None, start: str, end: str | None = None) -> dict:
    full = equity
    seg = full[(full.index >= start) & ((full.index < end) if end else True)]
    prior = full[full.index < start]
    base = prior.iloc[-1] if len(prior) else seg.iloc[0]
    rets = pd.concat([pd.Series([base]), seg.reset_index(drop=True)]).pct_change().dropna()
    years = max(len(seg) / 365.0, 1e-9)
    total = seg.iloc[-1] / base
    out = {
        "period": f"{seg.index[0].date()}..{seg.index[-1].date()}",
        "cagr": total ** (1 / years) - 1,
        "vol": rets.std() * math.sqrt(365),
        "sharpe": rets.mean() / rets.std() * math.sqrt(365) if rets.std() > 0 else float("nan"),
        "max_dd": (seg / seg.cummax().clip(lower=base) - 1).min(),
    }
    if trades is not None:
        ts = [t for t in trades if t["exit_date"] >= start and (end is None or t["exit_date"] < end)]
        rs = pd.Series([t["r"] for t in ts], dtype=float)
        wins = sum(t["pnl"] for t in ts if t["pnl"] > 0)
        losses = -sum(t["pnl"] for t in ts if t["pnl"] < 0)
        days = [(pd.Timestamp(t["exit_date"]) - pd.Timestamp(t["entry_date"])).days for t in ts]
        out.update({
            "trades": len(ts),
            "avg_days": sum(days) / len(days) if days else float("nan"),
            "win_rate": (rs > 0).mean() if len(rs) else float("nan"),
            "avg_r": rs.mean() if len(rs) else float("nan"),
            "t_stat_r": rs.mean() / rs.std() * math.sqrt(len(rs)) if len(rs) > 1 and rs.std() > 0 else float("nan"),
            "profit_factor": wins / losses if losses > 0 else float("nan"),
        })
    return out


def _fmt(name: str, m: dict) -> str:
    line = (f"{name:<34} {m['period']:<24} CAGR {m['cagr']:>7.1%}  vol {m['vol']:>6.1%}  "
            f"Sharpe {m['sharpe']:>5.2f}  maxDD {m['max_dd']:>7.1%}")
    if "trades" in m:
        line += (f"  trades {m['trades']:>4}  win {m['win_rate']:>5.1%}  days {m['avg_days']:>4.0f}  "
                 f"avgR {m['avg_r']:>5.2f}  "
                 f"t {m['t_stat_r']:>5.2f}  PF {m['profit_factor']:>4.2f}")
    return line


def _log_trial(strategy: str, settings: dict, label: str, m: dict) -> None:
    path = config.RESULTS_DIR / "trials.csv"
    slim = {k: v for k, v in settings.items() if k != "filters"}
    row = {"run_at": datetime.now().isoformat(timespec="seconds"), "strategy": strategy,
           "segment": label, **m, "settings": slim}
    new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def equity_series(account) -> pd.Series:
    return pd.Series({pd.Timestamp(d): v for d, v in account.equity_curve})


def report(name: str, account, settings: dict) -> pd.Series:
    eq = equity_series(account)
    trades = account.trades if account.trades and "r" in account.trades[0] else None
    start, oos = config.BACKTEST_START, config.OOS_START
    for label, s, e in (("full", start, None), ("in-sample", start, oos), ("out-of-sample", oos, None)):
        if eq[eq.index >= s].empty:
            continue
        m = metrics(eq, trades, s, e)
        print(_fmt(f"{name} [{label}]", m))
        _log_trial(name, settings, label, m)
    notes = []
    if account.skipped_min_notional:
        notes.append(f"{account.skipped_min_notional} orders skipped below the exchange minimum")
    if not trades:
        notes.append(f"{len(account.trades)} rebalancing orders")
    else:
        short = [t for t in account.trades if t.get("side") == -1]
        if short:
            notes.append(f"shorts: {len(short)} trades, ${sum(t['pnl'] for t in short):+.0f}")
        funding = sum(t.get("funding", 0.0) for t in account.trades)
        if funding:
            notes.append(f"funding paid ${funding:+.0f}")
    if notes:
        print("  (" + "; ".join(notes) + ")")
    return eq


def main() -> None:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    start = config.BACKTEST_START
    spot = data.load_universe(refresh=False, market="spot")
    futures = data.load_universe(refresh=False, market="futures")
    base = config.default_settings()
    print(f"Pool: {len(spot)} spot / {len(futures)} futures coins; fees {base['fee']:.2%} spot, 0.05% futures; "
          f"slippage {base['slippage']:.2%} per side\n")

    bench = buy_and_hold(spot[config.BENCHMARK], start, base)
    for label, s, e in (("full", start, None), ("in-sample", start, config.OOS_START),
                        ("out-of-sample", config.OOS_START, None)):
        print(_fmt(f"BTC buy&hold [{label}]", metrics(bench, None, s, e)))
    print()

    turtle = ALL["turtle"]
    old20 = {s: spot[s] for s in config.UNIVERSE if s in spot}
    acct = run(turtle, old20, start, strategy_settings(turtle))
    report("turtle old fixed 20 ($72)", acct, strategy_settings(turtle))
    print()

    curves = {}
    for name, acct_cfg in config.ACCOUNTS.items():
        strategy = ALL[acct_cfg["strategy"]]
        universe = futures if acct_cfg["market"] == "futures" else spot
        panel = Panel(strategy.prepare_universe(universe))
        for capital in (config.START_CAPITAL, config.PAPER_CAPITAL, 10_000.0):
            settings = account_settings(name)
            acct = run(strategy, universe, start, settings, capital, panel)
            eq = report(f"{name} (${capital:,.0f})", acct, settings)
            if capital > 1000:
                curves[name] = eq
                pd.DataFrame(acct.trades).to_csv(config.RESULTS_DIR / f"{name}_trades.csv", index=False)
                eq.to_csv(config.RESULTS_DIR / f"{name}_equity.csv", header=["equity"], index_label="date")
        print()

    rets = pd.DataFrame(curves).pct_change()
    print("Daily-return correlation ($10k runs):")
    print(rets.corr().round(2).to_string())
    print("\nOut-of-sample correlation:")
    print(rets[rets.index >= config.OOS_START].corr().round(2).to_string())


if __name__ == "__main__":
    main()
