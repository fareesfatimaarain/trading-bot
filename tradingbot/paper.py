import csv
import logging
import time

import pandas as pd
import requests

from . import config, confidence_paper, data, listing_paper
from .backtest import account_settings
from .engine import Panel, load_account, new_account, run_days
from .strategies import ALL

log = logging.getLogger("paper")


def _write_csv(path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _describe(t: dict) -> str:
    if "r" in t:
        side = "SHORT" if t.get("side") == -1 else "LONG"
        return (f"closed {side} {t['symbol']} @ {t['exit_price']:.6g} ({t['reason']}) "
                f"pnl ${t['pnl']:+.2f} = {t['r']:+.2f}R")
    return f"{t['side'].upper()} {t['symbol']} ${t['notional']:.2f} @ {t['price']:.6g}"


def run_account(name: str, universes: dict) -> None:
    cfg = config.ACCOUNTS[name]
    strategy = ALL[cfg["strategy"]]
    if cfg["market"] not in universes:
        universes[cfg["market"]] = data.load_universe(refresh=True, market=cfg["market"])
    universe = universes[cfg["market"]]
    if config.BENCHMARK not in universe:
        log.error("[%s] no %s data; will retry next run", name, config.BENCHMARK)
        return
    last_complete = universe[config.BENCHMARK].index[-1].strftime("%Y-%m-%d")

    state_path = config.PAPER_DIR / f"{name}_state.json"
    fresh = account_settings(name, kill_switch=True)
    if state_path.exists():
        acct = load_account(state_path, strategy)
        acct.settings.pop("min_notional_by_symbol", None)
        acct.settings.update(enforce_min_notional=True, filters=fresh["filters"], **cfg["settings"])
    else:
        acct = new_account(strategy, config.PAPER_CAPITAL, fresh)
        acct.last_date = last_complete
        log.info("[%s] new paper account with $%.2f, first signals at the close after %s",
                 name, acct.cash, last_complete)

    n_trades = [len(acct.trades)]

    def before_step(date, bars):
        missing = acct.watch() - bars.keys()
        if missing:
            log.warning("[%s] %s: no bar yet for %s; waiting for next run", name, date, sorted(missing))
            return False
        n_trades[0] = len(acct.trades)
        return True

    def after_step(date):
        account = acct
        for t in account.trades[n_trades[0]:]:
            log.info("[%s] %s %s", name, date, _describe(t))
        if hasattr(account, "positions"):
            for sym, pos in account.positions.items():
                if pos.partial_date == date:
                    log.info("[%s] %s took profit on half of %s at +%sR (realised $%+.2f); rest keeps running",
                             name, date, sym, account.settings.get("partial_take_r"), pos.realized_pnl)
                elif pos.partial_date == f"skipped {date}":
                    log.info("[%s] %s %s reached the profit target but half is below the minimum order; "
                             "holding it all", name, date, sym)
                if pos.entry_date == date:
                    log.info("[%s] %s opened %s %s qty %.6g @ %.6g stop %.6g", name, date,
                             "SHORT" if pos.side == -1 else "LONG", sym, pos.qty, pos.entry_price, pos.stop)
            queued = [f"{'short' if o.get('side') == -1 else 'long'} {o['symbol']}" for o in account.pending_entries]
            log.info("[%s] %s equity $%.2f, %d open, queued entries %s, queued exits %s%s", name, date,
                     account.equity(), len(account.positions), queued, account.pending_exits,
                     " HALTED (kill switch)" if account.halted else "")
        else:
            targets = {k: round(float(v), 4) for k, v in sorted(account.targets.items(), key=lambda kv: -kv[1])}
            log.info("[%s] %s equity $%.2f, %d holdings, targets for next open %s%s", name, date,
                     account.equity(), len(account.holdings), targets,
                     " HALTED (kill switch)" if account.halted else "")
        hours_late = (pd.Timestamp.now(tz="UTC").tz_localize(None)
                      - pd.Timestamp(date) - pd.Timedelta(days=1)) / pd.Timedelta(hours=1)
        if hours_late > 3:
            log.warning("[%s] %s processed %.0fh after its close (PC was off?); next-open fills are "
                        "simulated as if placed on time", name, date, hours_late)

    run_days(acct, Panel(strategy.prepare_universe(universe)), after=acct.last_date, until=last_complete,
             before_step=before_step, after_step=after_step)
    log.info("[%s] run ok: processed through %s, equity $%.2f", name, acct.last_date, acct.equity())
    acct.save(state_path)
    _write_csv(config.PAPER_DIR / f"{name}_trades.csv", acct.trades)
    _write_csv(config.PAPER_DIR / f"{name}_equity.csv", [{"date": d, "equity": e} for d, e in acct.equity_curve])


def wait_for_binance(timeout_s: float = 600, every_s: float = 20) -> bool:
    """After boot or wake, Wi-Fi and DNS can take minutes; wait instead of failing every download."""
    deadline = time.time() + timeout_s
    while True:
        try:
            requests.get("https://data-api.binance.vision/api/v3/ping", timeout=10).raise_for_status()
            requests.get("https://fapi.binance.com/fapi/v1/ping", timeout=10).raise_for_status()
            return True
        except requests.RequestException:
            if time.time() >= deadline:
                return False
            time.sleep(every_s)


def main() -> None:
    config.PAPER_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=config.PAPER_DIR / "run.log", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    if not wait_for_binance():
        log.warning("no connection to Binance after 10 minutes; skipping this run, will retry next hour")
        return
    filters_path = config.DATA_DIR / "filters.json"
    if not filters_path.exists() or time.time() - filters_path.stat().st_mtime > 86_400:
        try:
            data.save_exchange_filters()
        except Exception:
            log.exception("could not refresh exchange order rules; using the cached copy")
    universes: dict = {}
    for name in config.ACCOUNTS:
        try:
            run_account(name, universes)
        except Exception:
            log.exception("[%s] paper run failed; will retry next run", name)
    try:
        listing_paper.run_all()
    except Exception:
        log.exception("[listing shorts] paper run failed; will retry next run")
    try:
        confidence_paper.run_all()
    except Exception:
        log.exception("[confidence_model] paper run failed; will retry next run")


if __name__ == "__main__":
    main()
