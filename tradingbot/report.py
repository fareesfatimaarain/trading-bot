import json

from . import config


def _equity(s: dict) -> float:
    if s.get("kind") == "weights":
        return s["cash"] + sum(h["qty"] * h["last_price"] for h in s["holdings"].values())
    if s["settings"].get("market") == "futures":
        return s["cash"] + sum(p.get("side", 1) * p["qty"] * (p["last_price"] - p["entry_price"])
                               for p in s["positions"].values())
    return s["cash"] + sum(p["qty"] * p["last_price"] for p in s["positions"].values())


def _confidence_report(name: str, s: dict) -> None:
    equity = s["cash"] + sum(p["notional"] * (1 + p["side"] * (p["last"] / p["entry"] - 1)) for p in s["positions"])
    pnl = equity - s["start_capital"]
    print(f"=== {name} (hourly model, trades only its top 1% confidence; target 2x / stop 1x daily volatility, 48h max)")
    print(f"equity ${equity:.2f} ({equity * config.USD_PKR:,.0f} PKR)  P&L ${pnl:+.2f} ({pnl / s['start_capital']:+.1%})  "
          f"processed through {s['last_hour'][:16]} UTC")
    for p in s["positions"]:
        move = p["side"] * (p["last"] / p["entry"] - 1)
        print(f"  {'LONG ' if p['side'] == 1 else 'SHORT'} {p['symbol']:<10} since {p['entry_time'][:16]}  entry {p['entry']:.6g}  "
              f"target {p['target']:.6g}  stop {p['stop']:.6g}  last {p['last']:.6g} ({move:+.1%})  "
              f"confidence {p['confidence']:.0%}")
    if s["trades"]:
        rs = [t["ret"] for t in s["trades"]]
        print(f"  closed trades {len(rs)}, won {sum(r > 0 for r in rs) / len(rs):.0%}, avg {sum(rs) / len(rs):+.2%}")
        for t in s["trades"][-5:]:
            print(f"    {t['exit_time'][:16]} {'L' if t['side'] == 1 else 'S'} {t['symbol']:<10} {t['reason']:<10} "
                  f"{t['ret']:+.2%} (${t['pnl']:+.2f})")
    elif not s["positions"]:
        print("  no confident signal yet (expect roughly one trade a day across the 10 coins)")
    print()


def _listing_report(name: str, s: dict) -> None:
    equity = s["cash"] + sum(p["notional"] * p["unrealised"] for p in s["positions"])
    pnl = equity - s["start_capital"]
    stop = f"stop +{s['stop']:.0%}" if s["stop"] else "no stop"
    print(f"=== {name} (short new listings 24h after launch, hold {s['hold_days']} days, {stop})")
    print(f"equity ${equity:.2f} ({equity * config.USD_PKR:,.0f} PKR)  P&L ${pnl:+.2f} ({pnl / s['start_capital']:+.1%})  "
          f"watching since {s['created']} UTC")
    for p in s["positions"]:
        print(f"  SHORT {p['symbol']:<14} since {p['entry_time'][:16]}  entry {p['entry']:.6g}  last {p['last']:.6g}  "
              f"{p['unrealised']:+.1%} (${p['notional'] * p['unrealised']:+.2f})  closes {p['exit_time'][:10]}")
    if s["trades"]:
        rs = [t["ret"] for t in s["trades"]]
        print(f"  closed shorts {len(rs)}, won {sum(r > 0 for r in rs) / len(rs):.0%}, avg {sum(rs) / len(rs):+.1%}")
        for t in s["trades"][-5:]:
            print(f"    {t['exit_time'][:10]} {t['symbol']:<14} {t['reason']:<16} {t['ret']:+.1%} (${t['pnl']:+.2f})")
    if s["skipped"]:
        print(f"  {len(s['skipped'])} listings not shorted (e.g. {s['skipped'][-1]['symbol']}: {s['skipped'][-1]['why']})")
    if not s["positions"] and not s["trades"]:
        print("  no new listings shorted yet")
    print()


def main() -> None:
    states = sorted(config.PAPER_DIR.glob("*_state.json"))
    if not states:
        print("No paper accounts yet. Run: .venv\\Scripts\\python -m tradingbot.paper")
        return
    for path in states:
        s = json.loads(path.read_text())
        name = path.name.removesuffix("_state.json")
        if s.get("kind") == "listing":
            _listing_report(name, s)
            continue
        if s.get("kind") == "confidence":
            _confidence_report(name, s)
            continue
        equity = _equity(s)
        pnl = equity - s["start_capital"]
        print(f"=== {name} (last processed bar: {s['last_date']}){'  HALTED' if s['halted'] else ''}")
        print(f"equity ${equity:.2f} ({equity * config.USD_PKR:,.0f} PKR)  "
              f"P&L ${pnl:+.2f} ({pnl / s['start_capital']:+.1%})  cash ${s['cash']:.2f}")
        if s.get("skipped_min_notional"):
            print(f"  {s['skipped_min_notional']} orders skipped: below Binance's minimum order size for that coin")
        if s.get("below_min_orders"):
            print(f"  {s['below_min_orders']} orders were below the exchange minimum (would be blocked live)")

        if s.get("kind") == "weights":
            for sym, h in sorted(s["holdings"].items()):
                value = h["qty"] * h["last_price"]
                print(f"  hold {sym:<12} ${value:>7.2f}  ({value / equity:.1%} of equity)  "
                      f"unrealised ${value - h['cost']:+.2f}")
            if s["targets"]:
                top = sorted(s["targets"].items(), key=lambda kv: -kv[1])
                print("  targets for next open: " + ", ".join(f"{k} {v:.1%}" for k, v in top))
            realised = sum(t["pnl"] for t in s["trades"])
            print(f"  orders {len(s['trades'])}, realised P&L ${realised:+.2f}")
            print()
            continue

        for sym, p in s["positions"].items():
            side = p.get("side", 1)
            move = side * (p["last_price"] / p["entry_price"] - 1)
            take_r = s["settings"].get("partial_take_r")
            if p.get("partial_date") and not str(p["partial_date"]).startswith("skipped"):
                note = f"  half sold {p['partial_date']} (+${p['realized_pnl']:.2f})"
            elif take_r:
                note = f"  target {p['entry_price'] + side * take_r * p['risk_per_unit']:.6g}"
            else:
                note = ""
            print(f"  {'SHORT' if side == -1 else 'LONG ':<5} {sym:<12} since {p['entry_date']}  "
                  f"entry {p['entry_price']:.6g}  stop {p['stop']:.6g}  last {p['last_price']:.6g} ({move:+.1%}){note}")
        if s["pending_entries"]:
            queued = [f"{'short' if o.get('side') == -1 else 'long'} {o['symbol']}" for o in s["pending_entries"]]
            print(f"  entries queued for next open: {queued}")
        if s["pending_exits"]:
            print(f"  exits queued for next open: {s['pending_exits']}")
        trades = s["trades"]
        if trades:
            rs = [t["r"] for t in trades]
            print(f"  closed trades {len(trades)}, win rate {sum(r > 0 for r in rs) / len(rs):.0%}, "
                  f"avg {sum(rs) / len(rs):+.2f}R")
            for t in trades[-5:]:
                side = "S" if t.get("side") == -1 else "L"
                print(f"    {t['exit_date']} {side} {t['symbol']:<12} {t['reason']:<12} "
                      f"${t['pnl']:+.2f} ({t['r']:+.2f}R)")
        print()


if __name__ == "__main__":
    main()
