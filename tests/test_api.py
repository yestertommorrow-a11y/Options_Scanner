"""
Deterministic API tests: FastAPI driven in-process (no server, no network), Yahoo stubbed.

Run:  .venv\\Scripts\\python -B tests\\test_api.py      (exit code 1 on any failure)

Checks exact statuses, field-specific errors (before any fetch), exact returned rows, the
live-quote count, per-expiry chart data and partial provider responses.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import urlencode

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import main  # noqa: E402
import scanner_core as sc  # noqa: E402
from test_scanner import TODAY, fair_row, raw_row  # noqa: E402

FAILS: list[str] = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)


async def _get(path, params):
    events, sent = [], False

    async def receive():
        nonlocal sent
        if sent:
            await asyncio.Event().wait()
        sent = True
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(ev):
        events.append(ev)

    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "GET", "scheme": "http",
             "path": path, "root_path": "", "query_string": urlencode(params).encode(), "headers": [],
             "client": ("test", 0), "server": ("test", 80)}
    await main.app(scope, receive, send)
    status = next(e["status"] for e in events if e["type"] == "http.response.start")
    body = b"".join(e.get("body", b"") for e in events if e["type"] == "http.response.body")
    return status, json.loads(body)


def get(**params):
    return asyncio.run(_get("/api/scan", {"ticker": "TEST", **params}))


def fixture(ticker, min_dte, max_dte):
    """Stub for main.fetch_chain."""
    if ticker == "TEST":
        rows = [raw_row(dte=7), raw_row(k=104, dte=30), raw_row(k=106, bid=4.2, ask=3.8, dte=30)]  # last one crossed
    elif ticker == "EXPIRY":  # two expiries whose quotes imply different prices: $100 and $109
        rows = [fair_row(k, kind, dte=dte, spot=spot, iv=iv)
                for dte, spot, iv, ks in ((7, 100.0, .11, range(96, 105)), (45, 109.0, .22, range(103, 116)))
                for k in ks for kind in ("call", "put")]
    elif ticker == "FLAT":
        rows = [fair_row(k, kind) for k in range(94, 108, 2) for kind in ("call", "put")]
    else:
        raise ValueError(f"No price data found for '{ticker}'.")
    rows = [r for r in rows if min_dte <= (pd.Timestamp(r["expiry"]) - TODAY).days <= max_dte]
    if not rows:
        raise ValueError("No expiries in window.")
    ch = sc.prepare_chain(pd.DataFrame(rows), 100.0, TODAY)
    closes = pd.Series([100.0] * 25) if ticker == "FLAT" else None
    return 100.0, ch, sc.summarize(ch, 100.0, closes)


def run():
    main._cache.clear()
    with patch.object(main, "fetch_chain", side_effect=fixture) as fetch:
        # exact rows: the crossed 106 call is not live, so Tight Spreads returns only the two live calls
        st, b = get(mode="Tight Spreads")
        check(st == 200 and sorted((r["strike"], r["dte"]) for r in b["rows"]) == [(100, 7), (104, 30)],
              f"Tight Spreads rows {[(r['strike'], r['dte']) for r in b.get('rows', [])]}")
        check(b["two_sided"] == 2 and b["contracts"] == 3, f"live quote count {b.get('two_sided')} of {b.get('contracts')}")
        crossed = next((r for r in b["chain"] if r[2] == 106), None)
        check(crossed is not None, "crossed contract still listed in chart data")
        st, b = get(ticker="  test ", mode="Delta Bucket", min_price=0)
        check(st == 200 and b["ticker"] == "TEST" and len(b["rows"]) == 3 and b["cached"], "normalised ticker hits the cache")
        st, b = get(min_price=0.12, uoa_ratio=1.25, edge_pct=5.2, max_spread_pct=9.9)
        check(st == 200 and b["config"]["edge_pct"] == 5.2 and b["config"]["max_spread_pct"] == 9.9, "decimal settings")
        st, b = get(min_volume=0, min_oi=0, uoa_ratio=0, edge_pct=0, max_spread_pct=0, min_price=0)
        check(st == 200, "zero thresholds must be accepted")

        calls = fetch.call_count
        bad = [({"uoa_ratio": "inf"}, 422, "uoa_ratio"), ({"edge_pct": "inf"}, 422, "edge_pct"),
               ({"max_spread_pct": "1e309"}, 422, "max_spread_pct"), ({"min_price": "-inf"}, 422, "min_price"),
               ({"uoa_ratio": "nan"}, 422, "uoa_ratio"), ({"min_volume": -1}, 422, "min_volume"),
               ({"min_oi": ""}, 422, "min_oi"), ({"max_dte": 731}, 422, "max_dte"), ({"min_dte": 30, "max_dte": 7}, 400, None),
               ({"ticker": "SPY;X"}, 400, None), ({"ticker": "ABCDEFGHIJKLM"}, 400, None),
               ({"ticker": "^ABCDEFGHIJKL"}, 400, None), ({"ticker": "   "}, 400, None),
               ({"mode": "Magic"}, 400, None), ({"bucket": "Moonshot"}, 400, None)]
        for params, want, field in bad:
            st, b = get(**params)
            ok = st == want and (field is None or any(e["loc"][-1] == field for e in b.get("detail", [])))
            check(ok, f"{params}: HTTP {st} {str(b.get('detail'))[:120]}, expected {want} on {field}")
        check(fetch.call_count == calls, "invalid input reached the provider")
        for t in ("ABCDEFGHIJKL", "^ABCDEFGHIJK", "  abcdefghijkl  "):  # 12 characters after trimming: accepted
            st, b = get(ticker=t)
            check(st == 400 and "No price data" in str(b.get("detail")), f"{t!r} should pass validation and reach the stub")

        # per-expiry price and ATM IV for the dropdown charts
        st, b = get(ticker="EXPIRY")
        by = {e["dte"]: e for e in b["expiries"]}
        check(abs(by[7]["spot"] - 100) < 0.01 and abs(by[45]["spot"] - 109) < 0.01, f"per-expiry spots {by}")
        check(abs(by[7]["atm_iv"] - .11) < .002 and abs(by[45]["atm_iv"] - .22) < .002, "per-expiry ATM IV")

        # flat price history: RV 0, ratio omitted, still a valid 200
        st, b = get(ticker="FLAT")
        check(st == 200 and b["summary"]["rv20"] == 0 and "iv_rv" not in b["summary"], "flat history summary")

    # partial provider responses through the real fetch_chain
    e1, e2 = (TODAY + pd.Timedelta(days=7)).strftime("%Y-%m-%d"), (TODAY + pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    good = pd.DataFrame([raw_row()]).drop(columns=["type", "expiry"])
    for label, chains, want in (("missing first expiry", {e1: SimpleNamespace(calls=None, puts=None), e2: SimpleNamespace(calls=good, puts=good)}, 200),
                                ("all missing", {e1: SimpleNamespace(calls=None, puts=None), e2: None}, 400)):
        class T:
            fast_info = {"last_price": 100.0}
            options = [e1, e2]
            def history(self, period):
                return pd.DataFrame({"Close": [100.0] * 25})
            def option_chain(self, exp, c=chains):
                return c[exp]
        main._cache.clear()
        with patch.dict(sys.modules, {"yfinance": SimpleNamespace(Ticker=lambda _: T())}), \
                patch.object(sc, "market_today", return_value=TODAY):
            st, b = get(ticker="PART")
        check(st == want, f"{label}: HTTP {st} {str(b.get('detail'))[:100]}")
        if st == 200:
            check(b["summary"]["expiries_scanned"] == 1 and b["summary"]["expiries_missing"] == [e1], f"{label} coverage {b['summary']}")


if __name__ == "__main__":
    run()
    if FAILS:
        print(f"{len(FAILS)} FAILED")
        for f in FAILS:
            print(" -", f)
        sys.exit(1)
    print("All API tests passed.")
