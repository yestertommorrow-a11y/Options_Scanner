"""
main.py - Web version of the Options Scanner (FastAPI).

Vercel picks this file up automatically (FastAPI preset, top-level `app`).
The page itself is public/index.html, served by Vercel's CDN.

Local run:  uvicorn main:app --reload   then open http://127.0.0.1:8000
"""

from __future__ import annotations

import math
import re
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles

from scanner_core import (BUCKETS, DIVIDEND_YIELD, MODES, RISK_FREE_RATE, ScanConfig, fetch_chain,
                          format_summary, run_scans)

app = FastAPI(title="Options Scanner")

PUBLIC = Path(__file__).parent / "public"
TICKER_RE = re.compile(r"\^?[A-Z0-9][A-Z0-9.\-=]{0,11}")
CACHE_TTL = 120  # seconds; warm instances reuse a fetched chain for re-filtering
_cache: dict = {}

COLUMNS = ["type", "expiry", "dte", "strike", "bid", "ask", "last", "mid", "volume", "oi",
           "vol_oi", "iv", "delta", "gamma", "theta", "vega", "ref_iv", "theo", "edge_pct",
           "spread_pct", "bucket", "signal"]


def _clean(v):
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if hasattr(v, "item"):  # numpy scalar -> python
        return _clean(v.item())
    return v


def _get_chain(ticker: str, min_dte: int, max_dte: int):
    key = (ticker, min_dte, max_dte)
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_TTL:
        return hit[1], True
    data = fetch_chain(ticker, min_dte, max_dte)
    _cache.clear()  # keep memory small: one chain per instance
    _cache[key] = (time.time(), data)
    return data, False


def _chart_chain(chain) -> tuple[list, list]:
    """Compact per-contract arrays for the smile and activity charts."""
    if chain.empty:
        return [], []
    exp = chain["expiry"].dt.strftime("%Y-%m-%d")
    rows = [
        [e, t[0], round(float(k), 2), round(float(iv), 4), int(v), int(o), round(float(d), 3)]
        for e, t, k, iv, v, o, d in zip(exp, chain["type"], chain["strike"], chain["iv"],
                                        chain["volume"], chain["oi"], chain["delta"])
    ]
    expiries = (chain.assign(e=exp)[["e", "dte"]].drop_duplicates().sort_values("dte")
                .rename(columns={"e": "expiry"}).to_dict("records"))
    return rows, [{"expiry": x["expiry"], "dte": int(x["dte"])} for x in expiries]


@app.get("/api/meta")
def meta():
    return {"modes": MODES, "buckets": BUCKETS, "defaults": ScanConfig().__dict__}


@app.get("/api/scan")
def scan(
    ticker: str = Query(..., min_length=1, max_length=12),
    mode: str = MODES[0],
    bucket: str = "All",
    min_dte: int = Query(0, ge=0, le=730),
    max_dte: int = Query(45, ge=0, le=730),
    min_volume: int = Query(50, ge=0),
    min_oi: int = Query(10, ge=0),
    uoa_ratio: float = Query(1.0, ge=0),
    edge_pct: float = Query(15.0, ge=0),
    max_spread_pct: float = Query(10.0, ge=0),
    min_price: float = Query(0.10, ge=0),
):
    ticker = ticker.strip().upper()
    # Yahoo quietly ignores junk like "SPY;X" and returns SPY, which would label SPY data
    # with the wrong name. Accept only real symbol characters: letters, digits, . - ^ =
    if not TICKER_RE.fullmatch(ticker):
        raise HTTPException(400, "Please enter a ticker symbol such as SPY, AAPL, BRK-B or ^SPX "
                                 "(letters, digits, '.', '-', '^' or '=' only).")
    if mode not in MODES:
        raise HTTPException(400, f"Unknown scan type '{mode}'.")
    if bucket != "All" and bucket not in BUCKETS:
        raise HTTPException(400, f"Unknown delta bucket '{bucket}'.")
    if min_dte > max_dte:
        raise HTTPException(400, "Min DTE must not exceed Max DTE.")

    try:
        (spot, chain, summary), cached = _get_chain(ticker, min_dte, max_dte)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:  # network / Yahoo errors
        raise HTTPException(502, f"Data fetch failed: {exc}")

    cfg = ScanConfig(min_volume, min_oi, uoa_ratio, edge_pct, max_spread_pct, min_price, bucket)
    result = run_scans(chain, cfg, mode)

    chart_rows, expiries = _chart_chain(chain)
    rows = []
    for rec in result.reindex(columns=COLUMNS).to_dict("records"):
        rec["expiry"] = rec["expiry"].strftime("%Y-%m-%d") if rec["expiry"] is not None else None
        rows.append({k: _clean(v) for k, v in rec.items()})

    return {
        "ticker": ticker,
        "summary": {k: _clean(v) for k, v in summary.items()},
        "summary_text": format_summary(ticker, summary),
        "contracts": len(chain),
        "two_sided": int(((chain["bid"] > 0) & (chain["ask"] > 0)).sum()) if len(chain) else 0,
        "cached": cached,
        "r": RISK_FREE_RATE,
        "q": DIVIDEND_YIELD,
        "config": {**cfg.__dict__, "mode": mode},
        "rows": rows,
        "chain": chart_rows,  # [expiry, c/p, strike, iv, volume, oi, delta]
        "expiries": expiries,
    }


# Local development: serve public/ (index.html, charts.js). Registered last so /api routes win.
# On Vercel, public/ is served by the CDN before requests reach this app.
if PUBLIC.is_dir():
    app.mount("/", StaticFiles(directory=PUBLIC, html=True), name="public")
