"""
main.py - Web version of the Options Scanner (FastAPI).

Vercel picks this file up automatically (FastAPI preset, top-level `app`).
The page itself is public/index.html, served by Vercel's CDN.

Local run:  uvicorn main:app --reload   then open http://127.0.0.1:8000
"""

from __future__ import annotations

import math
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles

from scanner_core import (BUCKETS, DIVIDEND_YIELD, MAX_DTE, MODES, RISK_FREE_RATE, TICKER_RE,  # noqa: F401
                          ScanConfig, fetch_chain, format_summary, normalize_ticker, run_scans,
                          validate_settings)

app = FastAPI(title="Options Scanner")

PUBLIC = Path(__file__).parent / "public"
CACHE_TTL = 120  # seconds; warm instances reuse a fetched chain for re-filtering
_cache: dict = {}

COLUMNS = ["type", "expiry", "dte", "strike", "bid", "ask", "last", "mid", "volume", "oi",
           "vol_oi", "iv", "iv_yahoo", "iv_source", "quote", "otm", "spot_used", "moneyness", "ref_method", "ref_supported", "edge_note", "delta", "gamma", "theta", "vega", "ref_iv", "theo", "edge_pct",
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
    # Per-expiry price and at-the-money IV, so charts for a later expiry use that expiry's own
    # option-implied price (it differs from the summary expiry's) and its own strike window.
    expiries = []
    for (e, dte), g in chain.assign(e=exp).groupby(["e", "dte"], sort=False):
        spot = float(g["spot_used"].iloc[0])
        atm_k = g.loc[(g["strike"] - spot).abs().idxmin(), "strike"]
        atm_iv = float(g.loc[g["strike"] == atm_k, "iv"].mean())
        expiries.append({"expiry": e, "dte": int(dte), "spot": round(spot, 4),
                         "atm_iv": round(atm_iv, 6) if math.isfinite(atm_iv) else None})
    return rows, sorted(expiries, key=lambda x: x["dte"])


@app.get("/api/meta")
def meta():
    return {"modes": MODES, "buckets": BUCKETS, "defaults": ScanConfig().__dict__}


@app.get("/api/scan")
def scan(
    # The symbol rule (1-12 characters after trimming) is checked by normalize_ticker; the
    # raw length cap here only stops absurd inputs, so padded "  spy " is still accepted.
    ticker: str = Query(..., min_length=1, max_length=40),
    mode: str = MODES[0],
    bucket: str = "All",
    min_dte: int = Query(0, ge=0, le=MAX_DTE),
    max_dte: int = Query(45, ge=0, le=MAX_DTE),
    min_volume: int = Query(50, ge=0),
    min_oi: int = Query(10, ge=0),
    uoa_ratio: float = Query(1.0, ge=0, allow_inf_nan=False),
    edge_pct: float = Query(5.0, ge=0, allow_inf_nan=False),
    max_spread_pct: float = Query(10.0, ge=0, allow_inf_nan=False),
    min_price: float = Query(0.10, ge=0, allow_inf_nan=False),
):
    # Same validation as the desktop tab (scanner_core), checked before any data is fetched.
    if mode not in MODES:
        raise HTTPException(400, f"Unknown scan type '{mode}'.")
    cfg = ScanConfig(min_volume, min_oi, uoa_ratio, edge_pct, max_spread_pct, min_price, bucket)
    try:
        ticker = normalize_ticker(ticker)
        validate_settings(min_dte, max_dte, cfg)
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    try:
        (spot, chain, summary), cached = _get_chain(ticker, min_dte, max_dte)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:  # network / Yahoo errors
        raise HTTPException(502, f"Data fetch failed: {exc}")

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
        "two_sided": int((chain["quote"] == "live").sum()) if len(chain) else 0,  # same rule as pricing
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
