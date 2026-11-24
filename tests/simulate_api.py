"""
Simulated form entries against a running server (live Yahoo data).

Start the server first:  .venv\\Scripts\\python -m uvicorn main:app --port 8000
Run:                     .venv\\Scripts\\python tests\\simulate_api.py

Sends good, odd and bad inputs, prints the status and message for each, and checks
that every returned row obeys the rule for its scan type.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.parse
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
BUCKETS = ["Lotto", "Sensible", "Directional", "Deep ITM"]


def call(**params):
    url = BASE + "/api/scan?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            return e.code, json.loads(body)
        except ValueError:
            return e.code, {"detail": body.decode(errors="replace")[:200]}


def rule_ok(row, cfg, mode):
    v, oi = row["volume"] or 0, row["oi"] or 0
    if not (v >= cfg["min_volume"] and oi >= cfg["min_oi"]):
        return False
    uoa = v >= cfg["min_volume"] and row["vol_oi"] is not None and row["vol_oi"] >= cfg["uoa_ratio"]
    sp, mid, ed, d = row["spread_pct"], row["mid"] or 0, row["edge_pct"], abs(row["delta"])
    mis = 0.2 <= d <= 0.8 and mid >= cfg["min_price"] and sp is not None and sp <= cfg["max_spread_pct"] \
        and ed is not None and abs(ed) >= cfg["edge_pct"]
    tight = sp is not None and sp <= cfg["max_spread_pct"] and mid >= cfg["min_price"]
    buck = mid >= cfg["min_price"] and (cfg["bucket"] not in BUCKETS or row["bucket"] == cfg["bucket"])
    return {"UOA + Mispricing": uoa or mis, "Unusual Activity": uoa, "Mispricing": mis,
            "Tight Spreads": tight, "Delta Bucket": buck}[mode]


CASES = [
    ("defaults", dict(ticker="SPY")),
    ("lower-case + spaces", dict(ticker="  spy ")),
    ("single stock", dict(ticker="AAPL", mode="Mispricing")),
    ("each scan type", None),
    ("each bucket", None),
    ("loose filters", dict(ticker="SPY", mode="Tight Spreads", min_volume=0, min_oi=0, max_spread_pct=100, min_price=0)),
    ("impossible filters", dict(ticker="SPY", min_volume=10**9)),
    ("long DTE window", dict(ticker="SPY", min_dte=200, max_dte=730)),
    ("no expiries in window", dict(ticker="SPY", min_dte=729, max_dte=730)),
    ("min > max DTE", dict(ticker="SPY", min_dte=30, max_dte=5)),
    ("negative volume", dict(ticker="SPY", min_volume=-5)),
    ("text in number", dict(ticker="SPY", edge_pct="abc")),
    ("empty number", dict(ticker="SPY", min_oi="")),
    ("empty ticker", dict(ticker="")),
    ("blank ticker", dict(ticker="   ")),
    ("unknown ticker", dict(ticker="ZZZZQX")),
    ("symbol chars", dict(ticker="SPY;DROP")),
    ("too long ticker", dict(ticker="ABCDEFGHIJKLM")),
    ("unknown scan type", dict(ticker="SPY", mode="Magic")),
    ("unknown bucket", dict(ticker="SPY", mode="Delta Bucket", bucket="Moonshot")),
    ("index-style ticker", dict(ticker="^SPX")),
    ("class-share ticker", dict(ticker="BRK-B")),
]

problems = []
for name, params in CASES:
    if params is None:
        sets = ([dict(ticker="SPY", mode=m) for m in
                 ["UOA + Mispricing", "Unusual Activity", "Mispricing", "Tight Spreads", "Delta Bucket"]]
                if name == "each scan type" else
                [dict(ticker="SPY", mode="Delta Bucket", bucket=b) for b in ["All"] + BUCKETS])
    else:
        sets = [params]
    for p in sets:
        status, body = call(**p)
        if status == 200:
            bad = [r for r in body["rows"] if not rule_ok(r, body["config"], body["config"]["mode"])]
            msg = f"{len(body['rows'])} rows / {body['contracts']} contracts, {len(body['expiries'])} expiries"
            if bad:
                problems.append(f"{name} {p}: {len(bad)} rows break the rule")
                msg += f"  !! {len(bad)} rows break the rule"
        else:
            d = body.get("detail")
            msg = d if isinstance(d, str) else json.dumps(d)[:160]
        print(f"[{status}] {name:24s} {json.dumps(p)[:70]:72s} {msg}")

print("\nRule violations:", problems or "none")
