"""
Live smoke test: form entries against a running server with real Yahoo data.

Start the server first:  .venv\\Scripts\\python -m uvicorn main:app --port 8000
Run:                     .venv\\Scripts\\python tests\\simulate_api.py

Every case has an expected HTTP status. Each returned row must obey its scan rule. The script
exits with code 1 on any mismatch, so an outage (every request 502) is a failure, not a pass.
Deterministic, offline checks with exact expected rows live in tests/test_api.py.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.parse
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
BUCKETS = ["Lotto", "Sensible", "Directional", "Deep ITM"]
MODES = ["UOA + Mispricing", "Unusual Activity", "Mispricing", "Tight Spreads", "Delta Bucket"]


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
    except (urllib.error.URLError, OSError) as e:
        return 0, {"detail": f"server unreachable: {e}"}


def rule_ok(row, cfg, mode):
    v, oi = row["volume"] or 0, row["oi"] or 0
    if not (v >= cfg["min_volume"] and oi >= cfg["min_oi"]):
        return False
    uoa = v >= cfg["min_volume"] and row["vol_oi"] is not None and row["vol_oi"] >= cfg["uoa_ratio"]
    sp, mid, ed, d = row["spread_pct"], row["mid"] or 0, row["edge_pct"], abs(row["delta"])
    mis = 0.2 <= d <= 0.8 and mid >= cfg["min_price"] and sp is not None and sp <= cfg["max_spread_pct"] \
        and ed is not None and abs(ed) >= cfg["edge_pct"]
    tight = sp is not None and 0 <= sp <= cfg["max_spread_pct"] and mid >= cfg["min_price"]
    buck = mid >= cfg["min_price"] and (cfg["bucket"] not in BUCKETS or row["bucket"] == cfg["bucket"])
    return {"UOA + Mispricing": uoa or mis, "Unusual Activity": uoa, "Mispricing": mis,
            "Tight Spreads": tight, "Delta Bucket": buck}[mode]


CASES = (
    [("defaults", dict(ticker="SPY"), 200),
     ("lower-case + spaces", dict(ticker="  spy "), 200),
     ("single stock", dict(ticker="AAPL", mode="Mispricing"), 200)]
    + [(f"scan type {m}", dict(ticker="SPY", mode=m), 200) for m in MODES]
    + [(f"bucket {b}", dict(ticker="SPY", mode="Delta Bucket", bucket=b), 200) for b in ["All"] + BUCKETS]
    + [("loose filters", dict(ticker="SPY", mode="Tight Spreads", min_volume=0, min_oi=0, max_spread_pct=100, min_price=0), 200),
       ("decimal filters", dict(ticker="SPY", min_price=0.12, uoa_ratio=1.25, edge_pct=5.2, max_spread_pct=9.9), 200),
       ("impossible filters", dict(ticker="SPY", min_volume=10**9), 200),
       ("long DTE window", dict(ticker="SPY", min_dte=200, max_dte=730), 200),
       ("index-style ticker", dict(ticker="^SPX"), 200),
       ("class-share ticker", dict(ticker="BRK-B"), 200),
       ("no expiries in window", dict(ticker="SPY", min_dte=729, max_dte=730), 400),
       ("min > max DTE", dict(ticker="SPY", min_dte=30, max_dte=5), 400),
       ("unknown ticker", dict(ticker="ZZZZQX"), 400),
       ("symbol chars", dict(ticker="SPY;DROP"), 400),
       ("13-char ticker", dict(ticker="ABCDEFGHIJKLM"), 400),
       ("blank ticker", dict(ticker="   "), 400),
       ("unknown scan type", dict(ticker="SPY", mode="Magic"), 400),
       ("unknown bucket", dict(ticker="SPY", mode="Delta Bucket", bucket="Moonshot"), 400),
       ("empty ticker", dict(ticker=""), 422),
       ("negative volume", dict(ticker="SPY", min_volume=-5), 422),
       ("text in number", dict(ticker="SPY", edge_pct="abc"), 422),
       ("empty number", dict(ticker="SPY", min_oi=""), 422),
       ("infinite threshold", dict(ticker="SPY", uoa_ratio="inf"), 422),
       ("Max DTE too large", dict(ticker="SPY", max_dte=731), 422)]
)

if __name__ == "__main__":
    problems = []
    for name, params, want in CASES:
        status, body = call(**params)
        if status == 200:
            bad = [r for r in body["rows"] if not rule_ok(r, body["config"], body["config"]["mode"])]
            msg = f"{len(body['rows'])} rows / {body['contracts']} contracts, {len(body['expiries'])} expiries"
            if bad:
                problems.append(f"{name}: {len(bad)} rows break the rule")
                msg += f"  !! {len(bad)} rows break the rule"
        else:
            d = body.get("detail")
            msg = d if isinstance(d, str) else json.dumps(d)[:160]
        if status != want:
            problems.append(f"{name}: HTTP {status}, expected {want} ({msg[:120]})")
        flag = "ok " if status == want else "BAD"
        print(f"{flag} [{status}] {name:24s} {json.dumps(params)[:60]:62s} {msg}")

    print(f"\n{len(CASES)} cases, {len(problems)} problems")
    for p in problems:
        print(" -", p)
    sys.exit(1 if problems else 0)
