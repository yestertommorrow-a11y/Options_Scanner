"""
Logic tests on simulated option chains (no network).

Run:  .venv\\Scripts\\python tests\\test_scanner.py
Checks the pricing maths, chain preparation, every scan mode against random
settings (each listed row must pass its rule, and every passing contract must be
listed), edge cases, and the summary figures.
"""

from __future__ import annotations

import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scanner_core as sc  # noqa: E402

TODAY = pd.Timestamp("2026-11-24")
FAILS: list[str] = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)


def synthetic_chain(seed: int, spot: float = 100.0, n_exp: int = 4, junk: bool = True) -> pd.DataFrame:
    """A yfinance-shaped raw chain with skew, noise and the junk real data contains."""
    rng = np.random.default_rng(seed)
    rows = []
    for e in range(n_exp):
        dte = [0, 3, 10, 30, 60][e % 5]
        exp = (TODAY + pd.Timedelta(days=dte)).strftime("%Y-%m-%d")
        T = max(dte, 0.5) / 365
        for K in np.arange(spot * 0.7, spot * 1.3 + 1e-9, spot * 0.025):
            for kind in ("call", "put"):
                iv = 0.22 + 1.0 * max(0.0, (spot - K) / spot) + rng.normal(0, 0.005)
                p, _, _ = sc.bs_model(spot, K, T, 0.045, 0.0, iv, kind == "call")
                p = float(p) * rng.uniform(0.98, 1.02)  # IV is solved from this price, so keep noise modest
                half = max(0.01, p * rng.uniform(0.005, 0.2))
                bid, ask = max(p - half, 0.0), p + half
                r = rng.random() if junk else 1.0
                if r < 0.05:
                    bid = ask = 0.0                      # market closed / no quote
                elif r < 0.08:
                    iv = 0.00001                         # yfinance dead-contract IV
                elif r < 0.10:
                    iv = float("nan")
                rows.append(dict(
                    strike=round(K, 2), bid=round(bid, 2), ask=round(ask, 2), lastPrice=round(p, 2),
                    volume=float(rng.choice([np.nan, 0, rng.integers(1, 5000)])),
                    openInterest=float(rng.choice([np.nan, 0, rng.integers(1, 20000)])),
                    impliedVolatility=iv, type=kind, expiry=exp))
    return pd.DataFrame(rows)


def test_pricing():
    S, K, T, r, q, s = 100, 105, 30 / 365, 0.045, 0.01, 0.25
    c, dc, g = sc.bs_model(S, K, T, r, q, s, True)
    p, dp, _ = sc.bs_model(S, K, T, r, q, s, False)
    parity = float(c - p) - (S * math.exp(-q * T) - K * math.exp(-r * T))
    check(abs(parity) < 1e-9, f"put-call parity off by {parity}")
    check(abs(float(dc - dp) - math.exp(-q * T)) < 1e-9, "call delta - put delta != e^-qT")
    h = 1e-4
    fd = (sc.bs_model(S + h, K, T, r, q, s, True)[1] - sc.bs_model(S - h, K, T, r, q, s, True)[1]) / (2 * h)
    check(abs(float(fd) - float(g)) < 1e-5, "gamma != d(delta)/dS")
    for call in (True, False):
        th, ve = sc.bs_theta_vega(S, K, T, r, q, s, call)
        dt = 1e-5
        fd_th = (sc.bs_model(S, K, T - dt, r, q, s, call)[0] - sc.bs_model(S, K, T, r, q, s, call)[0]) / dt / 365
        fd_ve = (sc.bs_model(S, K, T, r, q, s + 1e-5, call)[0] - sc.bs_model(S, K, T, r, q, s, call)[0]) / 1e-5 / 100
        check(abs(float(th) - float(fd_th)) < 1e-5, f"theta mismatch call={call}")
        check(abs(float(ve) - float(fd_ve)) < 1e-5, f"vega mismatch call={call}")


def test_implied_vol():
    rng = np.random.default_rng(11)
    n = 400
    S, K = 100.0, rng.uniform(60, 140, n)
    T, iv = rng.uniform(1, 400, n) / 365, rng.uniform(0.05, 1.5, n)
    call = rng.random(n) < 0.5
    price = sc.bs_model(S, K, T, 0.045, 0.01, iv, call)[0]
    got = sc.implied_vol(price, S, K, T, 0.045, 0.01, call)
    vega = sc.bs_theta_vega(S, K, T, 0.045, 0.01, iv, call)[1]
    sens = vega > 1e-6  # deep wings have ~zero vega: many IVs give the same price
    check(np.all(np.abs(got[sens] - iv[sens]) < 1e-6), "implied_vol does not recover the IV")
    # below the zero-volatility price, or above the price at 500% vol: no IV exists
    floor = sc.bs_model(S, K, T, 0.045, 0.01, np.full(n, 1e-6), call)[0]
    check(np.isnan(sc.implied_vol(floor - 0.05, S, K, T, 0.045, 0.01, call)).all(), "IV below the zero-vol price")
    check(np.isnan(sc.implied_vol(np.full(n, S * 2), S, K, T, 0.045, 0.01, call)).all(), "IV for a price above the stock")


def test_prepare(chain: pd.DataFrame, spot: float):
    # IV solved from the mid reprices the mid, and then the edge has the sign of the IV gap
    m = (chain["iv_source"] == "mid").to_numpy()
    T = np.maximum(chain["dte"].to_numpy(dtype=float), 0.5) / 365
    rep = sc.bs_model(chain["spot_used"].to_numpy(), chain["strike"], T, sc.RISK_FREE_RATE, sc.DIVIDEND_YIELD, chain["iv"], (chain["type"] == "call").to_numpy())[0]
    check(np.allclose(rep[m], chain.loc[m, "mid"], rtol=1e-6, atol=1e-6), "mid-solved IV does not reprice the mid")
    g = m & chain["ref_iv"].notna().to_numpy() & (chain["edge_pct"].abs() > 0.5).to_numpy()
    check((np.sign(chain.loc[g, "edge_pct"]) == np.sign(chain.loc[g, "ref_iv"] - chain.loc[g, "iv"])).all(),
          "edge sign disagrees with the IV gap")
    check(chain["iv"].between(0.01, 5, inclusive="neither").all(), "junk IVs not dropped")
    check((chain["dte"] >= 0).all(), "negative DTE kept")
    two = (chain["bid"] > 0) & (chain["ask"] > 0) & (chain["ask"] >= chain["bid"])
    check((chain["quote"] == "live").eq(two).all(), "quote flag disagrees with the live-quote rule")
    check(np.allclose(chain.loc[two, "mid"], (chain.loc[two, "bid"] + chain.loc[two, "ask"]) / 2), "mid != (bid+ask)/2")
    check(chain.loc[~two, "spread_pct"].isna().all(), "spread shown without a live quote")
    check((chain["spread_pct"].dropna() >= 0).all(), "negative spread")
    check((chain.loc[chain["type"] == "call", "delta"] >= 0).all(), "negative call delta")
    check((chain.loc[chain["type"] == "put", "delta"] <= 0).all(), "positive put delta")
    ad = chain["delta"].abs()
    want = np.select([ad < 0.15, ad < 0.35, ad < 0.65], ["Lotto", "Sensible", "Directional"], "Deep ITM")
    check((chain["bucket"] == want).all(), "bucket does not match delta rule")
    ok = chain["edge_pct"].notna()
    edge = (chain["theo"] - chain["mid"]) / chain["mid"] * 100
    check(np.allclose(chain.loc[ok, "edge_pct"], edge[ok]), "edge % formula mismatch")
    elig = chain["otm"] & (chain["iv_source"] == "mid") & (chain["ref_method"] == "smile") & chain["ref_supported"]
    check(ok.eq(elig & (chain["mid"] > 0)).all(), "edge given outside the eligibility rule")
    check((chain["edge_note"] == "").eq(ok).all(), "edge_note must be empty exactly when an edge exists")
    check(chain.loc[~chain["otm"], "edge_pct"].isna().all(), "in-the-money options must not get a Rich/Cheap edge")
    otm = np.where(chain["type"] == "call", chain["strike"] >= chain["spot_used"], chain["strike"] <= chain["spot_used"])
    check((chain["otm"].to_numpy() == otm).all(), "otm flag wrong")
    itm_pair = chain[chain["iv_source"] == "pair"]
    for r in itm_pair.head(50).itertuples():  # ITM IV comes from the OTM option at the same strike
        twin = chain[(chain["expiry"] == r.expiry) & (chain["strike"] == r.strike) & chain["otm"]]
        check(len(twin) and abs(twin["iv"].mean() - r.iv) < 1e-12, f"pair IV mismatch at {r.strike}")
    # reference IV: the expiry's smile at this strike, fitted WITHOUT the contract itself when it
    # is one of the fitting points; supported only inside the strike range of the points used
    for exp, e in chain.groupby("expiry"):
        pts = sc.smile_points(e).index
        for label, r in e[e["ref_method"] == "smile"].head(40).iterrows():
            res = sc.fit_smile(e, exclude=label if label in pts else None)
            check(res is not None, f"{exp:%Y-%m-%d} K={r.strike}: marked smile but no fit")
            if res is None:
                continue
            coef, lo, hi = res
            check(abs(np.clip(np.polyval(coef, r.moneyness), 0.01, 5.0) - r.ref_iv) < 1e-9,
                  f"{exp:%Y-%m-%d} K={r.strike}: smile reference wrong")
            check(bool(r.ref_supported) == (lo <= r.moneyness <= hi), f"{exp:%Y-%m-%d} K={r.strike}: support flag wrong")


def skewed_chain(spot=500.0, stale=1.0, bump=None):
    """Clean quotes on a smooth index-style smile; IV depends on strike only (as parity requires).
    bump=(dte, strike, type, factor) multiplies one contract's price to plant a real mispricing."""
    rows = []
    for dte in (7, 30):
        exp = (TODAY + pd.Timedelta(days=dte)).strftime("%Y-%m-%d")
        for K in np.arange(spot * 0.85, spot * 1.15 + 1e-9, 5):
            k = math.log(K / spot)
            iv = 0.20 - 0.6 * k + 1.5 * k * k  # puts' side higher: normal skew
            for kind in ("call", "put"):
                p = float(sc.bs_model(spot, K, dte / 365, 0.045, 0.0, iv, kind == "call")[0])
                if bump and (dte, K, kind) == bump[:3]:
                    p *= bump[3]
                if p < 0.05:
                    continue
                rows.append(dict(strike=K, bid=round(p * 0.995, 2), ask=round(p * 1.005, 2), lastPrice=p,
                                 volume=1000, openInterest=800, impliedVolatility=iv, type=kind, expiry=exp))
    return sc.prepare_chain(pd.DataFrame(rows), spot * stale, TODAY)


def test_skew_is_not_mispricing(stale: float = 1.0):
    """Skew alone, even with a stale stock quote, must produce no Rich/Cheap signals at all.
    (The original code flagged 12-13 calls Cheap here with a 1.3% stale quote.)"""
    out = sc.run_scans(skewed_chain(stale=stale), sc.ScanConfig(), "Mispricing")
    check(len(out) == 0, f"skew with quote x{stale} flagged {len(out)} contracts "
                         f"({(out['type'] == 'call').sum()} calls, {(out['type'] == 'put').sum()} puts)")


def test_finds_real_mispricing():
    """A single contract priced 30% off the smile is flagged, in the right direction, and nothing else is."""
    for kind, K, factor, sign in (("put", 485.0, 1.3, "Rich"), ("call", 515.0, 0.7, "Cheap")):
        out = sc.run_scans(skewed_chain(bump=(30, K, kind, factor)), sc.ScanConfig(), "Mispricing")
        got = list(zip(out["type"], out["strike"]))
        check(got == [(kind, K)], f"planted {sign} {kind} {K}: scan returned {got}")
        if len(out):
            check(out["signal"].iloc[0].startswith(sign), f"planted {kind} {K} labelled {out['signal'].iloc[0]}")


def rule(df: pd.DataFrame, cfg: sc.ScanConfig, mode: str) -> pd.Series:
    """Independent restatement of the documented scan rules."""
    liq = (df["volume"] >= cfg.min_volume) & (df["oi"] >= cfg.min_oi)
    uoa = (df["volume"] >= cfg.min_volume) & (df["vol_oi"] >= cfg.uoa_ratio)
    mis = (df["delta"].abs().between(0.2, 0.8) & (df["mid"] >= cfg.min_price)
           & (df["spread_pct"] <= cfg.max_spread_pct) & (df["edge_pct"].abs() >= cfg.edge_pct))
    tight = (df["spread_pct"] <= cfg.max_spread_pct) & (df["mid"] >= cfg.min_price)
    buck = (df["mid"] >= cfg.min_price) & ((df["bucket"] == cfg.bucket) if cfg.bucket in sc.BUCKETS else True)
    m = {"UOA + Mispricing": uoa | mis, "Unusual Activity": uoa, "Mispricing": mis,
         "Tight Spreads": tight, "Delta Bucket": buck}[mode]
    return liq & m


def test_scans(chain: pd.DataFrame, trials: int = 300):
    rnd = random.Random(7)
    for i in range(trials):
        cfg = sc.ScanConfig(
            min_volume=rnd.choice([0, 1, 50, 500, 10**9]), min_oi=rnd.choice([0, 10, 1000]),
            uoa_ratio=rnd.choice([0, 0.5, 1.0, 3.0]), edge_pct=rnd.choice([0, 5, 15, 50]),
            max_spread_pct=rnd.choice([0, 2, 10, 100]), min_price=rnd.choice([0, 0.1, 1.0]),
            bucket=rnd.choice(["All"] + sc.BUCKETS))
        mode = rnd.choice(sc.MODES)
        out = sc.run_scans(chain, cfg, mode)
        expected = set(chain.index[rule(chain, cfg, mode)])
        got = set(out.index)
        tag = f"trial {i} {mode} {cfg}"
        check(got == expected, f"{tag}: missing {len(expected - got)}, extra {len(got - expected)}")
        check(out.index.is_unique, f"{tag}: duplicate rows")
        check(out["signal"].astype(str).str.len().gt(0).all() if len(out) else True, f"{tag}: empty signal")
        check(list(out["volume"]) == sorted(out["volume"], reverse=True), f"{tag}: not sorted by volume")


def test_parity_spot():
    """A stale stock quote must not move the prices: the option quotes imply the real spot."""
    true_spot = 100.0
    raw = synthetic_chain(21, true_spot, junk=False)
    for quoted in (true_spot, true_spot * 1.02, true_spot * 0.985):
        ch = sc.prepare_chain(raw, quoted, TODAY)
        est = ch.groupby("expiry")["spot_used"].first()
        check((est.sub(true_spot).abs() / true_spot < 0.003).all(),
              f"quoted {quoted:.2f}: implied spots {est.round(2).tolist()} not near {true_spot}")
    # without call/put pairs, fall back to the quoted price
    calls_only = raw[raw["type"] == "call"]
    ch = sc.prepare_chain(calls_only, 101.0, TODAY)
    check((ch["spot_used"] == 101.0).all(), "no parity pairs should fall back to the quoted spot")


def test_edge_cases():
    empty = sc.prepare_chain(synthetic_chain(1).iloc[0:0], 100.0, TODAY)
    for mode in sc.MODES:
        out = sc.run_scans(empty, sc.ScanConfig(), mode)
        check(len(out) == 0, f"empty chain gave rows in {mode}")
    check(sc.summarize(empty, 100.0) == {"spot": 100.0, "spot_quoted": 100.0}, "summary of empty chain")
    # all contracts expired
    raw = synthetic_chain(2)
    raw["expiry"] = (TODAY - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    check(sc.prepare_chain(raw, 100.0, TODAY).empty, "expired contracts kept")
    # strings/None in numeric columns (yfinance sometimes returns these)
    raw = synthetic_chain(3)
    raw = raw.astype({"bid": object, "volume": object})
    raw.loc[raw.index[:5], "bid"] = None
    raw.loc[raw.index[5:10], "volume"] = "n/a"
    ch = sc.prepare_chain(raw, 100.0, TODAY)
    check(not ch.empty and ch["volume"].notna().all(), "non-numeric volume not cleaned")
    # expiry picking spans the whole DTE window, keeps the nearest, never duplicates
    for n in (1, 5, 12, 13, 30, 250):
        exp = list(range(n))
        got = sc.pick_expiries(exp, 12)
        check(len(got) == min(n, 12) and got == sorted(set(got)), f"pick_expiries count/order n={n}: {got}")
        check(got[0] == 0 and got[-1] == n - 1, f"pick_expiries must keep nearest and farthest n={n}")
    # market_today is a plain (tz-naive) New York date
    t = sc.market_today()
    check(t.tzinfo is None and t == t.normalize(), "market_today not a naive date")


def test_summary(chain: pd.DataFrame, spot: float):
    closes = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(5).normal(0, 0.01, 60))))
    s = sc.summarize(chain, spot, closes)
    check(s["dte"] >= 7 or s["dte"] == chain["dte"].max(), "summary expiry rule")
    check(abs(s["exp_move"] - s["spot"] * s["atm_iv"] * math.sqrt(max(s["dte"], 0.5) / 365)) < 1e-9, "expected move formula")
    check(s["skew_25d"] > 0, "synthetic chain has put skew but skew_25d <= 0")
    check(abs(s["iv_rv"] - s["atm_iv"] / s["rv20"]) < 1e-12, "IV/RV ratio")
    check(abs(s["skew_25d"] - (s["p25_iv"] - s["c25_iv"]) * 100) < 1e-9, "skew = put IV - call IV")


def test_tkinter_tab(chain: pd.DataFrame):
    """Drive the desktop tab without showing it: refilter, sort, bad input."""
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception as exc:  # no display available
        print("  (skipped Tkinter test:", exc, ")")
        return
    root.withdraw()
    import tempfile
    from unittest.mock import patch
    import scanner_tab
    from tkinter import filedialog, messagebox
    said = []  # every dialog is stubbed so a headless run can never block on one
    stubs = [patch.object(messagebox, name, lambda *a, n=name, **k: said.append(n))
             for name in ("showwarning", "showinfo", "showerror")]
    for st in stubs:
        st.start()
    try:
        tab = scanner_tab.ScannerTab(root)
        tab.v["ticker"].set("TEST")
        deliver = lambda t, ch, win=(0, 45): (tab._queue.put(("ok", t, 100.0, ch, sc.summarize(ch, 100.0), win)), tab._poll())
        deliver("TEST", chain)
        check(str(tab.export_btn["state"]) == ("normal" if len(tab._result) else "disabled"), "Tk export state after scan")
        for mode in sc.MODES:
            tab.v["mode"].set(mode)
            tab.refilter_btn.invoke()
            n = len(tab.tree.get_children())
            check(n == min(len(tab._result), tab.MAX_ROWS), f"Tk rows shown {n} vs result {len(tab._result)} ({mode})")
        for col in ("volume", "edge_pct", "signal", "theta"):
            tab._sort_by(col)
        # filter-only change re-filters the cache
        tab.v["mode"].set("Delta Bucket")
        tab.v["min_vol"].set("0")
        tab.refilter_btn.invoke()
        before = len(tab._result)
        check(before > 0 and tab._ticker == "TEST", "Tk filter-only change should re-filter the cached chain")
        # ticker or DTE edits must not be applied to the old chain (TK-REFILTER-WINDOW)
        for key, val in (("ticker", "AAPL"), ("max_dte", "7"), ("min_dte", "1")):
            old = tab.v[key].get()
            tab.v[key].set(val)
            tab.refilter_btn.invoke()
            check(len(tab.tree.get_children()) == 0 and str(tab.export_btn["state"]) == "disabled"
                  and "Press Fetch & Scan" in tab.status.get() and tab.summary.get() == "",
                  f"Tk Re-apply after changing {key} must clear the view and ask for a fresh fetch")
            tab.v[key].set(old)
            tab.refilter_btn.invoke()
            check(len(tab._result) == before, f"Tk reverting {key} should restore the cached results")
        # nonempty -> empty scan: no stale rows, export disabled, button cannot save old rows (TK-EMPTY-EXPORT)
        deliver("EMPTY", chain.iloc[0:0])
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(filedialog, "asksaveasfilename", return_value=tmp + "/x.csv") as dialog:
            tab.export_btn.invoke()
            tab._export()  # even a direct call must refuse
            check(dialog.call_count == 0 and not os.path.exists(tmp + "/x.csv"),
                  "Tk empty scan must not export the previous ticker's rows")
        check(len(tab.tree.get_children()) == 0 and str(tab.export_btn["state"]) == "disabled"
              and tab._ticker == "EMPTY" and tab._result.empty, "Tk empty scan state")
        # failure after a success clears the old rows too
        deliver("TEST", chain)
        tab._queue.put(("err", "No price data found for 'MISSING'."))
        tab._poll()
        check(tab._result.empty and str(tab.export_btn["state"]) == "disabled" and tab._ticker is None,
              "Tk failed scan must clear the previous results and disable export")
        # every invalid field is rejected on its own, before any fetch (TK-VALIDATION)
        good = {"ticker": "SPY", "min_dte": "0", "max_dte": "45", "min_vol": "50", "min_oi": "10",
                "uoa": "1.0", "edge": "5", "spread": "10", "min_price": "0.10"}
        bad = [("ticker", "SPY;X"), ("ticker", "^ABCDEFGHIJKL"), ("ticker", " "), ("min_dte", "-2"),
               ("max_dte", "999"), ("min_dte", "50"), ("min_vol", "-1"), ("min_vol", "1.5"), ("min_oi", "-1"),
               ("uoa", "nan"), ("uoa", "abc"), ("edge", "inf"), ("spread", "1e309"), ("min_price", "-0.1")]
        with patch.object(scanner_tab, "fetch_chain") as fetch:
            for key, val in bad:
                for k, v in good.items():
                    tab.v[k].set(v)
                tab.v[key].set(val)
                n_warn = said.count("showwarning")
                tab.scan_btn.invoke()
                check(said.count("showwarning") == n_warn + 1 and not tab._busy,
                      f"Tk accepted {key}={val!r}")
            check(fetch.call_count == 0, f"Tk invalid input reached the provider: {fetch.call_args_list}")
        for k, v in good.items():
            tab.v[k].set(v)
        tab.v["ticker"].set("  spy ")
        with patch.object(scanner_tab, "fetch_chain", return_value=(100.0, chain, sc.summarize(chain, 100.0))) as fetch:
            tab.scan_btn.invoke()
            for _ in range(200):
                if not tab._queue.empty():
                    break
                time.sleep(0.01)
            tab._poll()
            check(fetch.call_args[0] == ("SPY", 0, 45) and tab._ticker == "SPY", "Tk valid scan normalises the ticker")
    finally:
        for st in stubs:
            st.stop()
        root.destroy()


def raw_row(k=100.0, kind="call", dte=30, bid=3.95, ask=4.05, last=4.0, iv=0.3, vol=100, oi=50):
    return dict(strike=k, bid=bid, ask=ask, lastPrice=last, volume=vol, openInterest=oi,
                impliedVolatility=iv, type=kind, expiry=(TODAY + pd.Timedelta(days=dte)).strftime("%Y-%m-%d"))


def fair_row(k, kind, dte=30, spot=100.0, iv=0.25, factor=1.0, oi=50):
    p = float(sc.bs_model(spot, k, max(dte, 0.5) / 365, sc.RISK_FREE_RATE, sc.DIVIDEND_YIELD, iv, kind == "call")[0]) * factor
    return raw_row(k, kind, dte, p * 0.995, p * 1.005, p, iv, oi=oi)


def test_quote_rule():
    """DATA-CROSSED-QUOTE: crossed quotes never count as live: no spread, no Tight Spreads hit,
    no IV from them, no effect on the parity spot. Locked quotes (bid == ask) are live."""
    cases = {"crossed": (4.20, 3.80), "locked": (4.00, 4.00), "missing": (0.0, 0.0), "nan": (np.nan, 4.1), "valid": (3.95, 4.05)}
    for name, (b, a) in cases.items():
        ch = sc.prepare_chain(pd.DataFrame([raw_row(bid=b, ask=a)]), 100.0, TODAY)
        r = ch.iloc[0]
        want = {"crossed": "crossed", "locked": "live", "valid": "live"}.get(name, "none")
        check(r.quote == want, f"{name} quote classified {r.quote}, expected {want}")
        tight = sc.run_scans(ch, sc.ScanConfig(), "Tight Spreads")
        check((len(tight) == 1) == (want == "live"), f"{name} quote: Tight Spreads returned {len(tight)} rows")
        if want != "live":
            check(math.isnan(r.spread_pct) and r.iv_source not in ("mid", "pair") and r.mid == 4.0,
                  f"{name} quote must fall back to the last trade without a spread or a quote-solved IV")
        else:
            check(r.spread_pct >= 0, f"{name} spread {r.spread_pct}")
    fair = [fair_row(k, kind) for k in (96, 98, 100, 102, 104) for kind in ("call", "put")]
    base = sc.prepare_chain(pd.DataFrame(fair), 100.0, TODAY)["spot_used"].iloc[0]
    bent = sc.prepare_chain(pd.DataFrame(fair + [raw_row(k=100.5, bid=9, ask=7), raw_row(k=100.5, kind="put", bid=.99, ask=1.01)]),
                            100.0, TODAY)["spot_used"].iloc[0]
    check(abs(bent - base) < 1e-9, f"a crossed pair moved the implied spot from {base:.4f} to {bent:.4f}")
    lone = sc.prepare_chain(pd.DataFrame([raw_row(bid=9, ask=7), raw_row(kind="put", bid=.99, ask=1.01)]), 100.0, TODAY)
    check((lone["spot_used"] == 100.0).all(), "crossed-only pairs must fall back to the quoted spot")


def test_impossible_mid():
    """DATA-IMPOSSIBLE-MID: a live quote no volatility can produce is 'rejected' and never Rich/Cheap,
    while ordinary Rich and Cheap contracts in the same chain are still found."""
    strikes = range(100, 124, 2)
    # above any IV: an OTM $105 call at $150; below the zero-volatility value: an ITM $90 call at
    # $9, under S - K*e^(-rT) = 10.37 (any positive price is reachable for an OTM option)
    for k_bad, quote in ((105, (149.5, 150.5)), (90, (8.95, 9.05))):
        rows = [fair_row(k, "call") for k in strikes]
        rows.append(raw_row(k=k_bad, bid=quote[0], ask=quote[1], last=sum(quote) / 2, iv=0.30))
        ch = sc.prepare_chain(pd.DataFrame(rows), 100.0, TODAY)
        bad = ch[ch["strike"] == k_bad].iloc[0]
        check(bad.iv_source == "rejected" and math.isnan(bad.edge_pct) and bad.edge_note == "price outside model bounds",
              f"impossible mid at {k_bad}: source={bad.iv_source} edge={bad.edge_pct} note={bad.edge_note!r}")
        out = sc.run_scans(ch, sc.ScanConfig(min_price=0), "Mispricing")
        check(k_bad not in set(out["strike"]), f"impossible mid at {k_bad} produced a signal")
    for factor, sign in ((1.3, "Rich"), (0.7, "Cheap")):  # controls: real mispricing still found
        rows = [fair_row(k, "call", factor=factor if k == 104 else 1) for k in strikes]
        out = sc.run_scans(sc.prepare_chain(pd.DataFrame(rows), 100.0, TODAY), sc.ScanConfig(), "Mispricing")
        check(list(out["strike"]) == [104] and out["signal"].iloc[0].startswith(sign),
              f"control {sign}: got {list(zip(out['strike'], out['signal']))}")


def test_smile_contamination():
    """CORE-SMILE-CONTAMINATION: an overpriced contract must not drag the smile and make a fair
    neighbour look Cheap. With thin support the verdict is withheld; with dense support the
    overpriced contract itself is flagged Rich, near its true -23% edge."""
    strikes = (95, 97, 99, 101, 103, 105)
    thin = lambda bump: sc.prepare_chain(pd.DataFrame(
        [fair_row(k, "put" if k < 100 else "call", factor=bump if k == 105 else 1) for k in strikes]), 100.0, TODAY)
    check(sc.run_scans(thin(1.0), sc.ScanConfig(), "Mispricing").empty, "clean thin chain produced signals")
    ch = thin(1.3)
    out = sc.run_scans(ch, sc.ScanConfig(), "Mispricing")
    check(out.empty, f"thin chain: overpriced 105 call produced {list(zip(out['strike'], out['signal']))}")
    check(ch.loc[ch["strike"] == 103, "edge_pct"].isna().all() or abs(ch.loc[ch["strike"] == 103, "edge_pct"].iloc[0]) < 1,
          "fair 103 call must not be judged against a curve bent by the 105 call")
    dense = sc.prepare_chain(pd.DataFrame(
        [fair_row(k, "put" if k < 100 else "call", factor=1.3 if k == 105 else 1) for k in range(85, 117)]), 100.0, TODAY)
    out = sc.run_scans(dense, sc.ScanConfig(), "Mispricing")
    check(list(out["strike"]) == [105] and out["signal"].iloc[0].startswith("Rich"),
          f"dense chain: expected only 105 Rich, got {list(zip(out['strike'], out['signal']))}")
    if len(out):
        check(abs(out["edge_pct"].iloc[0] - (1 / 1.3 - 1) * 100) < 2, f"105 edge {out['edge_pct'].iloc[0]:.2f}% vs true -23.08%")


def test_fetch_partial():
    """DATA-ABSENT-EXPIRY: missing legs or expiries keep the usable data and report coverage."""
    from types import SimpleNamespace
    from unittest.mock import patch
    e1, e2 = (TODAY + pd.Timedelta(days=7)).strftime("%Y-%m-%d"), (TODAY + pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    good = lambda kind: pd.DataFrame([fair_row(k, kind, dte=30) for k in (96, 98, 100, 102, 104)]).drop(columns=["type", "expiry"])

    def run(chains):
        class T:
            fast_info = {"last_price": 100.0}
            options = [e1, e2]
            def history(self, period):
                return pd.DataFrame({"Close": [100.0] * 25})
            def option_chain(self, exp):
                c = chains[exp]
                if isinstance(c, Exception):
                    raise c
                return c
        with patch.dict(sys.modules, {"yfinance": SimpleNamespace(Ticker=lambda _: T())}), \
                patch.object(sc, "market_today", return_value=TODAY):
            return sc.fetch_chain("PART", 0, 45)

    _, ch, s = run({e1: SimpleNamespace(calls=None, puts=None), e2: SimpleNamespace(calls=good("call"), puts=good("put"))})
    check(len(ch) == 10 and s["expiries_scanned"] == 1 and s["expiries_missing"] == [e1], f"missing first expiry: {s}")
    _, ch, s = run({e1: RuntimeError("timeout"), e2: SimpleNamespace(calls=good("call"), puts=pd.DataFrame())})
    check(set(ch["type"]) == {"call"} and s["expiries_scanned"] == 1, "missing single leg and failing expiry")
    try:
        run({e1: SimpleNamespace(calls=None, puts=None), e2: None})
        check(False, "all-missing window must raise a no-data error")
    except ValueError as exc:
        check("no option data" in str(exc), f"all-missing message: {exc}")


def test_inputs_and_caps():
    """API-SYMBOL-LENGTH, TK-VALIDATION (shared rules) and CORE-EXPIRY-CAP."""
    for t, ok in (("spy", True), (" ^spx ", True), ("ABCDEFGHIJKL", True), ("^ABCDEFGHIJK", True),
                  ("ABCDEFGHIJKLM", False), ("^ABCDEFGHIJKL", False), ("SPY;X", False), ("", False), ("^", False)):
        try:
            sc.normalize_ticker(t)
            got = True
        except ValueError:
            got = False
        check(got == ok, f"ticker {t!r}: accepted={got}")
    base = dict(min_dte=0, max_dte=45, cfg=sc.ScanConfig())
    sc.validate_settings(**base)
    sc.validate_settings(0, 0, sc.ScanConfig(min_volume=0, min_oi=0, uoa_ratio=0, edge_pct=0, max_spread_pct=0, min_price=0))
    bad = [dict(min_dte=-1), dict(max_dte=731), dict(min_dte=10, max_dte=5), dict(min_dte=1.5)]
    for field in ("min_volume", "min_oi", "uoa_ratio", "edge_pct", "max_spread_pct", "min_price"):
        bad += [{field: float("inf")}, {field: float("nan")}, {field: -1}]
    for b in bad:
        args = dict(base)
        cfg_fields = {k: v for k, v in b.items() if k not in ("min_dte", "max_dte")}
        args.update({k: v for k, v in b.items() if k in ("min_dte", "max_dte")})
        args["cfg"] = sc.ScanConfig(**{**sc.ScanConfig().__dict__, **cfg_fields})
        try:
            sc.validate_settings(**args)
            check(False, f"validate_settings accepted {b}")
        except ValueError:
            pass
    check(sc.pick_expiries([], 12) == [] and sc.pick_expiries([0, 1, 2, 3], 1) == [0]
          and sc.pick_expiries([0, 1, 2, 3], 2) == [0, 3], "pick_expiries small caps")
    got = sc.pick_expiries(list(range(40)), 12)
    check(len(got) == 12 and got[0] == 0 and got[-1] == 39, f"default cap coverage {got}")
    try:
        sc.pick_expiries([0, 1], 0)
        check(False, "cap 0 must raise")
    except ValueError:
        pass


if __name__ == "__main__":
    test_pricing()
    test_quote_rule()
    test_impossible_mid()
    test_smile_contamination()
    test_fetch_partial()
    test_inputs_and_caps()
    test_implied_vol()
    test_tkinter_tab(sc.prepare_chain(synthetic_chain(9), 100.0, TODAY))
    for seed in range(5):
        spot = [100.0, 722.29, 15.5, 2500.0, 48.0][seed]
        raw = synthetic_chain(seed, spot)
        ch = sc.prepare_chain(raw, spot, TODAY)
        test_prepare(ch, spot)
        test_scans(ch, 120)
        test_summary(ch, spot)
    test_edge_cases()
    test_skew_is_not_mispricing()
    test_skew_is_not_mispricing(stale=1.013)  # Yahoo's previous close vs live options, as seen on SPY
    test_skew_is_not_mispricing(stale=0.99)
    test_finds_real_mispricing()
    test_parity_spot()
    if FAILS:
        print(f"{len(FAILS)} FAILED")
        for f in FAILS[:25]:
            print(" -", f)
        sys.exit(1)
    print("All logic tests passed.")
