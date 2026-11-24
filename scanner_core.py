"""
scanner_core.py - UI-free logic shared by the Tkinter tab and the web API.

Layers:
    1. Pricing      - bs_model()                    (Black-Scholes price/delta/gamma)
    2. Data         - fetch_chain()                 (yfinance -> prepared chain + summary)
    3. Preparation  - prepare_chain(), summarize()  (mid, spread, delta, ref IV, edge)
    4. Scanners     - scan_uoa / scan_mispricing / scan_tight_spreads / scan_delta_bucket

No tkinter import here, so this module runs on servers (e.g. Vercel) too.
Requires: pandas, numpy, yfinance (yfinance imported lazily, only when fetching).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

RISK_FREE_RATE = float(os.environ.get("RISK_FREE_RATE", "0.045"))
DIVIDEND_YIELD = float(os.environ.get("DIVIDEND_YIELD", "0.0"))

BUCKETS = ["Lotto", "Sensible", "Directional", "Deep ITM"]
MODES = [
    "UOA + Mispricing",
    "Unusual Activity",
    "Mispricing",
    "Tight Spreads",
    "Delta Bucket",
]


# --------------------------------------------------------------------------
# 1. Pricing
# --------------------------------------------------------------------------
_erf = np.vectorize(math.erf, otypes=[float])


def _ncdf(x):
    return 0.5 * (1.0 + _erf(np.asarray(x, dtype=float) / math.sqrt(2.0)))


def _npdf(x):
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_model(S, K, T, r, q, sigma, is_call):
    """Vectorised Black-Scholes. Returns (price, delta, gamma)."""
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    is_call = np.asarray(is_call, dtype=bool)
    with np.errstate(all="ignore"):
        sqrt_t = np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t
        disc_r, disc_q = np.exp(-r * T), np.exp(-q * T)
        call = S * disc_q * _ncdf(d1) - K * disc_r * _ncdf(d2)
        put = K * disc_r * _ncdf(-d2) - S * disc_q * _ncdf(-d1)
        price = np.where(is_call, call, put)
        delta = np.where(is_call, disc_q * _ncdf(d1), disc_q * (_ncdf(d1) - 1.0))
        gamma = disc_q * _npdf(d1) / (S * sigma * sqrt_t)
    return price, delta, gamma


def bs_theta_vega(S, K, T, r, q, sigma, is_call):
    """Theta per calendar day and vega per 1 vol point (both in option-price units)."""
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    is_call = np.asarray(is_call, dtype=bool)
    with np.errstate(all="ignore"):
        sqrt_t = np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t
        disc_r, disc_q = np.exp(-r * T), np.exp(-q * T)
        decay = -S * disc_q * _npdf(d1) * sigma / (2.0 * sqrt_t)
        call = decay - r * K * disc_r * _ncdf(d2) + q * S * disc_q * _ncdf(d1)
        put = decay + r * K * disc_r * _ncdf(-d2) - q * S * disc_q * _ncdf(-d1)
        theta = np.where(is_call, call, put) / 365.0
        vega = S * disc_q * _npdf(d1) * sqrt_t / 100.0
    return theta, vega


# --------------------------------------------------------------------------
# 2. Data
# --------------------------------------------------------------------------
def market_today() -> pd.Timestamp:
    """Today's date in New York, where US options trade and expire.

    Using the computer's local date is wrong outside the US: during the US session
    it is already tomorrow in Australia, which would shift every DTE by one day and
    drop options expiring today.
    """
    return pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)


def pick_expiries(expiries: list, max_expiries: int) -> list:
    """Choose up to max_expiries dates spread across the whole window.

    Tickers like SPY list an expiry every trading day; taking only the first N would
    silently stop the scan a week or two out even when Max DTE is 45. Keep the two
    nearest (most active) and spread the rest evenly to the far end of the window.
    """
    if len(expiries) <= max_expiries:
        return list(expiries)
    # Space by calendar date, not list position: near-term dates are daily, later ones
    # weekly or monthly, so spacing by position would bunch the picks near the start.
    days = np.array([pd.Timestamp(e).value for e in expiries], dtype=float) if isinstance(expiries[0], str) \
        else np.asarray(expiries, dtype=float)
    head = 2
    keep = set(range(head))
    for target in np.linspace(days[head], days[-1], max_expiries - head):
        free = [i for i in range(head, len(days)) if i not in keep]
        keep.add(min(free, key=lambda i: abs(days[i] - target)))
    return [expiries[i] for i in sorted(keep)]


def fetch_chain(ticker: str, min_dte: int, max_dte: int, max_expiries: int = 12):
    """Fetch option chains from yfinance. Returns (spot, chain, summary)."""
    import yfinance as yf  # lazy import so the logic layer is testable offline

    t = yf.Ticker(ticker)
    hist = t.history(period="3mo")
    if hist.empty:
        raise ValueError(f"No price data found for '{ticker}'.")
    spot = float(hist["Close"].iloc[-1])
    try:
        live = t.fast_info["last_price"]
        if live and live > 0:
            spot = float(live)
    except Exception:
        pass

    today = market_today()
    in_window = [
        e for e in t.options
        if min_dte <= (pd.Timestamp(e) - today).days <= max_dte
    ]
    if not in_window:
        raise ValueError(f"No expiries for '{ticker}' between {min_dte} and {max_dte} DTE.")
    expiries = pick_expiries(in_window, max_expiries)

    frames = []
    for exp in expiries:
        oc = t.option_chain(exp)
        for kind, frame in (("call", oc.calls), ("put", oc.puts)):
            f = frame.copy()
            f["type"] = kind
            f["expiry"] = exp
            frames.append(f)
    raw = pd.concat(frames, ignore_index=True)

    chain = prepare_chain(raw, spot, today)
    summary = summarize(chain, spot, hist["Close"])
    summary["expiries_scanned"], summary["expiries_in_window"] = len(expiries), len(in_window)
    return spot, chain, summary


# --------------------------------------------------------------------------
# 3. Preparation
# --------------------------------------------------------------------------
def prepare_chain(raw: pd.DataFrame, spot: float, today=None,
                  r: float = RISK_FREE_RATE, q: float = DIVIDEND_YIELD) -> pd.DataFrame:
    """Normalise a yfinance-style chain and add mid, spread, delta, reference IV, edge."""
    today = (market_today() if today is None else pd.Timestamp(today)).normalize()
    df = raw.copy()
    df["expiry"] = pd.to_datetime(df["expiry"])
    for c in ("strike", "bid", "ask", "lastPrice", "volume", "openInterest", "impliedVolatility"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[["volume", "openInterest"]] = df[["volume", "openInterest"]].fillna(0)
    df = df.rename(columns={"openInterest": "oi", "impliedVolatility": "iv", "lastPrice": "last"})

    df["dte"] = (df["expiry"] - today).dt.days
    # yfinance returns near-zero IVs for dead contracts; drop them.
    df = df[(df["dte"] >= 0) & (df["iv"] > 0.01) & (df["iv"] < 5) & (df["strike"] > 0)].copy()
    if df.empty:
        return df.reset_index(drop=True)

    two_sided = (df["bid"] > 0) & (df["ask"] > 0)
    df["mid"] = np.where(two_sided, (df["bid"] + df["ask"]) / 2.0, df["last"])
    with np.errstate(all="ignore"):
        df["spread_pct"] = np.where(two_sided, (df["ask"] - df["bid"]) / df["mid"] * 100.0, np.nan)
        df["vol_oi"] = np.where(df["oi"] > 0, df["volume"] / df["oi"], np.nan)

    T = np.maximum(df["dte"].to_numpy(dtype=float), 0.5) / 365.0
    is_call = (df["type"] == "call").to_numpy()
    _, delta, gamma = bs_model(spot, df["strike"], T, r, q, df["iv"], is_call)
    df["delta"], df["gamma"] = delta, gamma
    df["theta"], df["vega"] = bs_theta_vega(spot, df["strike"], T, r, q, df["iv"], is_call)

    # Reference IV per expiry: median IV of near-the-money contracts (|delta| 0.3-0.7).
    core = df[df["delta"].abs().between(0.3, 0.7)]
    df["ref_iv"] = df["expiry"].map(core.groupby("expiry")["iv"].median())
    theo, _, _ = bs_model(spot, df["strike"], T, r, q, df["ref_iv"], is_call)
    df["theo"] = theo
    with np.errstate(all="ignore"):
        df["edge_pct"] = np.where(df["mid"] > 0, (df["theo"] - df["mid"]) / df["mid"] * 100.0, np.nan)

    ad = df["delta"].abs()
    df["bucket"] = np.select(
        [ad < 0.15, ad < 0.35, ad < 0.65],
        ["Lotto", "Sensible", "Directional"],
        default="Deep ITM",
    )
    return df.reset_index(drop=True)


def summarize(chain: pd.DataFrame, spot: float, closes: pd.Series | None = None) -> dict:
    """Ticker-level read: ATM IV, 1-sigma expected move, 25-delta skew, IV vs realised vol."""
    out: dict = {"spot": spot}
    if chain.empty:
        return out
    dtes = sorted(chain["dte"].unique())
    pick = next((d for d in dtes if d >= 7), dtes[-1])
    e = chain[chain["dte"] == pick]
    out["dte"] = int(pick)

    atm_strike = e.loc[(e["strike"] - spot).abs().idxmin(), "strike"]
    atm_iv = float(e.loc[e["strike"] == atm_strike, "iv"].mean())
    out["atm_strike"] = float(atm_strike)
    out["atm_iv"] = atm_iv
    out["exp_move"] = spot * atm_iv * math.sqrt(max(pick, 0.5) / 365.0)

    puts, calls = e[e["type"] == "put"], e[e["type"] == "call"]
    if not puts.empty and not calls.empty:
        p25 = puts.iloc[(puts["delta"] + 0.25).abs().to_numpy().argmin()]
        c25 = calls.iloc[(calls["delta"] - 0.25).abs().to_numpy().argmin()]
        out["skew_25d"] = float(p25["iv"] - c25["iv"]) * 100.0  # vol points
        out.update(p25_strike=float(p25["strike"]), p25_iv=float(p25["iv"]), p25_delta=float(p25["delta"]),
                   c25_strike=float(c25["strike"]), c25_iv=float(c25["iv"]), c25_delta=float(c25["delta"]))

    if closes is not None and len(closes) > 21:
        rv = float(np.log(closes).diff().tail(20).std() * math.sqrt(252))
        out["rv20"] = rv
        if rv > 0:
            out["iv_rv"] = atm_iv / rv
    return out


def format_summary(ticker: str, s: dict) -> str:
    parts = [f"{ticker} {s.get('spot', 0):.2f}"]
    if "atm_iv" in s:
        parts.append(f"ATM IV {s['atm_iv'] * 100:.1f}% ({s['dte']}d)")
        parts.append(f"1-sigma move +/-{s['exp_move']:.2f}")
    if "skew_25d" in s:
        parts.append(f"25d skew {s['skew_25d']:+.1f} vol pts")
    if "rv20" in s:
        parts.append(f"20d RV {s['rv20'] * 100:.1f}%")
    if "iv_rv" in s:
        parts.append(f"IV/RV {s['iv_rv']:.2f}")
    return "   |   ".join(parts)


# --------------------------------------------------------------------------
# 4. Scanners  (each returns a Series of signal text indexed like the chain)
# --------------------------------------------------------------------------
@dataclass
class ScanConfig:
    min_volume: int = 50
    min_oi: int = 10
    uoa_ratio: float = 1.0       # volume / open interest
    edge_pct: float = 15.0       # |theo - mid| / mid
    max_spread_pct: float = 10.0
    min_price: float = 0.10
    bucket: str = "All"


def scan_uoa(df: pd.DataFrame, cfg: ScanConfig) -> pd.Series:
    m = (df["volume"] >= cfg.min_volume) & (df["vol_oi"] >= cfg.uoa_ratio)
    sub = df[m]
    return sub["vol_oi"].map(lambda v: f"UOA {v:.1f}x")


def scan_mispricing(df: pd.DataFrame, cfg: ScanConfig) -> pd.Series:
    m = (
        df["delta"].abs().between(0.2, 0.8)
        & (df["mid"] >= cfg.min_price)
        & (df["spread_pct"] <= cfg.max_spread_pct)
        & (df["edge_pct"].abs() >= cfg.edge_pct)
    )
    sub = df[m]
    return sub["edge_pct"].map(lambda v: f"Cheap vs ref IV (+{v:.0f}%)" if v > 0 else f"Rich vs ref IV ({v:.0f}%)")


def scan_tight_spreads(df: pd.DataFrame, cfg: ScanConfig) -> pd.Series:
    m = (
        (df["spread_pct"] <= cfg.max_spread_pct)
        & (df["volume"] >= cfg.min_volume)
        & (df["mid"] >= cfg.min_price)
    )
    return df.loc[m, "spread_pct"].map(lambda v: f"Tight spread {v:.1f}%")


def scan_delta_bucket(df: pd.DataFrame, cfg: ScanConfig) -> pd.Series:
    m = df["mid"] >= cfg.min_price
    if cfg.bucket in BUCKETS:
        m &= df["bucket"] == cfg.bucket
    return df.loc[m, "bucket"].map(lambda b: f"Bucket: {b}")


def run_scans(chain: pd.DataFrame, cfg: ScanConfig, mode: str) -> pd.DataFrame:
    """Apply liquidity filters, run the selected scanners, merge signals per contract."""
    if chain.empty:
        return chain.assign(signal=pd.Series(dtype=str))
    df = chain[(chain["volume"] >= cfg.min_volume) & (chain["oi"] >= cfg.min_oi)]

    scanners = {
        "UOA + Mispricing": [scan_uoa, scan_mispricing],
        "Unusual Activity": [scan_uoa],
        "Mispricing": [scan_mispricing],
        "Tight Spreads": [scan_tight_spreads],
        "Delta Bucket": [scan_delta_bucket],
    }[mode]
    parts = [fn(df, cfg) for fn in scanners]
    parts = [p for p in parts if len(p)]
    if not parts:
        return chain.iloc[0:0].assign(signal="")
    signals = pd.concat(parts).groupby(level=0).agg("; ".join)
    out = df.loc[signals.index].copy()
    out["signal"] = signals
    return out.sort_values("volume", ascending=False)
