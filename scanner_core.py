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


def implied_vol(price, S, K, T, r, q, is_call, lo=1e-4, hi=5.0, iters=48):
    """Vectorised Black-Scholes implied volatility by bisection.

    Returns NaN where the price is outside what any volatility in [lo, hi] can produce
    (for example below intrinsic value).
    """
    price = np.asarray(price, dtype=float)
    K = np.broadcast_to(np.asarray(K, dtype=float), price.shape)
    T = np.broadcast_to(np.asarray(T, dtype=float), price.shape)
    is_call = np.broadcast_to(np.asarray(is_call, dtype=bool), price.shape)
    a, b = np.full(price.shape, lo), np.full(price.shape, hi)
    p_lo = bs_model(S, K, T, r, q, a, is_call)[0]
    p_hi = bs_model(S, K, T, r, q, b, is_call)[0]
    ok = np.isfinite(price) & (price > p_lo) & (price < p_hi)
    for _ in range(iters):
        m = (a + b) / 2.0
        below = bs_model(S, K, T, r, q, m, is_call)[0] < price
        a, b = np.where(below, m, a), np.where(below, b, m)
    return np.where(ok, (a + b) / 2.0, np.nan)


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
def parity_spot(df: pd.DataFrame, spot: float, r: float, q: float) -> pd.Series:
    """Underlying price implied by each expiry's option quotes (put-call parity).

    C - P = S*e^(-qT) - K*e^(-rT)  =>  S = (C - P + K*e^(-rT)) * e^(qT), using the 5 strikes
    where call and put mids are closest (nearest the money). Yahoo's stock quote can be stale
    (outside US hours it is the previous close while option quotes are not), and a spot that is
    off by 1% skews every IV: puts look expensive and calls cheap. Falls back to the quoted
    spot when an expiry has no two-sided call/put pair or the estimate is more than 10% away.
    """
    out = pd.Series(float(spot), index=df.index)
    live = df[(df["bid"] > 0) & (df["ask"] > 0)]
    for exp, g in live.groupby("expiry"):
        T = max(float(g["dte"].iloc[0]), 0.5) / 365.0
        c = g[g["type"] == "call"].groupby("strike")["mid"].mean()
        p = g[g["type"] == "put"].groupby("strike")["mid"].mean()
        both = c.index.intersection(p.index)
        if len(both) < 1:
            continue
        diff = c[both] - p[both]
        near = diff.abs().nsmallest(5).index
        est = float(np.median((diff[near] + near.to_numpy() * math.exp(-r * T)) * math.exp(q * T)))
        if 0.9 * spot < est < 1.1 * spot:
            out[df["expiry"] == exp] = est
    return out


def fit_smile(e: pd.DataFrame, min_points: int = 6):
    """Quadratic smile IV(k), k = ln(K/S), through one expiry's out-of-the-money live IVs.

    Weighted by vega so liquid near-the-money points count most; one refit drops points more
    than 3 robust standard deviations off the curve, so a single bad quote cannot bend it.
    Returns polynomial coefficients for np.polyval, or None if there are too few points.
    """
    pts = e[e["otm"] & (e["iv_source"] == "mid") & e["delta"].abs().between(0.05, 0.95)
            & (e["spread_pct"] <= 25)]
    if len(pts) < min_points:
        return None
    k, iv = pts["moneyness"].to_numpy(), pts["iv"].to_numpy()
    w = np.sqrt(np.maximum(pts["vega"].to_numpy(), 1e-8))
    coef = np.polyfit(k, iv, 2, w=w)
    resid = iv - np.polyval(coef, k)
    mad = np.median(np.abs(resid - np.median(resid))) * 1.4826
    keep = np.abs(resid) <= 3 * max(mad, 1e-4)
    if keep.sum() >= min_points and not keep.all():
        coef = np.polyfit(k[keep], iv[keep], 2, w=w[keep])
    return coef


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
    df = df[(df["dte"] >= 0) & (df["strike"] > 0)].copy()
    if df.empty:
        return df.reset_index(drop=True)

    two_sided = (df["bid"] > 0) & (df["ask"] > 0)
    df["mid"] = np.where(two_sided, (df["bid"] + df["ask"]) / 2.0, df["last"])
    with np.errstate(all="ignore"):
        df["spread_pct"] = np.where(two_sided, (df["ask"] - df["bid"]) / df["mid"] * 100.0, np.nan)
        df["vol_oi"] = np.where(df["oi"] > 0, df["volume"] / df["oi"], np.nan)

    # IV per strike, taken from the OUT-of-the-money option's live mid and shared with the
    # in-the-money option at the same strike (standard practice for an IV surface):
    # - solving from the mid keeps IV, greeks and edge consistent with the price shown
    #   (Yahoo's impliedVolatility comes from other prices and can be points off the mid);
    # - in-the-money US options are American and trade above European value, so IV solved
    #   from their own price is inflated (often 100%+), which would also distort delta.
    T = np.maximum(df["dte"].to_numpy(dtype=float), 0.5) / 365.0
    is_call = (df["type"] == "call").to_numpy()
    df["spot_used"] = parity_spot(df, spot, r, q)
    S = df["spot_used"].to_numpy()
    otm = np.where(is_call, df["strike"] >= S, df["strike"] <= S)
    df["otm"] = otm
    df["iv_yahoo"] = df["iv"]
    own = implied_vol(np.where(two_sided, df["mid"], np.nan), S, df["strike"], T, r, q, is_call)
    pair = (pd.Series(np.where(otm, own, np.nan), index=df.index)
            .groupby([df["expiry"], df["strike"]]).transform("mean").to_numpy())
    src = np.select([otm & np.isfinite(own), ~otm & np.isfinite(pair), np.isfinite(own)],
                    ["mid", "pair", "mid"], default="yahoo")
    df["iv_source"] = src
    df["iv"] = np.select([src == "pair", src == "mid"], [pair, own], default=df["iv_yahoo"])
    # yfinance returns near-zero or huge IVs for dead contracts; drop them.
    keep = ((df["iv"] > 0.01) & (df["iv"] < 5)).to_numpy()
    df, T, is_call, S = df[keep].copy(), T[keep], is_call[keep], S[keep]
    if df.empty:
        return df.reset_index(drop=True)
    _, delta, gamma = bs_model(S, df["strike"], T, r, q, df["iv"], is_call)
    df["delta"], df["gamma"] = delta, gamma
    df["theta"], df["vega"] = bs_theta_vega(S, df["strike"], T, r, q, df["iv"], is_call)

    # Reference IV = the expiry's smile, fitted through its out-of-the-money live IVs and read at this
    # contract's own strike. Comparing with one at-the-money level instead flags every far call
    # "Cheap" and every far put "Rich", because IV slopes along the curve (skew). Without enough
    # points to fit, fall back to the median IV of near-the-money calls (or puts).
    k = np.log(df["strike"].to_numpy() / S)
    df["moneyness"] = k
    df["ref_iv"] = np.nan
    df["ref_method"] = ""
    for exp, idx in df.groupby("expiry").groups.items():
        fit = fit_smile(df.loc[idx])
        if fit is not None:
            df.loc[idx, "ref_iv"] = np.clip(np.polyval(fit, df.loc[idx, "moneyness"]), 0.01, 5.0)
            df.loc[idx, "ref_method"] = "smile"
    core = df[df["delta"].abs().between(0.3, 0.7) & df["otm"] & (df["iv_source"] == "mid")]
    med = core.groupby(["expiry", "type"])["iv"].median()
    fallback = pd.MultiIndex.from_arrays([df["expiry"], df["type"]]).map(med.get).astype(float)
    need = df["ref_iv"].isna().to_numpy() & np.isfinite(fallback)
    df.loc[need, "ref_iv"] = fallback[need]
    df.loc[need, "ref_method"] = "median"
    theo, _, _ = bs_model(S, df["strike"], T, r, q, df["ref_iv"], is_call)
    df["theo"] = theo
    # Rich/Cheap is judged on out-of-the-money options only: in-the-money prices also carry
    # intrinsic and early-exercise value that a European model does not price.
    with np.errstate(all="ignore"):
        df["edge_pct"] = np.where((df["mid"] > 0) & df["otm"], (df["theo"] - df["mid"]) / df["mid"] * 100.0, np.nan)

    ad = df["delta"].abs()
    df["bucket"] = np.select(
        [ad < 0.15, ad < 0.35, ad < 0.65],
        ["Lotto", "Sensible", "Directional"],
        default="Deep ITM",
    )
    return df.reset_index(drop=True)


def summarize(chain: pd.DataFrame, spot: float, closes: pd.Series | None = None) -> dict:
    """Ticker-level read: ATM IV, 1-sigma expected move, 25-delta skew, IV vs realised vol."""
    out: dict = {"spot": spot, "spot_quoted": spot}
    if chain.empty:
        return out
    dtes = sorted(chain["dte"].unique())
    pick = next((d for d in dtes if d >= 7), dtes[-1])
    e = chain[chain["dte"] == pick]
    out["dte"] = int(pick)
    if "spot_used" in e:
        spot = float(e["spot_used"].iloc[0])  # option-implied price for this expiry
        out["spot"] = spot

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
    edge_pct: float = 5.0        # |theo - mid| / mid, versus the fitted smile
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
