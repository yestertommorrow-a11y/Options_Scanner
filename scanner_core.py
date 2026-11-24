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
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

RISK_FREE_RATE = float(os.environ.get("RISK_FREE_RATE", "0.045"))
DIVIDEND_YIELD = float(os.environ.get("DIVIDEND_YIELD", "0.0"))

BUCKETS = ["Lotto", "Sensible", "Directional", "Deep ITM"]
MAX_DTE = 730
# One symbol rule for every client: after trimming and upper-casing, 1-12 characters in total
# (a leading ^ counts), letters/digits first, then letters, digits, '.', '-' or '='.
TICKER_RE = re.compile(r"(?=.{1,12}$)\^?[A-Z0-9][A-Z0-9.\-=]*")
TICKER_HELP = ("Please enter a ticker symbol such as SPY, AAPL, BRK-B or ^SPX "
               "(up to 12 characters: letters, digits, '.', '-', '^' or '=').")
MODES = [
    "UOA + Mispricing",
    "Unusual Activity",
    "Mispricing",
    "Tight Spreads",
    "Delta Bucket",
]


def normalize_ticker(raw) -> str:
    """Trim and upper-case a symbol; raise ValueError(TICKER_HELP) if it is not a valid symbol.

    Yahoo silently ignores junk such as "SPY;X" and returns SPY, which would put SPY data
    under the wrong name, so anything outside the symbol alphabet is rejected up front.
    """
    t = str(raw or "").strip().upper()
    if not TICKER_RE.fullmatch(t):
        raise ValueError(TICKER_HELP)
    return t


def validate_settings(min_dte, max_dte, cfg: "ScanConfig") -> None:
    """Field-by-field checks shared by the web API and the desktop tab. Raises ValueError."""
    def num(label, v, integer=False):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"{label}: enter a finite number.")
        if integer and v != int(v):
            raise ValueError(f"{label}: enter a whole number.")
        if v < 0:
            raise ValueError(f"{label}: must be at least 0.")
    num("Min DTE", min_dte, True)
    num("Max DTE", max_dte, True)
    if max_dte > MAX_DTE:
        raise ValueError(f"Max DTE: must be at most {MAX_DTE}.")
    if min_dte > max_dte:
        raise ValueError("Min DTE must not exceed Max DTE.")
    num("Min Volume", cfg.min_volume, True)
    num("Min Open Interest", cfg.min_oi, True)
    num("UOA Vol/OI", cfg.uoa_ratio)
    num("Edge %", cfg.edge_pct)
    num("Max Spread %", cfg.max_spread_pct)
    num("Min Mid Price", cfg.min_price)
    if cfg.bucket != "All" and cfg.bucket not in BUCKETS:
        raise ValueError(f"Unknown delta bucket '{cfg.bucket}'.")


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
    if max_expiries < 1:
        raise ValueError("max_expiries must be at least 1.")
    if len(expiries) <= max_expiries:
        return list(expiries)
    if max_expiries == 1:
        return [expiries[0]]  # one slot: the nearest, most active expiry
    # Space by calendar date, not list position: near-term dates are daily, later ones
    # weekly or monthly, so spacing by position would bunch the picks near the start.
    # With 2+ slots the farthest expiry is always included, so the whole window is covered.
    days = np.array([pd.Timestamp(e).value for e in expiries], dtype=float) if isinstance(expiries[0], str) \
        else np.asarray(expiries, dtype=float)
    head = min(2, max_expiries - 1)
    keep = set(range(head))
    slots = max_expiries - head
    targets = [days[-1]] if slots == 1 else np.linspace(days[head], days[-1], slots)
    for target in targets:
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

    # Yahoo sometimes returns nothing (or None) for one expiry or one side of it. Keep what
    # did arrive and report the real coverage instead of failing the whole scan.
    frames, got, missing = [], [], []
    for exp in expiries:
        try:
            oc = t.option_chain(exp)
        except Exception:
            oc = None
        legs = 0
        for kind in ("call", "put"):
            frame = getattr(oc, kind + "s", None) if oc is not None else None
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                f = frame.copy()
                f["type"] = kind
                f["expiry"] = exp
                frames.append(f)
                legs += 1
        (got if legs else missing).append(exp)
    if not frames:
        raise ValueError(f"Yahoo returned no option data for '{ticker}' between {min_dte} and {max_dte} DTE.")
    raw = pd.concat(frames, ignore_index=True)

    chain = prepare_chain(raw, spot, today)
    summary = summarize(chain, spot, hist["Close"])
    summary["expiries_scanned"], summary["expiries_in_window"] = len(got), len(in_window)
    summary["expiries_missing"] = missing
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
    live = df[df["quote"] == "live"]  # crossed or missing quotes never move the spot
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


SMILE_MIN_STRIKES = 6


def smile_points(e: pd.DataFrame) -> pd.DataFrame:
    """Rows of one expiry that may shape its smile: out-of-the-money, live quote, IV solved
    from that quote, |delta| 0.05-0.95 and a spread of at most 25%."""
    return e[e["otm"] & (e["quote"] == "live") & (e["iv_source"] == "mid")
             & e["delta"].abs().between(0.05, 0.95) & (e["spread_pct"] <= 25)]


OUTLIER_FLOOR = 0.005  # 0.5 vol point: residuals smaller than this are never called outliers


def _fit(k, iv, w, strikes, min_strikes=SMILE_MIN_STRIKES):
    """Vega-weighted quadratic IV(k). Returns (coef, k_min, k_max) of the points used, or None
    if fewer than min_strikes distinct strikes support it (the verdict is then withheld)."""
    if len(k) == 0 or len(np.unique(strikes)) < min_strikes:
        return None
    coef = np.polyfit(k, iv, 2, w=w)
    return coef, float(k.min()), float(k.max())


def smile_inliers(k, iv, w, strikes, max_out=3):
    """Outlier screen, run once per expiry before any reference is built.

    Drop each point in turn and keep the drop that leaves the cleanest fit (smallest weighted
    RMS of the rest). If that point sits far off the clean curve (more than 4x the remaining
    scatter and more than 0.5 vol point), it is an outlier: remove it from every fit and repeat,
    up to max_out times while enough strikes remain. With few points a plain residual test fails,
    because every fit that contains the outlier is bent by it.
    """
    n = len(k)
    keep = np.ones(n, dtype=bool)
    for _ in range(max_out):
        idx = np.flatnonzero(keep)
        if len(np.unique(strikes[idx])) <= SMILE_MIN_STRIKES:
            break  # dropping one more would leave too little support to judge anything
        best = None
        for j in idx:
            m = keep.copy()
            m[j] = False
            res = _fit(k[m], iv[m], w[m], strikes[m], min_strikes=3)
            if res is None:
                continue
            fitted = np.polyval(res[0], k[m])
            rms = math.sqrt(np.average((iv[m] - fitted) ** 2, weights=w[m] ** 2))
            r_j = abs(iv[j] - np.polyval(res[0], k[j]))
            if best is None or rms < best[0]:
                best = (rms, j, r_j)
        if best is None:
            break
        rms, j, r_j = best
        if r_j > max(4 * rms, OUTLIER_FLOOR):
            keep[j] = False
        else:
            break
    return keep


def fit_smile(e: pd.DataFrame, exclude=None):
    """Quadratic smile IV(k), k = ln(K/S), through one expiry's out-of-the-money live IVs.

    Outliers found by smile_inliers() are removed first. exclude: an index label to leave out,
    so a contract never helps build its own reference (an overpriced option would otherwise drag
    the curve toward itself and push its fairly priced neighbours off it).
    Returns (coef, k_min, k_max) or None without enough support. prepare_chain uses the same steps.
    """
    pts = smile_points(e)
    if pts.empty:
        return None
    w = np.sqrt(np.maximum(pts["vega"].to_numpy(), 1e-8))
    pts = pts[smile_inliers(pts["moneyness"].to_numpy(), pts["iv"].to_numpy(), w, pts["strike"].to_numpy())]
    if exclude is not None:
        pts = pts.drop(index=exclude, errors="ignore")
    if pts.empty:
        return None
    return _fit(pts["moneyness"].to_numpy(), pts["iv"].to_numpy(),
                np.sqrt(np.maximum(pts["vega"].to_numpy(), 1e-8)), pts["strike"].to_numpy())


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

    # One quote rule for everything quote-driven (spread, parity spot, IV, smile, Rich/Cheap):
    # "live" = finite bid and ask, both > 0, ask >= bid (a locked quote, ask == bid, is fine).
    # A crossed quote (bid > ask) is not a market; it falls back to the last trade like a
    # missing quote, so it can never pass Tight Spreads or move the implied spot.
    bid, ask = df["bid"].to_numpy(dtype=float), df["ask"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        priced = np.isfinite(bid) & np.isfinite(ask) & (bid > 0) & (ask > 0)
        two_sided = priced & (ask >= bid)
        crossed = priced & (ask < bid)
    df["quote"] = np.select([two_sided, crossed], ["live", "crossed"], default="none")
    df["mid"] = np.where(two_sided, (bid + ask) / 2.0, df["last"])
    with np.errstate(all="ignore"):
        df["spread_pct"] = np.where(two_sided, (ask - bid) / df["mid"] * 100.0, np.nan)
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
    # Provenance: mid = solved from this quote; pair = from the OTM option at this strike;
    # rejected = live quote whose mid no volatility can produce (e.g. $150 for a $5 option);
    # crossed = bid above ask; yahoo = no usable quote. Only "mid" can be judged Rich/Cheap.
    src = np.select([otm & np.isfinite(own), ~otm & np.isfinite(pair), np.isfinite(own),
                     two_sided, crossed],
                    ["mid", "pair", "mid", "rejected", "crossed"], default="yahoo")
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

    # Reference IV = the expiry's smile, fitted through its out-of-the-money live IVs and read at
    # this contract's own strike. Comparing with one at-the-money level instead flags every far
    # call "Cheap" and every far put "Rich", because IV slopes along the curve (skew).
    # A contract that helps shape the smile is judged against a fit WITHOUT itself, and only
    # inside the strike range the remaining points cover (no extrapolation to the wings).
    k = np.log(df["strike"].to_numpy() / S)
    df["moneyness"] = k
    ref = np.full(len(df), np.nan)
    supported = np.zeros(len(df), dtype=bool)
    method = np.full(len(df), "", dtype=object)
    pos = {label: i for i, label in enumerate(df.index)}
    for _, idx in df.groupby("expiry").groups.items():
        e = df.loc[idx]
        pts = smile_points(e)
        pk, piv = pts["moneyness"].to_numpy(), pts["iv"].to_numpy()
        pw, pstrike = np.sqrt(np.maximum(pts["vega"].to_numpy(), 1e-8)), pts["strike"].to_numpy()
        clean = smile_inliers(pk, piv, pw, pstrike)  # outliers leave every fit, not just their own
        pts = pts[clean]
        pk, piv, pw, pstrike = pk[clean], piv[clean], pw[clean], pstrike[clean]
        full = _fit(pk, piv, pw, pstrike)
        in_fit = {label: j for j, label in enumerate(pts.index)}
        for label in idx:
            if label in in_fit:
                m = np.ones(len(pts), dtype=bool)
                m[in_fit[label]] = False
                res = _fit(pk[m], piv[m], pw[m], pstrike[m])
            else:
                res = full
            if res is None:
                continue
            coef, lo, hi = res
            i = pos[label]
            ref[i] = float(np.clip(np.polyval(coef, k[i]), 0.01, 5.0))
            method[i] = "smile"
            supported[i] = lo <= k[i] <= hi
    df["ref_iv"], df["ref_method"], df["ref_supported"] = ref, method, supported
    # Without a smile, show a near-the-money median for the worked example only; it is never
    # used for a Rich/Cheap verdict.
    core = df[df["delta"].abs().between(0.3, 0.7) & df["otm"] & (df["iv_source"] == "mid")]
    med = core.groupby(["expiry", "type"])["iv"].median()
    fallback = pd.MultiIndex.from_arrays([df["expiry"], df["type"]]).map(med.get).astype(float)
    need = df["ref_iv"].isna().to_numpy() & np.isfinite(fallback)
    df.loc[need, "ref_iv"] = fallback[need]
    df.loc[need, "ref_method"] = "median"
    theo, _, _ = bs_model(S, df["strike"], T, r, q, df["ref_iv"], is_call)
    df["theo"] = theo
    # Rich/Cheap only for out-of-the-money options with a live quote, an IV solved from that
    # quote, and a supported smile reference. edge_note says why any other row has no edge.
    eligible = (df["otm"] & (df["iv_source"] == "mid") & (df["ref_method"] == "smile")
                & df["ref_supported"] & (df["mid"] > 0)).to_numpy()
    with np.errstate(all="ignore"):
        df["edge_pct"] = np.where(eligible, (df["theo"] - df["mid"]) / df["mid"] * 100.0, np.nan)
    src = df["iv_source"].to_numpy()
    df["edge_note"] = np.select(
        [eligible, src == "crossed", src == "rejected", ~df["otm"].to_numpy(), src == "yahoo"],
        ["", "crossed quote", "price outside model bounds", "in the money", "no live quote"],
        default="smile lacks support at this strike")

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
