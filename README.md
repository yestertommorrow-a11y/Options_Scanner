# Options Scanner

Scan a US stock or ETF option chain for unusual activity, rich or cheap pricing, tight spreads and delta groups, with every number explained step by step and illustrated with interactive charts. Runs as a web page (FastAPI, deploys to Vercel) or as a Tkinter desktop tab.

> Educational tool, not financial advice. Data comes from Yahoo Finance via `yfinance`: quotes are delayed and open interest is end-of-day.

## Why use it

- **Learn by seeing the working.** Click any result and the page shows how its mid price, delta, theta, vega, model price and edge were calculated, with your numbers substituted into each formula.
- **Understand the concepts, not just the numbers.** Charts show expected move, volatility skew, time decay and the odds of finishing in the money, each with a plain-English caption built from the live figures.
- **Two audiences, one tool.** Beginner view explains every setting and column; Pro view is a dense trader layout with extra columns (Last, Gamma, Ref IV).
- **Free data, no API key.** Works out of the box with Yahoo Finance; the data layer is isolated so a paid feed can be swapped in.
- **Tested logic.** The scan rules are checked against hundreds of randomised simulated chains, and bad inputs are rejected with clear messages.

## Features

### Scans
| Scan type | What it finds |
|---|---|
| Unusual Activity (UOA) | Contracts where today's volume is large relative to open interest (fresh positioning) |
| Mispricing | Contracts priced rich or cheap versus a smooth volatility smile fitted through the expiry's quotes, read at the contract's own strike; out-of-the-money options only |
| UOA + Mispricing | Either of the above, with both signals merged per contract |
| Tight Spreads | Liquid contracts that are cheap to trade |
| Delta Bucket | Lotto (< 0.15), Sensible (0.15–0.35), Directional (0.35–0.65), Deep ITM (> 0.65) |

Adjustable filters: DTE window, minimum volume, minimum open interest, Vol/OI threshold, edge %, maximum spread % and minimum price.

### Market read
- ATM implied volatility, 1-sigma expected move, 25-delta skew, 20-day realised volatility and IV/RV ratio, each with its calculation shown.
- Charts: probability curve of the price at expiry with the expected-move band, implied vs realised volatility, volatility smile by strike, and volume vs open interest by strike.

### Per-contract working
- Step-by-step calculation of mid, spread, intrinsic and time value, Vol/OI, d1/d2, delta, theta, vega, reference IV, model price and edge, plus a ✓/✗ list of why the contract was listed.
- Charts: profit/loss at expiry vs today, time-decay curve, chance of finishing in the money, and option price vs implied volatility.

### Interface
- Sortable results table, CSV export of every row, and a field guide linking each setting to implied volatility and time decay.
- Beginner / Pro view toggle, Sunny / Dark theme toggle, full-screen view for any chart, and hover readouts on every chart.
- Warns when the market looks closed (few live quotes), since Yahoo then reports placeholder IVs.
- Works on phones; charts are dependency-free SVG with a colour-blind-safe palette.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt uvicorn   # Windows
.venv\Scripts\python -m uvicorn main:app --port 8000
```

Open http://127.0.0.1:8000, enter a ticker (e.g. SPY) and press **Fetch & Scan**. Scan during US market hours (9:30am–4pm New York time) for meaningful IVs.

Desktop version: `python scanner_tab.py`, or embed it in an existing Tkinter notebook:

```python
from scanner_tab import ScannerTab
notebook.add(ScannerTab(notebook, get_ticker=lambda: ticker_var.get()), text="Scanner")
```

## Deploy to Vercel

Import the repository in Vercel; the FastAPI preset is detected from `requirements.txt` and `main.py`. No build command is needed. Optional environment variables:

| Name | Default | Meaning |
|---|---|---|
| `RISK_FREE_RATE` | `0.045` | Annual risk-free rate used in Black-Scholes |
| `DIVIDEND_YIELD` | `0.0` | Annual dividend yield |

Note: Yahoo sometimes rate-limits requests from cloud servers, so test the deployed scan early.

## Project layout

| File | Role |
|---|---|
| `scanner_core.py` | Pricing, data fetch, chain preparation, scan rules (no UI) |
| `main.py` | FastAPI app: `/api/scan` and static file serving for local runs |
| `public/index.html` | Web page: settings, table, working panel, field guide |
| `public/charts.js` | SVG charts with hover tooltips and full-screen view |
| `scanner_tab.py` | Tkinter desktop tab using the same core |
| `tests/test_scanner.py` | Offline logic tests on simulated chains |
| `tests/simulate_api.py` | Good and bad form entries against a running server |

## Tests

```bash
.venv\Scripts\python -B tests\test_scanner.py   # offline: maths, scan rules, regression cases, Tkinter tab
.venv\Scripts\python -B tests\test_api.py       # offline: API statuses, field errors, exact rows (Yahoo stubbed)
.venv\Scripts\python tests\simulate_api.py      # live smoke test; needs the server running; exits 1 on any mismatch
```

## Limitations

- Yahoo data is delayed; open interest updates once a day, so Vol/OI is a rough signal.
- "Rich" and "Cheap" mean out of line with the fitted smile at that strike, i.e. versus neighbouring options, not versus a fair value. A quadratic smile can misfit far wings, so treat wing signals with care.
- Prices use the underlying price implied by the option quotes (put-call parity), because Yahoo's stock quote can be stale, e.g. the previous close outside US hours. The page warns when the two differ.
- IV is solved from each strike's out-of-the-money mid price and shared with the in-the-money option at that strike (US options are American, so in-the-money prices carry early-exercise value). Without a live quote, Yahoo's IV is used.
- A quote is "live" only if bid and ask are positive and ask ≥ bid. Crossed or missing quotes fall back to the last trade, never pass Tight Spreads and never move the implied price or the smile.
- Rich/Cheap needs a live out-of-the-money quote whose IV was solved from that quote, and a smile with support at the strike: each contract is judged against a fit without itself, after outliers are screened out, and only inside the range of strikes that fit covers. Otherwise the verdict is withheld and the worked example says why.
- Up to 12 expiries per scan, spread across the chosen DTE window.
