# HANDOFF: Scanner tab for the Options Analyser

Date: 2026-11-24
Files: `scanner_tab.py` (the deliverable), this note.

## Goal

Add a sixth "Scanner" tab to the existing five-tab Tkinter Options Analyser
(Options Chain, IV Analysis, Monte Carlo, Spreads, Scenario Analysis). The tab
scans an option chain for unusual activity, relative mispricing, tight spreads
and delta buckets. Patterns were adapted from public GitHub scanners.

## What exists

`scanner_tab.py` is a self-contained `ttk.Frame` subclass, `ScannerTab`. It was
written without access to the Options Analyser source, so it is not yet wired
into the app and does not use the app's own data layer.

Sections inside the file:

1. Pricing: `bs_model()` is a vectorised Black-Scholes returning price, delta, gamma.
2. Data: `fetch_chain()` pulls chains from yfinance (lazy import). **Replace this with the app's existing data layer.**
3. Preparation: `prepare_chain()` adds mid, spread %, vol/OI, delta, reference IV, theoretical price and edge %. `summarize()` gives ATM IV, 1-sigma expected move, 25-delta skew, 20-day realised vol and IV/RV.
4. Scanners: `scan_uoa`, `scan_mispricing`, `scan_tight_spreads`, `scan_delta_bucket`. `run_scans()` applies the liquidity filters and merges signal text per contract.
5. UI: `ScannerTab` has a threaded fetch (results returned through a queue polled with `after`), a Treeview with sortable headers, "Re-apply Filters" on the cached chain, and CSV export.

Integration:

```python
from scanner_tab import ScannerTab
notebook.add(ScannerTab(notebook, get_ticker=lambda: self.ticker_var.get()), text="Scanner")
```

Standalone: `python scanner_tab.py`. Requires pandas, numpy, yfinance. All UI labels are in English.

## Design choices (mine, not yet confirmed by the owner)

- **Mispricing is relative, not arbitrage.** Reference IV per expiry is the median IV of contracts with |delta| 0.3 to 0.7. Theoretical price is Black-Scholes at that reference IV, and edge % = (theo - mid) / mid. The scan is limited to |delta| 0.2 to 0.8 to reduce skew artefacts, but OTM puts will still often show as "Rich".
- **Delta buckets:** Lotto < 0.15, Sensible 0.15 to 0.35, Directional 0.35 to 0.65, Deep ITM > 0.65.
- **Defaults** (`ScanConfig`): min volume 50, min OI 10, UOA vol/OI >= 1.0, edge >= 15%, max spread 10%, min mid price 0.10.
- **Risk-free rate** 4.5% and dividend yield 0 (`RISK_FREE_RATE`, `DIVIDEND_YIELD`). Placeholders; set to the desk's convention.
- Contracts with IV below 0.01 or above 5 are dropped (yfinance returns junk IVs for dead contracts).
- Default fetch window is 0 to 45 DTE, capped at 8 expiries.
- Table display is capped at 500 rows; the CSV export contains every result row.

## Testing status

- Logic layer: ran on a synthetic Black-Scholes chain with Tkinter stubbed. All five scan modes returned sensible output, and the empty-chain case returned no rows.
- **Not tested:** the Tkinter UI itself (no Tkinter in the sandbox) and live yfinance data. Run `python scanner_tab.py` on a real machine first.

## Known caveats

- Yahoo data is delayed, and OI is end-of-day, so vol/OI is a rough UOA signal.
- The mispricing signal depends on the reference-IV choice; treat it as "rich or cheap versus the ATM vol level".
- Delta uses each contract's own IV; theoretical price uses the reference IV.

## Source repos reviewed

- `audreyarchibald/option_scanner`: read. Source of the scanner-per-module structure, the `DataProvider` pattern, the delta buckets and the mispricing idea. Only 2 commits and 0 stars, so a pattern source, not a dependency.
- `jwolberg/options-scanner`: read. React dashboard using a paid API. Source of the skew, expected-move and gamma-flip ideas.
- `suli99/options-scanner`, `SandiRidwan/option-scanner`, `artemiusgreat/Yahoo-Options-Scanner`: **not read** (fetch rate-limited or skipped).

## Suggested next steps

1. Read the real Options Analyser code, match its naming, layering and styling, and replace `fetch_chain()` with its data layer.
2. Add the tab to the notebook and run it on a few tickers (liquid ETF and single stock) to sanity-check units, IV scaling and DTE handling.
3. Add GEX per strike as an estimate: gamma x OI x 100 x spot, calls positive and puts negative. Label it an estimate (dealer-positioning assumption, end-of-day OI).
4. Optionally add a `DataProvider` interface so yfinance can be swapped for a paid feed.
5. Tune thresholds against what the desk actually finds useful.

## Web version (decided 2026-11-24: Tkinter tab and a web version)

- `scanner_core.py`: layers 1-4 (pricing, data, preparation, scanners), no tkinter. Both UIs import it.
- `scanner_tab.py`: now only the Tkinter UI.
- `main.py`: FastAPI app for Vercel (FastAPI preset, auto-detected). `GET /api/scan` takes the same settings as the tab. Fetched chains are cached for 120 s per warm instance.
- `public/index.html`: static page (settings, sortable table, CSV export of every row, 500-row display cap), Beginner/Pro view toggle, per-row step-by-step working, field guide.
- `public/charts.js`: dependency-free SVG charts with hover tooltips. Market: expected-move curve, IV vs RV, volatility smile, volume vs OI by strike. Per contract: payoff (expiry vs today), time-decay curve, chance of finishing ITM, price vs IV (vega and edge). Colours `--s1`/`--s2` passed the dataviz CVD validator in light and dark.
- Sunny/Dark theme button (saved per browser; defaults to the OS setting) and an Expand button on every chart (Esc or Close to exit).
- Tests: `.venv\Scripts\python tests\test_scanner.py` (offline: pricing maths, chain preparation, 600 random scans checked row-by-row against the documented rules, edge cases, Tkinter tab driven headless). `tests\simulate_api.py` sends ~30 good and bad form entries to a running server using live data.
- Fixed from testing: DTE now uses the New York date (local date was a day ahead in Australia during US hours); expiries are spread across the whole DTE window (was the first 8 only, ~10 days for SPY); junk tickers like `SPY;X` and unknown buckets are rejected (Yahoo silently returned SPY); API validation errors name the field; activity-chart ▲ uses the scanner's per-contract UOA rule; contract charts skip contracts with no usable price.
- Local run: `.venv\Scripts\python -m uvicorn main:app` and open http://127.0.0.1:8000.
- Optional Vercel env vars: `RISK_FREE_RATE`, `DIVIDEND_YIELD`.
- Tested locally with live SPY data. Outside US hours Yahoo returns almost no bids/asks and placeholder IVs; the page now shows a warning when fewer than half the contracts are two-sided. Not yet tested on Vercel: Yahoo may block Vercel's cloud IP addresses.
