"""
scanner_tab.py - Scanner tab (Tkinter) for the Options Analyser.

Drop-in ttk.Frame for an existing ttk.Notebook:

    from scanner_tab import ScannerTab
    notebook.add(ScannerTab(notebook, get_ticker=lambda: self.ticker_var.get()),
                 text="Scanner")

Standalone test:  python scanner_tab.py

All pricing, data and scan logic lives in scanner_core.py (shared with the web
version in main.py). This file is only the Tkinter UI layer.
"""

from __future__ import annotations

import math
import queue
import threading
import tkinter as tk
from datetime import date
from tkinter import filedialog, messagebox, ttk

import pandas as pd

from scanner_core import (BUCKETS, MODES, ScanConfig, fetch_chain, format_summary, normalize_ticker,
                          run_scans, validate_settings)

# --------------------------------------------------------------------------
# 5. UI
# --------------------------------------------------------------------------
def _f(fmt):
    def inner(v):
        try:
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return ""
            return format(v, fmt)
        except (TypeError, ValueError):
            return str(v)
    return inner


# (tree id, dataframe column, heading, width, formatter)
COLUMNS = [
    ("type", "type", "Type", 55, str),
    ("expiry", "expiry", "Expiry", 85, lambda v: pd.Timestamp(v).strftime("%Y-%m-%d")),
    ("dte", "dte", "DTE", 45, _f("d")),
    ("strike", "strike", "Strike", 70, _f(".2f")),
    ("bid", "bid", "Bid", 60, _f(".2f")),
    ("ask", "ask", "Ask", 60, _f(".2f")),
    ("mid", "mid", "Mid", 60, _f(".2f")),
    ("volume", "volume", "Volume", 70, _f(",.0f")),
    ("oi", "oi", "OI", 70, _f(",.0f")),
    ("vol_oi", "vol_oi", "Vol/OI", 60, _f(".1f")),
    ("iv", "iv", "IV %", 60, lambda v: _f(".1f")(v * 100)),
    ("delta", "delta", "Delta", 60, _f("+.2f")),
    ("theta", "theta", "Theta/day", 70, _f("+.3f")),
    ("vega", "vega", "Vega", 60, _f(".3f")),
    ("theo", "theo", "Theo", 60, _f(".2f")),
    ("edge", "edge_pct", "Edge %", 65, _f("+.1f")),
    ("spread", "spread_pct", "Spread %", 70, _f(".1f")),
    ("bucket", "bucket", "Bucket", 85, str),
    ("signal", "signal", "Signal", 230, str),
]


class ScannerTab(ttk.Frame):
    MAX_ROWS = 500

    def __init__(self, parent, get_ticker=None):
        super().__init__(parent)
        self._get_ticker = get_ticker
        self._queue: queue.Queue = queue.Queue()
        self._busy = False
        self._chain = pd.DataFrame()
        self._result = pd.DataFrame()
        self._ticker = None      # ticker of the chain in self._chain (and of any export)
        self._window = None      # (min_dte, max_dte) that chain was fetched for
        self._sort_col = None
        self._sort_desc = True
        self._build()
        self.after(100, self._poll)

    # ---- layout ---------------------------------------------------------
    def _build(self):
        v = {
            "ticker": tk.StringVar(value=(self._get_ticker() if self._get_ticker else "") or "SPY"),
            "mode": tk.StringVar(value=MODES[0]),
            "bucket": tk.StringVar(value="All"),
            "min_dte": tk.StringVar(value="0"),
            "max_dte": tk.StringVar(value="45"),
            "min_vol": tk.StringVar(value="50"),
            "min_oi": tk.StringVar(value="10"),
            "uoa": tk.StringVar(value="1.0"),
            "edge": tk.StringVar(value="5"),
            "spread": tk.StringVar(value="10"),
            "min_price": tk.StringVar(value="0.10"),
        }
        self.v = v

        top = ttk.LabelFrame(self, text="Scan Settings")
        top.pack(fill="x", padx=8, pady=(8, 4))

        def field(row, col, label, var, width=7, widget="entry", values=None):
            ttk.Label(top, text=label).grid(row=row, column=col * 2, sticky="e", padx=(8, 2), pady=3)
            if widget == "combo":
                w = ttk.Combobox(top, textvariable=var, values=values, width=width, state="readonly")
            else:
                w = ttk.Entry(top, textvariable=var, width=width)
            w.grid(row=row, column=col * 2 + 1, sticky="w", pady=3)
            return w

        field(0, 0, "Ticker", v["ticker"], 8)
        field(0, 1, "Scan Type", v["mode"], 18, "combo", MODES)
        field(0, 2, "Delta Bucket", v["bucket"], 12, "combo", ["All"] + BUCKETS)
        field(0, 3, "Min DTE", v["min_dte"], 5)
        field(0, 4, "Max DTE", v["max_dte"], 5)
        field(1, 0, "Min Volume", v["min_vol"], 8)
        field(1, 1, "Min Open Interest", v["min_oi"], 8)
        field(1, 2, "UOA Vol/OI >=", v["uoa"], 6)
        field(1, 3, "Edge % >=", v["edge"], 6)
        field(1, 4, "Max Spread %", v["spread"], 6)
        field(1, 5, "Min Mid Price", v["min_price"], 6)

        btns = ttk.Frame(self)
        btns.pack(fill="x", padx=8, pady=2)
        self.scan_btn = ttk.Button(btns, text="Fetch & Scan", command=self._start_scan)
        self.scan_btn.pack(side="left")
        self.refilter_btn = ttk.Button(btns, text="Re-apply Filters", command=self._refilter, state="disabled")
        self.refilter_btn.pack(side="left", padx=6)
        self.export_btn = ttk.Button(btns, text="Export CSV", command=self._export, state="disabled")
        self.export_btn.pack(side="left")
        self.status = tk.StringVar(value="Enter a ticker and press Fetch & Scan.")
        ttk.Label(btns, textvariable=self.status).pack(side="left", padx=12)

        self.summary = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.summary, font=("TkDefaultFont", 10, "bold")).pack(
            fill="x", padx=10, pady=(4, 2), anchor="w")

        frame = ttk.Frame(self)
        frame.pack(fill="both", expand=True, padx=8, pady=(2, 8))
        self.tree = ttk.Treeview(frame, columns=[c[0] for c in COLUMNS], show="headings")
        for cid, dfcol, head, width, _ in COLUMNS:
            self.tree.heading(cid, text=head, command=lambda c=dfcol: self._sort_by(c))
            anchor = "w" if cid in ("type", "expiry", "bucket", "signal") else "e"
            self.tree.column(cid, width=width, anchor=anchor, stretch=(cid == "signal"))
        ysb = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        xsb = ttk.Scrollbar(frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        ysb.grid(row=0, column=1, sticky="ns")
        xsb.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

    # ---- settings -------------------------------------------------------
    def _read_config(self) -> tuple[ScanConfig, int, int]:
        """Parse and validate every setting, naming the field that is wrong (ValueError)."""
        v = self.v

        def parse(key, label, kind):
            text = v[key].get().strip()
            try:
                return kind(text)
            except ValueError:
                word = "a whole number" if kind is int else "a number"
                raise ValueError(f"{label}: enter {word}.") from None

        cfg = ScanConfig(
            min_volume=parse("min_vol", "Min Volume", int),
            min_oi=parse("min_oi", "Min Open Interest", int),
            uoa_ratio=parse("uoa", "UOA Vol/OI", float),
            edge_pct=parse("edge", "Edge %", float),
            max_spread_pct=parse("spread", "Max Spread %", float),
            min_price=parse("min_price", "Min Mid Price", float),
            bucket=v["bucket"].get(),
        )
        min_dte, max_dte = parse("min_dte", "Min DTE", int), parse("max_dte", "Max DTE", int)
        validate_settings(min_dte, max_dte, cfg)  # same rules as the web API
        return cfg, min_dte, max_dte

    # ---- actions --------------------------------------------------------
    def _start_scan(self):
        if self._busy:
            return
        try:
            ticker = normalize_ticker(self.v["ticker"].get())
            _, min_dte, max_dte = self._read_config()
        except ValueError as exc:
            messagebox.showwarning("Scanner", str(exc))
            return
        self._busy = True
        self.scan_btn.configure(state="disabled")
        self.status.set(f"Fetching option chains for {ticker}...")
        threading.Thread(target=self._worker, args=(ticker, min_dte, max_dte), daemon=True).start()

    def _worker(self, ticker, min_dte, max_dte):
        try:
            spot, chain, summary = fetch_chain(ticker, min_dte, max_dte)
            self._queue.put(("ok", ticker, spot, chain, summary, (min_dte, max_dte)))
        except Exception as exc:  # surfaced to the UI thread
            self._queue.put(("err", str(exc)))

    def _poll(self):
        try:
            while True:
                msg = self._queue.get_nowait()
                self._busy = False
                self.scan_btn.configure(state="normal")
                if msg[0] == "err":
                    # A failed scan must not leave the previous ticker's rows exportable.
                    self._clear_view(f"Scan failed: {msg[1]}")
                    self._chain, self._ticker, self._window = pd.DataFrame(), None, None
                    self.summary.set("")
                    self.refilter_btn.configure(state="disabled")
                    messagebox.showerror("Scanner", msg[1])
                else:
                    _, ticker, spot, chain, summary, *rest = msg
                    window = rest[0] if rest else self._entry_window()
                    # Replace everything tied to a scan together: chain, ticker, window,
                    # summary, result, sort state and button states.
                    self._chain, self._ticker, self._window = chain, ticker, window
                    self.summary.set(self._format_summary(ticker, summary))
                    self._apply_filters()
        except queue.Empty:
            pass
        if self.winfo_exists():
            self.after(100, self._poll)

    def _entry_window(self):
        try:
            return int(self.v["min_dte"].get()), int(self.v["max_dte"].get())
        except ValueError:
            return None

    def _clear_view(self, status: str):
        self._result = pd.DataFrame()
        self._sort_col = None
        self._render(self._result)
        self.export_btn.configure(state="disabled")
        self.status.set(status)

    def _refilter(self):
        """Re-apply Filters: re-run the scan rules on the cached chain, but only while the ticker
        and DTE window in the entries are still the ones that chain was fetched for."""
        try:
            cfg, min_dte, max_dte = self._read_config()
            ticker = normalize_ticker(self.v["ticker"].get())
        except ValueError as exc:
            messagebox.showwarning("Scanner", str(exc))
            return
        if self._ticker is not None and (ticker, (min_dte, max_dte)) != (self._ticker, self._window):
            # The visible rows no longer match the entries: clear them rather than mislabel them.
            old = f"{self._ticker}, {self._window[0]}-{self._window[1]} DTE" if self._window else self._ticker
            self._clear_view(f"Ticker or DTE window changed since the last fetch ({old}). "
                             f"Press Fetch & Scan to load {ticker}, {min_dte}-{max_dte} DTE.")
            self.summary.set("")
            return
        self._apply_filters(cfg)

    def _apply_filters(self, cfg: ScanConfig | None = None):
        if self._chain.empty:
            self._clear_view("No usable contracts returned.")
            self.refilter_btn.configure(state="disabled")
            return
        if cfg is None:
            try:
                cfg, _, _ = self._read_config()
            except ValueError as exc:
                self._clear_view(str(exc))
                messagebox.showwarning("Scanner", str(exc))
                return
        self._result = run_scans(self._chain, cfg, self.v["mode"].get())
        self._sort_col = None
        self._render(self._result)
        self.refilter_btn.configure(state="normal")
        self.export_btn.configure(state="normal" if len(self._result) else "disabled")
        total, shown = len(self._result), min(len(self._result), self.MAX_ROWS)
        self.status.set(f"{total} signals from {len(self._chain)} contracts (showing {shown}).")

    _format_summary = staticmethod(format_summary)

    # ---- table ----------------------------------------------------------
    def _render(self, df: pd.DataFrame):
        self.tree.delete(*self.tree.get_children())
        for _, row in df.head(self.MAX_ROWS).iterrows():
            self.tree.insert("", "end", values=[fmt(row[dfcol]) for _, dfcol, _, _, fmt in COLUMNS])

    def _sort_by(self, col: str):
        if self._result.empty or col not in self._result.columns:
            return
        self._sort_desc = (not self._sort_desc) if self._sort_col == col else True
        self._sort_col = col
        self._result = self._result.sort_values(col, ascending=not self._sort_desc, na_position="last")
        self._render(self._result)

    def _export(self):
        if self._result.empty or self._ticker is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile=f"{self._ticker}_scanner_{date.today():%Y%m%d}.csv",
        )
        if path:
            self._result.to_csv(path, index=False)
            self.status.set(f"Exported {len(self._result)} rows to {path}")


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Options Analyser - Scanner")
    root.geometry("1250x620")
    nb = ttk.Notebook(root)
    nb.pack(fill="both", expand=True)
    nb.add(ScannerTab(nb), text="Scanner")
    root.mainloop()
