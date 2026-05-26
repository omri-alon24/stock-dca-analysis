"""LRS backtest extended to 1985 using ^GSPC and ^NDX price indices.

Caveat: ^GSPC and ^NDX are price-only (no dividends). Absolute CAGRs
understated by ~2% (S&P) and ~0.7% (NDX). Relative LRS-vs-B&H comparison
remains valid since the same drag affects all three series in each family.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

START = "1985-01-01"
END   = datetime.today().strftime("%Y-%m-%d")
LEV   = 3.0
ANNUAL_DRAG = 0.0475   # calibrated to real UPRO 2009-2025 (1% expense + ~3.75% financing)
DAILY_FEE = ANNUAL_DRAG / 252
MA_WINDOW = 200
INITIAL = 10_000

def fetch(ticker):
    df = yf.download(ticker, start=START, end=END, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df["Close"].rename(ticker)

print(f"Fetching ^GSPC and ^NDX ({START} -> {END})...")
spx = fetch("^GSPC")
ndx = fetch("^NDX")
print(f"^GSPC: {spx.index[0].date()} -> {spx.index[-1].date()} ({len(spx)} days)")
print(f"^NDX:  {ndx.index[0].date()} -> {ndx.index[-1].date()} ({len(ndx)} days)")

def build(name, underlying):
    df = pd.DataFrame({"px": underlying})
    df["ret"] = df["px"].pct_change()
    df["sma200"] = df["px"].rolling(MA_WINDOW).mean()
    df["above"] = (df["px"] > df["sma200"]).shift(1).fillna(False)
    df["bh_under"] = (1 + df["ret"].fillna(0)).cumprod() * INITIAL
    df["lev_ret"] = LEV * df["ret"] - DAILY_FEE
    df["bh_lev"] = (1 + df["lev_ret"].fillna(0)).cumprod() * INITIAL
    df["lrs_ret"] = np.where(df["above"], df["lev_ret"], 0.0)
    df["lrs"] = (1 + df["lrs_ret"].fillna(0)).cumprod() * INITIAL
    df = df.iloc[MA_WINDOW:].copy()
    for col in ["bh_under", "bh_lev", "lrs"]:
        df[col] = df[col] / df[col].iloc[0] * INITIAL
    return df

spx_df = build("S&P 500", spx)
ndx_df = build("NASDAQ-100", ndx)

def metrics(series, label):
    rets = series.pct_change().dropna()
    n_years = (series.index[-1] - series.index[0]).days / 365.25
    cagr = (series.iloc[-1] / series.iloc[0]) ** (1/n_years) - 1
    vol = rets.std() * np.sqrt(252)
    sharpe = (rets.mean() * 252) / vol if vol > 0 else 0
    cummax = series.cummax()
    dd = (series / cummax - 1).min()
    return dict(label=label, end=series.iloc[-1], cagr=cagr, vol=vol, sharpe=sharpe, mdd=dd)

results = []
for name, df, lev_label in [("S&P 500", spx_df, "UPRO"), ("NASDAQ-100", ndx_df, "TQQQ")]:
    rotations = int(df["above"].astype(int).diff().abs().sum())
    pct_in = df["above"].mean() * 100
    n_years = (df.index[-1] - df.index[0]).days / 365.25
    print(f"\n{name}: {df.index[0].date()} -> {df.index[-1].date()} ({n_years:.1f}y), {rotations} rotations, in-market {pct_in:.1f}% of days")
    results.append(metrics(df["bh_under"], f"B&H {name}"))
    results.append(metrics(df["bh_lev"],   f"B&H 3x {lev_label} (synthetic)"))
    results.append(metrics(df["lrs"],      f"LRS 3x {lev_label} (200d MA)"))

print("\n" + "="*100)
print(f"{'Strategy':<40} {'End $':>16} {'CAGR':>8} {'Vol':>8} {'Sharpe':>8} {'MaxDD':>8}")
print("="*100)
for r in results:
    print(f"{r['label']:<40} ${r['end']:>15,.0f} {r['cagr']*100:>7.2f}% {r['vol']*100:>7.1f}% {r['sharpe']:>8.2f} {r['mdd']*100:>7.1f}%")

# Plots
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

ax = axes[0,0]
ax.plot(spx_df.index, spx_df["bh_under"], label="B&H S&P 500", color="#888")
ax.plot(spx_df.index, spx_df["bh_lev"],   label="B&H 3x UPRO (synth)", color="#d44")
ax.plot(spx_df.index, spx_df["lrs"],      label="LRS 3x UPRO (200d MA)", color="#27a")
ax.set_yscale("log")
ax.set_title(f"S&P 500 family: Growth of $10,000 (log)\n{spx_df.index[0]:%Y-%m-%d} → {spx_df.index[-1]:%Y-%m-%d}")
ax.legend(); ax.grid(alpha=0.3)

ax = axes[0,1]
ax.plot(ndx_df.index, ndx_df["bh_under"], label="B&H NASDAQ-100", color="#888")
ax.plot(ndx_df.index, ndx_df["bh_lev"],   label="B&H 3x TQQQ (synth)", color="#d44")
ax.plot(ndx_df.index, ndx_df["lrs"],      label="LRS 3x TQQQ (200d MA)", color="#27a")
ax.set_yscale("log")
ax.set_title(f"NASDAQ-100 family: Growth of $10,000 (log)\n{ndx_df.index[0]:%Y-%m-%d} → {ndx_df.index[-1]:%Y-%m-%d}")
ax.legend(); ax.grid(alpha=0.3)

ax = axes[1,0]
for col, lbl, c in [("bh_under","B&H S&P 500","#888"),("bh_lev","B&H 3x UPRO","#d44"),("lrs","LRS 3x UPRO","#27a")]:
    s = spx_df[col]
    dd = s/s.cummax() - 1
    ax.plot(spx_df.index, dd*100, label=lbl, color=c)
ax.set_title("S&P 500 family: Drawdowns (%)")
ax.legend(); ax.grid(alpha=0.3)

ax = axes[1,1]
for col, lbl, c in [("bh_under","B&H NASDAQ-100","#888"),("bh_lev","B&H 3x TQQQ","#d44"),("lrs","LRS 3x TQQQ","#27a")]:
    s = ndx_df[col]
    dd = s/s.cummax() - 1
    ax.plot(ndx_df.index, dd*100, label=lbl, color=c)
ax.set_title("NASDAQ-100 family: Drawdowns (%)")
ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("/tmp/lrs_backtest_long.png", dpi=110)
print(f"\nChart saved to /tmp/lrs_backtest_long.png")

pd.DataFrame(results).to_csv("/tmp/lrs_results_long.csv", index=False)
print("Results saved to /tmp/lrs_results_long.csv")
