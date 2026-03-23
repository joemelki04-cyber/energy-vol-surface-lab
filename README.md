# Energy Vol Surface Lab

A front-office style side project for building implied volatility smiles and ATM term structures for energy options on futures.

## What this project does

This repo takes a simple end-of-day energy option chain and converts it into:

- an implied volatility table
- strike-by-strike smiles
- an ATM term structure
- clean CSV outputs that can later feed a dashboard or HTML report

Version 1 focuses on:

- one valuation date
- one market setup
- European options on futures
- Black-76 pricing
- CSV-based inputs

## Why this matters

On an energy desk, raw option premiums are not enough.

Traders and structurers usually want to know:

- what volatility the market is implying
- whether front-month vol is rich relative to back-month vol
- how skew behaves across strikes
- whether the smile looks stable or distorted

This repo is meant to look like a small but realistic desk analytics tool rather than a generic options notebook.

## Scope

Version 1 includes:

- Black-76 pricing
- implied volatility solving
- moneyness calculations
- nearest-ATM term structure extraction
- CSV outputs in `reports/`

Version 1 does **not** yet include:

- Greeks
- SABR or spline fitting
- interactive charts
- historical realized vs implied comparison
- strategy pricing
- live exchange/API feeds

## Folder structure

```text
energy-vol-surface-lab/
├─ README.md
├─ requirements.txt
├─ data/
│  └─ raw/
│     ├─ futures_prices.csv
│     ├─ option_quotes.csv
│     └─ rates.csv
├─ src/
│  ├─ black76.py
│  ├─ iv_solver.py
│  └─ surface_builder.py
└─ reports/
