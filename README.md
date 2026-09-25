# Portfolio Performance

Portfolio performance tracker for calculating accurate money-weighted and time-weighted returns from brokerage transaction history.

This is a lightweight command-line tracker that currently supports Fidelity account history CSV exports. It can merge downloaded history files, de-duplicate overlapping exports, rebuild holdings from transactions, value the portfolio with market prices, account for cash/core cash, handle stock splits, and calculate performance metrics like XIRR and time-weighted CAGR.

Future plans include a UI and support for other brokerages, including Robinhood.

## Features

- Merges Fidelity account-history CSV exports and removes duplicate transaction rows.
- Reconstructs holdings and cash balances from transaction history.
- Handles SPAXX/core cash, free cash, external contributions, withdrawals, dividends, reinvestments, and stock splits.
- Calculates XIRR and date-level approximate time-weighted CAGR.
- Compares results against benchmarks (QQQM, VTI, VGT, and a 50/30/20 blend by default) by mirroring external cash flows and reinvesting benchmark dividends, with weighted blends and additional tickers supported.
- Stops with clear errors when unsupported cash movements or corporate actions could make results unreliable.

## How to use

### 1. Export your Fidelity account history

Download your Fidelity account history CSV files (from `Accounts & Trade` > `Portfolio` > `Activity & Orders` > filter `Orders` + `Transfer` + `Dividends/Interest` and a custom time-range). This will download a file `Accounts_History.csv`. For the most accurate results, use complete transaction history from account inception through the end date you want to analyze.

If Fidelity only gives you partial downloads, merge the latest export with your existing historical exports before running performance.

### 2. Install dependencies

```bash
python -m pip install pandas numpy yfinance
```

### 3. Merge multiple Fidelity exports

```bash
python merge_account_history.py Accounts_History_20250110_20260320.csv Accounts_History_20260320-20260502.csv
```

Use a custom merged filename:

```bash
python merge_account_history.py Accounts_History_20250110_20260320.csv Accounts_History_20260320-20260502.csv -o Accounts_History_20250110_20260430.csv
```

The merge command writes a clean CSV, skips Fidelity footer text, removes duplicate transactions, and prints merge stats.

### 4. Run a performance report

Use one of the supported period shortcuts:

```bash
python account_performance.py Accounts_History_20250110_20260430.csv YOUR_ACCOUNT_NUMBER 1Y
python account_performance.py Accounts_History_20250110_20260430.csv YOUR_ACCOUNT_NUMBER 6M
python account_performance.py Accounts_History_20250110_20260430.csv YOUR_ACCOUNT_NUMBER 3M
python account_performance.py Accounts_History_20250110_20260430.csv YOUR_ACCOUNT_NUMBER YTD
python account_performance.py Accounts_History_20250110_20260430.csv YOUR_ACCOUNT_NUMBER 2025
```

Or use an exact custom date range:

```bash
python account_performance.py Accounts_History_20250110_20260430.csv YOUR_ACCOUNT_NUMBER --start 2025-01-01 --end 2025-12-31
```

### 5. Show detailed diagnostics

```bash
python account_performance.py Accounts_History_20250110_20260430.csv YOUR_ACCOUNT_NUMBER 1Y --details
```

This shows ending positions, external flows, cash/core cash, and split adjustments.

### 6. Validate against Fidelity totals

If you know the cash/core balance or total account value from Fidelity, pass those values as checks:

```bash
python account_performance.py Accounts_History_20250110_20260430.csv YOUR_ACCOUNT_NUMBER 1Y --expected-cash 502.41 --expected-value 40107.84
```

You can also validate calculated quantities against a Fidelity holdings CSV:

```bash
python account_performance.py Accounts_History_20250110_20260430.csv YOUR_ACCOUNT_NUMBER 1Y --holdings-csv Holdings.csv
```

### 7. Compare against a benchmark

Every report compares against QQQM, VTI, VGT, and a 50% QQQM + 30% VTI + 20% VGT blend by default. Each comparison replays the same external cash flows (start-of-period value plus every contribution and withdrawal) into the benchmark, reinvests benchmark dividends within the same holding, lets the blend drift without rebalancing, and reports the same P/L %, XIRR, and TWR CAGR next to the portfolio's.

Pass `--benchmark` and/or `--blend` to replace the default set:

```bash
python account_performance.py Accounts_History_20250110_20260430.csv YOUR_ACCOUNT_NUMBER 1Y --benchmark QQQM --benchmark SPY
python account_performance.py Accounts_History_20250110_20260430.csv YOUR_ACCOUNT_NUMBER 1Y --blend QQQM:60,VTI:40
```

Skip all comparisons:

```bash
python account_performance.py Accounts_History_20250110_20260430.csv YOUR_ACCOUNT_NUMBER 1Y --no-benchmark
```

Use `--details` to list each reinvested dividend with its ticker, ex-date, per-share amount, and shares added.

### 8. Run tests

```bash
python -m unittest -q
```

## Notes

- Keep brokerage exports private. This repo ignores `Account*` files and all `*.csv` files by default.
- The performance script assumes the transaction CSV is complete from account inception through the end date.
- TWR is date-level approximate because Fidelity CSV exports do not provide intraday flow timing.
- Benchmark comparisons mirror external cash flows at the next trading day's close, reinvest dividends at the ex-dividend close (Yahoo does not expose payable dates), and assume fractional shares with no fees or taxes.
- Blended allocations split each cash flow by the target weights and then drift; they are not rebalanced.
- The benchmark starts with the portfolio's start-of-period value, so comparisons stay apples-to-apples for periods like YTD or 1Y.
