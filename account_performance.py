#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import sys

from benchmark_core import DEFAULT_BENCHMARKS, BenchmarkResult, simulate_benchmark
from performance_core import (
    PerformanceError,
    analyze_performance,
    download_market_data,
    load_account_history,
    money,
    pct,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate Fidelity account performance from transaction history."
    )
    parser.add_argument("history_csv", help="Merged Fidelity account history CSV.")
    parser.add_argument("account_number", help="Fidelity account number to analyze.")
    parser.add_argument(
        "period",
        nargs="?",
        help="Period shortcut: 1Y, 6M, 3M, YTD, or a calendar year such as 2025.",
    )
    parser.add_argument("--start", help="Custom start date, YYYY-MM-DD. Requires --end.")
    parser.add_argument("--end", help="Custom end date, YYYY-MM-DD. Requires --start.")
    parser.add_argument("--holdings-csv", help="Optional Fidelity holdings CSV for quantity validation.")
    parser.add_argument(
        "--expected-cash",
        type=float,
        help="Optional expected cash/core total to compare with calculated free cash plus SPAXX.",
    )
    parser.add_argument(
        "--expected-value",
        type=float,
        help="Optional expected total account value to compare with calculated end value.",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Show holdings, cash, external-flow, split-adjustment, and benchmark dividend details.",
    )
    parser.add_argument(
        "--refresh-prices",
        action="store_true",
        help="Ignore cached Yahoo prices and download fresh data.",
    )
    parser.add_argument(
        "--benchmark",
        action="append",
        metavar="SYMBOL",
        help="Benchmark ticker to compare against, repeatable. Defaults to QQQM.",
    )
    parser.add_argument(
        "--no-benchmark",
        action="store_true",
        help="Skip the benchmark comparison.",
    )
    return parser.parse_args()


def requested_benchmarks(args: argparse.Namespace) -> list[str]:
    if args.no_benchmark:
        return []
    symbols: list[str] = []
    for symbol in args.benchmark or DEFAULT_BENCHMARKS:
        normalized = str(symbol).strip().upper()
        if normalized and normalized not in symbols:
            symbols.append(normalized)
    return symbols


def signed_money(value: float) -> str:
    return f"{'-' if value < 0 else '+'}${abs(value):,.2f}"


def signed_pct(value: float) -> str:
    if not math.isfinite(value):
        return "N/A"
    return f"{value * 100:+.2f}%"


def print_result(result, details: bool) -> None:
    print(f"Account: {result.account}")
    print(f"Period:  {result.period_label}")
    print(f"Range:   {result.start.date()} to {result.end.date()} ({result.years:.3f} years)")
    print(f"Start value:       {money(result.start_valuation.total_value)}")
    print(f"End value:         {money(result.end_valuation.total_value)}")
    print(f"Net external cash: {money(result.net_external)} (contributions minus withdrawals)")
    print(f"External flows:    {len(result.flows)}")
    print()
    print(f"Period P/L (End - Start - Net external): {money(result.pl)}")
    print(f"Period P/L vs capital (Start + Net ext.): {pct(result.pl_pct)}")
    print()
    print(f"XIRR (annualized money-weighted return): {pct(result.xirr_rate)}")
    print(f"TWR CAGR (date-level approximate):       {pct(result.twr_annualized)}")

    print()
    print("Valuation detail:")
    print(
        f"  Start: securities {money(result.start_valuation.security_value)}, "
        f"SPAXX {money(result.start_valuation.spaxx_value)}, "
        f"free cash {money(result.start_valuation.free_cash)}"
    )
    print(
        f"  End:   securities {money(result.end_valuation.security_value)}, "
        f"SPAXX {money(result.end_valuation.spaxx_value)}, "
        f"free cash {money(result.end_valuation.free_cash)}"
    )

    if result.validations:
        print()
        print("Validation:")
        for item in result.validations:
            status = "OK" if item.ok else "CHECK"
            print(
                f"  {status} {item.label}: calculated {money(item.actual)}, "
                f"expected {money(item.expected)}, diff {money(item.diff)}"
            )

    warnings = list(result.warnings)
    if not all(v.ok for v in result.validations):
        warnings.append("One or more manual validation totals did not match.")
    if warnings:
        print()
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")

    if details:
        print()
        print("Split adjustments:")
        if result.split_adjustments:
            for msg in result.split_adjustments:
                print(f"  - {msg}")
        else:
            print("  None")

        print()
        print("External flows:")
        if result.flows.empty:
            print("  None")
        else:
            for _, row in result.flows.iterrows():
                print(f"  {row['Run Date'].date()} {money(float(row['Amount']))} {row['Action']}")

        print()
        print("Ending positions:")
        for item in result.end_valuation.position_values:
            factor = f", split factor {item.split_factor:g}" if not math.isclose(item.split_factor, 1.0) else ""
            print(
                f"  {item.symbol}: qty {item.quantity:.6f}, priced qty {item.priced_quantity:.6f}, "
                f"price {money(item.price)}, value {money(item.value)}{factor}"
            )


def print_benchmark_comparison(result, benchmarks: list[BenchmarkResult], details: bool) -> None:
    print()
    print("Benchmark comparison (same external cash flows; benchmark dividends reinvested).")
    print("Difference column is benchmark minus portfolio.")
    for bench in benchmarks:
        buys = sum(1 for event in bench.events if event.kind == "buy")
        sells = sum(1 for event in bench.events if event.kind == "sell")
        print()
        print(
            f"  {bench.symbol} ({buys} purchases, {sells} sales, "
            f"{bench.dividend_count} dividends reinvested):"
        )
        rows = (
            (
                "P/L %",
                pct(result.pl_pct),
                pct(bench.pl_pct),
                signed_pct(bench.pl_pct - result.pl_pct),
            ),
            (
                "XIRR",
                pct(result.xirr_rate),
                pct(bench.xirr_rate),
                signed_pct(bench.xirr_rate - result.xirr_rate),
            ),
            (
                "TWR CAGR",
                pct(result.twr_annualized),
                pct(bench.twr_annualized),
                signed_pct(bench.twr_annualized - result.twr_annualized),
            ),
            ("P/L", money(result.pl), money(bench.pl), signed_money(bench.pl - result.pl)),
            (
                "End value",
                money(result.end_valuation.total_value),
                money(bench.end_value),
                signed_money(bench.end_value - result.end_valuation.total_value),
            ),
        )
        print(f"    {'Metric':<12}{'Portfolio':>14}{bench.symbol:>14}{'Difference':>14}")
        for label, portfolio_value, benchmark_value, difference in rows:
            print(f"    {label:<12}{portfolio_value:>14}{benchmark_value:>14}{difference:>14}")
        print(
            f"    Dividends reinvested: {money(bench.dividends_reinvested)} "
            f"across {bench.dividend_count} event(s), end shares {bench.shares_end:,.4f}"
        )
        for warning in bench.warnings:
            print(f"    Warning: {warning}")

    if details:
        for bench in benchmarks:
            dividends = [event for event in bench.events if event.kind == "dividend"]
            splits = [event for event in bench.events if event.kind == "split"]
            print()
            print(f"  {bench.symbol} dividend reinvestment:")
            if not dividends:
                print("    None")
            for event in dividends:
                held = event.shares_after - event.shares
                print(
                    f"    {event.date.date()} {event.dividend_per_share:.4f}/share on "
                    f"{held:.6f} shares = {money(event.amount)} reinvested at "
                    f"{money(event.price)} (+{event.shares:.6f} shares)"
                )
            for event in splits:
                print(
                    f"    {event.date.date()} split x{event.shares:g}, "
                    f"shares after {event.shares_after:.6f}"
                )


def main() -> int:
    args = parse_args()
    try:
        df = load_account_history(args.history_csv, args.account_number)
        result = analyze_performance(
            df,
            account=args.account_number,
            period=args.period,
            start_arg=args.start,
            end_arg=args.end,
            holdings_csv=args.holdings_csv,
            expected_cash=args.expected_cash,
            expected_value=args.expected_value,
            refresh_prices=args.refresh_prices,
        )
    except PerformanceError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    benchmarks: list[BenchmarkResult] = []
    failures: list[str] = []
    for symbol in requested_benchmarks(args):
        try:
            market_data = download_market_data(
                [symbol],
                result.start,
                result.end,
                refresh_prices=args.refresh_prices,
            )
            benchmarks.append(
                simulate_benchmark(
                    symbol,
                    result.flows,
                    result.start_valuation.total_value,
                    result.start,
                    result.end,
                    market_data,
                )
            )
        except PerformanceError as exc:
            failures.append(f"{symbol}: {exc}")

    print_result(result, args.details)
    if benchmarks:
        print_benchmark_comparison(result, benchmarks, args.details)
    for failure in failures:
        print(f"ERROR: benchmark {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
