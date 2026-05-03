#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import sys

from performance_core import (
    PerformanceError,
    analyze_performance,
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
        help="Show holdings, cash, external-flow, and split-adjustment details.",
    )
    return parser.parse_args()


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
        )
        print_result(result, args.details)
        return 0
    except PerformanceError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
