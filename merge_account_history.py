#!/usr/bin/env python3
"""
Merge Fidelity account-history exports into a clean, de-duplicated CSV.
"""

from __future__ import annotations

import argparse
import sys

from performance_core import PerformanceError, merge_history_files, write_merged_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge Fidelity account-history CSV exports.")
    parser.add_argument("csv_paths", nargs="+", help="Fidelity account-history CSV exports.")
    parser.add_argument("-o", "--output", help="Output CSV path. Defaults to Accounts_History_<first>_<last>.csv.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = merge_history_files(args.csv_paths, args.output)
        write_merged_history(result)
        print(f"Wrote {result.output_path}")
        print(f"Input rows:       {result.input_rows}")
        print(f"Valid rows:       {result.valid_rows}")
        print(f"Duplicate rows:   {result.duplicate_rows}")
        print(f"Output rows:      {result.output_rows}")
        print(f"Date range:       {result.first_date.date()} to {result.last_date.date()}")
        return 0
    except PerformanceError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
