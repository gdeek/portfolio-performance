#!/usr/bin/env python3

from __future__ import annotations

import csv
import math
import pathlib
import re
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import yfinance as yf


HISTORY_COLUMNS = [
    "Run Date",
    "Account",
    "Account Number",
    "Action",
    "Symbol",
    "Description",
    "Type",
    "Exchange Quantity",
    "Exchange Currency",
    "Currency",
    "Price",
    "Quantity",
    "Exchange Rate",
    "Commission",
    "Fees",
    "Accrued Interest",
    "Amount",
    "Settlement Date",
]

NUMERIC_COLUMNS = [
    "Exchange Quantity",
    "Price",
    "Quantity",
    "Exchange Rate",
    "Commission",
    "Fees",
    "Accrued Interest",
    "Amount",
]

TRANSACTION_KEY_COLUMNS = [
    "Run Date",
    "Account",
    "Account Number",
    "Action",
    "Symbol",
    "Description",
    "Type",
    "Exchange Quantity",
    "Exchange Currency",
    "Currency",
    "Price",
    "Quantity",
    "Exchange Rate",
    "Commission",
    "Fees",
    "Accrued Interest",
    "Amount",
    "Settlement Date",
]

EXTERNAL_CASH_PATTERNS = (
    "Electronic Funds Transfer Received",
    "Electronic Funds Transfer Paid",
    "Electronic Funds Transfer Sent",
    "DIRECT DEPOSIT",
    "DIRECT DEBIT",
    "CHECK RECEIVED",
    "CHECK PAID",
    "WIRE TRANSFER RECEIVED",
    "WIRE TRANSFER SENT",
)

INTERNAL_CASH_PATTERNS = (
    "YOU BOUGHT",
    "YOU SOLD",
    "REINVESTMENT",
    "DIVIDEND RECEIVED",
    "FOREIGN TAX PAID",
    "INTEREST EARNED",
    "MARGIN INTEREST",
    "FEE",
    "COMMISSION",
    "DISTRIBUTION",
)

CASH_SYMBOLS = {"", "SPAXX"}
SPLIT_TOLERANCE_ABS = 0.01
SPLIT_TOLERANCE_REL = 0.005
VALUE_TOLERANCE_ABS = 1.00
VALUE_TOLERANCE_REL = 0.0005


class PerformanceError(RuntimeError):
    """Raised when the tool cannot produce a trustworthy performance result."""


class UnsupportedTransactionError(PerformanceError):
    """Raised for transaction types that are not safely classified."""


@dataclass
class CleanHistoryResult:
    frame: pd.DataFrame
    path: pathlib.Path
    raw_rows: int
    valid_rows: int


@dataclass
class MergeResult:
    frame: pd.DataFrame
    output_path: pathlib.Path
    input_rows: int
    valid_rows: int
    duplicate_rows: int
    output_rows: int
    first_date: pd.Timestamp
    last_date: pd.Timestamp


@dataclass
class MarketData:
    prices: pd.DataFrame
    splits: pd.DataFrame
    basis_end: pd.Timestamp


@dataclass
class PositionValue:
    symbol: str
    quantity: float
    split_factor: float
    priced_quantity: float
    price: float
    value: float


@dataclass
class Valuation:
    date: pd.Timestamp
    total_value: float
    security_value: float
    free_cash: float
    spaxx_value: float
    positions: pd.Series
    position_values: list[PositionValue] = field(default_factory=list)

    @property
    def cash_core_value(self) -> float:
        return self.free_cash + self.spaxx_value


@dataclass
class ValidationResult:
    label: str
    expected: float
    actual: float
    ok: bool

    @property
    def diff(self) -> float:
        return self.actual - self.expected


@dataclass
class PerformanceResult:
    account: str
    period_label: str
    start: pd.Timestamp
    end: pd.Timestamp
    years: float
    start_valuation: Valuation
    end_valuation: Valuation
    flows: pd.DataFrame
    net_external: float
    pl: float
    pl_pct: float
    xirr_rate: float
    twr_total: float
    twr_annualized: float
    warnings: list[str]
    validations: list[ValidationResult]
    split_adjustments: list[str]


def normalize_date(value) -> pd.Timestamp:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid date: {value}")
    return pd.Timestamp(ts).normalize()


def find_header_row(path: str | pathlib.Path, first_column: str = "Run Date") -> int:
    path = pathlib.Path(path)
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        for idx, line in enumerate(f):
            if line.lstrip("\ufeff").startswith(f"{first_column},"):
                return idx
    raise PerformanceError(f"Could not find Fidelity header row in {path}")


def clean_fidelity_history(path: str | pathlib.Path) -> CleanHistoryResult:
    path = pathlib.Path(path)
    header_row = find_header_row(path, "Run Date")
    df = pd.read_csv(path, skiprows=header_row)
    raw_rows = len(df)

    missing = [col for col in HISTORY_COLUMNS if col not in df.columns]
    if missing:
        raise PerformanceError(f"{path} is missing required columns: {', '.join(missing)}")

    df = df[HISTORY_COLUMNS].copy()
    df["Run Date"] = pd.to_datetime(df["Run Date"], errors="coerce")
    df = df[df["Run Date"].notna()].copy()
    df["Run Date"] = df["Run Date"].dt.normalize()

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    for col in df.columns:
        if col not in NUMERIC_COLUMNS and col != "Run Date":
            df[col] = df[col].fillna("").astype(str).str.strip()

    df.sort_values(["Run Date", "Account Number", "Action", "Symbol"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return CleanHistoryResult(df, path, raw_rows, len(df))


def stable_transaction_key(df: pd.DataFrame) -> pd.Series:
    parts = []
    for col in TRANSACTION_KEY_COLUMNS:
        if col == "Run Date":
            part = df[col].dt.strftime("%Y-%m-%d")
        elif col in NUMERIC_COLUMNS:
            part = df[col].map(lambda x: f"{float(x):.10g}")
        else:
            part = df[col].fillna("").astype(str).str.strip()
        parts.append(part)

    key = parts[0]
    for part in parts[1:]:
        key = key + "\x1f" + part
    return key


def merge_history_files(
    csv_paths: Iterable[str | pathlib.Path],
    output_path: str | pathlib.Path | None = None,
) -> MergeResult:
    paths = [pathlib.Path(p) for p in csv_paths]
    if not paths:
        raise PerformanceError("Pass at least one CSV path.")

    cleaned = [clean_fidelity_history(p) for p in paths]
    merged = pd.concat([item.frame for item in cleaned], ignore_index=True)
    input_rows = sum(item.raw_rows for item in cleaned)
    valid_rows = sum(item.valid_rows for item in cleaned)

    merged["_transaction_key"] = stable_transaction_key(merged)
    duplicate_rows = int(merged.duplicated("_transaction_key", keep="first").sum())
    merged = merged.drop_duplicates("_transaction_key", keep="first").drop(columns=["_transaction_key"])
    merged.sort_values(["Run Date", "Account Number", "Action", "Symbol"], ascending=[False, True, True, True], inplace=True)
    merged.reset_index(drop=True, inplace=True)

    first_date = merged["Run Date"].min()
    last_date = merged["Run Date"].max()
    if pd.isna(first_date) or pd.isna(last_date):
        raise PerformanceError("No valid transaction dates found after cleaning inputs.")

    if output_path is None:
        output_path = pathlib.Path(f"Accounts_History_{first_date:%Y%m%d}_{last_date:%Y%m%d}.csv")
    else:
        output_path = pathlib.Path(output_path)

    return MergeResult(
        frame=merged,
        output_path=output_path,
        input_rows=input_rows,
        valid_rows=valid_rows,
        duplicate_rows=duplicate_rows,
        output_rows=len(merged),
        first_date=first_date,
        last_date=last_date,
    )


def write_merged_history(result: MergeResult) -> None:
    out = result.frame.copy()
    out["Run Date"] = out["Run Date"].dt.strftime("%Y-%m-%d")
    out.to_csv(result.output_path, index=False, quoting=csv.QUOTE_MINIMAL)


def load_account_history(path: str | pathlib.Path, account_number: str) -> pd.DataFrame:
    df = clean_fidelity_history(path).frame
    account_number = str(account_number).strip()
    df = df[df["Account Number"] == account_number].copy()
    if df.empty:
        raise PerformanceError(f"No rows found for account number {account_number} in {path}.")
    df.sort_values("Run Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def determine_period(
    df: pd.DataFrame,
    period: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp, str]:
    overall_end = df["Run Date"].max().normalize()
    earliest = df["Run Date"].min().normalize()

    if start or end:
        if not start or not end:
            raise PerformanceError("Use --start and --end together.")
        start_ts = normalize_date(start)
        end_ts = normalize_date(end)
        label = f"{start_ts.date()}_{end_ts.date()}"
    else:
        if not period:
            raise PerformanceError("Provide a period or use --start and --end.")
        period = period.upper()
        end_ts = overall_end
        if period == "1Y":
            start_ts = end_ts - timedelta(days=365)
        elif period == "6M":
            start_ts = end_ts - timedelta(days=182)
        elif period == "3M":
            start_ts = end_ts - timedelta(days=91)
        elif period == "YTD":
            start_ts = pd.Timestamp(end_ts.year, 1, 1)
        elif re.fullmatch(r"\d{4}", period):
            year = int(period)
            start_ts = pd.Timestamp(year, 1, 1)
            end_ts = min(pd.Timestamp(year, 12, 31), overall_end)
        else:
            raise PerformanceError(f"Unsupported period {period}")
        label = period

    if start_ts > end_ts:
        raise PerformanceError(f"Start date {start_ts.date()} is after end date {end_ts.date()}.")
    if start_ts < earliest:
        raise PerformanceError(
            f"Requested start {start_ts.date()} is before earliest history {earliest.date()}. "
            "Use complete history back to account inception or provide a later start."
        )
    if end_ts > overall_end:
        raise PerformanceError(
            f"Requested end {end_ts.date()} is after latest history {overall_end.date()}."
        )
    return start_ts, end_ts, label


def symbols_for_pricing(df: pd.DataFrame) -> list[str]:
    syms = sorted({str(s).strip().upper() for s in df["Symbol"].dropna()})
    return [sym for sym in syms if sym and sym not in CASH_SYMBOLS]


def extract_yfinance_series(data: pd.DataFrame, field: str, symbol: str) -> pd.Series | None:
    if data.empty:
        return None

    if isinstance(data.columns, pd.MultiIndex):
        if field in data.columns.get_level_values(0):
            obj = data[field]
            if isinstance(obj, pd.DataFrame):
                if symbol in obj.columns:
                    return obj[symbol]
                if len(obj.columns) == 1:
                    return obj.iloc[:, 0]
            return obj
        key = (field, symbol)
        if key in data.columns:
            return data[key]
        return None

    if field in data.columns:
        return data[field]
    return None


def download_market_data(
    symbols: Iterable[str],
    earliest_date: pd.Timestamp,
    end: pd.Timestamp,
    max_retries: int = 3,
) -> MarketData:
    symbols = list(symbols)
    if not symbols:
        empty = pd.DataFrame()
        return MarketData(empty, empty, end)

    today = pd.Timestamp.today().normalize()
    basis_end = max(end.normalize(), today)
    start_dl = earliest_date.normalize() - timedelta(days=10)
    end_dl = basis_end + timedelta(days=1)
    all_prices: dict[str, pd.Series] = {}
    all_splits: dict[str, pd.Series] = {}

    for sym in symbols:
        last_err = None
        for _ in range(max_retries):
            try:
                data = yf.download(
                    sym,
                    start=start_dl,
                    end=end_dl,
                    progress=False,
                    auto_adjust=False,
                    actions=True,
                )
                close = extract_yfinance_series(data, "Close", sym)
                if close is None or close.dropna().empty:
                    last_err = RuntimeError(f"Empty price data for {sym}")
                    continue

                close = pd.to_numeric(close, errors="coerce").dropna()
                close.index = pd.to_datetime(close.index).normalize()
                close = close[~close.index.duplicated(keep="last")]
                close.name = sym
                all_prices[sym] = close

                split = extract_yfinance_series(data, "Stock Splits", sym)
                if split is None:
                    split = pd.Series(dtype=float)
                split = pd.to_numeric(split, errors="coerce").fillna(0.0)
                split.index = pd.to_datetime(split.index).normalize()
                split = split[split != 0.0]
                split.name = sym
                all_splits[sym] = split
                break
            except Exception as exc:  # pragma: no cover - network/library failures vary.
                last_err = exc
        else:
            raise PerformanceError(f"Failed to download prices for {sym}: {last_err}")

    prices = pd.concat(all_prices.values(), axis=1).sort_index().ffill()
    splits = pd.concat(all_splits.values(), axis=1).sort_index() if all_splits else pd.DataFrame()
    if not splits.empty:
        splits = splits.fillna(0.0)
    return MarketData(prices=prices, splits=splits, basis_end=basis_end)


def get_price(market_data: MarketData, symbol: str, date: pd.Timestamp) -> float:
    symbol = str(symbol).strip().upper()
    if symbol == "SPAXX":
        return 1.0
    if not symbol:
        return 0.0
    if symbol not in market_data.prices.columns:
        raise PerformanceError(f"No price series available for {symbol}")
    s = market_data.prices[symbol].loc[:date]
    s = s.dropna()
    if s.empty:
        raise PerformanceError(f"No price available for {symbol} on or before {date.date()}")
    return float(s.iloc[-1])


def actual_positions(df: pd.DataFrame, up_to_date: pd.Timestamp) -> pd.Series:
    sub = df[(df["Run Date"] <= up_to_date) & df["Symbol"].astype(bool)]
    if sub.empty:
        return pd.Series(dtype=float)
    pos = sub.groupby("Symbol")["Quantity"].sum()
    pos = pos[pos.abs() > 1e-9]
    pos.index = pos.index.astype(str).str.strip().str.upper()
    return pos.sort_index()


def cash_balance(df: pd.DataFrame, up_to_date: pd.Timestamp) -> float:
    sub = df[(df["Run Date"] <= up_to_date) & (df["Type"].str.upper() == "CASH")]
    return float(sub["Amount"].sum()) if not sub.empty else 0.0


def split_factor_after(market_data: MarketData, symbol: str, date: pd.Timestamp) -> float:
    symbol = str(symbol).strip().upper()
    if market_data.splits.empty or symbol not in market_data.splits.columns:
        return 1.0
    s = market_data.splits[symbol]
    s = s[(s.index > date.normalize()) & (s.index <= market_data.basis_end)]
    s = s[(s > 0.0) & (s != 1.0)]
    if s.empty:
        return 1.0
    return float(s.prod())


def compute_valuation(df: pd.DataFrame, market_data: MarketData, date: pd.Timestamp) -> Valuation:
    date = date.normalize()
    positions = actual_positions(df, date)
    free_cash = cash_balance(df, date)
    security_value = 0.0
    spaxx_value = 0.0
    position_values: list[PositionValue] = []

    for sym, qty in positions.items():
        qty = float(qty)
        if sym == "SPAXX":
            value = qty
            spaxx_value += value
            position_values.append(PositionValue(sym, qty, 1.0, qty, 1.0, value))
            continue

        factor = split_factor_after(market_data, sym, date)
        priced_qty = qty * factor
        price = get_price(market_data, sym, date)
        value = priced_qty * price
        security_value += value
        position_values.append(PositionValue(sym, qty, factor, priced_qty, price, value))

    total = security_value + spaxx_value + free_cash
    return Valuation(
        date=date,
        total_value=total,
        security_value=security_value,
        free_cash=free_cash,
        spaxx_value=spaxx_value,
        positions=positions,
        position_values=position_values,
    )


def action_contains(action: str, patterns: Iterable[str]) -> bool:
    action_upper = str(action).upper()
    return any(pattern.upper() in action_upper for pattern in patterns)


def is_external_cash_action(action: str) -> bool:
    return action_contains(action, EXTERNAL_CASH_PATTERNS)


def is_internal_cash_action(action: str) -> bool:
    return action_contains(action, INTERNAL_CASH_PATTERNS)


def external_flows(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    period = df[(df["Run Date"] > start) & (df["Run Date"] <= end)].copy()
    flows = []
    unsupported = []

    for idx, row in period.iterrows():
        amount = float(row["Amount"])
        type_name = str(row["Type"]).upper()
        action = str(row["Action"])
        if abs(amount) < 1e-9:
            continue
        if type_name != "CASH":
            continue
        if is_external_cash_action(action):
            flows.append(idx)
        elif is_internal_cash_action(action):
            continue
        else:
            unsupported.append(row)

    if unsupported:
        details = "\n".join(
            f"  {row['Run Date'].date()} {row['Action']} amount={float(row['Amount']):,.2f}"
            for row in unsupported[:10]
        )
        raise UnsupportedTransactionError(
            "Unsupported nonzero cash transaction(s); classify before trusting returns:\n" + details
        )

    result = period.loc[flows].copy() if flows else period.iloc[0:0].copy()
    result.sort_values("Run Date", inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


def validate_cash_classification(df: pd.DataFrame, end: pd.Timestamp) -> None:
    period = df[(df["Run Date"] <= end) & (df["Type"].str.upper() == "CASH")].copy()
    unsupported = []
    for _, row in period.iterrows():
        amount = float(row["Amount"])
        if abs(amount) < 1e-9:
            continue
        action = str(row["Action"])
        if is_external_cash_action(action) or is_internal_cash_action(action):
            continue
        unsupported.append(row)

    if unsupported:
        details = "\n".join(
            f"  {row['Run Date'].date()} {row['Action']} amount={float(row['Amount']):,.2f}"
            for row in unsupported[:10]
        )
        raise UnsupportedTransactionError(
            "Unsupported nonzero cash transaction(s) before or inside this period; "
            "classify before trusting valuations:\n" + details
        )


def validate_share_actions(df: pd.DataFrame, market_data: MarketData, end: pd.Timestamp) -> list[str]:
    messages: list[str] = []

    share_rows = df[
        (df["Run Date"] <= end)
        & ((df["Type"].str.upper() == "SHARES") | df["Action"].str.upper().str.contains("DISTRIBUTION"))
        & (df["Quantity"].abs() > 1e-9)
    ].copy()

    split_events: list[tuple[str, pd.Timestamp, float]] = []
    if not market_data.splits.empty:
        for sym in market_data.splits.columns:
            series = market_data.splits[sym]
            for split_date, ratio in series.items():
                ratio = float(ratio)
                split_date = pd.Timestamp(split_date).normalize()
                if split_date <= end and ratio > 0.0 and ratio != 1.0:
                    split_events.append((str(sym).upper(), split_date, ratio))

    matched_share_indexes: set[int] = set()
    for sym, split_date, ratio in split_events:
        before = actual_positions(df[df["Run Date"] < split_date], split_date)
        qty_before = float(before.get(sym, 0.0))
        expected_distribution = qty_before * (ratio - 1.0)
        same_day = share_rows[
            (share_rows["Symbol"].str.upper() == sym)
            & (share_rows["Run Date"] == split_date)
        ]
        actual_distribution = float(same_day["Quantity"].sum()) if not same_day.empty else 0.0
        tolerance = max(SPLIT_TOLERANCE_ABS, abs(expected_distribution) * SPLIT_TOLERANCE_REL)

        if abs(expected_distribution) > tolerance or abs(actual_distribution) > tolerance:
            if abs(actual_distribution - expected_distribution) > tolerance:
                raise PerformanceError(
                    f"Split mismatch for {sym} on {split_date.date()}: Yahoo ratio {ratio:g} "
                    f"implies distribution {expected_distribution:.6f}, but Fidelity rows sum to "
                    f"{actual_distribution:.6f}."
                )
            matched_share_indexes.update(same_day.index.tolist())
            messages.append(
                f"{sym} {ratio:g}-for-1 split on {split_date.date()} reconciled "
                f"with {actual_distribution:.6f} distributed shares."
            )

    unmatched = share_rows[~share_rows.index.isin(matched_share_indexes)]
    if not unmatched.empty:
        details = "\n".join(
            f"  {row['Run Date'].date()} {row['Action']} {row['Symbol']} qty={float(row['Quantity']):.6f}"
            for _, row in unmatched.head(10).iterrows()
        )
        raise PerformanceError(
            "Unsupported share distribution/corporate action; cannot safely value split-adjusted prices:\n"
            + details
        )

    return messages


def xnpv(rate: float, cashflows: list[float], dates: list[pd.Timestamp]) -> float:
    if rate <= -1.0:
        return math.inf
    base = dates[0]
    return float(
        sum(cf / ((1.0 + rate) ** ((date - base).days / 365.0)) for cf, date in zip(cashflows, dates))
    )


def xirr(cashflows: list[float], dates: list[pd.Timestamp]) -> float:
    if len(cashflows) != len(dates):
        raise ValueError("cashflows and dates must have the same length")

    pairs = [(float(cf), normalize_date(date)) for cf, date in zip(cashflows, dates) if abs(float(cf)) > 1e-9]
    if len(pairs) < 2:
        raise PerformanceError("Need at least two nonzero cashflows for XIRR.")
    cashflows = [p[0] for p in pairs]
    dates = [p[1] for p in pairs]
    if not any(cf < 0 for cf in cashflows) or not any(cf > 0 for cf in cashflows):
        raise PerformanceError("XIRR needs at least one positive and one negative cashflow.")

    candidates = [
        -0.999999,
        -0.999,
        -0.99,
        -0.95,
        -0.9,
        -0.75,
        -0.5,
        -0.25,
        -0.1,
        0.0,
        0.1,
        0.25,
        0.5,
        1.0,
        2.0,
        5.0,
        10.0,
        25.0,
        50.0,
        100.0,
        250.0,
        1000.0,
    ]
    values = [(rate, xnpv(rate, cashflows, dates)) for rate in candidates]
    brackets = []
    for (left, f_left), (right, f_right) in zip(values, values[1:]):
        if not math.isfinite(f_left) or not math.isfinite(f_right):
            continue
        if abs(f_left) < 1e-7:
            return left
        if f_left * f_right < 0:
            brackets.append((left, right))

    if not brackets:
        raise PerformanceError("Could not bracket an XIRR root; cashflows may have no valid IRR.")

    left, right = min(brackets, key=lambda pair: abs(sum(pair) / 2.0 - 0.1))
    f_left = xnpv(left, cashflows, dates)
    for _ in range(200):
        mid = (left + right) / 2.0
        f_mid = xnpv(mid, cashflows, dates)
        if abs(f_mid) < 1e-7 or abs(right - left) < 1e-10:
            return mid
        if f_left * f_mid <= 0:
            right = mid
        else:
            left = mid
            f_left = f_mid

    raise PerformanceError("XIRR solver did not converge.")


def build_xirr_cashflows(
    start_value: float,
    end_value: float,
    flows: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[list[float], list[pd.Timestamp]]:
    cfs: list[float] = []
    dates: list[pd.Timestamp] = []
    if abs(start_value) > 1e-9:
        cfs.append(-float(start_value))
        dates.append(start)
    for _, row in flows.iterrows():
        cfs.append(-float(row["Amount"]))
        dates.append(pd.Timestamp(row["Run Date"]).normalize())
    cfs.append(float(end_value))
    dates.append(end)
    return cfs, dates


def compute_twr(
    df: pd.DataFrame,
    market_data: MarketData,
    start: pd.Timestamp,
    end: pd.Timestamp,
    flows: pd.DataFrame,
) -> tuple[float, float]:
    if start >= end:
        return float("nan"), float("nan")

    flow_by_date = flows.groupby("Run Date")["Amount"].sum() if not flows.empty else pd.Series(dtype=float)
    prev_value = compute_valuation(df, market_data, start).total_value
    chain = 1.0

    for date in pd.date_range(start + timedelta(days=1), end, freq="D"):
        value = compute_valuation(df, market_data, date).total_value
        flow = float(flow_by_date.get(date.normalize(), 0.0))
        if abs(prev_value) < 1e-9:
            prev_value = value
            continue
        daily_return = (value - flow) / prev_value - 1.0
        chain *= 1.0 + daily_return
        prev_value = value

    total_return = chain - 1.0
    years = (end - start).days / 365.0
    annualized = chain ** (1.0 / years) - 1.0 if years > 0 and chain > 0 else float("nan")
    return total_return, annualized


def validation_ok(diff: float, expected: float) -> bool:
    tolerance = max(VALUE_TOLERANCE_ABS, abs(expected) * VALUE_TOLERANCE_REL)
    return abs(diff) <= tolerance


def load_holdings_snapshot(path: str | pathlib.Path) -> pd.Series:
    path = pathlib.Path(path)
    header_row = None
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        for idx, line in enumerate(f):
            cols = [c.strip().strip('"') for c in line.split(",")]
            if "Symbol" in cols and any(col in cols for col in ("Quantity", "Current Quantity", "Shares")):
                header_row = idx
                break
    if header_row is None:
        raise PerformanceError(f"Could not find Symbol/Quantity columns in holdings file {path}")

    df = pd.read_csv(path, skiprows=header_row)
    symbol_col = "Symbol"
    quantity_col = next(
        (col for col in ("Quantity", "Current Quantity", "Shares") if col in df.columns),
        None,
    )
    if quantity_col is None:
        raise PerformanceError(f"Could not find a holdings quantity column in {path}")

    df[symbol_col] = df[symbol_col].fillna("").astype(str).str.strip().str.upper()
    df[quantity_col] = pd.to_numeric(df[quantity_col], errors="coerce").fillna(0.0)
    df = df[df[symbol_col].astype(bool)]
    holdings = df.groupby(symbol_col)[quantity_col].sum()
    holdings = holdings[holdings.abs() > 1e-9]
    return holdings.sort_index()


def validate_holdings(expected: pd.Series, actual: pd.Series) -> list[str]:
    messages: list[str] = []
    symbols = sorted(set(expected.index) | set(actual.index))
    for sym in symbols:
        exp = float(expected.get(sym, 0.0))
        act = float(actual.get(sym, 0.0))
        tolerance = max(0.001, abs(exp) * 0.001)
        if abs(act - exp) > tolerance:
            messages.append(f"{sym}: calculated {act:.6f}, holdings file {exp:.6f}")
    return messages


def analyze_performance(
    df: pd.DataFrame,
    account: str,
    period: str | None = None,
    start_arg: str | None = None,
    end_arg: str | None = None,
    holdings_csv: str | pathlib.Path | None = None,
    expected_cash: float | None = None,
    expected_value: float | None = None,
    market_loader: Callable[[Iterable[str], pd.Timestamp, pd.Timestamp], MarketData] = download_market_data,
) -> PerformanceResult:
    start, end, label = determine_period(df, period, start_arg, end_arg)
    warnings = [
        "Assuming the transaction CSV is complete from account inception through the end date."
    ]

    market_data = market_loader(symbols_for_pricing(df), df["Run Date"].min(), end)
    validate_cash_classification(df, end)
    split_messages = validate_share_actions(df, market_data, end)
    start_valuation = compute_valuation(df, market_data, start)
    end_valuation = compute_valuation(df, market_data, end)
    flows = external_flows(df, start, end)
    net_external = float(flows["Amount"].sum()) if not flows.empty else 0.0
    cfs, cf_dates = build_xirr_cashflows(
        start_valuation.total_value,
        end_valuation.total_value,
        flows,
        start,
        end,
    )
    xirr_rate = xirr(cfs, cf_dates)
    twr_total, twr_annualized = compute_twr(df, market_data, start, end, flows)

    pl = end_valuation.total_value - start_valuation.total_value - net_external
    denom = start_valuation.total_value + net_external
    pl_pct = pl / denom if abs(denom) > 1e-9 else float("nan")

    validations: list[ValidationResult] = []
    if expected_cash is not None:
        actual_cash = end_valuation.cash_core_value
        validations.append(
            ValidationResult(
                "Cash/core",
                float(expected_cash),
                actual_cash,
                validation_ok(actual_cash - float(expected_cash), float(expected_cash)),
            )
        )
    if expected_value is not None:
        actual_value = end_valuation.total_value
        validations.append(
            ValidationResult(
                "Total value",
                float(expected_value),
                actual_value,
                validation_ok(actual_value - float(expected_value), float(expected_value)),
            )
        )
    if holdings_csv is not None:
        expected_holdings = load_holdings_snapshot(holdings_csv)
        holding_warnings = validate_holdings(expected_holdings, end_valuation.positions)
        if holding_warnings:
            warnings.append("Holdings validation mismatches: " + "; ".join(holding_warnings[:10]))
        else:
            warnings.append("Holdings validation matched calculated end quantities.")

    return PerformanceResult(
        account=account,
        period_label=label,
        start=start,
        end=end,
        years=(end - start).days / 365.0 if end > start else 0.0,
        start_valuation=start_valuation,
        end_valuation=end_valuation,
        flows=flows,
        net_external=net_external,
        pl=pl,
        pl_pct=pl_pct,
        xirr_rate=xirr_rate,
        twr_total=twr_total,
        twr_annualized=twr_annualized,
        warnings=warnings,
        validations=validations,
        split_adjustments=split_messages,
    )


def money(value: float) -> str:
    return f"${value:,.2f}"


def pct(value: float) -> str:
    if not math.isfinite(value):
        return "N/A"
    return f"{value * 100:.2f}%"
