#!/usr/bin/env python3

from __future__ import annotations

import csv
import hashlib
import json
import math
import pathlib
import re
import time
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
YAHOO_DOWNLOAD_CACHE_TTL_SECONDS = 60 * 60
YAHOO_DOWNLOAD_CACHE_VERSION = 1
YAHOO_DOWNLOAD_AUTO_ADJUST = False
YAHOO_DOWNLOAD_ACTIONS = True

_YahooDownloadCacheKey = tuple[str, str, str, bool, bool, int]
_YahooDownloadCacheEntry = tuple[float, pd.DataFrame]
_YAHOO_DOWNLOAD_CACHE: dict[_YahooDownloadCacheKey, _YahooDownloadCacheEntry] = {}


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


def clear_yahoo_download_cache() -> None:
    _YAHOO_DOWNLOAD_CACHE.clear()


def yahoo_download_cache_dir() -> pathlib.Path:
    return pathlib.Path.home() / ".cache" / "fidelity-portfolio-tracker" / "yahoo"


def yahoo_download_cache_key(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> _YahooDownloadCacheKey:
    return (
        symbol,
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
        YAHOO_DOWNLOAD_AUTO_ADJUST,
        YAHOO_DOWNLOAD_ACTIONS,
        YAHOO_DOWNLOAD_CACHE_VERSION,
    )


def yahoo_download_cache_paths(
    cache_key: _YahooDownloadCacheKey,
    cache_dir: str | pathlib.Path | None = None,
) -> tuple[pathlib.Path, pathlib.Path]:
    symbol, start, end, auto_adjust, actions, version = cache_key
    payload = {
        "actions": actions,
        "auto_adjust": auto_adjust,
        "end": end,
        "start": start,
        "symbol": symbol,
        "version": version,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    safe_symbol = re.sub(r"[^A-Z0-9._-]+", "_", symbol.upper()).strip("_") or "SYMBOL"
    stem = f"{safe_symbol}_{start}_{end}_{digest}"
    root = pathlib.Path(cache_dir).expanduser() if cache_dir is not None else yahoo_download_cache_dir()
    return root / f"{stem}.pkl", root / f"{stem}.json"


def read_yahoo_disk_cache(
    cache_key: _YahooDownloadCacheKey,
    now: float,
    cache_dir: str | pathlib.Path | None = None,
) -> pd.DataFrame | None:
    data_path, metadata_path = yahoo_download_cache_paths(cache_key, cache_dir)
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        fetched_at = float(metadata["fetched_at"])
        if metadata.get("version") != YAHOO_DOWNLOAD_CACHE_VERSION:
            return None
        if now - fetched_at >= YAHOO_DOWNLOAD_CACHE_TTL_SECONDS:
            return None
        if (
            metadata.get("symbol"),
            metadata.get("start"),
            metadata.get("end"),
            metadata.get("auto_adjust"),
            metadata.get("actions"),
            metadata.get("version"),
        ) != cache_key:
            return None

        data = pd.read_pickle(data_path)
        if not isinstance(data, pd.DataFrame):
            return None
    except Exception:
        return None

    _YAHOO_DOWNLOAD_CACHE[cache_key] = (fetched_at, data.copy(deep=True))
    return data.copy(deep=True)


def write_yahoo_disk_cache(
    cache_key: _YahooDownloadCacheKey,
    data: pd.DataFrame,
    fetched_at: float,
    cache_dir: str | pathlib.Path | None = None,
) -> None:
    data_path, metadata_path = yahoo_download_cache_paths(cache_key, cache_dir)
    metadata = {
        "actions": cache_key[4],
        "auto_adjust": cache_key[3],
        "end": cache_key[2],
        "fetched_at": fetched_at,
        "start": cache_key[1],
        "symbol": cache_key[0],
        "version": cache_key[5],
    }
    data_tmp = data_path.with_name(f"{data_path.name}.{time.time_ns()}.tmp")
    metadata_tmp = metadata_path.with_name(f"{metadata_path.name}.{time.time_ns()}.tmp")

    try:
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_pickle(data_tmp)
        metadata_tmp.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
        data_tmp.replace(data_path)
        metadata_tmp.replace(metadata_path)
    except Exception:
        try:
            data_tmp.unlink(missing_ok=True)
            metadata_tmp.unlink(missing_ok=True)
        except OSError:
            pass


def download_yahoo_data(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    refresh_prices: bool = False,
    cache_dir: str | pathlib.Path | None = None,
) -> pd.DataFrame:
    symbol = str(symbol).strip().upper()
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    cache_key = yahoo_download_cache_key(symbol, start, end)
    now = time.time()

    if not refresh_prices:
        cached = _YAHOO_DOWNLOAD_CACHE.get(cache_key)
        if cached is not None:
            fetched_at, data = cached
            if now - fetched_at < YAHOO_DOWNLOAD_CACHE_TTL_SECONDS:
                return data.copy(deep=True)
            del _YAHOO_DOWNLOAD_CACHE[cache_key]

        cached_data = read_yahoo_disk_cache(cache_key, now, cache_dir)
        if cached_data is not None:
            return cached_data

    data = yf.download(
        symbol,
        start=start,
        end=end,
        progress=False,
        auto_adjust=YAHOO_DOWNLOAD_AUTO_ADJUST,
        actions=YAHOO_DOWNLOAD_ACTIONS,
    )
    if not data.empty:
        fetched_at = time.time()
        cached_copy = data.copy(deep=True)
        _YAHOO_DOWNLOAD_CACHE[cache_key] = (fetched_at, cached_copy)
        write_yahoo_disk_cache(cache_key, cached_copy, fetched_at, cache_dir)
    return data


def download_market_data(
    symbols: Iterable[str],
    earliest_date: pd.Timestamp,
    end: pd.Timestamp,
    max_retries: int = 3,
    refresh_prices: bool = False,
    cache_dir: str | pathlib.Path | None = None,
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
                data = download_yahoo_data(
                    sym,
                    start_dl,
                    end_dl,
                    refresh_prices=refresh_prices,
                    cache_dir=cache_dir,
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


def priced_positions(df: pd.DataFrame, market_data: MarketData, up_to_date: pd.Timestamp) -> pd.Series:
    """
    Return positions converted to the same split-adjusted basis as Yahoo Close.

    Fidelity may or may not emit a share-distribution row for a split. Pricing
    transaction quantities directly into the price provider's split-adjusted
    basis handles both cases and avoids double-counting distribution rows.
    """
    sub = df[
        (df["Run Date"] <= up_to_date)
        & df["Symbol"].astype(bool)
        & (df["Type"].str.upper() != "SHARES")
        & (~df["Action"].str.upper().str.contains("DISTRIBUTION", na=False))
    ].copy()
    if sub.empty:
        return pd.Series(dtype=float)

    totals: dict[str, float] = {}
    for _, row in sub.iterrows():
        sym = str(row["Symbol"]).strip().upper()
        qty = float(row["Quantity"])
        if abs(qty) < 1e-12:
            continue
        factor = 1.0 if sym == "SPAXX" else split_factor_after(market_data, sym, row["Run Date"])
        totals[sym] = totals.get(sym, 0.0) + qty * factor

    pos = pd.Series(totals, dtype=float)
    pos = pos[pos.abs() > 1e-9]
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
    adjusted_positions = priced_positions(df, market_data, date)
    free_cash = cash_balance(df, date)
    security_value = 0.0
    spaxx_value = 0.0
    position_values: list[PositionValue] = []

    for sym in sorted(set(positions.index) | set(adjusted_positions.index)):
        qty = float(positions.get(sym, 0.0))
        priced_qty = float(adjusted_positions.get(sym, qty))
        if sym == "SPAXX":
            priced_qty = qty
            value = priced_qty
            spaxx_value += value
            position_values.append(PositionValue(sym, qty, 1.0, priced_qty, 1.0, value))
            continue

        factor = priced_qty / qty if abs(qty) > 1e-12 else 1.0
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
            if abs(actual_distribution) <= tolerance:
                messages.append(
                    f"{sym} {ratio:g}-for-1 split on {split_date.date()} has no Fidelity "
                    "distribution row; valuation uses split-adjusted transaction quantities."
                )
                continue
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


def cumulative_deltas_on_dates(deltas: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    if deltas.empty:
        return pd.Series(0.0, index=dates)

    deltas = deltas.groupby(deltas.index).sum().sort_index()
    combined_index = deltas.index.union(dates).sort_values()
    return (
        deltas.reindex(combined_index, fill_value=0.0)
        .cumsum()
        .reindex(dates)
        .fillna(0.0)
        .astype(float)
    )


def total_value_series(df: pd.DataFrame, market_data: MarketData, dates: pd.DatetimeIndex) -> pd.Series:
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).normalize().drop_duplicates().sort_values()
    if dates.empty:
        return pd.Series(dtype=float)

    end = dates.max()
    run_dates = pd.to_datetime(df["Run Date"]).dt.normalize()
    symbols = df["Symbol"].fillna("").astype(str).str.strip().str.upper()
    types = df["Type"].fillna("").astype(str).str.upper()
    actions = df["Action"].fillna("").astype(str).str.upper()

    cash_mask = (run_dates <= end) & (types == "CASH")
    cash_deltas = df.loc[cash_mask].groupby(run_dates[cash_mask])["Amount"].sum()
    free_cash = cumulative_deltas_on_dates(cash_deltas, dates)

    spaxx_mask = (run_dates <= end) & (symbols == "SPAXX")
    spaxx_deltas = df.loc[spaxx_mask].groupby(run_dates[spaxx_mask])["Quantity"].sum()
    spaxx_value = cumulative_deltas_on_dates(spaxx_deltas, dates)

    priced_mask = (
        (run_dates <= end)
        & symbols.astype(bool)
        & (symbols != "SPAXX")
        & (types != "SHARES")
        & (~actions.str.contains("DISTRIBUTION", na=False))
    )
    security_value = pd.Series(0.0, index=dates)
    if priced_mask.any():
        priced_rows = df.loc[priced_mask, ["Run Date", "Quantity"]].copy()
        priced_rows["Run Date"] = run_dates[priced_mask].values
        priced_rows["Symbol"] = symbols[priced_mask].values
        priced_rows["Adjusted Quantity"] = [
            float(quantity) * split_factor_after(market_data, symbol, date)
            for symbol, date, quantity in zip(
                priced_rows["Symbol"],
                priced_rows["Run Date"],
                priced_rows["Quantity"],
            )
        ]

        quantity_deltas = (
            priced_rows.groupby(["Run Date", "Symbol"])["Adjusted Quantity"]
            .sum()
            .unstack(fill_value=0.0)
            .sort_index()
        )
        combined_index = quantity_deltas.index.union(dates).sort_values()
        quantities = (
            quantity_deltas.reindex(combined_index, fill_value=0.0)
            .cumsum()
            .reindex(dates)
            .fillna(0.0)
        )
        active_symbols = [sym for sym in quantities.columns if quantities[sym].abs().max() > 1e-9]

        if active_symbols:
            missing_symbols = [sym for sym in active_symbols if sym not in market_data.prices.columns]
            if missing_symbols:
                raise PerformanceError(f"No price series available for {missing_symbols[0]}")

            price_index = market_data.prices.index.union(dates).sort_values()
            prices = market_data.prices.reindex(price_index).ffill().reindex(dates)
            active_prices = prices[active_symbols]
            active_quantities = quantities[active_symbols]
            missing_price = active_prices.isna() & (active_quantities.abs() > 1e-9)
            if missing_price.any().any():
                date = missing_price.any(axis=1).idxmax()
                symbol = missing_price.loc[date][missing_price.loc[date]].index[0]
                raise PerformanceError(f"No price available for {symbol} on or before {date.date()}")

            security_value = (active_quantities * active_prices).sum(axis=1)

    return security_value + spaxx_value + free_cash


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
    values = total_value_series(df, market_data, pd.date_range(start, end, freq="D"))
    prev_value = float(values.loc[start.normalize()])
    chain = 1.0

    for date, value in values.iloc[1:].items():
        value = float(value)
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
    refresh_prices: bool = False,
) -> PerformanceResult:
    start, end, label = determine_period(df, period, start_arg, end_arg)
    warnings = [
        "Assuming the transaction CSV is complete from account inception through the end date."
    ]

    if market_loader is download_market_data:
        market_data = market_loader(
            symbols_for_pricing(df),
            df["Run Date"].min(),
            end,
            refresh_prices=refresh_prices,
        )
    else:
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
