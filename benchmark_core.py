#!/usr/bin/env python3

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from performance_core import (
    MarketData,
    PerformanceError,
    build_xirr_cashflows,
    chain_daily_returns,
    xirr,
)


DEFAULT_BENCHMARKS = ("QQQM",)
DIVIDEND_WARNING_PERIOD_DAYS = 120


@dataclass
class BenchmarkEvent:
    date: pd.Timestamp
    kind: str
    amount: float
    price: float
    shares: float
    shares_after: float
    dividend_per_share: float = 0.0


@dataclass
class BenchmarkResult:
    symbol: str
    start: pd.Timestamp
    end: pd.Timestamp
    years: float
    start_value: float
    end_value: float
    net_external: float
    pl: float
    pl_pct: float
    xirr_rate: float
    twr_total: float
    twr_annualized: float
    dividends_reinvested: float
    dividend_count: int
    shares_end: float
    events: list[BenchmarkEvent] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def event_series(frame: pd.DataFrame, symbol: str) -> pd.Series:
    if frame.empty or symbol not in frame.columns:
        return pd.Series(dtype=float)
    series = pd.to_numeric(frame[symbol], errors="coerce").fillna(0.0)
    series = series[series > 0.0]
    return series.sort_index()


def event_map(series: pd.Series, combine: str) -> dict[pd.Timestamp, float]:
    if series.empty:
        return {}
    grouped = series.groupby(level=0).prod() if combine == "prod" else series.groupby(level=0).sum()
    return {pd.Timestamp(index).normalize(): float(value) for index, value in grouped.items()}


def simulate_benchmark(
    symbol: str,
    flows: pd.DataFrame,
    start_value: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
    market: MarketData,
) -> BenchmarkResult:
    """
    Replay the portfolio's external cash flows into a single benchmark symbol.

    Contributions and withdrawals execute at the close of the next trading day,
    dividends are reinvested at the ex-dividend close, and splits adjust the
    share count. Fractional shares are allowed and no fees or taxes are modeled.
    """
    symbol = str(symbol).strip().upper()
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    if start >= end:
        raise PerformanceError(
            f"Benchmark comparison needs a start before the end date "
            f"({start.date()} to {end.date()})."
        )
    if market.prices.empty or symbol not in market.prices.columns:
        raise PerformanceError(f"No price series available for benchmark {symbol}.")

    prices = pd.to_numeric(market.prices[symbol], errors="coerce").dropna().sort_index()
    if prices.empty:
        raise PerformanceError(f"No price data available for benchmark {symbol}.")
    if prices.index.min() > start:
        raise PerformanceError(
            f"{symbol} has no price data on or before {start.date()} "
            f"(earliest {prices.index.min().date()}); cannot benchmark this period."
        )

    if not flows.empty:
        outside = (flows["Run Date"] <= start) | (flows["Run Date"] > end)
        if outside.any():
            raise PerformanceError("Benchmark flows must fall inside the analysis period.")

    dates = pd.date_range(start, end, freq="D")
    price_series = prices.reindex(dates, method="ffill")
    trading_days = set(prices.index)
    dividend_map = event_map(event_series(market.dividends, symbol), "sum")
    split_events = event_series(market.splits, symbol)
    split_map = event_map(split_events[split_events != 1.0], "prod")
    flow_by_date = flows.groupby("Run Date")["Amount"].sum() if not flows.empty else pd.Series(dtype=float)

    cash = float(start_value)
    shares = 0.0
    values: list[float] = []
    events: list[BenchmarkEvent] = []
    dividends_reinvested = 0.0
    dividend_count = 0
    pending_dividend_cash = 0.0
    pending_dividend_per_share = 0.0

    for date in dates:
        date = pd.Timestamp(date)

        if date in split_map:
            ratio = float(split_map[date])
            shares *= ratio
            events.append(BenchmarkEvent(date, "split", 0.0, float(price_series.loc[date]), ratio, shares))

        flow_amount = float(flow_by_date.get(date, 0.0))
        if abs(flow_amount) > 1e-9:
            cash += flow_amount

        if date in dividend_map and shares > 1e-12:
            dividend_per_share = float(dividend_map[date])
            amount = shares * dividend_per_share
            cash += amount
            pending_dividend_cash += amount
            pending_dividend_per_share += dividend_per_share

        if date in trading_days and abs(cash) > 1e-9:
            price = float(prices.loc[date])
            flow_cash = cash - pending_dividend_cash
            if pending_dividend_cash > 1e-9:
                dividend_shares = pending_dividend_cash / price
                shares += dividend_shares
                dividends_reinvested += pending_dividend_cash
                dividend_count += 1
                events.append(
                    BenchmarkEvent(
                        date,
                        "dividend",
                        pending_dividend_cash,
                        price,
                        dividend_shares,
                        shares,
                        pending_dividend_per_share,
                    )
                )
                pending_dividend_cash = 0.0
                pending_dividend_per_share = 0.0
            if abs(flow_cash) > 1e-9:
                flow_shares = flow_cash / price
                shares += flow_shares
                events.append(
                    BenchmarkEvent(
                        date,
                        "buy" if flow_shares > 0 else "sell",
                        flow_cash,
                        price,
                        flow_shares,
                        shares,
                    )
                )
            cash = 0.0

        values.append(cash + shares * float(price_series.loc[date]))

    value_series = pd.Series(values, index=dates)
    benchmark_start = float(value_series.iloc[0])
    end_value = float(value_series.iloc[-1])
    net_external = float(flows["Amount"].sum()) if not flows.empty else 0.0
    pl = end_value - benchmark_start - net_external
    denominator = benchmark_start + net_external
    pl_pct = pl / denominator if abs(denominator) > 1e-9 else float("nan")

    try:
        cashflows, cashflow_dates = build_xirr_cashflows(benchmark_start, end_value, flows, start, end)
        xirr_rate = xirr(cashflows, cashflow_dates)
    except PerformanceError:
        xirr_rate = float("nan")

    twr_total, twr_annualized = chain_daily_returns(value_series, flows, start, end)

    warnings: list[str] = []
    if math.isnan(xirr_rate):
        warnings.append(f"{symbol}: XIRR could not be computed for this period.")
    if dividend_count == 0 and (end - start).days >= DIVIDEND_WARNING_PERIOD_DAYS:
        warnings.append(
            f"{symbol}: no dividend events found in this period; verify benchmark dividend data."
        )

    return BenchmarkResult(
        symbol=symbol,
        start=start,
        end=end,
        years=(end - start).days / 365.0,
        start_value=benchmark_start,
        end_value=end_value,
        net_external=net_external,
        pl=pl,
        pl_pct=pl_pct,
        xirr_rate=xirr_rate,
        twr_total=twr_total,
        twr_annualized=twr_annualized,
        dividends_reinvested=dividends_reinvested,
        dividend_count=dividend_count,
        shares_end=float(shares),
        events=events,
        warnings=warnings,
    )
