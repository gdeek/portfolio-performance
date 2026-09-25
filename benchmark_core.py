#!/usr/bin/env python3

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd

from performance_core import (
    MarketData,
    PerformanceError,
    build_xirr_cashflows,
    chain_daily_returns,
    xirr,
)


@dataclass(frozen=True)
class BenchmarkHolding:
    symbol: str
    weight: float


DEFAULT_BENCHMARKS = ("QQQM", "VTI", "VGT")
DEFAULT_BLENDS = (
    (
        BenchmarkHolding("QQQM", 0.5),
        BenchmarkHolding("VTI", 0.3),
        BenchmarkHolding("VGT", 0.2),
    ),
)
DIVIDEND_WARNING_PERIOD_DAYS = 120


@dataclass
class BenchmarkEvent:
    date: pd.Timestamp
    kind: str
    symbol: str
    amount: float
    price: float
    shares: float
    shares_after: float
    dividend_per_share: float = 0.0


@dataclass
class BenchmarkResult:
    label: str
    holdings: list[BenchmarkHolding]
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
    positions: dict[str, float]
    events: list[BenchmarkEvent] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class BenchmarkLeg:
    holding: BenchmarkHolding
    prices: pd.Series
    price_series: pd.Series
    trading_days: set[pd.Timestamp]
    dividend_map: dict[pd.Timestamp, float]
    cash: float
    shares: float = 0.0
    pending_dividend_cash: float = 0.0
    pending_dividend_per_share: float = 0.0


def holdings_label(holdings: Iterable[BenchmarkHolding]) -> str:
    holdings = list(holdings)
    if len(holdings) == 1:
        return holdings[0].symbol
    return " + ".join(f"{holding.weight * 100:g}% {holding.symbol}" for holding in holdings)


def normalize_holdings(holdings: Iterable[BenchmarkHolding]) -> tuple[BenchmarkHolding, ...]:
    combined: dict[str, float] = {}
    for holding in holdings:
        symbol = str(holding.symbol).strip().upper()
        if not symbol:
            raise PerformanceError("Benchmark holdings need a ticker symbol.")
        if holding.weight <= 0.0:
            raise PerformanceError(f"Benchmark weight for {symbol} must be positive.")
        combined[symbol] = combined.get(symbol, 0.0) + float(holding.weight)
    if not combined:
        raise PerformanceError("At least one benchmark ticker is required.")
    total = sum(combined.values())
    return tuple(
        BenchmarkHolding(symbol, weight / total) for symbol, weight in combined.items()
    )


def parse_blend(value: str) -> tuple[BenchmarkHolding, ...]:
    holdings: list[BenchmarkHolding] = []
    for token in str(value).split(","):
        token = token.strip()
        symbol, separator, weight_text = token.partition(":")
        symbol = symbol.strip().upper()
        if not separator or not symbol or not weight_text.strip():
            raise PerformanceError(
                f"Invalid blend {value!r}; expected TICKER:WEIGHT pairs separated by commas."
            )
        try:
            weight = float(weight_text)
        except ValueError:
            raise PerformanceError(f"Invalid blend weight {weight_text!r} in {token!r}.")
        if weight <= 0.0:
            raise PerformanceError(f"Blend weight for {symbol} must be positive.")
        holdings.append(BenchmarkHolding(symbol, weight))
    return normalize_holdings(holdings)


def event_series(frame: pd.DataFrame, symbol: str) -> pd.Series:
    if frame.empty or symbol not in frame.columns:
        return pd.Series(dtype=float)
    series = pd.to_numeric(frame[symbol], errors="coerce").fillna(0.0)
    series = series[series > 0.0]
    return series.sort_index()


def event_map(series: pd.Series) -> dict[pd.Timestamp, float]:
    if series.empty:
        return {}
    grouped = series.groupby(level=0).sum()
    return {pd.Timestamp(index).normalize(): float(value) for index, value in grouped.items()}


def simulate_allocation(
    holdings: Iterable[BenchmarkHolding],
    flows: pd.DataFrame,
    start_value: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
    market: MarketData,
) -> BenchmarkResult:
    """
    Replay the portfolio's external cash flows into a fixed allocation of symbols.

    Each flow is split by the target weights at arrival, contributions and
    withdrawals execute at the close of the next trading day, dividends are
    reinvested at the ex-dividend close within the same leg, and holdings drift
    between flows (no rebalancing). Yahoo price and dividend series are already
    split-adjusted, so quantities are simulated in that basis without extra
    share adjustments. Fractional shares are allowed and no fees or taxes are
    modeled.
    """
    normalized = normalize_holdings(holdings)
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    if start >= end:
        raise PerformanceError(
            f"Benchmark comparison needs a start before the end date "
            f"({start.date()} to {end.date()})."
        )
    if not flows.empty:
        outside = (flows["Run Date"] <= start) | (flows["Run Date"] > end)
        if outside.any():
            raise PerformanceError("Benchmark flows must fall inside the analysis period.")

    dates = pd.date_range(start, end, freq="D")
    legs: list[BenchmarkLeg] = []
    for holding in normalized:
        symbol = holding.symbol
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

        legs.append(
            BenchmarkLeg(
                holding=holding,
                prices=prices,
                price_series=prices.reindex(dates, method="ffill"),
                trading_days=set(prices.index),
                dividend_map=event_map(event_series(market.dividends, symbol)),
                cash=float(start_value) * holding.weight,
            )
        )

    flow_by_date = flows.groupby("Run Date")["Amount"].sum() if not flows.empty else pd.Series(dtype=float)

    values: list[float] = []
    events: list[BenchmarkEvent] = []
    dividends_reinvested = 0.0
    dividend_count = 0

    for date in dates:
        date = pd.Timestamp(date)
        flow_amount = float(flow_by_date.get(date, 0.0))

        for leg in legs:
            symbol = leg.holding.symbol

            leg.cash += leg.holding.weight * flow_amount

            if date in leg.dividend_map and leg.shares > 1e-12:
                dividend_per_share = float(leg.dividend_map[date])
                amount = leg.shares * dividend_per_share
                leg.cash += amount
                leg.pending_dividend_cash += amount
                leg.pending_dividend_per_share += dividend_per_share

            if date in leg.trading_days and abs(leg.cash) > 1e-9:
                price = float(leg.prices.loc[date])
                flow_cash = leg.cash - leg.pending_dividend_cash
                if leg.pending_dividend_cash > 1e-9:
                    dividend_shares = leg.pending_dividend_cash / price
                    leg.shares += dividend_shares
                    dividends_reinvested += leg.pending_dividend_cash
                    dividend_count += 1
                    events.append(
                        BenchmarkEvent(
                            date,
                            "dividend",
                            symbol,
                            leg.pending_dividend_cash,
                            price,
                            dividend_shares,
                            leg.shares,
                            leg.pending_dividend_per_share,
                        )
                    )
                    leg.pending_dividend_cash = 0.0
                    leg.pending_dividend_per_share = 0.0
                if abs(flow_cash) > 1e-9:
                    flow_shares = flow_cash / price
                    leg.shares += flow_shares
                    events.append(
                        BenchmarkEvent(
                            date,
                            "buy" if flow_shares > 0 else "sell",
                            symbol,
                            flow_cash,
                            price,
                            flow_shares,
                            leg.shares,
                        )
                    )
                leg.cash = 0.0

        values.append(
            sum(leg.cash + leg.shares * float(leg.price_series.loc[date]) for leg in legs)
        )

    value_series = pd.Series(values, index=dates)
    benchmark_start = float(value_series.iloc[0])
    end_value = float(value_series.iloc[-1])
    net_external = float(flows["Amount"].sum()) if not flows.empty else 0.0
    pl = end_value - benchmark_start - net_external
    denominator = benchmark_start + net_external
    pl_pct = pl / denominator if abs(denominator) > 1e-9 else float("nan")
    label = holdings_label(normalized)

    try:
        cashflows, cashflow_dates = build_xirr_cashflows(benchmark_start, end_value, flows, start, end)
        xirr_rate = xirr(cashflows, cashflow_dates)
    except PerformanceError:
        xirr_rate = float("nan")

    twr_total, twr_annualized = chain_daily_returns(value_series, flows, start, end)

    warnings: list[str] = []
    if math.isnan(xirr_rate):
        warnings.append(f"{label}: XIRR could not be computed for this period.")
    if dividend_count == 0 and (end - start).days >= DIVIDEND_WARNING_PERIOD_DAYS:
        warnings.append(
            f"{label}: no dividend events found in this period; verify benchmark dividend data."
        )

    return BenchmarkResult(
        label=label,
        holdings=list(normalized),
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
        positions={leg.holding.symbol: float(leg.shares) for leg in legs},
        events=events,
        warnings=warnings,
    )


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
    """
    return simulate_allocation(
        (BenchmarkHolding(str(symbol).strip().upper(), 1.0),),
        flows,
        start_value,
        start,
        end,
        market,
    )
