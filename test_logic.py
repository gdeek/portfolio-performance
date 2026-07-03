from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from performance_core import (
    HISTORY_COLUMNS,
    MarketData,
    PerformanceError,
    UnsupportedTransactionError,
    actual_positions,
    clear_yahoo_download_cache,
    clean_fidelity_history,
    compute_twr,
    compute_valuation,
    download_market_data,
    external_flows,
    merge_history_files,
    validate_cash_classification,
    validate_share_actions,
    xirr,
)


def row(
    date,
    action,
    symbol="",
    type_name="Cash",
    price=0.0,
    quantity=0.0,
    amount=0.0,
    account="Z1",
):
    return {
        "Run Date": pd.Timestamp(date),
        "Account": "Individual",
        "Account Number": account,
        "Action": action,
        "Symbol": symbol,
        "Description": symbol,
        "Type": type_name,
        "Exchange Quantity": 0.0,
        "Exchange Currency": "",
        "Currency": "USD",
        "Price": price,
        "Quantity": quantity,
        "Exchange Rate": 0.0,
        "Commission": 0.0,
        "Fees": 0.0,
        "Accrued Interest": 0.0,
        "Amount": amount,
        "Settlement Date": "",
    }


def frame(rows):
    df = pd.DataFrame(rows, columns=HISTORY_COLUMNS)
    df["Run Date"] = pd.to_datetime(df["Run Date"]).dt.normalize()
    for col in (
        "Exchange Quantity",
        "Price",
        "Quantity",
        "Exchange Rate",
        "Commission",
        "Fees",
        "Accrued Interest",
        "Amount",
    ):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    for col in df.columns:
        if col not in (
            "Run Date",
            "Exchange Quantity",
            "Price",
            "Quantity",
            "Exchange Rate",
            "Commission",
            "Fees",
            "Accrued Interest",
            "Amount",
        ):
            df[col] = df[col].fillna("").astype(str).str.strip()
    return df


def market(prices, splits=None, basis_end="2026-01-01"):
    price_df = pd.DataFrame(prices)
    price_df.index = pd.to_datetime(price_df.index).normalize()
    split_df = pd.DataFrame(splits or {})
    if not split_df.empty:
        split_df.index = pd.to_datetime(split_df.index).normalize()
    return MarketData(price_df, split_df, pd.Timestamp(basis_end))


class PerformanceCoreTests(unittest.TestCase):
    def assertClose(self, actual, expected, places=6):
        self.assertAlmostEqual(actual, expected, places=places)

    def test_clean_history_drops_footer_and_merge_dedupes(self):
        with self.subTest("clean and merge"):
            import tempfile
            from pathlib import Path

            csv_text = """
Run Date,Account,Account Number,Action,Symbol,Description,Type,Exchange Quantity,Exchange Currency,Currency,Price,Quantity,Exchange Rate,Commission,Fees,Accrued Interest,Amount,Settlement Date
2025-01-01,Individual,Z1,Electronic Funds Transfer Received (Cash),,,,0,,USD,0,0,0,,,,100,
2025-01-02,Individual,Z1,YOU BOUGHT ABC (ABC) (Cash),ABC,ABC,Cash,0,,USD,10,10,0,,,,-100,
"The data and information in this spreadsheet is provided to you solely for your use"
Date downloaded 01/03/2025
""".lstrip()
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                first = tmp_path / "first.csv"
                second = tmp_path / "second.csv"
                first.write_text(csv_text)
                second.write_text(csv_text)

                cleaned = clean_fidelity_history(first)
                self.assertEqual(len(cleaned.frame), 2)

                merged = merge_history_files([first, second], tmp_path / "merged.csv")
                self.assertEqual(merged.valid_rows, 4)
                self.assertEqual(merged.duplicate_rows, 2)
                self.assertEqual(merged.output_rows, 2)

    def test_cash_balance_includes_free_cash_and_spaxx(self):
        df = frame(
            [
                row("2025-01-01", "Electronic Funds Transfer Received (Cash)", amount=1000),
                row("2025-01-02", "YOU BOUGHT ABC (ABC) (Cash)", "ABC", price=100, quantity=4, amount=-400),
                row("2025-01-31", "DIVIDEND RECEIVED FIDELITY GOVERNMENT MONEY MARKET (SPAXX) (Cash)", "SPAXX", amount=1),
                row("2025-01-31", "REINVESTMENT FIDELITY GOVERNMENT MONEY MARKET (SPAXX) (Cash)", "SPAXX", price=1, quantity=1, amount=-1),
            ]
        )
        valuation = compute_valuation(df, market({"ABC": {"2025-01-31": 120}}), pd.Timestamp("2025-01-31"))
        self.assertClose(valuation.security_value, 480)
        self.assertClose(valuation.free_cash, 600)
        self.assertClose(valuation.spaxx_value, 1)
        self.assertClose(valuation.total_value, 1081)

    def test_split_adjusted_valuation_reconciles_distribution(self):
        df = frame(
            [
                row("2025-01-01", "Electronic Funds Transfer Received (Cash)", amount=1000),
                row("2025-01-02", "YOU BOUGHT NETFLIX INC (NFLX) (Cash)", "NFLX", price=1000, quantity=1, amount=-1000),
                row("2025-11-17", "DISTRIBUTION NETFLIX INC (NFLX) (Cash)", "NFLX", type_name="Shares", quantity=9, amount=900),
            ]
        )
        md = market(
            {"NFLX": {"2025-01-02": 100, "2025-11-17": 110}},
            {"NFLX": {"2025-11-17": 10}},
            basis_end="2025-12-31",
        )
        messages = validate_share_actions(df, md, pd.Timestamp("2025-12-31"))
        self.assertIn("NFLX", messages[0])

        before = compute_valuation(df, md, pd.Timestamp("2025-01-02"))
        after = compute_valuation(df, md, pd.Timestamp("2025-11-17"))
        self.assertClose(before.total_value, 1000)
        self.assertClose(after.total_value, 1100)

    def test_unknown_share_distribution_stops(self):
        df = frame(
            [
                row("2025-01-02", "YOU BOUGHT XYZ (XYZ) (Cash)", "XYZ", price=100, quantity=1, amount=-100),
                row("2025-03-01", "DISTRIBUTION XYZ (XYZ) (Cash)", "XYZ", type_name="Shares", quantity=1, amount=100),
            ]
        )
        with self.assertRaises(PerformanceError):
            validate_share_actions(df, market({"XYZ": {"2025-03-01": 100}}), pd.Timestamp("2025-03-01"))

    def test_missing_split_distribution_is_synthesized_for_valuation(self):
        df = frame(
            [
                row("2026-05-29", "YOU BOUGHT KLA CORP COM NEW (KLAC) (Cash)", "KLAC", price=1900, quantity=0.338, amount=-642.2),
            ]
        )
        md = market(
            {"KLAC": {"2026-06-15": 200}},
            {"KLAC": {"2026-06-12": 10}},
            basis_end="2026-06-15",
        )
        messages = validate_share_actions(df, md, pd.Timestamp("2026-06-15"))
        self.assertIn("no Fidelity distribution row", messages[0])
        valuation = compute_valuation(df, md, pd.Timestamp("2026-06-15"))
        self.assertClose(valuation.position_values[0].quantity, 0.338)
        self.assertClose(valuation.position_values[0].priced_quantity, 3.38)
        self.assertClose(valuation.security_value, 676)

    def test_unclassified_nonzero_cash_stops(self):
        df = frame([row("2025-01-02", "MYSTERY CASH EVENT", amount=5)])
        with self.assertRaises(UnsupportedTransactionError):
            external_flows(df, pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-02"))
        with self.assertRaises(UnsupportedTransactionError):
            validate_cash_classification(df, pd.Timestamp("2025-01-02"))

    def test_xirr_bracketed_solver_and_failure(self):
        self.assertClose(
            xirr([-1000, 1100], [pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01")]),
            0.1,
        )
        with self.assertRaises(PerformanceError):
            xirr([1000, 1100], [pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01")])

    def test_twr_removes_external_flow_timing(self):
        df = frame(
            [
                row("2025-01-01", "Electronic Funds Transfer Received (Cash)", amount=100),
                row("2025-01-01", "YOU BOUGHT ABC (ABC) (Cash)", "ABC", price=100, quantity=1, amount=-100),
                row("2025-01-03", "Electronic Funds Transfer Received (Cash)", amount=100),
            ]
        )
        md = market({"ABC": {"2025-01-01": 100, "2025-01-02": 110, "2025-01-03": 110}})
        flows = external_flows(df, pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-03"))
        total, annualized = compute_twr(df, md, pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-03"), flows)
        self.assertClose(total, 0.10)
        self.assertGreater(annualized, total)

    def test_actual_positions_preserve_actual_post_split_quantity(self):
        df = frame(
            [
                row("2025-01-02", "YOU BOUGHT NETFLIX INC (NFLX) (Cash)", "NFLX", quantity=1, amount=-1000),
                row("2025-11-17", "DISTRIBUTION NETFLIX INC (NFLX) (Cash)", "NFLX", type_name="Shares", quantity=9, amount=900),
            ]
        )
        self.assertClose(actual_positions(df, pd.Timestamp("2025-01-02")).get("NFLX"), 1)
        self.assertClose(actual_positions(df, pd.Timestamp("2025-11-17")).get("NFLX"), 10)

    def test_yahoo_downloads_use_disk_cache_across_memory_clears(self):
        clear_yahoo_download_cache()
        data = pd.DataFrame(
            {
                "Close": [100.0, 101.0],
                "Stock Splits": [0.0, 0.0],
            },
            index=pd.to_datetime(["2026-01-02", "2026-01-03"]),
        )

        try:
            with tempfile.TemporaryDirectory() as tmp:
                with patch("performance_core.time.time", side_effect=[0.0, 0.0, 3599.0]):
                    with patch("performance_core.yf.download", return_value=data.copy()) as mock_download:
                        first = download_market_data(
                            ["ABC"], pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-03"), cache_dir=tmp
                        )
                        clear_yahoo_download_cache()
                        second = download_market_data(
                            ["ABC"], pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-03"), cache_dir=tmp
                        )

            self.assertEqual(mock_download.call_count, 1)
            self.assertClose(first.prices.loc[pd.Timestamp("2026-01-03"), "ABC"], 101.0)
            self.assertClose(second.prices.loc[pd.Timestamp("2026-01-03"), "ABC"], 101.0)
        finally:
            clear_yahoo_download_cache()

    def test_yahoo_disk_cache_expires_after_one_hour(self):
        clear_yahoo_download_cache()
        first_data = pd.DataFrame(
            {
                "Close": [100.0, 101.0],
                "Stock Splits": [0.0, 0.0],
            },
            index=pd.to_datetime(["2026-01-02", "2026-01-03"]),
        )
        refreshed_data = pd.DataFrame(
            {
                "Close": [200.0, 201.0],
                "Stock Splits": [0.0, 0.0],
            },
            index=pd.to_datetime(["2026-01-02", "2026-01-03"]),
        )

        try:
            with tempfile.TemporaryDirectory() as tmp:
                with patch("performance_core.time.time", side_effect=[0.0, 0.0, 3599.0, 3600.0, 3600.0]):
                    with patch("performance_core.yf.download", side_effect=[first_data.copy(), refreshed_data.copy()]) as mock_download:
                        first = download_market_data(
                            ["ABC"], pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-03"), cache_dir=tmp
                        )
                        clear_yahoo_download_cache()
                        second = download_market_data(
                            ["ABC"], pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-03"), cache_dir=tmp
                        )
                        clear_yahoo_download_cache()
                        third = download_market_data(
                            ["ABC"], pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-03"), cache_dir=tmp
                        )

            self.assertEqual(mock_download.call_count, 2)
            self.assertClose(first.prices.loc[pd.Timestamp("2026-01-03"), "ABC"], 101.0)
            self.assertClose(second.prices.loc[pd.Timestamp("2026-01-03"), "ABC"], 101.0)
            self.assertClose(third.prices.loc[pd.Timestamp("2026-01-03"), "ABC"], 201.0)
        finally:
            clear_yahoo_download_cache()

    def test_refresh_prices_bypasses_valid_yahoo_disk_cache(self):
        clear_yahoo_download_cache()
        first_data = pd.DataFrame(
            {
                "Close": [100.0, 101.0],
                "Stock Splits": [0.0, 0.0],
            },
            index=pd.to_datetime(["2026-01-02", "2026-01-03"]),
        )
        refreshed_data = pd.DataFrame(
            {
                "Close": [200.0, 201.0],
                "Stock Splits": [0.0, 0.0],
            },
            index=pd.to_datetime(["2026-01-02", "2026-01-03"]),
        )

        try:
            with tempfile.TemporaryDirectory() as tmp:
                with patch("performance_core.time.time", side_effect=[0.0, 0.0, 10.0, 10.0, 20.0]):
                    with patch("performance_core.yf.download", side_effect=[first_data.copy(), refreshed_data.copy()]) as mock_download:
                        first = download_market_data(
                            ["ABC"], pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-03"), cache_dir=tmp
                        )
                        clear_yahoo_download_cache()
                        refreshed = download_market_data(
                            ["ABC"],
                            pd.Timestamp("2026-01-01"),
                            pd.Timestamp("2026-01-03"),
                            refresh_prices=True,
                            cache_dir=tmp,
                        )
                        clear_yahoo_download_cache()
                        cached_refresh = download_market_data(
                            ["ABC"], pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-03"), cache_dir=tmp
                        )

            self.assertEqual(mock_download.call_count, 2)
            self.assertClose(first.prices.loc[pd.Timestamp("2026-01-03"), "ABC"], 101.0)
            self.assertClose(refreshed.prices.loc[pd.Timestamp("2026-01-03"), "ABC"], 201.0)
            self.assertClose(cached_refresh.prices.loc[pd.Timestamp("2026-01-03"), "ABC"], 201.0)
        finally:
            clear_yahoo_download_cache()

    def test_corrupt_yahoo_disk_cache_falls_back_to_download(self):
        clear_yahoo_download_cache()
        first_data = pd.DataFrame(
            {
                "Close": [100.0, 101.0],
                "Stock Splits": [0.0, 0.0],
            },
            index=pd.to_datetime(["2026-01-02", "2026-01-03"]),
        )
        refreshed_data = pd.DataFrame(
            {
                "Close": [200.0, 201.0],
                "Stock Splits": [0.0, 0.0],
            },
            index=pd.to_datetime(["2026-01-02", "2026-01-03"]),
        )

        try:
            with tempfile.TemporaryDirectory() as tmp:
                with patch("performance_core.time.time", side_effect=[0.0, 0.0, 10.0, 10.0]):
                    with patch("performance_core.yf.download", side_effect=[first_data.copy(), refreshed_data.copy()]) as mock_download:
                        first = download_market_data(
                            ["ABC"], pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-03"), cache_dir=tmp
                        )
                        metadata_files = list(Path(tmp).glob("*.json"))
                        self.assertEqual(len(metadata_files), 1)
                        metadata_files[0].write_text("{invalid json", encoding="utf-8")
                        clear_yahoo_download_cache()
                        second = download_market_data(
                            ["ABC"], pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-03"), cache_dir=tmp
                        )

            self.assertEqual(mock_download.call_count, 2)
            self.assertClose(first.prices.loc[pd.Timestamp("2026-01-03"), "ABC"], 101.0)
            self.assertClose(second.prices.loc[pd.Timestamp("2026-01-03"), "ABC"], 201.0)
        finally:
            clear_yahoo_download_cache()

    def test_cached_yahoo_data_is_copied(self):
        clear_yahoo_download_cache()
        data = pd.DataFrame(
            {
                "Close": [100.0],
                "Stock Splits": [0.0],
            },
            index=pd.to_datetime(["2026-01-03"]),
        )

        try:
            with tempfile.TemporaryDirectory() as tmp:
                with patch("performance_core.time.time", side_effect=[0.0, 0.0, 10.0]):
                    with patch("performance_core.yf.download", return_value=data.copy()) as mock_download:
                        first = download_market_data(
                            ["ABC"], pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-03"), cache_dir=tmp
                        )
                        first.prices.loc[pd.Timestamp("2026-01-03"), "ABC"] = 999.0
                        second = download_market_data(
                            ["ABC"], pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-03"), cache_dir=tmp
                        )

            self.assertEqual(mock_download.call_count, 1)
            self.assertClose(second.prices.loc[pd.Timestamp("2026-01-03"), "ABC"], 100.0)
        finally:
            clear_yahoo_download_cache()


if __name__ == "__main__":
    unittest.main()
