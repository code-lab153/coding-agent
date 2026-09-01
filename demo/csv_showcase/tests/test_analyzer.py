"""Acceptance tests for deterministic public analysis functions."""

from pathlib import Path

import pytest

from csv_analyzer import SalesRecord, load_sales_csv, sales_by_region, summarize_sales


SAMPLE_CSV = Path(__file__).parents[1] / "data" / "sales.csv"


def sample_records() -> list[SalesRecord]:
    return load_sales_csv(SAMPLE_CSV).records


def test_summary_row_count() -> None:
    assert summarize_sales(sample_records()).row_count == 20


def test_summary_total_sales() -> None:
    assert summarize_sales(sample_records()).total_sales == pytest.approx(9300.0)


def test_summary_average_sales() -> None:
    assert summarize_sales(sample_records()).average_sales == pytest.approx(465.0)


def test_empty_summary_has_deterministic_zero_values() -> None:
    summary = summarize_sales([])

    assert summary.row_count == 0
    assert summary.total_sales == 0.0
    assert summary.average_sales == 0.0
    assert summary.top_region is None


def test_sales_by_region_returns_expected_totals() -> None:
    assert sales_by_region(sample_records()) == [
        ("East", 3000.0),
        ("West", 2400.0),
        ("South", 2050.0),
        ("North", 1850.0),
    ]


def test_summary_reports_the_region_with_highest_total() -> None:
    assert summarize_sales(sample_records()).top_region == "East"


def test_equal_region_totals_use_deterministic_name_tie_breaker() -> None:
    records = [
        SalesRecord("2026-02-01", "Beta", "Mouse", 100.0),
        SalesRecord("2026-02-02", "Alpha", "Keyboard", 100.0),
    ]

    assert sales_by_region(records) == [("Alpha", 100.0), ("Beta", 100.0)]
    assert summarize_sales(records).top_region == "Alpha"
