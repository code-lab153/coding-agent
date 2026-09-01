"""Phase 2 acceptance tests for product and extended summary analysis."""

from pathlib import Path

import csv_analyzer


SAMPLE_CSV = Path(__file__).parents[1] / "data" / "sales.csv"


def sample_records() -> list[csv_analyzer.SalesRecord]:
    return csv_analyzer.load_sales_csv(SAMPLE_CSV).records


def test_summary_reports_top_product() -> None:
    summary = csv_analyzer.summarize_sales(sample_records())

    assert summary.top_product == "Laptop"


def test_summary_reports_maximum_individual_sale() -> None:
    summary = csv_analyzer.summarize_sales(sample_records())

    assert summary.max_sale == 1500.0


def test_sales_by_product_returns_expected_totals() -> None:
    sales_by_product = getattr(csv_analyzer, "sales_by_product")

    assert sales_by_product(sample_records()) == [
        ("Laptop", 4600.0),
        ("Monitor", 2500.0),
        ("Headphones", 970.0),
        ("Keyboard", 780.0),
        ("Mouse", 450.0),
    ]


def test_product_aggregation_uses_deterministic_name_tie_breaker() -> None:
    sales_by_product = getattr(csv_analyzer, "sales_by_product")
    records = [
        csv_analyzer.SalesRecord("2026-02-01", "East", "Beta", 100.0),
        csv_analyzer.SalesRecord("2026-02-02", "West", "Alpha", 100.0),
    ]

    assert sales_by_product(records) == [("Alpha", 100.0), ("Beta", 100.0)]


def test_top_product_uses_the_product_aggregation_tie_breaker() -> None:
    records = [
        csv_analyzer.SalesRecord("2026-02-01", "East", "Beta", 100.0),
        csv_analyzer.SalesRecord("2026-02-02", "West", "Alpha", 100.0),
    ]

    assert csv_analyzer.summarize_sales(records).top_product == "Alpha"


def test_empty_summary_has_empty_extended_values() -> None:
    summary = csv_analyzer.summarize_sales([])

    assert summary.top_product is None
    assert summary.max_sale is None
