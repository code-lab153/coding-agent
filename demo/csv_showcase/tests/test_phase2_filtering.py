"""Phase 2 acceptance tests for composable product and sales-range filters."""

from pathlib import Path

import csv_analyzer


SAMPLE_CSV = Path(__file__).parents[1] / "data" / "sales.csv"


def sample_records() -> list[csv_analyzer.SalesRecord]:
    return csv_analyzer.load_sales_csv(SAMPLE_CSV).records


def test_filter_by_product_returns_only_requested_product_in_original_order() -> None:
    filter_by_product = getattr(csv_analyzer, "filter_by_product")

    filtered = filter_by_product(sample_records(), "Laptop")

    assert [record.date for record in filtered] == [
        "2026-01-01",
        "2026-01-06",
        "2026-01-11",
        "2026-01-16",
    ]
    assert {record.product for record in filtered} == {"Laptop"}


def test_filter_by_product_is_case_insensitive() -> None:
    filter_by_product = getattr(csv_analyzer, "filter_by_product")

    assert filter_by_product(sample_records(), "laptop") == filter_by_product(
        sample_records(), "LAPTOP"
    )


def test_minimum_sales_filter_is_inclusive() -> None:
    filter_sales = getattr(csv_analyzer, "filter_sales")

    filtered = filter_sales(sample_records(), min_sales=1000.0)

    assert [record.sales for record in filtered] == [1500.0, 1200.0, 1000.0]


def test_maximum_sales_filter_is_inclusive() -> None:
    filter_sales = getattr(csv_analyzer, "filter_sales")

    filtered = filter_sales(sample_records(), max_sales=100.0)

    assert [record.sales for record in filtered] == [100.0, 80.0]


def test_both_sales_range_boundaries_are_inclusive() -> None:
    filter_sales = getattr(csv_analyzer, "filter_sales")

    filtered = filter_sales(sample_records(), min_sales=500.0, max_sales=1000.0)

    assert [record.date for record in filtered] == [
        "2026-01-02",
        "2026-01-07",
        "2026-01-11",
        "2026-01-12",
        "2026-01-16",
        "2026-01-17",
    ]


def test_region_and_product_filters_compose_with_and_semantics() -> None:
    filter_sales = getattr(csv_analyzer, "filter_sales")

    filtered = filter_sales(sample_records(), region="East", product="Laptop")

    assert [(record.region, record.product) for record in filtered] == [
        ("East", "Laptop")
    ]


def test_region_product_and_range_filters_all_compose() -> None:
    filter_sales = getattr(csv_analyzer, "filter_sales")

    filtered = filter_sales(
        sample_records(),
        region="east",
        product="laptop",
        min_sales=1500.0,
        max_sales=1500.0,
    )

    assert [record.date for record in filtered] == ["2026-01-01"]


def test_combined_filters_return_empty_list_when_nothing_matches() -> None:
    filter_sales = getattr(csv_analyzer, "filter_sales")

    assert filter_sales(
        sample_records(), region="East", product="Laptop", max_sales=1000.0
    ) == []
