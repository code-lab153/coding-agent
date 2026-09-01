"""Acceptance tests for public region filtering behavior."""

from pathlib import Path

from csv_analyzer import SalesRecord, filter_by_region, load_sales_csv


SAMPLE_CSV = Path(__file__).parents[1] / "data" / "sales.csv"


def sample_records() -> list[SalesRecord]:
    return load_sales_csv(SAMPLE_CSV).records


def test_filter_by_region_returns_only_requested_region() -> None:
    filtered = filter_by_region(sample_records(), "East")

    assert len(filtered) == 5
    assert {record.region for record in filtered} == {"East"}


def test_filter_by_region_is_case_insensitive() -> None:
    expected = filter_by_region(sample_records(), "East")

    assert filter_by_region(sample_records(), "east") == expected
    assert filter_by_region(sample_records(), "EAST") == expected


def test_filter_by_region_preserves_original_record_order() -> None:
    records = sample_records()
    filtered = filter_by_region(records, "West")

    assert [record.date for record in filtered] == [
        "2026-01-06",
        "2026-01-07",
        "2026-01-08",
        "2026-01-09",
        "2026-01-10",
    ]


def test_filter_by_region_returns_empty_list_for_no_match() -> None:
    assert filter_by_region(sample_records(), "Central") == []
