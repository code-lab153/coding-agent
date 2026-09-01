"""Acceptance tests for public CSV loading and validation behavior."""

from pathlib import Path

import pytest

from csv_analyzer import InvalidCSVError, load_sales_csv


SAMPLE_CSV = Path(__file__).parents[1] / "data" / "sales.csv"


def test_sample_csv_loads_all_expected_records() -> None:
    result = load_sales_csv(SAMPLE_CSV)

    assert len(result.records) == 20
    assert result.skipped_rows == 0


def test_sales_record_fields_are_parsed_and_sales_is_float() -> None:
    record = load_sales_csv(SAMPLE_CSV).records[0]

    assert record.date == "2026-01-01"
    assert record.region == "East"
    assert record.product == "Laptop"
    assert record.sales == 1500.0
    assert isinstance(record.sales, float)


def test_missing_required_column_raises_invalid_csv_error(tmp_path: Path) -> None:
    path = tmp_path / "missing-sales.csv"
    path.write_text(
        "date,region,product\n2026-01-01,East,Laptop\n",
        encoding="utf-8",
    )

    with pytest.raises(InvalidCSVError):
        load_sales_csv(path)


def test_invalid_numeric_sales_row_is_skipped_and_reported(tmp_path: Path) -> None:
    path = tmp_path / "invalid-number.csv"
    path.write_text(
        "date,region,product,sales\n"
        "2026-01-01,East,Laptop,100\n"
        "2026-01-02,West,Monitor,not-a-number\n",
        encoding="utf-8",
    )

    result = load_sales_csv(path)

    assert [record.product for record in result.records] == ["Laptop"]
    assert result.records[0].sales == 100.0
    assert result.skipped_rows == 1


def test_missing_required_values_are_skipped_once_per_row(tmp_path: Path) -> None:
    path = tmp_path / "missing-values.csv"
    path.write_text(
        "date,region,product,sales\n"
        "2026-01-01,East,Laptop,100\n"
        "2026-01-02,,Monitor,200\n"
        ",West,Mouse,50\n"
        "2026-01-04,North,,75\n",
        encoding="utf-8",
    )

    result = load_sales_csv(path)

    assert len(result.records) == 1
    assert result.records[0].product == "Laptop"
    assert result.skipped_rows == 3
