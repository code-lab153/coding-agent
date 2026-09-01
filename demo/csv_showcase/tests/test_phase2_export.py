"""Phase 2 acceptance tests for deterministic filtered-record CSV export."""

import csv
from pathlib import Path

import csv_analyzer


SAMPLE_CSV = Path(__file__).parents[1] / "data" / "sales.csv"


def sample_records() -> list[csv_analyzer.SalesRecord]:
    return csv_analyzer.load_sales_csv(SAMPLE_CSV).records


def test_export_writes_the_supplied_filtered_records(tmp_path: Path) -> None:
    export_sales_csv = getattr(csv_analyzer, "export_sales_csv")
    records = csv_analyzer.filter_by_region(sample_records(), "East")
    destination = tmp_path / "east.csv"

    export_sales_csv(records, destination)

    with destination.open("r", encoding="utf-8", newline="") as stream:
        exported = list(csv.DictReader(stream))
    assert len(exported) == 5
    assert {row["region"] for row in exported} == {"East"}
    assert [row["product"] for row in exported] == [
        "Laptop",
        "Monitor",
        "Keyboard",
        "Mouse",
        "Headphones",
    ]


def test_export_uses_the_required_header(tmp_path: Path) -> None:
    export_sales_csv = getattr(csv_analyzer, "export_sales_csv")
    destination = tmp_path / "header.csv"

    export_sales_csv([], destination)

    assert destination.read_text(encoding="utf-8").splitlines()[0] == (
        "date,region,product,sales"
    )


def test_export_preserves_the_supplied_record_order(tmp_path: Path) -> None:
    export_sales_csv = getattr(csv_analyzer, "export_sales_csv")
    records = sample_records()
    selected = [records[6], records[0], records[14]]
    destination = tmp_path / "ordered.csv"

    export_sales_csv(selected, destination)

    with destination.open("r", encoding="utf-8", newline="") as stream:
        exported = list(csv.DictReader(stream))
    assert [row["date"] for row in exported] == [
        selected[0].date,
        selected[1].date,
        selected[2].date,
    ]
    assert [float(row["sales"]) for row in exported] == [
        selected[0].sales,
        selected[1].sales,
        selected[2].sales,
    ]
