# CSV Data Analyzer

Build a desktop CSV data-analysis application as a Python package named
`csv_analyzer`. The package does not exist yet. Design a small, focused source
structure and implement the complete application using only the Python standard
library.

The finished application must launch with either command:

```text
python -m csv_analyzer
python -m csv_analyzer data/sales.csv
```

With no path, open an empty application ready to load a file. With a path, open
the GUI and automatically load that CSV after startup. Importing `csv_analyzer`
or `csv_analyzer.__main__` must not create a Tk root or start `mainloop`; only an
explicit call to `main()` or module execution may launch the GUI.

## Required public API

These names must be importable directly from `csv_analyzer`:

```python
from csv_analyzer import (
    InvalidCSVError,
    LoadResult,
    SalesRecord,
    SalesSummary,
    export_sales_csv,
    filter_by_product,
    filter_by_region,
    filter_sales,
    load_sales_csv,
    main,
    sales_by_product,
    sales_by_region,
    summarize_sales,
)
```

Public classes and functions must have type annotations. Private helpers and
the internal module layout are implementation choices.

### Data objects

`SalesRecord` exposes:

- `date: str`
- `region: str`
- `product: str`
- `sales: float`

`LoadResult` exposes:

- `records: list[SalesRecord]`
- `skipped_rows: int`

`SalesSummary` exposes:

- `row_count: int`
- `total_sales: float`
- `average_sales: float`
- `top_region: str | None`
- `top_product: str | None`
- `max_sale: float | None`

`InvalidCSVError` is raised when a CSV file lacks one or more required columns.

## CSV loading and validation

Implement:

```python
load_sales_csv(path: str | pathlib.Path) -> LoadResult
```

The required header names are exactly:

```text
date,region,product,sales
```

Additional columns may be ignored. Trim surrounding whitespace from required
field values. Each valid row becomes a `SalesRecord`, and `sales` is stored as
a `float`.

Use these deterministic malformed-data rules:

- Missing required header columns raise `InvalidCSVError`.
- A row with an empty required value is skipped.
- A row whose sales value is not a finite numeric value is skipped.
- Valid rows remain available even when other rows are skipped.
- Every skipped data row increments `LoadResult.skipped_rows` exactly once.
- A header-only valid CSV produces no records and zero skipped rows.

## Analysis

Implement:

```python
summarize_sales(records: list[SalesRecord]) -> SalesSummary
sales_by_region(records: list[SalesRecord]) -> list[tuple[str, float]]
sales_by_product(records: list[SalesRecord]) -> list[tuple[str, float]]
```

The summary contains row count, total sales, average sales, the region with the
largest total sales, the product with the largest total sales, and the maximum
individual sale. Empty input produces:

```text
row_count = 0
total_sales = 0.0
average_sales = 0.0
top_region = None
top_product = None
max_sale = None
```

`sales_by_region` groups records by their stored region names and returns
`(region, total)` pairs ordered by descending total. Equal totals use the region
name in case-insensitive ascending order, then the original region name as a
stable final tie-breaker. `top_region` follows the same ordering rule.

`sales_by_product` follows the same behavior for stored product names. It
returns `(product, total)` pairs ordered by descending total. Equal totals use
the product name in case-insensitive ascending order, then the original product
name as a stable final tie-breaker. `top_product` follows this same rule.

`max_sale` is the largest individual `SalesRecord.sales` value, not an
aggregate.

## Filtering

Implement:

```python
filter_by_region(
    records: list[SalesRecord], region: str
) -> list[SalesRecord]
filter_by_product(
    records: list[SalesRecord], product: str
) -> list[SalesRecord]
filter_sales(
    records: list[SalesRecord],
    *,
    region: str | None = None,
    product: str | None = None,
    min_sales: float | None = None,
    max_sales: float | None = None,
) -> list[SalesRecord]
```

Region matching is case-insensitive. Preserve the original record order and
return an empty list when no region matches.

Product matching is also case-insensitive. `filter_by_product` preserves the
original record order and returns an empty list when no product matches.

`filter_sales` composes all supplied restrictions with AND semantics:

- `region=None` means any region.
- `product=None` means any product.
- `min_sales=None` means there is no lower bound.
- `max_sales=None` means there is no upper bound.
- A minimum is inclusive: `record.sales >= min_sales`.
- A maximum is inclusive: `record.sales <= max_sales`.
- If both bounds are supplied, both must be satisfied.
- If the lower bound is greater than the upper bound, return an empty list.
- The original input ordering is always preserved.

The GUI maps its **All** region/product choices and blank range entries to
`None`. Invalid non-numeric range input should produce a clear status message
without crashing or changing the currently visible records.

## CSV export

Implement a testable core helper:

```python
export_sales_csv(
    records: list[SalesRecord], path: str | pathlib.Path
) -> None
```

Write a UTF-8 CSV containing exactly this header:

```text
date,region,product,sales
```

Export records in their existing order and write sales as a numeric value that
can be parsed back as `float`. The GUI exports the currently visible records to
a path selected with a normal save-file dialog. It must never automatically
overwrite the original input file.

## Tkinter desktop interface

Create a simple single-window application using `tkinter` and `tkinter.ttk`.
Keep loading/analysis logic separate from presentation. Do not use matplotlib.

The window must provide:

1. A visible `CSV Data Analyzer` header/title.
2. A file section showing the current path and a **Load CSV** button using a
   standard file dialog.
3. A summary area showing **Rows**, **Total Sales**, **Average Sales**,
   **Top Region**, **Top Product**, and **Max Sale**.
4. A filter area with a Region `ttk.Combobox`, Product `ttk.Combobox`, Min
   Sales entry, Max Sales entry, **Apply Filters**, **Reset**, and
   **Export Filtered CSV**.
5. A `ttk.Treeview` with Date, Region, Product, and Sales columns.
6. A chart-mode control with **By Region** and **By Product**, plus a
   `tkinter.Canvas` horizontal bar chart for the selected aggregation.
7. A human-readable status bar.

The chart draws one horizontal bar per visible region or product, scaled
relative to the largest visible total. Label every bar with its group name and
numeric total. Redraw it whenever a file is loaded, filters are applied, Reset
is used, or chart mode changes. Continue using `tkinter.Canvas`; do not add a
plotting dependency.

Applying any combination of region, product, minimum sales, and maximum sales
must recompute the table, all six summary values, and the selected chart from
the filtered records. Reset clears all filter controls and restores the
complete loaded data. Both Combobox choice lists should be deterministic and
include **All**.

Use clear status messages such as:

```text
Loaded 20 valid rows from sales.csv
Loaded 20 valid rows, skipped 2 invalid rows from sales.csv
Showing 5 rows for region East
Showing all 20 rows
Invalid CSV: missing required column "sales"
```

Handle a user-selected invalid file without crashing the GUI. The visible GUI
behavior will be checked manually; automated tests intentionally avoid display
automation.

Run the full suite with `python -m pytest` before declaring completion.
