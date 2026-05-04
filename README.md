# SmartCleaner

A production-quality data cleaning and quality scoring engine for CSV and Excel files,
built with Python and Pandas.

---

## Tagline

Automated data cleaning with measurable quality improvement — from raw, messy files
to analysis-ready datasets in seconds.

---

## Problem Statement

Raw data files from the real world are almost never clean. Data collected from forms,
databases, scraping tools, or manual entry consistently suffers from the same set of
problems: column headers with inconsistent casing and extra whitespace, rows that are
exact duplicates, string values in mixed case, numeric columns with missing values,
text columns with missing values, and entire columns that contain no data at all.

Before any analysis, reporting, machine learning, or data pipeline can run reliably,
this data must be cleaned. Doing this manually is slow, error-prone, and not
reproducible. Doing it with one-off scripts produces code that cannot be tested,
audited, or reused.

There was no simple, self-contained tool that a data engineer or analyst could run
against any CSV or Excel file, get a cleaned output, and receive a structured report
explaining exactly what changed and by how much the data quality improved.

SmartCleaner was built to fill that gap.

---

## Solution Overview

SmartCleaner is a modular data cleaning pipeline with three interfaces: a command-line
tool, a browser-based web application, and a Python API that any other script or
system can call directly.

It accepts a raw CSV or Excel file, runs it through a five-step cleaning pipeline,
calculates a data quality score before and after cleaning, and returns both the cleaned
data and a structured audit report. The report records every change made: how many
duplicates were removed, how many columns were dropped, which columns had missing
values filled and with what values, and what the quality score improvement was.

The entire system is covered by 59 unit tests. The tests caught two real bugs in the
production code during development, before the code was ever run against real data.

---

## System Architecture

SmartCleaner follows a strict layered architecture. Each layer has one responsibility
and communicates only with the layer directly below it.

```
Layer 1 — Interfaces
    app.py            Streamlit web UI. Handles file upload, rendering, download.
    main.py           Command-line entry point. Handles file loading and saving.

Layer 2 — Pipeline Orchestration
    src/cleaner.py    Calls each cleaning function in order. Collects the report.
                      Calls the scoring module before and after cleaning.

Layer 3 — Core Cleaning Functions
    src/cleaner.py    Five independent cleaning functions. Each does one thing.
                      None of them know about files, the web, or the score.

Layer 4 — Quality Scoring
    src/quality_score.py    Measures data quality metrics. Calculates the score.
                            Has no knowledge of the cleaning functions.

Layer 5 — Tests
    tests/conftest.py       Shared fixtures used across all test classes.
    tests/test_cleaner.py   59 unit tests. One test class per function.
```

Each layer is independently testable. Changing the web interface does not require
touching the cleaning logic. Changing the scoring formula does not require touching
the pipeline. This is the core design principle that makes the system maintainable.

---

## Tech Stack

| Component          | Technology   | Version  | Purpose                                    |
|--------------------|--------------|----------|--------------------------------------------|
| Language           | Python       | 3.10+    | Core language                              |
| Data processing    | Pandas       | 2.0+     | DataFrame operations, cleaning, export     |
| Web interface      | Streamlit    | 1.35+    | Browser-based UI                           |
| Excel support      | openpyxl     | 3.1+     | Reading .xlsx and .xls files via Pandas    |
| Testing framework  | pytest       | 9.0+     | Unit test runner and assertion library     |

No machine learning libraries. No database. No external APIs. The only runtime
dependency for the core cleaning and scoring engine is Pandas.

---

## Features

**Data Cleaning**
- Standardises column names to snake_case by stripping whitespace, lowercasing,
  and replacing spaces with underscores
- Removes columns where every value is null
- Removes fully blank rows
- Removes exact duplicate rows, preserving the first occurrence
- Strips whitespace and applies title case to all string values
- Fills missing text values with the string "Unknown"
- Fills missing numeric values with the column median, which is resistant to outliers

**Quality Scoring**
- Calculates a score from 0 to 100 before and after cleaning
- Score is based on three weighted penalty factors: missing values (40 points),
  duplicate rows (30 points), and empty columns (30 points)
- Returns a human-readable label: Excellent, Good, Fair, Poor, or Critical
- Reports the exact improvement in score points

**Validation Report**
- Structured dictionary returned alongside the cleaned DataFrame
- Records original and final row and column counts, empty rows dropped,
  empty columns dropped, duplicate rows removed, text columns normalised,
  per-column missing value fill details, and before and after quality scores

**Interfaces**
- Command-line tool that accepts any CSV or Excel file as an argument
- Streamlit web app with file upload, side-by-side before and after table views,
  colour-coded score cards, metrics comparison table, and one-click CSV download
- Python API: call clean_dataset(df) from any script and receive (DataFrame, report)

**Testing**
- 59 unit tests across 9 test classes
- Tests for every public function in both cleaner.py and quality_score.py
- Shared fixtures in conftest.py
- Zero external file reads during testing

---

## Sample Output

**Command-line output:**

```
Loading CSV  : data/sample_dirty.csv
Shape: 11 rows x 7 columns

Starting SmartCleaner pipeline...
  Input  -> 11 rows x 7 columns
  Quality score before cleaning: 88.96/100

  Step 1/5 - Column names cleaned.
  Step 2/5 - Empty columns removed.
  [pipeline] Dropped 1 fully-empty row(s).
  [remove_duplicates] Removed 1 duplicate row(s).
  Step 3/5 - Duplicate rows removed.
  Step 4/5 - Text values normalised.
  Step 5/5 - Missing values filled.

Cleaning complete!
  Output -> 9 rows x 7 columns
  Quality score after cleaning: 96.67/100
  Improvement: +7.71 points

--- Quality Score ---
  Before :  88.96/100  (Good)
  After  :  96.67/100  (Excellent)
  Gain   : +7.71 points

--- Validation Report ---
  Rows            : 11 -> 9
  Columns         : 7 -> 7
  Empty rows dropped     : 1
  Empty columns dropped  : 0
  Duplicate rows removed : 1
  Missing values filled:
    - last_name: 1 gap(s) -> 'Unknown'
    - age: 2 gap(s) -> 36.0
    - department: 1 gap(s) -> 'Unknown'
    - salary: 1 gap(s) -> 64500.0
    - notes: 4 gap(s) -> 'Unknown'
```

**Validation report dictionary:**

```python
{
    "original_rows": 11,
    "original_columns": 7,
    "empty_rows_dropped": 1,
    "empty_columns_dropped": 0,
    "duplicate_rows_removed": 1,
    "text_columns_normalised": ["first_name", "last_name", "department", "notes"],
    "missing_filled": {
        "age":    {"count": 2, "fill_value": 36.0},
        "salary": {"count": 1, "fill_value": 64500.0}
    },
    "quality_score_before": 88.96,
    "quality_score_after":  96.67,
    "quality_improvement":  7.71,
    "final_rows": 9,
    "final_columns": 7
}
```

---

## Project Structure

```
SmartCleaner/
├── src/
│   ├── __init__.py           Makes src/ a Python package
│   ├── cleaner.py            Five cleaning functions and the pipeline orchestrator
│   └── quality_score.py      Quality metrics, scoring formula, and score labels
├── tests/
│   ├── __init__.py
│   ├── conftest.py           Shared pytest fixtures
│   └── test_cleaner.py       59 unit tests across 9 test classes
├── data/
│   ├── sample_dirty.csv      Raw input sample with deliberate quality problems
│   └── sample_cleaned.csv    Output produced by running main.py
├── app.py                    Streamlit web interface
├── main.py                   Command-line entry point
├── requirements.txt          Python dependencies
└── README.md
```

---

## Installation and Setup

**Step 1 — Clone the repository**

```bash
git clone https://github.com/PraveenMinindu/smartcleaner.git
cd smartcleaner
```

**Step 2 — Create a virtual environment**

```bash
python -m venv venv
```

**Step 3 — Activate the virtual environment**

On Windows:
```bash
venv\Scripts\activate
```

On macOS and Linux:
```bash
source venv/bin/activate
```

**Step 4 — Install dependencies**

```bash
pip install -r requirements.txt
```

---

## Usage and How to Run

**Command-line pipeline**

Run on the default sample file:
```bash
python main.py
```

Run on a specific CSV file:
```bash
python main.py data/yourfile.csv
```

Run on an Excel file:
```bash
python main.py data/yourfile.xlsx
```

The cleaned file is saved in the same folder as the input, with a "cleaned_" prefix.
For example, data/sales.csv produces data/cleaned_sales.csv.

**Web application**

```bash
python -m streamlit run app.py
```

Open http://localhost:8501 in your browser. Upload any CSV or Excel file. The app
cleans it immediately, displays the before and after quality scores, shows both tables
side by side, and provides a download button for the cleaned file.

**Python API**

```python
import pandas as pd
from src.cleaner import clean_dataset

df = pd.read_csv("data/yourfile.csv")
df_clean, report = clean_dataset(df)

print(report["quality_score_before"])
print(report["quality_score_after"])
print(report["quality_improvement"])

df_clean.to_csv("output.csv", index=False)
```

**Run tests**

```bash
python -m pytest tests/ -v
```

Run a single test class:
```bash
python -m pytest tests/ -v -k "TestFillMissingValues"
```

Stop on first failure:
```bash
python -m pytest tests/ -x
```

---

## How It Works

**Step 1 — clean_column_names**
Every column header is stripped of leading and trailing whitespace, converted to
lowercase, and has spaces replaced with underscores. The result is consistent
snake_case headers that are safe to reference in code. This step runs first because
all subsequent steps reference columns by name.

**Step 2 — remove_empty_columns**
Any column where every single value is null is dropped. These columns carry no
information and would interfere with downstream operations that iterate over all
columns. This step runs before filling so there is no wasted work filling a column
that will be dropped.

**Step 2b — drop empty rows**
Rows where every value is null are removed. CSV files exported from Excel and other
tools frequently include blank rows between sections of data. These are not real
records and must be removed before deduplication.

**Step 3 — remove_duplicates**
Rows that are identical across every column are removed. Only the first occurrence is
kept. Deduplication runs before filling because filling changes values. If you fill
two rows that are both null in the same column, they may no longer be identical and
the duplicate would be missed.

**Step 4 — clean_text_values**
Every string column has its values stripped of surrounding whitespace and converted to
title case. Numeric columns are not touched. This step runs after deduplication so
that casing normalisation does not accidentally merge records that were intentionally
different in the source.

**Step 5 — fill_missing_values**
Remaining null values are filled. Text columns receive the string "Unknown". Numeric
columns receive the column median. The median is used instead of the mean because it
is not distorted by outliers. For example, if a salary column contains values of
40000, 45000, and 1000000, the mean is 361666 but the median is 45000, which is
actually representative of the real distribution.

**Quality scoring**
The score formula starts at 100 and subtracts three penalty terms. Missing values
carry a weight of 40 points, duplicate rows carry 30 points, and empty columns carry
30 points. Each penalty is proportional. A 50 percent missing rate subtracts 20 points
from the missing penalty term, not the full 40. The score is clamped to a minimum of
0 and a maximum of 100. The score is calculated once on the raw DataFrame before any
cleaning step runs, and once on the fully cleaned DataFrame after all steps complete.

---

## Quality Score Formula

```
score = 100
      - (missing_percent   / 100) * 40
      - (duplicate_percent / 100) * 30
      - (empty_col_percent / 100) * 30

Where:
  missing_percent   = (total NaN cells / total cells) * 100
  duplicate_percent = (duplicate row count / total rows) * 100
  empty_col_percent = (empty column count / total columns) * 100

Score is clamped to the range [0.0, 100.0]
```

Score interpretation:

| Range    | Label     |
|----------|-----------|
| 90 - 100 | Excellent |
| 75 - 89  | Good      |
| 50 - 74  | Fair      |
| 25 - 49  | Poor      |
| 0  - 24  | Critical  |

---

## Test Coverage

| Test Class                   | Function Tested              | Tests |
|------------------------------|------------------------------|-------|
| TestCleanColumnNames         | clean_column_names()         | 6     |
| TestRemoveEmptyColumns       | remove_empty_columns()       | 4     |
| TestRemoveDuplicates         | remove_duplicates()          | 5     |
| TestCleanTextValues          | clean_text_values()          | 5     |
| TestFillMissingValues        | fill_missing_values()        | 6     |
| TestCleanDataset             | clean_dataset()              | 9     |
| TestCalculateQualityMetrics  | calculate_quality_metrics()  | 7     |
| TestCalculateQualityScore    | calculate_quality_score()    | 7     |
| TestScoreLabel               | score_label()                | 10    |
| Total                        |                              | 59    |

The tests caught two real bugs during development:

Bug 1: fill_missing_values was using dtype == "object" to detect text columns. On
Python 3.11 with Pandas 2.x, string columns use StringDtype rather than object dtype.
The check silently skipped all text columns so no text nulls were ever filled. The
tests test_fills_missing_text_with_unknown and test_no_nan_remains_after_filling both
failed immediately, identifying the exact function and line.

Bug 2: select_dtypes(include=["object", "str"]) raised a TypeError on Python 3.10.
The integration test for the full pipeline caught this before any real data was run.

---

## Limitations

- The cleaning logic is general-purpose. It does not understand domain-specific data
  shapes. A column containing a block of text that mixes director and actor names is
  cleaned as a single string but not parsed into separate fields. Domain-specific
  transformations require custom extensions.

- Missing value fill strategies are fixed. Text always fills with "Unknown" and
  numbers always fill with the column median. There is no per-column configuration.
  A status column might be better filled with "Pending" than with "Unknown".

- Duplicate detection is exact. Two rows must be identical in every column to be
  considered duplicates. Near-duplicate detection where rows are similar but not
  identical due to minor variation is not supported.

- The quality score weights are fixed constants. They are not configurable at runtime
  without editing the source code. Different use cases may call for different weight
  distributions across the three penalty factors.

- File encoding is assumed to be UTF-8 for CSV files. Files in other encodings such
  as latin-1 or windows-1252 may produce errors or garbled characters without manual
  encoding specification.

- The web application runs locally only. There is no authentication, no multi-user
  support, and no persistent storage. Each session is independent and stateless.

---

## Future Improvements

**Version 2 — Configurable Cleaning**
- YAML configuration file for per-column fill values, custom cleaning rules, and
  adjustable score weights
- Column-level data type inference that detects and converts columns that look like
  dates, booleans, or categories but were read as plain strings
- Near-duplicate detection using string similarity thresholds
- HTML cleaning report exportable alongside the cleaned CSV

**Version 3 — API and Deployment**
- FastAPI REST endpoint that accepts a file upload and returns the cleaned file and
  quality report as JSON
- Docker container for one-command deployment with no local Python setup required
- Support for JSON and Parquet input and output formats
- Configurable encoding detection for non-UTF-8 CSV files

**Version 4 — Scheduling and Monitoring**
- Directory watcher that automatically processes new files as they arrive
- Structured JSON log for every cleaning run recording filename, timestamp,
  row counts, and quality scores
- Historical dashboard showing quality trends across multiple cleaning runs
- Alerting when a file quality score falls below a configured threshold

---

## Contributing

Contributions are welcome. Please follow these steps:

1. Fork the repository on GitHub.
2. Create a new branch for your feature or fix.
3. Make your changes.
4. Add or update unit tests in tests/test_cleaner.py to cover your changes.
5. Run the full test suite and confirm all 59 tests pass.
6. Commit with a clear message describing what changed and why.
7. Open a pull request against the main branch with a description of the change.

All pull requests must pass the full test suite before they will be reviewed.
New features must include corresponding unit tests. Changes to the scoring formula
must include an explanation of why the new weights or logic are more appropriate
than the existing ones.

---

## Author

Praveen Minindu

Built with Python, Pandas, pytest, and Streamlit
