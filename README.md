# SmartCleaner

A production-quality data cleaning, quality scoring, and security-tested data engineering platform built with Python.

---

## Tagline

Automated data cleaning with measurable quality improvement, machine learning imputation, outlier detection, schema drift protection, and a REST API — from raw messy files to analysis-ready datasets in seconds.

---

## Problem Statement

Raw data files from the real world are almost never clean. Data collected from forms, databases, scraping tools, or manual entry consistently suffers from the same set of problems: column headers with inconsistent casing and extra whitespace, rows that are exact duplicates, string values in mixed case, numeric columns with missing values, text columns with missing values, and entire columns that contain no data at all.

Before any analysis, reporting, machine learning, or data pipeline can run reliably, this data must be cleaned. Doing this manually is slow, error-prone, and not reproducible. Doing it with one-off scripts produces code that cannot be tested, audited, or reused.

SmartCleaner was built to solve this at a production level — not just cleaning data but measuring quality, detecting structural changes, catching outliers with machine learning, filling gaps intelligently, and exposing everything through a secure REST API.

---

## Solution Overview

SmartCleaner is a modular seven-step data cleaning platform with four interfaces: a command-line tool, a browser-based web application, a REST API, and a direct Python API. It accepts raw CSV or Excel files, runs them through the full pipeline, calculates a quality score before and after cleaning, checks for schema drift, and returns both the cleaned data and a structured audit report.

The system is covered by 151 unit and integration tests. The test suite caught multiple real bugs during development before the code was ever run against real data. The codebase passes a full Bandit static security scan with zero issues and a full OWASP ZAP penetration test with zero medium or high severity findings.

---

## System Architecture

SmartCleaner follows a strict layered architecture where each layer has one responsibility and communicates only with the layer directly below it.

```
Layer 1 — Interfaces
    app.py              Streamlit web UI with drift warnings, score cards, download
    main.py             Command-line entry point
    api.py              FastAPI REST server with five endpoints

Layer 2 — Pipeline Orchestration
    src/cleaner.py      Calls each step in order, collects the report,
                        calls scoring before and after

Layer 3 — Core Cleaning Functions
    src/cleaner.py      Seven independent cleaning functions, each doing one thing

Layer 4 — Intelligence Modules
    src/outlier_detection.py    Isolation Forest outlier detection
    src/knn_imputer.py          KNN imputation for missing numeric values
    src/schema_drift.py         Schema drift detection and reference management

Layer 5 — Quality and Config
    src/quality_score.py        Quality metrics, scoring formula, score labels
    src/config.py               Environment variable loading via .env file

Layer 6 — Tests
    tests/conftest.py           Shared pytest fixtures
    tests/test_cleaner.py       58 tests covering all cleaning and scoring functions
    tests/test_outlier_detection.py   18 tests
    tests/test_knn_imputer.py         13 tests
    tests/test_schema_drift.py        26 tests
    tests/test_api.py                 27 tests
    tests/test_config_and_limits.py   17 tests
    tests/test_security.py            8 automated Bandit security tests
```

---
<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/1e751dab-cf5e-4ca2-92a4-3884fadd3a33" />

## Tech Stack

Language: Python 3.10 or higher

Data processing: Pandas 2.0 or higher for all DataFrame operations, cleaning, and export

Machine learning: scikit-learn 1.3 or higher for Isolation Forest outlier detection and KNN imputation

REST API: FastAPI 0.110 or higher with Uvicorn 0.29 or higher as the ASGI server

Web interface: Streamlit 1.35 or higher

Excel support: openpyxl 3.1 or higher for reading and writing xlsx and xls files

Configuration: python-dotenv for loading settings from a dot env file

Testing: pytest 9.0 or higher as the test runner and assertion library

Security static analysis: Bandit for scanning Python source code

Security penetration testing: OWASP ZAP 2.17.0 for external API scanning

---

## Features

**Seven-Step Cleaning Pipeline**

Step 1 standardises every column header to snake_case by stripping whitespace, lowercasing, and replacing spaces with underscores. Step 2 removes columns where every value is null and drops fully blank rows. Step 3 removes exact duplicate rows preserving the first occurrence. Step 4 strips whitespace and applies title case to all string values while leaving numeric columns untouched. Step 5 fills missing numeric values using KNN imputation which finds the most similar rows and uses their values rather than a global median. Step 6 fills missing text values with the string Unknown. Step 7 detects and removes statistical outliers using Isolation Forest, which examines all numeric columns simultaneously to catch values that are unusual in combination.

**Quality Scoring System**

Calculates a score from zero to one hundred based on three weighted penalty factors. Missing values carry a weight of 40 points, duplicate rows carry 30 points, and empty columns carry 30 points. The score is calculated once before cleaning and once after so the improvement is measurable and reportable. Scores map to human-readable labels: 90 to 100 is Excellent, 75 to 89 is Good, 50 to 74 is Fair, 25 to 49 is Poor, and 0 to 24 is Critical.

**Schema Drift Detection**

Saves the column structure of a reference file and compares every new upload against it. Detects missing columns, new columns, data type changes, column order changes, and possible renames using a character similarity score. Assigns severity levels of none, low, medium, or high. Shown in the web interface before cleaning begins so structural problems are caught before any transformation happens.

**REST API**

Five endpoints accessible to any HTTP client. GET /health returns server status and configuration. GET /schema returns the saved reference schema. POST /schema/save saves a reference schema from an uploaded file. POST /clean accepts a file, runs the full pipeline, and returns the cleaned CSV with quality metrics in the response headers. POST /inspect returns a quality score and drift report without modifying the data.

**Security**

File size limit of 25 megabytes enforced before any parsing to prevent memory exhaustion attacks. All settings loaded from environment variables with no secrets in source code. Security headers middleware adding X-Content-Type-Options, X-Frame-Options, and X-XSS-Protection to every response. Zero Bandit issues across 1200 lines of code. Zero medium or high severity OWASP ZAP findings. API fuzzing with 22 categories of malicious input confirmed no server crashes.

**Validation Report**

Structured dictionary returned alongside the cleaned DataFrame recording every change: original and final row and column counts, empty rows dropped, empty columns dropped, duplicate rows removed, text columns normalised, per-column KNN fill details, per-column text fill details, outliers removed, quality score before and after, and the full improvement delta.

---

## Sample Output

Command-line output:

```
Loading CSV  : data/sample_dirty.csv
Shape: 11 rows x 7 columns

Starting SmartCleaner pipeline...
  Input  -> 11 rows x 7 columns
  Quality score before cleaning: 61.5/100

  Step 1/7 - Column names cleaned.
  Step 2/7 - Empty columns removed.
  Step 3/7 - Duplicate rows removed.
  Step 4/7 - Text values normalised.
  Step 5/7 - Numeric gaps filled using KNN Imputation.
  Step 6/7 - Text gaps filled with Unknown.
  Step 7/7 - Outliers detected and removed.

Cleaning complete!
  Output -> 9 rows x 7 columns
  Quality score after cleaning: 96.67/100
  Improvement: +35.17 points

--- Quality Score ---
  Before :  61.50/100  (Fair)
  After  :  96.67/100  (Excellent)
  Gain   : +35.17 points
```

---

## Project Structure

```
SmartCleaner/
├── src/
│   ├── __init__.py
│   ├── cleaner.py                 Seven-step pipeline and cleaning functions
│   ├── quality_score.py           Quality metrics, scoring formula, score labels
│   ├── outlier_detection.py       Isolation Forest outlier detection
│   ├── knn_imputer.py             KNN imputation for numeric missing values
│   ├── schema_drift.py            Schema drift detection and reference management
│   └── config.py                  Environment variable loading
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_cleaner.py            58 tests
│   ├── test_outlier_detection.py  18 tests
│   ├── test_knn_imputer.py        13 tests
│   ├── test_schema_drift.py       26 tests
│   ├── test_api.py                27 tests
│   ├── test_config_and_limits.py  17 tests
│   └── test_security.py          8 security tests
├── data/
│   ├── sample_dirty.csv           Raw input sample with deliberate quality problems
│   └── sample_cleaned.csv         Output produced by running main.py
├── api.py                         FastAPI REST server
├── app.py                         Streamlit web interface
├── main.py                        Command-line entry point
├── security_scan.py               Bandit security scan runner
├── security_fuzz_test.py          API fuzzing tool
├── sample.env                     Environment variable template
├── requirements.txt
└── README.md
```

---

## Installation and Setup

Clone the repository:

```bash
git clone https://github.com/PraveenMinindu/smartcleaner.git
cd smartcleaner
```

Create and activate a virtual environment. On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

On macOS and Linux:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create your environment file by copying the template:

```bash
copy sample.env .env
```

---

## Usage and How to Run

**Command-line pipeline**

```bash
python main.py
python main.py data/yourfile.csv
python main.py data/yourfile.xlsx
```

The cleaned file is saved in the same folder as the input with a cleaned_ prefix.

**Web application**

```bash
python -m streamlit run app.py
```

Open http://localhost:8501. Upload any CSV or Excel file. The app runs schema drift detection before cleaning, displays before and after quality score cards with colour coding, shows both tables side by side, and provides a download button for the cleaned file.

**REST API**

Start the server:

```bash
uvicorn api:app --reload
```

Open http://localhost:8000/docs for interactive API documentation.

Example using curl:

```bash
curl -X POST http://localhost:8000/clean \
  -F "file=@data/sample_dirty.csv" \
  --output cleaned.csv
```

Example using Python requests:

```python
import requests

with open("data/sample_dirty.csv", "rb") as f:
    response = requests.post(
        "http://localhost:8000/clean",
        files={"file": f}
    )

print(response.headers["X-Quality-Score-Before"])
print(response.headers["X-Quality-Score-After"])
print(response.headers["X-Quality-Improvement"])

with open("cleaned.csv", "wb") as f:
    f.write(response.content)
```

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
python -m pytest tests/ -v -k "TestFillMissingValues"
python -m pytest tests/ -x
```

**Run security scan**

```bash
python security_scan.py
```

**Run API fuzzer** (requires API to be running first):

```bash
python security_fuzz_test.py
```

---

## Quality Score Formula

```
score = 100
      - (missing_percent   / 100) * 40
      - (duplicate_percent / 100) * 30
      - (empty_col_percent / 100) * 30

Score is clamped to the range [0.0, 100.0]

90 - 100   Excellent
75 -  89   Good
50 -  74   Fair
25 -  49   Poor
 0 -  24   Critical
```

---

## API Endpoints

GET /health returns server status, version, environment, and maximum file size limit.

GET /schema returns the saved reference schema showing column names and types.

POST /schema/save accepts a file and saves its column structure as the reference. All future uploads are compared against this reference for drift detection.

POST /clean accepts a CSV or Excel file, runs the full seven-step pipeline, and returns the cleaned file as a downloadable CSV. Quality metrics are in the response headers: X-Quality-Score-Before, X-Quality-Score-After, X-Quality-Improvement, X-Rows-Before, X-Rows-After, X-Outliers-Removed, X-Drift-Detected, and X-Drift-Severity.

POST /inspect accepts a file and returns a JSON quality report and drift analysis without modifying the data.

---

## Security

**Static Analysis — Bandit**

Zero high severity issues. Zero medium severity issues. Zero low severity issues. Scanned across 1200 lines of production code including all src modules and the API.

**API Fuzzing**

22 categories of malicious input tested including wrong file types, SQL injection payloads in cell values, cross-site scripting payloads, null bytes, path traversal filenames, Unicode, emoji, broken CSV structure, 500-column files, and wrong HTTP methods. All 22 tests passed with no server crashes.

**OWASP ZAP Penetration Test**

Zero high severity findings. Zero medium severity findings. One low severity finding — a missing X-Content-Type-Options header — was identified and fixed by adding a security headers middleware to the API. The subsequent scan returned zero findings at all severity levels.

**File Size Limit**

Files larger than 25 megabytes are rejected with HTTP 413 before any parsing begins. This prevents memory exhaustion denial of service attacks. The limit is configurable via the MAX_FILE_SIZE_MB environment variable.

**Environment Variables**

All settings and secrets are loaded from a dot env file. No credentials, keys, or configuration values are hardcoded in source files.

---

## Test Coverage

58 tests in test_cleaner.py covering all cleaning functions, the quality scoring system, and full pipeline integration.

18 tests in test_outlier_detection.py covering Isolation Forest detection, removal, and edge cases.

13 tests in test_knn_imputer.py covering numeric imputation, text column exclusion, small dataset handling, and report accuracy.

26 tests in test_schema_drift.py covering schema saving and loading, all drift types, severity levels, and rename detection.

27 tests in test_api.py covering all five endpoints with valid and invalid inputs, file size limits, and response header validation.

17 tests in test_config_and_limits.py covering environment variable loading and file size enforcement across all endpoints.

8 tests in test_security.py that run Bandit automatically and assert zero high and medium severity issues in all source files.

Total: 151 tests. All passing.

---

## Development History

The project was built in four weekly iterations, each adding a meaningful capability.

The initial version delivered a five-step cleaning pipeline with quality scoring, a Streamlit web interface, and 59 unit tests.

Week 1 added Isolation Forest outlier detection as Step 7 in the pipeline. The algorithm examines all numeric columns simultaneously to catch values that are statistically unusual in combination, not just individually.

Week 2 replaced the median fill strategy for numeric columns with KNN imputation. KNN finds the five most similar rows and uses their values to fill each gap, producing estimates that reflect actual patterns in the data rather than a global statistic.

Week 3 added schema drift detection. The system saves the column structure of a reference file and compares every new upload against it, detecting missing columns, new columns, type changes, and possible renames before any cleaning begins.

Week 4 added a FastAPI REST API with five endpoints, a 25 megabyte file size limit, environment-based configuration, Bandit static security analysis, API fuzzing, and OWASP ZAP penetration testing.

---

## Limitations

The cleaning logic is general-purpose and does not understand domain-specific data shapes. A column mixing director and actor names is cleaned as a single string.

Missing value fill strategies are fixed globally. Text fills with Unknown and numbers fill with KNN estimates. Per-column configuration is not yet supported.

Duplicate detection is exact. Near-duplicate detection using string similarity is not yet implemented.

The web application and API run locally only. There is no authentication, multi-user support, or cloud deployment in the current version.

File encoding is assumed to be UTF-8. Non-UTF-8 encoded CSV files may produce errors.

---

## Future Improvements

YAML configuration file for per-column fill values and adjustable score weights.

Column-level data type inference for dates, booleans, and categorical values.

Near-duplicate detection using string similarity thresholds.

Docker container for one-command deployment.

Support for JSON and Parquet input formats.

Directory watcher for automatic processing of new files.

Historical quality score dashboard across multiple runs.

---

## Contributing

Fork the repository and create a new branch for your feature or fix. Make your changes and add or update tests to cover them. Run the full test suite and confirm all 151 tests pass. Run the security scan and confirm zero medium or high severity issues. Commit with a clear message describing what changed and why. Open a pull request against the main branch.

---

## Author

Praveen Minindu

Built with Python, Pandas, scikit-learn, FastAPI, Streamlit, and pytest
