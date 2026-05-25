"""
SmartCleaner - tests/test_api.py
----------------------------------
Unit tests for api.py

Uses FastAPI's built-in TestClient which simulates HTTP requests
without needing a running server. This means tests run instantly
and do not require uvicorn to be started.

Every test:
  - Sends a real HTTP request to a real endpoint
  - Checks the response status code
  - Checks the response body or headers

Run with:
    python -m pytest tests/test_api.py -v
"""

import io
import os
import sys
import json
import pytest
import pandas as pd
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api import app

# Create a test client — this simulates a browser or Postman
client = TestClient(app)


# ── Helper: create a CSV file in memory ──────────────────────────────────────

def make_csv(data: dict) -> bytes:
    """
    Convert a dictionary into CSV bytes for upload in tests.
    This simulates a user selecting a file in Postman or a browser.
    """
    df = pd.DataFrame(data)
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def make_dirty_csv() -> bytes:
    """
    Create a realistic dirty CSV file for testing the clean endpoint.
    Contains: duplicate rows, missing values, mixed-case text.
    """
    return make_csv({
        "First Name":  ["Alice", "BOB", "Alice", "  Carol  ", None],
        "Age":         [25.0, 30.0, 25.0, None, 28.0],
        "Salary":      [50000.0, 60000.0, 50000.0, 55000.0, None],
        "Department":  ["engineering", "MARKETING", "engineering", "HR", "Sales"],
    })


# ════════════════════════════════════════════════════════════════
#  SECTION 1 — Health check endpoint
# ════════════════════════════════════════════════════════════════

class TestHealthEndpoint:
    """Tests for GET /health"""

    def test_health_returns_200(self):
        """
        The health endpoint must return HTTP 200 OK.
        200 means the server is running and responding.
        """
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_status_ok(self):
        """
        The response body must contain status: ok.
        """
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "ok"

    def test_health_returns_service_name(self):
        """
        The response must identify itself as SmartCleaner API.
        """
        response = client.get("/health")
        data = response.json()
        assert "SmartCleaner" in data["service"]

    def test_health_returns_timestamp(self):
        """
        The response must include a timestamp so the caller
        knows the server is live and not returning cached data.
        """
        response = client.get("/health")
        data = response.json()
        assert "timestamp" in data
        assert len(data["timestamp"]) > 0


# ════════════════════════════════════════════════════════════════
#  SECTION 2 — Schema endpoints
# ════════════════════════════════════════════════════════════════

class TestSchemaEndpoints:
    """Tests for GET /schema and POST /schema/save"""

    def test_save_schema_returns_200(self):
        """
        Uploading a valid CSV to /schema/save must return 200.
        """
        csv_bytes = make_csv({
            "name":   ["Alice", "Bob"],
            "age":    [25, 30],
            "salary": [50000, 60000],
        })
        response = client.post(
            "/schema/save",
            files={"file": ("reference.csv", csv_bytes, "text/csv")},
        )
        assert response.status_code == 200

    def test_save_schema_returns_column_count(self):
        """
        The response from /schema/save must tell the caller
        how many columns were saved.
        """
        csv_bytes = make_csv({
            "name":   ["Alice"],
            "age":    [25],
            "salary": [50000],
        })
        response = client.post(
            "/schema/save",
            files={"file": ("ref.csv", csv_bytes, "text/csv")},
        )
        data = response.json()
        assert data["columns_saved"] == 3

    def test_save_schema_returns_column_names(self):
        """
        The response must list the exact columns that were saved.
        """
        csv_bytes = make_csv({
            "name":   ["Alice"],
            "salary": [50000],
        })
        response = client.post(
            "/schema/save",
            files={"file": ("ref.csv", csv_bytes, "text/csv")},
        )
        data = response.json()
        assert "name"   in data["columns"]
        assert "salary" in data["columns"]

    def test_get_schema_returns_200_after_save(self):
        """
        After saving a schema, GET /schema must return 200.
        """
        csv_bytes = make_csv({"col_a": [1], "col_b": [2]})
        client.post(
            "/schema/save",
            files={"file": ("ref.csv", csv_bytes, "text/csv")},
        )
        response = client.get("/schema")
        assert response.status_code == 200

    def test_get_schema_returns_correct_columns(self):
        """
        The schema returned by GET /schema must match what was saved.
        """
        csv_bytes = make_csv({"product": ["Apple"], "price": [1.5]})
        client.post(
            "/schema/save",
            files={"file": ("ref.csv", csv_bytes, "text/csv")},
        )
        response = client.get("/schema")
        data = response.json()
        assert "product" in data["columns"]
        assert "price"   in data["columns"]

    def test_save_schema_unsupported_format_returns_400(self):
        """
        Uploading a .txt file must return HTTP 400 Bad Request.
        The API only accepts CSV and Excel files.
        """
        response = client.post(
            "/schema/save",
            files={"file": ("data.txt", b"some text", "text/plain")},
        )
        assert response.status_code == 400


# ════════════════════════════════════════════════════════════════
#  SECTION 3 — Clean endpoint
# ════════════════════════════════════════════════════════════════

class TestCleanEndpoint:
    """Tests for POST /clean"""

    def test_clean_returns_200(self):
        """
        Sending a valid CSV to /clean must return HTTP 200.
        """
        response = client.post(
            "/clean",
            files={"file": ("dirty.csv", make_dirty_csv(), "text/csv")},
        )
        assert response.status_code == 200

    def test_clean_returns_csv_content_type(self):
        """
        The response must be a CSV file, not JSON.
        Content-Type must be text/csv.
        """
        response = client.post(
            "/clean",
            files={"file": ("dirty.csv", make_dirty_csv(), "text/csv")},
        )
        assert "text/csv" in response.headers["content-type"]

    def test_clean_response_has_quality_score_headers(self):
        """
        The response headers must include quality scores so the caller
        can read them without downloading and parsing the full CSV.
        """
        response = client.post(
            "/clean",
            files={"file": ("dirty.csv", make_dirty_csv(), "text/csv")},
        )
        assert "x-quality-score-before" in response.headers
        assert "x-quality-score-after"  in response.headers
        assert "x-quality-improvement"  in response.headers

    def test_clean_response_has_row_count_headers(self):
        """
        The response headers must include row counts before and after
        so the caller knows how many rows were removed.
        """
        response = client.post(
            "/clean",
            files={"file": ("dirty.csv", make_dirty_csv(), "text/csv")},
        )
        assert "x-rows-before" in response.headers
        assert "x-rows-after"  in response.headers

    def test_clean_quality_score_after_is_float(self):
        """
        The quality score in the response header must be a valid number.
        """
        response = client.post(
            "/clean",
            files={"file": ("dirty.csv", make_dirty_csv(), "text/csv")},
        )
        score = float(response.headers["x-quality-score-after"])
        assert 0.0 <= score <= 100.0

    def test_clean_response_body_is_valid_csv(self):
        """
        The response body must be parseable as a valid CSV file.
        It must have at least one column.
        """
        response = client.post(
            "/clean",
            files={"file": ("dirty.csv", make_dirty_csv(), "text/csv")},
        )
        df = pd.read_csv(io.StringIO(response.text))
        assert df.shape[1] > 0

    def test_clean_removes_duplicates(self):
        """
        The cleaned file must have fewer rows than the dirty file
        because duplicates get removed by the pipeline.
        """
        csv_bytes = make_csv({
            "name":   ["Alice", "Alice", "Bob"],
            "salary": [50000.0, 50000.0, 60000.0],
        })
        response = client.post(
            "/clean",
            files={"file": ("data.csv", csv_bytes, "text/csv")},
        )
        df_clean = pd.read_csv(io.StringIO(response.text))
        assert len(df_clean) < 3

    def test_clean_no_missing_values_in_output(self):
        """
        The cleaned CSV must have zero NaN values.
        The pipeline must fill all gaps before returning.
        """
        csv_bytes = make_csv({
            "name":   ["Alice", None, "Bob"],
            "salary": [50000.0, None, 60000.0],
        })
        response = client.post(
            "/clean",
            files={"file": ("data.csv", csv_bytes, "text/csv")},
        )
        df_clean = pd.read_csv(io.StringIO(response.text))
        assert df_clean.isna().sum().sum() == 0

    def test_clean_unsupported_format_returns_400(self):
        """
        Sending a .txt file must return HTTP 400.
        """
        response = client.post(
            "/clean",
            files={"file": ("data.txt", b"hello", "text/plain")},
        )
        assert response.status_code == 400

    def test_clean_has_content_disposition_header(self):
        """
        The response must have a Content-Disposition header so the
        browser or Postman knows to treat it as a file download.
        """
        response = client.post(
            "/clean",
            files={"file": ("sales.csv", make_dirty_csv(), "text/csv")},
        )
        assert "content-disposition" in response.headers
        assert "cleaned_" in response.headers["content-disposition"]

    def test_clean_report_header_is_valid_json(self):
        """
        The X-Report header must contain valid JSON with all
        expected fields so the caller can parse it programmatically.
        """
        response = client.post(
            "/clean",
            files={"file": ("dirty.csv", make_dirty_csv(), "text/csv")},
        )
        report = json.loads(response.headers["x-report"])
        assert "quality_score_before" in report
        assert "quality_score_after"  in report
        assert "rows_before"          in report
        assert "rows_after"           in report


# ════════════════════════════════════════════════════════════════
#  SECTION 4 — Inspect endpoint
# ════════════════════════════════════════════════════════════════

class TestInspectEndpoint:
    """Tests for POST /inspect"""

    def test_inspect_returns_200(self):
        """
        Sending a valid CSV to /inspect must return HTTP 200.
        """
        response = client.post(
            "/inspect",
            files={"file": ("data.csv", make_dirty_csv(), "text/csv")},
        )
        assert response.status_code == 200

    def test_inspect_returns_quality_score(self):
        """
        The inspect response must include a quality_score field
        with a number between 0 and 100.
        """
        response = client.post(
            "/inspect",
            files={"file": ("data.csv", make_dirty_csv(), "text/csv")},
        )
        data = response.json()
        assert "quality_score" in data
        assert 0.0 <= data["quality_score"] <= 100.0

    def test_inspect_returns_row_and_column_counts(self):
        """
        The inspect response must include the exact row and column
        counts from the uploaded file.
        """
        csv_bytes = make_csv({
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": [7, 8, 9],
        })
        response = client.post(
            "/inspect",
            files={"file": ("data.csv", csv_bytes, "text/csv")},
        )
        data = response.json()
        assert data["rows"]    == 3
        assert data["columns"] == 3

    def test_inspect_returns_drift_info(self):
        """
        The inspect response must include a drift section
        with has_drift and drift_severity fields.
        """
        response = client.post(
            "/inspect",
            files={"file": ("data.csv", make_dirty_csv(), "text/csv")},
        )
        data = response.json()
        assert "drift"          in data
        assert "has_drift"      in data["drift"]
        assert "drift_severity" in data["drift"]

    def test_inspect_does_not_modify_data(self):
        """
        The inspect endpoint must return JSON, not a CSV file.
        It looks at the data but never changes it.
        """
        response = client.post(
            "/inspect",
            files={"file": ("data.csv", make_dirty_csv(), "text/csv")},
        )
        assert response.headers["content-type"] == "application/json"

    def test_inspect_returns_missing_percent(self):
        """
        The inspect response must tell the caller what percentage
        of cells are missing in the uploaded file.
        """
        response = client.post(
            "/inspect",
            files={"file": ("data.csv", make_dirty_csv(), "text/csv")},
        )
        data = response.json()
        assert "missing_percent" in data
        assert data["missing_percent"] >= 0.0
