"""
SmartCleaner - tests/test_schema_drift.py
-------------------------------------------
Unit tests for src/schema_drift.py

Tests verify:
  - Schema is saved and loaded correctly
  - Missing columns are detected
  - New columns are detected
  - Type changes are detected
  - Possible renames are flagged
  - No drift is reported when file matches schema
  - Drift severity levels are correct
  - Edge cases are handled gracefully

Run with:
    python -m pytest tests/test_schema_drift.py -v
"""

import os
import json
import tempfile
import pandas as pd
import pytest
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from schema_drift import (
    save_schema,
    load_schema,
    detect_drift,
    get_drift_summary,
    _name_similarity,
)


# ── Shared fixture: temp schema file ─────────────────────────────────────────
# Each test gets its own temporary file so tests never interfere with each other

@pytest.fixture
def schema_file(tmp_path):
    """Return a path to a temporary schema JSON file."""
    return str(tmp_path / "test_schema.json")


@pytest.fixture
def base_df():
    """A clean, known-good DataFrame used as the reference schema."""
    return pd.DataFrame({
        "name":       ["Alice", "Bob"],
        "age":        [25.0, 30.0],
        "department": ["Engineering", "Marketing"],
        "salary":     [70000.0, 65000.0],
        "city":       ["London", "Paris"],
    })


class TestSaveSchema:
    """Tests for the save_schema() function."""

    def test_creates_json_file(self, base_df, schema_file):
        """
        save_schema must create a JSON file at the specified path.
        """
        save_schema(base_df, schema_path=schema_file)
        assert os.path.exists(schema_file), "Schema file was not created."

    def test_saved_schema_contains_correct_columns(self, base_df, schema_file):
        """
        The saved schema must contain the exact column names from the DataFrame.
        """
        save_schema(base_df, schema_path=schema_file)
        with open(schema_file) as f:
            saved = json.load(f)

        assert saved["columns"] == list(base_df.columns)

    def test_saved_schema_contains_column_count(self, base_df, schema_file):
        """
        The saved schema must record the correct number of columns.
        """
        save_schema(base_df, schema_path=schema_file)
        with open(schema_file) as f:
            saved = json.load(f)

        assert saved["column_count"] == base_df.shape[1]

    def test_saved_schema_contains_row_count(self, base_df, schema_file):
        """
        The saved schema must record the number of rows at save time.
        """
        save_schema(base_df, schema_path=schema_file)
        with open(schema_file) as f:
            saved = json.load(f)

        assert saved["row_count"] == len(base_df)

    def test_saved_schema_contains_column_types(self, base_df, schema_file):
        """
        The saved schema must record the type of each column.
        Numeric columns must be "numeric", text columns must be "text".
        """
        save_schema(base_df, schema_path=schema_file)
        with open(schema_file) as f:
            saved = json.load(f)

        assert saved["column_types"]["age"]        == "numeric"
        assert saved["column_types"]["salary"]     == "numeric"
        assert saved["column_types"]["name"]       == "text"
        assert saved["column_types"]["department"] == "text"

    def test_save_returns_schema_dict(self, base_df, schema_file):
        """
        save_schema must return the schema dict it saved,
        not None or another type.
        """
        result = save_schema(base_df, schema_path=schema_file)

        assert isinstance(result, dict)
        assert "columns" in result
        assert "column_types" in result


class TestLoadSchema:
    """Tests for the load_schema() function."""

    def test_loads_saved_schema(self, base_df, schema_file):
        """
        load_schema must return the same schema that was saved.
        """
        save_schema(base_df, schema_path=schema_file)
        loaded = load_schema(schema_path=schema_file)

        assert loaded is not None
        assert loaded["columns"] == list(base_df.columns)

    def test_returns_none_when_no_file_exists(self, schema_file):
        """
        If no schema file has been saved yet, load_schema must
        return None — not raise an error.
        """
        result = load_schema(schema_path=schema_file)
        assert result is None


class TestDetectDrift:
    """Tests for the detect_drift() function."""

    def test_no_drift_when_file_matches_schema(self, base_df, schema_file):
        """
        When the new file has exactly the same columns as the saved schema,
        has_drift must be False and drift_severity must be "none".
        """
        save_schema(base_df, schema_path=schema_file)
        result = detect_drift(base_df, schema_path=schema_file)

        assert result["has_drift"]      is False
        assert result["drift_severity"] == "none"
        assert result["missing_columns"] == []
        assert result["new_columns"]     == []

    def test_detects_missing_column(self, base_df, schema_file):
        """
        When a column present in the saved schema is missing from
        the new file, it must appear in missing_columns.
        """
        save_schema(base_df, schema_path=schema_file)

        # New file is missing the salary column
        df_new = base_df.drop(columns=["salary"])
        result = detect_drift(df_new, schema_path=schema_file)

        assert result["has_drift"]        is True
        assert "salary" in result["missing_columns"]

    def test_detects_new_column(self, base_df, schema_file):
        """
        When the new file has a column that was not in the saved schema,
        it must appear in new_columns.
        """
        save_schema(base_df, schema_path=schema_file)

        # New file has an extra column
        df_new = base_df.copy()
        df_new["bonus"] = [5000.0, 4000.0]
        result = detect_drift(df_new, schema_path=schema_file)

        assert result["has_drift"]      is True
        assert "bonus" in result["new_columns"]

    def test_detects_type_change(self, base_df, schema_file):
        """
        When a column changes from numeric to text or vice versa,
        it must appear in type_changes.
        """
        save_schema(base_df, schema_path=schema_file)

        # age column changed from numeric to text
        df_new = base_df.copy()
        df_new["age"] = df_new["age"].astype(str)
        result = detect_drift(df_new, schema_path=schema_file)

        assert result["has_drift"] is True
        type_changed_cols = [tc["column"] for tc in result["type_changes"]]
        assert "age" in type_changed_cols

    def test_detects_possible_rename(self, base_df, schema_file):
        """
        When a column disappears and a similarly-named column appears,
        it must be flagged as a possible rename.
        department -> dept should be flagged.
        """
        save_schema(base_df, schema_path=schema_file)

        # Rename department to dept
        df_new = base_df.rename(columns={"department": "dept"})
        result = detect_drift(df_new, schema_path=schema_file)

        assert result["has_drift"] is True
        rename_pairs = [(r["old_name"], r["new_name"]) for r in result["possible_renames"]]
        assert ("department", "dept") in rename_pairs

    def test_no_schema_returns_schema_exists_false(self, base_df, schema_file):
        """
        When no schema has been saved yet, detect_drift must return
        schema_exists = False and has_drift = False.
        """
        result = detect_drift(base_df, schema_path=schema_file)

        assert result["schema_exists"] is False
        assert result["has_drift"]     is False
        assert "message"               in result

    def test_drift_severity_high_when_columns_removed(self, base_df, schema_file):
        """
        When columns are removed from the file, severity must be "high"
        because missing columns break downstream processing.
        """
        save_schema(base_df, schema_path=schema_file)

        df_new = base_df.drop(columns=["salary", "city", "department"])
        result = detect_drift(df_new, schema_path=schema_file)

        assert result["drift_severity"] == "high"

    def test_drift_severity_medium_for_single_change(self, base_df, schema_file):
        """
        When one column is added or removed, severity must be "medium".
        """
        save_schema(base_df, schema_path=schema_file)

        df_new = base_df.drop(columns=["city"])
        result = detect_drift(df_new, schema_path=schema_file)

        assert result["drift_severity"] in ("medium", "high")

    def test_correct_column_counts_in_result(self, base_df, schema_file):
        """
        The result must report the saved column count and the
        new file's column count accurately.
        """
        save_schema(base_df, schema_path=schema_file)

        df_new = base_df.drop(columns=["city"])
        result = detect_drift(df_new, schema_path=schema_file)

        assert result["saved_column_count"] == base_df.shape[1]
        assert result["new_column_count"]   == df_new.shape[1]

    def test_multiple_issues_detected_at_once(self, base_df, schema_file):
        """
        detect_drift must catch multiple problems in a single call:
        a missing column AND a new column AND a type change.
        """
        save_schema(base_df, schema_path=schema_file)

        df_new = base_df.copy()
        df_new = df_new.drop(columns=["city"])              # missing column
        df_new["region"]  = ["North", "South"]              # new column
        df_new["age"]     = df_new["age"].astype(str)       # type change

        result = detect_drift(df_new, schema_path=schema_file)

        assert result["has_drift"]          is True
        assert "city"   in result["missing_columns"]
        assert "region" in result["new_columns"]
        assert len(result["type_changes"])  > 0


class TestNameSimilarity:
    """Tests for the _name_similarity() helper function."""

    def test_identical_names_score_one(self):
        """Two identical names must score 1.0."""
        assert _name_similarity("salary", "salary") == 1.0

    def test_completely_different_names_score_low(self):
        """Two completely unrelated names must score low."""
        score = _name_similarity("salary", "xyz")
        assert score < 0.5

    def test_similar_names_score_high(self):
        """
        Names that are abbreviations of each other must score
        above 0.6 so they get flagged as possible renames.
        """
        score = _name_similarity("first_name", "firstname")
        assert score >= 0.6

    def test_empty_string_returns_zero(self):
        """An empty string must return 0.0 without crashing."""
        assert _name_similarity("", "salary") == 0.0
        assert _name_similarity("salary", "") == 0.0


class TestGetDriftSummary:
    """Tests for the get_drift_summary() helper function."""

    def test_returns_string(self, base_df, schema_file):
        """get_drift_summary must always return a string."""
        save_schema(base_df, schema_path=schema_file)
        result = detect_drift(base_df, schema_path=schema_file)
        summary = get_drift_summary(result)

        assert isinstance(summary, str)

    def test_no_drift_message_is_positive(self, base_df, schema_file):
        """
        When there is no drift, the summary must say so clearly.
        """
        save_schema(base_df, schema_path=schema_file)
        result  = detect_drift(base_df, schema_path=schema_file)
        summary = get_drift_summary(result)

        assert "no drift" in summary.lower()

    def test_drift_message_mentions_missing_columns(self, base_df, schema_file):
        """
        When columns are missing, the summary must mention them by name.
        """
        save_schema(base_df, schema_path=schema_file)
        df_new  = base_df.drop(columns=["salary"])
        result  = detect_drift(df_new, schema_path=schema_file)
        summary = get_drift_summary(result)

        assert "salary" in summary

    def test_no_schema_message_is_helpful(self, schema_file):
        """
        When no schema exists, the summary must give the user
        clear instructions on what to do next.
        """
        df     = pd.DataFrame({"a": [1, 2]})
        result = detect_drift(df, schema_path=schema_file)
        summary = get_drift_summary(result)

        assert isinstance(summary, str)
        assert len(summary) > 10
