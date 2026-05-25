"""
SmartCleaner - tests/test_knn_imputer.py
------------------------------------------
Unit tests for src/knn_imputer.py

Tests verify that:
  - KNN correctly fills missing numeric values
  - Text columns are never touched
  - The report is accurate
  - Edge cases are handled gracefully
  - The original DataFrame is never mutated

Run with:
    python -m pytest tests/test_knn_imputer.py -v
"""

import pandas as pd
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from knn_imputer import knn_fill_numeric


class TestKnnFillNumeric:
    """Tests for the knn_fill_numeric() function."""

    def test_fills_missing_numeric_values(self):
        """
        Missing values in a numeric column must be filled.
        After calling knn_fill_numeric there must be zero NaNs
        in numeric columns.
        """
        df = pd.DataFrame({
            "age":    [25.0, 30.0, None, 28.0, 35.0],
            "salary": [50000.0, 60000.0, 55000.0, None, 70000.0],
        })
        result, report = knn_fill_numeric(df)

        assert result["age"].isna().sum() == 0,    "age still has NaN after KNN fill."
        assert result["salary"].isna().sum() == 0, "salary still has NaN after KNN fill."

    def test_returns_tuple_of_dataframe_and_dict(self):
        """
        knn_fill_numeric must return a tuple where the first element
        is a DataFrame and the second is a dictionary.
        """
        df = pd.DataFrame({
            "age":    [25.0, 30.0, None],
            "salary": [50000.0, 60000.0, 55000.0],
        })
        result = knn_fill_numeric(df)

        assert isinstance(result, tuple),           "Must return a tuple."
        assert isinstance(result[0], pd.DataFrame), "First element must be DataFrame."
        assert isinstance(result[1], dict),         "Second element must be dict."

    def test_text_columns_are_not_touched(self):
        """
        KNN only works on numbers. Text columns must pass through
        completely unchanged — including their NaN values.
        """
        df = pd.DataFrame({
            "name":   ["Alice", None, "Carol"],
            "salary": [50000.0, 55000.0, None],
        })
        result, _ = knn_fill_numeric(df)

        # Text column NaN must still be NaN — KNN does not touch text
        assert pd.isna(result["name"][1]), "Text column NaN was changed by KNN."

    def test_report_contains_correct_keys(self):
        """
        The report dict must contain all expected keys.
        """
        df = pd.DataFrame({
            "age":    [25.0, 30.0, None],
            "salary": [50000.0, None, 55000.0],
        })
        _, report = knn_fill_numeric(df)

        expected_keys = [
            "columns_filled", "total_values_filled",
            "fill_details", "n_neighbors_used", "method",
        ]
        for key in expected_keys:
            assert key in report, f"Missing key in report: {key}"

    def test_columns_filled_lists_correct_columns(self):
        """
        report columns_filled must list exactly the columns
        that had missing values — not all numeric columns.
        """
        df = pd.DataFrame({
            "age":    [25.0, 30.0, 28.0],     # no missing values
            "salary": [50000.0, None, 55000.0],  # has missing
        })
        _, report = knn_fill_numeric(df)

        assert "salary" in report["columns_filled"]
        assert "age"    not in report["columns_filled"]

    def test_total_values_filled_is_accurate(self):
        """
        total_values_filled in the report must equal the exact
        number of NaN cells that were filled.
        """
        df = pd.DataFrame({
            "age":    [25.0, None, None],       # 2 missing
            "salary": [50000.0, None, 55000.0], # 1 missing
        })
        _, report = knn_fill_numeric(df)

        assert report["total_values_filled"] == 3, (
            f"Expected 3 values filled, got {report['total_values_filled']}."
        )

    def test_no_numeric_columns_returns_unchanged_dataframe(self):
        """
        If the DataFrame has no numeric columns, return it unchanged
        with an informative skipped_reason in the report.
        """
        df = pd.DataFrame({
            "name": ["Alice", "Bob"],
            "city": ["London", "Paris"],
        })
        result, report = knn_fill_numeric(df)

        assert result.equals(df),                        "DataFrame was changed despite no numeric columns."
        assert report["total_values_filled"] == 0
        assert report["columns_filled"]      == []
        assert "skipped_reason"              in report

    def test_no_missing_values_returns_unchanged_dataframe(self):
        """
        If there are no missing values in numeric columns,
        the DataFrame must be returned completely unchanged.
        """
        df = pd.DataFrame({
            "age":    [25.0, 30.0, 28.0],
            "salary": [50000.0, 60000.0, 55000.0],
        })
        result, report = knn_fill_numeric(df)

        assert result["age"].tolist()    == df["age"].tolist()
        assert result["salary"].tolist() == df["salary"].tolist()
        assert report["total_values_filled"] == 0
        assert "skipped_reason" in report

    def test_does_not_mutate_original_dataframe(self):
        """
        The original DataFrame passed into knn_fill_numeric
        must remain completely unchanged after the call.
        """
        df = pd.DataFrame({
            "age":    [25.0, None, 28.0],
            "salary": [50000.0, 55000.0, None],
        })
        original_nan_count = df.isna().sum().sum()

        knn_fill_numeric(df)

        assert df.isna().sum().sum() == original_nan_count, (
            "Original DataFrame was mutated — NaN count changed."
        )

    def test_method_field_in_report_is_knn(self):
        """
        The method field in the report must always say KNN Imputation
        so callers know what strategy was used.
        """
        df = pd.DataFrame({
            "age": [25.0, None, 28.0],
        })
        _, report = knn_fill_numeric(df)

        assert report["method"] == "KNN Imputation"

    def test_filled_values_are_numeric(self):
        """
        After filling, all values in numeric columns must be
        actual numbers — not NaN, not strings, not None.
        """
        df = pd.DataFrame({
            "age":    [25.0, None, 30.0, 28.0, None],
            "salary": [50000.0, 55000.0, None, 60000.0, 52000.0],
        })
        result, _ = knn_fill_numeric(df)

        for col in ["age", "salary"]:
            assert result[col].isna().sum() == 0, f"{col} still has NaN."
            assert pd.api.types.is_numeric_dtype(result[col]), f"{col} is not numeric."

    def test_k_is_reduced_for_small_datasets(self):
        """
        If the dataset has fewer complete rows than n_neighbors,
        K must be automatically reduced so the imputer does not crash.
        """
        # Only 2 complete rows — default K=5 would fail
        df = pd.DataFrame({
            "age":    [25.0, 30.0, None],
            "salary": [50000.0, 60000.0, None],
        })
        # This should not raise an error
        result, report = knn_fill_numeric(df, n_neighbors=5)

        assert result.isna().sum().sum() == 0, "NaNs remain after small-dataset fill."
        assert report["n_neighbors_used"] <= 2, "K was not reduced for small dataset."

    def test_mixed_columns_only_numeric_filled(self):
        """
        In a DataFrame with both text and numeric columns,
        only numeric columns with NaN should be affected.
        """
        df = pd.DataFrame({
            "name":       ["Alice", "Bob",  "Carol"],
            "department": ["Eng",   None,   "HR"],
            "age":        [25.0,    None,   30.0],
            "salary":     [50000.0, 55000.0, None],
        })
        result, report = knn_fill_numeric(df)

        # Numeric NaNs must be filled
        assert result["age"].isna().sum()    == 0
        assert result["salary"].isna().sum() == 0

        # Text NaN must remain untouched
        assert pd.isna(result["department"][1]), "Text NaN was incorrectly filled by KNN."

        # Only numeric columns should appear in report
        assert "age"        in report["columns_filled"] or "salary" in report["columns_filled"]
        assert "department" not in report["columns_filled"]
