"""
SmartCleaner - tests/test_outlier_detection.py
------------------------------------------------
Unit tests for src/outlier_detection.py

Every test follows the Arrange - Act - Assert pattern:
  Arrange: build a small DataFrame with a specific condition
  Act:     call the function being tested
  Assert:  check the output is exactly what we expect

Run with:
    python -m pytest tests/test_outlier_detection.py -v
"""

import pandas as pd
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from outlier_detection import detect_outliers, remove_outliers, get_outlier_rows


class TestDetectOutliers:
    """Tests for the detect_outliers() function."""

    def test_detects_obvious_outlier(self):
        """
        A value that is extremely far from all others must be flagged.
        Salaries are all around 50000 except one row which is 9999999.
        That row must be detected as an outlier.
        """
        df = pd.DataFrame({
            "salary": [50000, 52000, 48000, 51000, 49000,
                       53000, 50500, 51500, 49500, 9999999],
        })
        result = detect_outliers(df, contamination=0.1)

        assert result["has_outliers"] is True, "Obvious outlier was not detected."
        assert result["outlier_count"] > 0, "Expected at least one outlier."

    def test_returns_correct_keys(self):
        """
        The report dict must contain all expected keys.
        """
        df = pd.DataFrame({"score": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]})
        result = detect_outliers(df)

        expected_keys = [
            "outlier_count", "outlier_indices", "outlier_percent",
            "numeric_columns_used", "contamination_used", "has_outliers",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key in report: {key}"

    def test_clean_data_flags_minimal_outliers(self):
        """
        Perfectly uniform data should flag very few or zero outliers.
        All values are identical so no row is unusual.
        """
        df = pd.DataFrame({
            "value": [100.0] * 50,
        })
        result = detect_outliers(df, contamination=0.05)

        # With identical values, no row is more isolated than another
        # so outlier count should be very low (0 or just rounding artifacts)
        assert result["outlier_count"] <= 3, (
            f"Too many outliers flagged in uniform data: {result['outlier_count']}"
        )

    def test_no_numeric_columns_returns_empty_report(self):
        """
        If the DataFrame has no numeric columns, detection cannot run.
        The function must return gracefully with has_outliers = False
        and a skipped_reason explaining why.
        """
        df = pd.DataFrame({
            "name": ["Alice", "Bob", "Carol"],
            "city": ["London", "Paris", "Tokyo"],
        })
        result = detect_outliers(df)

        assert result["has_outliers"] is False
        assert result["outlier_count"] == 0
        assert "skipped_reason" in result

    def test_outlier_indices_are_valid_index_values(self):
        """
        Every index in outlier_indices must be a valid index
        value that exists in the original DataFrame.
        """
        df = pd.DataFrame({
            "age":    [25, 27, 26, 24, 28, 25, 27, 26, 999, 25],
            "salary": [50000, 51000, 49000, 52000, 50500,
                       51500, 49500, 50200, 50100, 51200],
        })
        result = detect_outliers(df, contamination=0.1)

        for idx in result["outlier_indices"]:
            assert idx in df.index, f"Outlier index {idx} not in DataFrame index."

    def test_numeric_columns_used_is_correct(self):
        """
        The report must list the exact numeric columns that
        were passed to the Isolation Forest model.
        """
        df = pd.DataFrame({
            "name":   ["Alice", "Bob", "Carol"],
            "age":    [25, 30, 28],
            "salary": [50000, 60000, 55000],
        })
        result = detect_outliers(df)

        assert "age"    in result["numeric_columns_used"]
        assert "salary" in result["numeric_columns_used"]
        assert "name"   not in result["numeric_columns_used"]

    def test_outlier_percent_is_calculated_correctly(self):
        """
        outlier_percent = (outlier_count / total_rows) * 100
        If 2 out of 10 rows are flagged, percent should be 20.0
        """
        df = pd.DataFrame({
            "value": [10, 12, 11, 13, 10, 12, 11, 9999, 9998, 10],
        })
        result = detect_outliers(df, contamination=0.2)

        expected_percent = round((result["outlier_count"] / len(df)) * 100, 2)
        assert result["outlier_percent"] == expected_percent

    def test_does_not_mutate_original_dataframe(self):
        """
        detect_outliers must never modify the DataFrame passed in.
        It only reads data and returns a report.
        """
        df = pd.DataFrame({"salary": [50000, 52000, 48000, 9999999, 51000]})
        original_shape  = df.shape
        original_values = df["salary"].tolist()

        detect_outliers(df)

        assert df.shape == original_shape
        assert df["salary"].tolist() == original_values


class TestRemoveOutliers:
    """Tests for the remove_outliers() function."""

    def test_removes_outlier_rows(self):
        """
        After calling remove_outliers, the rows flagged as outliers
        must not be present in the returned DataFrame.
        """
        df = pd.DataFrame({
            "salary": [50000, 52000, 48000, 51000, 49000,
                       53000, 50500, 51500, 49500, 9999999],
        })
        df_clean, report = remove_outliers(df, contamination=0.1)

        assert len(df_clean) < len(df), "No rows were removed."
        assert report["outliers_removed"] > 0

    def test_returns_tuple_of_dataframe_and_dict(self):
        """
        remove_outliers must return a tuple where the first element
        is a DataFrame and the second is a dictionary.
        """
        df = pd.DataFrame({"value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 1000]})
        result = remove_outliers(df, contamination=0.1)

        assert isinstance(result, tuple)
        assert isinstance(result[0], pd.DataFrame)
        assert isinstance(result[1], dict)

    def test_index_is_reset_after_removal(self):
        """
        After removing outlier rows the index must be continuous:
        0, 1, 2, ... — not 0, 1, 3, 4 (with gaps).
        """
        df = pd.DataFrame({
            "salary": [50000, 52000, 48000, 51000, 49000,
                       53000, 50500, 51500, 49500, 9999999],
        })
        df_clean, _ = remove_outliers(df, contamination=0.1)

        assert list(df_clean.index) == list(range(len(df_clean))), (
            "Index was not reset after outlier removal."
        )

    def test_report_contains_correct_keys(self):
        """
        The report dict from remove_outliers must have all expected keys.
        """
        df = pd.DataFrame({"score": [10, 20, 30, 40, 50, 60, 70, 80, 90, 9999]})
        _, report = remove_outliers(df, contamination=0.1)

        expected_keys = [
            "rows_before", "rows_after", "outliers_removed",
            "outlier_percent", "numeric_columns_used",
            "outlier_indices", "has_outliers", "contamination_used",
        ]
        for key in expected_keys:
            assert key in report, f"Missing key in report: {key}"

    def test_rows_before_and_after_are_correct(self):
        """
        rows_before must equal the original row count.
        rows_after must equal rows_before minus outliers_removed.
        """
        df = pd.DataFrame({
            "salary": [50000, 52000, 48000, 51000, 49000,
                       53000, 50500, 51500, 49500, 9999999],
        })
        df_clean, report = remove_outliers(df, contamination=0.1)

        assert report["rows_before"] == len(df)
        assert report["rows_after"]  == len(df_clean)
        assert report["rows_after"]  == report["rows_before"] - report["outliers_removed"]

    def test_does_not_mutate_original_dataframe(self):
        """
        The original DataFrame passed to remove_outliers must
        remain completely unchanged.
        """
        df = pd.DataFrame({
            "salary": [50000, 52000, 9999999, 51000, 49000]
        })
        original_len = len(df)

        remove_outliers(df)

        assert len(df) == original_len, "Original DataFrame was mutated."

    def test_no_numeric_columns_returns_unchanged_dataframe(self):
        """
        If there are no numeric columns, the DataFrame must be
        returned unchanged with outliers_removed = 0.
        """
        df = pd.DataFrame({
            "name": ["Alice", "Bob", "Carol"],
            "city": ["London", "Paris", "Tokyo"],
        })
        df_clean, report = remove_outliers(df)

        assert len(df_clean) == len(df)
        assert report["outliers_removed"] == 0
        assert report["has_outliers"] is False


class TestGetOutlierRows:
    """Tests for the get_outlier_rows() helper function."""

    def test_returns_dataframe(self):
        """
        get_outlier_rows must always return a DataFrame,
        never None or another type.
        """
        df = pd.DataFrame({"value": [10, 20, 30, 40, 50, 60, 70, 80, 90, 9999]})
        result = get_outlier_rows(df, contamination=0.1)

        assert isinstance(result, pd.DataFrame)

    def test_returns_empty_dataframe_when_no_outliers(self):
        """
        When no outliers are found, the function must return
        an empty DataFrame, not raise an error.
        """
        df = pd.DataFrame({"name": ["Alice", "Bob"], "city": ["London", "Paris"]})
        result = get_outlier_rows(df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_outlier_rows_are_subset_of_original(self):
        """
        Every row returned by get_outlier_rows must exist in
        the original DataFrame. It is a subset, not new data.
        """
        df = pd.DataFrame({
            "salary": [50000, 52000, 48000, 51000, 49000,
                       53000, 50500, 51500, 49500, 9999999],
        })
        outlier_df = get_outlier_rows(df, contamination=0.1)

        for idx in outlier_df.index:
            assert idx in df.index, f"Row {idx} in outlier_df not found in original df."
