"""
SmartCleaner - tests/test_cleaner.py
--------------------------------------
Unit tests for every function in src/cleaner.py.

NOTE on fill_missing_values (Week 2 change):
  Since Week 2, fill_missing_values() handles TEXT columns only.
  Numeric columns are handled by knn_fill_numeric().
  Tests in TestFillMissingValues reflect this updated behaviour.

Run with:
    python -m pytest tests/ -v
"""

import pandas as pd
import numpy as np
import pytest

from src.cleaner import (
    clean_column_names,
    remove_empty_columns,
    remove_duplicates,
    clean_text_values,
    fill_missing_values,
    clean_dataset,
)
from src.quality_score import (
    calculate_quality_metrics,
    calculate_quality_score,
    score_label,
)


# ════════════════════════════════════════════════════════════════
#  SECTION 1 — clean_column_names()
# ════════════════════════════════════════════════════════════════

class TestCleanColumnNames:

    def test_strips_whitespace_from_column_names(self):
        df = pd.DataFrame({"  Age  ": [25], "Name": ["Alice"]})
        result = clean_column_names(df)
        assert "age" in result.columns

    def test_converts_column_names_to_lowercase(self):
        df = pd.DataFrame({"SALARY": [50000], "Department": ["HR"]})
        result = clean_column_names(df)
        assert "salary" in result.columns
        assert "department" in result.columns

    def test_replaces_spaces_with_underscores(self):
        df = pd.DataFrame({"First Name": ["Bob"], "Email Address": ["b@t.com"]})
        result = clean_column_names(df)
        assert "first_name" in result.columns
        assert "email_address" in result.columns

    def test_handles_all_transformations_together(self):
        df = pd.DataFrame({"  Email Address  ": ["test@test.com"]})
        result = clean_column_names(df)
        assert "email_address" in result.columns

    def test_does_not_mutate_original_dataframe(self):
        df = pd.DataFrame({"First Name": ["Alice"]})
        original_columns = list(df.columns)
        clean_column_names(df)
        assert list(df.columns) == original_columns

    def test_already_clean_columns_are_unchanged(self):
        df = pd.DataFrame({"first_name": ["Alice"], "age": [30]})
        result = clean_column_names(df)
        assert list(result.columns) == ["first_name", "age"]


# ════════════════════════════════════════════════════════════════
#  SECTION 2 — remove_empty_columns()
# ════════════════════════════════════════════════════════════════

class TestRemoveEmptyColumns:

    def test_removes_fully_empty_column(self):
        df = pd.DataFrame({"name": ["Alice", "Bob"], "notes": [None, None]})
        result = remove_empty_columns(df)
        assert "notes" not in result.columns
        assert "name" in result.columns

    def test_keeps_column_with_at_least_one_value(self):
        df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [None, 30]})
        result = remove_empty_columns(df)
        assert "age" in result.columns

    def test_removes_multiple_empty_columns(self):
        df = pd.DataFrame({"name": ["Alice"], "col_a": [None], "col_b": [None]})
        result = remove_empty_columns(df)
        assert "col_a" not in result.columns
        assert "col_b" not in result.columns
        assert "name" in result.columns

    def test_no_columns_removed_when_all_have_data(self):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        result = remove_empty_columns(df)
        assert result.shape == df.shape


# ════════════════════════════════════════════════════════════════
#  SECTION 3 — remove_duplicates()
# ════════════════════════════════════════════════════════════════

class TestRemoveDuplicates:

    def test_removes_exact_duplicate_row(self):
        df = pd.DataFrame({"name": ["Alice", "Alice"], "salary": [70000, 70000]})
        result = remove_duplicates(df)
        assert len(result) == 1

    def test_keeps_unique_rows_intact(self):
        df = pd.DataFrame({"name": ["Alice", "Bob", "Carol"], "salary": [70000, 80000, 65000]})
        result = remove_duplicates(df)
        assert len(result) == 3

    def test_keeps_first_occurrence_of_duplicate(self):
        df = pd.DataFrame({"name": ["Alice", "Bob", "Alice"], "salary": [70000, 80000, 70000]})
        result = remove_duplicates(df)
        assert len(result) == 2
        assert list(result["name"]) == ["Alice", "Bob"]

    def test_index_is_reset_after_dedup(self):
        df = pd.DataFrame({"name": ["Alice", "Alice", "Bob"], "salary": [70000, 70000, 80000]})
        result = remove_duplicates(df)
        assert list(result.index) == list(range(len(result)))

    def test_no_rows_removed_when_all_unique(self):
        df = pd.DataFrame({"id": [1, 2, 3], "val": ["a", "b", "c"]})
        result = remove_duplicates(df)
        assert len(result) == 3


# ════════════════════════════════════════════════════════════════
#  SECTION 4 — clean_text_values()
# ════════════════════════════════════════════════════════════════

class TestCleanTextValues:

    def test_strips_leading_and_trailing_whitespace(self):
        df = pd.DataFrame({"name": ["  Alice  ", "  Bob"]})
        result = clean_text_values(df)
        assert result["name"][0] == "Alice"
        assert result["name"][1] == "Bob"

    def test_converts_uppercase_to_title_case(self):
        df = pd.DataFrame({"name": ["JOHN DOE", "JANE SMITH"]})
        result = clean_text_values(df)
        assert result["name"][0] == "John Doe"
        assert result["name"][1] == "Jane Smith"

    def test_converts_lowercase_to_title_case(self):
        df = pd.DataFrame({"department": ["marketing", "engineering"]})
        result = clean_text_values(df)
        assert result["department"][0] == "Marketing"
        assert result["department"][1] == "Engineering"

    def test_numeric_columns_are_not_touched(self):
        df = pd.DataFrame({"name": ["Alice"], "salary": [75000]})
        result = clean_text_values(df)
        assert result["salary"][0] == 75000

    def test_nan_in_text_column_remains_nan(self):
        """NaN must stay NaN after clean_text_values — fill handles it later."""
        df = pd.DataFrame({"name": ["Alice", None]})
        result = clean_text_values(df)
        assert pd.isna(result["name"][1])


# ════════════════════════════════════════════════════════════════
#  SECTION 5 — fill_missing_values()
# ════════════════════════════════════════════════════════════════

class TestFillMissingValues:
    """
    fill_missing_values() handles TEXT columns only since Week 2.
    Numeric columns are handled by knn_fill_numeric().
    """

    def test_fills_missing_text_with_unknown(self):
        """NaN in a text column must become 'Unknown'."""
        df = pd.DataFrame({"department": ["HR", None, "Sales"]})
        result = fill_missing_values(df)
        assert result["department"][1] == "Unknown", (
            f"Expected 'Unknown', got '{result['department'][1]}'."
        )

    def test_multiple_missing_text_values_all_filled(self):
        """Every NaN in a text column must be replaced."""
        df = pd.DataFrame({"city": [None, "London", None, None]})
        result = fill_missing_values(df)
        assert (result["city"] == "Unknown").sum() == 3

    def test_numeric_columns_not_touched(self):
        """
        Numeric NaNs must remain NaN after fill_missing_values.
        KNN imputation handles numeric columns — not this function.
        """
        df = pd.DataFrame({"name": ["Alice", None], "salary": [50000.0, None]})
        result = fill_missing_values(df)
        # Text must be filled
        assert result["name"][1] == "Unknown"
        # Numeric must remain NaN
        assert pd.isna(result["salary"][1]), (
            "fill_missing_values must not touch numeric columns."
        )

    def test_columns_without_missing_values_unchanged(self):
        df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [30, 25]})
        result = fill_missing_values(df)
        assert list(result["name"]) == ["Alice", "Bob"]
        assert list(result["age"])  == [30, 25]

    def test_does_not_mutate_original_dataframe(self):
        df = pd.DataFrame({"department": ["HR", None, "Sales"]})
        original_nan_count = df.isna().sum().sum()
        fill_missing_values(df)
        assert df.isna().sum().sum() == original_nan_count


# ════════════════════════════════════════════════════════════════
#  SECTION 6 — clean_dataset() full pipeline
# ════════════════════════════════════════════════════════════════

class TestCleanDataset:

    def test_pipeline_cleans_column_names(self):
        df = pd.DataFrame({"First Name": ["Alice"], "Last Name": ["Smith"]})
        result, report = clean_dataset(df)
        assert "first_name" in result.columns
        assert "last_name"  in result.columns

    def test_pipeline_removes_duplicates(self):
        df = pd.DataFrame({
            "name":   ["Alice", "Alice", "Bob"],
            "salary": [70000.0, 70000.0, 80000.0],
        })
        result, report = clean_dataset(df)
        assert len(result) == 2, f"Expected 2 rows after dedup, got {len(result)}."

    def test_pipeline_removes_empty_columns(self):
        df = pd.DataFrame({"name": ["Alice", "Bob"], "notes": [None, None]})
        result, report = clean_dataset(df)
        assert "notes" not in result.columns

    def test_pipeline_fills_all_missing_values(self):
        df = pd.DataFrame({"name": ["Alice", None], "salary": [50000.0, None]})
        result, report = clean_dataset(df)
        assert result.isna().sum().sum() == 0, "NaN values remain after full pipeline."

    def test_pipeline_handles_real_world_messy_dataframe(self):
        df = pd.DataFrame({
            "  First Name  ": ["  alice  ", "BOB",     "  alice  ", None    ],
            "  SALARY  ":     [50000.0,     None,       50000.0,    75000.0 ],
            "Empty Col":      [None,         None,       None,       None    ],
        })
        result, report = clean_dataset(df)

        assert "first_name" in result.columns,       "Column name not cleaned."
        assert "salary"     in result.columns,       "Column name not cleaned."
        assert "empty_col"  not in result.columns,   "Empty column not removed."
        assert "Alice" in result["first_name"].values, "Text not title-cased."
        assert "Bob"   in result["first_name"].values, "Text not title-cased."
        assert result.isna().sum().sum() == 0,       "NaN values remain."

    def test_pipeline_returns_tuple(self):
        df = pd.DataFrame({"name": ["Alice"], "age": [30.0]})
        output = clean_dataset(df)
        assert isinstance(output,    tuple)
        assert isinstance(output[0], pd.DataFrame)
        assert isinstance(output[1], dict)

    def test_pipeline_does_not_mutate_input(self):
        df = pd.DataFrame({"First Name": ["  alice  "], "Salary": [None]})
        original_shape   = df.shape
        original_columns = list(df.columns)
        clean_dataset(df)
        assert df.shape         == original_shape
        assert list(df.columns) == original_columns

    def test_report_contains_quality_scores(self):
        df = pd.DataFrame({"name": ["Alice", None], "salary": [50000.0, None]})
        result, report = clean_dataset(df)
        assert "quality_score_before" in report
        assert "quality_score_after"  in report
        assert "quality_improvement"  in report
        assert isinstance(report["quality_score_before"], float)
        assert isinstance(report["quality_score_after"],  float)
        assert isinstance(report["quality_improvement"],  float)

    def test_quality_score_improves_after_cleaning(self):
        df = pd.DataFrame({
            "name":   ["Alice", "Alice", None],
            "salary": [50000.0, 50000.0, None],
            "empty":  [None, None, None],
        })
        result, report = clean_dataset(df)
        assert report["quality_score_after"] >= report["quality_score_before"]


# ════════════════════════════════════════════════════════════════
#  SECTION 7 — quality_score module
# ════════════════════════════════════════════════════════════════

class TestCalculateQualityMetrics:

    def test_returns_correct_row_and_column_counts(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        metrics = calculate_quality_metrics(df)
        assert metrics["rows"]    == 3
        assert metrics["columns"] == 2

    def test_calculates_missing_percent_correctly(self):
        df = pd.DataFrame({"a": [1, None], "b": [3, 4]})
        metrics = calculate_quality_metrics(df)
        assert metrics["missing_percent"] == 25.0

    def test_zero_missing_when_data_is_complete(self):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        metrics = calculate_quality_metrics(df)
        assert metrics["missing_percent"] == 0.0

    def test_calculates_duplicate_percent_correctly(self):
        df = pd.DataFrame({"a": [1, 2, 1], "b": ["x", "y", "x"]})
        metrics = calculate_quality_metrics(df)
        assert metrics["duplicate_percent"] == round((1 / 3) * 100, 2)

    def test_zero_duplicates_when_all_rows_unique(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        metrics = calculate_quality_metrics(df)
        assert metrics["duplicate_percent"] == 0.0

    def test_counts_empty_columns_correctly(self):
        df = pd.DataFrame({"good": [1, 2], "empty": [None, None]})
        metrics = calculate_quality_metrics(df)
        assert metrics["empty_columns"] == 1

    def test_zero_empty_columns_when_all_have_data(self):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        metrics = calculate_quality_metrics(df)
        assert metrics["empty_columns"] == 0


class TestCalculateQualityScore:

    def test_perfect_data_scores_100(self):
        df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [30, 25]})
        score = calculate_quality_score(df)
        assert score == 100.0

    def test_score_decreases_with_missing_values(self):
        df_clean   = pd.DataFrame({"a": [1, 2, 3, 4]})
        df_missing = pd.DataFrame({"a": [1, None, None, None]})
        assert calculate_quality_score(df_missing) < calculate_quality_score(df_clean)

    def test_score_decreases_with_duplicates(self):
        df_unique = pd.DataFrame({"a": [1, 2, 3]})
        df_dupes  = pd.DataFrame({"a": [1, 1, 1]})
        assert calculate_quality_score(df_dupes) < calculate_quality_score(df_unique)

    def test_score_decreases_with_empty_columns(self):
        df_good  = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df_empty = pd.DataFrame({"a": [1, 2], "b": [None, None]})
        assert calculate_quality_score(df_empty) < calculate_quality_score(df_good)

    def test_score_never_goes_below_zero(self):
        df = pd.DataFrame({"a": [None, None], "b": [None, None]})
        assert calculate_quality_score(df) >= 0.0

    def test_score_never_exceeds_100(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        assert calculate_quality_score(df) <= 100.0

    def test_score_is_float(self):
        df = pd.DataFrame({"a": [1, 2]})
        assert isinstance(calculate_quality_score(df), float)


class TestScoreLabel:

    def test_100_is_excellent(self):
        assert score_label(100.0) == "Excellent"

    def test_90_is_excellent(self):
        assert score_label(90.0) == "Excellent"

    def test_89_is_good(self):
        assert score_label(89.0) == "Good"

    def test_75_is_good(self):
        assert score_label(75.0) == "Good"

    def test_74_is_fair(self):
        assert score_label(74.0) == "Fair"

    def test_50_is_fair(self):
        assert score_label(50.0) == "Fair"

    def test_49_is_poor(self):
        assert score_label(49.0) == "Poor"

    def test_25_is_poor(self):
        assert score_label(25.0) == "Poor"

    def test_24_is_critical(self):
        assert score_label(24.0) == "Critical"

    def test_zero_is_critical(self):
        assert score_label(0.0) == "Critical"
