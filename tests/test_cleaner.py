"""
SmartCleaner - tests/test_cleaner.py
--------------------------------------
Unit tests for every function in src/cleaner.py.

Philosophy:
  - Each test is self-contained: it builds its own tiny DataFrame,
    calls exactly one function, and asserts the expected result.
  - Tests are intentionally small and focused — if a test fails,
    you know immediately which behaviour broke and why.
  - No external files are read; no side-effects on disk.

Run all tests with:
    pytest tests/
    pytest tests/ -v          # verbose — shows each test name
    pytest tests/ -v --tb=short  # short traceback on failure
"""

import math
import pandas as pd
import pytest

# Import all functions under test from our src package
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
    """All tests that verify column-name normalisation."""

    def test_strips_whitespace_from_column_names(self):
        """
        Columns with leading/trailing spaces must be trimmed.
        '  Age  ' → 'age'  (whitespace gone, also lowercased)
        """
        df = pd.DataFrame({"  Age  ": [25], "Name": ["Alice"]})
        result = clean_column_names(df)

        assert "age" in result.columns, "Leading/trailing spaces were not stripped."

    def test_converts_column_names_to_lowercase(self):
        """
        Uppercase column headers must become lowercase.
        'SALARY' → 'salary'
        """
        df = pd.DataFrame({"SALARY": [50000], "Department": ["HR"]})
        result = clean_column_names(df)

        assert "salary" in result.columns
        assert "department" in result.columns

    def test_replaces_spaces_with_underscores(self):
        """
        Multi-word column names must use underscores, not spaces.
        'First Name' → 'first_name'
        'Email Address' → 'email_address'
        """
        df = pd.DataFrame({"First Name": ["Bob"], "Email Address": ["bob@test.com"]})
        result = clean_column_names(df)

        assert "first_name" in result.columns
        assert "email_address" in result.columns

    def test_handles_all_transformations_together(self):
        """
        A messy column header like '  Email Address  ' must survive
        all three transformations: strip → lower → underscores.
        """
        df = pd.DataFrame({"  Email Address  ": ["test@test.com"]})
        result = clean_column_names(df)

        assert "email_address" in result.columns

    def test_does_not_mutate_original_dataframe(self):
        """
        The function must return a *copy*, never modify the caller's DataFrame.
        This is critical for pipeline safety — steps should be independent.
        """
        df = pd.DataFrame({"First Name": ["Alice"]})
        original_columns = list(df.columns)

        clean_column_names(df)  # call — but ignore return value

        assert list(df.columns) == original_columns, (
            "Original DataFrame was mutated — function must work on a copy."
        )

    def test_already_clean_columns_are_unchanged(self):
        """
        If columns are already snake_case, the function must leave them alone.
        Idempotent behaviour: calling it twice gives the same result.
        """
        df = pd.DataFrame({"first_name": ["Alice"], "age": [30]})
        result = clean_column_names(df)

        assert list(result.columns) == ["first_name", "age"]


# ════════════════════════════════════════════════════════════════
#  SECTION 2 — remove_empty_columns()
# ════════════════════════════════════════════════════════════════

class TestRemoveEmptyColumns:
    """Tests that verify fully-NaN columns are dropped."""

    def test_removes_fully_empty_column(self):
        """
        A column that contains only NaN values must be dropped entirely.
        """
        df = pd.DataFrame({
            "name":  ["Alice", "Bob"],
            "notes": [None, None],      # ← completely empty column
        })
        result = remove_empty_columns(df)

        assert "notes" not in result.columns, "Fully-empty column was not removed."
        assert "name" in result.columns,      "Non-empty column was incorrectly removed."

    def test_keeps_column_with_at_least_one_value(self):
        """
        A column with even a single non-NaN value must be kept.
        Partially-empty ≠ fully-empty.
        """
        df = pd.DataFrame({
            "name": ["Alice", "Bob"],
            "age":  [None, 30],         # ← only one value, but not fully empty
        })
        result = remove_empty_columns(df)

        assert "age" in result.columns, "Partially-filled column should not be removed."

    def test_removes_multiple_empty_columns(self):
        """
        Multiple fully-NaN columns must all be dropped in one pass.
        """
        df = pd.DataFrame({
            "name":    ["Alice"],
            "col_a":   [None],
            "col_b":   [None],
        })
        result = remove_empty_columns(df)

        assert "col_a" not in result.columns
        assert "col_b" not in result.columns
        assert "name"  in result.columns

    def test_no_columns_removed_when_all_have_data(self):
        """
        If every column has at least one value, the shape must be unchanged.
        """
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        result = remove_empty_columns(df)

        assert result.shape == df.shape


# ════════════════════════════════════════════════════════════════
#  SECTION 3 — remove_duplicates()
# ════════════════════════════════════════════════════════════════

class TestRemoveDuplicates:
    """Tests that verify duplicate rows are correctly removed."""

    def test_removes_exact_duplicate_row(self):
        """
        When two rows are identical in every column, only the first is kept.
        The duplicate second row must be gone.
        """
        df = pd.DataFrame({
            "name":   ["Alice", "Alice"],
            "salary": [70000,   70000],
        })
        result = remove_duplicates(df)

        assert len(result) == 1, f"Expected 1 row, got {len(result)}."
        assert result.iloc[0]["name"] == "Alice"

    def test_keeps_unique_rows_intact(self):
        """
        Rows that are genuinely different must all survive.
        """
        df = pd.DataFrame({
            "name":   ["Alice", "Bob", "Carol"],
            "salary": [70000,   80000,  65000],
        })
        result = remove_duplicates(df)

        assert len(result) == 3, "Unique rows were incorrectly removed."

    def test_keeps_first_occurrence_of_duplicate(self):
        """
        The first occurrence of a duplicated row must be kept,
        not the second. (keep='first' behaviour)
        """
        df = pd.DataFrame({
            "name":   ["Alice", "Bob",   "Alice"],
            "salary": [70000,   80000,   70000],
        })
        result = remove_duplicates(df)

        # After dedup: Alice (row 0) + Bob (row 1)  →  2 rows
        assert len(result) == 2
        assert list(result["name"]) == ["Alice", "Bob"]

    def test_index_is_reset_after_dedup(self):
        """
        After removing duplicates the DataFrame index must be
        continuous: 0, 1, 2, … — not 0, 2 (with a gap).
        """
        df = pd.DataFrame({
            "name":   ["Alice", "Alice", "Bob"],
            "salary": [70000,   70000,   80000],
        })
        result = remove_duplicates(df)

        assert list(result.index) == list(range(len(result))), (
            "Index was not reset after deduplication."
        )

    def test_no_rows_removed_when_all_unique(self):
        """
        When no duplicates exist the row count must stay the same.
        """
        df = pd.DataFrame({"id": [1, 2, 3], "val": ["a", "b", "c"]})
        result = remove_duplicates(df)

        assert len(result) == 3


# ════════════════════════════════════════════════════════════════
#  SECTION 4 — clean_text_values()
# ════════════════════════════════════════════════════════════════

class TestCleanTextValues:
    """Tests that verify string normalisation (strip + title-case)."""

    def test_strips_leading_and_trailing_whitespace(self):
        """
        Cell values with surrounding spaces must be trimmed.
        '  Alice  ' → 'Alice'
        """
        df = pd.DataFrame({"name": ["  Alice  ", "  Bob"]})
        result = clean_text_values(df)

        assert result["name"][0] == "Alice"
        assert result["name"][1] == "Bob"

    def test_converts_uppercase_to_title_case(self):
        """
        ALL-CAPS text must become Title Case.
        'JOHN DOE' → 'John Doe'
        """
        df = pd.DataFrame({"name": ["JOHN DOE", "JANE SMITH"]})
        result = clean_text_values(df)

        assert result["name"][0] == "John Doe"
        assert result["name"][1] == "Jane Smith"

    def test_converts_lowercase_to_title_case(self):
        """
        all-lowercase text must become Title Case.
        'marketing' → 'Marketing'
        """
        df = pd.DataFrame({"department": ["marketing", "engineering"]})
        result = clean_text_values(df)

        assert result["department"][0] == "Marketing"
        assert result["department"][1] == "Engineering"

    def test_numeric_columns_are_not_touched(self):
        """
        Numeric columns must pass through completely unchanged.
        Applying string ops to numbers would corrupt the data.
        """
        df = pd.DataFrame({"name": ["Alice"], "salary": [75000]})
        result = clean_text_values(df)

        assert result["salary"][0] == 75000
        assert result["salary"].dtype != object

    def test_nan_in_text_column_remains_nan(self):
        """
        NaN values inside a text column must stay NaN after cleaning —
        they should be handled later by fill_missing_values(), not here.
        """
        df = pd.DataFrame({"name": ["Alice", None]})
        result = clean_text_values(df)

        assert pd.isna(result["name"][1]), "NaN was unexpectedly changed by clean_text_values."


# ════════════════════════════════════════════════════════════════
#  SECTION 5 — fill_missing_values()
# ════════════════════════════════════════════════════════════════

class TestFillMissingValues:
    """Tests that verify NaN-filling logic for text and numeric columns."""

    def test_fills_missing_text_with_unknown(self):
        """
        NaN cells in a text column must be replaced with the
        string 'Unknown' — never left as NaN.
        """
        df = pd.DataFrame({"department": ["HR", None, "Sales"]})
        result = fill_missing_values(df)

        assert result["department"][1] == "Unknown", (
            f"Expected 'Unknown', got '{result['department'][1]}'."
        )

    def test_fills_missing_numeric_with_median(self):
        """
        NaN cells in a numeric column must be replaced with the
        column's median value.
        Median of [10, 20, 30] = 20.0
        """
        df = pd.DataFrame({"salary": [10.0, None, 30.0]})
        result = fill_missing_values(df)

        # Median of [10, 30] = 20.0  (NaN is excluded from median calc)
        assert result["salary"][1] == 20.0, (
            f"Expected median 20.0, got {result['salary'][1]}."
        )

    def test_no_nan_remains_after_filling(self):
        """
        After filling, the DataFrame must contain zero NaN values.
        This is the most fundamental contract of the function.
        """
        df = pd.DataFrame({
            "name":   ["Alice", None],
            "salary": [50000.0, None],
        })
        result = fill_missing_values(df)

        total_missing = result.isna().sum().sum()
        assert total_missing == 0, f"Still {total_missing} NaN(s) after filling."

    def test_multiple_missing_text_values_all_filled(self):
        """
        Every NaN in a text column must be replaced, not just the first one.
        """
        df = pd.DataFrame({"city": [None, "London", None, None]})
        result = fill_missing_values(df)

        assert (result["city"] == "Unknown").sum() == 3

    def test_median_is_outlier_resistant(self):
        """
        Median should be used (not mean) so one extreme outlier
        doesn't distort the fill value.

        Values: [10, 20, 30, 1000]
          Mean   = 265.0  ← skewed by 1000
          Median = 25.0   ← robust, middle value
        """
        df = pd.DataFrame({"score": [10.0, 20.0, 30.0, 1000.0, None]})
        result = fill_missing_values(df)

        filled_value = result["score"][4]
        median_value = pd.Series([10.0, 20.0, 30.0, 1000.0]).median()  # 25.0

        assert filled_value == median_value, (
            f"Expected median {median_value}, got {filled_value}. "
            "Are you filling with mean instead of median?"
        )

    def test_columns_without_missing_values_unchanged(self):
        """
        Columns that are already complete must not be modified.
        """
        df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [30, 25]})
        result = fill_missing_values(df)

        assert list(result["name"]) == ["Alice", "Bob"]
        assert list(result["age"])  == [30, 25]


# ════════════════════════════════════════════════════════════════
#  SECTION 6 — clean_dataset()  (full pipeline)
# ════════════════════════════════════════════════════════════════

class TestCleanDataset:
    """
    Integration-style tests for the complete cleaning pipeline.

    clean_dataset() now returns a tuple: (DataFrame, report_dict).
    Every test unpacks the tuple — result, report = clean_dataset(df) —
    so that assertions can target either the cleaned data or the report.
    """

    def test_pipeline_cleans_column_names(self):
        """
        clean_dataset() must produce snake_case column headers,
        even when the raw headers have mixed case and spaces.
        """
        df = pd.DataFrame({"First Name": ["Alice"], "Last Name": ["Smith"]})
        result, report = clean_dataset(df)

        assert "first_name" in result.columns
        assert "last_name"  in result.columns

    def test_pipeline_removes_duplicates(self):
        """
        clean_dataset() must deduplicate rows as part of the pipeline.
        """
        df = pd.DataFrame({
            "name":   ["Alice", "Alice", "Bob"],
            "salary": [70000,   70000,   80000],
        })
        result, report = clean_dataset(df)

        assert len(result) == 2, f"Expected 2 rows after dedup, got {len(result)}."

    def test_pipeline_removes_empty_columns(self):
        """
        clean_dataset() must drop columns that are entirely NaN.
        """
        df = pd.DataFrame({
            "name":  ["Alice", "Bob"],
            "notes": [None, None],
        })
        result, report = clean_dataset(df)

        assert "notes" not in result.columns

    def test_pipeline_fills_all_missing_values(self):
        """
        clean_dataset() must leave no NaN values in the output.
        """
        df = pd.DataFrame({
            "name":   ["Alice", None],
            "salary": [50000.0, None],
        })
        result, report = clean_dataset(df)

        assert result.isna().sum().sum() == 0, "NaN values remain after full pipeline."

    def test_pipeline_handles_real_world_messy_dataframe(self):
        """
        Stress test: apply all types of messiness at once and verify
        the final output is fully clean.
        """
        df = pd.DataFrame({
            "  First Name  ": ["  alice  ", "BOB",      "  alice  ", None    ],
            "  SALARY  ":     [50000.0,     None,        50000.0,    75000.0 ],
            "Empty Col":      [None,         None,        None,       None    ],
        })

        result, report = clean_dataset(df)

        assert "first_name" in result.columns,       "Column name not cleaned."
        assert "salary"     in result.columns,       "Column name not cleaned."
        assert "empty_col"  not in result.columns,   "Empty column was not removed."
        assert len(result) == 3,                     f"Expected 3 unique rows, got {len(result)}."
        assert "Alice" in result["first_name"].values, "Text value not title-cased."
        assert "Bob"   in result["first_name"].values, "Text value not title-cased."
        assert result.isna().sum().sum() == 0,       "NaN values remain after pipeline."

    def test_pipeline_returns_tuple(self):
        """
        clean_dataset() must return a tuple of (DataFrame, dict).
        The first element must be a DataFrame, the second a dict.
        """
        df = pd.DataFrame({"name": ["Alice"], "age": [30]})
        output = clean_dataset(df)

        assert isinstance(output, tuple),            "clean_dataset must return a tuple."
        assert isinstance(output[0], pd.DataFrame),  "First element must be a DataFrame."
        assert isinstance(output[1], dict),           "Second element must be a dict."

    def test_pipeline_does_not_mutate_input(self):
        """
        The original DataFrame passed to clean_dataset() must
        remain completely unchanged — the pipeline works on copies.
        """
        df = pd.DataFrame({"First Name": ["  alice  "], "Salary": [None]})
        original_shape   = df.shape
        original_columns = list(df.columns)

        clean_dataset(df)

        assert df.shape          == original_shape,   "Input shape was mutated."
        assert list(df.columns)  == original_columns, "Input columns were mutated."

    def test_report_contains_quality_scores(self):
        """
        The report dict must contain quality_score_before, quality_score_after,
        and quality_improvement keys with numeric values.
        """
        df = pd.DataFrame({
            "name":   ["Alice", None],
            "salary": [50000.0, None],
        })
        result, report = clean_dataset(df)

        assert "quality_score_before"  in report, "Missing quality_score_before in report."
        assert "quality_score_after"   in report, "Missing quality_score_after in report."
        assert "quality_improvement"   in report, "Missing quality_improvement in report."
        assert isinstance(report["quality_score_before"], float)
        assert isinstance(report["quality_score_after"],  float)
        assert isinstance(report["quality_improvement"],  float)

    def test_quality_score_improves_after_cleaning(self):
        """
        The quality score after cleaning must be greater than or equal to
        the score before cleaning. Cleaning should never make data worse.
        """
        df = pd.DataFrame({
            "name":   ["Alice", "Alice", None],   # duplicate + missing
            "salary": [50000.0, 50000.0, None],   # duplicate + missing
            "empty":  [None, None, None],          # fully empty column
        })
        result, report = clean_dataset(df)

        assert report["quality_score_after"] >= report["quality_score_before"], (
            "Quality score decreased after cleaning — cleaning made the data worse."
        )


# ════════════════════════════════════════════════════════════════
#  SECTION 7 — quality_score module
# ════════════════════════════════════════════════════════════════

class TestCalculateQualityMetrics:
    """Tests for the calculate_quality_metrics() function."""

    def test_returns_correct_row_and_column_counts(self):
        """
        The metrics dict must report the exact shape of the DataFrame.
        """
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        metrics = calculate_quality_metrics(df)

        assert metrics["rows"]    == 3
        assert metrics["columns"] == 2

    def test_calculates_missing_percent_correctly(self):
        """
        missing_percent = (total NaN cells / total cells) * 100.
        DataFrame: 2 rows x 2 cols = 4 cells, 1 is NaN → 25%.
        """
        df = pd.DataFrame({"a": [1, None], "b": [3, 4]})
        metrics = calculate_quality_metrics(df)

        assert metrics["missing_percent"] == 25.0

    def test_zero_missing_when_data_is_complete(self):
        """
        A fully populated DataFrame must report 0.0% missing.
        """
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        metrics = calculate_quality_metrics(df)

        assert metrics["missing_percent"] == 0.0

    def test_calculates_duplicate_percent_correctly(self):
        """
        duplicate_percent = (duplicate row count / total rows) * 100.
        3 rows, 1 duplicate → 33.33%.
        """
        df = pd.DataFrame({"a": [1, 2, 1], "b": ["x", "y", "x"]})
        metrics = calculate_quality_metrics(df)

        assert metrics["duplicate_percent"] == round((1 / 3) * 100, 2)

    def test_zero_duplicates_when_all_rows_unique(self):
        """
        A DataFrame with no duplicate rows must report 0.0% duplicates.
        """
        df = pd.DataFrame({"a": [1, 2, 3]})
        metrics = calculate_quality_metrics(df)

        assert metrics["duplicate_percent"] == 0.0

    def test_counts_empty_columns_correctly(self):
        """
        empty_columns counts columns where every value is NaN.
        """
        df = pd.DataFrame({
            "good":  [1, 2],
            "empty": [None, None],
        })
        metrics = calculate_quality_metrics(df)

        assert metrics["empty_columns"] == 1

    def test_zero_empty_columns_when_all_have_data(self):
        """
        A DataFrame with no fully-empty columns must report empty_columns = 0.
        """
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        metrics = calculate_quality_metrics(df)

        assert metrics["empty_columns"] == 0


class TestCalculateQualityScore:
    """Tests for the calculate_quality_score() function."""

    def test_perfect_data_scores_100(self):
        """
        A DataFrame with no missing values, no duplicates, and no empty
        columns must score exactly 100.0.
        """
        df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [30, 25]})
        score = calculate_quality_score(df)

        assert score == 100.0, f"Expected 100.0, got {score}."

    def test_score_decreases_with_missing_values(self):
        """
        Adding missing values must reduce the score below 100.
        """
        df_clean   = pd.DataFrame({"a": [1, 2, 3, 4]})
        df_missing = pd.DataFrame({"a": [1, None, None, None]})

        score_clean   = calculate_quality_score(df_clean)
        score_missing = calculate_quality_score(df_missing)

        assert score_missing < score_clean, (
            "Score should decrease when missing values are present."
        )

    def test_score_decreases_with_duplicates(self):
        """
        Adding duplicate rows must reduce the score below 100.
        """
        df_unique = pd.DataFrame({"a": [1, 2, 3]})
        df_dupes  = pd.DataFrame({"a": [1, 1, 1]})

        assert calculate_quality_score(df_dupes) < calculate_quality_score(df_unique)

    def test_score_decreases_with_empty_columns(self):
        """
        Adding empty columns must reduce the score below 100.
        """
        df_good  = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df_empty = pd.DataFrame({"a": [1, 2], "b": [None, None]})

        assert calculate_quality_score(df_empty) < calculate_quality_score(df_good)

    def test_score_never_goes_below_zero(self):
        """
        Even with 100% missing values, 100% duplicates, and all empty columns,
        the score must be clamped to a minimum of 0.0 — never negative.
        """
        df = pd.DataFrame({
            "a": [None, None],
            "b": [None, None],
        })
        score = calculate_quality_score(df)

        assert score >= 0.0, f"Score went negative: {score}."

    def test_score_never_exceeds_100(self):
        """
        The score must be capped at 100.0 — it cannot exceed perfect.
        """
        df    = pd.DataFrame({"a": [1, 2, 3]})
        score = calculate_quality_score(df)

        assert score <= 100.0, f"Score exceeded 100: {score}."

    def test_score_is_float(self):
        """
        The return type must always be a float, not an int or None.
        """
        df    = pd.DataFrame({"a": [1, 2]})
        score = calculate_quality_score(df)

        assert isinstance(score, float), f"Expected float, got {type(score)}."


class TestScoreLabel:
    """Tests for the score_label() helper function."""

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

    def test_24_is_poor(self):
        assert score_label(24.0) == "Poor"

    def test_zero_is_poor(self):
        assert score_label(0.0) == "Poor"
