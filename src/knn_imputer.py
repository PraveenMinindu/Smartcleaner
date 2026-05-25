"""
SmartCleaner - src/knn_imputer.py
-----------------------------------
AI-powered missing value filling using K-Nearest Neighbors (KNN).

WHY THIS EXISTS — THE PROBLEM WITH MEDIAN FILL
------------------------------------------------
The old approach fills every missing number with the median of that
column. The median is the middle value of all values in that column.

The problem: it ignores every other column.

Example:
  You have age, department, and salary columns.
  A row has age=28, department=Engineering, salary=???

  Median fill looks only at the salary column and uses the middle
  salary of the entire dataset. It does not care that this person
  is 28 years old and works in Engineering.

  KNN fill looks at this row and asks: "Which other rows in the
  dataset are most similar to this one — similar age, similar
  department?" Then it uses the salaries of those similar rows
  to fill the gap. Much smarter. Much more accurate.

HOW KNN IMPUTATION WORKS
--------------------------
KNN stands for K-Nearest Neighbors. The K is a number you choose,
like 5. For each row with a missing value:

  Step 1: Find the 5 most similar rows (neighbors) in the dataset
          based on all other columns that DO have values.

  Step 2: Take the average of that column's value across those
          5 neighbors.

  Step 3: Use that average as the fill value.

The result is a fill value that reflects the actual patterns in
your data, not just the overall middle value.

WHAT THIS MODULE DOES
----------------------
  knn_fill_numeric(df)  -> tuple(DataFrame, dict)
      Fills missing values in ALL numeric columns using KNN.
      Text columns are not touched — they still use "Unknown".
      Returns the filled DataFrame and a report of what changed.
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer


# ── Default number of neighbors ───────────────────────────────────────────────
# K=5 means: look at the 5 most similar rows to fill each gap.
# This is the standard default used in most data science applications.
# Higher K = smoother fills but slower.
# Lower K = fills closer to the actual nearest match but noisier.
DEFAULT_N_NEIGHBORS = 5


def knn_fill_numeric(
    df: pd.DataFrame,
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
) -> tuple:
    """
    Fill missing values in numeric columns using KNN Imputation.

    This replaces the old median-fill approach for numeric columns.
    Text columns are NOT touched by this function — they should
    still be filled with "Unknown" by fill_missing_values().

    How it works step by step:
      1. Identify all numeric columns in the DataFrame.
      2. Check which of those columns have at least one missing value.
      3. If none have missing values, return immediately — nothing to do.
      4. Extract just the numeric columns into a separate array.
      5. Run sklearn's KNNImputer on that array.
         KNNImputer finds the K most similar rows for each missing
         cell and fills it with the weighted average of their values.
      6. Put the filled values back into the original DataFrame.
      7. Return the filled DataFrame and a detailed report.

    Args:
        df: DataFrame after text cleaning and deduplication.
            Should still have NaN values in numeric columns.
        n_neighbors: Number of similar rows to look at when filling.
                     Default is 5. Must be at least 1.

    Returns:
        Tuple of (filled_df, report_dict).

        filled_df: DataFrame with numeric NaNs filled using KNN.
                   Text columns and their NaNs are unchanged.

        report_dict keys:
          - columns_filled:     list of column names that had gaps filled
          - total_values_filled: total number of NaN cells filled
          - fill_details:       per-column dict with count and method
          - n_neighbors_used:   the K value used
          - method:             always "KNN Imputation"
          - skipped_reason:     present only if KNN was skipped
    """
    df = df.copy()

    # ── Find numeric columns ──────────────────────────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        # No numeric columns — nothing for KNN to do
        return df, {
            "columns_filled":      [],
            "total_values_filled": 0,
            "fill_details":        {},
            "n_neighbors_used":    n_neighbors,
            "method":              "KNN Imputation",
            "skipped_reason":      "No numeric columns found.",
        }

    # ── Check which numeric columns actually have missing values ──────────────
    cols_with_missing = [
        col for col in numeric_cols
        if df[col].isna().sum() > 0
    ]

    if not cols_with_missing:
        # Numeric columns exist but none have missing values — nothing to do
        return df, {
            "columns_filled":      [],
            "total_values_filled": 0,
            "fill_details":        {},
            "n_neighbors_used":    n_neighbors,
            "method":              "KNN Imputation",
            "skipped_reason":      "No missing values found in numeric columns.",
        }

    # ── Record how many values are missing before filling ─────────────────────
    # We capture this now so the report is accurate
    fill_details = {}
    for col in cols_with_missing:
        missing_count = int(df[col].isna().sum())
        fill_details[col] = {
            "missing_count": missing_count,
            "method":        f"KNN (k={n_neighbors})",
        }

    total_values_filled = sum(d["missing_count"] for d in fill_details.values())

    # ── Adjust n_neighbors if dataset is too small ────────────────────────────
    # KNN cannot use more neighbors than there are complete rows.
    # If K=5 but only 3 rows exist, we must reduce K to 3.
    complete_rows = df[numeric_cols].dropna().shape[0]
    actual_k = min(n_neighbors, max(1, complete_rows))

    if actual_k != n_neighbors:
        print(f"  [knn_fill_numeric] Dataset has only {complete_rows} complete rows. "
              f"Reduced K from {n_neighbors} to {actual_k}.")

    # ── Run KNN Imputer ───────────────────────────────────────────────────────
    # KNNImputer works on numpy arrays so we extract numeric columns,
    # fill them, then put them back into the DataFrame.
    imputer = KNNImputer(n_neighbors=actual_k)

    # Extract numeric data as numpy array, impute, put back
    numeric_data_filled = imputer.fit_transform(df[numeric_cols])

    # Put the filled values back into the DataFrame column by column
    for i, col in enumerate(numeric_cols):
        df[col] = numeric_data_filled[:, i]

    # ── Print summary ─────────────────────────────────────────────────────────
    for col, detail in fill_details.items():
        print(f"  [knn_fill_numeric] '{col}': filled {detail['missing_count']} "
              f"gap(s) using KNN (k={actual_k}).")

    report = {
        "columns_filled":      cols_with_missing,
        "total_values_filled": total_values_filled,
        "fill_details":        fill_details,
        "n_neighbors_used":    actual_k,
        "method":              "KNN Imputation",
    }

    return df, report
