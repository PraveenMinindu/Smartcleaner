"""
SmartCleaner - src/outlier_detection.py
-----------------------------------------
Outlier detection engine using Isolation Forest.

What is an outlier?
  An outlier is a value that is very different from all the other
  values in the same column. For example, if every salary in your
  dataset is between 30000 and 80000, and one row has 9999999,
  that row is an outlier. It might be a data entry error, a system
  glitch, or a genuinely unusual case.

Why Isolation Forest?
  Isolation Forest is an unsupervised machine learning algorithm
  designed specifically for outlier detection. It works by randomly
  splitting data and measuring how quickly each row gets isolated
  from all the others. Normal rows take many splits to isolate.
  Outliers get isolated very quickly because they are already far
  from everything else.

  Compared to simple statistical methods:
    - Z-Score: only works well when data follows a bell curve
    - IQR method: only looks at one column at a time
    - Isolation Forest: looks at all numeric columns together,
      finds outliers that only appear unusual in combination
      (e.g. age=25 is fine, salary=200000 is fine, but
       age=25 AND salary=200000 together is suspicious)

Public functions:
  detect_outliers(df)   -> dict   (just detect, do not remove)
  remove_outliers(df)   -> tuple  (remove and return report)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


# ── Default contamination rate ────────────────────────────────────────────────
# contamination = the proportion of rows we expect to be outliers.
# 0.05 means we expect about 5% of rows to be outliers.
# This is a standard default used in most real-world applications.
# A higher value (e.g. 0.1) flags more rows as outliers.
# A lower value (e.g. 0.01) flags fewer rows as outliers.
DEFAULT_CONTAMINATION = 0.05


def detect_outliers(
    df: pd.DataFrame,
    contamination: float = DEFAULT_CONTAMINATION,
    random_state: int = 42,
) -> dict:
    """
    Detect outlier rows in a DataFrame using Isolation Forest.

    This function ONLY detects — it does not remove anything.
    The original DataFrame is never modified.

    How it works:
      1. Select all numeric columns (text columns are ignored
         because Isolation Forest only works on numbers).
      2. Drop rows that have any NaN in numeric columns before
         running the algorithm (NaN confuses the model).
      3. Train an Isolation Forest on the numeric data.
      4. Each row gets a prediction: 1 = normal, -1 = outlier.
      5. Collect the row indices of all outlier rows.
      6. Return a detailed report about what was found.

    Args:
        df: Any pandas DataFrame — should be called after basic
            cleaning (column names cleaned, duplicates removed)
            but before filling missing values.
        contamination: Expected proportion of outliers in the data.
                       Default is 0.05 (5%). Range: 0.0 to 0.5.
        random_state: Seed for reproducibility. Same seed always
                      gives the same result on the same data.

    Returns:
        dict with keys:
          - outlier_count:   number of rows flagged as outliers
          - outlier_indices: list of row index numbers
          - outlier_percent: percentage of total rows flagged
          - numeric_columns_used: which columns the model analysed
          - contamination_used: the contamination value used
          - has_outliers: True if any outliers were found
    """
    df_work = df.copy()

    # ── Select numeric columns only ───────────────────────────────────────────
    # Isolation Forest cannot process text. We only pass numeric columns.
    numeric_cols = df_work.select_dtypes(include=[np.number]).columns.tolist()

    # If there are no numeric columns, outlier detection cannot run.
    # Return an empty report rather than crashing.
    if len(numeric_cols) == 0:
        return {
            "outlier_count":         0,
            "outlier_indices":       [],
            "outlier_percent":       0.0,
            "numeric_columns_used":  [],
            "contamination_used":    contamination,
            "has_outliers":          False,
            "skipped_reason":        "No numeric columns found in dataset.",
        }

    # ── Prepare data for the model ────────────────────────────────────────────
    # Get only the numeric columns
    numeric_data = df_work[numeric_cols]

    # Find rows where ALL numeric columns have a value (no NaN).
    # We can only run the model on complete rows.
    complete_mask = numeric_data.notna().all(axis=1)
    complete_data = numeric_data[complete_mask]

    # If too few complete rows, skip detection (model needs at least 2 rows).
    if len(complete_data) < 2:
        return {
            "outlier_count":         0,
            "outlier_indices":       [],
            "outlier_percent":       0.0,
            "numeric_columns_used":  numeric_cols,
            "contamination_used":    contamination,
            "has_outliers":          False,
            "skipped_reason":        "Not enough complete rows to run outlier detection.",
        }

    # ── Train Isolation Forest ────────────────────────────────────────────────
    # n_estimators=100 means the model builds 100 decision trees.
    # More trees = more accurate but slower. 100 is a good balance.
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=random_state,
    )

    # fit_predict trains the model and predicts at the same time.
    # Returns: 1 for normal rows, -1 for outlier rows.
    predictions = model.fit_predict(complete_data)

    # ── Collect outlier row indices ───────────────────────────────────────────
    # Get the actual DataFrame index values (not position numbers)
    # for every row the model labelled as -1 (outlier).
    complete_indices  = complete_data.index.tolist()
    outlier_indices   = [
        idx for idx, pred in zip(complete_indices, predictions)
        if pred == -1
    ]

    outlier_count   = len(outlier_indices)
    total_rows      = len(df_work)
    outlier_percent = round((outlier_count / total_rows) * 100, 2) if total_rows > 0 else 0.0

    return {
        "outlier_count":         outlier_count,
        "outlier_indices":       outlier_indices,
        "outlier_percent":       outlier_percent,
        "numeric_columns_used":  numeric_cols,
        "contamination_used":    contamination,
        "has_outliers":          outlier_count > 0,
    }


def remove_outliers(
    df: pd.DataFrame,
    contamination: float = DEFAULT_CONTAMINATION,
    random_state: int = 42,
) -> tuple:
    """
    Detect and remove outlier rows from a DataFrame.

    This function calls detect_outliers() internally, then drops
    the flagged rows and returns both the cleaned DataFrame and
    a full report of what was removed.

    The original DataFrame is never modified — this function
    always works on a copy.

    Args:
        df: Any pandas DataFrame.
        contamination: Expected proportion of outliers. Default 0.05.
        random_state: Seed for reproducibility. Default 42.

    Returns:
        Tuple of (cleaned_df, report_dict).

        cleaned_df: DataFrame with outlier rows removed and
                    index reset to 0, 1, 2, ...

        report_dict keys:
          - rows_before:         row count before removal
          - rows_after:          row count after removal
          - outliers_removed:    number of rows dropped
          - outlier_percent:     percentage of rows that were outliers
          - numeric_columns_used: columns the model analysed
          - outlier_indices:     original index numbers of removed rows
          - has_outliers:        True if any rows were removed
          - contamination_used:  contamination value used
    """
    df = df.copy()
    rows_before = len(df)

    # Run detection
    detection = detect_outliers(df, contamination=contamination, random_state=random_state)

    # Remove the flagged rows
    if detection["has_outliers"]:
        df = df.drop(index=detection["outlier_indices"])
        df = df.reset_index(drop=True)
        print(f"  [remove_outliers] Removed {detection['outlier_count']} outlier row(s) "
              f"({detection['outlier_percent']}% of data) "
              f"using Isolation Forest on columns: {detection['numeric_columns_used']}")
    else:
        reason = detection.get("skipped_reason", "No outliers detected.")
        print(f"  [remove_outliers] {reason}")

    rows_after = len(df)

    report = {
        "rows_before":          rows_before,
        "rows_after":           rows_after,
        "outliers_removed":     detection["outlier_count"],
        "outlier_percent":      detection["outlier_percent"],
        "numeric_columns_used": detection["numeric_columns_used"],
        "outlier_indices":      detection["outlier_indices"],
        "has_outliers":         detection["has_outliers"],
        "contamination_used":   contamination,
    }

    return df, report


def get_outlier_rows(df: pd.DataFrame, contamination: float = DEFAULT_CONTAMINATION) -> pd.DataFrame:
    """
    Return just the outlier rows as a DataFrame for inspection.

    Useful when you want to review what would be removed before
    actually removing it. Does not modify the original DataFrame.

    Args:
        df: Any pandas DataFrame.
        contamination: Expected proportion of outliers. Default 0.05.

    Returns:
        DataFrame containing only the rows flagged as outliers.
        Returns an empty DataFrame if no outliers are found.
    """
    detection = detect_outliers(df, contamination=contamination)

    if not detection["has_outliers"]:
        return pd.DataFrame()

    return df.loc[detection["outlier_indices"]].copy()
