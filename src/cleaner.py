"""
SmartCleaner - src/cleaner.py
------------------------------
Core data cleaning engine for CSV files.
"""

import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quality_score import calculate_quality_score, calculate_quality_metrics
from outlier_detection import remove_outliers
from knn_imputer import knn_fill_numeric


# ── Helper ────────────────────────────────────────────────────────────────────

def _text_cols(df: pd.DataFrame) -> list:
    """Return all non-numeric column names. Works on all Pandas versions."""
    return df.select_dtypes(exclude="number").columns.tolist()


# ──────────────────────────────────────────────
# 1. CLEAN COLUMN NAMES
# ──────────────────────────────────────────────

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip, lowercase, and underscore every column header."""
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    return df


# ──────────────────────────────────────────────
# 2. REMOVE EMPTY COLUMNS
# ──────────────────────────────────────────────

def remove_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop any column that contains only NaN values."""
    df = df.copy()
    before = df.shape[1]
    df = df.dropna(axis=1, how="all")
    dropped = before - df.shape[1]
    if dropped:
        print(f"  [remove_empty_columns] Dropped {dropped} fully-empty column(s).")
    return df


# ──────────────────────────────────────────────
# 3. REMOVE DUPLICATE ROWS
# ──────────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows. Keep first occurrence."""
    df = df.copy()
    before = len(df)
    df = df.drop_duplicates(keep="first")
    removed = before - len(df)
    if removed:
        print(f"  [remove_duplicates] Removed {removed} duplicate row(s).")
    df = df.reset_index(drop=True)
    return df


# ──────────────────────────────────────────────
# 4. CLEAN TEXT VALUES
# ──────────────────────────────────────────────

def clean_text_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip whitespace and apply Title Case to every non-numeric column.
    NaN values are preserved exactly — fill_missing_values handles them later.
    select_dtypes(exclude='number') catches all string dtypes on all
    Pandas versions (object, StringDtype, etc).
    """
    df = df.copy()
    for col in _text_cols(df):
        # Only apply to non-null cells so NaN stays NaN
        mask = df[col].notna()
        if mask.any():
            df.loc[mask, col] = (
                df.loc[mask, col]
                .astype(str)
                .str.strip()
                .str.title()
            )
    return df


# ──────────────────────────────────────────────
# 5. FILL MISSING VALUES (text only)
# ──────────────────────────────────────────────

def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill NaN in non-numeric columns with 'Unknown'.
    Numeric columns are handled by knn_fill_numeric().
    """
    df = df.copy()
    for col in _text_cols(df):
        missing_count = df[col].isna().sum()
        if missing_count:
            df[col] = df[col].fillna("Unknown")
            print(f"  [fill_missing_values] '{col}': filled {missing_count} "
                  f"text gap(s) with 'Unknown'.")
    return df


# ──────────────────────────────────────────────
# 6. MASTER CLEANING PIPELINE
# ──────────────────────────────────────────────

def clean_dataset(df: pd.DataFrame) -> tuple:
    """
    7-step cleaning pipeline. Returns (cleaned_df, report_dict).

    Steps:
      1. clean_column_names
      2. remove_empty_columns + drop empty rows
      3. remove_duplicates
      4. clean_text_values
      5. knn_fill_numeric  (AI numeric imputation)
      6. fill_missing_values (text -> Unknown)
      7. remove_outliers
    """
    score_before   = calculate_quality_score(df)
    metrics_before = calculate_quality_metrics(df)

    report = {
        "original_rows":           len(df),
        "original_columns":        df.shape[1],
        "empty_rows_dropped":      0,
        "empty_columns_dropped":   0,
        "duplicate_rows_removed":  0,
        "text_columns_normalised": [],
        "missing_filled":          {},
        "knn_columns_filled":      [],
        "knn_values_filled":       0,
        "knn_neighbors_used":      0,
        "outliers_removed":        0,
        "outlier_percent":         0.0,
        "outlier_columns_used":    [],
        "quality_score_before":    score_before,
        "quality_score_after":     0.0,
        "quality_improvement":     0.0,
        "metrics_before":          metrics_before,
        "metrics_after":           {},
        "final_rows":              0,
        "final_columns":           0,
    }

    print("\nStarting SmartCleaner pipeline...")
    print(f"  Input  -> {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Quality score before cleaning: {score_before}/100\n")

    # Step 1
    df = clean_column_names(df)
    print("  Step 1/7 - Column names cleaned.")

    # Step 2
    before_cols = df.shape[1]
    df = remove_empty_columns(df)
    report["empty_columns_dropped"] = before_cols - df.shape[1]
    print("  Step 2/7 - Empty columns removed.")

    before_rows = len(df)
    df = df.dropna(how="all").reset_index(drop=True)
    report["empty_rows_dropped"] = before_rows - len(df)
    if report["empty_rows_dropped"]:
        print(f"  [pipeline] Dropped {report['empty_rows_dropped']} fully-empty row(s).")

    # Step 3
    before_rows = len(df)
    df = remove_duplicates(df)
    report["duplicate_rows_removed"] = before_rows - len(df)
    print("  Step 3/7 - Duplicate rows removed.")

    # Step 4
    report["text_columns_normalised"] = _text_cols(df)
    df = clean_text_values(df)
    print("  Step 4/7 - Text values normalised.")

    # Step 5 — KNN numeric imputation
    df, knn_report = knn_fill_numeric(df)
    report["knn_columns_filled"] = knn_report["columns_filled"]
    report["knn_values_filled"]  = knn_report["total_values_filled"]
    report["knn_neighbors_used"] = knn_report["n_neighbors_used"]
    print("  Step 5/7 - Numeric gaps filled using KNN Imputation.")

    # Step 6 — text fill
    for col in _text_cols(df):
        n = int(df[col].isna().sum())
        if n:
            report["missing_filled"][col] = {"count": n, "fill_value": "Unknown"}
    df = fill_missing_values(df)
    print("  Step 6/7 - Text gaps filled with 'Unknown'.")

    # Step 7 — outliers
    df, outlier_report = remove_outliers(df)
    report["outliers_removed"]     = outlier_report["outliers_removed"]
    report["outlier_percent"]      = outlier_report["outlier_percent"]
    report["outlier_columns_used"] = outlier_report["numeric_columns_used"]
    print("  Step 7/7 - Outliers detected and removed.")

    # Final quality score
    score_after   = calculate_quality_score(df)
    metrics_after = calculate_quality_metrics(df)

    report["quality_score_after"] = score_after
    report["quality_improvement"] = round(score_after - score_before, 2)
    report["metrics_after"]       = metrics_after
    report["final_rows"]          = len(df)
    report["final_columns"]       = df.shape[1]

    print(f"\nCleaning complete!")
    print(f"  Output -> {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Quality score after cleaning:  {score_after}/100")
    print(f"  Improvement: +{report['quality_improvement']} points")
    if report["knn_values_filled"]:
        print(f"  KNN filled: {report['knn_values_filled']} numeric gap(s)")
    print(f"  Outliers removed: {report['outliers_removed']}\n")

    return df, report
