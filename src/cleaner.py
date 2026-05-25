"""
SmartCleaner - src/cleaner.py
------------------------------
Core data cleaning engine for CSV files.
Each function handles one specific cleaning responsibility,
making the code easy to understand, test, and extend.
"""

import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quality_score import calculate_quality_score, calculate_quality_metrics
from outlier_detection import remove_outliers
from knn_imputer import knn_fill_numeric


# ──────────────────────────────────────────────
# 1. CLEAN COLUMN NAMES
# ──────────────────────────────────────────────

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise every column header so they are consistent and
    safe to use as Python attribute names.

    Steps performed:
      - Strip leading/trailing whitespace  (e.g. ' Email Address ' → 'Email Address')
      - Convert to lowercase               (e.g. 'First Name'      → 'first name')
      - Replace spaces with underscores    (e.g. 'first name'      → 'first_name')

    Args:
        df: Raw DataFrame with potentially messy column names.

    Returns:
        DataFrame with clean, snake_case column names.
    """
    df = df.copy()  # Never mutate the caller's DataFrame
    df.columns = (
        df.columns
        .str.strip()          # Remove surrounding whitespace
        .str.lower()          # Lowercase everything
        .str.replace(" ", "_", regex=False)  # Spaces → underscores
    )
    return df


# ──────────────────────────────────────────────
# 2. REMOVE EMPTY COLUMNS
# ──────────────────────────────────────────────

def remove_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop any column that contains *only* NaN / empty values
    — these carry zero information and waste memory.

    Args:
        df: DataFrame that may contain fully-empty columns.

    Returns:
        DataFrame with all-NaN columns removed.
    """
    df = df.copy()

    # dropna(axis=1) drops columns; how='all' requires every value to be NaN
    before = df.shape[1]
    df = df.dropna(axis=1, how="all")
    after = df.shape[1]

    dropped = before - after
    if dropped:
        print(f"  [remove_empty_columns] Dropped {dropped} fully-empty column(s).")

    return df


# ──────────────────────────────────────────────
# 3. REMOVE DUPLICATE ROWS
# ──────────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows that are exact duplicates of a previous row.

    Only the *first* occurrence of each unique row is kept;
    all subsequent copies are deleted.

    Args:
        df: DataFrame that may contain duplicate rows.

    Returns:
        DataFrame with duplicate rows removed.
    """
    df = df.copy()

    before = len(df)
    df = df.drop_duplicates(keep="first")
    after = len(df)

    removed = before - after
    if removed:
        print(f"  [remove_duplicates] Removed {removed} duplicate row(s).")

    # Reset the index so row numbers are continuous (0, 1, 2 …)
    df = df.reset_index(drop=True)
    return df


# ──────────────────────────────────────────────
# 4. CLEAN TEXT VALUES
# ──────────────────────────────────────────────

def clean_text_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise the *content* of every text (object) column:

      - Strip leading/trailing whitespace from each cell value
      - Convert the entire value to Title Case
        (e.g. '  DIANA  ' → 'Diana', 'jane' → 'Jane')

    Numeric columns are left completely untouched.

    Args:
        df: DataFrame whose string columns need normalisation.

    Returns:
        DataFrame with clean, consistently-cased string values.
    """
    df = df.copy()

    # Select only columns that Pandas treats as text.
    # We use 'str' for Pandas 2.x (StringDtype) and keep 'object' for
    # backwards-compatibility with 1.x — union covers both.
    text_columns = df.select_dtypes(include="object").columns

    for col in text_columns:
        df[col] = (
            df[col]
            .str.strip()    # Remove surrounding whitespace
            .str.title()    # Title-case  (handles None/NaN safely)
        )

    return df


# ──────────────────────────────────────────────
# 5. FILL MISSING VALUES
# ──────────────────────────────────────────────

def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values in TEXT columns only.

    Numeric columns are now handled by knn_fill_numeric() which
    uses KNN Imputation — a smarter, AI-powered approach.

    This function only fills text (string) columns with "Unknown".
    It deliberately skips numeric columns so KNN can handle them.

    Args:
        df: DataFrame that may contain NaN values in text columns.

    Returns:
        DataFrame with text NaN values replaced with "Unknown".
        Numeric columns are unchanged — KNN will handle them.
    """
    df = df.copy()

    for col in df.columns:
        # Only process text columns — skip numeric entirely
        if pd.api.types.is_string_dtype(df[col]) and not pd.api.types.is_numeric_dtype(df[col]):
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
    Run every cleaning step in the correct order and return
    the fully-cleaned DataFrame plus a detailed report.

    Pipeline order matters:
      1. clean_column_names   — standardise headers first so later
                                steps can reference columns reliably.
      2. remove_empty_columns — drop useless columns early to reduce work.
      3. remove_duplicates    — remove identical rows before we fill gaps
                                (avoids filling then deduplicating).
      4. clean_text_values    — normalise string content.
      5. fill_missing_values  — fill remaining NaNs last, after duplicates
                                and empty columns are already gone.

    Quality scoring runs BEFORE step 1 and AFTER step 5 so the report
    shows the full improvement from raw to clean.

    Args:
        df: Raw, uncleaned DataFrame.

    Returns:
        Tuple of (cleaned_df, report_dict).
        report_dict contains row/column counts, change summaries,
        missing value details, and before/after quality scores.
    """
    # ── Quality score BEFORE cleaning ───────────────────────────────────────
    # Measure the raw DataFrame before any transformation.
    # This gives the baseline that we compare against after cleaning.
    score_before   = calculate_quality_score(df)
    metrics_before = calculate_quality_metrics(df)

    report = {
        "original_rows":          len(df),
        "original_columns":       df.shape[1],
        "empty_rows_dropped":     0,
        "empty_columns_dropped":  0,
        "duplicate_rows_removed": 0,
        "text_columns_normalised": [],
        "missing_filled":         {},
        "knn_columns_filled":     [],
        "knn_values_filled":      0,
        "knn_neighbors_used":     0,
        "outliers_removed":       0,
        "outlier_percent":        0.0,
        "outlier_columns_used":   [],
        "quality_score_before":   score_before,
        "quality_score_after":    0.0,
        "quality_improvement":    0.0,
        "metrics_before":         metrics_before,
        "metrics_after":          {},
        "final_rows":             0,
        "final_columns":          0,
    }

    print("\nStarting SmartCleaner pipeline...")
    print(f"  Input  -> {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Quality score before cleaning: {score_before}/100\n")

    # ── Step 1 — clean column names ──────────────────────────────────────────
    df = clean_column_names(df)
    print("  Step 1/7 - Column names cleaned.")

    # ── Step 2 — remove empty columns ───────────────────────────────────────
    before_cols = df.shape[1]
    df = remove_empty_columns(df)
    report["empty_columns_dropped"] = before_cols - df.shape[1]
    print("  Step 2/7 - Empty columns removed.")

    # ── Step 2b — drop fully-empty rows ─────────────────────────────────────
    before_rows = len(df)
    df = df.dropna(how="all").reset_index(drop=True)
    report["empty_rows_dropped"] = before_rows - len(df)
    if report["empty_rows_dropped"]:
        print(f"  [pipeline] Dropped {report['empty_rows_dropped']} fully-empty row(s).")

    # ── Step 3 — remove duplicates ───────────────────────────────────────────
    before_rows = len(df)
    df = remove_duplicates(df)
    report["duplicate_rows_removed"] = before_rows - len(df)
    print("  Step 3/7 - Duplicate rows removed.")

    # ── Step 4 — clean text values ───────────────────────────────────────────
    text_cols = df.select_dtypes(include="object").columns.tolist()
    df = clean_text_values(df)
    report["text_columns_normalised"] = text_cols
    print("  Step 4/7 - Text values normalised.")

    # ── Step 5 — KNN fill for numeric columns (AI-powered) ───────────────────
    # KNN Imputation looks at similar rows to fill missing numbers.
    # This is smarter than median because it uses patterns across
    # all numeric columns together, not just one column at a time.
    df, knn_report = knn_fill_numeric(df)
    report["knn_columns_filled"] = knn_report["columns_filled"]
    report["knn_values_filled"]  = knn_report["total_values_filled"]
    report["knn_neighbors_used"] = knn_report["n_neighbors_used"]
    print("  Step 5/7 - Numeric gaps filled using KNN Imputation.")

    # ── Step 6 — fill remaining text missing values ───────────────────────────
    # After KNN handles all numeric columns, fill text columns
    # with "Unknown". Record what gets filled for the report.
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) and not pd.api.types.is_numeric_dtype(df[col]):
            n = df[col].isna().sum()
            if n:
                report["missing_filled"][col] = {"count": int(n), "fill_value": "Unknown"}

    df = fill_missing_values(df)
    print("  Step 6/7 - Text gaps filled with 'Unknown'.")

    # ── Step 7 — detect and remove outliers ──────────────────────────────────
    # Outlier detection runs last — after all NaNs are filled —
    # because Isolation Forest cannot handle NaN values.
    df, outlier_report = remove_outliers(df)
    report["outliers_removed"]     = outlier_report["outliers_removed"]
    report["outlier_percent"]      = outlier_report["outlier_percent"]
    report["outlier_columns_used"] = outlier_report["numeric_columns_used"]
    print("  Step 7/7 - Outliers detected and removed.")

    # ── Quality score AFTER cleaning ─────────────────────────────────────────
    # Measure the cleaned DataFrame so we can show improvement.
    score_after   = calculate_quality_score(df)
    metrics_after = calculate_quality_metrics(df)

    report["quality_score_after"]  = score_after
    report["quality_improvement"]  = round(score_after - score_before, 2)
    report["metrics_after"]        = metrics_after
    report["final_rows"]           = len(df)
    report["final_columns"]        = df.shape[1]

    print(f"\nCleaning complete!")
    print(f"  Output -> {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Quality score after cleaning:  {score_after}/100")
    print(f"  Improvement: +{report['quality_improvement']} points")
    if report["knn_values_filled"]:
        print(f"  KNN filled: {report['knn_values_filled']} numeric gap(s) across {len(report['knn_columns_filled'])} column(s)")
    if report["outliers_removed"]:
        print(f"  Outliers removed: {report['outliers_removed']} rows ({report['outlier_percent']}%)\n")
    else:
        print(f"  Outliers removed: 0\n")

    return df, report
