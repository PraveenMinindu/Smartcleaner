import pandas as pd
import os
import sys
import pathlib

# This makes sure Python can always find quality_score.py
# by loading it directly from the same folder as this file.
# It works regardless of how the project is launched.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quality_score import calculate_quality_score, calculate_quality_metrics

try:
    import yaml
except ImportError:
    yaml = None


# Default configuration values (used when config.yaml is missing or incomplete)
DEFAULT_CONFIG = {
    "fill_text": "Unknown",
    "fill_numeric": "median",
    "drop_empty_columns": True,
    "remove_duplicates": True,
}


def load_config(config_path: pathlib.Path = None) -> dict:
    """
    Load cleaning configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml. If None, searches for config.yaml in project root.
    
    Returns:
        Dictionary with config values, merged with defaults for any missing keys.
    """
    if config_path is None:
        # Search for config.yaml in the project root (parent of src/)
        config_path = pathlib.Path(__file__).parent.parent / "config.yaml"
    
    config = DEFAULT_CONFIG.copy()
    
    if yaml is None:
        print("  [load_config] PyYAML not installed; using defaults.")
        return config
    
    if not config_path.exists():
        print(f"  [load_config] config.yaml not found at {config_path}; using defaults.")
        return config
    
    try:
        with open(config_path, "r") as f:
            file_config = yaml.safe_load(f) or {}
        
        # Merge file config with defaults (file config takes precedence)
        config.update(file_config)
        
        # Validate numeric fill strategy
        if config["fill_numeric"] not in ["median", "mean", "mode"]:
            print(f"  [load_config] Invalid fill_numeric '{config['fill_numeric']}'; using 'median'.")
            config["fill_numeric"] = "median"
        
        print(f"  [load_config] Loaded config from {config_path}")
        
    except Exception as e:
        print(f"  [load_config] Error reading config.yaml: {e}; using defaults.")
        return DEFAULT_CONFIG.copy()
    
    return config


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    return df


def remove_empty_columns(df: pd.DataFrame, enabled: bool = True) -> pd.DataFrame:
    df = df.copy()
    if not enabled:
        return df
    
    before = df.shape[1]
    df = df.dropna(axis=1, how="all")
    after  = df.shape[1]
    dropped = before - after
    if dropped:
        print(f"  [remove_empty_columns] Dropped {dropped} fully-empty column(s).")
    return df


def remove_duplicates(df: pd.DataFrame, enabled: bool = True) -> pd.DataFrame:
    df = df.copy()
    if not enabled:
        return df
    
    before = len(df)
    df = df.drop_duplicates(keep="first")
    after   = len(df)
    removed = before - after
    if removed:
        print(f"  [remove_duplicates] Removed {removed} duplicate row(s).")
    df = df.reset_index(drop=True)
    return df


def clean_text_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    text_columns = df.select_dtypes(include="object").columns
    for col in text_columns:
        df[col] = df[col].str.strip().str.title()
    return df


def fill_missing_values(
    df: pd.DataFrame,
    fill_text: str = "Unknown",
    fill_numeric: str = "median",
) -> pd.DataFrame:
    """
    Fill missing values in the dataset.
    
    Args:
        df: DataFrame with missing values.
        fill_text: Value to use for text columns (e.g., "Unknown", "N/A").
        fill_numeric: Strategy for numeric columns: "median", "mean", or "mode".
    
    Returns:
        DataFrame with missing values filled.
    """
    df = df.copy()
    for col in df.columns:
        # Check if column is object (text/string) type
        if df[col].dtype == 'object':
            missing_count = df[col].isna().sum()
            if missing_count:
                df[col] = df[col].fillna(fill_text)
                print(f"  [fill_missing_values] '{col}': filled {missing_count} text gap(s) with '{fill_text}'.")
        # Check if column is numeric
        elif pd.api.types.is_numeric_dtype(df[col]):
            missing_count = df[col].isna().sum()
            if missing_count:
                if fill_numeric == "mean":
                    fill_val = df[col].mean()
                elif fill_numeric == "mode":
                    mode_result = df[col].mode()
                    fill_val = mode_result[0] if len(mode_result) > 0 else df[col].median()
                else:  # Default to median
                    fill_val = df[col].median()
                
                df[col] = df[col].fillna(fill_val)
                print(f"  [fill_missing_values] '{col}': filled {missing_count} numeric gap(s) with {fill_numeric} ({round(float(fill_val), 2)}).")
    return df


def clean_dataset(df: pd.DataFrame, config_path: pathlib.Path = None) -> tuple:
    """
    Clean a dataset using a configurable pipeline.
    
    Args:
        df: Raw DataFrame to clean.
        config_path: Optional path to config.yaml. If None, uses default config.
    
    Returns:
        Tuple of (cleaned_df, report_dict)
    """
    # Load configuration
    config = load_config(config_path)
    
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
        "quality_score_before":    score_before,
        "quality_score_after":     0.0,
        "quality_improvement":     0.0,
        "metrics_before":          metrics_before,
        "metrics_after":           {},
        "final_rows":              0,
        "final_columns":           0,
        "config_used":             config,
    }

    print("\nStarting SmartCleaner pipeline...")
    print(f"  Input  -> {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Quality score before cleaning: {score_before}/100\n")

    df = clean_column_names(df)
    print("  Step 1/5 - Column names cleaned.")

    before_cols = df.shape[1]
    df = remove_empty_columns(df, enabled=config["drop_empty_columns"])
    report["empty_columns_dropped"] = before_cols - df.shape[1]
    print(f"  Step 2/5 - Empty columns {'removed' if config['drop_empty_columns'] else 'kept'}.")

    before_rows = len(df)
    df = df.dropna(how="all").reset_index(drop=True)
    report["empty_rows_dropped"] = before_rows - len(df)
    if report["empty_rows_dropped"]:
        print(f"  [pipeline] Dropped {report['empty_rows_dropped']} fully-empty row(s).")

    before_rows = len(df)
    df = remove_duplicates(df, enabled=config["remove_duplicates"])
    report["duplicate_rows_removed"] = before_rows - len(df)
    print(f"  Step 3/5 - Duplicate rows {'removed' if config['remove_duplicates'] else 'kept'}.")

    text_cols = df.select_dtypes(include="object").columns.tolist()
    df = clean_text_values(df)
    report["text_columns_normalised"] = text_cols
    print("  Step 4/5 - Text values normalised.")

    # Build missing_filled report before filling
    for col in df.columns:
        if df[col].dtype == 'object':
            n = df[col].isna().sum()
            if n:
                report["missing_filled"][col] = {"count": int(n), "fill_value": config["fill_text"]}
        elif pd.api.types.is_numeric_dtype(df[col]):
            n = df[col].isna().sum()
            if n:
                # Calculate what the fill value will be based on strategy
                if config["fill_numeric"] == "mean":
                    fill_val = df[col].mean()
                elif config["fill_numeric"] == "mode":
                    mode_result = df[col].mode()
                    fill_val = mode_result[0] if len(mode_result) > 0 else df[col].median()
                else:  # median (default)
                    fill_val = df[col].median()
                
                report["missing_filled"][col] = {"count": int(n), "fill_value": round(float(fill_val), 2)}

    df = fill_missing_values(
        df,
        fill_text=config["fill_text"],
        fill_numeric=config["fill_numeric"],
    )
    print("  Step 5/5 - Missing values filled.")

    score_after   = calculate_quality_score(df)
    metrics_after = calculate_quality_metrics(df)

    report["quality_score_after"]  = score_after
    report["quality_improvement"]  = round(score_after - score_before, 2)
    report["metrics_after"]        = metrics_after
    report["final_rows"]           = len(df)
    report["final_columns"]        = df.shape[1]

    print(f"\nCleaning complete!")
    print(f"  Output -> {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Quality score after cleaning: {score_after}/100")
    print(f"  Improvement: +{report['quality_improvement']} points\n")

    return df, report