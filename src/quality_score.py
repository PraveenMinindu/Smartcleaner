import pandas as pd

MISSING_VALUES_WEIGHT  = 40
DUPLICATE_ROWS_WEIGHT  = 30
EMPTY_COLUMNS_WEIGHT   = 30


def calculate_quality_metrics(df: pd.DataFrame) -> dict:
    total_rows    = len(df)
    total_columns = df.shape[1]
    total_cells   = total_rows * total_columns
    total_missing = int(df.isna().sum().sum())

    missing_percent   = round((total_missing / total_cells) * 100, 2) if total_cells > 0 else 0.0
    duplicate_count   = int(df.duplicated(keep="first").sum())
    duplicate_percent = round((duplicate_count / total_rows) * 100, 2) if total_rows > 0 else 0.0
    empty_columns     = int(df.isna().all().sum())

    return {
        "rows":               total_rows,
        "columns":            total_columns,
        "missing_percent":    missing_percent,
        "duplicate_percent":  duplicate_percent,
        "empty_columns":      empty_columns,
    }


def calculate_quality_score(df: pd.DataFrame) -> float:
    metrics = calculate_quality_metrics(df)

    missing_penalty   = (metrics["missing_percent"]   / 100) * MISSING_VALUES_WEIGHT
    duplicate_penalty = (metrics["duplicate_percent"] / 100) * DUPLICATE_ROWS_WEIGHT

    if metrics["columns"] > 0:
        empty_col_percent = (metrics["empty_columns"] / metrics["columns"]) * 100
    else:
        empty_col_percent = 0.0

    empty_col_penalty = (empty_col_percent / 100) * EMPTY_COLUMNS_WEIGHT
    raw_score         = 100 - missing_penalty - duplicate_penalty - empty_col_penalty

    return round(max(0.0, min(100.0, raw_score)), 2)


def score_label(score: float) -> str:
    if score >= 90:
        return "Excellent"
    elif score >= 75:
        return "Good"
    elif score >= 50:
        return "Fair"
    else:
        return "Poor"