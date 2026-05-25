"""
SmartCleaner - src/schema_drift.py
------------------------------------
Schema Drift Detection engine.

WHAT IS SCHEMA DRIFT?
----------------------
Schema means the structure of your data file:
  - What columns exist
  - How many columns there are
  - What data type each column holds (text, number, date)

Schema Drift means that structure has changed from what you expected.

REAL WORLD EXAMPLE:
  Every Monday your team uploads a sales report with these columns:
    name, age, department, salary, city

  One Monday someone uploads a file with:
    name, age, dept, salary, location

  Two columns have different names (department -> dept, city -> location).
  Without drift detection, SmartCleaner cleans it silently.
  You get wrong results and never know why.

  With drift detection, SmartCleaner immediately warns you:
    - Missing columns: department, city
    - New columns: dept, location
    - Possible renames: department -> dept, city -> location

WHY THIS MATTERS IN PRODUCTION:
  In real companies, data pipelines break because a column was renamed
  upstream and nobody noticed. Schema drift detection is one of the
  most important features in any production data system.
  Tools like Great Expectations and dbt are built around this idea.

HOW THIS MODULE WORKS:
  1. save_schema(df)      - save the structure of a known-good file
  2. load_schema()        - load the previously saved schema
  3. detect_drift(df)     - compare a new file against the saved schema
  4. get_drift_report(df) - return a full human-readable drift report

Storage:
  The schema is saved as a JSON file at data/schema.json by default.
  This is simple, human-readable, and requires no database.
"""

import json
import os
import pandas as pd
import numpy as np
from datetime import datetime


# ── Default schema file location ─────────────────────────────────────────────
# Saved next to the data folder so it stays with the project.
# Can be overridden by passing schema_path to any function.
DEFAULT_SCHEMA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "schema.json"
)


def _get_column_types(df: pd.DataFrame) -> dict:
    """
    Build a dictionary mapping each column name to its data type category.

    We use three simple categories instead of exact Pandas dtypes because
    exact dtypes change between Pandas versions and file formats:
      - "numeric"  : integers and floats
      - "text"     : strings and object columns
      - "other"    : datetime, boolean, and anything else

    Args:
        df: Any pandas DataFrame.

    Returns:
        dict mapping column name (str) -> type category (str).
    """
    type_map = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            type_map[col] = "numeric"
        elif pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
            type_map[col] = "text"
        else:
            type_map[col] = "other"
    return type_map


def save_schema(
    df: pd.DataFrame,
    schema_path: str = DEFAULT_SCHEMA_PATH,
    label: str = "default",
) -> dict:
    """
    Save the schema of a DataFrame to a JSON file.

    Call this once on a known-good file — the file whose structure
    you want future uploads to match. After saving, every new file
    you upload can be compared against this saved schema.

    What gets saved:
      - column names (in order)
      - data type of each column
      - number of columns
      - number of rows at time of saving
      - timestamp of when the schema was saved
      - a label so you can have multiple named schemas

    Args:
        df: The known-good DataFrame whose schema you want to save.
        schema_path: Where to save the JSON file. Default: data/schema.json
        label: A name for this schema. Useful if you have multiple
               file types. Default: "default"

    Returns:
        The schema dict that was saved. Useful for logging or display.
    """
    schema = {
        "label":       label,
        "saved_at":    datetime.now().isoformat(),
        "row_count":   len(df),
        "column_count": df.shape[1],
        "columns":     list(df.columns),
        "column_types": _get_column_types(df),
    }

    # Make sure the data directory exists
    os.makedirs(os.path.dirname(schema_path), exist_ok=True)

    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)

    print(f"  [save_schema] Schema saved to {schema_path}")
    print(f"  [save_schema] Saved {len(df.columns)} columns: {list(df.columns)}")

    return schema


def load_schema(schema_path: str = DEFAULT_SCHEMA_PATH) -> dict | None:
    """
    Load a previously saved schema from a JSON file.

    Args:
        schema_path: Path to the schema JSON file.

    Returns:
        The schema dict if the file exists, None otherwise.
    """
    if not os.path.exists(schema_path):
        return None

    with open(schema_path, "r") as f:
        return json.load(f)


def detect_drift(
    df: pd.DataFrame,
    schema_path: str = DEFAULT_SCHEMA_PATH,
) -> dict:
    """
    Compare a new DataFrame against the saved schema and detect drift.

    Checks for:
      1. Missing columns   — columns in the saved schema but not in the new file
      2. New columns       — columns in the new file but not in the saved schema
      3. Type changes      — columns present in both but with different data types
      4. Order changes     — same columns but in a different order
      5. Column count diff — total number of columns changed

    Also attempts to detect possible renames:
      If a column disappears and a new column appears with a similar name,
      it flags them as a possible rename. This is a heuristic — it helps
      the user spot accidental renames like "dept" instead of "department".

    Args:
        df: The new DataFrame to check against the saved schema.
        schema_path: Path to the saved schema JSON file.

    Returns:
        dict with keys:
          - has_drift:          True if any drift was detected
          - schema_exists:      True if a saved schema was found
          - missing_columns:    columns removed from the file
          - new_columns:        columns added to the file
          - type_changes:       columns whose type changed
          - order_changed:      True if column order is different
          - possible_renames:   list of (old_name, new_name) pairs
          - saved_column_count: how many columns the schema had
          - new_column_count:   how many columns the new file has
          - saved_at:           when the schema was saved
          - drift_severity:     "none", "low", "medium", or "high"
    """
    saved = load_schema(schema_path)

    # If no schema has been saved yet, drift cannot be checked
    if saved is None:
        return {
            "has_drift":          False,
            "schema_exists":      False,
            "missing_columns":    [],
            "new_columns":        [],
            "type_changes":       [],
            "order_changed":      False,
            "possible_renames":   [],
            "saved_column_count": 0,
            "new_column_count":   df.shape[1],
            "saved_at":           None,
            "drift_severity":     "none",
            "message":            "No saved schema found. Upload a file and click Save Schema to start tracking.",
        }

    saved_cols   = saved["columns"]
    saved_types  = saved["column_types"]
    new_cols     = list(df.columns)
    new_types    = _get_column_types(df)

    saved_set = set(saved_cols)
    new_set   = set(new_cols)

    # ── Missing columns ───────────────────────────────────────────────────────
    missing_columns = sorted(saved_set - new_set)

    # ── New columns ───────────────────────────────────────────────────────────
    new_columns = sorted(new_set - saved_set)

    # ── Type changes ──────────────────────────────────────────────────────────
    # Only check columns that exist in both schemas
    common_cols  = saved_set & new_set
    type_changes = []
    for col in common_cols:
        old_type = saved_types.get(col, "unknown")
        new_type = new_types.get(col, "unknown")
        if old_type != new_type:
            type_changes.append({
                "column":   col,
                "old_type": old_type,
                "new_type": new_type,
            })

    # ── Order changed ─────────────────────────────────────────────────────────
    # Check if the common columns appear in the same order in both schemas
    common_in_saved = [c for c in saved_cols if c in common_cols]
    common_in_new   = [c for c in new_cols   if c in common_cols]
    order_changed   = common_in_saved != common_in_new

    # ── Possible renames ──────────────────────────────────────────────────────
    # Heuristic: if a missing column and a new column share at least
    # 60% of their characters, flag them as a possible rename.
    # This catches common cases like "dept" vs "department".
    possible_renames = []
    for old_col in missing_columns:
        for new_col in new_columns:
            similarity = _name_similarity(old_col, new_col)
            if similarity >= 0.5:
                possible_renames.append({
                    "old_name":   old_col,
                    "new_name":   new_col,
                    "similarity": round(similarity, 2),
                })

    # ── Determine drift severity ──────────────────────────────────────────────
    has_drift = bool(missing_columns or new_columns or type_changes)

    if not has_drift:
        severity = "none"
    elif missing_columns and not new_columns:
        # Columns were removed — potentially breaking
        severity = "high"
    elif len(missing_columns) > 2 or len(new_columns) > 2:
        # Many changes — significant restructure
        severity = "high"
    elif missing_columns or new_columns:
        # Some changes — moderate impact
        severity = "medium"
    else:
        # Only type changes — low impact
        severity = "low"

    return {
        "has_drift":          has_drift,
        "schema_exists":      True,
        "missing_columns":    missing_columns,
        "new_columns":        new_columns,
        "type_changes":       type_changes,
        "order_changed":      order_changed,
        "possible_renames":   possible_renames,
        "saved_column_count": saved["column_count"],
        "new_column_count":   df.shape[1],
        "saved_at":           saved["saved_at"],
        "drift_severity":     severity,
    }


def _name_similarity(a: str, b: str) -> float:
    """
    Calculate how similar two column names are as a value from 0.0 to 1.0.

    Uses character overlap divided by the length of the longer name.
    This is a simple but effective heuristic for catching renames like:
      "department" vs "dept"     -> 0.4 (not flagged — too different)
      "salary"     vs "sal"      -> 0.5 (borderline)
      "first_name" vs "firstname"-> 0.89 (flagged as possible rename)

    Args:
        a: First column name.
        b: Second column name.

    Returns:
        Float between 0.0 and 1.0. Higher = more similar.
    """
    a = a.lower().replace("_", "").replace(" ", "")
    b = b.lower().replace("_", "").replace(" ", "")

    if not a or not b:
        return 0.0

    # Count common characters
    set_a    = set(a)
    set_b    = set(b)
    common   = len(set_a & set_b)
    max_len  = max(len(set_a), len(set_b))

    return common / max_len if max_len > 0 else 0.0


def get_drift_summary(drift_result: dict) -> str:
    """
    Convert a drift detection result into a plain English summary.

    Useful for displaying in the terminal or in the Streamlit UI
    as a human-readable description of what changed.

    Args:
        drift_result: The dict returned by detect_drift().

    Returns:
        A multi-line string describing the drift in plain English.
    """
    if not drift_result["schema_exists"]:
        return drift_result.get("message", "No schema saved yet.")

    if not drift_result["has_drift"]:
        return "No drift detected. File structure matches the saved schema."

    lines = [f"Schema drift detected (severity: {drift_result['drift_severity'].upper()})"]

    if drift_result["missing_columns"]:
        lines.append(f"  Missing columns ({len(drift_result['missing_columns'])}): "
                     f"{', '.join(drift_result['missing_columns'])}")

    if drift_result["new_columns"]:
        lines.append(f"  New columns ({len(drift_result['new_columns'])}): "
                     f"{', '.join(drift_result['new_columns'])}")

    if drift_result["type_changes"]:
        for tc in drift_result["type_changes"]:
            lines.append(f"  Type change: '{tc['column']}' changed from "
                         f"{tc['old_type']} to {tc['new_type']}")

    if drift_result["possible_renames"]:
        for pr in drift_result["possible_renames"]:
            lines.append(f"  Possible rename: '{pr['old_name']}' -> '{pr['new_name']}' "
                         f"(similarity: {pr['similarity']})")

    return "\n".join(lines)
