"""
SmartCleaner - api.py
----------------------
FastAPI REST API for the SmartCleaner pipeline.

This file adds a third way to use SmartCleaner alongside the
command-line tool (main.py) and the web app (app.py).

Anyone with an HTTP client — Postman, Python requests, a mobile
app, another server — can send a file to these endpoints and get
back a cleaned file and a full quality report as JSON.

Endpoints:
  GET  /health          Check if the API is running
  GET  /schema          Get the currently saved reference schema
  POST /schema/save     Save a new reference schema from a file
  POST /clean           Clean a file and return the quality report

Run with:
  uvicorn api:app --reload

Then open:
  http://localhost:8000/docs   (interactive API documentation)
  http://localhost:8000/health (quick health check)
"""

import os
import sys
import io
import json
import tempfile
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

# ── Path setup ────────────────────────────────────────────────────────────────
# Make sure Python can find the src/ package regardless of
# where uvicorn launches from
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.cleaner import clean_dataset
from src.quality_score import calculate_quality_score, calculate_quality_metrics, score_label
from src.schema_drift import save_schema, load_schema, detect_drift, get_drift_summary
from src.config import settings


# ── Create the FastAPI app ────────────────────────────────────────────────────
app = FastAPI(
    title="SmartCleaner API",
    description="Automated data cleaning and quality scoring for CSV and Excel files.",
    version=settings.API_VERSION,
)


# ── Helper: read uploaded file into DataFrame ─────────────────────────────────

def _check_file_size(contents: bytes, filename: str) -> None:
    """Reject files larger than MAX_FILE_SIZE_MB with HTTP 413."""
    if len(contents) > settings.MAX_FILE_SIZE_BYTES:
        size_mb = len(contents) / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=(
                f"File '{filename}' is too large. "
                f"Size: {size_mb:.1f} MB. "
                f"Maximum allowed: {settings.MAX_FILE_SIZE_MB} MB."
            )
        )


def _read_upload(file: UploadFile) -> pd.DataFrame:
    """Read an uploaded CSV or Excel file into a DataFrame."""
    filename = file.filename.lower()
    contents = file.file.read()

    # Check size BEFORE parsing
    _check_file_size(contents, file.filename)

    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Please upload .csv, .xlsx, or .xls."
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Could not read file '{file.filename}': {str(e)}"
        )

    return df 

    """
    Read an uploaded file (CSV or Excel) into a pandas DataFrame.

    Args:
        file: The uploaded file object from FastAPI.

    Returns:
        pandas DataFrame.

    Raises:
        HTTPException 400 if the file format is not supported.
        HTTPException 422 if the file cannot be parsed.
    """
    filename = file.filename.lower()
    contents = file.file.read()

    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Please upload a .csv, .xlsx, or .xls file."
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Could not read file '{file.filename}': {str(e)}"
        )

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 1 — Health check
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["System"])
def health_check():
    """
    Check if the SmartCleaner API is running.

    Returns a simple status message and the current server time.
    Use this to verify the API is up before sending files.

    Example response:
        {
            "status": "ok",
            "service": "SmartCleaner API",
            "timestamp": "2026-05-25T10:30:00"
        }
    """
    return {
        "status":           "ok",
        "service":          "SmartCleaner API",
        "version":          settings.API_VERSION,
        "environment":      settings.ENVIRONMENT,
        "max_file_size_mb": settings.MAX_FILE_SIZE_MB,
        "timestamp":        datetime.now().isoformat(),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 2 — Get saved schema
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/schema", tags=["Schema"])
def get_schema():
    """
    Get the currently saved reference schema.

    The schema is the saved structure of a known-good file —
    column names, data types, and column count.

    Returns 404 if no schema has been saved yet.

    Example response:
        {
            "label": "default",
            "saved_at": "2026-05-25T10:30:00",
            "column_count": 5,
            "columns": ["name", "age", "department", "salary", "city"],
            "column_types": {
                "name": "text",
                "age": "numeric"
            }
        }
    """
    schema = load_schema()

    if schema is None:
        raise HTTPException(
            status_code=404,
            detail="No reference schema saved yet. POST a file to /schema/save first."
        )

    return schema


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 3 — Save schema from uploaded file
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/schema/save", tags=["Schema"])
def save_reference_schema(file: UploadFile = File(...)):
    """
    Save the structure of an uploaded file as the reference schema.

    Upload your known-good file here once. Every file you send to
    /clean will then be automatically compared against this reference
    to detect any structural changes (schema drift).

    Accepts: .csv, .xlsx, .xls

    Example response:
        {
            "message": "Schema saved successfully.",
            "columns_saved": 5,
            "columns": ["name", "age", "department", "salary", "city"]
        }
    """
    df = _read_upload(file)

    schema = save_schema(df, label=file.filename)

    return {
        "message":      "Schema saved successfully.",
        "columns_saved": len(schema["columns"]),
        "columns":       schema["columns"],
        "saved_at":      schema["saved_at"],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 4 — Clean a file
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/clean", tags=["Cleaning"])
def clean_file(file: UploadFile = File(...)):
    """
    Clean an uploaded CSV or Excel file and return a full quality report.

    This is the main endpoint. It runs the full SmartCleaner pipeline:
      1. Detect schema drift against saved reference (if one exists)
      2. Calculate quality score before cleaning
      3. Clean column names
      4. Remove empty columns and rows
      5. Remove duplicate rows
      6. Clean text values
      7. Fill numeric gaps with KNN imputation
      8. Fill text gaps with Unknown
      9. Remove outliers using Isolation Forest
      10. Calculate quality score after cleaning

    The cleaned file is returned as a downloadable CSV attachment.
    The full quality report is returned in the response headers as JSON.

    Accepts: .csv, .xlsx, .xls

    Returns:
        Streaming CSV file download with these response headers:
          X-Quality-Score-Before
          X-Quality-Score-After
          X-Quality-Improvement
          X-Rows-Before
          X-Rows-After
          X-Outliers-Removed
          X-Drift-Detected
          X-Drift-Severity
          X-Report (full JSON report)
    """
    # ── Read the uploaded file ────────────────────────────────────────────────
    df_raw   = _read_upload(file)
    filename = file.filename

    # ── Schema drift check ────────────────────────────────────────────────────
    # Check if the file structure matches the saved reference schema.
    # This does not block cleaning — it just adds drift info to the report.
    drift = detect_drift(df_raw)

    # ── Run the cleaning pipeline ─────────────────────────────────────────────
    try:
        df_clean, report = clean_dataset(df_raw)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Cleaning pipeline failed: {str(e)}"
        )

    # ── Build output filename ─────────────────────────────────────────────────
    base_name  = filename.rsplit(".", 1)[0]
    out_name   = f"cleaned_{base_name}.csv"

    # ── Convert cleaned DataFrame to CSV bytes ────────────────────────────────
    csv_buffer = io.StringIO()
    df_clean.to_csv(csv_buffer, index=False)
    csv_bytes  = csv_buffer.getvalue().encode("utf-8")

    # ── Build summary report for headers ─────────────────────────────────────
    # We put key metrics in response headers so the caller can read
    # them without parsing the entire file.
    summary = {
        "quality_score_before":  report["quality_score_before"],
        "quality_score_after":   report["quality_score_after"],
        "quality_improvement":   report["quality_improvement"],
        "quality_label_before":  score_label(report["quality_score_before"]),
        "quality_label_after":   score_label(report["quality_score_after"]),
        "rows_before":           report["original_rows"],
        "rows_after":            report["final_rows"],
        "columns_before":        report["original_columns"],
        "columns_after":         report["final_columns"],
        "duplicates_removed":    report["duplicate_rows_removed"],
        "empty_rows_dropped":    report["empty_rows_dropped"],
        "empty_columns_dropped": report["empty_columns_dropped"],
        "outliers_removed":      report["outliers_removed"],
        "knn_values_filled":     report["knn_values_filled"],
        "missing_filled":        report["missing_filled"],
        "drift_detected":        drift["has_drift"],
        "drift_severity":        drift["drift_severity"],
        "drift_missing_columns": drift["missing_columns"],
        "drift_new_columns":     drift["new_columns"],
    }

    # ── Return streaming CSV with report in headers ───────────────────────────
    headers = {
        "Content-Disposition":    f'attachment; filename="{out_name}"',
        "X-Quality-Score-Before": str(report["quality_score_before"]),
        "X-Quality-Score-After":  str(report["quality_score_after"]),
        "X-Quality-Improvement":  str(report["quality_improvement"]),
        "X-Rows-Before":          str(report["original_rows"]),
        "X-Rows-After":           str(report["final_rows"]),
        "X-Outliers-Removed":     str(report["outliers_removed"]),
        "X-Drift-Detected":       str(drift["has_drift"]),
        "X-Drift-Severity":       drift["drift_severity"],
        "X-Report":               json.dumps(summary),
    }

    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers=headers,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 5 — Inspect a file without cleaning
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/inspect", tags=["Cleaning"])
def inspect_file(file: UploadFile = File(...)):
    """
    Get the quality score and drift report for a file WITHOUT cleaning it.

    Use this when you want to see what problems exist in a file
    before deciding whether to clean it. No changes are made to the data.

    Returns:
        JSON with quality metrics, quality score, and drift report.

    Example response:
        {
            "filename": "sales.csv",
            "rows": 150,
            "columns": 7,
            "quality_score": 61.5,
            "quality_label": "Fair",
            "missing_percent": 12.3,
            "duplicate_percent": 5.0,
            "empty_columns": 0,
            "drift": {
                "has_drift": true,
                "drift_severity": "medium",
                "missing_columns": ["salary"],
                "new_columns": ["wage"]
            }
        }
    """
    df      = _read_upload(file)
    metrics = calculate_quality_metrics(df)
    score   = calculate_quality_score(df)
    drift   = detect_drift(df)

    return {
        "filename":          file.filename,
        "rows":              metrics["rows"],
        "columns":           metrics["columns"],
        "quality_score":     score,
        "quality_label":     score_label(score),
        "missing_percent":   metrics["missing_percent"],
        "duplicate_percent": metrics["duplicate_percent"],
        "empty_columns":     metrics["empty_columns"],
        "drift": {
            "has_drift":        drift["has_drift"],
            "schema_exists":    drift["schema_exists"],
            "drift_severity":   drift["drift_severity"],
            "missing_columns":  drift["missing_columns"],
            "new_columns":      drift["new_columns"],
            "type_changes":     drift["type_changes"],
            "possible_renames": drift["possible_renames"],
        },
    }
