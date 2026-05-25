"""
SmartCleaner - app.py
----------------------
Streamlit web interface for the SmartCleaner pipeline.

Displays:
  - File upload for CSV and Excel files
  - Raw vs cleaned data side by side
  - Quality score before and after cleaning
  - Score improvement with colour-coded feedback
  - Detailed validation report
  - One-click download of the cleaned file

Run with:
    python -m streamlit run app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import streamlit as st

from src.cleaner import clean_dataset
from src.quality_score import calculate_quality_score, calculate_quality_metrics, score_label
from src.schema_drift import save_schema, detect_drift, get_drift_summary


# ── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartCleaner",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🧹</text></svg>",
    layout="wide",
)


# ── Helper: colour a score value for display ─────────────────────────────────

def score_colour(score: float) -> str:
    """Return a hex colour string based on the score range."""
    if score >= 90:
        return "#2ecc71"   # green
    elif score >= 75:
        return "#27ae60"   # darker green
    elif score >= 50:
        return "#f39c12"   # orange
    elif score >= 25:
        return "#e67e22"   # dark orange
    else:
        return "#e74c3c"   # red


def render_score_card(label: str, score: float) -> None:
    """Render a styled score card using st.markdown."""
    colour  = score_colour(score)
    quality = score_label(score)
    st.markdown(
        f"""
        <div style="
            border: 2px solid {colour};
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            background-color: {colour}18;
        ">
            <p style="margin:0; font-size:14px; color:#888;">{label}</p>
            <p style="margin:4px 0; font-size:42px; font-weight:bold; color:{colour};">{score}</p>
            <p style="margin:0; font-size:16px; color:{colour};">{quality}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Page header ──────────────────────────────────────────────────────────────
st.title("SmartCleaner")
st.caption("Upload a messy CSV or Excel file. Get a clean one back with a full quality report.")

st.divider()

# ── File upload ──────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    label="Upload your file",
    type=["csv", "xlsx", "xls"],
    help="Supported formats: CSV, Excel (.xlsx / .xls)",
)

if uploaded_file is None:
    st.info("Upload a file above to get started.")
    st.stop()

# ── Load the file ─────────────────────────────────────────────────────────────
filename = uploaded_file.name

try:
    if filename.endswith(".csv"):
        df_raw = pd.read_csv(uploaded_file)
    else:
        df_raw = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

# ── Schema drift detection ────────────────────────────────────────────────────
# Check if the uploaded file matches the previously saved schema.
# This runs BEFORE cleaning so the user can see structural problems
# before any transformation happens.

drift_result = detect_drift(df_raw)

if not drift_result["schema_exists"]:
    # No schema saved yet — show a button to save this file as the reference
    st.info("No reference schema saved yet. Click below to use this file as your reference schema.")
    if st.button("Save this file as reference schema"):
        save_schema(df_raw)
        st.success(f"Schema saved. {df_raw.shape[1]} columns recorded as the reference structure.")

elif drift_result["has_drift"]:
    # Drift detected — show a warning with details
    severity = drift_result["drift_severity"]
    colour   = "#e74c3c" if severity == "high" else "#f39c12"

    st.markdown(
        f"""
        <div style="
            border: 2px solid {colour};
            border-radius: 10px;
            padding: 16px 20px;
            background-color: {colour}18;
            margin-bottom: 16px;
        ">
            <p style="margin:0 0 8px; font-weight:bold; color:{colour}; font-size:16px;">
                Schema Drift Detected — Severity: {severity.upper()}
            </p>
            <p style="margin:0; font-size:14px; color:var(--text);">
                This file's structure is different from your saved reference schema.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Show drift details in columns
    d1, d2, d3 = st.columns(3)
    d1.metric("Missing columns",  len(drift_result["missing_columns"]))
    d2.metric("New columns",      len(drift_result["new_columns"]))
    d3.metric("Type changes",     len(drift_result["type_changes"]))

    if drift_result["missing_columns"]:
        st.warning(f"Missing columns: {', '.join(drift_result['missing_columns'])}")

    if drift_result["new_columns"]:
        st.info(f"New columns: {', '.join(drift_result['new_columns'])}")

    if drift_result["possible_renames"]:
        for pr in drift_result["possible_renames"]:
            st.info(f"Possible rename detected: '{pr['old_name']}' may have been renamed to '{pr['new_name']}'")

    if drift_result["type_changes"]:
        for tc in drift_result["type_changes"]:
            st.warning(f"Type change: '{tc['column']}' changed from {tc['old_type']} to {tc['new_type']}")

    st.divider()

    # Let user update the schema to this new file if the change was intentional
    if st.button("Update reference schema to this file"):
        save_schema(df_raw)
        st.success("Reference schema updated.")

else:
    # No drift — file matches the saved schema
    st.success(f"File structure matches reference schema. {df_raw.shape[1]} columns verified.")

st.divider()

# ── Run the pipeline ──────────────────────────────────────────────────────────
with st.spinner("Cleaning your data..."):
    df_clean, report = clean_dataset(df_raw)

# ── Section 1: Quality Score ──────────────────────────────────────────────────
st.subheader("Data Quality Score")

score_before     = report["quality_score_before"]
score_after      = report["quality_score_after"]
improvement      = report["quality_improvement"]

col_before, col_after, col_improvement = st.columns(3)

with col_before:
    render_score_card("Score Before Cleaning", score_before)

with col_after:
    render_score_card("Score After Cleaning", score_after)

with col_improvement:
    # Improvement card — always green if positive
    arrow  = "+" if improvement >= 0 else ""
    colour = "#2ecc71" if improvement >= 0 else "#e74c3c"
    st.markdown(
        f"""
        <div style="
            border: 2px solid {colour};
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            background-color: {colour}18;
        ">
            <p style="margin:0; font-size:14px; color:#888;">Improvement</p>
            <p style="margin:4px 0; font-size:42px; font-weight:bold; color:{colour};">
                {arrow}{improvement}
            </p>
            <p style="margin:0; font-size:16px; color:{colour};">points gained</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# ── Section 2: Metrics comparison table ──────────────────────────────────────
st.subheader("Quality Metrics Breakdown")

mb = report["metrics_before"]
ma = report["metrics_after"]

metrics_table = pd.DataFrame({
    "Metric":         ["Rows", "Columns", "Missing Values %", "Duplicate Rows %", "Empty Columns"],
    "Before":         [
        mb["rows"],
        mb["columns"],
        f"{mb['missing_percent']}%",
        f"{mb['duplicate_percent']}%",
        mb["empty_columns"],
    ],
    "After":          [
        ma["rows"],
        ma["columns"],
        f"{ma['missing_percent']}%",
        f"{ma['duplicate_percent']}%",
        ma["empty_columns"],
    ],
})

st.dataframe(metrics_table, use_container_width=True, hide_index=True)

st.divider()

# ── Section 3: Data preview ───────────────────────────────────────────────────
st.subheader("Data Preview")

left, right = st.columns(2)

with left:
    st.caption(f"Raw data — {report['original_rows']} rows x {report['original_columns']} columns")
    st.dataframe(df_raw, use_container_width=True, height=300)

with right:
    st.caption(f"Cleaned data — {report['final_rows']} rows x {report['final_columns']} columns")
    st.dataframe(df_clean, use_container_width=True, height=300)

st.divider()

# ── Section 4: Validation report ─────────────────────────────────────────────
st.subheader("Cleaning Report")

r1, r2, r3, r4 = st.columns(4)
r1.metric("Duplicate rows removed",  report["duplicate_rows_removed"])
r2.metric("Empty columns dropped",   report["empty_columns_dropped"])
r3.metric("Empty rows dropped",      report["empty_rows_dropped"])
r4.metric("Columns with gaps filled", len(report["missing_filled"]))

if report["missing_filled"]:
    st.markdown("**Missing value fill details:**")
    for col, detail in report["missing_filled"].items():
        st.write(f"- `{col}`: {detail['count']} gap(s) filled with `{detail['fill_value']!r}`")

if report["text_columns_normalised"]:
    cols_str = ", ".join(f"`{c}`" for c in report["text_columns_normalised"])
    st.markdown(f"**Text columns normalised:** {cols_str}")

st.divider()

# ── Section 5: Download ───────────────────────────────────────────────────────
st.subheader("Download")

csv_bytes = df_clean.to_csv(index=False).encode("utf-8")
out_name  = "cleaned_" + filename.replace(".xlsx", ".csv").replace(".xls", ".csv")

st.download_button(
    label="Download cleaned CSV",
    data=csv_bytes,
    file_name=out_name,
    mime="text/csv",
)
