import sys
import os

# Fix import path so src/ is always found regardless of
# how Streamlit launches the script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import pandas as pd
import streamlit as st

from src.cleaner import clean_dataset
from src.quality_score import calculate_quality_score, calculate_quality_metrics, score_label


st.set_page_config(page_title="SmartCleaner", layout="wide")


def score_colour(score: float) -> str:
    if score >= 90:
        return "#2ecc71"  # Green - Excellent
    elif score >= 75:
        return "#27ae60"  # Dark green - Good
    elif score >= 50:
        return "#f39c12"  # Orange - Fair
    else:
        return "#e74c3c"  # Red - Poor


def render_score_card(label: str, score: float) -> None:
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


st.title("SmartCleaner")
st.caption("Upload a messy CSV or Excel file. Get a clean one back with a full quality report.")
st.divider()

uploaded_file = st.file_uploader(
    label="Upload your file",
    type=["csv", "xlsx", "xls"],
)

if uploaded_file is None:
    st.info("Upload a file above to get started.")
    st.stop()

filename = uploaded_file.name

try:
    if filename.endswith(".csv"):
        df_raw = pd.read_csv(uploaded_file)
    else:
        df_raw = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

with st.spinner("Cleaning your data..."):
    df_clean, report = clean_dataset(df_raw)

# Quality scores
st.subheader("Data Quality Score")

score_before = report["quality_score_before"]
score_after  = report["quality_score_after"]
improvement  = report["quality_improvement"]

col_before, col_after, col_improvement = st.columns(3)

with col_before:
    render_score_card("Score Before Cleaning", score_before)

with col_after:
    render_score_card("Score After Cleaning", score_after)

with col_improvement:
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

# Before vs After Metrics
st.subheader("Before vs After Metrics")

mb = report["metrics_before"]
ma = report["metrics_after"]

# Create two columns for side-by-side comparison
metric_col1, metric_col2, metric_col3 = st.columns(3)

with metric_col1:
    st.markdown("**Missing Values**")
    st.write(f"Before: {mb['missing_percent']}%")
    st.write(f"After: {ma['missing_percent']}%")
    improvement = mb['missing_percent'] - ma['missing_percent']
    st.write(f"📉 Improved by {improvement}%" if improvement > 0 else f"(no change)" if improvement == 0 else f"(increased)")

with metric_col2:
    st.markdown("**Duplicate Rows**")
    st.write(f"Before: {mb['duplicate_percent']}%")
    st.write(f"After: {ma['duplicate_percent']}%")
    improvement = mb['duplicate_percent'] - ma['duplicate_percent']
    st.write(f" Improved by {improvement}%" if improvement > 0 else f"(no change)" if improvement == 0 else f"(increased)")

with metric_col3:
    st.markdown("**Row Count**")
    st.write(f"Before: {mb['rows']} rows")
    st.write(f"After: {ma['rows']} rows")
    change = mb['rows'] - ma['rows']
    st.write(f" Removed {change}" if change > 0 else f"(no change)" if change == 0 else f"(added {-change})")

st.divider()

# Detailed metrics table
st.subheader("Quality Metrics Breakdown")

metrics_table = pd.DataFrame({
    "Metric":  ["Rows", "Columns", "Missing Values %", "Duplicate Rows %", "Empty Columns"],
    "Before":  [mb["rows"], mb["columns"], f"{mb['missing_percent']}%", f"{mb['duplicate_percent']}%", mb["empty_columns"]],
    "After":   [ma["rows"], ma["columns"], f"{ma['missing_percent']}%", f"{ma['duplicate_percent']}%", ma["empty_columns"]],
})

st.dataframe(metrics_table, use_container_width=True, hide_index=True)
st.divider()

# Data preview
st.subheader("Data Preview")

left, right = st.columns(2)

with left:
    st.caption(f"Raw data - {report['original_rows']} rows x {report['original_columns']} columns")
    st.dataframe(df_raw, use_container_width=True, height=300)

with right:
    st.caption(f"Cleaned data - {report['final_rows']} rows x {report['final_columns']} columns")
    st.dataframe(df_clean, use_container_width=True, height=300)

st.divider()

# Cleaning report
st.subheader("Cleaning Report")

r1, r2, r3, r4 = st.columns(4)
r1.metric("Duplicate rows removed",   report["duplicate_rows_removed"])
r2.metric("Empty columns dropped",    report["empty_columns_dropped"])
r3.metric("Empty rows dropped",       report["empty_rows_dropped"])
r4.metric("Columns with gaps filled", len(report["missing_filled"]))

if report["missing_filled"]:
    st.markdown("**Missing value fill details:**")
    for col, detail in report["missing_filled"].items():
        st.write(f"- `{col}`: {detail['count']} gap(s) filled with `{detail['fill_value']!r}`")

if report["text_columns_normalised"]:
    cols_str = ", ".join(f"`{c}`" for c in report["text_columns_normalised"])
    st.markdown(f"**Text columns normalised:** {cols_str}")

st.divider()

# Configuration used
st.subheader("Configuration Used")

config = report.get("config_used", {})
c1, c2 = st.columns(2)

with c1:
    st.write(f"**Fill text:** `{config.get('fill_text', 'N/A')!r}`")
    st.write(f"**Fill numeric:** `{config.get('fill_numeric', 'N/A')}`")

with c2:
    st.write(f"**Drop empty columns:** `{config.get('drop_empty_columns', 'N/A')}`")
    st.write(f"**Remove duplicates:** `{config.get('remove_duplicates', 'N/A')}`")

st.divider()

# Download
st.subheader("Download")

# CSV download
csv_bytes = df_clean.to_csv(index=False).encode("utf-8")
out_name  = "cleaned_" + filename.replace(".xlsx", ".csv").replace(".xls", ".csv")

col_csv, col_json = st.columns(2)

with col_csv:
    st.download_button(
        label=" Download Cleaned CSV",
        data=csv_bytes,
        file_name=out_name,
        mime="text/csv",
    )

# JSON report download
with col_json:
    report_json = json.dumps(report, indent=2, default=str)
    report_bytes = report_json.encode("utf-8")
    report_name = "report_" + filename.replace(".xlsx", ".json").replace(".xls", ".json")
    
    st.download_button(
        label=" Download Report (JSON)",
        data=report_bytes,
        file_name=report_name,
        mime="application/json",
    )