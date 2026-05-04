import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pathlib
import pandas as pd

from src.cleaner import clean_dataset
from src.quality_score import score_label

INPUT_PATH  = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("data/sample_dirty.csv")
OUTPUT_PATH = INPUT_PATH.parent / ("cleaned_" + INPUT_PATH.stem + ".csv")


def load_file(path: pathlib.Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        print(f"Loading CSV  : {path}")
        return pd.read_csv(path)
    elif suffix in (".xlsx", ".xls"):
        print(f"Loading Excel: {path}")
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported format: '{suffix}'.")


def main():
    if not INPUT_PATH.exists():
        print(f"File not found: {INPUT_PATH}")
        sys.exit(1)

    df_raw = load_file(INPUT_PATH)
    print(f"Shape: {df_raw.shape[0]} rows x {df_raw.shape[1]} columns\n")

    df_clean, report = clean_dataset(df_raw)

    df_clean.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to: {OUTPUT_PATH}")

    before = report["quality_score_before"]
    after  = report["quality_score_after"]
    gain   = report["quality_improvement"]

    print("\n--- Quality Score ---")
    print(f"  Before : {before:>6}/100  ({score_label(before)})")
    print(f"  After  : {after:>6}/100  ({score_label(after)})")
    print(f"  Gain   : +{gain} points")

    print("\n--- Validation Report ---")
    print(f"  Rows            : {report['original_rows']} -> {report['final_rows']}")
    print(f"  Columns         : {report['original_columns']} -> {report['final_columns']}")
    print(f"  Empty rows dropped     : {report['empty_rows_dropped']}")
    print(f"  Empty columns dropped  : {report['empty_columns_dropped']}")
    print(f"  Duplicate rows removed : {report['duplicate_rows_removed']}")

    if report["missing_filled"]:
        print("  Missing values filled:")
        for col, detail in report["missing_filled"].items():
            print(f"    - {col}: {detail['count']} gap(s) -> {detail['fill_value']!r}")

    print("\n--- Configuration Used ---")
    config = report.get("config_used", {})
    print(f"  Fill text            : {config.get('fill_text', 'N/A')!r}")
    print(f"  Fill numeric         : {config.get('fill_numeric', 'N/A')!r}")
    print(f"  Drop empty columns   : {config.get('drop_empty_columns', 'N/A')}")
    print(f"  Remove duplicates    : {config.get('remove_duplicates', 'N/A')}")


if __name__ == "__main__":
    main()