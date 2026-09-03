"""
clean_and_explore.py :

Cleans raw MPLADS 'recommended works' and 'completed works' exports into the
two CSVs that match_works_tiered.py (and match_works.py) expect:

    cleaned_mplads_recommended_works.csv
    cleaned_mplads_completed_works.csv

Run this BEFORE match_works_tiered.py — it's step 1 of the pipeline.
"""

import os
import zipfile
import pandas as pd

#  Raw input files
RAW_REC_FILE   = 'mplads_recommended_works_raw.csv'
RAW_COMP_FILE  = 'mplads_completed_works_raw.csv'
DATASET_ZIP    = 'dataset(1).zip'   # raw CSVs are inside this zip

#  Cleaned output files 
CLEAN_REC_OUT  = 'cleaned_mplads_recommended_works.csv'
CLEAN_COMP_OUT = 'cleaned_mplads_completed_works.csv'

REQUIRED_REC_COLS = [
    'Work ID', 'Work Description', 'Category', 'MP Name', 'IDA',
    'Recommended Amount (₹)', 'Recommendation Date', 'Has Images',
]
REQUIRED_COMP_COLS = [
    'Work ID', 'Work Description', 'Category', 'MP Name', 'Constituency',
    'State', 'House', 'IDA', 'Final Amount (₹)', 'Completed Date',
    'Has Images', 'Average Rating',
]


def _maybe_extract_zip():
    """If the raw CSVs aren't present but dataset(1).zip is, unzip it first."""
    if os.path.exists(DATASET_ZIP) and not (
        os.path.exists(RAW_REC_FILE) and os.path.exists(RAW_COMP_FILE)
    ):
        print(f"-> Extracting {DATASET_ZIP}...")
        with zipfile.ZipFile(DATASET_ZIP) as z:
            z.extractall('.')


def _basic_clean(df: pd.DataFrame, required_cols: list, date_col: str, amount_col: str) -> pd.DataFrame:
    """Shared cleaning steps for both recommended and completed dataframes."""
    # Drop fully-empty rows
    df = df.dropna(how='all').copy()

    # Strip whitespace on every string/object column
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip()

    # Parse the date column
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    # Coerce amount column to numeric, stripping currency symbols/commas
    if amount_col in df.columns:
        df[amount_col] = (
            df[amount_col].astype(str)
            .str.replace(r'[₹,]', '', regex=True)
            .str.strip()
        )
        df[amount_col] = pd.to_numeric(df[amount_col], errors='coerce')

    # Drop exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    print(f"   Dropped {before - len(df):,} exact duplicate rows")

    # Warn (don't crash) on any missing required columns
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"   WARNING: missing expected columns: {missing}")
        print(f"   Actual columns present: {list(df.columns)}")

    return df


def clean_recommended(df: pd.DataFrame) -> pd.DataFrame:
    print("-> Cleaning recommended works...")
    return _basic_clean(
        df, REQUIRED_REC_COLS,
        date_col='Recommendation Date',
        amount_col='Recommended Amount (₹)',
    )


def clean_completed(df: pd.DataFrame) -> pd.DataFrame:
    print("-> Cleaning completed works...")
    return _basic_clean(
        df, REQUIRED_COMP_COLS,
        date_col='Completed Date',
        amount_col='Final Amount (₹)',
    )


def explore(df: pd.DataFrame, name: str):
    print(f"\n{'='*70}\n  {name} — quick look\n{'='*70}")
    print(f"  Rows: {len(df):,}   Columns: {len(df.columns)}")
    print(f"  Columns: {list(df.columns)}")
    print("\n  Null counts per column:")
    print(df.isna().sum().to_string())
    print(f"{'='*70}")


if __name__ == '__main__':
    _maybe_extract_zip()

    df_rec_raw  = pd.read_csv(RAW_REC_FILE)
    df_comp_raw = pd.read_csv(RAW_COMP_FILE)

    df_rec  = clean_recommended(df_rec_raw)
    df_comp = clean_completed(df_comp_raw)

    explore(df_rec, 'Recommended works (cleaned)')
    explore(df_comp, 'Completed works (cleaned)')

    df_rec.to_csv(CLEAN_REC_OUT, index=False)
    df_comp.to_csv(CLEAN_COMP_OUT, index=False)
    print(f"\n-> Saved '{CLEAN_REC_OUT}' and '{CLEAN_COMP_OUT}'")
