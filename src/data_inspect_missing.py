# =============================================
# src/data_inspect_missing.py
# =============================================
# Purpose:
#   Inspect missing values in train.csv and test.csv
#   before performing any cleaning or imputation.
#
# Steps:
#   1. Load train and test data
#   2. Compute missing values per column
#   3. Compute missing values per row
#   4. Summarize which features or periods are most affected
#   5. Visualize or print distributions (no filling yet)
#
# Notes:
#   • Financial datasets often have missing data for real reasons
#     (e.g., indicator not available before certain year).
#   • This script is only for *diagnosis*, not modification.
# =============================================

import pandas as pd
import numpy as np
from pathlib import Path

# ------------------------------------------------------------
# 1️⃣ Setup paths and load data
# ------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
train_path = DATA_DIR / "train.csv"
test_path  = DATA_DIR / "test.csv"

train = pd.read_csv(train_path)
test  = pd.read_csv(test_path)

print("Loaded train shape:", train.shape)
print("Loaded test shape:", test.shape)
print("=" * 80)

# ------------------------------------------------------------
# 2️⃣ Missing values per column
# ------------------------------------------------------------
def missing_summary(df: pd.DataFrame, name: str):
    """Prints a summary of missing values per column."""
    print(f"\n=== Missing value summary for {name} ===")
    missing_counts = df.isna().sum()
    missing_perc = (missing_counts / len(df)) * 100
    summary = pd.DataFrame({
        "missing_count": missing_counts,
        "missing_percent": missing_perc
    }).sort_values(by="missing_percent", ascending=False)
    
    # Print top 15 most missing columns
    print(summary.head(15))
    print(f"\nTotal columns with missing values: {(missing_counts > 0).sum()} / {len(df.columns)}")
    print("=" * 80)
    return summary

train_missing_summary = missing_summary(train, "train.csv")
test_missing_summary  = missing_summary(test, "test.csv")

# ------------------------------------------------------------
# 3️⃣ Missing values per row
# ------------------------------------------------------------
# Count how many columns are missing per row (to identify rows with heavy NaNs)
train['missing_per_row'] = train.isna().sum(axis=1)
test['missing_per_row']  = test.isna().sum(axis=1)

print("\n=== Missing data per row (train) ===")
print(train['missing_per_row'].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]))

print("\n=== Missing data per row (test) ===")
print(test['missing_per_row'].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]))

# ------------------------------------------------------------
# 4️⃣ Investigate which features groups are most missing
# ------------------------------------------------------------
# Use feature prefixes (M*, E*, I*, P*, V*, S*, MOM*, D*) to group by type
def group_missing_by_prefix(df: pd.DataFrame):
    prefixes = ['M', 'E', 'I', 'P', 'V', 'S', 'MOM', 'D']
    counts = {}
    for prefix in prefixes:
        cols = [c for c in df.columns if c.startswith(prefix)]
        if cols:
            missing = df[cols].isna().sum().sum()
            total = len(cols) * len(df)
            counts[prefix] = (missing / total) * 100
    return pd.Series(counts).sort_values(ascending=False)

train_group_missing = group_missing_by_prefix(train)
test_group_missing  = group_missing_by_prefix(test)

print("\n=== Missing percentages by feature group (train) ===")
print(train_group_missing)
print("\n=== Missing percentages by feature group (test) ===")
print(test_group_missing)
print("=" * 80)

# ------------------------------------------------------------
# 5️⃣ Optional: investigate by date_id
# ------------------------------------------------------------
# Some older date_ids (earlier years) might have more missing data.
if 'date_id' in train.columns:
    train_date_missing = (
        train.groupby('date_id')['missing_per_row']
        .mean()
        .reset_index(name='avg_missing_per_row')
    )
    print("\n=== Average missing features per row over time (train) ===")
    print(train_date_missing.head())
    print("...")
    print(train_date_missing.tail())

    # Identify earliest and latest dates for context
    print(f"\nDate range in train: {train['date_id'].min()} → {train['date_id'].max()}")

# ------------------------------------------------------------
# 6️⃣ Save summaries for later review (optional)
# ------------------------------------------------------------
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
train_missing_summary.to_csv(OUT_DIR / "train_missing_summary.csv", index=True)
test_missing_summary.to_csv(OUT_DIR / "test_missing_summary.csv", index=True)
train[['date_id', 'missing_per_row']].to_csv(OUT_DIR / "train_missing_per_row.csv", index=False)
test[['date_id', 'missing_per_row']].to_csv(OUT_DIR / "test_missing_per_row.csv", index=False)

print("\nSaved missing data summaries to /outputs folder.")
print("Next step: inspect the CSVs to decide why data is missing before cleaning.")
