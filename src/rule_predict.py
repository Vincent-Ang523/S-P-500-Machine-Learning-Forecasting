# =============================================
# src/rule_predict.py
# =============================================
# This script generates a simple *rule-based* prediction
# for the Hull Tactical Market Prediction project.
#
# Instead of using a machine learning model, it applies a
# straightforward trading rule:
#   - If yesterday’s (lagged) S&P 500 return was positive,
#       → slightly overweight the market (predict weight = 1.2)
#   - If yesterday’s return was negative,
#       → slightly underweight (predict weight = 0.8)
#
# This serves as a sanity check before implementing real models.
# =============================================

import pandas as pd
from pathlib import Path

# -------------------------------------------------------------
# 1️⃣ Define file paths
# -------------------------------------------------------------
# Get the project root directories relative to this file.
# Example:
#   src/ → go up one level → project root
#   Then join "data" and "submissions" folders.
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_DIR  = Path(__file__).resolve().parents[1] / "submissions"

# Create the submissions directory if it doesn't exist
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# 2️⃣ Load training and test datasets
# -------------------------------------------------------------
# train.csv: contains true forward_returns and risk_free_rate
# test.csv:  mock test set with lagged features for prediction
train = pd.read_csv(DATA_DIR / "train.csv")
test  = pd.read_csv(DATA_DIR / "test.csv")

# -------------------------------------------------------------
# 3️⃣ Define the rule-based strategy
# -------------------------------------------------------------
# The column 'lagged_forward_returns' in test.csv contains
# yesterday’s return for each date_id. It allows you to
# approximate "momentum" — if market was up yesterday, it may
# continue rising (momentum effect), and vice versa.
lag_col = 'lagged_forward_returns'

if lag_col in test.columns:
    # ✅ Case 1: lagged_forward_returns is available in test set
    #
    # Apply a simple lambda function to each row:
    #   - If lagged return > 0 → predict 1.2 (overweight)
    #   - Else → predict 0.8 (underweight)
    #
    # This is a basic "trend-following" rule.
    test['prediction'] = test[lag_col].apply(lambda x: 1.2 if x > 0 else 0.8)
else:
    # ⚠️ Case 2: test set does NOT contain lagged_forward_returns
    #
    # In this case, fall back to using the *last known forward_return*
    # from the training set as a rough proxy for the most recent trend.
    last_return = train['forward_returns'].iloc[-1]
    test['prediction'] = 1.2 if last_return > 0 else 0.8

# -------------------------------------------------------------
# 4️⃣ Post-process and enforce valid bounds
# -------------------------------------------------------------
# Kaggle rules: predictions (portfolio weights) must be between [0, 2].
#   0 → fully out of market (all cash)
#   1 → fully invested (neutral)
#   2 → fully leveraged long (2x)
#
# Clipping ensures we never exceed those limits, even if a bug occurs.
test['prediction'] = test['prediction'].clip(0, 2)

# -------------------------------------------------------------
# 5️⃣ Determine identifier column name
# -------------------------------------------------------------
# The test.csv file uses 'date_id' as the unique identifier, but
# to keep this script flexible for other datasets, we automatically
# search for common id column names (row_id, id, index, etc.).
row_id_col = next(
    (c for c in test.columns if c in ['row_id', 'id', 'index', 'date_id', 'rowid']),
    test.columns[0]  # fallback if none of the above exist
)

# -------------------------------------------------------------
# 6️⃣ Save submission file
# -------------------------------------------------------------
# The final CSV will contain only two columns:
#   1. id column (e.g. date_id)
#   2. prediction (portfolio weight)
#
# The file will be saved as:
#   submissions/rule_pred.csv
test[[row_id_col, 'prediction']].to_csv(OUT_DIR / 'rule_pred.csv', index=False)

print("Saved rule_pred.csv")
