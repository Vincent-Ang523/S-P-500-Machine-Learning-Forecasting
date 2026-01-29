# =============================================
# src/metric.py
# =============================================
# This script implements the *custom evaluation metric*
# used in the Hull Tactical Market Prediction competition.
#
# Specifically, it computes a *volatility-adjusted Sharpe ratio*,
# which measures risk-adjusted performance of a trading strategy.
#
# The metric compares the model's "strategy" returns against
# the market's returns and penalizes:
#   1) Excess volatility (being too risky),
#   2) Underperforming mean return (earning less than market).
#
# This version is written for local evaluation (using train.csv
# as a stand-in for the hidden test ground-truth).
#
# =============================================

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------
# Define data directory (relative to this file)
# ---------------------------------------------
# This locates the project's "data/" folder by going
# up one directory from src/ and appending /data.
# Using Path objects makes this OS-independent.
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


# =====================================================
# Function: compute_adjusted_sharpe
# =====================================================
# Computes the competition's custom adjusted Sharpe ratio.
# Arguments:
#   solution:  ground-truth DataFrame (contains forward_returns, risk_free_rate, etc.)
#   submission: predicted portfolio weights DataFrame (contains prediction column)
#   row_id_col: name of the ID column used to merge (default 'row_id')
#
# Returns:
#   adjusted_sharpe: float score (higher = better)
# =====================================================
def compute_adjusted_sharpe(solution: pd.DataFrame, submission: pd.DataFrame, row_id_col='row_id'):
    # -------------------------------------------------
    # 1️⃣ Merge predictions with ground truth
    # -------------------------------------------------
    # Combine the "solution" (train.csv) and "submission" (your predictions)
    # by matching on the identifier column (e.g. date_id).
    # The result df will contain:
    #   - model predictions
    #   - forward_returns, risk_free_rate (actuals)
    df = solution.merge(submission, on=row_id_col, how='left')

    # Ensure every row in solution has a prediction
    if df['prediction'].isnull().any():
        raise ValueError("Missing predictions for some rows after merge")

    # Ensure prediction values are numeric (not strings)
    if not np.issubdtype(df['prediction'].dtype, np.number):
        raise ValueError("Predictions must be numeric")

    # Ensure predictions are within allowed bounds [0, 2]
    # because competition rules require portfolio weights between 0x and 2x leverage
    if df['prediction'].max() > 2 or df['prediction'].min() < 0:
        raise ValueError("Predictions out of bounds [0,2]")

    # -------------------------------------------------
    # 2️⃣ Compute "strategy returns"
    # -------------------------------------------------
    # Each day's portfolio return = weighted mix between:
    #   - risk-free rate (cash portion)
    #   - market forward return (equity portion)
    #
    # If prediction (position) = 1.0 → fully invested in market.
    # If <1 → partially invested; >1 → leveraged.
    df['position'] = df['prediction']
    df['strategy_returns'] = (
        df['risk_free_rate'] * (1 - df['position'])
        + df['position'] * df['forward_returns']
    )

    # -------------------------------------------------
    # 3️⃣ Compute "excess returns" of strategy
    # -------------------------------------------------
    # Excess return = return - risk-free rate
    # i.e., how much better the strategy did than staying in cash.
    strategy_excess = df['strategy_returns'] - df['risk_free_rate']

    # Product of (1 + returns) across all days gives cumulative growth.
    # This is used to calculate *geometric mean* of returns.
    strategy_excess_cum = (1 + strategy_excess).prod()
    n = len(df)  # number of days

    # Edge case: if cumulative product ≤ 0 (e.g., massive negative returns)
    # → geometric mean undefined, fallback to arithmetic mean.
    if strategy_excess_cum <= 0:
        strategy_mean_excess = strategy_excess.mean()
    else:
        # Geometric mean excess return per day
        strategy_mean_excess = strategy_excess_cum ** (1 / n) - 1

    # -------------------------------------------------
    # 4️⃣ Compute volatility and Sharpe ratio
    # -------------------------------------------------
    # Standard deviation of daily returns (volatility)
    strategy_std = df['strategy_returns'].std(ddof=0)
    trading_days_per_yr = 252  # approximate number of trading days

    # If std == 0 → all returns identical (undefined Sharpe)
    if strategy_std == 0:
        raise ValueError("strategy std is zero")

    # Annualized Sharpe ratio:
    # (mean excess return / std deviation) * sqrt(252)
    sharpe = strategy_mean_excess / strategy_std * np.sqrt(trading_days_per_yr)

    # Annualized volatility in percentage terms (for penalty calc)
    strategy_volatility = float(strategy_std * np.sqrt(trading_days_per_yr) * 100.0)

    # -------------------------------------------------
    # 5️⃣ Compute *market* metrics for comparison
    # -------------------------------------------------
    # Same calculations as above, but for the raw market returns.
    market_excess = df['forward_returns'] - df['risk_free_rate']
    market_excess_cum = (1 + market_excess).prod()
    if market_excess_cum <= 0:
        market_mean_excess = market_excess.mean()
    else:
        market_mean_excess = market_excess_cum ** (1 / n) - 1

    market_std = df['forward_returns'].std(ddof=0)
    market_volatility = float(market_std * np.sqrt(trading_days_per_yr) * 100.0)

    if market_volatility == 0:
        raise ValueError("market std is zero")

    # -------------------------------------------------
    # 6️⃣ Apply competition penalties
    # -------------------------------------------------
    # Two penalties are applied to discourage excessive risk-taking
    # or underperformance relative to the market.

    # ① Volatility penalty:
    # If strategy volatility > 1.2x market volatility,
    # apply a proportional penalty.
    excess_vol = max(0, strategy_volatility / market_volatility - 1.2)
    vol_penalty = 1 + excess_vol

    # ② Return penalty:
    # If your mean excess return < market's,
    # penalize quadratically by how much you underperform.
    return_gap = max(0, (market_mean_excess - strategy_mean_excess) * 100 * trading_days_per_yr)
    return_penalty = 1 + (return_gap ** 2) / 100

    # -------------------------------------------------
    # 7️⃣ Final adjusted Sharpe ratio
    # -------------------------------------------------
    # Divide raw Sharpe by both penalties.
    adjusted_sharpe = sharpe / (vol_penalty * return_penalty)

    # Cap at 1,000,000 to prevent runaway large scores (Kaggle safeguard)
    return min(float(adjusted_sharpe), 1_000_000)


# =====================================================
# CLI Entry Point
# =====================================================
# This block allows running the script directly to test
# the metric locally using your baseline submission.
# =====================================================
if __name__ == "__main__":
    # Load train.csv as local ground-truth (since Kaggle’s test labels are private)
    solution = pd.read_csv(DATA_DIR / "train.csv")

    # Load your submission file from submissions/baseline_pred.csv
    sub = pd.read_csv(
        Path(__file__).resolve().parents[1] / "submissions" / "baseline_pred.csv"
    )

    # Detect which column to use as the id (date_id or row_id, etc.)
    candidate_ids = ['row_id', 'id', 'index', 'date_id', 'rowid']
    row_id_col = next((c for c in sub.columns if c in candidate_ids), sub.columns[0])

    print("Using id column:", row_id_col)

    # Compute and print baseline adjusted Sharpe score
    score = compute_adjusted_sharpe(solution, sub, row_id_col=row_id_col)
    print("Adjusted Sharpe (baseline):", score)
    