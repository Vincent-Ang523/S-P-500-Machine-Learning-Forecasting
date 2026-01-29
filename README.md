# Stock-Market-ML-Forecasting
**A machine learning framework for risk-adjusted stock market forecasting**

This project builds a complete ML pipeline for predicting the **daily excess returns** of the S&P 500 and optimizing a portfolio’s **risk-adjusted Sharpe ratio**. **More details on algorithm and portfolio strategy used can be found in Project Report.pdf file**

---

## Overview
- **Goal:** Predict daily portfolio exposure (0–2x leverage) based on macro, sentiment, and volatility features.
- **Evaluation Metric:** Custom **volatility-adjusted Sharpe ratio**.
- **Core Idea:** Combine feature engineering, cross-validation, and ensemble models to generate stable, low-volatility predictions.

---

## Features
- Time-series cross-validation (no leakage)
- Feature engineering (lags, momentum, rolling stats)
- Ridge, LightGBM, and ensemble models
- Custom Sharpe metric implementation
- Full backtesting and evaluation pipeline

---

## Project Structure
