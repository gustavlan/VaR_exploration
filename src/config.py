import os
from datetime import datetime

from scipy.stats import norm

# Base project directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data Directories
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Market Data Settings
START_DATE = "2017-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

# Tickers
TICKERS = {"S&P 500": "^GSPC", "NASDAQ": "^IXIC"}

# VaR Calculation Settings
CONFIDENCE_LEVEL = 0.95
MONTE_CARLO_SIMULATIONS = 10000
TRADING_DAYS_PER_YEAR = 252

# Risk-free Rate (annualized)
RISK_FREE_RATE = 0.02  # Adjust based on current market conditions

# VaR horizon (days)
VAR_HORIZON_DAYS = 10

# Compute the corresponding z‐score for the one‐tailed lower‐quantile VaR
# e.g. if CONFIDENCE_LEVEL = 0.95 → norm.ppf(1 - 0.95) ≈ -1.645
CONFIDENCE_Z = norm.ppf(1 - CONFIDENCE_LEVEL)

# EWMA smoothing parameter
EWMA_LAMBDA = 0.72  # decay factor λ
EWMA_ALPHA = 1 - EWMA_LAMBDA  # smoothing factor α
