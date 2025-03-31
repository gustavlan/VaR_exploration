import os
from datetime import datetime

# Base project directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data Directories
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# Market Data Settings
START_DATE = '2017-01-01'
END_DATE = datetime.today().strftime('%Y-%m-%d')

# Tickers
TICKERS = {
    'S&P 500': '^GSPC',
    'NASDAQ': '^IXIC'
}

# VaR Calculation Settings
CONFIDENCE_LEVEL = 0.95
MONTE_CARLO_SIMULATIONS = 10000
TRADING_DAYS_PER_YEAR = 252

# Risk-free Rate (annualized)
RISK_FREE_RATE = 0.02  # Adjust based on current market conditions
