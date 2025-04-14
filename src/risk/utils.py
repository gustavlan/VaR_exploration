import os
import numpy as np
import pandas as pd
from datetime import datetime
from config import TRADING_DAYS_PER_YEAR, RISK_FREE_RATE

def annualize_volatility(daily_volatility, trading_days=TRADING_DAYS_PER_YEAR):
    return daily_volatility * np.sqrt(trading_days)

def annualize_return(daily_return, trading_days=TRADING_DAYS_PER_YEAR):
    return daily_return * trading_days

def sharpe_ratio(returns, risk_free_rate=RISK_FREE_RATE, trading_days=TRADING_DAYS_PER_YEAR):
    excess_daily_returns = returns - risk_free_rate / trading_days
    annualized_excess_return = annualize_return(excess_daily_returns.mean(), trading_days)
    annualized_vol = annualize_volatility(excess_daily_returns.std(), trading_days)
    
    if annualized_vol == 0:
        raise ValueError("Volatility is zero, Sharpe ratio undefined.")
    
    return annualized_excess_return / annualized_vol

def calculate_daily_returns(price_series):
    """
    Calculates daily returns of a given price series.
    """
    if price_series.empty:
        raise ValueError("Price series is empty.")

    returns = price_series.pct_change().dropna()
    return returns

def calculate_log_returns(prices: pd.Series) -> pd.Series:
    """
    Calculates daily log returns of a given price series.
    """
    return np.log(prices / prices.shift(1))


def calculate_forward_log_returns(prices: pd.Series, days_forward: int = 10) -> pd.Series:
    """
    Calculates forward-looking log returns over a specified number of days.
    """
    return np.log(prices.shift(-days_forward) / prices)


def calculate_rolling_volatility(returns: pd.Series, window: int = 21) -> pd.Series:
    """
    Calculates rolling standard deviation (volatility) of returns.
    """
    return returns.rolling(window=window).std()


def calculate_parametric_var(volatility: pd.Series, confidence_z: float = -2.33, horizon_days: int = 10) -> pd.Series:
    """
    Calculates parametric VaR given volatility, confidence level (z-score),
    and time horizon (days).
    """
    return confidence_z * np.sqrt(horizon_days) * volatility

def load_latest_price_data(directory: str, keyword: str) -> pd.DataFrame:
    """
    Loads the latest CSV file for a given keyword, ensuring numeric data types and clean index.

    Args:
        directory (str): Directory path with CSV files.
        keyword (str): Keyword identifying the CSV files.

    Returns:
        pd.DataFrame: Clean, numeric DataFrame with date index.
    """
    files = [
        f for f in os.listdir(directory)
        if keyword in f and f.endswith('.csv')
    ]

    if not files:
        raise FileNotFoundError(f"No files found for keyword '{keyword}' in {directory}")

    # Extract dates and find latest file
    files_dates = [
        (f, datetime.strptime(f.split('_')[0], '%Y-%m-%d'))
        for f in files
    ]
    latest_file = max(files_dates, key=lambda x: x[1])[0]
    latest_filepath = os.path.join(directory, latest_file)

    # Explicitly load CSV, ensuring numeric conversion and clean index
    df = pd.read_csv(latest_filepath, index_col=0, parse_dates=True)

    # Rename index clearly
    df.index.name = 'Date'

    # Explicit numeric conversion for all columns robustly
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Explicitly remove rows with any NaN values
    df.dropna(inplace=True)

    # Explicitly convert columns to numeric type (float64)
    df = df.astype('float64')

    print(f"Loaded data from: {latest_filepath}")
    return df

def detect_var_breaches(df: pd.DataFrame, return_col: str, var_col: str, breach_col: str = 'breach') -> pd.DataFrame:
    """
    Adds a column indicating where a VaR breach occurred.

    Args:
        df (pd.DataFrame): DataFrame with return and VaR columns.
        return_col (str): Name of the column with returns.
        var_col (str): Name of the column with the calculated VaR.
        breach_col (str): Name of the output column to flag breaches (default 'breach').

    Returns:
        pd.DataFrame: Same DataFrame with a new boolean column indicating breaches.
    """
    df[breach_col] = (df[return_col] < df[var_col]) & (df[return_col] < 0)
    return df


def summarize_var_breaches(df: pd.DataFrame, breach_col: str = 'breach') -> dict:
    """
    Summarizes the number and percentage of VaR breaches.

    Args:
        df (pd.DataFrame): DataFrame with a boolean breach column.
        breach_col (str): Name of the breach column.

    Returns:
        dict: Dictionary with breach count and percentage.
    """
    count = df[breach_col].sum()
    pct = round(df[breach_col].mean(), 3)
    return {'count': count, 'percentage': pct}
