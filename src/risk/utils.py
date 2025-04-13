import numpy as np
import pandas as pd
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
