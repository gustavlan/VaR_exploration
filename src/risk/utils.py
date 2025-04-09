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
    Converts a series of asset prices into daily returns.

    Args:
        price_series (pd.Series): Asset prices.

    Returns:
        pd.Series: Daily returns.
    """
    if price_series.empty:
        raise ValueError("Price series is empty.")

    returns = price_series.pct_change().dropna()
    return returns