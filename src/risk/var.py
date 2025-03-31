import numpy as np
import pandas as pd
import os
from src.config import PROCESSED_DATA_DIR, CONFIDENCE_LEVEL, MONTE_CARLO_SIMULATIONS
from src.risk.utils import calculate_daily_returns

def historical_var(returns, confidence_level=0.95):
    """
    Calculates historical VaR given returns.

    Args:
        returns (pd.Series): Daily returns of the asset.
        confidence_level (float): Confidence level for VaR calculation (default 0.95).

    Returns:
        float: Historical VaR.
    """
    if returns.empty:
        raise ValueError("Returns series is empty.")

    var = np.percentile(returns.dropna(), (1 - confidence_level) * 100)
    return abs(var)


def parametric_var(returns, confidence_level=0.95):
    """
    Calculates parametric VaR assuming returns are normally distributed.

    Args:
        returns (pd.Series): Daily returns of the asset.
        confidence_level (float): Confidence level for VaR calculation (default 0.95).

    Returns:
        float: Parametric VaR.
    """
    if returns.empty:
        raise ValueError("Returns series is empty.")

    mean = returns.mean()
    std_dev = returns.std()
    z_score = np.abs(np.percentile(np.random.normal(0, 1, 100000), (1 - confidence_level) * 100))
    var = mean - z_score * std_dev

    return abs(var)


def monte_carlo_var(returns, confidence_level=0.95, simulations=10000):
    """
    Calculates VaR using Monte Carlo simulations assuming normality of returns.

    Args:
        returns (pd.Series): Daily returns of the asset.
        confidence_level (float): Confidence level for VaR calculation (default 0.95).
        simulations (int): Number of Monte Carlo simulations (default 10,000).

    Returns:
        float: Monte Carlo VaR.
    """
    if returns.empty:
        raise ValueError("Returns series is empty.")

    mean = returns.mean()
    std_dev = returns.std()

    simulated_returns = np.random.normal(mean, std_dev, simulations)
    var = np.percentile(simulated_returns, (1 - confidence_level) * 100)

    return abs(var)


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


if __name__ == "__main__":
    example_data_path = os.path.join(PROCESSED_DATA_DIR, '2025-03-30_sp500.csv')  # adjust date accordingly

    df = pd.read_csv(example_data_path, index_col=0)
    price_series = df.iloc[:, 0]

    returns = calculate_daily_returns(price_series)

    print("Historical VaR:", historical_var(returns, CONFIDENCE_LEVEL))
    print("Parametric VaR:", parametric_var(returns, CONFIDENCE_LEVEL))
    print("Monte Carlo VaR:", monte_carlo_var(returns, CONFIDENCE_LEVEL, MONTE_CARLO_SIMULATIONS))
