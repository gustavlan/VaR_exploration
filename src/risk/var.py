import os
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from config import PROCESSED_DATA_DIR, CONFIDENCE_LEVEL, MONTE_CARLO_SIMULATIONS
from risk.utils import calculate_daily_returns


def historical_var(
    returns: pd.Series,
    confidence_level: float = CONFIDENCE_LEVEL
) -> float:
    """
    Compute historical VaR at a specified confidence level.

    Parameters
    ----------
    returns
        Series of daily returns.
    confidence_level
        VaR confidence level (e.g. 0.95 for 95%).

    Returns
    -------
    float
        Absolute historical VaR (positive number).

    Raises
    ------
    ValueError
        If the returns series is empty.
    """
    if returns.empty:
        raise ValueError("Returns series is empty.")
    var_pct = (1 - confidence_level) * 100
    var_value = np.percentile(returns.dropna(), var_pct)
    return abs(var_value)


def parametric_var(
    returns: pd.Series,
    confidence_level: float = CONFIDENCE_LEVEL
) -> float:
    """
    Compute parametric (Gaussian) VaR assuming normal returns.

    Parameters
    ----------
    returns
        Series of daily returns.
    confidence_level
        VaR confidence level.

    Returns
    -------
    float
        Absolute parametric VaR.
    """
    if returns.empty:
        raise ValueError("Returns series is empty.")
    mean = returns.mean()
    std = returns.std()
    # deterministic z-score for the one-tailed lower quantile
    z = norm.ppf(1 - confidence_level)
    var = -(mean + z * std)
    return abs(var)


def monte_carlo_var(
    returns: pd.Series,
    confidence_level: float = CONFIDENCE_LEVEL,
    simulations: int = MONTE_CARLO_SIMULATIONS
) -> float:
    """
    Compute VaR via Monte Carlo simulation under normality assumption.

    Parameters
    ----------
    returns
        Series of daily returns.
    confidence_level
        VaR confidence level.
    simulations
        Number of simulated return paths.

    Returns
    -------
    float
        Absolute Monte Carlo VaR.
    """
    if returns.empty:
        raise ValueError("Returns series is empty.")
    mean = returns.mean()
    std = returns.std()
    sims = np.random.normal(mean, std, simulations)
    var = np.percentile(sims, (1 - confidence_level) * 100)
    # np.percentile returns a numpy scalar; cast to float for clarity
    return float(abs(var))


if __name__ == "__main__":
    # Simple example usage
    example = os.path.join(PROCESSED_DATA_DIR, '2025-03-30_sp500.csv')
    df = pd.read_csv(example, index_col=0)
    prices = df.iloc[:, 0]
    rets = calculate_daily_returns(prices)
    print("Historical VaR:", historical_var(rets))
    print("Parametric VaR:", parametric_var(rets))
    print("Monte Carlo VaR:", monte_carlo_var(rets))
