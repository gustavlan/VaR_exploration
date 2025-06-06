import os
from datetime import datetime
from typing import Union, Dict, List, Optional
import logging

import numpy as np
import pandas as pd

from config import TRADING_DAYS_PER_YEAR, RISK_FREE_RATE

logger = logging.getLogger(__name__)


def annualize_volatility(
    daily_volatility: Union[pd.Series, float],
    trading_days: int = TRADING_DAYS_PER_YEAR
) -> Union[pd.Series, float]:
    """
    Annualize a daily volatility series or scalar.

    Parameters
    ----------
    daily_volatility
        Daily standard deviation of returns (series or scalar).
    trading_days
        Number of trading days in a year.

    Returns
    -------
    pd.Series or float
        Annualized volatility: daily_volatility * sqrt(trading_days).
    """
    return daily_volatility * np.sqrt(trading_days)


def annualize_return(
    daily_return: Union[pd.Series, float],
    trading_days: int = TRADING_DAYS_PER_YEAR
) -> Union[pd.Series, float]:
    """
    Annualize a daily return series or scalar.

    Parameters
    ----------
    daily_return
        Daily return (series or scalar).
    trading_days
        Number of trading days in a year.

    Returns
    -------
    pd.Series or float
        Annualized return: daily_return * trading_days.
    """
    return daily_return * trading_days


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = RISK_FREE_RATE,
    trading_days: int = TRADING_DAYS_PER_YEAR
) -> float:
    """
    Compute the annualized Sharpe ratio of a return series.

    Parameters
    ----------
    returns
        Series of daily returns.
    risk_free_rate
        Annual risk-free rate (as a decimal, e.g. 0.02 for 2%).
    trading_days
        Number of trading days per year.

    Returns
    -------
    float
        Annualized Sharpe ratio: (annualized excess return) / (annualized volatility).

    Raises
    ------
    ValueError
        If volatility is zero (to avoid division by zero).
    """
    excess = returns - risk_free_rate / trading_days
    ann_excess_ret = annualize_return(excess.mean(), trading_days)
    ann_vol = annualize_volatility(excess.std(), trading_days)
    if ann_vol == 0:
        raise ValueError("Volatility is zero, Sharpe ratio undefined.")
    return ann_excess_ret / ann_vol


def calculate_daily_returns(price_series: pd.Series) -> pd.Series:
    """
    Calculate simple daily returns from a price series.

    Parameters
    ----------
    price_series
        Time-indexed price series.

    Returns
    -------
    pd.Series
        Daily simple returns (pct_change), with leading NaN dropped.

    Raises
    ------
    ValueError
        If the input series is empty.
    """
    if price_series.empty:
        raise ValueError("Price series is empty.")
    returns = price_series.pct_change().dropna()
    return returns


def calculate_log_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate daily log returns from a price series.

    Parameters
    ----------
    prices
        Time-indexed price series.

    Returns
    -------
    pd.Series
        Daily log returns: log(p_t / p_{t-1}).
    """
    return np.log(prices / prices.shift(1)).dropna()


def calculate_forward_log_returns(
    prices: pd.Series,
    days_forward: int = 10
) -> pd.Series:
    """
    Calculates forward‑looking log returns over a specified number of days.
    Drops the trailing NaNs created by the shift.

    Returns
    -------
    pd.Series
        Forward log returns: log(p_{t+days_forward} / p_t), with last
        `days_forward` rows removed.
    """
    fwd = np.log(prices.shift(-days_forward) / prices)
    return fwd.dropna()


def calculate_rolling_volatility(
    returns: pd.Series,
    window: int = 21
) -> pd.Series:
    """
    Calculate rolling volatility (standard deviation) of returns.

    Parameters
    ----------
    returns
        Series of returns.
    window
        Rolling window size in days.

    Returns
    -------
    pd.Series
        Rolling standard deviation of returns.
    """
    return returns.rolling(window=window).std()


def calculate_parametric_var(
    volatility: pd.Series,
    confidence_z: float = -2.33,
    horizon_days: int = 10
) -> pd.Series:
    """
    Calculate parametric VaR from a volatility series.

    Parameters
    ----------
    volatility
        Rolling volatility (std of returns).
    confidence_z
        Negative z-score corresponding to the VaR quantile
        (e.g. -1.645 for 95%, -2.33 for 99%).
    horizon_days
        VaR horizon in days.

    Returns
    -------
    pd.Series
        Parametric VaR series: confidence_z * sqrt(horizon_days) * volatility.
    """
    return confidence_z * np.sqrt(horizon_days) * volatility


def load_latest_price_data(
    directory: str,
    keyword: str,
    date_threshold: float = 0.9
) -> pd.DataFrame:
    """
    Load the most recent CSV in `directory` whose filename contains `keyword`,
    auto-detect its date column, and return a clean time-indexed DataFrame.

    Parameters
    ----------
    directory : str
        Directory containing CSVs named like 'YYYY-MM-DD_<keyword>.csv'.
    keyword : str
        Substring to match in filenames (e.g. 'nasdaq').
    date_threshold : float
        Minimum fraction of parseable datetimes in a column to pick it
        as the date column (default 0.9).

    Returns
    -------
    pd.DataFrame
        Time-indexed DataFrame of float64 values with no NaNs.

    Raises
    ------
    FileNotFoundError
        If no file matching `keyword` is found.
    ValueError
        If no column can be confidently parsed as dates.
    """
    # 1) find the latest file
    files: List[str] = [
        f for f in os.listdir(directory)
        if keyword in f and f.endswith('.csv')
    ]
    if not files:
        raise FileNotFoundError(f"No CSVs for '{keyword}' in {directory!r}")

    dated = []
    for fname in files:
        try:
            dt = datetime.strptime(fname.split('_')[0], '%Y-%m-%d')
            dated.append((fname, dt))
        except ValueError:
            continue
    if not dated:
        raise FileNotFoundError(f"No properly dated files for '{keyword}' in {directory!r}")

    latest_file = max(dated, key=lambda x: x[1])[0]
    path = os.path.join(directory, latest_file)
    logger.info(f"Loading data from {path!r}")

    # 2) read the CSV raw
    df_raw = pd.read_csv(path)
    if df_raw.shape[1] < 2:
        raise ValueError(f"Expected ≥2 columns in {path!r}, got {df_raw.shape[1]}")

    # 3) auto-detect date column
    date_col: Optional[str] = None
    for col in df_raw.columns:
        parsed = pd.to_datetime(df_raw[col], errors='coerce', infer_datetime_format=True)
        frac = parsed.notna().mean()
        if frac >= date_threshold:
            date_col = col
            df_raw[col] = parsed
            logger.info(f"Detected date column '{col}' ({frac:.0%} parseable)")
            break

    if date_col is None:
        raise ValueError(f"No column in {path!r} had ≥{date_threshold:.0%} parseable dates")

    # 4) set index (drop the date column automatically)
    df = df_raw.set_index(date_col, drop=True)

    # 5) convert all remaining columns to float, drop NaNs
    df = df.apply(pd.to_numeric, errors='coerce') \
           .dropna(how='any') \
           .astype('float64', copy=False)

    logger.info(f"Loaded {len(df)} rows with columns {list(df.columns)}")
    return df


def detect_var_breaches(
    df: pd.DataFrame,
    return_col: str,
    var_col: str,
    breach_col: str = 'breach'
) -> pd.DataFrame:
    """
    Flag rows where returns breach the VaR threshold.

    Parameters
    ----------
    df
        DataFrame containing return and VaR columns.
    return_col
        Column name for returns.
    var_col
        Column name for VaR values.
    breach_col
        Name for the boolean output column (default 'breach').

    Returns
    -------
    pd.DataFrame
        Original DataFrame with an added boolean breach column.
    """
    df[breach_col] = (df[return_col] < df[var_col]) & (df[return_col] < 0)
    return df


def summarize_var_breaches(
    df: pd.DataFrame,
    breach_col: str = 'breach'
) -> Dict[str, Union[int, float]]:
    """
    Summarize the count and percentage of VaR breaches.

    Parameters
    ----------
    df
        DataFrame containing a boolean breach column.
    breach_col
        Name of the breach flag column.

    Returns
    -------
    dict
        {'count': int, 'percentage': float}
    """
    count = int(df[breach_col].sum())
    pct = float(round(df[breach_col].mean(), 3))
    return {'count': count, 'percentage': pct}
