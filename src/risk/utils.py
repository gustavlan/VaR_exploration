import os
from datetime import datetime
from typing import Union, Dict, List, Optional
import logging

import numpy as np
import pandas as pd

from config import TRADING_DAYS_PER_YEAR, RISK_FREE_RATE

logger = logging.getLogger(__name__)


def annualize_volatility(
    daily_volatility: Union[pd.Series, float], trading_days: int = TRADING_DAYS_PER_YEAR
) -> Union[pd.Series, float]:
    """Annualized volatility as ``daily_volatility * sqrt(trading_days)``."""
    return daily_volatility * np.sqrt(trading_days)


def annualize_return(
    daily_return: Union[pd.Series, float], trading_days: int = TRADING_DAYS_PER_YEAR
) -> Union[pd.Series, float]:
    """Annualized return as ``daily_return * trading_days``."""
    return daily_return * trading_days


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = RISK_FREE_RATE,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized Sharpe ratio of ``returns``."""
    excess = returns - risk_free_rate / trading_days
    ann_excess_ret = annualize_return(excess.mean(), trading_days)
    ann_vol = annualize_volatility(excess.std(), trading_days)
    if ann_vol == 0:
        raise ValueError("Volatility is zero, Sharpe ratio undefined.")
    return ann_excess_ret / ann_vol


def calculate_daily_returns(price_series: pd.Series) -> pd.Series:
    """Simple daily returns. Raises ``ValueError`` if series is empty."""
    if price_series.empty:
        raise ValueError("Price series is empty.")
    returns = price_series.pct_change().dropna()
    return returns


def calculate_log_returns(prices: pd.Series) -> pd.Series:
    """Daily log returns."""
    return np.log(prices / prices.shift(1)).dropna()


def calculate_forward_log_returns(
    prices: pd.Series, days_forward: int = 10
) -> pd.Series:
    """Forward log returns over ``days_forward`` days."""
    fwd = np.log(prices.shift(-days_forward) / prices)
    return fwd.dropna()


def calculate_rolling_volatility(returns: pd.Series, window: int = 21) -> pd.Series:
    """Rolling standard deviation of ``returns``."""
    return returns.rolling(window=window).std()


def calculate_parametric_var(
    volatility: pd.Series, confidence_z: float = -2.33, horizon_days: int = 10
) -> pd.Series:
    """Parametric VaR: ``confidence_z * sqrt(horizon_days) * volatility``."""
    return confidence_z * np.sqrt(horizon_days) * volatility


def load_latest_price_data(
    directory: str, keyword: str, date_threshold: float = 0.9
) -> pd.DataFrame:
    """Return latest CSV containing ``keyword`` as a numeric DataFrame.

    The date column is chosen using ``date_threshold``.
    """
    # 1) find the latest file
    files: List[str] = [
        f for f in os.listdir(directory) if keyword in f and f.endswith(".csv")
    ]
    if not files:
        raise FileNotFoundError(f"No CSVs for '{keyword}' in {directory!r}")

    dated = []
    for fname in files:
        try:
            dt = datetime.strptime(fname.split("_")[0], "%Y-%m-%d")
            dated.append((fname, dt))
        except ValueError:
            continue
    if not dated:
        raise FileNotFoundError(
            f"No properly dated files for '{keyword}' in {directory!r}"
        )

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
        parsed = pd.to_datetime(df_raw[col], errors="coerce")
        frac = parsed.notna().mean()
        if frac >= date_threshold:
            date_col = col
            df_raw[col] = parsed
            logger.info(f"Detected date column '{col}' ({frac:.0%} parseable)")
            break

    if date_col is None:
        raise ValueError(
            f"No column in {path!r} had ≥{date_threshold:.0%} parseable dates"
        )

    # set index
    df = df_raw.set_index(date_col, drop=True)

    # numeric columns
    df = (
        df.apply(pd.to_numeric, errors="coerce")
        .dropna(how="any")
        .astype("float64", copy=False)
    )

    logger.info(f"Loaded {len(df)} rows with columns {list(df.columns)}")
    return df


def detect_var_breaches(
    df: pd.DataFrame, return_col: str, var_col: str, breach_col: str = "breach"
) -> pd.DataFrame:
    """Add a boolean column marking VaR breaches."""
    df[breach_col] = (df[return_col] < df[var_col]) & (df[return_col] < 0)
    return df


def summarize_var_breaches(
    df: pd.DataFrame, breach_col: str = "breach"
) -> Dict[str, Union[int, float]]:
    """Return breach count and percentage."""
    count = int(df[breach_col].sum())
    pct = float(round(df[breach_col].mean(), 3))
    return {"count": count, "percentage": pct}
