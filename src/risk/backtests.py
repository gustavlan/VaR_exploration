import numpy as np
import pandas as pd
from scipy.stats import chi2


def kupiec_pof_test(breaches: pd.Series, alpha: float) -> dict:
    """
    Kupiec Proportion of Failures (POF) test for VaR backtesting.

    Parameters
    ----------
    breaches
        Boolean Series where True indicates a VaR breach.
    alpha
        VaR confidence level (e.g. 0.95 for 95% VaR).

    Returns
    -------
    dict with keys
      - 'n': total observations
      - 'x': number of breaches
      - 'p_hat': observed failure rate (x/n)
      - 'LR': likelihood ratio statistic
      - 'p_value': p-value under Chi2(1)
    """
    n = len(breaches)
    x = int(breaches.sum())
    p = 1 - alpha
    p_hat = x / n

    # avoid zeros in log
    if p_hat in (0, 1):
        LR = np.nan
        p_value = np.nan
    else:
        # Log‐likelihood ratio
        num = (1 - p) ** (n - x) * p**x
        den = (1 - p_hat) ** (n - x) * p_hat**x
        LR = -2 * np.log(num / den)
        p_value = 1 - chi2.cdf(LR, df=1)

    return {"n": n, "x": x, "p_hat": p_hat, "LR": LR, "p_value": p_value}


def christoffersen_independence_test(breaches: pd.Series) -> dict:
    """
    Christoffersen test for independence of VaR breaches.

    Builds a 2×2 transition matrix:
        -- from no‐breach (0) -- from breach (1)
    to no‐breach (0): N00, N10
       breach   (1): N01, N11

    Under H0 (independence), the probability of breach does not
    depend on the previous day’s state.

    Returns
    -------
    dict with keys
      - transition_counts: dict of N00, N01, N10, N11
      - LR: likelihood‐ratio statistic
      - p_value: p‐value under Chi2(1)
    """
    # build transitions
    b = breaches.astype(int).values
    N00 = np.sum((b[:-1] == 0) & (b[1:] == 0))
    N01 = np.sum((b[:-1] == 0) & (b[1:] == 1))
    N10 = np.sum((b[:-1] == 1) & (b[1:] == 0))
    N11 = np.sum((b[:-1] == 1) & (b[1:] == 1))

    # probs
    pi0 = N01 / (N00 + N01) if (N00 + N01) > 0 else 0
    pi1 = N11 / (N10 + N11) if (N10 + N11) > 0 else 0
    pi = (N01 + N11) / (N00 + N01 + N10 + N11)

    # log‐likelihoods
    def ll(n0, n1, p):
        return n0 * np.log(1 - p) + n1 * np.log(p)

    ll_ind = ll(N00 + N10, N01 + N11, pi)
    ll_markov = ll(N00, N01, pi0) + ll(N10, N11, pi1)
    LR = -2 * (ll_ind - ll_markov)
    p_value = 1 - chi2.cdf(LR, df=1)

    return {
        "N00": N00,
        "N01": N01,
        "N10": N10,
        "N11": N11,
        "pi0": pi0,
        "pi1": pi1,
        "pi": pi,
        "LR": LR,
        "p_value": p_value,
    }


def expected_shortfall(returns: pd.Series, alpha: float) -> float:
    """
    Compute Expected Shortfall (Conditional VaR) at level alpha.

    ES_α = E[–r | r ≤ –VaR_α]

    Parameters
    ----------
    returns
        Series of portfolio returns.
    alpha
        VaR confidence level (e.g. 0.95).

    Returns
    -------
    float
        ES (positive number).
    """
    # losses are -returns; ES is average loss beyond VaR
    var_level = np.percentile(returns.dropna(), (1 - alpha) * 100)
    tail = returns[returns <= var_level]
    if len(tail) == 0:
        return 0.0
    return -tail.mean()
