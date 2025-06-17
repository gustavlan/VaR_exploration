import numpy as np
import pandas as pd
from scipy.stats import chi2


def kupiec_pof_test(breaches: pd.Series, alpha: float) -> dict:
    """Kupiec's POF test comparing the breach rate to ``1 - alpha``.

    Returns ``{'n', 'x', 'p_hat', 'LR', 'p_value'}``.
    """
    n = len(breaches)
    x = int(breaches.sum())
    p = 1 - alpha
    p_hat = x / n

    # avoid log(0)
    if p_hat in (0, 1):
        LR = np.nan
        p_value = np.nan
    else:
        # log-likelihood ratio
        num = (1 - p) ** (n - x) * p**x
        den = (1 - p_hat) ** (n - x) * p_hat**x
        LR = -2 * np.log(num / den)
        p_value = 1 - chi2.cdf(LR, df=1)

    return {"n": n, "x": x, "p_hat": p_hat, "LR": LR, "p_value": p_value}


def christoffersen_independence_test(breaches: pd.Series) -> dict:
    """Christoffersen independence test for a breach sequence.

    Returns the transition counts and test statistics.
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
    """Expected shortfall at level ``alpha`` for the given returns."""
    # average loss beyond VaR
    var_level = np.percentile(returns.dropna(), (1 - alpha) * 100)
    tail = returns[returns <= var_level]
    if len(tail) == 0:
        return 0.0
    return -tail.mean()
