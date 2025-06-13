import numpy as np
import pandas as pd
import pytest

from risk.backtests import (
    kupiec_pof_test,
    christoffersen_independence_test,
    expected_shortfall,
)


def test_kupiec_pof_counts_and_pvalue():
    breaches = pd.Series([False] * 95 + [True] * 5)
    res = kupiec_pof_test(breaches, alpha=0.95)
    assert res['n'] == 100
    assert res['x'] == 5
    assert res['p_hat'] == pytest.approx(0.05)
    assert res['p_value'] == pytest.approx(1.0)
    assert res['LR'] == pytest.approx(0.0, abs=1e-12)


def test_christoffersen_transition_counts_and_finite_stats():
    breaches = pd.Series([False, False, True, False, True, True])
    res = christoffersen_independence_test(breaches)
    assert res['N00'] == 1
    assert res['N01'] == 2
    assert res['N10'] == 1
    assert res['N11'] == 1
    assert np.isfinite(res['LR'])
    assert np.isfinite(res['p_value'])


def test_expected_shortfall_manual_example():
    returns = pd.Series([-0.02, 0.01, -0.05, -0.10])
    es = expected_shortfall(returns, alpha=0.95)
    assert es == pytest.approx(0.10)

