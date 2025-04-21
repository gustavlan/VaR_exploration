import pytest
import numpy as np
import pandas as pd

from risk.var import historical_var, parametric_var, monte_carlo_var


def test_historical_var_simple():
    # returns: [-1,0,1,2,3], confidence_level=0.8 → percentile(20)= -0.2 → abs=0.2
    rets = pd.Series([-1, 0, 1, 2, 3])
    hv = historical_var(rets, confidence_level=0.8)
    assert hv == pytest.approx(0.2, rel=1e-3)


def test_parametric_and_monte_carlo_on_constant():
    const = pd.Series([0.0, 0.0, 0.0])
    # both should give var=0
    assert parametric_var(const, confidence_level=0.95) == 0.0
    # Monte Carlo with zero std → all sims zero → var zero
    mc = monte_carlo_var(const, confidence_level=0.95, simulations=1000)
    assert mc == pytest.approx(0.0)


def test_empty_series_raises():
    empty = pd.Series(dtype=float)
    with pytest.raises(ValueError):
        historical_var(empty)
    with pytest.raises(ValueError):
        parametric_var(empty)
    with pytest.raises(ValueError):
        monte_carlo_var(empty)
