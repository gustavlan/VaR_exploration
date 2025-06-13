import pytest
import pandas as pd
import numpy as np

from risk.utils import (
    annualize_return,
    annualize_volatility,
    calculate_daily_returns,
    calculate_log_returns,
    calculate_forward_log_returns,
    calculate_rolling_volatility,
    calculate_parametric_var,
    sharpe_ratio,
    detect_var_breaches,
    summarize_var_breaches,
)


def test_annualize_return_and_vol():
    # daily return .01 over 252 days → ~2.52 annual
    assert annualize_return(0.01, trading_days=252) == pytest.approx(2.52)
    # daily vol .02 → annual vol .02*sqrt(252)
    expected = 0.02 * np.sqrt(252)
    assert annualize_volatility(0.02, trading_days=252) == pytest.approx(expected)


def test_calculate_daily_returns():
    prices = pd.Series([100, 110, 121], index=pd.date_range("2020-01-01", periods=3))
    # returns = [10%, 10%]
    ret = calculate_daily_returns(prices)
    assert np.allclose(ret.values, [0.1, 0.1])

    # empty series → ValueError
    with pytest.raises(ValueError):
        calculate_daily_returns(pd.Series(dtype=float))


def test_calculate_log_and_forward_returns():
    prices = pd.Series([100, 110, 121], index=pd.date_range("2020-01-01", periods=3))
    log_ret = calculate_log_returns(prices)
    assert np.allclose(log_ret.values, np.log([110 / 100, 121 / 110]))

    fwd = calculate_forward_log_returns(prices, days_forward=2)
    # only one value: ln( price[t+2]/price[t] ) = ln(121/100)
    assert len(fwd) == 1
    assert fwd.iloc[0] == pytest.approx(np.log(121 / 100))


def test_rolling_vol_and_parametric_var():
    # simple returns series
    rets = pd.Series([1.0, 2.0, 3.0, 4.0])
    # window=2 volatility: sample std([1,2]) = sqrt(0.5)
    vol = calculate_rolling_volatility(rets, window=2)
    expected_std = np.sqrt(0.5)
    # first value is NaN, second is ≈0.7071
    assert vol.iloc[1] == pytest.approx(expected_std)

    # parametric var: with volatility series [1,2,3], z=-1, horizon=1 → [-1,-2,-3]
    vol_series = pd.Series([1.0, 2.0, 3.0])
    var = calculate_parametric_var(vol_series, confidence_z=-1.0, horizon_days=1)
    assert np.allclose(var.values, [-1.0, -2.0, -3.0])


def test_sharpe_ratio_and_edge():
    # constant zero returns → Sharpe undefined
    zero = pd.Series([0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        sharpe_ratio(zero, risk_free_rate=0.0, trading_days=252)

    # simple positive returns with zero RF
    rets = pd.Series([0.01, 0.02, 0.015])
    sr = sharpe_ratio(rets, risk_free_rate=0.0, trading_days=252)
    # check it's a finite number
    assert isinstance(sr, float)
    assert not np.isnan(sr)


def test_detect_and_summarize_breaches():
    # make a tiny DataFrame
    df = pd.DataFrame(
        {
            "ret": [0.05, -0.10, -0.02],
            "var": [-0.01, -0.05, -0.05],
        }
    )
    df2 = detect_var_breaches(
        df.copy(), return_col="ret", var_col="var", breach_col="breach"
    )
    # breaches where ret < var and ret < 0 → only middle row
    assert df2["breach"].tolist() == [False, True, False]

    summary = summarize_var_breaches(df2, breach_col="breach")
    assert summary["count"] == 1
    assert summary["percentage"] == pytest.approx(1 / 3, rel=1e-3)
