import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from risk.plotting import plot_var_breaches, plot_multiple_var_breaches
from risk.utils import detect_var_breaches


def make_dummy_df():
    df = pd.DataFrame(
        {
            "ret_10d": [0.02, -0.05, 0.01],
            "var_10d": [-0.03, -0.03, -0.03],
        },
        index=pd.date_range("2020-01-01", periods=3),
    )
    return detect_var_breaches(
        df, return_col="ret_10d", var_col="var_10d", breach_col="breach"
    )


def test_plot_var_breaches_returns_figure():
    df = make_dummy_df()
    fig, ax = plt.subplots()
    result = plot_var_breaches(df, ax, title="Test")
    assert result is fig
    assert hasattr(result, "savefig")
    plt.close(fig)


def test_plot_multiple_var_breaches_returns_figure():
    df1 = make_dummy_df()
    df2 = make_dummy_df()
    fig = plot_multiple_var_breaches([df1, df2], titles=["A", "B"])
    assert hasattr(fig, "savefig")
    plt.close(fig)

    fig_single = plot_multiple_var_breaches([df1], titles=["Single"])
    assert hasattr(fig_single, "savefig")
    plt.close(fig_single)
