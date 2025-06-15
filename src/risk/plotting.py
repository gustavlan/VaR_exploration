import logging

# Silence that categorical‐units INFO
logging.getLogger("matplotlib.category").setLevel(logging.WARNING)

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import cast
import pandas as pd
from typing import List


def plot_var_breaches(
    data: pd.DataFrame,
    ax: plt.Axes,
    title: str,
    breach_col: str = "breach",
    return_col: str = "ret_10d",
    var_col: str = "var_10d",
) -> Figure:
    """
    Plot returns, VaR threshold, and breach points on a single axes.
    """
    # Force index into real Python datetimes
    x = pd.to_datetime(data.index)

    # Plot the 10-day returns
    ax.plot(x, data[return_col], label="10-day returns")

    # Mark breaches
    breaches = data[data[breach_col]]
    bx = pd.to_datetime(breaches.index)
    ax.scatter(bx, breaches[return_col], marker="x", label="VaR Breaches")

    # Plot the 10-day VaR
    ax.plot(x, data[var_col], label="10-day VaR")

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Returns")
    ax.legend()

    fig = ax.get_figure()
    assert fig is not None
    return cast(Figure, fig)


def plot_multiple_var_breaches(
    data_list: List[pd.DataFrame],
    titles: List[str],
    breach_col: str = "breach",
    return_col: str = "ret_10d",
    var_col: str = "var_10d",
    figsize: tuple = (20, 5),
) -> plt.Figure:
    """
    Stack multiple VaR breach plots vertically for comparison.
    """
    n = len(data_list)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=figsize)

    if n == 1:
        plot_var_breaches(
            data_list[0], axes, titles[0], breach_col, return_col, var_col
        )
    else:
        for df, title, ax in zip(data_list, titles, axes):
            plot_var_breaches(df, ax, title, breach_col, return_col, var_col)

    plt.tight_layout()
    plt.show()
    return fig
