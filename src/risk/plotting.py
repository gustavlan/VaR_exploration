from typing import List

import matplotlib.pyplot as plt
import pandas as pd


def plot_var_breaches(
    data: pd.DataFrame,
    ax: plt.Axes,
    title: str,
    breach_col: str = 'breach',
    return_col: str = 'ret_10d',
    var_col: str = 'var_10d'
) -> None:
    """
    Plot returns, VaR threshold, and breach points on a single axes.

    Parameters
    ----------
    data
        DataFrame containing return_col, var_col, and breach_col.
    ax
        Matplotlib Axes to draw on.
    title
        Title of the subplot.
    breach_col
        Column name for breach flags.
    return_col
        Column name for returns.
    var_col
        Column name for VaR values.

    Returns
    -------
    None
    """
    ax.plot(data.index, data[return_col], label='10-day returns')
    breaches = data[data[breach_col]]
    ax.scatter(breaches.index, breaches[return_col], color='red', marker='x', label='VaR Breaches')
    ax.plot(data.index, data[var_col], label='10-day VaR')
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Returns')
    ax.legend()


def plot_multiple_var_breaches(
    data_list: List[pd.DataFrame],
    titles: List[str],
    breach_col: str = 'breach',
    return_col: str = 'ret_10d',
    var_col: str = 'var_10d',
    figsize: tuple = (20, 5)
) -> None:
    """
    Stack multiple VaR breach plots vertically for comparison.

    Parameters
    ----------
    data_list
        List of DataFrames to plot.
    titles
        Corresponding list of subplot titles.
    breach_col
        Column name for breach flags.
    return_col
        Column name for returns.
    var_col
        Column name for VaR values.
    figsize
        Figure size (width, height).

    Returns
    -------
    None
    """
    n = len(data_list)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=figsize)

    # If only one DataFrame, axes is not iterable
    if n == 1:
        plot_var_breaches(data_list[0], axes, titles[0], breach_col, return_col, var_col)
    else:
        for df, title, ax in zip(data_list, titles, axes):
            plot_var_breaches(df, ax, title, breach_col, return_col, var_col)

    plt.tight_layout()
    plt.show()
