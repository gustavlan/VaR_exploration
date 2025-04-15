import matplotlib.pyplot as plt

def plot_var_breaches(data, ax, title, breach_col='breach', return_col='ret_10d', var_col='var_10d'):
    """
    Plots 10-day returns, VaR breaches, and VaR line on a given Axes object.
    
    Args:
        data (pd.DataFrame): Data with columns for returns, var, and breach flags.
        ax (matplotlib.axes.Axes): Axes object on which to plot.
        title (str): Plot title.
        breach_col (str): Column name for the boolean breach indicator.
        return_col (str): Column name for returns.
        var_col (str): Column name for VaR values.
    """
    # Plot the 10-day returns
    ax.plot(data.index, data[return_col], label='10-day returns', color='blue')
    
    # Mark breaches
    breaches = data[data[breach_col]]
    ax.scatter(breaches.index, breaches[return_col], color='red', marker='x', label='VaR Breaches')
    
    # Plot the 10-day VaR
    ax.plot(data.index, data[var_col], label='10-day VaR', color='green')
    
    # Titles and labels
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Returns')
    ax.legend()

def plot_multiple_var_breaches(
    data_list,
    titles,
    breach_col='breach',
    return_col='ret_10d',
    var_col='var_10d',
    figsize=(20, 5)
):
    """
    Plots multiple DataFrames in stacked subplots for side-by-side VaR breach comparison.

    Args:
        data_list (List[pd.DataFrame]): List of DataFrames to plot.
        titles (List[str]): Plot titles for each DataFrame.
        breach_col (str): Column name with breach flags.
        return_col (str): Column name with returns.
        var_col (str): Column name with VaR values.
        figsize (tuple): Figure size (width, height).
    """
    n = len(data_list)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(figsize[0], figsize[1]*n))

    # If only 1 data in data_list, axes won't be a list
    if n == 1:
        plot_var_breaches(
            data_list[0],
            ax=axes,
            title=titles[0],
            breach_col=breach_col,
            return_col=return_col,
            var_col=var_col
        )
    else:
        for i, (df, title) in enumerate(zip(data_list, titles)):
            plot_var_breaches(
                df,
                ax=axes[i],
                title=title,
                breach_col=breach_col,
                return_col=return_col,
                var_col=var_col
            )

    plt.tight_layout()
    plt.show()
