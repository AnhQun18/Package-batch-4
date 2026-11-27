# coding=utf-8
"""Plot distribution of a numeric column."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_numeric_distribution(df: pd.DataFrame, column: str, bins=30, percentiles=(5, 95), **params):
    """
        A function plot histogram with mean, median and percentile lines.
                Parameters:
                            df: DataFrame
                                Input data
                            column: str
                                Numeric column
                            bins: int
                                Number of bins. Default 30
                            percentiles: Tuple
                                Percentiles (0-100) drawn as lines. Default (5, 95). Empty tuple = no lines
                            **params: The parameters of figure.
                                figsize: Tuple of figure size. Default (12,6)
                Returns:
                            figure: Figure
                            stats: Series (count, mean, median, percentiles)
    """
    values = df[column].replace([np.inf, -np.inf], np.nan).dropna()
    stats = pd.Series({'count': len(values), 'mean': values.mean(), 'median': values.median()})
    for p in percentiles:
        stats[f'p{p}'] = values.quantile(p / 100)

    figure, ax = plt.subplots(figsize=params.get('figsize', (12, 6)))
    ax.hist(values, bins=bins)
    ax.axvline(stats['mean'], color='r', linestyle='--', label=f"mean = {stats['mean']:.2f}")
    ax.axvline(stats['median'], color='g', linestyle='--', label=f"median = {stats['median']:.2f}")
    for p in percentiles:
        ax.axvline(stats[f'p{p}'], color='gray', linestyle=':', label=f"p{p} = {stats[f'p{p}']:.2f}")
    ax.set_xlabel(column)
    ax.set_ylabel('Count')
    ax.set_title(f'Distribution of {column}')
    ax.legend()

    return figure, stats
