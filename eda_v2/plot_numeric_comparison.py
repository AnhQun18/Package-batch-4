# coding=utf-8
"""Compare distribution of a numeric column between groups."""
import matplotlib.pyplot as plt
import pandas as pd


def plot_numeric_comparison(df: pd.DataFrame, numeric_column: str, group_column: str, top_k=10, hide_outliers=False, **params):
    """
        A function plot boxplot of a numeric column for each group.
                Parameters:
                            df: DataFrame
                                Input data
                            numeric_column: str
                                Numeric column to compare
                            group_column: str
                                Column that defines groups
                            top_k: int
                                Only plot K largest groups (by rows). If top_k = None, plot all groups
                            hide_outliers: Bool
                                If True, outlier points (beyond 1.5 * IQR) are not drawn. Default False (show).
                                Only affects the chart; the returned table still uses all data
                            **params: The parameters of figure.
                                figsize: Tuple of figure size. Default (12,6)
                Returns:
                            figure: Figure
                            table: DataFrame (describe of numeric column by group)
    """
    data = df[[group_column, numeric_column]].dropna()
    order = data[group_column].value_counts().index
    if top_k is not None:
        order = order[:top_k]
    data = data[data[group_column].isin(order)]
    values = [data.loc[data[group_column] == g, numeric_column] for g in order]
    table = data.groupby(group_column)[numeric_column].describe().loc[order]

    figure, ax = plt.subplots(figsize=params.get('figsize', (12, 6)))
    ax.boxplot(values, showmeans=True, showfliers=not hide_outliers)
    ax.set_xticklabels([str(g) for g in order])  # tự gán nhãn: tên tham số labels/tick_labels đổi theo bản matplotlib
    ax.set_xlabel(group_column)
    ax.set_ylabel(numeric_column)
    ax.set_title(f'{numeric_column} by {group_column}')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    return figure, table
