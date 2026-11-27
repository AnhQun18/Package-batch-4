# coding=utf-8
"""Plot distribution of a categorical column."""
import matplotlib.pyplot as plt
import pandas as pd


def plot_category_distribution(df: pd.DataFrame, column: str, top_k=10, other_label='Other', **params):
    """
        A function plot bar of category counts (top K + Other), with count and percentage.
                Parameters:
                            df: DataFrame
                                Input data
                            column: str
                                Categorical column
                            top_k: int
                                Keep K most frequent categories, the rest are merged into one group.
                                If top_k = None, keep all categories
                            other_label: str
                                Name of the merged group. Default 'Other'
                            **params: The parameters of figure.
                                figsize: Tuple of figure size. Default (12,6)
                Returns:
                            figure: Figure
                            table: DataFrame (category, count, percentage)
    """
    counts = df[column].value_counts()
    if top_k is not None and len(counts) > top_k:
        other = counts.iloc[top_k:].sum()
        counts = pd.concat([counts.iloc[:top_k], pd.Series({other_label: other})])
    table = pd.DataFrame({'category': counts.index.astype(str), 'count': counts.values})
    table['percentage'] = table['count'] / table['count'].sum() * 100

    figure, ax = plt.subplots(figsize=params.get('figsize', (12, 6)))
    bars = ax.barh(table['category'], table['count'])
    ax.invert_yaxis()  # category nhiều nhất ở trên cùng
    ax.bar_label(bars, labels=[f'{c:,} ({p:.1f}%)' for c, p in zip(table['count'], table['percentage'])], padding=3)
    ax.set_xlabel('Count')
    ax.set_title(f'Distribution of {column}')
    ax.margins(x=0.15)

    return figure, table
