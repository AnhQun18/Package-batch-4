# coding=utf-8
"""Plot entity activity over time."""
import matplotlib.pyplot as plt
import pandas as pd


def plot_entity_coverage(df: pd.DataFrame, entity_column: str, time_column: str, freq='M', top_k=30, **params):
    """
        A function plot heatmap of entity activity over time (rows = entity, columns = period).
                Parameters:
                            df: DataFrame
                                Input data
                            entity_column: str
                                Entity column (e.g. customer)
                            time_column: str
                                Datetime column
                            freq: str
                                Pandas period frequency, e.g. 'D', 'W', 'M'. Default 'M'
                            top_k: int
                                Only show K most active entities. If top_k = None, show all
                            **params: The parameters of figure.
                                figsize: Tuple of figure size. Default (14,8)
                Returns:
                            figure: Figure
                            table: DataFrame (number of records of entity by period)
    """
    data = df[[entity_column, time_column]].dropna()
    period = pd.to_datetime(data[time_column]).dt.to_period(freq)
    table = pd.crosstab(data[entity_column], period)
    table = table.loc[table.sum(axis=1).sort_values(ascending=False).index]
    if top_k is not None:
        table = table.iloc[:top_k]

    figure, ax = plt.subplots(figsize=params.get('figsize', (14, 8)))
    image = ax.imshow(table.values, aspect='auto', cmap='Blues')
    ax.set_xticks(range(table.shape[1]))
    ax.set_xticklabels([str(p) for p in table.columns], rotation=90, ha='right')
    ax.set_yticks(range(table.shape[0]))
    ax.set_yticklabels(table.index)
    ax.set_xlabel(f'Period ({freq})')
    ax.set_ylabel(entity_column)
    ax.set_title(f'Activity of {entity_column} over time')
    figure.colorbar(image, ax=ax, label='Records')

    return figure, table
