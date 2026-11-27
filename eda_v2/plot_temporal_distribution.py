# coding=utf-8
"""Plot a metric aggregated over time."""
import matplotlib.pyplot as plt
import pandas as pd


def plot_temporal_distribution(df: pd.DataFrame, time_column: str, value_column=None, freq='D', agg='count', **params):
    """
        A function plot line of a metric aggregated (resampled) by time.
                Parameters:
                            df: DataFrame
                                Input data
                            time_column: str
                                Datetime column
                            value_column: str
                                Column to aggregate. If value_column = None, count rows
                            freq: str
                                Pandas resample frequency, e.g. 'D', 'W', 'ME' (month end, pandas >= 2.2). Default 'D'
                            agg: str
                                Aggregation: 'count', 'sum', 'mean', 'median', ... Default 'count'
                            **params: The parameters of figure.
                                figsize: Tuple of figure size. Default (14,6)
                Returns:
                            figure: Figure
                            series: Series (aggregated value by period)
    """
    data = df.copy()
    data[time_column] = pd.to_datetime(data[time_column])
    data = data.dropna(subset=[time_column]).set_index(time_column)
    if value_column is None:
        series = data.resample(freq).size()
        agg = 'count'
    else:
        series = data[value_column].resample(freq).agg(agg)

    figure, ax = plt.subplots(figsize=params.get('figsize', (14, 6)))
    ax.plot(series.index, series.values, marker='o')
    ax.set_xlabel(time_column)
    ax.set_ylabel(f'{agg} of {value_column}' if value_column else 'Count')
    ax.set_title(f'{agg} by {freq}')
    ax.grid(True, alpha=0.3)

    return figure, series
