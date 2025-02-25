import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import DEFAULT_PLOTLY_COLORS
import plotly.express as px
from plotly.colors import qualitative
from typing import Optional, Union, List, Tuple
import re
import numpy as np
import pandas as pd


def heatmap(table, width, height, upper=False, title="", decimals=2, **kwargs):
    """Plotly heatmap of a dataframe."""
    args = dict(
        zmin=None,
        zmax=None,
        colorscale='RdBu',
        reversescale=False,
    )
    args.update(kwargs)

    matrix = table.values
    if upper:
        mask = np.triu(np.ones_like(table))
        matrix = np.where(mask, table, np.nan)

    # Create the heatmap
    map = go.Heatmap(
        z=matrix,
        x=table.columns,
        y=table.index,
        text=np.round(table.values, decimals=decimals),
        texttemplate="%{text}",
        **args,
    )
    fig = go.Figure(map)
    # Update the layout
    fig.update_layout(
        title=title,
        yaxis_title=table.index.name,
        xaxis_title=table.columns.name,
        yaxis={"autorange":"reversed"},
        width=width*100,
        height=height*100,
    )
    return fig


def traces(X, categories, stds={}, cmap="turbo", col=1, cols=1, fig=None):
    """Plot traces from a 2d matrix, splitting on / in column names."""
    steps = X.index.values
    task_names = sorted(list(set(v.split("/")[1] for v in X.columns)))
    colours = px.colors.sample_colorscale(cmap, len(task_names))
    task_colors = dict(zip(task_names, colours))

    if fig is None:
        fig = make_subplots(
            rows=len(categories),
            cols=cols,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=([""] * cols)
        )

    for row, cat in enumerate(categories):
        for task in task_names:
            err_y = None
            if f"{cat}_std/{task}" in stds:
                err_y=dict(
                    type='data',
                    array=stds[f"{cat}_std/{task}"],
                    color=task_colors[task],
                    thickness=1,
                    width=3,
                    visible=True,
                )
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=X[f"{cat}/{task}"],
                    mode="lines+markers",
                    marker=dict(size=5, color=task_colors[task]),
                    error_y=err_y,
                    name=task,
                    showlegend=row==0,
                    line=dict(color=task_colors[task]),
                ),
                row=row+1, col=col,
            )
            fig.update_yaxes(title_text=cat, row=row+1, col=col)

    # Add to final subplot
    fig.update_xaxes(title_text="step", row=len(categories), col=col)
    fig.update_xaxes(type="log")  # most of the action happens early

    # fig.write_image("traces.pdf")
    fig.update_layout(
        width=800,
        height=700,
    )
    return fig


def tensor_traces(X, dimensions, split=-1, cmap="turbo"):
    """Plot traces of data in tensor format, labeling by dimensions."""
    if split < 0:
        split = len(dimensions) + split
    ndim = len(dimensions)
    dimensions = [*dimensions]  # I'll be doing inplace list operations
    categories = dimensions.pop(split)
    steps, task_names = dimensions
    task_colors = dict(zip(
        task_names,
        px.colors.sample_colorscale(cmap, len(task_names))
    ))

    fig = make_subplots(
        rows=len(categories),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        # subplot_titles=categories,
    )

    for row, cat in enumerate(categories):
        # extract a 2d array
        idx = [slice(None)] * ndim
        idx[split] = row
        Q = X[*idx]

        for j, task in enumerate(task_names):
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=Q[:,j],
                    mode="lines+markers",
                    marker=dict(size=5, color=task_colors[task]),
                    name=task,
                    showlegend=row==0,
                    line=dict(color=task_colors[task]),
                ),
                row=row+1, col=1,
            )
            fig.update_yaxes(title_text=cat, row=row+1, col=1)

    # Add to final subplot
    fig.update_xaxes(title_text="step", row=len(categories), col=1)
    fig.update_xaxes(type="log")  # most of the action happens early

    # fig.write_image("traces.pdf")
    fig.update_layout(
        width=800,
        height=700,
    )
    return fig


def crossval(heldout_err, heldout_std, fit_err, method_name):
    """Plot a cross validation result."""
    n_factors = np.arange(len(heldout_err)) + 1
    fig = go.Figure()

    # Add traces for train and test data
    fig.add_trace(
        go.Scatter(
            x=n_factors,
            y=fit_err,
            mode='lines+markers',
            name='Train Data',
            line=dict(color='blue'),
            marker=dict(color='blue')
        )
    )

    fig.add_trace(
        go.Scatter(
            x=n_factors,
            y=heldout_err,
            mode='lines+markers',
            name='Test Data',
            error_y=dict(
                type='data',
                array=2*heldout_std,  # 95%?
                visible=True,
                color='red'
            ),
            line=dict(color='red'),
            marker=dict(color='red')
        )
    )

    # Update layout
    fig.update_layout(
        title=f'{method_name} cross validation',
        yaxis_title='Mean Squared Error',
        showlegend=True,
        width=800,
        height=600,
    )
    return fig


# Plotly has this bug where the first figure doesn't show --> prime the system
fig = go.Figure()
fig.update_layout(width=10, height=10)  # non-display size
fig.show()
