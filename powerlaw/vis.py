import plotly.graph_objects as go
import plotly.express as px
import numpy as np


def dict2txt(params_dict, on="<br>"):
    """Convert an annotated parameters dictionary into a string."""
    return on.join(f"{k}: {v:.2f}" for k, v in params_dict.items())


def assign_cols(names, cmap="turbo", seed=42):
    """Annotate a set of names with a set of colours."""
    import plotly.express as px
    from random import Random
    cols = px.colors.sample_colorscale(cmap, len(names)+1)
    local_rng = Random(seed)
    local_rng.shuffle(cols)
    return dict(zip(names, cols))


def fill_between(fig, x, y1, y2, color='blue', alpha=0.2, name=None, showlegend=False):
    """
    Add a filled region between two curves to a plotly figure.

    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        The figure to add the fill to
    x : array-like
        x coordinates
    y1 : array-like
        Upper y coordinates
    y2 : array-like
        Lower y coordinates
    color : str
        Color name or RGB(A) string for the fill
    alpha : float
        Opacity of the fill (0 to 1)
    name : str, optional
        Name for the legend
    showlegend : bool
        Whether to show in legend

    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The figure with the filled region added
    """
    # Convert color to rgba if alpha < 1
    if alpha < 1:
        if color.startswith('rgb('):
            # Convert rgb(r,g,b) to rgba(r,g,b,a)
            color = color.replace('rgb', 'rgba').replace(')', f',{alpha})')
        elif color.startswith('#'):
            # Convert hex to rgba
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            color = f'rgba({r},{g},{b},{alpha})'
        else:
            # Assume it's a named color, use rgba
            color = f'rgba(0,0,255,{alpha})'  # Default to blue if color name not recognized

    fig.add_trace(go.Scatter(
        x=x,
        y=y1,
        mode='lines',
        line=dict(width=0),
        showlegend=False,
    ))

    fig.add_trace(go.Scatter(
        x=x,
        y=y2,
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor=color,
        name=name,
        showlegend=showlegend,
    ))


def plot_split(fig, train, test, shift=None):
    """Plot the train / test datapoints."""
    for fold, col, xs, ys, ss in (
            ("Training Data", "gray", *train),
            ("Test Data", "black", *test)):

        if shift and "x*" in shift:
            xs = xs - shift["x*"]
        if shift and "y*" in shift:
            ys = ys - shift["y*"]


        fig.add_trace(go.Scatter(
            x=xs, y=ys, customdata=ss,
            mode='markers',
            showlegend=True,
            name=fold,
            marker=dict(color=col, size=8),
            hovertemplate="Step: %{customdata:.0f}<br><extra></extra>",
        ))
    return fig


def x_plot(x, res=100):
    if len(x) < res:
        # We probably want the curve at a higher resolution than x
        alpha = np.linspace(0, len(x)-1, res)
        x_pred = np.interp(alpha, np.arange(len(x)), np.sort(x))
    else:
        x_pred = np.sort(x)
    return x_pred


def plot_result(fig, x, result, name="Model fit", res=100, showlegend=True,
                shift=None):
    x_pred = x_plot(x, res=res)
    y_pred = result.f(x_pred)

    if shift and "x*" in shift:
        x_pred = x_pred - shift["x*"]
    if shift and "y*" in shift:
        y_pred = y_pred - shift["y*"]

    fig.add_trace(go.Scatter(
        x=x_pred, y=y_pred,
        mode='lines',
        name=name,
        showlegend=showlegend,
        hovertemplate=(
            dict2txt(result.params_dict)
        )
    ))
    return fig


def sample_result(fig, x, result, col, name="Model fit", res=100, sigma=2.,
                  noise=True, showlegend=True):

    x_p = x_plot(x, res=res)
    _, y_mu, y_std = result.sample(x_p, 30, noise=noise)

    fill_between(fig, x_p, y_mu - sigma * y_std,
                 y_mu + sigma * y_std, color=col)

    fig.add_trace(go.Scatter(
        x=x_p, y=y_mu,
        mode='lines',
        line=dict(color=col),
        name=name,
        showlegend=showlegend,
        hovertemplate=(
            dict2txt(result.params_dict)
        )
    ))
    return fig

