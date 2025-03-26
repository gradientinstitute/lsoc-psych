import plotly.graph_objects as go  # temporarily don't drop
import plotly.express as px
import numpy as np
from .data import Trace
from .fit import FitResult


llc_desc = r"$\text{Estimated and transformed LLC }\,\frac{1}{100}\hat{\lambda}$"
loss_desc = r"$\text{Loss }L$"


def dict2txt(params_dict, on="<br>"):
    """Convert an annotated parameters dictionary into a string."""
    return on.join(f"{k}: {v:.2f}" for k, v in params_dict.items())


def assign_cols(names, cmap="turbo", seed=42, shuffle=False):
    """Annotate a set of names with a set of colours."""
    import plotly.express as px
    from random import Random
    cols = px.colors.sample_colorscale(cmap, len(names))
    if shuffle:
        local_rng = Random(seed)
        local_rng.shuffle(cols)
    return dict(zip(names, cols))


def fade(color):
    # pull towards gray
    return add_color(color, 0.85, 211)


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


def plot_data(fig, xs, ys=None, ss=None, shift=None,
              color="black", name="",
              subplot={}, showlegend=None, size=5, xscale=1.,
              mode='markers', **kwargs):
    """Plot the train / test datapoints."""

    if showlegend is None:
        showlegend = name!=""

    if isinstance(xs, Trace):
        xs, ys, ss = xs  # unpack

    if shift:
        fig.update_xaxes(type="log", **subplot)  # Set x-axis to log scale for first subplot
        fig.update_yaxes(type="log", **subplot)  # Set y-axis to log scale for first subplot

        if "x*" in shift:
            xs = xs - shift["x*"]
        if "y*" in shift:
            ys = ys - shift["y*"]

    fig.add_trace(
        go.Scatter(
            x=xs * xscale, y=ys, customdata=ss,
            mode=mode,
            showlegend=showlegend,
            name=name,
            marker=dict(
                color=color,
                size=size,
                line=dict(                   # This creates and styles the marker outline
                    color='rgba(255,255,255,.5)',  #'white',        # Outline color
                    width=.25                  # Outline width
                ),
            ),
            hovertemplate=f"Name:{name}<br>" + "Step: %{customdata}<br><extra></extra>",
            **kwargs,
        ),
        **subplot,
    )
    return fig


def add_color(rgb_str, alpha=.6, target=255):
    """
    Takes a color string of form 'rgb(48, 18, 59)' and returns a lighter version
    amount: how much to lighten (0-255)
    """
    # Extract the numbers using string operations
    nums = rgb_str.strip('rgb()').split(',')
    rgb = np.array(nums).astype(float)
    rgb = rgb*(1. - alpha) + target * alpha
    rgb = np.maximum(0, np.minimum(255, rgb)).astype(np.uint8)

    return 'rgb({}, {}, {})'.format(*rgb)


def x_plot(x, res=100):
    if len(x) < res:
        # We probably want the curve at a higher resolution than x
        alpha = np.linspace(0, len(x)-1, res)
        x_pred = np.interp(alpha, np.arange(len(x)), np.sort(x))
    else:
        x_pred = np.sort(x)
    return x_pred


def plot_result(fig, x, result, res=100, shift=None, color=None, subplot=None,
                xscale=1., outline="white", **kwargs):

    x_pred = x_plot(x, res=res)

    if isinstance(result, FitResult):
        y_pred = result.f(x_pred)
        params_dict = result.params_dict
    else:
        # assume function
        y_pred = result(x_pred)
        params_dict = {}

    if shift and "x*" in shift:
        x_pred = x_pred - shift["x*"]
    if shift and "y*" in shift:
        y_pred = y_pred - shift["y*"]

    lineval = dict(width=1.5)

    if color is not None:
        lineval["color"] = color

    if subplot is None:
        subplot = {}

    fig.add_trace(
        go.Scatter(
            x=x_pred * xscale, y=y_pred,
            mode='lines',
            line=lineval,
            hovertemplate=(
                dict2txt(params_dict)
            ),
            **kwargs,
        ),
        **subplot
    )
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


# Plotly has this bug where the first figure doesn't show --> prime the system
fig = go.Figure()
fig.update_layout(width=10, height=10)  # non-display size
fig.show()
