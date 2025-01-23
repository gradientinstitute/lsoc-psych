import pandas as pd
import autograd.numpy as np
import autograd
from scipy import optimize
import pickle
from typing import Callable, List, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


# Data Loading
def load_traces(msize, data_path):
    """Load the pickle files in George's format."""
    with open(f'{data_path}/pile_{msize}_subset_loss_df.pkl', 'rb') as file:
       df_loss = pickle.load(file)

    with open(f'{data_path}/pile_{msize}_subset_llc_df.pkl', 'rb') as file:
      df_llc = pickle.load(file)

    step = "step" if "step" in df_loss.columns else "Step"
    df_loss.set_index(step, inplace=True)
    df_llc.set_index(step, inplace=True)
    df_llc = df_llc / 100.  # adjust the scale

    return df_llc, df_loss


def trim_trace(df_llc, df_loss, task, start, finish):
    """
    Extract and trim (by step) a task from the data.
    Returns 1D arrays for LLC, Loss and Step.
    """
    s = df_llc[task].index.values
    x = df_llc[task].values
    y = df_loss[task].values
    keep = (s >= start) & (s<=finish)
    x = x[keep]
    y = y[keep]
    s = s[keep]
    return x, y, s


class ModelSpec(ABC):
    """Abstract base class to specify a scaling law model."""

    par0: List[float]  # Specify initial parameters
    par_names: List[str]  # Names of these parameters

    @staticmethod
    @abstractmethod
    def function(x: np.ndarray, params: List[float]) -> np.ndarray:
        """Encode the parametric form of the relationship."""
        pass

    @classmethod
    def bounds(cls, x: np.ndarray, y: np.ndarray) -> List[tuple[float, float]]:
        """Inspect the data to set dynamic optimization bounds."""
        return [(None, None)] * len(cls.par0)


class ShiftedPowerLaw(ModelSpec):
    """
    Single Offset Power Law model.
    y = L0 + c x^r
    """

    @staticmethod
    def function(x, params):
        L0, logc, r = params
        return L0 + np.exp(logc - r * np.log(x))

    par0 = [0., 0.1, 1.]
    par_names = ["L*", "logc", "r"]

    @staticmethod
    def bounds(x, y):
        return [
            (0., float(y.min())),
            (-30., 30.),
            (-30., 30.)
        ]


class PolynomialModel(ModelSpec):
    """
    Polynomial model
    y = a0 + a1*x + a2*x^2 + a3*x^3
    """

    @staticmethod
    def function(x, params):
        a0, a1, a2, a3 = params
        return a0 + a1*x + a2*x**2 + a3*x**3

    # Inherited bounds (unbounded)
    par0 = [0.1, 0.1, 0.1, 0.1]
    par_names = ["a0", "a1", "a2", "a3"]



class ShiftedExponential(ModelSpec):
    """
    Shifted exponential model.
    y = y0 + exp(-r (x - x0))
    """

    @staticmethod
    def function(x, params):
        x0, y0, r = params
        return y0 + np.exp(-r * (x - x0))

    par0 = [1., 1., 4.]
    par_names = "LLC*", "L*", "r"

    @staticmethod
    def bounds(x, y):
        return [
            (float(x.min()), float(x.max())),
            (0., float(y.min())),
            (0, 20.),
        ]


class DoubleShiftedPowerLaw(ModelSpec):
    """
    Double offset power law model.
    y = y0 + c (x - x0)^r
    """

    @staticmethod
    def function(x, params):
        x0, y0, logc, r = params
        return y0 + np.exp(logc - r * np.log(x - x0))

    par0 = [0.1, .1, 0.1, 4.]
    par_names = ["x*", "y*", "logc", "r"]

    @staticmethod
    def bounds(x, y):
        eps = 1e-12
        return [
            (0., float(x.min() - eps)),
            (0., float(y.min())),
            (-20., 20.),
            (-20., 20.)
        ]

    @staticmethod
    def params_dict(par):
        x0, y0, logc, r = [float(p) for p in par]
        return {
            'L*': L0,
            'logc': logc,
            'r': r,
        }


# List all the candidate models
models = {
    "Power Law": ShiftedPowerLaw,
    "Power Law (4P)": DoubleShiftedPowerLaw,
    "Exponential": ShiftedExponential,
    "Polynomial": PolynomialModel,
}


@dataclass
class ModelFit:
    """Collection to specify a fit result."""
    f: Callable  # the function
    params_dict: dict  # annotated parameters
    params: list  # parameter list
    pcov: Optional[np.ndarray]  # solution covariance (if applicable)
    raw: Optional  # raw minimize result (if applicable)


def _params_dict(model, par):
    return dict(zip(model.par_names, par))


def curve_fit(x, y, model):

    bounds = list(zip(*model.bounds(x, y)))
    def _model(x, *args):
        return model.function(x, args)

    popt, pcov = optimize.curve_fit(_model, x, y, p0=model.par0, bounds=bounds)

    def f(x):
        return model.function(x, popt)

    return ModelFit(f, _params_dict(model, popt), popt, pcov, None)


def min_fit(x, y, model, method="L-BFGS-B", hs=False):

    # Optimise =========================================
    def opt_loss(params):
        y_pred = model.function(x, params)
        if hs:
            return np.sum((y_pred / y - 1.)**2)
            # or y/y_pred if we believe y_pred is the truth
        else:
            return np.sum((y - y_pred) ** 2)  # Sum of squared errors

    bounds = model.bounds(x, y)
    res = optimize.minimize(
        opt_loss, model.par0, method=method,
        jac=autograd.grad(opt_loss), bounds=bounds)
    popt = res.x

    # estimate pcov (trying to replicate curve_fit)
    # We can use analytic hessian rather than the optimiser
    hess = autograd.hessian(opt_loss)(popt)
    inv_hess = np.linalg.inv(hess)
    # inv_hess = np.asarray(res.hess_inv.todense())
    p = len(model.par0)
    n = len(x)
    ssq = res.fun / (len(x) - p)
    pcov = ssq * inv_hess

    def f(x):
        return model.function(x, popt)

    return ModelFit(f, _params_dict(model, popt), popt, pcov, res)


def odr_fit(x, y, model, method="L-BFGS-B", hs=False):
    """
    Experimental orthogonal distance regression.
    Jointly estimates x and parameters.

    Limitations:
    - scaling wrt data size
    - currently no learned anisotropy (assuming equal noise on x and y)
    """

    n_par = len(model.par0)

    def opt_loss(paramsx):
        """Squared error in X and Y"""
        params = paramsx[:n_par]
        x_pred = paramsx[n_par:]
        y_pred = model.function(x_pred, params)

        if hs:
            # y is a stable approximation for y_pred
            yerr = np.sum((y_pred / y - 1.)**2)
            xerr = np.sum((x_pred / x - 1.)**2)
        else:
            yerr = np.sum((y - y_pred) ** 2)
            xerr = np.sum((x - x_pred) ** 2)
        return yerr + xerr

    bounds = model.bounds(x, y)

    # Augment model parameters with best guess for x*
    par0 = np.concatenate((model.par0, x))  # x is a good guess for x*
    bounds.extend([(None, None)] * len(x))

    res = optimize.minimize(
        opt_loss, par0, method=method,
        jac=autograd.grad(opt_loss), bounds=bounds)
    popt = res.x[:n_par]

    # Report ===============================
    def f(x):
        return model.function(x, popt)

    return ModelFit(f, _params_dict(model, popt), popt, None, res)


fit_methods = {
    "Curvefit": curve_fit,
    "Minimimize": min_fit,
    "Orthogonal": odr_fit,
}


def logspace_r2(y_true, y_pred):
    """
    Metric - R2 in logspace.
    Note - this is not in the *shifted space* which requires model parameters.
    """
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    return 1. - (ss_res / ss_tot)


def pcov_diagnostics(result, model):
    """Compute fit diagnostics for sum-of-squared-error objectives."""

    if result.pcov is not None:
        stderrs = np.diag(result.pcov) ** .5
        named = _params_dict(model, stderrs)
        report = {k + "_stderr": v for k, v in named.items()}

        # Some additional diagnostics
        eigvals = np.linalg.eigvals(result.pcov)
        abs_eigvals = np.abs(eigvals)
        min_eigval, max_eigval = abs_eigvals.min(), abs_eigvals.max()
        report["min_eigenval"] = min_eigval
        report["max_eigenval"] = max_eigval
        report["condition_number"] = max_eigval / min_eigval
        return report
    else:
        return {}



def dict2txt(params_dict, join="<br>"):
    """Convert an annotated parameters dictionary into a string."""
    return "<br>".join(f"{k}: {v:.2f}" for k, v in params_dict.items())


def assign_cols(names, cmap="turbo", seed=42):
    """Annotate a set of names with a set of colours."""
    import plotly.express as px
    from random import Random
    cols = px.colors.sample_colorscale(cmap, len(names))
    local_rng = Random(seed)
    local_rng.shuffle(cols)
    return dict(zip(names, cols))


def autotrim_trace(df_llc, df_loss, task):
    """"
    Extract and trim (by heuristic) a task from the data.
    The heuristic extracts a powerlaw-like region of a loss/LLC trajectory.
    Returns 1D arrays for LLC, Loss and Step.

    Heuristic:
    * Finds the longest monotonically increasing interval in LLC,
    * Crops the start to begin from the steepest slope.
    * Note - may not play well with noisy data.
    """
    # Interval end heuristic - minimum loss
    step = df_llc.index.values  # step
    x_full = df_llc[task].values  # llc
    y_full = df_loss[task].values  # loss
    task_finish = step[y_full.argmin()]

    # Get monotonically increasing LLC region
    st, fn = _longest_monotonic(x_full)
    x = x_full[st:fn]
    y= y_full[st:fn]
    s = step[st:fn]

    dx = np.diff(x)
    dy = np.diff(y)
    data_grad = dy / np.maximum(1e-8, dx)
    ix = np.argmin(data_grad)  # find steepest descent
    x = x[ix:]
    y= y[ix:]
    s = s[ix:]
    return x, y, s


def _longest_monotonic(arr):
    """
    Find the longest monotonically increasing interval in a vector.
    Returns start and end indices.
    """
    if len(arr) == 0:
        return None, None

    # Convert to numpy array if not already
    arr = np.array(arr)

    # Keep track of current sequence
    current_start = 0
    current_length = 1

    # Keep track of longest sequence
    max_start = 0
    max_length = 1

    for i in range(1, len(arr)):
        if arr[i] > arr[i-1]:
            # Continue current sequence
            current_length += 1
        else:
            # Start new sequence
            current_start = i
            current_length = 1

        # Update longest sequence if current is longer
        if current_length > max_length:
            max_start = current_start
            max_length = current_length

    # Return start and end indices (inclusive)
    return max_start, max_start + max_length
