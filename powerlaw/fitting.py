import autograd.numpy as np
import autograd
from scipy import optimize
from typing import Callable, List, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
from inspect import signature


class ModelSpec(ABC):
    """Class to specify a scaling law model."""

    par0: List[float]  # Initial parameters
    par_names: List[str]  # Parameter names

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
        y0, c, r = params
        return y0 + c * (x**(-r))

    par0 = [0., 1., 1.]
    par_names = ["y*", "c", "r"]


class ShiftedPowerLaw2(ModelSpec):
    """
    Single Offset Power Law model.
    y = L0 + c x^r
    """
    # I'm finding it much more robust (for convex functions)
    @staticmethod
    def function(x, params):
        y0, logc, r = params
        return y0 + np.exp(logc - r * np.log(x))

    par0 = [0., 1., 1.]
    par_names = ["y*", "logc", "r"]


class XShiftedPowerLaw(ModelSpec):

    @staticmethod
    def function(x, params):
        x0, c, r = params
        return c * (x-x0)**r

    par0 = [0., 1., .25]
    par_names = ["x*", "c", "r"]

    @staticmethod
    def bounds(x, y):
        # Set an X bound to avoid complex numbers
        eps = 1e-10
        return [
            (None, float(x.min()-eps)),
            (None, None),
            (None, None)
        ]


class XShiftedPowerLaw2(ModelSpec):

    @staticmethod
    def function(x, params):
        x0, logc, r = params
        return np.exp(logc + np.log(x-x0)*r)

    par0 = [0., 1., .25]
    par_names = ["x*", "logc", "r"]

    @staticmethod
    def bounds(x, y):
        # Set an X bound to avoid complex numbers
        eps = 1e-10
        return [
            (None, float(x.min()-eps)),
            (None, None),
            (None, None)
        ]


class ShiftedLogarithm(ModelSpec):

    @staticmethod
    def function(x, params):
        y0, a = params
        return a * np.log(x) + y0

    par_names = ["y*", "a"]
    par0 = [0., 1.]


class Cubic(ModelSpec):

    @staticmethod
    def function(x, params):
        a0, a1, a2, a3 = params
        return a0 + a1*x + a2*x**2 + a3*x**3

    # Inherited bounds (unbounded)
    par0 = [0.1, 0.1, 0.1, 0.1]
    par_names = ["a0", "a1", "a2", "a3"]


class ShiftedExponential(ModelSpec):

    @staticmethod
    def function(x, params):
        y0, a, r = params
        return y0 + np.exp(a - r * x)

    par0 = [1., 1., 1.]
    par_names = "y*", "a", "r"


class DoubleExponential(ModelSpec):
    """
    Double exponential model.
    y = y0 + a * np.exp(-r * np.exp(x/b))
    """

    @staticmethod
    def function(x, params):
        y0, loga, r, b = params
        return y0 + np.exp(-r * np.exp(x/b) + loga)

    par0 = [1., 1., 4., 4.]
    par_names = ["y*", "loga", "r", "b"]

    @staticmethod
    def bounds(x, y):
        return [
            (0., None),
            (-50, 50),
            (0., 50.),
            (.1, 50.),
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

    par0 = [0.1, .1, 0.1, 1.]
    par_names = ["x*", "y*", "logc", "r"]

    @staticmethod
    def bounds(x, y):
        eps = 1e-8
        return [
            (None, None),
            (None, float(x.min() - eps)),
            (-30., 30.),
            (-30., 30.)
        ]


@dataclass
class FitResult:
    """Collection to specify a fit result."""
    # This is still evolving a bit...
    f: Callable  # the function
    model: ModelSpec  # The functional form learned
    params: list  # parameter list
    pcov: Optional[np.ndarray]  # solution covariance (if applicable)
    nvar: Optional[float]  # learned noise covariance
    raw: Optional  # raw minimize result (if applicable)

    @property
    def params_dict(self):
        return dict(zip(self.model.par_names, self.params))

    def pcov_diagnostics(self):
        """Compute fit diagnostics for sum-of-squared-error objectives."""

        if self.pcov is not None:
            stderrs = np.diag(self.pcov) ** .5
            named = dict(zip(self.model.par_names, stderrs))
            report = {k + "_stderr": v for k, v in named.items()}

            # Some additional diagnostics
            eigvals = np.linalg.eigvals(self.pcov)
            abs_eigvals = np.abs(eigvals)
            min_eigval, max_eigval = abs_eigvals.min(), abs_eigvals.max()
            report["min_eigenval"] = min_eigval
            report["max_eigenval"] = max_eigval
            report["condition_number"] = max_eigval / min_eigval
            return report
        else:
            return {}

    def sample(self, x, n_samp, noise=True):
        """Approximately estimate output uncertainty by sampling parameters."""

        # Don't place too much weight on this
        # Firstly its an approximation
        # secondly - maybe needs to be using chi-squared
        pars = np.random.multivariate_normal(
            mean=self.params, cov=self.pcov, size=n_samp)
        draws = []
        for par in pars:
            draws.append(self.model.function(x, par))

        y_mu = np.mean(draws, axis=0)
        y_std = np.std(draws, axis=0)

        if noise:
            y_std = (y_std **2 + self.nvar)**.5

        return draws, y_mu, y_std


def curve_fit(x, y, model, rel_noise=None, par0=None):

    if par0 is None:
        par0 = model.par0

    bounds = np.array(list(zip(*model.bounds(x, y))))
    def _model(x, *args):
        return model.function(x, args)

    eps = 1e-12
    par0 = np.minimum(np.maximum(par0, bounds[0]+eps), bounds[1]-eps)
    popt, pcov = optimize.curve_fit(_model, x, y, p0=par0, bounds=bounds)

    def f(x):
        return model.function(x, popt)

    residual = np.sum((f(x) - y)**2)
    s2 = residual / (len(x) - len(model.par0))

    return FitResult(f, model, popt, pcov, s2, None)


def min_fit(x, y, model, method="L-BFGS-B", rel_noise=False, par0=None):

    # Optimise =========================================
    def opt_loss(params):
        y_pred = model.function(x, params)
        if rel_noise:
            return np.sum((y_pred / y - 1.)**2)
            # or y/y_pred if we believe y_pred is the truth
        else:
            return np.sum((y - y_pred) ** 2)  # Sum of squared errors

    bounds = model.bounds(x, y)
    if par0 is None:
        par0 = model.par0

    res = optimize.minimize(
        opt_loss, par0, method=method,
        jac=autograd.grad(opt_loss), bounds=bounds)
    popt = res.x

    # estimate pcov (trying to replicate curve_fit)
    # We can use analytic hessian rather than the optimiser
    hess = autograd.hessian(opt_loss)(popt)
    inv_hess = np.linalg.inv(hess)
    # inv_hess = np.asarray(res.hess_inv.todense())
    s2 = res.fun / (len(x) - len(model.par0))
    pcov = s2 * inv_hess

    def f(x):
        return model.function(x, popt)

    return FitResult(f, model, popt, pcov, s2, res)


def odr_fit(x, y, model, method="L-BFGS-B", rel_noise=False, par0=None):
    """
    Experimental orthogonal distance regression.
    Jointly estimates x and parameters.

    Limitations:
    - scaling wrt data size
    - currently no learned anisotropy (assuming equal noise on x and y)
    """

    n_par = len(model.par0)

    if par0 is None:
        par0 = model.par0

    def opt_loss(paramsx):
        """Squared error in X and Y"""
        params = paramsx[:n_par]
        x_pred = paramsx[n_par:]
        y_pred = model.function(x_pred, params)

        if rel_noise:
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

    return FitResult(f, model, popt, None, None, res)


def logspace_r2(y_true, y_pred):
    """
    Metric - R2 in logspace.
    Note - this is not in the *shifted space* which requires model parameters.
    """
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    return 1. - (ss_res / ss_tot)


def normal_log_likelihood(y_true, y_mu, y_sig):
    # Compute the negative log likelihood for Gaussian predictions
    nll = 0.5 * np.log(2 * np.pi * y_sig**2) + 0.5 * ((y_true - y_mu)**2) / y_sig**2
    return -np.sum(nll)




