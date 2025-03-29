# Copyright (c) Gradient Institute and Timaeus. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Masked Factorisation For Model Selection"""
from scipy.optimize import minimize
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD, FactorAnalysis


class FA:
    """Implement Factor Analysis for masked data."""

    def __init__(self, iters=20, **kwargs):
        self.name = "FA"
        self.iters = iters
        # Wrap scikit-learn...
        self.model = FactorAnalysis(kwargs)

    def fit(self, X, dims, mask=None):
        """Masked factorisation."""
        assert mask is None or mask.dtype == bool

        if isinstance(X, pd.DataFrame):
            X = X.values

        if mask is None:
            # Dense PCA
            self._fit(X, dims)
            return self.R

        # EM Approach
        Xd = fill_column_mean(X, mask)

        for iteration in range(self.iters):
            self._fit(Xd, dims)
            Xd[mask] = self.R[mask]

        return self.R


    def _fit(self, X, dims):
        """Dense factorisation."""
        FA = self.model
        FA.n_components = dims
        self.U = FA.fit_transform(X)
        self.V = FA.components_
        self.mu = FA.mean_
        self.R = self.U @ self.V + self.mu


class PCA:
    """Implement PCA for missing-at-random data."""

    def __init__(self, iters=20):
        self.name = "PCA"  # Used for plotting etc.
        self.iters = iters

    def fit(self, X, dims, mask=None):

        assert mask is None or mask.dtype == bool

        if isinstance(X, pd.DataFrame):
            X = X.values

        if mask is None:
            # Dense PCA
            self._fit(X, dims)
            return self.R

        # EM Approach
        Xd = fill_column_mean(X, mask)

        for iteration in range(self.iters):
            self._fit(Xd, dims)
            Xd[mask] = self.R[mask]

        return self.R


    def _fit(self, X, dims):
        """Apply dense PCA factorisation."""
        self.mu = X.mean(axis=0)
        U, S, Vt = np.linalg.svd(X - self.mu, full_matrices=False)
        U = U[:, :dims]
        S = S[:dims]
        self.V = Vt[:dims, :]
        self.U = U * S[None, :]
        self.R = self.U @ self.V + self.mu


class NMF:

    def __init__(self, L2=1e-4, method="opt", max_iters=500):
        self.name = "NMF"
        self.L2 = L2
        # regulariser (optional) but a *tiny* amount helps with stability
        self.method = method
        self.max_iters = max_iters


    def heuristic(self, X, dims, mask=None):
        """Initialisation heuristic for NMF."""
        if mask is not None:
            X = fill_column_mean(X, mask)  # ensure dense

        # Should we automatically offset this?
        offset = X.min(axis=0)
        Z = X - offset

        # Use a clustering initialisation
        kmeans = KMeans(n_clusters=dims).fit(Z)
        V = np.maximum(0, kmeans.cluster_centers_)
        V = unit_norm(V)
        U = np.maximum(0, Z @ np.linalg.pinv(V))
        # THe positivity can be problematic...
        return U, V, offset


    def fit(self, X, dims, mask=None):
        """Fit a nonnegative factorisation on masked data."""

        if isinstance(X, pd.DataFrame):
            X = X.values

        # Initialise and pack
        if mask is not None:
            X = X.copy()  # so we can write to it
            assert mask.dtype == bool
            X[mask] = 0  # should be unused

        if self.method == "opt":
            return self.opt_fit(X, dims, mask)
        elif self.method == "mult":
            return self.mult_fit(X, dims, mask)
        else:
            raise NotImplementedError()


    def opt_fit(self, X, dims, mask=None):
        n_models, n_tasks = X.shape

        def unpack(q):
            """Unpack the parameters vector."""
            mat = q.reshape((n_models+n_tasks, dims))
            U = mat[:n_models]
            V = mat[n_models:].T
            return U, V

        # Turn initial state into parameters
        # (Or could start random)
        U0, V0, mu = self.heuristic(X, dims, mask)
        p0 = np.vstack((U0, V0.T)).ravel()
        L2 = self.L2  # Optional regulariser hyperparameter

        def masked_loss_fn(p):
            """Define fitting loss function."""
            U, V = unpack(p)
            F = U @ V + mu  # reconstruct
            err = F - X  # fit error
            if mask is not None:
                # We also need to support *unmasked* computation
                err[mask] = 0  # loss masking
            loss = np.sum(err**2) + L2*np.sum(p**2)
            dldu = 2*err@V.T  # Note masked errors are zero
            dldv = 2*U.T@err  # So we get gradient masking for free
            fgrad = np.vstack((dldu, dldv.T)).ravel()
            # mugrad = 2*np.sum(err, axis=0)  # Sign must match
            # grad = np.concatenate((mugrad, fgrad)) + 2*L2*p
            grad = fgrad + 2*L2*p
            return loss, grad

        # Apply a positivity constraint:
        bounds = ([(0, None)] * (n_models + n_tasks) * dims)
        # bounds = [(None, None)] * n_tasks  # Allow unbounded mean offset
        # bounds.extend([(0, None)] * (n_models + n_tasks) * dims)

        self.res = result = minimize(
            fun=masked_loss_fn, x0=p0, method='L-BFGS-B',
            jac=True, bounds=bounds,
            # tol=1e-3,
        )
        par = result.x
        assert result.success, result

        # Add-on - sort by variance
        U, V = unpack(par)
        dvars = np.array([np.var(np.outer(u, v)) for u, v in zip(U.T, V)])
        order = np.argsort(dvars)[::-1]
        U = U[:, order]
        V = V[order, :]
        self.U = U
        self.V = V
        self.O = mu
        self.R = U @ V + mu
        return self.R


    def mult_fit(self, X, dims, mask=None):
        """Multiplicative update of factors."""

        U, V, mu = self.heuristic(X, dims, mask)
        Z = X - mu

        # This is an update derived from the gradient of the opt_fit
        # But with per-parameter updates that ensure positivity and 
        # fast convergence (from the literature)
        eps = 1e-5  # avoid HUGE updates


        for alpha in np.linspace(1., 0., self.max_iters):
            if mask is not None:
                # Ensure masked do not contribute
                R = U @ V
                Z[mask] = R[mask]

            # learning rate schedule
            beta = 1. - alpha

            # Update U
            numerator = Z @ V.T
            denominator = U @ (V @ V.T)
            U *= beta + alpha * numerator / (denominator + eps)

            # Update W
            numerator = U.T @ Z
            denominator = (U.T @ U) @ V
            V *= beta + alpha * numerator / (denominator + eps)

            # U *= (Z @ V.T) / (U @ (V @ V.T))
            # V *= (U.T @ Z) / ((U.T @ U)@V)

        self.U = U
        self.V = V
        self.O = mu
        self.R = U @ V + mu
        return self.R


class TRD:
    """Tensor rank decomposition."""

    def __init__(self, dims=4, positive=[]):
        self.dims = dims
        self.positive = positive
        self.factors = None
        self.L2 = 1e-3
        self.L1 = 0.
        self.F = None
        self.R = None

    def fit(self, X, dims=None, mask=[]):
        # TODO: non-joint optimisation
        if dims:
            self.dims = dims
        else:
            dims = self.dims

        s = len(X.shape)
        X = X.copy()
        xshape = X.shape
        X[mask] = 0  # to stop cheating.. during development
        assert len(X.shape) > 2, "This is a tensor decomposition"

        # each component has 1d scores and loadings
        n_par = np.sum(X.shape) * dims

        # random initialisation
        # TODO: think more about this and its scale etc
        w0 = 1e-1 * np.random.rand(n_par)

        def loss_fn(w):
            parts = TRD.unpack(w, xshape, dims)

            F, R = TRD.reconstruct(parts)
            l, lg = mse(F, X, mask)

            # The gradient is simple, otherwise will switch to torch
            stack = []

            # Provided values are nonzero (they can be small) we can super-efficiently figure out the gradient by dividing the result.
            for i, p in enumerate(parts):
                pshape = [dims] + [1 for _ in range(s)]
                pshape[i+1] = p.shape[1]
                p = p.reshape(pshape)
                G = R / p
                G = G * lg[None, ...]
                # Do in reverse order to avoid changing the indices
                for j in range(len(parts), 0, -1):
                    if j == i+1:
                        continue
                    G = G.sum(axis=j)
                stack.append(G.ravel())

            grad = np.concat(stack)

            # Add a *tiny amount of regularisation to give a better behaved solution?
            L2 = self.L2
            l += L2 * np.sum(w**2)
            grad += 2 * L2 * w

            L1 = self.L1
            l += L1 * np.sum(np.abs(w))
            grad += L1 * np.sign(w)
            return l, grad

        result = minimize(fun=loss_fn, x0=w0, method='L-BFGS-B', jac=True)

        # Optionally, restrict various task loadings to be positivej
        if len(self.positive):
            # Build up the restricted bounds for the optimiser
            bounds = []
            eps = 1e-6
            for ix, n in enumerate(X.shape):
                term = [(None, None)]
                if ix in self.positive:
                    term = [(eps, None)]
                bounds.extend(term * (n * dims))
        else:
            bounds = None

        result = minimize(fun=loss_fn, x0=w0, method='L-BFGS-B', jac=True,
                        bounds=bounds)
        # [(eps, None)] * n_par)

        # Order the parts by variance (approximately)
        parts = TRD.unpack(result.x, xshape, dims)
        F, R = TRD.reconstruct(parts)

        # Order by reconstruction variance
        var = R.reshape(dims, -1).var(axis=1)
        order = np.argsort(var)[::-1]
        for i in range(len(parts)):
            parts[i] = parts[i][order]

        self.factors = parts
        self.F = F
        self.R = R
        return F


    @staticmethod
    def unpack(w, xshape, dims):
        i = 0
        parts = []
        for n in xshape:
            block = n * dims
            parts.append(w[i:i+block].reshape(dims, n))
            i += block
        return parts

    @staticmethod
    def reconstruct(parts):
        R = parts[0]
        pshape = list(R.shape)

        for p in parts[1:]:
            R = R[..., None]  # add a new axis
            pshape[-1] = 1  # flatten previous dimension
            pshape.append(p.shape[1])
            p = p.reshape(pshape)
            R = R * p  # broadcast the multiplication
        F = R.sum(axis=0)

        return F, R


def mse(F, X, mask):
    """Masked squared error optimisation objective."""
    err = (F - X)
    err[mask] = 0  # apply loss masking
    l = np.sum((err**2))
    lg = 2*err
    return l, lg


def fill_column_mean(X, mask):
    """Fill masked values with column means."""
    assert not np.isnan(X[~mask]).any()  # ensure all nan values are masked
    Xd = X.copy()
    Xd[mask] = 0
    mu = Xd.sum(axis=0) / (~mask).sum(axis=0)
    col_ix = np.tile(np.arange(X.shape[1]), (X.shape[0], 1))[mask]
    Xd[mask] = mu[col_ix]  # Fill with column mean
    return Xd


def unit_norm(v):
    # matrix is v x n_features
    # normalise along axis 1
    norm = (v**2).sum(axis=1) ** .5
    return v / norm[:, None]
