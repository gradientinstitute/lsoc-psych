import numpy as np
from sklearn.model_selection import KFold
import pandas as pd


def cross_validate(X, model, max_factors, n_folds=10, repeats=10):

    if isinstance(X, pd.DataFrame):
        X = X.values

    # Use crossvalidation to select the number of latent factors...
    n_factors = np.arange(max_factors) + 1

    fit_err = []  # seeing all the data
    heldout_err = []  # crossval
    heldout_std = []

    indices = np.arange(np.prod(X.shape))

    for dims in n_factors:
        print(f"Evaluating {dims} dimensions...", flush=True)
        MSEs = []  # Samples losses

        for r in range(repeats):
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42+r)
            for train_idx, test_idx in kf.split(indices):
                mask = np.zeros(X.shape, bool)
                mask.flat[test_idx] = True
                if len(X.shape) == 2:
                    # never completely mask
                    assert all(mask.shape[1] != mask.sum(axis=1))
                R = model.fit(X, dims, mask)
                mse = np.mean((R[mask] - X[mask])**2)
                MSEs.append(mse)

        heldout_err.append(np.mean(MSEs))  # should be the same
        heldout_std.append(np.std(MSEs) / np.sqrt(len(MSEs)))

        # Control - no masking
        recon = model.fit(X, dims)
        fit_err.append(np.mean((recon - X)**2))

    heldout_err = np.array(heldout_err)
    heldout_std = np.array(heldout_std)
    return heldout_err, heldout_std, fit_err


def row_cross_validate(X, model, max_factors, n_folds=10, repeats=1):
    """Cross validate a scikit learn estimator row-wise wrt holdout error."""
    if isinstance(X, pd.DataFrame):
        X = X.values

    fit_err = []  # seeing all the data
    heldout_err = []  # crossval
    heldout_std = []

    indices = np.arange(X.shape[0])  # by rows rather than at random

    for dims in n_factors:
        print(f"Evaluating {dims} dimensions...", flush=True)
        MSEs = []  # Samples losses

        for r in range(repeats):
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42+r)
            for train_idx, test_idx in kf.split(indices):
                mask = np.zeros(X.shape, bool)
                mask.flat[test_idx] = True
                if len(X.shape) == 2:
                    # never completely mask
                    assert all(mask.shape[1] != mask.sum(axis=1))
                R = model.fit(X, dims, mask)
                mse = np.mean((R[mask] - X[mask])**2)
                MSEs.append(mse)

        heldout_err.append(np.mean(MSEs))  # should be the same
        heldout_std.append(np.std(MSEs) / np.sqrt(len(MSEs)))

        # Control - no masking
        recon = model.fit(X, dims)
        fit_err.append(np.mean((recon - X)**2))

    heldout_err = np.array(heldout_err)
    heldout_std = np.array(heldout_std)
    return heldout_err, heldout_std, fit_err

