"""Validation for sparse PCA with held out rows and columns."""

import numpy as np
import pandas as pd

# Local files
from scipy import linalg
from sklearn.linear_model import ridge_regression
from sklearn.decomposition import SparsePCA
from sklearn.model_selection import train_test_split
from tqdm import tqdm



def block_train_test_split(n_samples, block_size, test_size=0.2, random_state=42):
    """Like a shuffled train_test_split but selects contiguous blocks to handle autocorrelation"""
    
    # Select a shuffled set of blocks
    n_blocks = n_samples // block_size
    n_blocks = int(np.ceil(n_samples / block_size))
    rng = np.random.RandomState(random_state)
    shuffled_blocks = rng.permutation(n_blocks)
    
    # Create a boolean array indicating which blocks are for training
    block_is_train = np.ones(n_blocks, dtype=bool)
    n_test_blocks = int(np.ceil(n_blocks * test_size))
    block_is_train[shuffled_blocks[:n_test_blocks]] = False
    
    # Map each sample to its corresponding block's train/test status
    # Consecutive blocks of length block_size are assigned to the same block
    block_indices = np.arange(n_samples) // block_size
    is_train = block_is_train[block_indices]
    
    # Get indices
    train_indices = np.where(is_train)[0]
    test_indices = np.where(~is_train)[0]
    
    return train_indices, test_indices



def bi_validate(model, X, row_frac=0.3, col_frac=0.3, row_autocorr=2, seed=42):
    """Submatrix holdout validation, suitable for factorisation models."""
    assert isinstance(model, SparsePCA), "The methodology assumes a Sparse PCA model."
    if isinstance(X, pd.DataFrame):
        X = X.values

    n_rows, n_cols = X.shape
    # If the rows are in checkpoint order, do we need to do a timeseries split?
    r1, r0 = block_train_test_split(n_rows, row_autocorr, test_size=row_frac, random_state=seed)
    c1, c0 = train_test_split(np.arange(n_cols), test_size=col_frac, random_state=seed+1)

    # X = (A B)
    #     (C D)
    A = X[np.ix_(r0, c0)]
    B = X[np.ix_(r0, c1)]
    CD = X[r1]

    # Predict A only observing B, C, D
    model.fit(CD)  # Learn factorisation of non heldout rows [C, D]

    # Extract factorisation components corresponding to D
    sub_components = model.components_[:, c1]
    sub_mean = model.mean_[c1]

    # Apply these components to "score" B
    AB_score = ridge_regression(
        sub_components.T, (B-sub_mean).T, model.ridge_alpha, solver="cholesky"
    )

    # Extrapolate these scores to reconstruct A
    AB_pred = model.inverse_transform(AB_score)
    A_pred = AB_pred[:, c0]

    # Compute the validation loss on A (which was unseen by the methodology)
    MSE = np.sum((A - A_pred)**2)  # your loss here

    return MSE



# Testing stuff
# Test BCV
def synthetic(n_samples, n_features, rank, seed=15):
    rng = np.random.RandomState(seed)
    U, _, _ = linalg.svd(rng.randn(n_features, n_features))
    X = np.dot(rng.randn(n_samples, rank), U[:, :rank].T)
    X /= X.std()
    return X


def apply_noise(T, sigma=0.05, sigma_hs=0., sigma_p=0.05, seed=45):
    """Noise model with heteroskedastic and proportional noise."""
    rng = np.random.RandomState(seed)
    n, n_features = T.shape

    sigmas = sigma + sigma_hs * rng.rand(n_features)
    sigmas = sigmas[None, :]
    sigmas = sigmas + sigma_p * T  # proportional noise level
    X = T + sigmas * rng.randn(n, n_features)

    return X


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Supress M4 accelerate warnings
    import warnings
    warnings.filterwarnings('ignore', message='invalid value')
    warnings.filterwarnings('ignore', message='divide by zero')
    warnings.filterwarnings('ignore', message='overflow')


    # Run a demo:
    V = synthetic(50, 20, 5) # how many dimensions
    X = apply_noise(V, sigma=.2, sigma_hs=0., sigma_p=0.)

    model = SparsePCA(alpha=1)

    MSEs = []

    components = np.arange(1, 12, dtype=int)

    for n_components in tqdm(components):
        model.n_components = n_components
        MSEs.append(
            bi_validate(model, X)
        )

    plt.figure
    plt.plot(components, MSEs, label="Block train/test split")
    plt.title("CrossValidation")
    plt.legend()
    plt.show()
