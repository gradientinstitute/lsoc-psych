import pandas as pd
import numpy as np
from scipy import linalg
from scipy.cluster import hierarchy


def load(name):
    if name == "p70m":
        return load_pythia_tensor("70m")[0]
    elif name == "helm":
        return load_helm()
    elif name == "synthetic":
        return synthetic()
    else:
        raise NotImplementedError()


def load_pythia_tensor(model_size):
    assert model_size == "70m"

    df = pd.read_csv(f'data/p{model_size}.csv')
    df['step'] = df['models'].str.extract(r'_(\d+)').astype(int)
    df = df.set_index('step').drop('models', axis=1)  # Go numerical for this dataset
    df.columns.name = "Tasks"

    # TODO: we recently got some standard deviations from the estimation process!
    Xcols = [r for r in df.columns if "std" not in r]
    Scols = [r for r in df.columns if "std" in r]
    X = df[Xcols]
    stds = df[Scols]

    return X, stds


def load_helm():
    """Load scraped HELM leaderboard (single modality)."""
    df = pd.read_csv('data/helm.csv')
    df.set_index('Unnamed: 0', inplace=True)
    df.index.name = "Model"
    return df


def synthetic(n_samples, n_features, rank):
    rng = np.random.RandomState(42)
    U, _, _ = linalg.svd(rng.randn(n_features, n_features))
    X = np.dot(rng.randn(n_samples, rank), U[:, :rank].T)
    X /= X.std()
    return preorder(X)


def apply_noise(T, sigma=0.05, sigma_hs=0., sigma_p=0.05, seed=42):
    """Noise model with heteroskedastic and proportional noise."""
    rng = np.random.RandomState(seed)
    n, n_features = T.shape

    sigmas = sigma + sigma_hs * rng.rand(n_features)
    sigmas = sigmas[None, :]
    sigmas = sigmas + sigma_p * T  # proportional noise level
    X = T + sigmas * rng.randn(n, n_features)

    return X


def preorder(X):
    """Structure data by pre-ordering columns with heirarchical clustering"""
    V = X
    if isinstance(X, pd.DataFrame):
        V = X.values
    linkage = hierarchy.linkage(V.T, method='ward')
    ix = hierarchy.leaves_list(linkage)

    if isinstance(X, pd.DataFrame):
        ordered = X[X.columns[ix]]
    elif isinstance(X, np.ndarray):
        ordered = X[:, ix]
    else:
        raise NotImplementedError("Unsupported format.")

    return ordered
