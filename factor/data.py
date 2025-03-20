import pandas as pd
import numpy as np
from scipy import linalg
from scipy.cluster import hierarchy


def load(name):
    if name == "p70m":
        return load_pythia_table("70m")[0]
    elif name == "helm":
        return load_helm()
    elif name == "synthetic":
        return synthetic()
    else:
        raise NotImplementedError()


def load_pythia_table(model_size):
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


def scale(X, method="standard"):
    if method == "none":
        return X.copy()
    elif method == "modality":
        T, indices = df_to_tensor(X)
        for i in range(T.shape[-1]):
            T[:, :, i] /= np.std(T[:, :, i].ravel())
        return tensor_to_df(T, indices)

    if isinstance(X, pd.DataFrame):
        values = X.values
    else:
        values = X

    scaled = values.copy()  # We're about to modify them inplace

    if method == "positive":
        scaled -= scaled.min(axis=0)
        scaled /= scaled.max(axis=0)
    elif method == "standard":
        scaled -= scaled.mean(axis=0)
        scaled /= scaled.std(axis=0)

    else:
        raise NotImplementedError()

    if isinstance(X, pd.DataFrame):
        scaled = pd.DataFrame(
            scaled,
            columns=X.columns,
            index=X.index,
        )

    return scaled


def df_to_tensor(X):
    """Convert a dataframe into a tensor."""
    # Step 1: Split the column names into measure and task
    tasks = list(X.columns.str.split('/').str[1].unique())
    measures = list(X.columns.str.split('/').str[0].unique())

    # Step 2: Reshape into 3D array
    steps = X.index.values
    T = np.zeros((len(steps), len(tasks), len(measures)))

    #T *= np.array([1, 1, .25])[None, None, :]

    # Fill the array (or could cast as a pd.MultiIndex)
    for i, task in enumerate(tasks):
        for j, measure in enumerate(measures):
            T[:, i, j] = X[f'{measure}/{task}'].values

    return T, (steps, tasks, measures)


def tensor_to_df(T, indices):
    index, tasks, measures = indices
    shape = tuple(len(v) for v in indices)
    assert shape==T.shape, "Indices don't match"
    columns = {}
    ix = 0
    for j, measure in enumerate(measures):
        for i, task in enumerate(tasks):
            columns[f'{measure}/{task}'] = T[:, i, j]
            ix += 1

    return pd.DataFrame(columns, index=index)
