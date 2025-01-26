import pandas as pd
import pickle
import numpy as np
from collections import namedtuple




def load_dfs(msize, data_path):
    """Data Loading - updated to 24/1/2024 format"""

    if load_dfs.df_loss is None:
        with open(f'data/pythia-test-losses-new-seed-1024.pkl', 'rb') as file:
            df_loss = pickle.load(file)

        with open(f'data/pythia-pile-subset-llcs.pkl', 'rb') as file:
            df_llc = pickle.load(file)


        # LLC on full pile came as a "patch"
        with open(f'data/pythia-full-llc.pkl', 'rb') as file:
            df_full_llc = pickle.load(file)

        df_llc.set_index('Step', inplace=True)
        df_loss.set_index('Step', inplace=True)
        df_full_llc.set_index('Step', inplace=True)

        # Patch "full llc" into the results:
        renames = {c:c+"-full" for c in df_full_llc.columns}
        df_full_llc.rename(columns=renames, inplace=True)
        df_llc = df_llc.join(df_full_llc)
        load_dfs.df_loss = df_loss
        load_dfs.df_llc = df_llc
    else:
        df_loss = load_dfs.df_loss
        df_llc = load_dfs.df_llc

    sizes = set(c.split("-")[1] for c in df_loss.columns if "-" in c)
    assert msize in sizes, sizes

    # Extract this size:
    columns = [c for c in df_loss.columns if msize in c]
    rename = {c: c.split(msize+"-")[-1] for c in columns}
    task_loss = df_loss[columns].rename(columns=rename)
    task_llc = df_llc[columns].rename(columns=rename)

    # Note these may have NANs now...
    return task_llc / 100., task_loss
# Cache across calls
load_dfs.df_loss = None
load_dfs.df_llc = None



Trace = namedtuple('Trace', ['x','y','s'])


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

    # Sometimes LLCs are missing in the tail...
    if np.isnan(x).any():
        ix = np.argmax(np.isnan(x))
        x = x[:ix]
        y = y[:ix]
        s = s[:ix]

    assert not np.isnan(x).any()
    assert not np.isnan(y).any()

    return Trace(x, y, s)


def split(trace, cutoff_step):
    """Used for a simple train/test split."""
    x, y, s = trace
    cut = np.argmax(s > cutoff_step)
    x_train, y_train, s_train = x[:cut], y[:cut], s[:cut]
    x_test, y_test, s_test = x[cut:], y[cut:], s[cut:]
    return Trace(x_train, y_train, s_train), Trace(x_test, y_test, s_test)


def old_load_dfs(msize, data_path):
    """Data loading pre 24/1/2024 fmt"""

    with open(f'{data_path}/pile_{msize}_subset_loss_df.pkl', 'rb') as file:
       df_loss = pickle.load(file)

    with open(f'{data_path}/pile_{msize}_subset_llc_df.pkl', 'rb') as file:
      df_llc = pickle.load(file)

    step = "step" if "step" in df_loss.columns else "Step"
    df_loss.set_index(step, inplace=True)
    df_llc.set_index(step, inplace=True)
    df_llc = df_llc / 100.  # adjust the scale

    return df_llc, df_loss



