# Liam's explanation

## Pertoken losses

For each Pile subset $P$ (aside from `dm_mathematics` which I'll write about below), we constructed a dataset $D_P$ of $K=512$ contexts $\{S_k^P\}_{k=1}^K$, where each $|S_k| \leq 512$ in token length (some datasets, like `enron_emails`, have small contexts, but most max out at 512 tokens). Each $S_k$ is randomly sampled from a context from Timaeus' pile-subsets (in particular: we first randomly sample a context, and then randomly sample the token slice from within that context, so that the contexts didn't just represent text beginnings).

Given a dataset $D_P$, for each Pythia model size $M_t$ in the suite $mathcal{M} = [14m, ..., 6.9b]$ at training checkpoint $t$, we evaluate the next-token prediction loss on each context $S_K^P$, which I will denote $\ell_{M_t, S_K^P}$. (The 0th token is always padded with $\ell=0$, since pythia doesn't use a `<BOS>` token). Thus, for each model $M \in \mathcal{M}$ and each dataset $D_P$, we construct a trajectory pertoken loss matrix $L_{M, P}$ of size `(num_checkpoints, total_tokens )` where `total_tokens = \sum_k |S_k|` with each $\ell_{M_t, S_K^P}$ in the appropriate place. 

`dm_mathematics` is a special dataset, where we constructed 560 problems $x$ with 10 contexts per category (where a category is a module and a template, in the language of that paper - e.g. `algebra - linear_1d`). Each problem has a solution, which also follows after the `"\n"` token, which allows for more fine-grained analysis. This allows us to construct two datasets:
- $D_{DM, zero}$ is the dataset of 560 contexts such that each context is one problem with one answer (thus zero-shot asking the question). 
- $D_{DM, few}$ is the dataset of 56 contexts such that each context represents a _category_, where all problems from a given category are concatenated (up to a certain max-token-threshold [I think 512]) to form a few-shot context. The problems are separated by a `"\n"` token, which we do not track the pertoken loss of. 

This means that $L_{M, D_{DM, zero}}$ and $L_{M,D_{DM, few}}$ have the exact same dimensions, where each row and column matches; the only thing that is different is the loss calculated upon either the few-shot or zero-shot strategy. 

## Pertoken loss HTMLs
Let $L_{M, P'}$ be the pertoken loss matrix above shortened to only contain the data of contexts $D_P' = \{S_k\}_{k=1}^50$, which we restrict for computational efficiency (otherwise the matrices are too big to dsiplay/do PCA on). Then for non-`dm_mathematics` Pile subsets: 
- The "raw loss" pane of the pertoken loss HTMLs is just displaying the pure data in $L_{M, P'}$.
- Given a subset of checkpoints $T \subseteq \mathcal{T}$ as selected by the user, the "stepwise difference" calculates $L_{M_t, P'} - L_{M_{t-1}, P'}$ for each index of $T$. (Therefore each row displays the difference between that row and the one before it, e.g. `step1000 - step512`).
- The "model difference" pane displays $L_{M_2, P'} - L_{M_1, P'}$ for two models $M_1, M_2 \in \mathcal{M}$. 

For `dm_mathematics` specifically, all of the above applies, plus: 
- The raw loss is either showing $L_{M, D_{DM, zero}}$ or $L_{M,D_{DM, few}}$
- The prompting difference is showing $L_{M,D_{DM, few}} - L_{M, D_{DM, zero}}$. 

## PCA 

As in the Essential Dynamics project, for each dataset $D_P'$ we construct a large matrix $L_{\mathcal{M}, P}$ which vertically concatenates $[L_{M_1, P'}, \dots, L_{M_{max}, P'}]^{\top}$ to create a joint trajectory matrix of size `(num_models*num_checkpoints, total_num_tokens)`. We then apply some different methods: 
- Full PCA on $L_{\mathcal{M}, P}$, extracting 10 components. 
- Sparse PCA on $L_{\mathcal{M}, P}$, extracting 10 components. Here I used 

```
default_sparse_params = {
                'alpha': 5.0,  # L1 penalty parameter
                'ridge_alpha': 0.01,  # Ridge penalty parameter
                'max_iter': 1000,
                'tol': 1e-6,
                'random_state': 42}
```
, but in one experiment I changed `alpha` to 15. 
- Hybrid PCA: first apply full PCA to extract `n_components`. Then calculate: 
```
reconstructed_data = self.pca.inverse_transform(pca_transformed)
model_residuals = original_data - reconstructed_data
``` 
Then apply sparse PCA as above to the matrix `model_residuals`. 

