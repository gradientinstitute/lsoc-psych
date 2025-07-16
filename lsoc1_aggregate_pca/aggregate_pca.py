import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors as pc

class PCAAnalysis:
    """
    A class to perform PCA analysis and store all results in one place.
    """
    
    def __init__(self, df, n_components=10, standardise=True, name="Data"):
        """
        Initialize PCA analysis.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input data matrix
        n_components : int
            Number of principal components to compute
        standardise : bool
            Whether to standardize the data before PCA
        name : str
            Name for this dataset (used in plots)
        """
        self.df = df
        if n_components is None:
            self.n_components = min(df.shape)
        else:
            self.n_components = n_components
        self.standardise = standardise
        self.name = name
        
        # Apply PCA
        self._apply_pca()
        
        # Create formatted labels
        self._create_labels()
    
    def _apply_pca(self):
        """Apply PCA to the input data and store results."""
        # Standardize the features (columns)
        if self.standardise:
            self.scaler = StandardScaler()
            standardised = self.scaler.fit_transform(self.df)
        else:
            standardised = self.df.values
        
        # Apply PCA
        self.pca = PCA(n_components=self.n_components)
        scores = self.pca.fit_transform(standardised)
        
        # Create DataFrame with original row names (variance-scaled scores)
        self.scores = pd.DataFrame(
            scores, 
            index=self.df.index,
            columns=[f'PC{i+1}' for i in range(self.n_components)]
        )
        
        # Create unit scores (left singular vectors, not scaled by variance)
        # These are the scores divided by sqrt(explained_variance)
        unit_scores = scores / np.sqrt(self.pca.explained_variance_)
        self.unit_scores = pd.DataFrame(
            unit_scores,
            index=self.df.index,
            columns=[f'PC{i+1}' for i in range(self.n_components)]
        )
        
        # Create loadings DataFrame
        self.loadings = pd.DataFrame(
            self.pca.components_.T,
            index=self.df.columns,
            columns=[f'PC{i+1}' for i in range(self.n_components)]
        )

        # if pile in name, also calculate average loadings matrix. Each column is is a Pile subset. Each column of the original is like "{category} {num}".
        if 'pile' in self.name.lower():
            # Extract categories from the column names
            categories = self.df.columns.str.split(' ', expand=True).get_level_values(0)
            unique_categories = categories.unique()
            
            # Calculate mean loadings for each category
            self.mean_loadings = pd.DataFrame(
                index=unique_categories,
                columns=self.loadings.columns
            )
            
            for category in unique_categories:
                category_mask = categories == category
                self.mean_loadings.loc[category] = self.loadings.loc[category_mask].mean(axis=0)
        else:
            self.mean_loadings = None

        # Store explained variance
        self.explained_variance = {
            f'PC{i+1}': self.pca.explained_variance_[i].item() 
            for i in range(self.n_components)
        }
        
        self.explained_variance_ratio = {
            f'PC{i+1}': self.pca.explained_variance_ratio_[i].item() 
            for i in range(self.n_components)
        }
    
    def _create_labels(self):
        """Create formatted labels for plots."""
        self.pc_labels = [
            f'PC{i+1} ({self.explained_variance_ratio[f"PC{i+1}"]:.1%})' 
            for i in range(self.n_components)
        ]
        
        self.pc_names = [f'PC{i+1}' for i in range(self.n_components)]


## Plotting ##

def create_sorted_data(pca_X, pca_Y, sort_matrix, sort_component, use_unit=False):
    """Create sorted data for a specific matrix and component combination"""
    # Get the component index
    sort_col = int(sort_component[2:]) - 1
    
    # Choose which scores to use for sorting
    if use_unit:
        sort_scores_X = pca_X.unit_scores
        sort_scores_Y = pca_Y.unit_scores
    else:
        sort_scores_X = pca_X.scores
        sort_scores_Y = pca_Y.scores
    
    # Sort by the specified matrix and component
    if sort_matrix == 'X':
        sorted_indices = sort_scores_X.iloc[:, sort_col].sort_values().index
    else:  # sort_matrix == 'Y'
        sorted_indices = sort_scores_Y.iloc[:, sort_col].sort_values().index
    
    # Reorder all dataframes using the same indices
    scores_X_sorted = pca_X.scores.loc[sorted_indices]
    scores_Y_sorted = pca_Y.scores.loc[sorted_indices]
    unit_scores_X_sorted = pca_X.unit_scores.loc[sorted_indices]
    unit_scores_Y_sorted = pca_Y.unit_scores.loc[sorted_indices]
    
    return scores_X_sorted, scores_Y_sorted, unit_scores_X_sorted, unit_scores_Y_sorted

def plot_pc_score_heatmaps(pca_X, pca_Y):
    """Create PCA heatmaps with dropdown menu for sorting options and unit/scaled toggle"""

    # Get all possible sorting combinations
    matrices = ['X', 'Y']
    components = list(pca_X.scores.columns)
    
    # Create initial plot (sorted by Y matrix, PC1, using scaled scores)
    initial_X, initial_Y, initial_unit_X, initial_unit_Y = create_sorted_data(
        pca_X, pca_Y, 'Y', 'PC1', use_unit=False
    )
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f'{pca_X.name} - PCA Scores (Scaled)',
            f'{pca_Y.name} - PCA Scores (Scaled)'
        ],
        horizontal_spacing=0.15
    )
    
    # Add initial heatmaps (scaled scores)
    fig.add_trace(
        go.Heatmap(
            z=initial_X.values,
            x=pca_X.pc_labels,
            y=initial_X.index,
            colorscale='RdBu_r',
            zmid=0,
            zmin=-50, zmax=50,
            name=pca_X.name,
            showscale=True,
            colorbar=dict(x=0.45, len=0.9),
            hovertemplate="%{x}<br>model: %{y}<br>score: %{z}<extra></extra>",
            visible=True
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Heatmap(
            z=initial_Y.values,
            x=pca_Y.pc_labels,
            y=initial_Y.index,
            colorscale='RdBu_r',
            zmid=0,
            name=pca_Y.name,
            showscale=True,
            colorbar=dict(x=1.02, len=0.9),
            hovertemplate="%{x}<br>model: %{y}<br>score: %{z}<extra></extra>",
            visible=True
        ),
        row=1, col=2
    )
    
    # Add unit score heatmaps (initially hidden)
    fig.add_trace(
        go.Heatmap(
            z=initial_unit_X.values,
            x=pca_X.pc_labels,
            y=initial_unit_X.index,
            colorscale='RdBu_r',
            zmid=0,
            zmin=-5, zmax=5,
            name=pca_X.name,
            showscale=True,
            colorbar=dict(x=0.45, len=0.9),
            hovertemplate="%{x}<br>model: %{y}<br>score: %{z}<extra></extra>",
            visible=False
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Heatmap(
            z=initial_unit_Y.values,
            x=pca_Y.pc_labels,
            y=initial_unit_Y.index,
            colorscale='RdBu_r',
            zmid=0,
            zmin=-5, zmax=5,
            name=pca_Y.name,
            showscale=True,
            colorbar=dict(x=1.02, len=0.9),
            hovertemplate="%{x}<br>model: %{y}<br>score: %{z}<extra></extra>",
            visible=False
        ),
        row=1, col=2
    )
    
    # Create score type dropdown buttons
    score_type_buttons = []
    
    # Create score type buttons
    score_type_buttons.append(
        dict(
            label="SV Scaled Scores",
            method="update",
            args=[{
                "visible": [True, True, False, False]
            }, {
                "annotations[0].text": f'{pca_X.name} - PCA Scores (Scaled)',
                "annotations[1].text": f'{pca_Y.name} - PCA Scores (Scaled)'
            }]
        )
    )
    
    score_type_buttons.append(
        dict(
            label="Unit Scores",
            method="update",
            args=[{
                "visible": [False, False, True, True]
            }, {
                "annotations[0].text": f'{pca_X.name} - PCA Scores (Unit)',
                "annotations[1].text": f'{pca_Y.name} - PCA Scores (Unit)'
            }]
        )
    )
    
    # Create sorting dropdown buttons
    sort_buttons = []
    
    for matrix in matrices:
        for component in components:
            # Create buttons for scaled scores
            sorted_X, sorted_Y, _, _ = create_sorted_data(
                pca_X, pca_Y, matrix, component, use_unit=False
            )
            
            # Create buttons for unit scores  
            _, _, sorted_unit_X, sorted_unit_Y = create_sorted_data(
                pca_X, pca_Y, matrix, component, use_unit=True
            )
            
            button = dict(
                label=f"{pca_X.name if matrix == 'X' else pca_Y.name} {component}",
                method="restyle",
                args=[{
                    "z": [sorted_X.values, sorted_Y.values, sorted_unit_X.values, sorted_unit_Y.values],
                    "y": [sorted_X.index, sorted_Y.index, sorted_unit_X.index, sorted_unit_Y.index]
                }]
            )
            sort_buttons.append(button)
    
    # Update layout with both dropdowns
    fig.update_layout(
        title=f'PCA Scores Comparison: {pca_X.name} vs {pca_Y.name} (n={len(pca_X.scores.index)})',
        height=max(400, len(pca_X.scores.index) * 20),
        width=1000,
        font=dict(size=10),
        updatemenus=[
            # Score type dropdown
            dict(
                buttons=score_type_buttons,
                direction="down",
                showactive=True,
                x=0.45,
                xanchor="center",
                y=1.11,
                yanchor="top"
            ),
            # Sort by dropdown
            dict(
                buttons=sort_buttons,
                direction="down",
                showactive=True,
                x=0.85,
                xanchor="center",
                y=1.11,
                yanchor="top"
            )
        ]
    )

    # Add annotations for both dropdowns
    fig.add_annotation(
        text="Score Type:",
        x=0.25, y=1.1,
        showarrow=False,
        font=dict(size=12),
        xref="paper", yref="paper",
        align="center"
    )
    
    fig.add_annotation(
        text="Sort by:",
        x=0.74, y=1.1,
        showarrow=False,
        font=dict(size=12),
        xref="paper", yref="paper",
        align="center"
    )
    
    # Update axes
    fig.update_xaxes(title_text="Principal Components", row=1, col=1)
    fig.update_xaxes(title_text="Principal Components", row=1, col=2)
    fig.update_yaxes(title_text="Models", row=1, col=1)
    fig.update_yaxes(title_text="Models", showticklabels=False, row=1, col=2)
    
    return fig

def plot_score_correlation_heatmap(pca_X, pca_Y):
   """Create a heatmap showing the correlation between scores of two PCAAnalysis objects."""
   # Calculate correlation matrix
   X_scores = pca_X.scores.values
   Y_scores = pca_Y.scores.values
   corr_matrix = np.corrcoef(X_scores.T, Y_scores.T)[:X_scores.shape[1], X_scores.shape[1]:]
   
   # Reverse the order of pca_X components so PC1 is at the top
   corr_matrix = corr_matrix[::-1, :]
   
   # Create text annotations
   text_values = [[f"{val:.2f}" for val in row] for row in corr_matrix]
   
   # Create heatmap
   fig = go.Figure(data=go.Heatmap(
       z=corr_matrix,
       x=pca_Y.pc_labels,
       y=pca_X.pc_labels[::-1],  # Reverse the labels to match
       colorscale='RdBu_r',
       zmid=0,
       text=text_values,
       texttemplate="%{text}",
       textfont={"size": 10},
       colorbar=dict(title='Correlation', len=0.9, title_side='right'),
       hovertemplate=f"{pca_X.name}: %{{y}}<br>{pca_Y.name}: %{{x}}<br>corr: %{{z:.2f}}<extra></extra>"
   ))
   
   fig.update_layout(
       title=f'Score Correlation Heatmap: {pca_X.name} vs {pca_Y.name} (n={len(pca_X.scores.index)})',
       xaxis_title=f'Y {pca_Y.name} Components',
       yaxis_title=f'X {pca_X.name} Components',
       height=600,
       width=800,
       font=dict(size=10)
   )
   
   return fig


def plot_pc_loading_heatmap(pca_analysis, mean_loadings=False):
    """Create PCA loadings heatmap with dropdown menu for sorting by different PCs and sort order"""
    if mean_loadings and pca_analysis.mean_loadings is not None:
        loadings = pca_analysis.mean_loadings
        name = f"{pca_analysis.name} (mean loadings)"
    else:
        loadings = pca_analysis.loadings
        name = pca_analysis.name

    print(len(loadings.index))

    # Get all components
    components = list(pca_analysis.loadings.columns)
    
    # Create initial plot sorted by PC1
    initial_sort_col = components[0]
    initial_sorted_indices = loadings[initial_sort_col].sort_values(ascending=True).index
    initial_sorted_loadings = loadings.loc[initial_sorted_indices]
    
    # Create the figure
    fig = go.Figure()
    
    # Add initial heatmap
    fig.add_trace(
        go.Heatmap(
            z=initial_sorted_loadings.values,
            x=pca_analysis.pc_labels,
            y=initial_sorted_loadings.index,
            colorscale='RdBu_r',
            zmid=0,
            name=name,
            showscale=True,
            colorbar=dict(len=0.9, title='Loading', title_side='right'),
            hovertemplate="%{x}<br>feature: %{y}<br>loading: %{z:.3f}<extra></extra>"
        )
    )
    
    # Create sorting dropdown buttons for all component/order combinations
    sort_buttons = []
    
    for component in components:
        # Create buttons for both ascending and descending for each component
        for ascending, order_label in [(True, "↑"), (False, "↓")]:
            sorted_indices = loadings[component].sort_values(ascending=ascending).index
            sorted_loadings = loadings.loc[sorted_indices]
            
            button = dict(
                label=f"{component} {order_label}",
                method="restyle",
                args=[{
                    "z": [sorted_loadings.values],
                    "y": [sorted_loadings.index]
                }]
            )
            sort_buttons.append(button)

    # Set y_vals for positioning
    if len(loadings.index) < 50:
        y_title_pos = 0.93
        y_sort_by_pos = 1.13
        y_sort_by_button_pos = 1.14
    elif len(loadings.index) >= 50 and len(loadings.index) < 110:
        y_title_pos = 0.97
        y_sort_by_pos = 1.04
        y_sort_by_button_pos = 1.045
    elif len(loadings.index) > 1250:
        y_title_pos = 0.998
        y_sort_by_pos = 1.002
        y_sort_by_button_pos = 1.0025
    else:
        y_title_pos = 0.985
        y_sort_by_pos = 1.02
        y_sort_by_button_pos = 1.025
    
    # Update layout with single dropdown
    fig.update_layout(
        title=dict(text=f'PCA Loadings Heatmap: {name} (n={len(pca_analysis.scores.index)})', y=y_title_pos),
        height=max(600, len(loadings.index) * 16),  # Adjust height based on number of features
        width=800,
        font=dict(size=10),
        updatemenus=[
            # Combined component and order selection dropdown
            dict(
                buttons=sort_buttons,
                direction="down",
                showactive=True,
                x=0.85,
                xanchor="center",
                y=y_sort_by_button_pos,
                yanchor="top"
            )
        ]
    )

    # Add annotation for dropdown
    fig.add_annotation(
        text="Sort by:",
        x=0.72, y=y_sort_by_pos,
        showarrow=False,
        font=dict(size=12),
        xref="paper", yref="paper",
        align="center"
    )
    
    # Update axes
    fig.update_xaxes(title_text="Principal Components")
    fig.update_yaxes(title_text="Features")
    
    return fig


## D(k) Analysis Functions ##

def calculate_dk_values(X, Y, k_X, standardise=False):
    """Calculate D(k_Y) values for a single k_X value across all k_Y values"""
    X = X.df
    Y = Y.df
    n, p = X.shape
    n_y, q = Y.shape
    assert n == n_y, "X and Y must have the same number of rows"
    
    # Center the matrices
    X_centered = X - X.mean(axis=0)
    Y_centered = Y - Y.mean(axis=0)

    # Optionally standardise the columns
    if standardise:
        X_centered = X_centered / X.std(axis=0)
        Y_centered = Y_centered / Y.std(axis=0)

    # Compute SVDs
    U_X, S_X, Vt_X = np.linalg.svd(X_centered, full_matrices=False)
    U_Y, S_Y, Vt_Y = np.linalg.svd(Y_centered, full_matrices=False)
    
    # Frobenius norm of Y (denominator)
    Y_frob_norm_sq = np.linalg.norm(Y_centered, 'fro') ** 2
    
    # Maximum k_Y value (minimum of n and q, but also limited by available singular values)
    max_k_Y = min(n, q, len(S_Y))
    
    # Calculate D(k_Y) for each k_Y
    dk_values = np.zeros(max_k_Y)

    # Truncate U_X to the first k_X columns
    U_X_truncate = U_X[:, :k_X]
    
    for k_Y in range(1, max_k_Y + 1):
        # Get first k_Y columns of U_Y and first k_Y singular values
        U_Y_k = U_Y[:, :k_Y]
        S_Y_k = np.diag(S_Y[:k_Y])
        
        # Calculate U_Y_k * S_Y_k
        U_Y_k_S_Y_k = U_Y_k @ S_Y_k
        
        # Calculate U_X * U_X^T * U_Y_k * S_Y_k
        projection_term = U_X_truncate @ (U_X_truncate.T @ U_Y_k_S_Y_k)
        
        # Calculate the difference
        difference = U_Y_k_S_Y_k - projection_term
        
        # Calculate Frobenius norm squared of the difference
        numerator = np.linalg.norm(difference, 'fro') ** 2
        
        # Calculate D(k_Y)
        dk_values[k_Y-1] = numerator / Y_frob_norm_sq
    
    return dk_values

def calculate_dk_dict(pca_X, pca_Y, max_k_X=None, standardise=True):
    """Calculate D(k_Y) values over a dict of k_X values"""
    if max_k_X is None:
        n, p = pca_X.df.shape
        max_k_X = min(n, p)  # Use rank of X if None
    
    k_X_values = list(range(1, max_k_X + 1))
    dk_dict = {}
    
    for k_X in k_X_values:
        dk_values = calculate_dk_values(pca_X, pca_Y, k_X=k_X, standardise=standardise)
        dk_dict[k_X] = dk_values
    
    return dk_dict

def plot_dk_dict(dk_dict, pca_X_name="X", pca_Y_name="Y", max_k_Y=11):
    """
    Plot D(k_Y) values for multiple k_X values with viridis colorscale.
    """
    # Convert dict to list for plotting
    data = []
    for k_X, dk_values in dk_dict.items():
        for k_Y, dk in enumerate(dk_values, 1):
            data.append({'k_Y': k_Y, 'D(k_Y)': dk, 'k_X': k_X})
    
    df = pd.DataFrame(data)
    # subset to max_k_Y
    if max_k_Y is not None:
        df = df[df['k_Y'] <= max_k_Y]
    
    fig = go.Figure()
    
    # Get unique k_X values
    k_X_values = sorted(df['k_X'].unique())
    
    # Get viridis colors
    import plotly.colors as pc
    viridis_colors = pc.sample_colorscale('viridis', len(k_X_values))
    
    # Add traces for each k_X
    for i, k_X in enumerate(k_X_values):
        df_subset = df[df['k_X'] == k_X]
        fig.add_trace(go.Scatter(
            x=df_subset['k_Y'],
            y=df_subset['D(k_Y)'],
            mode='lines+markers',
            name=f'k_X={k_X}',
            line=dict(width=2, color=viridis_colors[i]),
            marker=dict(size=4)
        ))
    
    fig.update_layout(
        title=f'D(k_Y) vs k_Y for different k_X values: {pca_X_name} → {pca_Y_name}',
        xaxis_title='k_Y (number of singular vectors of Y)',
        yaxis_title='D(k_Y)',
        width=800,
        height=500,
    )
    
    return fig

def plot_raw_data_heatmap(pca_analysis, sort_by='PC1', ascending=True, drop_columns=None, standardise=False,
                          negate_values=False):
    """
    Create a simple heatmap of the original data ordered by a specified PC component.
    """
    # Start with the original data
    data = pca_analysis.df.copy()
    
    # Negate the values if requested (convert loss to negative loss)
    if negate_values:
        data = -data
    
    # Get the PC scores for sorting
    pc_scores = pca_analysis.scores[sort_by]
    
    # Sort the dataframe by the specified PC component
    sorted_indices = pc_scores.sort_values(ascending=ascending).index
    sorted_data = data.loc[sorted_indices]
    
    # Drop specified columns if provided
    if drop_columns is not None:
        sorted_data = sorted_data.drop(columns=drop_columns, errors='ignore')
    
    # Sort columns by PC loadings
    pc1_loadings = pca_analysis.loadings[sort_by]
    sorted_columns = pc1_loadings.loc[sorted_data.columns].sort_values(ascending=False).index
    sorted_data = sorted_data[sorted_columns]

    # Standardise the columns
    if standardise:
        scaler = StandardScaler()
        sorted_data = pd.DataFrame(scaler.fit_transform(sorted_data),
                                   index=sorted_data.index,
                                   columns=sorted_data.columns)

    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=sorted_data.values.T,
        x=sorted_data.index,  # evaluations on x-axis
        y=sorted_data.columns,    # models on y-axis (ordered by PC)
        colorscale='RdBu_r',
        hoverongaps=False,
        zmid=0 if standardise else None,
        colorbar=dict(title='Negative Loss (standardised)' if negate_values else 'Loss', title_side='right')
    ))
    
    # Update layout
    fig.update_layout(
        title=f'Raw {"(Negated) " if negate_values else ""}{"(Standardised) " if standardise else ""}{pca_analysis.name} Data for n={len(pca_analysis.scores.index)} - Ordered by {pca_analysis.name}-{sort_by}.',
        xaxis_title='Evaluations',
        yaxis_title='Models',
        height=2000,
        width=1500
    )
    
    return fig


# Null hypothesis testing functions

def permute_matrix_columns(matrix):
    """
    Randomly permute the rows of each column independently.
    
    Parameters:
    matrix: numpy array or pandas DataFrame
    
    Returns:
    Permuted matrix of the same shape
    """
    if hasattr(matrix, 'values'):  # pandas DataFrame
        matrix_array = matrix.values
    else:
        matrix_array = matrix
    
    permuted = matrix_array.copy()
    
    # Permute each column independently
    for col in range(permuted.shape[1]):
        np.random.shuffle(permuted[:, col])
    
    # Return same type as input
    if hasattr(matrix, 'values'):
        return pd.DataFrame(permuted, columns=matrix.columns, index=matrix.index)
    else:
        return permuted

def run_null_hypothesis_experiment(pca_X, pca_Y, max_k_X=None, standardise=True, n_trials=100, permute_Y=False):
    """
    Run null hypothesis experiment by permuting X and Y matrices and calculating D(k_Y) values.
    
    Parameters:
    pca_X, pca_Y: PCA objects with .df attribute
    max_k_X: maximum k_X value to test (creates range 1 to max_k_X)
    standardise: whether to normalize the matrices
    n_trials: number of permutation trials
    
    Returns:
    Dictionary with statistics for each k_X and k_Y combination
    """
    if max_k_X is None:
        n, p = pca_X.df.shape
        max_k_X = min(n, p)  # Use rank of X if None
    
    k_X_values = list(range(1, max_k_X + 1))
    
    # Get original matrices
    X_orig = pca_X.df
    Y_orig = pca_Y.df
    
    # Storage for results
    null_results = {k_X: [] for k_X in k_X_values}
    
    print(f"Running {n_trials} permutation trials...")
    
    for trial in range(n_trials):
        if trial % 10 == 0:
            print(f"Trial {trial}/{n_trials}")
        
        # Create permuted matrices
        X_perm = permute_matrix_columns(X_orig)
        if permute_Y:
            Y_perm = permute_matrix_columns(Y_orig)
        else:
            Y_perm = Y_orig.copy()  # Use original Y for now
        
        # Create temporary PCA-like objects for the permuted data
        class TempPCA:
            def __init__(self, df, n_components):
                self.df = df
                self.n_components = n_components
        
        pca_X_perm = TempPCA(X_perm, pca_X.n_components)
        pca_Y_perm = TempPCA(Y_perm, pca_Y.n_components)
        
        # Calculate D(k_Y) for each k_X
        for k_X in k_X_values:
            dk_values = calculate_dk_values(pca_X_perm, pca_Y_perm, k_X=k_X, standardise=standardise)
            null_results[k_X].append(dk_values)
    
    # Convert to numpy arrays and calculate statistics
    null_stats = {}
    for k_X in k_X_values:
        dk_array = np.array(null_results[k_X])  # shape: (n_trials, max_k_Y)
        
        null_stats[k_X] = {
            'mean': np.mean(dk_array, axis=0),
            'std': np.std(dk_array, axis=0),
            'median': np.median(dk_array, axis=0),
            'percentile_5': np.percentile(dk_array, 5, axis=0),
            'percentile_95': np.percentile(dk_array, 95, axis=0),
            'all_trials': dk_array
        }
    
    return null_stats

def plot_dk_with_null_comparison(observed_dk_dict, null_stats, pca_X_name="X", pca_Y_name="Y", max_k_Y=11):
    """
    Plot observed D(k_Y) values with null hypothesis confidence intervals.
    
    Parameters:
    observed_dk_dict: Dictionary of observed D(k_Y) values
    null_stats: Dictionary of null hypothesis statistics
    pca_X_name, pca_Y_name: Names for plot labels
    max_k_Y: Maximum k_Y value to plot
    
    Returns:
    Plotly figure
    """
    fig = go.Figure()

    if max_k_Y is None:
        max_k_Y = len(observed_dk_dict[1])
    
    # Get unique k_X values
    k_X_values = sorted(observed_dk_dict.keys())
    
    # Get viridis colors
    viridis_colors = pc.sample_colorscale('viridis', len(k_X_values))
    
    # Add traces for each k_X
    for i, k_X in enumerate(k_X_values):
        # Observed data
        observed_dk = observed_dk_dict[k_X]
        k_Y_values = list(range(1, len(observed_dk) + 1))
        
        # Limit to max_k_Y
        k_Y_values = [k_Y for k_Y in k_Y_values if k_Y <= max_k_Y]
        observed_dk_subset = observed_dk[:max_k_Y]
        
        if k_X in null_stats:
            null_mean = null_stats[k_X]['mean'][:max_k_Y]
            null_std = null_stats[k_X]['std'][:max_k_Y]
            null_p5 = null_stats[k_X]['percentile_5'][:max_k_Y]
            null_p95 = null_stats[k_X]['percentile_95'][:max_k_Y]
            
            # Add null hypothesis confidence interval (95% percentile range)
            fig.add_trace(go.Scatter(
                x=k_Y_values + k_Y_values[::-1],  # x, then x reversed
                y=list(null_p95) + list(null_p5[::-1]),  # upper, then lower reversed
                fill='toself',
                fillcolor=viridis_colors[i].replace('rgb', 'rgba').replace(')', ', 0.2)'),
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                name=f'Null 95% CI (k_X={k_X})',
                hoverinfo='skip'
            ))
            
            # Add null hypothesis mean
            fig.add_trace(go.Scatter(
                x=k_Y_values,
                y=null_mean,
                mode='lines',
                name=f'Null mean (k_X={k_X})',
                line=dict(width=1, color=viridis_colors[i], dash='dash'),
                opacity=0.7
            ))
        
        # Add observed data
        fig.add_trace(go.Scatter(
            x=k_Y_values,
            y=observed_dk_subset,
            mode='lines+markers',
            name=f'Observed (k_X={k_X})',
            line=dict(width=2, color=viridis_colors[i]),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title=f'D(k_Y) vs k_Y: Observed vs Null Hypothesis<br>{pca_X_name}, {pca_X_name}_perturb → {pca_Y_name}',
        xaxis_title='k_Y (number of singular vectors of Y)',
        yaxis_title='D(k_Y)',
        width=1000,
        height=700,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    return fig


import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats

def plot_marginal_pdf(pca_analysis, x_col, y_col):
    """
    Create a scatter plot with marginal PDFs for two columns from a PCA analysis.
    
    Parameters:
    pca_analysis: PCA analysis object with .df attribute containing the data
    x_col: string, column name for x-axis
    y_col: string, column name for y-axis
    title_suffix: string, optional suffix for the plot title
    save_path: string, optional path to save the figure (without extension)
    scale: int, scale factor for saved image (default 3)
    
    Returns:
    Plotly figure object
    """
    
    # Create subplots with secondary axes
    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.85, 0.15],
        row_heights=[0.15, 0.85],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]],
        horizontal_spacing=0.02,
        vertical_spacing=0.02
    )

    # Main scatter plot (bottom left)
    fig.add_trace(go.Scatter(
        x=pca_analysis.df[x_col],
        y=pca_analysis.df[y_col],
        mode='markers+text',
        text=pca_analysis.df.index,
        textposition='top center',
        textfont=dict(size=8, color='black'),
        marker=dict(size=8, color='blue'),
        name='Models'
    ), row=2, col=1)

    # Top marginal (PDF for x-axis)
    x_data = pca_analysis.df[x_col]
    x_range = np.linspace(x_data.min(), x_data.max(), 100)
    kde_x = stats.gaussian_kde(x_data)
    fig.add_trace(go.Scatter(
        x=x_range,
        y=kde_x(x_range),
        mode='lines',
        fill='tozeroy',
        name=f'{x_col} PDF',
        line=dict(color='lightblue')
    ), row=1, col=1)

    # Right marginal (PDF for y-axis)
    y_data = pca_analysis.df[y_col]
    y_range = np.linspace(y_data.min(), y_data.max(), 100)
    kde_y = stats.gaussian_kde(y_data)
    fig.add_trace(go.Scatter(
        x=kde_y(y_range),
        y=y_range,
        mode='lines',
        fill='tozerox',
        name=f'{y_col} PDF',
        line=dict(color='lightcoral')
    ), row=2, col=2)

    # Create title
    title_prefix = f"{x_col} vs {y_col} with marginal PDFs"
    title = f"Raw {pca_analysis.name} data - {title_prefix}"

    # Update layout
    fig.update_layout(
        title=title,
        showlegend=False,
        width=1200,
        height=1200
    )

    # Update axes labels
    fig.update_xaxes(title_text=x_col, row=2, col=1)
    fig.update_yaxes(title_text=y_col, row=2, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=1)  # Hide x-axis ticks for top plot
    fig.update_yaxes(showticklabels=False, row=2, col=2)  # Hide y-axis ticks for right plot

    # Align the x-axis range of the top plot with the main plot
    x_range_main = [x_data.min(), x_data.max()]
    fig.update_xaxes(range=x_range_main, row=1, col=1)
    
    return fig


def analyse_reconstruction_errors(X, Y, max_k_y=None, standardise=False):
    """
    Analyze reconstruction errors across different k_x values for predicting Y from X.
    """
    X_data = X.df
    Y_data = Y.df
    Y_index = Y_data.index
    Y_columns = Y_data.columns
    n, p = X_data.shape
    n_y, q = Y_data.shape
    assert n == n_y, "X and Y must have the same number of rows"

    n, p = X_data.shape  # X: n samples × p features
    n_y, q = Y_data.shape  # Y: n samples × q targets
    assert n == n_y, "X and Y must have the same number of rows"
    
    # Center the matrices
    X_centered = X_data - X_data.mean(axis=0)  # Shape: (n, p)
    Y_centered = Y_data - Y_data.mean(axis=0)  # Shape: (n, q)

    # Optionally standardise the columns
    if standardise:
        X_centered = X_centered / X_data.std(axis=0)  # Shape: (n, p)
        Y_centered = Y_centered / Y_data.std(axis=0)  # Shape: (n, q)

    # Compute SVDs
    U_X, S_X, Vt_X = np.linalg.svd(X_centered, full_matrices=False)  # U_X: (n, min(n,p)), S_X: (min(n,p),), Vt_X: (min(n,p), p)
    U_Y, S_Y, Vt_Y = np.linalg.svd(Y_centered, full_matrices=False)  # U_Y: (n, min(n,q)), S_Y: (min(n,q),), Vt_Y: (min(n,q), q)
    
    # Determine max_k_y
    max_k_y_available = min(n, q, len(S_Y))
    if max_k_y is None:
        max_k_y = max_k_y_available
    else:
        max_k_y = min(max_k_y, max_k_y_available)
    
    # Precompute Y reconstruction components up to max_k_y
    U_Y_k = U_Y[:, :max_k_y]  # Shape: (n, max_k_y)
    S_Y_k = np.diag(S_Y[:max_k_y])  # Shape: (max_k_y, max_k_y)
    U_Y_k_S_Y_k = U_Y_k @ S_Y_k  # Shape: (n, max_k_y)
    
    # Dictionary to store results
    results = {}
    
    # Range of k_x values to test (1 to number of rows, as there are fewer rows than columns)
    max_k_x = min(n, p, len(S_X))
    
    for k_x in range(max_k_x + 1):
        if k_x == 0:
            # If k_x is 0, we cannot project anything, so skip this iteration
            results[k_x] = {
                'reconstructed_Y_matrix': np.square(Y_centered),
                'total_error': np.linalg.norm(Y_centered, 'fro').item() 
            }
            continue

        # Get first k_x columns of U_X
        U_X_k = U_X[:, :k_x]  # Shape: (n, k_x)
        
        # Calculate projection: U_X_k * U_X_k^T * U_Y_k * S_Y_k
        projection_term = U_X_k @ (U_X_k.T @ U_Y_k_S_Y_k)  # Shape: (n, max_k_y)
        
        # Calculate the reconstruction error matrix in latent Y space
        latent_error_matrix = U_Y_k_S_Y_k - projection_term  # Shape: (n, max_k_y)
        
        # Project back to original Y space to see which Y variables are hard to predict
        reconstructed_Y_errors = latent_error_matrix @ Vt_Y[:max_k_y, :]  # Shape: (n, q)

        # Take square of each value 
        reconstructed_Y_errors = np.square(reconstructed_Y_errors)  # Shape: (n, q)
        
        # Convert back to DataFrame if original Y was a DataFrame
        if Y_index is not None and Y_columns is not None:
            import pandas as pd
            reconstructed_Y_errors = pd.DataFrame(
                reconstructed_Y_errors, 
                index=Y_index, 
                columns=Y_columns
            )
        
        # Calculate total error (Frobenius norm) - this is the same in both spaces
        total_error = np.linalg.norm(latent_error_matrix, 'fro').item()  # Scalar
        
        # For normalized error (like your original D(k) formula), we could divide by Y_frob_norm
        # Y_frob_norm_sq = np.linalg.norm(Y_centered, 'fro') ** 2  # This would be a scalar
        # normalized_total_error = (total_error ** 2) / Y_frob_norm_sq
        
        # Store results
        results[k_x] = {
            'reconstructed_Y_matrix': reconstructed_Y_errors,  # Shape: (n, q) - same as original Y
            'total_error': total_error  # Scalar
        }
    
    return results


def plot_reconstruction_errors_with_slider(reconstruction_results, sorted_evals=None, sorted_models=None, 
                                         hparam_name="Latent factors of X (k_X)", standardise=False, 
                                         negate_values=False, title_prefix="Reconstruction Square Errors"):
    """
    Create an interactive heatmap with slider to explore reconstruction errors across different k_x values.
    
    Parameters:
    -----------
    reconstruction_results : dict
        Results from analyze_reconstruction_errors function
    sorted_evals : list, optional
        List of evaluation names in desired order. If None, uses original order.
    sorted_models : list, optional  
        List of model names in desired order. If None, uses original order.
    hparam_name : str, default "Latent factors of X (k_X)"
        Label for the slider parameter
    standardise : bool, default False
        Whether to standardise the data within each k_x
    negate_values : bool, default False
        Whether to negate the values
    title_prefix : str, default "Reconstruction Errors"
        Prefix for the plot title
    """
    # Get all k_x values
    k_x_values = sorted(reconstruction_results.keys())
    
    # Process all reconstruction matrices to determine consistent sorting and color scale
    processed_data = {}
    all_values = []
    
    for k_x in k_x_values:
        data = reconstruction_results[k_x]['reconstructed_Y_matrix'].copy()
        
        # Negate values if requested
        if negate_values:
            data = -data
            
        # Sort rows (evaluations) if sorted_evals provided
        if sorted_evals is not None:
            available_evals = [eval_name for eval_name in sorted_evals if eval_name in data.index]
            data = data.loc[available_evals]
        
        # Sort columns (models) if sorted_models provided  
        if sorted_models is not None:
            available_models = [model_name for model_name in sorted_models if model_name in data.columns]
            data = data[available_models]
        
        # Standardise within this k_x if requested
        if standardise:
            scaler = StandardScaler()
            data = pd.DataFrame(scaler.fit_transform(data),
                               index=data.index,
                               columns=data.columns)
        
        processed_data[k_x] = data
        all_values.extend(data.values.flatten())
    
    # Calculate consistent color scale range
    vmin, vmax = np.min(all_values), np.max(all_values)
    
    # Create subplot with slider
    fig = make_subplots(rows=1, cols=1)
    
    # Add traces for each k_x value (initially all invisible except first)
    for i, k_x in enumerate(k_x_values):
        data = processed_data[k_x]
        total_error = reconstruction_results[k_x]['total_error']
        
        fig.add_trace(go.Heatmap(
            z=data.values.T,
            x=data.index,
            y=data.columns,
            colorscale='RdBu_r',
            hoverongaps=False,
            zmid=0 if standardise else None,
            zmin=vmin,
            zmax=10,
            #zmax=vmax,
            visible=(i == 0),  # Only first trace visible initially
            colorbar=dict(
                title='Negative Reconstruction Error (standardised)' if negate_values else 'Reconstruction Error', 
                title_side='right'
            ),
            name=f'{hparam_name}={k_x}'
        ))
    
    # Create slider steps
    steps = []
    for i, k_x in enumerate(k_x_values):
        total_error = reconstruction_results[k_x]['total_error']
        step = dict(
            method="update",
            args=[{"visible": [False] * len(k_x_values)},
                  {"title": f'{title_prefix} {"(Negated) " if negate_values else ""}{"(Standardised) " if standardise else ""}- {hparam_name}={k_x} (Total Error: {total_error:.4f})'}],
            label=str(k_x)
        )
        step["args"][0]["visible"][i] = True
        steps.append(step)
    
    # Add slider
    sliders = [dict(
        active=0,
        currentvalue={"prefix": f"{hparam_name}: "},
        pad={"t": 150},
        steps=steps,
        y=1.17
    )]
    
    # Update layout
    fig.update_layout(
        title=f'{title_prefix} {"(Negated) " if negate_values else ""}{"(Standardised) " if standardise else ""}- {hparam_name}={k_x_values[0]} (Total Error: {reconstruction_results[k_x_values[0]]["total_error"]:.4f})',
        xaxis_title='Evaluations',
        yaxis_title='Models',
        height=2000,
        width=1500,
        sliders=sliders
    )
    
    return fig