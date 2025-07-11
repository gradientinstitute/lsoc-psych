import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class PCAAnalysis:
    """
    A class to perform PCA analysis and store all results in one place.
    """
    
    def __init__(self, df, n_components=10, normalise=True, name="Data"):
        """
        Initialize PCA analysis.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input data matrix
        n_components : int
            Number of principal components to compute
        normalise : bool
            Whether to standardize the data before PCA
        name : str
            Name for this dataset (used in plots)
        """
        self.df = df
        self.n_components = n_components
        self.normalise = normalise
        self.name = name
        
        # Apply PCA
        self._apply_pca()
        
        # Create formatted labels
        self._create_labels()
    
    def _apply_pca(self):
        """Apply PCA to the input data and store results."""
        # Standardize the features (columns)
        if self.normalise:
            self.scaler = StandardScaler()
            normalised = self.scaler.fit_transform(self.df)
        else:
            normalised = self.df.values
        
        # Apply PCA
        self.pca = PCA(n_components=self.n_components)
        scores = self.pca.fit_transform(normalised)
        
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
            label="Scaled Scores",
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
        title=f'PCA Scores Comparison: {pca_X.name} vs {pca_Y.name}',
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
       title=f'Score Correlation Heatmap: {pca_X.name} vs {pca_Y.name}',
       xaxis_title=f'{pca_Y.name} Components',
       yaxis_title=f'{pca_X.name} Components',
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
        print('got here')
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
        title=dict(text=f'PCA Loadings Heatmap: {name}', y=y_title_pos),
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

def calculate_dk_values(X, Y, max_k_X=None, normalise=False):
    X = X.df
    Y = Y.df
    n, p = X.shape
    n_y, q = Y.shape
    if max_k_X is None:
        max_k_X = min(n, p)
    assert n == n_y, "X and Y must have the same number of rows"
    
    # Center the matrices
    X_centered = X - X.mean(axis=0)
    Y_centered = Y - Y.mean(axis=0)

    # Optionally normalise the columns
    if normalise:
        X_centered = X_centered / X.std(axis=0)
        Y_centered = Y_centered / Y.std(axis=0)

    # Compute SVDs
    U_X, S_X, Vt_X = np.linalg.svd(X_centered, full_matrices=False)
    U_Y, S_Y, Vt_Y = np.linalg.svd(Y_centered, full_matrices=False)
    
    # Frobenius norm of Y (denominator)
    Y_frob_norm_sq = np.linalg.norm(Y_centered, 'fro') ** 2
    
    # Maximum k value (minimum of n and q, but also limited by available singular values)
    max_k = min(n, q, len(S_Y))
    
    # Calculate D(k) for each k
    dk_values = np.zeros(max_k)

    # Truncate U_X to the first max_k_X columns
    U_X_truncate = U_X[:, :max_k_X]
    
    for k in range(1, max_k + 1):
        # Get first k columns of U_Y and first k singular values
        U_Y_k = U_Y[:, :k]
        S_Y_k = np.diag(S_Y[:k])
        
        # Calculate U_Y_k * S_Y_k
        U_Y_k_S_Y_k = U_Y_k @ S_Y_k
        
        # Calculate U_X * U_X^T * U_Y_k * S_Y_k
        projection_term = U_X_truncate @ (U_X_truncate.T @ U_Y_k_S_Y_k)
        
        # Calculate the difference
        difference = U_Y_k_S_Y_k - projection_term
        
        # Calculate Frobenius norm squared of the difference
        numerator = np.linalg.norm(difference, 'fro') ** 2
        
        # Calculate D(k)
        dk_values[k-1] = numerator / Y_frob_norm_sq
    
    return dk_values

def calculate_dk_dict(pca_X, pca_Y, max_k_X_values=None, normalise=True):
    """Calculate D(k) values over a dict"""
    if max_k_X_values is None:
        max_k_X_values = list(range(1, pca_X.n_components + 1))
    
    dk_dict = {}
    
    for max_k_X in max_k_X_values:
        dk_values = calculate_dk_values(pca_X, pca_Y, max_k_X=max_k_X, normalise=normalise)
        dk_dict[max_k_X] = dk_values
    
    return dk_dict

def plot_dk_dict(dk_dict, pca_X_name="X", pca_Y_name="Y", max_k_Y = 11):
    """
    Plot D(k) values for multiple max_k_X values with viridis colorscale.
    """
    # Convert dict to list for plotting
    data = []
    for max_k_X, dk_values in dk_dict.items():
        for k, dk in enumerate(dk_values, 1):
            data.append({'k': k, 'D(k)': dk, 'max_k_X': max_k_X})
    
    df = pd.DataFrame(data)
    # subset to max_k_Y
    df = df[df['k'] <= max_k_Y]
    
    fig = go.Figure()
    
    # Get unique max_k_X values
    max_k_X_values = sorted(df['max_k_X'].unique())
    
    # Get viridis colors
    import plotly.colors as pc
    viridis_colors = pc.sample_colorscale('viridis', len(max_k_X_values))
    
    # Add traces for each max_k_X
    for i, max_k_X in enumerate(max_k_X_values):
        df_subset = df[df['max_k_X'] == max_k_X]
        fig.add_trace(go.Scatter(
            x=df_subset['k'],
            y=df_subset['D(k)'],
            mode='lines+markers',
            name=f'max_k_X={max_k_X}',
            line=dict(width=2, color=viridis_colors[i]),
            marker=dict(size=4)
        ))

    # Update y axis range
    #fig.update_xaxes(range=[0, max_k_Y])
    
    fig.update_layout(
        title=f'D(k_Y) vs k_Y for different max_k_X values: {pca_X_name} → {pca_Y_name}',
        xaxis_title='k_Y (number of singular vectors of Y)',
        yaxis_title='D(k)',
        width=800,
        height=500,
    )
    
    return fig


