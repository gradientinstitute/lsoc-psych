import os
import pandas as pd
import numpy as np
import glob
from tqdm import tqdm
from sklearn.decomposition import PCA, SparsePCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pickle
from io import StringIO
import openpyxl
import csv

### LOAD DATA ###

def load_token_mapping(experiment_name, dataset_name):
    """
    Load token mapping from the tokens pickle file.
    
    Args:
        experiment_name (str): Name of the experiment (e.g., "EXP000")
        dataset_name (str): Name of the dataset (e.g., "dm_mathematics")
    
    Returns:
        dict: Dictionary mapping context_id and token_position to actual token values
    """
    # Construct the path to the tokens pickle file
    tokens_path = f"/Users/liam/quests/lsoc-psych/datasets/experiments/{experiment_name}/trajectories/tokens/{dataset_name}_tokens.pkl"
    
    # Load the pickle file
    try:
        with open(tokens_path, 'rb') as f:
            token_data = pickle.load(f)
    except Exception as e:
        print(f"Error loading token file: {e}")
        return None
    
    # Create a lookup dictionary for token mapping
    token_mapping = {}
    
    # Process based on the structure of the token data
    # This structure might need adjustment based on the actual format
    for context_id, context_data in token_data.items():
        for token_idx, token_value in enumerate(context_data):
            # Create a key in format that matches your column names
            key = f"context_{context_id}_pos_{token_idx}"
            token_mapping[key] = token_value
    
    return token_mapping


def load_trajectory_data(experiment_name, dataset_name, model_sizes=None, column_fraction=1.0):
    """
    Load trajectory data for a given dataset name and experiment name.
    
    Args:
        experiment_name (str): Name of the experiment (e.g., "EXP000")
        dataset_name (str): Name of the dataset
        model_sizes (list, optional): List of model sizes to load. If None, loads all available.
        column_fraction (float, optional): Fraction of columns to load (between 0 and 1).
                                          Helps with loading large files more quickly.
    
    Returns:
        dict: Dictionary where keys are model sizes and values are the corresponding dataframes
    """
    # Define the base path for trajectories
    base_path = f"/Users/liam/quests/lsoc-psych/datasets/experiments/{experiment_name}/trajectories/csv"
    
    # Find all available model sizes for the dataset if not specified
    if model_sizes is None:
        pattern = os.path.join(base_path, f"*_{dataset_name}.csv")
        file_paths = glob.glob(pattern)
        model_sizes = [os.path.basename(file_path).split('_')[0] for file_path in file_paths]
        
    print(model_sizes)    
    # Dictionary to store dataframes for each model size
    trajectory_data = {}
    
    # Sample columns if column_fraction < 1.0
    sampled_columns = None
    
    # Load data for each model size
    for model_size in tqdm(model_sizes, desc="Loading trajectory data"):
        file_path = os.path.join(base_path, f"{model_size}_{dataset_name}.csv")
        
        if not os.path.exists(file_path):
            print(f"Warning: File not found for model size {model_size}, dataset {dataset_name}")
            continue
        
        # For the first model, determine columns to sample if needed
        if column_fraction < 1.0 and sampled_columns is None:
            # Read just the header to get column names
            # header_df = pd.read_csv(file_path, nrows=0)
            # all_columns = header_df.columns.tolist()
            with open(file_path, 'r') as f:
                all_columns = next(csv.reader(f))
            
            # Always include 'step' column
            non_step_columns = [col for col in all_columns if col != 'step']
            
            # Calculate number of columns to sample
            num_cols_to_sample = max(1, int(len(non_step_columns) * column_fraction))
            
            # Randomly sample columns
            # np.random.seed(42)  # For reproducibility
            # sampled_non_step_columns = np.random.choice(non_step_columns, 
            #                                            size=num_cols_to_sample, 
            #                                            replace=False)
            # Get first num_cols_to_sample columns
            sampled_non_step_columns = non_step_columns[:num_cols_to_sample]
            
            # Create final list of columns to load (step + sampled)
            sampled_columns = ['step'] + sampled_non_step_columns
            
            print(f"Loading {len(sampled_columns)-1} out of {len(non_step_columns)} columns")
        
        # Load the data, using sampled columns if applicable
        if column_fraction < 1.0:
            df = pd.read_csv(file_path, usecols=sampled_columns)
        else:
            df = pd.read_csv(file_path)
        
        # Remove columns with NaN values
        cols_with_nan = df.columns[df.isna().any()].tolist()
        if len(cols_with_nan) > 0:
            print(f"Removing {len(cols_with_nan)} columns with NaN values from model {model_size}")
            df = df.drop(columns=cols_with_nan)
        
        trajectory_data[model_size] = df

    # Helper function to convert model size to numeric value for sorting
    def model_size_to_numeric(size_str):
        if 'm' in size_str:
            return float(size_str.replace('m', '')) * 1e6
        elif 'b' in size_str:
            return float(size_str.replace('b', '')) * 1e9
        else:
            try:
                return float(size_str)
            except ValueError:
                return 0  # Default for unknown format
    
    # Sort model sizes by numeric value
    sorted_model_sizes = sorted(trajectory_data.keys(), key=model_size_to_numeric)
    
    # Create a new ordered dictionary
    ordered_trajectory_data = {model_size: trajectory_data[model_size] for model_size in sorted_model_sizes}
    
        
    return ordered_trajectory_data

### APPLY TRAJECTORY PCA ###

class TrajectoryPCA:
    """
    Class for performing PCA on concatenated trajectory data across multiple model sizes.
    
    This class handles:
    1. Concatenating trajectory data from different model sizes
    2. Applying PCA to the concatenated data
    3. Optionally applying sparse PCA to the residual matrix after regular PCA
    4. Transforming the original trajectories into the PCA space (both regular and sparse components)
    5. Storing the transformed trajectories back in the original data structure
    """
    
    def __init__(self, trajectory_data: Dict[str, pd.DataFrame], start_step, 
                 n_components: int = 10, n_sparse_components: int = 0, scale: bool = False,
                 sparse_pca_params: Optional[Dict] = None, run_at_init: bool = False,
                 dataset_name=None):
        """
        Initialize TrajectoryPCA with trajectory data.
        
        Args:
            trajectory_data (dict): Dictionary where keys are model sizes and 
                                    values are the corresponding dataframes
            start_step: Starting step to include in the analysis
            n_components (int): Number of regular PCA components to use
            n_sparse_components (int): Number of sparse PCA components to extract from residuals
            scale (bool): Whether to standardize the data before PCA
            sparse_pca_params (dict, optional): Parameters for the sparse PCA
            run_at_init (bool): Whether to run the PCA pipeline during initialization
        """
        # Input data and parameters
        self.trajectory_data = trajectory_data
        self.start_step = start_step
        self.n_components = n_components
        self.n_sparse_components = n_sparse_components
        self.scale = scale
        self.sparse_pca_params = sparse_pca_params or {}
        self.dataset_name = dataset_name
        
        # Filter data by start_step
        for model_size in self.trajectory_data:
            self.trajectory_data[model_size] = self.trajectory_data[model_size][
                self.trajectory_data[model_size]['step'] >= start_step
            ]

        # Model attributes
        self.model_sizes = list(trajectory_data.keys())
        self.pca = None
        self.sparse_pca = None
        self.scaler = None
        
        # Data containers
        self.common_columns = None
        self.concatenated_matrix = None
        self.raw_concatenated_matrix = None
        self.pca_residual_matrix = None
        self.row_indices = None
        
        # Run pipeline at initialization if requested
        if run_at_init:
            self.run_pca_pipeline()
        
    def find_common_columns(self) -> List[str]:
        """
        Find columns that are common across all model size dataframes.
        
        Returns:
            list: List of common column names
        """
        if not self.model_sizes:
            raise ValueError("No model sizes found in trajectory data")
            
        # Start with all columns from the first model
        common_cols = set(self.trajectory_data[self.model_sizes[0]].columns)
        
        # Intersect with columns from other models
        for model_size in self.model_sizes[1:]:
            model_cols = set(self.trajectory_data[model_size].columns)
            common_cols = common_cols.intersection(model_cols)
        
        # Convert back to list and remove 'step' if it exists (we'll handle it separately)
        common_cols = list(common_cols)
        if 'step' in common_cols:
            common_cols.remove('step')
            
        if not common_cols:
            raise ValueError("No common columns found across model sizes")
            
        print(f"Found {len(common_cols)} common columns across all {len(self.model_sizes)} model sizes")
        self.common_columns = common_cols
        return common_cols
    
    def concatenate_trajectories(self) -> np.ndarray:
        """
        Concatenate trajectories from all model sizes into a single matrix.
        
        Returns:
            np.ndarray: Concatenated matrix where rows are model checkpoints
                        across all model sizes
        """
        if self.common_columns is None:
            self.find_common_columns()
            
        all_data = []
        row_indices = {}
        start_idx = 0
        
        for model_size in self.model_sizes:
            df = self.trajectory_data[model_size]
            # Extract just the common columns
            model_data = df[self.common_columns].values
            
            # Store the row indices for this model size
            end_idx = start_idx + len(model_data)
            row_indices[model_size] = (start_idx, end_idx)
            start_idx = end_idx
            
            all_data.append(model_data)
            
        # Concatenate all model data
        concatenated = np.vstack(all_data)
        
        self.concatenated_matrix = concatenated
        self.raw_concatenated_matrix = concatenated.copy()  # Store the original unmodified matrix
        self.row_indices = row_indices
        
        print(f"Concatenated matrix shape: {concatenated.shape}")
        return concatenated
    
    def fit_pca(self) -> Tuple[PCA, Optional[SparsePCA]]:
        """
        Fit PCA on the concatenated trajectory data, followed by sparse PCA on residuals if requested.
        
        Returns:
            Tuple[PCA, Optional[SparsePCA]]: Fitted PCA and SparsePCA objects (if applicable)
        """
        if self.concatenated_matrix is None:
            self.concatenate_trajectories()
            
        # Scale the data if required
        if self.scale:
            self.scaler = StandardScaler()
            scaled_data = self.scaler.fit_transform(self.concatenated_matrix)
        else:
            scaled_data = self.concatenated_matrix  # PCA will handle centering internally
            
        # Fit regular PCA
        self.pca = PCA(n_components=self.n_components)
        pca_transformed = self.pca.fit_transform(scaled_data)
        
        # Print explained variance for regular PCA
        explained_var = self.pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)
        
        print(f"Regular PCA: Top {self.n_components} components explain {cumulative_var[-1]*100:.2f}% of variance")
        print(f"Individual explained variance: {explained_var}")
        
        # Calculate residuals after regular PCA
        if self.n_sparse_components > 0:
            # Calculate residuals using inverse_transform
            pca_transformed = self.pca.transform(scaled_data)
            reconstructed_data = self.pca.inverse_transform(pca_transformed)
            self.pca_residual_matrix = scaled_data - reconstructed_data
            
            print(f"Residual matrix shape after regular PCA: {self.pca_residual_matrix.shape}")
            
            # Default sparse PCA parameters
            default_sparse_params = {
                'alpha': 1.0,  # L1 penalty parameter
                'ridge_alpha': 0.01,  # Ridge penalty parameter
                'max_iter': 1000,
                'tol': 1e-6,
                'random_state': 42
            }
            
            # Update with user-provided parameters
            sparse_params = {**default_sparse_params, **self.sparse_pca_params}
            
            # Fit sparse PCA on residuals
            self.sparse_pca = SparsePCA(n_components=self.n_sparse_components, **sparse_params)
            sparse_transformed = self.sparse_pca.fit_transform(self.pca_residual_matrix)
            
            # Calculate sparsity of components
            component_sparsity = self.get_sparse_component_sparsity()
                
            print(f"Sparse PCA: {self.n_sparse_components} components extracted from residuals")
            print(f"Sparsity of components (fraction of zero values): {component_sparsity}")
        else:
            self.sparse_pca = None
            
        return self.pca, self.sparse_pca
    
    def transform_trajectories(self) -> Dict[str, pd.DataFrame]:
        """
        Transform the original trajectories into the PCA space (both regular and sparse if applicable)
        and store them back in the trajectory data dictionary.
        
        Returns:
            dict: Dictionary of transformed trajectories for each model size
        """
        if self.pca is None:
            raise ValueError("PCA not fitted yet. Call fit_pca() first.")
            
        transformed_data = {}
        
        for model_size in self.model_sizes:
            df = self.trajectory_data[model_size]
            
            # Get the data for this model size
            model_data = df[self.common_columns].values
            
            # Scale if needed
            if self.scaler is not None:
                processed_data = self.scaler.transform(model_data)
            else:
                processed_data = model_data  # PCA.transform will handle centering
                
            # Transform using regular PCA
            pca_transformed = self.pca.transform(processed_data)
            
            # Create column names for regular PCA components
            pca_component_cols = [f"PC{i+1}" for i in range(pca_transformed.shape[1])]
            
            # Initialize transformed DataFrame with regular PCA components
            transformed_df = pd.DataFrame(pca_transformed, columns=pca_component_cols)
            
            # Add sparse PCA components if applicable
            if self.sparse_pca is not None:
                # Calculate residuals using inverse_transform
                reconstructed_data = self.pca.inverse_transform(pca_transformed)
                model_residuals = processed_data - reconstructed_data
                
                # Transform residuals using sparse PCA
                sparse_transformed = self.sparse_pca.transform(model_residuals)
                
                # Create column names for sparse PCA components
                sparse_component_cols = [f"SPC{i+1}" for i in range(sparse_transformed.shape[1])]
                
                # Add sparse components to DataFrame
                sparse_df = pd.DataFrame(sparse_transformed, columns=sparse_component_cols)
                transformed_df = pd.concat([transformed_df, sparse_df], axis=1)
            
            # Add the step column if it exists
            if 'step' in df.columns:
                transformed_df['step'] = df['step'].values
                
            # Store the transformed data
            transformed_key = f"{model_size}_transformed"
            self.trajectory_data[transformed_key] = transformed_df
            transformed_data[model_size] = transformed_df
            
        return transformed_data
    
    def run_pca_pipeline(self) -> Dict[str, pd.DataFrame]:
        """
        Run the complete PCA pipeline using the instance parameters.
        
        Returns:
            dict: Dictionary of transformed trajectories for each model size
        """
        self.find_common_columns()
        self.concatenate_trajectories()
        self.fit_pca()
        return self.transform_trajectories()

    def get_sparse_component_sparsity(self):
        """
        Calculate the sparsity of each sparse PCA component.
        
        Returns:
            list: List of sparsity values (fraction of zero values) for each component
        """
        if self.sparse_pca is None:
            return None
            
        component_sparsity = []
        for component in self.sparse_pca.components_:
            non_zero = np.count_nonzero(component)
            sparsity = 1.0 - (non_zero / len(component))
            component_sparsity.append(sparsity)
            
        return component_sparsity
    
    def get_sparse_component_variance(self, n_samples=1000):
        """
        Estimate the variance explained by sparse PCA components in the residual space.
        
        Args:
            n_samples (int): Number of samples to use for variance estimation
            
        Returns:
            np.ndarray: Array of explained variance ratios
        """
        if self.sparse_pca is None or self.pca_residual_matrix is None:
            return None
        
        # Use a subset of samples if the residual matrix is very large
        if self.pca_residual_matrix.shape[0] > n_samples:
            indices = np.random.choice(self.pca_residual_matrix.shape[0], n_samples, replace=False)
            residual_subset = self.pca_residual_matrix[indices]
        else:
            residual_subset = self.pca_residual_matrix
        
        # Calculate total variance in residual space
        total_variance = np.var(residual_subset, axis=0).sum()
        
        # Transform the subset using sparse PCA
        transformed = self.sparse_pca.transform(residual_subset)
        
        # Calculate variance explained by each component
        component_variances = []
        for i in range(transformed.shape[1]):
            # Project back to the original feature space
            component_projection = np.outer(transformed[:, i], self.sparse_pca.components_[i])
            component_var = np.var(component_projection, axis=0).sum()
            component_variances.append(component_var)
        
        # Convert to explained variance ratio
        explained_variance_ratio = np.array(component_variances) / total_variance
        
        return explained_variance_ratio

# Helper function to get colors from viridis colormap
def get_viridis_colors(n):
    """
    Generate n colors from the viridis colormap with yellow for largest value
    and purple for smallest.
    
    Args:
        n (int): Number of colors to generate
        
    Returns:
        list: List of colors in hex format
    """
    return px.colors.sample_colorscale("viridis", np.linspace(0, 1, n))

### PLOT PCA ###

class TrajectoryPlotter:
    """
    Class for plotting PCA trajectories with consistent colors across visualizations.
    Works with TrajectoryPCA class.
    """
    
    def __init__(self, pca_handler):
        """
        Initialize TrajectoryPlotter with a TrajectoryPCA instance.
        
        Args:
            pca_handler: TrajectoryPCA instance with fitted PCA
        """
        self.pca_handler = pca_handler
        self.model_sizes = pca_handler.model_sizes
        
        # Get model size ordering for color assignment
        # We want smallest models to be purple and largest to be yellow
        self.model_order = self._get_model_size_order()
        
        # Assign colors from viridis colormap
        colors = get_viridis_colors(len(self.model_order))
        
        # Create color mapping for models
        self.model_colors = {}
        for i, model in enumerate(self.model_order):
            self.model_colors[model] = colors[i]
    
    def _get_model_size_order(self):
        """
        Get model sizes in order from smallest to largest.
        
        Returns:
            list: Sorted list of model sizes
        """
        # Helper function to convert model size to numeric value for sorting
        def model_size_to_numeric(size_str):
            if 'm' in size_str.lower():
                return float(size_str.lower().replace('m', '')) * 1e6
            elif 'b' in size_str.lower():
                return float(size_str.lower().replace('b', '')) * 1e9
            else:
                try:
                    return float(size_str)
                except ValueError:
                    return 0  # Default for unknown format
        
        # Sort model sizes from smallest to largest
        return sorted(self.model_sizes, key=model_size_to_numeric)
    
    def plot_pca_components(self, 
                            pc_x: int = 1, 
                            pc_y: int = 2, 
                            title: str = None,
                            width: int = 900, 
                            height: int = 600,
                            show_legend: bool = True,
                            line_width: int = 2,
                            markers: bool = True,
                            marker_size: int = 6):
        """
        Plot two PCA components against each other using Plotly.
        
        Args:
            pc_x (int): PCA component number for x-axis (1-based)
            pc_y (int): PCA component number for y-axis (1-based)
            title (str, optional): Plot title
            width (int): Plot width in pixels
            height (int): Plot height in pixels
            show_legend (bool): Whether to show the legend
            line_width (int): Width of trajectory lines
            markers (bool): Whether to show markers
            marker_size (int): Size of markers if shown
            
        Returns:
            plotly.graph_objects.Figure: Plotly figure
        """
        # Check if PCA has been fitted
        if self.pca_handler.pca is None:
            raise ValueError("PCA not fitted yet. Call pca_handler.fit_pca() first.")
        
        # Construct column names for the PCA components
        col_x = f"PC{pc_x}"
        col_y = f"PC{pc_y}"
        
        # Create a figure
        fig = go.Figure()
        
        # Add traces for each model size
        for model_size in self.model_order:
            # Get transformed data
            transformed_key = f"{model_size}_transformed"
            if transformed_key not in self.pca_handler.trajectory_data:
                continue
                
            df = self.pca_handler.trajectory_data[transformed_key]
            
            # Check if required columns exist
            if col_x not in df.columns or col_y not in df.columns:
                continue
            
            # Add trace for this model
            marker_dict = dict(size=marker_size) if markers else dict(size=0)
            
            # Create hover text including step information if available
            hover_text = None
            if 'step' in df.columns:
                hover_text = [f"Step: {step}<br>{col_x}: {x:.4f}<br>{col_y}: {y:.4f}" 
                             for step, x, y in zip(df['step'], df[col_x], df[col_y])]
            
            fig.add_trace(go.Scatter(
                x=df[col_x], 
                y=df[col_y],
                mode='lines+markers' if markers else 'lines',
                name=f"Model {model_size}",
                line=dict(color=self.model_colors[model_size], width=line_width),
                marker=marker_dict,
                text=hover_text,
                hoverinfo='text' if hover_text is not None else 'x+y+name'
            ))
        
        # Update layout
        if title is None:
            title = f"PCA Components {pc_x} vs {pc_y}"
            
        fig.update_layout(
            title=title,
            xaxis_title=f"Principal Component {pc_x}",
            yaxis_title=f"Principal Component {pc_y}",
            width=width,
            height=height,
            showlegend=show_legend,
            template='plotly_white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Add grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        
        return fig
    
    def plot_multiple_pca_components_vs_step(self,
                                       title: str = "PCA Components vs Training Steps",
                                       width: int = 900,
                                       height_per_component: int = 250,
                                       show_legend: bool = True,
                                       line_width: int = 2,
                                       markers: bool = False,
                                       marker_size: int = 6,
                                       min_step=0,
                                       shared_xaxis: bool = False):
        """
        Plot multiple PCA components and sparse PCA components against training steps as vertically stacked subplots.
        
        Args:
            num_pca_components (int): Number of regular PCA components to plot (starting from PC1)
            num_sparse_components (int): Number of sparse PCA components to plot (starting from SPC1)
            title (str): Overall plot title
            width (int): Plot width in pixels
            height_per_component (int): Height per component subplot in pixels
            show_legend (bool): Whether to show the legend
            line_width (int): Width of trajectory lines
            markers (bool): Whether to show markers
            marker_size (int): Size of markers if shown
            min_step: Minimum step to include in the plot
            shared_xaxis (bool): Whether to share x-axis across subplots
            
        Returns:
            plotly.graph_objects.Figure: Plotly figure with subplots
        """
        # Check if PCA has been fitted
        if self.pca_handler.pca is None:
            raise ValueError("PCA not fitted yet. Call pca_handler.fit_pca() first.")
            
        # Limit number of regular PCA components to what's available
        num_pca_components = self.pca_handler.pca.n_components_

        # Get explained variance for regular PCA components
        pca_explained_var = self.pca_handler.pca.explained_variance_ratio_
        pc_titles = [f"PC{i+1} (E.V. {pca_explained_var[i]*100:.2f}%)" for i in range(num_pca_components)]
        
        # Handle sparse PCA components if available
        if self.pca_handler.sparse_pca is not None:
            num_sparse_components = self.pca_handler.sparse_pca.n_components_
            
            # Get sparsity for sparse components
            if hasattr(self.pca_handler, 'get_sparse_component_sparsity'):
                sparse_sparsity = self.pca_handler.get_sparse_component_sparsity()
                sparse_explained_variance = self.pca_handler.get_sparse_component_variance()
                spc_titles = [f"SPC{i+1} (Sparsity: {sparse_sparsity[i]*100:.2f}%, E.V. {sparse_explained_variance[i]*100:.2f}%)" for i in range(num_sparse_components)]
            else:
                # Fallback if sparsity info not available
                spc_titles = [f"SPC{i+1}" for i in range(num_sparse_components)]
        else:
            num_sparse_components = 0
            spc_titles = []
        
        # Calculate total number of components to plot
        total_components = num_pca_components + num_sparse_components
        subplot_titles = pc_titles + spc_titles

        # Subset steps by min_step
        for model_size in self.model_order:
            transformed_key = f"{model_size}_transformed"
            if transformed_key not in self.pca_handler.trajectory_data:
                continue
            df = self.pca_handler.trajectory_data[transformed_key]
            df = df[df['step'] >= min_step]
            self.pca_handler.trajectory_data[transformed_key] = df
        
        # Calculate total height
        total_height = height_per_component * total_components
        
        # Create a subplot structure
        fig = go.Figure()
        
        fig = make_subplots(
            rows=total_components,
            cols=1,
            shared_xaxes=shared_xaxis,
            vertical_spacing=0.05,
            subplot_titles=subplot_titles
        )
        
        # Add traces for each regular PCA component and model size
        for pc_idx in range(num_pca_components):
            pc = pc_idx + 1  # 1-based PC numbering
            col_pc = f"PC{pc}"
            
            for model_size in self.model_order:
                # Get transformed data
                transformed_key = f"{model_size}_transformed"
                if transformed_key not in self.pca_handler.trajectory_data:
                    continue
                    
                df = self.pca_handler.trajectory_data[transformed_key]
                
                # Check if required columns exist
                if col_pc not in df.columns or 'step' not in df.columns:
                    continue
                
                # Create hover text
                hover_text = [f"Model: {model_size}<br>Step: {step}<br>{col_pc}: {val:.4f}" 
                            for step, val in zip(df['step'], df[col_pc])]
                
                # Add trace for this model and component
                marker_dict = dict(size=marker_size) if markers else dict(size=0)
                
                fig.add_trace(
                    go.Scatter(
                        x=df['step'], 
                        y=df[col_pc],
                        mode='lines+markers' if markers else 'lines',
                        name=f"Model {model_size}",
                        line=dict(color=self.model_colors[model_size], width=line_width),
                        marker=marker_dict,
                        text=hover_text,
                        hoverinfo='text',
                        showlegend=True if pc_idx == 0 else False  # Only show legend once
                    ),
                    row=pc_idx+1, 
                    col=1
                )
                fig.update_yaxes(title_text=f"PC{pc}", row=pc_idx+1, col=1)
        
        # Add traces for each sparse PCA component and model size
        for spc_idx in range(num_sparse_components):
            spc = spc_idx + 1  # 1-based SPC numbering
            col_spc = f"SPC{spc}"
            plot_row = num_pca_components + spc_idx + 1  # Plot after regular PCs
            
            for model_size in self.model_order:
                # Get transformed data
                transformed_key = f"{model_size}_transformed"
                if transformed_key not in self.pca_handler.trajectory_data:
                    continue
                    
                df = self.pca_handler.trajectory_data[transformed_key]
                
                # Check if required columns exist
                if col_spc not in df.columns or 'step' not in df.columns:
                    continue
                
                # Create hover text
                hover_text = [f"Model: {model_size}<br>Step: {step}<br>{col_spc}: {val:.4f}" 
                            for step, val in zip(df['step'], df[col_spc])]
                
                # Add trace for this model and component
                marker_dict = dict(size=marker_size) if markers else dict(size=0)
                
                fig.add_trace(
                    go.Scatter(
                        x=df['step'], 
                        y=df[col_spc],
                        mode='lines+markers' if markers else 'lines',
                        name=f"Model {model_size}",
                        line=dict(color=self.model_colors[model_size], width=line_width),
                        marker=marker_dict,
                        text=hover_text,
                        hoverinfo='text',
                        showlegend=False  # No need to show legend again for sparse components
                    ),
                    row=plot_row, 
                    col=1
                )
                fig.update_yaxes(title_text=f"SPC{spc}", row=plot_row, col=1)
        
        # Update layout
        fig.update_layout(
            title=title,
            width=width,
            height=total_height,
            template='plotly_white',
            showlegend=show_legend,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.01,
                xanchor="right",
                x=1
            ),
            margin=dict(l=20, r=20, t=150, b=20)
        )


        # Update x-axis titles (only for the bottom subplot if shared)
        if shared_xaxis:
            fig.update_xaxes(title="Training Steps", row=total_components, col=1)
        else:
            for i in range(total_components):
                fig.update_xaxes(title="Training Steps", row=i+1, col=1)
        
        # Add grid to all subplots
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', type='log')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        
        return fig
    
    def plot_pca_component_vs_step(self, 
                                  pc: int = 1, 
                                  title: str = None,
                                  width: int = 900, 
                                  height: int = 600,
                                  show_legend: bool = True,
                                  line_width: int = 2,
                                  markers: bool = False,
                                  marker_size: int = 6):
        """
        Plot a PCA component against training steps.
        
        Args:
            pc (int): PCA component number (1-based)
            title (str, optional): Plot title
            width (int): Plot width in pixels
            height (int): Plot height in pixels
            show_legend (bool): Whether to show the legend
            line_width (int): Width of trajectory lines
            markers (bool): Whether to show markers
            marker_size (int): Size of markers if shown
            
        Returns:
            plotly.graph_objects.Figure: Plotly figure
        """
        # Check if PCA has been fitted
        if self.pca_handler.pca is None:
            raise ValueError("PCA not fitted yet. Call pca_handler.fit_pca() first.")
        
        # Construct column name for the PCA component
        col_pc = f"PC{pc}"
        
        # Create a figure
        fig = go.Figure()
        
        # Add traces for each model size
        for model_size in self.model_order:
            # Get transformed data
            transformed_key = f"{model_size}_transformed"
            if transformed_key not in self.pca_handler.trajectory_data:
                continue
                
            df = self.pca_handler.trajectory_data[transformed_key]
            
            # Check if required columns exist
            if col_pc not in df.columns or 'step' not in df.columns:
                continue
            
            # Add trace for this model
            marker_dict = dict(size=marker_size) if markers else dict(size=0)
            
            fig.add_trace(go.Scatter(
                x=df['step'], 
                y=df[col_pc],
                mode='lines+markers' if markers else 'lines',
                name=f"Model {model_size}",
                line=dict(color=self.model_colors[model_size], width=line_width),
                marker=marker_dict
            ))
        
        # Update layout
        if title is None:
            title = f"PCA Component {pc} vs Training Steps"
            
        fig.update_layout(
            title=title,
            xaxis_title="Training Steps",
            yaxis_title=f"Principal Component {pc}",
            width=width,
            height=height,
            showlegend=show_legend,
            template='plotly_white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Add grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', type='log')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        
        return fig
    
    def plot_explained_variance(self, 
                               n_components: int = None,
                               width: int = 900, 
                               height: int = 600,
                               title: str = "PCA Explained Variance"):
        """
        Plot the explained variance for PCA components using Plotly.
        
        Args:
            n_components (int, optional): Number of components to display
            width (int): Plot width in pixels
            height (int): Plot height in pixels
            title (str): Plot title
            
        Returns:
            plotly.graph_objects.Figure: Plotly figure
        """
        # Check if PCA has been fitted
        if self.pca_handler.pca is None:
            raise ValueError("PCA not fitted yet. Call pca_handler.fit_pca() first.")
        
        # Get explained variance
        explained_var = self.pca_handler.pca.explained_variance_ratio_
        
        # Set number of components to display
        if n_components is None:
            n_components = len(explained_var)
        else:
            n_components = min(n_components, len(explained_var))
            
        explained_var = explained_var[:n_components]
        cumulative_var = np.cumsum(explained_var)
        component_indices = np.arange(1, n_components + 1)
        
        # Create figure with secondary y-axis
        fig = go.Figure()
        
        # Add individual explained variance as bars
        fig.add_trace(go.Bar(
            x=component_indices,
            y=explained_var,
            name="Individual",
            marker_color="royalblue",
            opacity=0.7
        ))
        
        # Add cumulative explained variance as line
        fig.add_trace(go.Scatter(
            x=component_indices,
            y=cumulative_var,
            name="Cumulative",
            mode="lines+markers",
            marker=dict(size=8, color="firebrick"),
            line=dict(width=2, color="firebrick"),
            yaxis="y2"
        ))
        
        # Add 90% threshold line
        fig.add_trace(go.Scatter(
            x=component_indices,
            y=[0.9] * n_components,
            name="90% Explained",
            mode="lines",
            line=dict(width=2, color="green", dash="dash"),
            yaxis="y2"
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis=dict(
                title="Principal Component",
                tickmode="linear",
                tick0=1,
                dtick=1
            ),
            yaxis=dict(
                title="Explained Variance Ratio",
                range=[0, max(explained_var) * 1.1],
                side="left"
            ),
            yaxis2=dict(
                title="Cumulative Explained Variance",
                range=[0, 1],
                side="right",
                overlaying="y",
                tickmode="linear",
                tick0=0,
                dtick=0.1
            ),
            width=width,
            height=height,
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode="x unified"
        )
        
        # Add grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        
        return fig
    
    def plot_3d_pca_components(self,
                              pc_x: int = 1,
                              pc_y: int = 2, 
                              pc_z: int = 3,
                              title: str = None,
                              width: int = 900,
                              height: int = 700,
                              line_width: int = 3,
                              markers: bool = True,
                              marker_size: int = 4):
        """
        Plot three PCA components in 3D using Plotly.
        
        Args:
            pc_x (int): PCA component number for x-axis (1-based)
            pc_y (int): PCA component number for y-axis (1-based)
            pc_z (int): PCA component number for z-axis (1-based)
            title (str, optional): Plot title
            width (int): Plot width in pixels
            height (int): Plot height in pixels
            line_width (int): Width of trajectory lines
            markers (bool): Whether to show markers
            marker_size (int): Size of markers if shown
            
        Returns:
            plotly.graph_objects.Figure: Plotly figure
        """
        # Check if PCA has been fitted
        if self.pca_handler.pca is None:
            raise ValueError("PCA not fitted yet. Call pca_handler.fit_pca() first.")
        
        # Construct column names for the PCA components
        col_x = f"PC{pc_x}"
        col_y = f"PC{pc_y}"
        col_z = f"PC{pc_z}"
        
        # Create a figure
        fig = go.Figure()
        
        # Add traces for each model size
        for model_size in self.model_order:
            # Get transformed data
            transformed_key = f"{model_size}_transformed"
            if transformed_key not in self.pca_handler.trajectory_data:
                continue
                
            df = self.pca_handler.trajectory_data[transformed_key]
            
            # Check if required columns exist
            if col_x not in df.columns or col_y not in df.columns or col_z not in df.columns:
                continue
            
            # Add trace for this model
            marker_dict = dict(size=marker_size) if markers else dict(size=0)
            
            fig.add_trace(go.Scatter3d(
                x=df[col_x], 
                y=df[col_y],
                z=df[col_z],
                mode='lines+markers' if markers else 'lines',
                name=f"Model {model_size}",
                line=dict(color=self.model_colors[model_size], width=line_width),
                marker=marker_dict
            ))
        
        # Update layout
        if title is None:
            title = f"PCA Components {pc_x} vs {pc_y} vs {pc_z}"
            
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=f"Principal Component {pc_x}",
                yaxis_title=f"Principal Component {pc_y}",
                zaxis_title=f"Principal Component {pc_z}",
            ),
            width=width,
            height=height,
            template='plotly_white'
        )
        
        return fig

    def plot_raw_trajectory_heatmap(self,
                            token_mapping,
                            colorscale="Viridis",
                            title="Raw Trajectory Data Heatmap with Tokens",
                            width=1200,
                            height=None,
                            subplot_height=300,
                            shared_colorscale=True,
                            log_scale=False,
                            max_columns=100,
                            min_steps=0,
                            max_color=10):
        """
        Plot heatmaps of the raw trajectory data with token information in hover text.
        
        Args:
            token_mapping (dict): Dictionary mapping column names to token values
            colorscale (str): Color scale for the heatmap (e.g., "Viridis", "Reds")
            title (str): Overall plot title
            width (int): Width of the figure in pixels
            height (int): Total height of the figure in pixels (if None, calculated based on model count)
            subplot_height (int): Height for each subplot in pixels
            shared_colorscale (bool): Whether to use the same color scale range for all models
            log_scale (bool): Whether to use log scaling for the color values
            max_columns (int): Maximum number of columns to display (to prevent overcrowding)
            min_steps (int/float): Minimum step threshold to include in the visualization
            max_color (float): Maximum value for color scale capping
            
        Returns:
            plotly.graph_objects.Figure: Plotly figure with token hover information
        """
        
        # Check if we have trajectory data
        if not self.model_order:
            raise ValueError("No model data available")
            
        # Access the original (not transformed) trajectory data
        trajectory_data = self.pca_handler.trajectory_data
        
        # Determine number of models and calculate total height if not specified
        n_models = len(self.model_order)
        if height is None:
            height = subplot_height * n_models
            
        # Create subplot structure
        fig = make_subplots(
            rows=n_models,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.01,
            subplot_titles=[f"Model: {model_size}" for model_size in self.model_order]
        )
        
        # Find global min and max values if using shared colorscale
        global_min, global_max = None, None
        if shared_colorscale:
            all_values = []
            for model_size in self.model_order:
                if model_size not in trajectory_data:
                    continue
                    
                df = trajectory_data[model_size]
                
                # Filter steps based on threshold
                step_mask = df['step'] > min_steps
                if not any(step_mask):
                    print(f"Warning: No steps above {min_steps} for model {model_size}")
                    continue
                    
                value_cols = [col for col in df.columns if col != 'step']
                
                # Limit columns if there are too many
                if len(value_cols) > max_columns:
                    value_cols = value_cols[:max_columns]
                    
                # Filter data by steps
                filtered_data = df.loc[step_mask, value_cols].values
                    
                # For numerical stability with log scale
                if log_scale:
                    # Add a small positive value to zeros before taking log
                    data_array = filtered_data.copy()
                    data_array[data_array <= 0] = 1e-10  # Avoid log(0)
                    data_array = np.log10(data_array)
                else:
                    data_array = filtered_data
                    
                all_values.extend(data_array.flatten())
            
            if all_values:
                global_min, global_max = np.nanmin(all_values), np.nanmax(all_values)
        
        # Create a heatmap for each model
        for i, model_size in enumerate(self.model_order):
            if model_size not in trajectory_data:
                continue
                
            # Get the data
            df = trajectory_data[model_size]
            
            # Filter steps based on threshold
            step_mask = df['step'] > min_steps
            if not any(step_mask):
                print(f"Warning: No steps above {min_steps} for model {model_size}")
                continue
                
            # Get steps for y-axis (rows) and token columns for x-axis
            filtered_steps = df.loc[step_mask, 'step'].values
            value_cols = [col for col in df.columns if col != 'step']
            
            # Limit columns if there are too many
            if len(value_cols) > max_columns:
                print(f"Limiting display to {max_columns} columns out of {len(value_cols)} for model {model_size}")
                value_cols = value_cols[:max_columns]
            
            # Create column display names with token preview
            col_display_names = []
            for col_idx, col_name in enumerate(value_cols):
                # Get token text if available in the mapping
                token_text = token_mapping.get(col_name, "")
                # Create a short preview for display (first 8 chars)
                short_token = token_text[:8] + '...' if len(token_text) > 8 else token_text
                # Strip newlines for display
                short_token = short_token.replace('\n', '\\n')
                col_display_names.append(f"T{col_idx}: {short_token}")
            
            # Get the filtered data
            filtered_data = df.loc[step_mask, value_cols].values
            
            # Apply log scaling if requested
            if log_scale and np.any(filtered_data > 0):
                # Add a small positive value to zeros before taking log
                data_array = filtered_data.copy()
                data_array[data_array <= 0] = 1e-10  # Avoid log(0)
                data_array = np.log10(data_array)
                z_title = "Loss (Log10)"
            else:
                data_array = filtered_data
                z_title = "Value"
            
            # Set color scale range
            colorscale_args = {}
            if shared_colorscale and global_min is not None and global_max is not None:
                zmax = global_max if global_max < max_color else max_color
                colorscale_args = dict(zmin=global_min, zmax=zmax)
            
            # Create custom hover text with token information
            hover_text = []
            for step_idx, step in enumerate(filtered_steps):
                step_hover = []
                for col_idx, col_name in enumerate(value_cols):
                    # Get the value
                    value = data_array[step_idx, col_idx]
                    
                    # Get token text if available in the mapping
                    # print(col_name)
                    # print(token_mapping[col_name])
                    token_text = token_mapping.get(col_name, "Unknown")
                    # Handle potential newlines in tokens for hover display
                    token_text = token_text.replace('\n', '\\n')
                    
                    # Create hover text
                    text = (
                        f"Step: {step}<br>"
                        f"Column: {col_name}<br>"
                        f"Loss: {10**value:.4f}<br>"
                        f"Token: '{token_text}'"
                    )
                    step_hover.append(text)
                hover_text.append(step_hover)
            
            # Add heatmap for this model
            fig.add_trace(
                go.Heatmap(
                    z=data_array,  # steps as rows (y), tokens as columns (x)
                    x=col_display_names,  # token columns with preview
                    y=filtered_steps,  # filtered training steps
                    colorscale=colorscale,
                    colorbar=dict(
                        title=z_title,
                        # Only show the colorbar for the last model or if not shared
                        showticklabels=(i == n_models-1) if shared_colorscale else True,
                    ),
                    showscale=(i == n_models-1) if shared_colorscale else True,
                    hoverinfo="text",
                    text=hover_text,
                    **colorscale_args
                ),
                row=i+1,
                col=1
            )
            
            # Customize layout for this subplot
            fig.update_yaxes(title="Training Steps", row=i+1, col=1, type='log')
        
        # Set x-axis title only for the bottom subplot
        fig.update_xaxes(title="Token Features", row=n_models, col=1)
        
        # Update overall layout
        fig.update_layout(
            title=title,
            width=width,
            height=height,
            template="plotly_white",
            margin=dict(l=20, r=20, t=80, b=20)
        )
        
        return fig
    
    def plot_top_loading_columns(self,
                              pc_index: int = 1,
                              num_columns: int = 10,
                              title: str = None,
                              width: int = 1200,
                              height_per_subplot: int = 200,
                              show_legend: bool = True,
                              line_width: int = 2,
                              markers: bool = False,
                              marker_size: int = 6,
                              min_step=0,
                              token_mapping=None,
                              shared_yaxis: bool = False,
                              log_step_axis: bool = True):
        """
        Plot trajectories of top loading columns from a specific principal component.
        
        Args:
            pc_index (int): Principal component index (1-based)
            num_columns (int): Number of top loading columns to plot
            title (str, optional): Overall plot title
            width (int): Plot width in pixels
            height_per_subplot (int): Height per subplot in pixels
            show_legend (bool): Whether to show the legend
            line_width (int): Width of trajectory lines
            markers (bool): Whether to show markers
            marker_size (int): Size of markers if shown
            min_step (int): Minimum step to include in the plot
            token_mapping (dict, optional): Dictionary mapping column names to token values
            shared_yaxis (bool): Whether to use the same y-axis scale for all subplots
            log_step_axis (bool): Whether to use log scale for the step axis
            
        Returns:
            plotly.graph_objects.Figure: Plotly figure with subplots for top loading columns
        """
        # Check if PCA has been fitted
        if self.pca_handler.pca is None:
            raise ValueError("PCA not fitted yet. Call pca_handler.fit_pca() first.")
        
        # Get PCA loadings for the specified component
        pc_idx = pc_index - 1  # Convert to 0-based indexing
        if pc_idx >= self.pca_handler.pca.components_.shape[0]:
            raise ValueError(f"PC index {pc_index} exceeds available components ({self.pca_handler.pca.components_.shape[0]})")
        
        # Get the loadings for this component
        component_loadings = self.pca_handler.pca.components_[pc_idx]
        
        # Get the feature names (column names)
        feature_names = self.pca_handler.common_columns
        
        # Create a DataFrame with feature names and their loadings
        import pandas as pd
        loadings_df = pd.DataFrame({
            'Feature': feature_names,
            'Loading': component_loadings
        })
        
        # Sort by absolute loading value (to get most influential features)
        loadings_df['Abs_Loading'] = abs(loadings_df['Loading'])
        loadings_df = loadings_df.sort_values('Abs_Loading', ascending=False)
        
        # Get top N features
        top_columns = loadings_df.head(num_columns)['Feature'].tolist()
        top_loadings = loadings_df.head(num_columns)['Loading'].tolist()
        
        # Set up subplot grid
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        
        # Calculate total height
        total_height = height_per_subplot * min(num_columns, len(top_columns))
        
        # Create subplot titles with loading values
        subplot_titles = []
        for i, (col, loading) in enumerate(zip(top_columns, top_loadings)):
            # Add token info if available
            token_info = ""
            if token_mapping and col in token_mapping:
                # Format token for display (truncate if too long, handle newlines)
                token_text = token_mapping[col]
                token_text = token_text.replace('\n', '\\n')
                if len(token_text) > 20:
                    token_text = token_text[:17] + "..."
                token_info = f" - '{token_text}'"
            
            subplot_titles.append(f"{col}{token_info} (Loading: {loading:.4f})")
        
        # Create subplot layout
        fig = make_subplots(
            rows=len(top_columns),
            cols=1,
            #shared_xaxes=True,
            shared_yaxes=shared_yaxis,
            vertical_spacing=0.02,
            subplot_titles=subplot_titles
        )
        
        # Dictionary to track y-axis range for each subplot if using shared y-axis
        y_ranges = {}
        
        # Add traces for each feature and model size
        for i, feature in enumerate(top_columns):
            for model_size in self.model_order:
                # Get raw trajectory data
                if model_size not in self.pca_handler.trajectory_data:
                    continue
                    
                df = self.pca_handler.trajectory_data[model_size]
                
                # Check if required columns exist
                if feature not in df.columns or 'step' not in df.columns:
                    continue
                
                # Filter by minimum step
                df = df[df['step'] >= min_step]
                
                # Create hover text
                hover_text = [f"Model: {model_size}<br>Step: {step}<br>Value: {val:.6f}" 
                            for step, val in zip(df['step'], df[feature])]
                
                # Add trace for this model and feature
                marker_dict = dict(size=marker_size) if markers else dict(size=0)
                
                fig.add_trace(
                    go.Scatter(
                        x=df['step'], 
                        y=df[feature],
                        mode='lines+markers' if markers else 'lines',
                        name=f"Model {model_size}",
                        line=dict(color=self.model_colors[model_size], width=line_width),
                        marker=marker_dict,
                        text=hover_text,
                        hoverinfo='text',
                        showlegend=True if i == 0 else False  # Only show legend once
                    ),
                    row=i+1, 
                    col=1
                )
                
                # Track y-axis range for this subplot
                if shared_yaxis:
                    if i not in y_ranges:
                        y_ranges[i] = [float('inf'), float('-inf')]
                    y_min = min(df[feature])
                    y_max = max(df[feature])
                    y_ranges[i][0] = min(y_ranges[i][0], y_min)
                    y_ranges[i][1] = max(y_ranges[i][1], y_max)
        
        # Apply shared y-axis ranges if needed
        if shared_yaxis and y_ranges:
            global_min = min([r[0] for r in y_ranges.values()])
            global_max = max([r[1] for r in y_ranges.values()])
            for i in range(len(top_columns)):
                fig.update_yaxes(range=[global_min, global_max], row=i+1, col=1)
        
        # Update layout
        if title is None:
            explained_var = self.pca_handler.pca.explained_variance_ratio_[pc_idx]
            title = f"Top {num_columns} Loading Features for PC{pc_index} ({explained_var*100:.2f}% variance)"
        
        fig.update_layout(
            title=title,
            width=width,
            height=total_height,
            template='plotly_white',
            showlegend=show_legend,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.01,
                xanchor="right",
                x=1
            ),
            margin=dict(l=20, r=20, t=80, b=20)
        )
        
        # Update x-axis (only for the bottom subplot)
        fig.update_xaxes(
            title="Training Steps", 
            row=len(top_columns), 
            col=1,
            type='log' if log_step_axis else 'linear'
        )
        
        # Add grid to all subplots
        if log_step_axis:
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', type='log')
        else:
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        
        return fig
    
### OTHER DATA STUFF ###

def save_pc_loadings_excel(pca_handler, filename, num_components=None, top_n=20, 
                        include_sparse=True, token_mapping=None):
    """
    Save PC loadings to an Excel file with separate sheets for regular and sparse components.
    
    Args:
        pca_handler: TrajectoryPCA instance with fitted PCA
        filename (str): Output Excel file path
        num_components (int, optional): Number of principal components to include
        top_n (int): Number of top loading features per component
        include_sparse (bool): Whether to include sparse components
        token_mapping (dict, optional): Dictionary mapping column names to token values
        
    Returns:
        bool: Success status
    """

    # Check if PCA has been fitted
    if pca_handler.pca is None:
        raise ValueError("PCA not fitted yet. Call fit_pca() first.")
    
    # Create Excel writer
    with pd.ExcelWriter(filename) as writer:
        # Add summary sheet
        summary_data = []
        
        # Regular PCA summary
        explained_var = pca_handler.pca.explained_variance_ratio_
        
        if num_components is None:
            num_pca_components = len(explained_var)
        else:
            num_pca_components = min(num_components, len(explained_var))
            
        for i in range(num_pca_components):
            summary_data.append({
                "Component": f"PC{i+1}",
                "Type": "Regular",
                "Explained Variance (%)": explained_var[i] * 100,
                "Cumulative Variance (%)": np.sum(explained_var[:i+1]) * 100
            })
        
        # Get feature names
        feature_names = pca_handler.common_columns
        
        # Process regular PCA components
        loadings = pca_handler.pca.components_
        
        all_pc_data = []
        
        # For each regular PCA component
        for i in range(num_pca_components):
            # Get the loadings for this component
            component_loadings = loadings[i]
            
            # Create a DataFrame with feature names and their loadings
            loadings_df = pd.DataFrame({
                'Feature': feature_names,
                'Loading': component_loadings
            })
            
            # Sort by absolute loading value
            loadings_df['Abs_Loading'] = abs(loadings_df['Loading'])
            loadings_df = loadings_df.sort_values('Abs_Loading', ascending=False)
            
            # Get top N features
            top_loadings = loadings_df.head(top_n)[['Feature', 'Loading']].copy()
            
            # If token mapping provided, add token information
            if token_mapping:
                top_loadings['Token'] = top_loadings['Feature'].apply(
                    lambda x: token_mapping.get(x, "N/A")
                )
            
            # Add component information
            top_loadings['Component'] = f"PC{i+1}"
            top_loadings['Explained Variance (%)'] = explained_var[i] * 100
            
            all_pc_data.append(top_loadings)
        
        # Combine all PC data
        if all_pc_data:
            all_pc_df = pd.concat(all_pc_data, ignore_index=True)
            all_pc_df.to_excel(writer, sheet_name='Regular PCs', index=False)
        
        # Process sparse PCA components if available
        all_spc_data = []
        
        if pca_handler.sparse_pca is not None and include_sparse:
            sparse_loadings = pca_handler.sparse_pca.components_
            
            # Get sparsity
            sparse_sparsity = None
            if hasattr(pca_handler, 'get_sparse_component_sparsity'):
                sparse_sparsity = pca_handler.get_sparse_component_sparsity()
            
            # Get variance explained
            sparse_variance = None
            if hasattr(pca_handler, 'get_sparse_component_variance'):
                sparse_variance = pca_handler.get_sparse_component_variance()
            
            num_sparse_components = min(num_components if num_components else float('inf'), 
                                    sparse_loadings.shape[0])
            
            # Add to summary
            for i in range(num_sparse_components):
                summary_entry = {
                    "Component": f"SPC{i+1}",
                    "Type": "Sparse",
                }
                
                if sparse_sparsity is not None:
                    summary_entry["Sparsity (%)"] = sparse_sparsity[i] * 100
                
                if sparse_variance is not None and i < len(sparse_variance):
                    summary_entry["Explained Variance of Residuals (%)"] = sparse_variance[i] * 100
                
                summary_data.append(summary_entry)
            
            # For each sparse component
            for i in range(num_sparse_components):
                component_loadings = sparse_loadings[i]
                
                # Calculate sparsity for this component
                sparsity = 1.0 - (np.count_nonzero(component_loadings) / len(component_loadings))
                
                # Create DataFrame
                loadings_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Loading': component_loadings
                })
                
                # Sort by absolute loading
                loadings_df['Abs_Loading'] = abs(loadings_df['Loading'])
                loadings_df = loadings_df.sort_values('Abs_Loading', ascending=False)
                
                # Only include non-zero loadings
                non_zero_mask = loadings_df['Loading'] != 0
                non_zero_loadings = loadings_df[non_zero_mask]
                
                # Get top features
                top_loadings = non_zero_loadings.head(top_n)[['Feature', 'Loading']].copy()
                
                # Add token information if available
                if token_mapping:
                    top_loadings['Token'] = top_loadings['Feature'].apply(
                        lambda x: token_mapping.get(x, "N/A")
                    )
                
                # Add component information
                top_loadings['Component'] = f"SPC{i+1}"
                top_loadings['Sparsity (%)'] = sparsity * 100
                
                if sparse_variance is not None and i < len(sparse_variance):
                    top_loadings['Explained Variance of Residuals (%)'] = sparse_variance[i] * 100
                
                all_spc_data.append(top_loadings)
            
            # Combine all SPC data
            if all_spc_data:
                all_spc_df = pd.concat(all_spc_data, ignore_index=True)
                all_spc_df.to_excel(writer, sheet_name='Sparse PCs', index=False)
        
        # Write summary sheet
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"PC loadings saved to {filename}")
    return True


def print_enhanced_pc_loadings(pca_handler, num_components=None, top_n=10,            
                               include_sparse=True, 
                                token_mapping=None, save_to_file=None, max_token_length=30):
    """
    Generate well-formatted tables showing the top loading columns for each principal component
    with optional token information and explained variance.
    
    Args:
        pca_handler: TrajectoryPCA instance with fitted PCA
        num_components (int, optional): Number of principal components to show (default: all)
        top_n (int): Number of top loading features to show for each component
        include_sparse (bool): Whether to include sparse components
        token_mapping (dict, optional): Dictionary mapping column names to token values
        save_to_file (str, optional): If provided, save output to this file path
        max_token_length (int): Maximum length for token display
        
    Returns:
        str: Formatted text output
    """
    
    
    # Check if PCA has been fitted
    if pca_handler.pca is None:
        raise ValueError("PCA not fitted yet. Call fit_pca() first.")
    
    # Get the feature names
    feature_names = pca_handler.common_columns
    
    # String buffer to collect all output
    output_buffer = StringIO()
    
    # Function to write to both buffer and print to console
    def write_output(text):
        print(text)
        output_buffer.write(text + "\n")
    
    # Helper function to format token text
    def format_token(token_text, max_length=max_token_length):
        if token_text is None:
            return "N/A"
        # Replace newlines and tabs with visible representations
        token_text = token_text.replace('\n', '\\n').replace('\t', '\\t')
        # Truncate if too long
        if len(token_text) > max_length:
            return token_text[:max_length-3] + "..."
        return token_text
    
    # Process regular PCA components
    # Get the PCA components (loadings)
    loadings = pca_handler.pca.components_
    
    # Get the explained variance ratio for each component
    explained_var = pca_handler.pca.explained_variance_ratio_
    
    # Set number of components to display
    if num_components is None:
        num_pca_components = loadings.shape[0]
    else:
        num_pca_components = min(num_components, loadings.shape[0])
    
    # Header for regular PCA components
    header = "=" * 100
    write_output(header)
    write_output(f"REGULAR PCA COMPONENTS (Top {top_n} Features per Component)")
    write_output(header)
    
    # Print summary of explained variance
    write_output("\nEXPLAINED VARIANCE SUMMARY:")
    summary_data = []
    for i in range(num_pca_components):
        summary_data.append({
            "Component": f"PC{i+1}",
            "Explained Variance (%)": f"{explained_var[i]*100:.2f}%",
            "Cumulative Variance (%)": f"{np.sum(explained_var[:i+1])*100:.2f}%"
        })
    
    summary_df = pd.DataFrame(summary_data)
    # Format the summary table
    write_output(summary_df.to_string(index=False))
    write_output("\n" + "=" * 100 + "\n")
    
    # For each regular PCA component
    for i in range(num_pca_components):
        # Get the loadings for this component
        component_loadings = loadings[i]
        
        # Create a DataFrame with feature names and their loadings
        loadings_df = pd.DataFrame({
            'Feature': feature_names,
            'Loading': component_loadings
        })
        
        # Sort by absolute loading value (to get most influential features)
        loadings_df['Abs_Loading'] = abs(loadings_df['Loading'])
        loadings_df = loadings_df.sort_values('Abs_Loading', ascending=False)
        
        # Get top N features
        top_loadings = loadings_df.head(top_n)[['Feature', 'Loading']].reset_index(drop=True)
        
        # If token mapping provided, add token information
        if token_mapping:
            top_loadings['Token'] = top_loadings['Feature'].apply(
                lambda x: format_token(token_mapping.get(x))
            )
        
        # Component header
        write_output(f"\nPRINCIPAL COMPONENT {i+1} ({explained_var[i]*100:.2f}% variance)")
        write_output("-" * 100)
        
        # Print the formatted table
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.width', 100)
        pd.set_option('display.float_format', '{:.4f}'.format)
        
        write_output(top_loadings.to_string(index=False))
        write_output("\n")
    
    # Process sparse PCA components if they exist and requested
    if pca_handler.sparse_pca is not None and include_sparse:
        sparse_loadings = pca_handler.sparse_pca.components_
        
        # Get sparsity for sparse components
        sparse_sparsity = None
        if hasattr(pca_handler, 'get_sparse_component_sparsity'):
            sparse_sparsity = pca_handler.get_sparse_component_sparsity()
        
        # Get variance explained by sparse components if available
        sparse_variance = None
        if hasattr(pca_handler, 'get_sparse_component_variance'):
            try:
                sparse_variance = pca_handler.get_sparse_component_variance()
            except:
                pass
        
        # Limit to specified number of components
        num_sparse_components = min(num_components if num_components else float('inf'), 
                                sparse_loadings.shape[0])
        
        # Header for sparse PCA components
        write_output("=" * 100)
        write_output(f"SPARSE PCA COMPONENTS FROM RESIDUALS (Top {top_n} Features per Component)")
        write_output("=" * 100)
        
        # Print summary for sparse components
        write_output("\nSPARSE COMPONENT SUMMARY:")
        summary_data = []
        for i in range(num_sparse_components):
            summary_entry = {
                "Component": f"SPC{i+1}",
            }
            
            # Add sparsity if available
            if sparse_sparsity is not None:
                summary_entry["Sparsity (%)"] = f"{sparse_sparsity[i]*100:.2f}%"
            
            # Add variance if available
            if sparse_variance is not None and i < len(sparse_variance):
                summary_entry["Explained Variance of Residuals (%)"] = f"{sparse_variance[i]*100:.2f}%"
            
            summary_data.append(summary_entry)
        
        summary_df = pd.DataFrame(summary_data)
        write_output(summary_df.to_string(index=False))
        write_output("\n" + "=" * 100 + "\n")
        
        # For each sparse PCA component
        for i in range(num_sparse_components):
            # Get the loadings for this component
            component_loadings = sparse_loadings[i]
            
            # Create a DataFrame with feature names and their loadings
            loadings_df = pd.DataFrame({
                'Feature': feature_names,
                'Loading': component_loadings
            })
            
            # Sort by absolute loading value (to get most influential features)
            loadings_df['Abs_Loading'] = abs(loadings_df['Loading'])
            loadings_df = loadings_df.sort_values('Abs_Loading', ascending=False)
            
            # Get top N features with non-zero loadings
            non_zero_mask = loadings_df['Loading'] != 0
            non_zero_loadings = loadings_df[non_zero_mask]
            top_loadings = non_zero_loadings.head(top_n)[['Feature', 'Loading']].reset_index(drop=True)
            
            # If token mapping provided, add token information
            if token_mapping:
                top_loadings['Token'] = top_loadings['Feature'].apply(
                    lambda x: format_token(token_mapping.get(x))
                )
            
            # Calculate sparsity
            sparsity = 1.0 - (np.count_nonzero(component_loadings) / len(component_loadings))
            
            # Sparse component header
            sparse_header = f"SPARSE PRINCIPAL COMPONENT {i+1} (Sparsity: {sparsity*100:.2f}%"
            
            # Add variance info if available
            if sparse_variance is not None and i < len(sparse_variance):
                sparse_header += f", Variance of Residuals: {sparse_variance[i]*100:.2f}%"
            sparse_header += ")"
            
            write_output(sparse_header)
            write_output("-" * 100)
            
            # Handle empty case
            if len(top_loadings) == 0:
                write_output("No non-zero loadings found in this component")
            else:
                write_output(top_loadings.to_string(index=False))
            
            write_output("\n")
    
    # Save to file if requested
    if save_to_file:
        with open(save_to_file, 'w') as f:
            f.write(output_buffer.getvalue())
        print(f"Output saved to {save_to_file}")
    
    # Return the full text
    return output_buffer.getvalue()
