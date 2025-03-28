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


def load_trajectory_data(experiment_name, dataset_name, model_sizes=None, num_contexts=None):
    """
    Load trajectory data for a given dataset name and experiment name.
    
    Args:
        experiment_name (str): Name of the experiment (e.g., "EXP000")
        dataset_name (str): Name of the dataset
        model_sizes (list, optional): List of model sizes to load. If None, loads all available.
        num_contexts (int, optional): Number of contexts to load. If None, loads all contexts.
                                     This will select columns containing "context_0", "context_1", etc.
                                     up to "context_{num_contexts-1}"
    
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
    
    # Determine columns to select if num_contexts is specified
    context_columns = None
    if num_contexts is not None:
        # Use the first model size's file to get column names
        first_model_size = model_sizes[0]
        first_file_path = os.path.join(base_path, f"{first_model_size}_{dataset_name}.csv")
        
        if not os.path.exists(first_file_path):
            print(f"Warning: File not found for model size {first_model_size}, dataset {dataset_name}")
            return {}
            
        # Read just the header to get column names
        with open(first_file_path, 'r') as f:
            all_columns = next(csv.reader(f))
        
        # Always include 'step' column
        step_column = ['step']
        
        # Extract all unique context indices that exist in the column names
        context_indices = set()
        for col in all_columns:
            if col.startswith('context_'):
                try:
                    # Extract the context index from the column name
                    # Format: context_X_pos_Y where X is the context index
                    parts = col.split('_pos_')
                    if len(parts) == 2:
                        context_prefix = parts[0]  # e.g., "context_123"
                        context_index = int(context_prefix.split('_')[1])
                        context_indices.add(context_index)
                except (ValueError, IndexError):
                    continue  # Skip columns that don't match the expected format
        
        # Sort the context indices and take the first num_contexts
        sorted_context_indices = sorted(context_indices)
        if len(sorted_context_indices) < num_contexts:
            print(f"Warning: Requested {num_contexts} contexts but only {len(sorted_context_indices)} available")
            selected_context_indices = sorted_context_indices
        else:
            selected_context_indices = sorted_context_indices[:num_contexts]
        
        # Create a set of valid context prefixes for the selected indices
        valid_prefixes = set(f"context_{i}_" for i in selected_context_indices)
        
        # Filter columns for the selected context indices
        context_specific_columns = []
        for col in all_columns:
            parts = col.split('_pos_')
            if len(parts) == 2 and parts[0] + '_' in valid_prefixes:
                context_specific_columns.append(col)
        
        # Create final list of columns to load (step + context-specific)
        context_columns = step_column + context_specific_columns
        
        print(f"Loading columns for {len(selected_context_indices)} contexts")
        print(f"Selected context indices: {selected_context_indices[:5]}... (total: {len(selected_context_indices)})")
    
    # Load data for each model size
    for model_size in tqdm(model_sizes, desc="Loading trajectory data"):
        file_path = os.path.join(base_path, f"{model_size}_{dataset_name}.csv")
        
        if not os.path.exists(file_path):
            print(f"Warning: File not found for model size {model_size}, dataset {dataset_name}")
            continue
        
        # Load the data, using filtered columns if applicable
        if num_contexts is not None:
            df = pd.read_csv(file_path, usecols=context_columns)
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
    
    def __init__(self, trajectory_data: Dict[str, pd.DataFrame], 
             step_range=[None, None], 
             n_components: int = 10, n_sparse_components: int = 0, scale: bool = False,
             sparse_pca_params: Optional[Dict] = None, run_at_init: bool = False,
             dataset_name=None, num_contexts=None):
        """
        Initialize TrajectoryPCA with trajectory data.
        
        Args:
            trajectory_data (dict): Dictionary where keys are model sizes and 
                                    values are the corresponding dataframes
            step_range (list): [min_step, max_step] to include in the analysis
            n_components (int): Number of regular PCA components to use
            n_sparse_components (int): Number of sparse PCA components to extract from residuals
            scale (bool): Whether to standardize the data before PCA
            sparse_pca_params (dict, optional): Parameters for the sparse PCA
            run_at_init (bool): Whether to run the PCA pipeline during initialization
        """
        # Input data and parameters
        self.trajectory_data = {}
        self.step_range = step_range
        self.n_components = n_components
        self.n_sparse_components = n_sparse_components
        self.scale = scale
        self.sparse_pca_params = sparse_pca_params or {}
        self.dataset_name = dataset_name
        self.num_contexts = num_contexts
        
        # Copy the data, filtering by step_range
        for model_size, df in trajectory_data.items():
            df_copy = df.copy()
            min_step = self.step_range[0] if self.step_range[0] is not None else df_copy['step'].min()
            max_step = self.step_range[1] if self.step_range[1] is not None else df_copy['step'].max()
            self.trajectory_data[model_size] = df_copy[
                (df_copy['step'] >= min_step) & 
                (df_copy['step'] <= max_step)
            ]

        self.min_step = min_step
        self.max_step = max_step

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
    
    def fit_pca(self) -> Tuple[Optional[PCA], Optional[SparsePCA]]:
        """
        Fit PCA on the concatenated trajectory data, followed by sparse PCA on residuals if requested.
        
        Returns:
            Tuple[Optional[PCA], Optional[SparsePCA]]: Fitted PCA and SparsePCA objects (if applicable)
        """
        if self.concatenated_matrix is None:
            self.concatenate_trajectories()
            
        # Scale the data if required
        if self.scale:
            self.scaler = StandardScaler()
            scaled_data = self.scaler.fit_transform(self.concatenated_matrix)
        else:
            scaled_data = self.concatenated_matrix
            
        # Initialize PCA and sparse PCA objects to None
        self.pca = None
        self.sparse_pca = None
        
        # Fit regular PCA if n_components > 0
        if self.n_components > 0:
            self.pca = PCA(n_components=self.n_components)
            self.pca.fit_transform(scaled_data)
            
            # Print explained variance for regular PCA
            explained_var = self.pca.explained_variance_ratio_
            cumulative_var = np.cumsum(explained_var)
            
            print(f"Regular PCA: Top {self.n_components} components explain {cumulative_var[-1]*100:.2f}% of variance")
            print(f"Individual explained variance: {explained_var}")
        
        # Fit sparse PCA if n_sparse_components > 0
        if self.n_sparse_components > 0:
            # Calculate residuals if regular PCA was performed
            if self.pca is not None:
                pca_transformed = self.pca.transform(scaled_data)
                reconstructed_data = self.pca.inverse_transform(pca_transformed)
                self.pca_residual_matrix = scaled_data - reconstructed_data
                print(f"Residual matrix shape after regular PCA: {self.pca_residual_matrix.shape}")
            else:
                # If no regular PCA was performed, use the original scaled data
                self.pca_residual_matrix = scaled_data
                print(f"Using full matrix for sparse PCA (no regular PCA performed): {self.pca_residual_matrix.shape}")
            
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
            
            # Fit sparse PCA on residuals or original data
            self.sparse_pca = SparsePCA(n_components=self.n_sparse_components, **sparse_params)
            self.sparse_pca.fit_transform(self.pca_residual_matrix)
            
            # Calculate sparsity of components
            component_sparsity = self.get_sparse_component_sparsity()
                
            print(f"Sparse PCA: {self.n_sparse_components} components extracted")
            print(f"Sparsity of components (fraction of zero values): {component_sparsity}")
            
        return self.pca, self.sparse_pca
    
    def transform_trajectories(self) -> Dict[str, pd.DataFrame]:
        """
        Transform the original trajectories into the PCA space (both regular and sparse if applicable)
        and store them back in the trajectory data dictionary.
        
        Returns:
            dict: Dictionary of transformed trajectories for each model size
        """
        if self.pca is None and self.sparse_pca is None:
            raise ValueError("Neither PCA nor sparse PCA fitted. Call fit_pca() first.")
            
        transformed_data = {}
        
        for model_size in self.model_sizes:
            df = self.trajectory_data[model_size]
            
            # Get the data for this model size
            model_data = df[self.common_columns].values
            
            # Scale if needed
            if self.scaler is not None:
                processed_data = self.scaler.transform(model_data)
            else:
                processed_data = model_data
                
            # Initialize transformed DataFrame
            transformed_df = pd.DataFrame()
            
            # Transform using regular PCA if it was fitted
            if self.pca is not None:
                pca_transformed = self.pca.transform(processed_data)
                
                # Create column names for regular PCA components
                pca_component_cols = [f"PC{i+1}" for i in range(pca_transformed.shape[1])]
                
                # Add regular PCA components to DataFrame
                pca_df = pd.DataFrame(pca_transformed, columns=pca_component_cols)
                transformed_df = pd.concat([transformed_df, pca_df], axis=1)
                
                # Calculate residuals for sparse PCA if both were fitted
                if self.sparse_pca is not None:
                    reconstructed_data = self.pca.inverse_transform(pca_transformed)
                    model_residuals = processed_data - reconstructed_data
                else:
                    model_residuals = None
            else:
                # If no regular PCA, use original data for sparse PCA
                model_residuals = processed_data
            
            # Add sparse PCA components if fitted
            if self.sparse_pca is not None and model_residuals is not None:
                # Transform using sparse PCA
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
        self.transform_trajectories()  # Don't capture the return value yet
        self.normalize_component_signs()  # Modify the transformed data
        
        # After normalization, collect the transformed data to return
        transformed_data = {}
        for model_size in self.model_sizes:
            transformed_key = f"{model_size}_transformed"
            if transformed_key in self.trajectory_data:
                transformed_data[model_size] = self.trajectory_data[transformed_key]
        
        return transformed_data
    
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
    
    def normalize_component_signs(self, reference_model=None):
        """
        Normalize PCA component signs so the reference model (default: largest) 
        has positive values at the first step.
        """
        # Select reference model (default to last/largest model)
        if reference_model is None:
            reference_model = self.model_sizes[-1]
        
        # Get transformed data for reference model
        transformed_key = f"{reference_model}_transformed"
        if transformed_key not in self.trajectory_data:
            print("Warning: Reference model not found in transformed data. Skipping sign normalization.")
            return False
        
        reference_data = self.trajectory_data[transformed_key]
        
        # Get first step row
        if 'step' not in reference_data.columns:
            print("Warning: Step column not found in reference data. Skipping sign normalization.")
            return False
            
        min_step_row = reference_data.loc[reference_data['step'].idxmin()]
        
        # Normalize regular PCA components if they exist
        if self.pca is not None:
            pc_cols = [col for col in reference_data.columns if col.startswith('PC')]
            for pc_col in pc_cols:
                pc_idx = int(pc_col[2:]) - 1  # Extract PC index (0-based)
                first_step_value = min_step_row[pc_col]
                
                # Flip sign if negative
                if first_step_value < 0:
                    self.pca.components_[pc_idx] *= -1
                    for model_size in self.model_sizes:
                        model_key = f"{model_size}_transformed"
                        if model_key in self.trajectory_data and pc_col in self.trajectory_data[model_key].columns:
                            self.trajectory_data[model_key][pc_col] *= -1
        
        # Normalize sparse PCA components if they exist
        if self.sparse_pca is not None:
            spc_cols = [col for col in reference_data.columns if col.startswith('SPC')]
            for spc_col in spc_cols:
                spc_idx = int(spc_col[3:]) - 1
                first_step_value = min_step_row[spc_col]
                
                if first_step_value < 0:
                    self.sparse_pca.components_[spc_idx] *= -1
                    for model_size in self.model_sizes:
                        model_key = f"{model_size}_transformed"
                        if model_key in self.trajectory_data and spc_col in self.trajectory_data[model_key].columns:
                            self.trajectory_data[model_key][spc_col] *= -1
        
        return True
    
    def get_specific_column_loadings(self, columns_of_interest):
        """
        Generate a table showing loadings of specific columns on PCs and sparse PCs.
        
        Args:
            columns_of_interest (list): List of column names to include (e.g., "context_0_pos_1")
        
        Returns:
            pd.DataFrame: Table with columns as rows and components as columns
        """
        # Initialize the results DataFrame with feature names
        results = pd.DataFrame(columns_of_interest, columns=['Feature'])
        results.set_index('Feature', inplace=True)
        
        # Process regular PCA components
        if self.pca is not None:
            feature_names = self.common_columns
            
            # Add PC loadings for each feature
            for i in range(self.pca.components_.shape[0]):
                pc_col = f"PC{i+1}"
                
                # Get loadings for this component
                pc_loadings = {}
                for feature in columns_of_interest:
                    if feature in feature_names:
                        feature_idx = feature_names.index(feature)
                        pc_loadings[feature] = self.pca.components_[i, feature_idx]
                
                # Add to results
                results[pc_col] = pd.Series(pc_loadings)
        
        # Process sparse PCA components if available
        if self.sparse_pca is not None:
            feature_names = self.common_columns
            
            # Add sparse PC loadings for each feature
            for i in range(self.sparse_pca.components_.shape[0]):
                spc_col = f"SPC{i+1}"
                
                # Get loadings for this component
                spc_loadings = {}
                for feature in columns_of_interest:
                    if feature in feature_names:
                        feature_idx = feature_names.index(feature)
                        spc_loadings[feature] = self.sparse_pca.components_[i, feature_idx]
                
                # Add to results
                results[spc_col] = pd.Series(spc_loadings)
        
        # Reset index to make Feature a regular column
        results.reset_index(inplace=True)
        
        return results
    
    def compute_cosine_with_spc(self, columns_of_interest, pc_idx=('sparse', 6)):
        """
        Compute the cosine similarity between the sum of one-hot vectors for specified columns 
        and a specific sparse principal component.
        
        Args:
            columns_of_interest (list): List of column names to include
            spc_index (int): Index of the sparse PC to compare with (default: 6)
            
        Returns:
            float: Cosine similarity score
        """
        from scipy.spatial.distance import cosine
        import numpy as np

        # Get all feature names
        feature_names = self.common_columns
        
        # Create one-hot vector for each column (1 at the column's position, 0 elsewhere)
        token_vector = np.zeros(len(feature_names))
        
        # Set 1 for each column of interest
        for col in columns_of_interest:
            if col in feature_names:
                idx = feature_names.index(col)
                token_vector[idx] = 1
        
        # Get the sparse PC vector (0-indexed, so SPC6 is at index 5)
        if pc_idx[0] == 'sparse':
            spc_index = pc_idx[1]
            vector = self.sparse_pca.components_[spc_index-1]
        elif idx[0] == 'pca':
            pc_index = pc_idx[1]
            vector = self.pca.components_[pc_index-1]
        
        # Compute cosine similarity (1 - cosine distance)
        similarity = 1 - cosine(token_vector, vector)
        
        return similarity

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

    def format_step_number(self, step):
        """Format step number to show 'k' for thousands"""
        if step is None:
            return "None"
        elif step >= 1000:
            return f"{step/1000:.0f}k" if step % 1000 == 0 else f"{step/1000:.0f}k"
        else:
            return str(step)


    def _get_filename(self, base_plotting_path = "/Users/liam/quests/lsoc-psych/datasets/plotting/joint_pca"):
        """
        Generate a filename based on the dataset name and PCA parameters.
        """
        dataset_name = self.pca_handler.dataset_name
        n_components = self.pca_handler.n_components
        n_sparse_components = self.pca_handler.n_sparse_components
        
        # Format the step range
        min_step = self.format_step_number(self.pca_handler.min_step)
        max_step = self.format_step_number(self.pca_handler.max_step)
        step_range_str = f"steps[{min_step}-{max_step}]"
        
        # Rest of the method remains the same
        sparse_params_str = ""
        if self.pca_handler.sparse_pca_params:
            params = []
            for key, value in self.pca_handler.sparse_pca_params.items():
                params.append(f"{key}={value}")
            if params:
                sparse_params_str = f", {', '.join(params)}"

        filename = f"{dataset_name}_PC{n_components}_SPC{n_sparse_components}_{step_range_str}{sparse_params_str}"
        path = os.path.join(base_plotting_path, filename)
        
        return path
    
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
            raise ValueError("Regular PCA not fitted. Cannot plot PCA components.")
        
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
    
    def plot_pcs_over_time(self):
        """
        Plot PCA components vs training steps in a 2x5 grid layout.
        
        Returns:
            plotly.graph_objects.Figure: Plotly figure with grid subplots
        """
        # Set fixed parameters
        width = 1300
        height = 1200
        line_width = 2
        
        # Initialize variables for component titles
        pc_titles = []
        spc_titles = []
        num_pca_components = 0
        num_sparse_components = 0
        
        # Get regular PCA components info if available
        if self.pca_handler.pca is not None:
            num_pca_components = min(self.pca_handler.pca.n_components_, 10)
            pca_explained_var = self.pca_handler.pca.explained_variance_ratio_
            pc_titles = [f"PC{i+1} (EV: {pca_explained_var[i]*100:.2f}%)" for i in range(num_pca_components)]
        
        # Get sparse PCA components info if available
        has_sparse = False
        if self.pca_handler.sparse_pca is not None:
            has_sparse = True
            # If no regular PC components, we can use all 10 slots for sparse components
            max_sparse_slots = 10 - num_pca_components
            num_sparse_components = min(self.pca_handler.sparse_pca.n_components_, max_sparse_slots)
            
            # Get sparsity for sparse components
            if hasattr(self.pca_handler, 'get_sparse_component_sparsity'):
                sparse_sparsity = self.pca_handler.get_sparse_component_sparsity()
                sparse_explained_variance = self.pca_handler.get_sparse_component_variance()
                
                # Data source label
                data_source = "REV" if self.pca_handler.pca is not None else "EV"
                
                if sparse_explained_variance is not None:
                    spc_titles = [
                        f"SPC{i+1} ({data_source}: {sparse_explained_variance[i]*100:.2f}%, Sparsity: {sparse_sparsity[i]*100:.2f}%)" 
                        for i in range(num_sparse_components)
                    ]
                else:
                    spc_titles = [
                        f"SPC{i+1} (Sparsity: {sparse_sparsity[i]*100:.2f}%)" 
                        for i in range(num_sparse_components)
                    ]
            else:
                # Fallback if sparsity info not available
                spc_titles = [f"SPC{i+1}" for i in range(num_sparse_components)]
        
        # Auto-generate title
        dataset_name = self.pca_handler.dataset_name or "Unknown"
        n_components = self.pca_handler.n_components
        n_sparse_components = self.pca_handler.n_sparse_components
        
        # Check for non-default sparse PCA params
        sparse_params_str = ""
        if self.pca_handler.sparse_pca_params:
            params = []
            for key, value in self.pca_handler.sparse_pca_params.items():
                params.append(f"{key}={value}")
            if params:
                sparse_params_str = f", {', '.join(params)}"
        
        min_step = self.format_step_number(self.pca_handler.min_step)
        max_step = self.format_step_number(self.pca_handler.max_step)
        step_range_str = f"steps [{min_step}-{max_step}]"

        title = f"PCA on {dataset_name} - {n_components} PCs, {n_sparse_components} SPCs ({step_range_str}{sparse_params_str}, num_contexts={self.pca_handler.num_contexts})"

        # Calculate total components to plot (max 10)
        total_components = min(num_pca_components + num_sparse_components, 10)
        
        # Check if we have any components to plot
        if total_components == 0:
            raise ValueError("No PCA or sparse PCA components available to plot.")
        
        # Re-arrange subplot titles to follow proper order (row by row)
        subplot_titles = []
        for i in range(5):  # 5 rows
            for j in range(2):  # 2 columns
                idx = i + j * 5
                if idx < len(pc_titles + spc_titles):
                    combined_titles = pc_titles + spc_titles
                    subplot_titles.append(combined_titles[idx])
                else:
                    subplot_titles.append("")
        
        # Create subplot layout
        fig = make_subplots(
            rows=5,
            cols=2,
            subplot_titles=subplot_titles,
            vertical_spacing=0.06,
            horizontal_spacing=0.08
        )
        
        # Helper function to get row, col from component index (0-based)
        def get_subplot_position(idx):
            # For a 2x5 grid, ordering goes row by row
            row = (idx % 5) + 1   # Remainder gives row (0-4, then add 1)
            col = (idx // 5) + 1  # Integer division gives column (0 or 1, then add 1)
            return row, col
        
        # Add traces for regular PCA components if they exist
        if self.pca_handler.pca is not None:
            for pc_idx in range(num_pca_components):
                pc = pc_idx + 1  # 1-based PC numbering
                col_pc = f"PC{pc}"
                row, col = get_subplot_position(pc_idx)
                
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
                    fig.add_trace(
                        go.Scatter(
                            x=df['step'], 
                            y=df[col_pc],
                            mode='lines',
                            name=f"Pythia {model_size}",
                            line=dict(color=self.model_colors[model_size], width=line_width),
                            text=hover_text,
                            hoverinfo='text',
                            showlegend=True if pc_idx == 0 else False  # Only show legend for first component
                        ),
                        row=row, 
                        col=col
                    )
        
        # Add traces for sparse PCA components if available
        if has_sparse:
            for spc_idx in range(num_sparse_components):
                spc = spc_idx + 1  # 1-based SPC numbering
                col_spc = f"SPC{spc}"
                
                # Calculate position in grid
                plot_idx = num_pca_components + spc_idx
                row, col = get_subplot_position(plot_idx)
                
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
                    fig.add_trace(
                        go.Scatter(
                            x=df['step'], 
                            y=df[col_spc],
                            mode='lines',
                            name=f"Pythia {model_size}",
                            line=dict(color=self.model_colors[model_size], width=line_width),
                            text=hover_text,
                            hoverinfo='text',
                            # Only show legend for the first component and only if there are no regular PCs
                            showlegend=True if spc_idx == 0 and num_pca_components == 0 else False
                        ),
                        row=row, 
                        col=col
                    )
        
        # Add axis lines and labels for each subplot
        for i in range(total_components):
            row, col = get_subplot_position(i)
            component_idx = i
            
            # Determine if this is a PC or SPC
            if component_idx < num_pca_components:
                y_title = f"PC{component_idx+1}"
            else:
                y_title = f"SPC{component_idx-num_pca_components+1}"
            
            # Add borders to plot
            fig.update_xaxes(
                showline=True,
                linewidth=1,
                linecolor='black',
                mirror=True,
                showticklabels=True,  # Always show x-axis ticks
                row=row,
                col=col,
                # range=[self.pca_handler.step_range[0], self.pca_handler.step_range[1]]
            )
            
            # Add the "Training Steps" title only to the bottom plots
            if row == 5:
                fig.update_xaxes(title="Training Steps", row=row, col=col)
            
            # Add y-axis title for all plots
            fig.update_yaxes(
                title=y_title,
                showline=True,
                title_standoff=5,
                linewidth=1,
                linecolor='black',
                mirror=True,
                row=row,
                col=col
            )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,  # Center the title
                y=0.99,
                font=dict(
                    family="Arial",
                    size=18,
                )
            ),
            width=width,
            height=height,
            template='plotly_white',
            showlegend=True,
            legend=dict(
                orientation="h",  # horizontal orientation
                yanchor="top",
                y=1.06,  # Position at the top
                xanchor="center",
                x=0.5  # Position to the right of the plots
            ),
            margin=dict(l=60, r=80, t=100, b=20)  # Increased bottom margin for legend
        )
        
        # Update all x-axes to log scale
        fig.update_xaxes(type='log', showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        
        return fig
    
    def plot_top_loaded_features_for_component(self, component: int, token_mapping=None, token_data=None):
        """
        Plot the top 10 loaded features for a specific principal component.
        
        Args:
            component_idx (int): Index of the component (0-based, counts through regular PCs then sparse PCs)
            token_mapping (dict, optional): Dictionary mapping column names to token values
            
        Returns:
            plotly.graph_objects.Figure: Plotly figure
        """
        # Fixed values
        width = 1300
        height = 1200
        line_width = 2
        min_step = 0
        num_features = 10

        is_dual_pca = 'DualTrajectoryPCA' in str(type(self.pca_handler))

        component_idx = component - 1  # Convert to 0-based indexing
        
        # Determine if this is a regular PC or sparse PC
        num_pca_components = 0 if self.pca_handler.pca is None else self.pca_handler.pca.n_components_
        
        if component_idx < num_pca_components and self.pca_handler.pca is not None:
            # Regular PC
            is_sparse = False
            pc_type = "PC"
            pc_num = component_idx + 1  # Convert to 1-based indexing
            component_loadings = self.pca_handler.pca.components_[component_idx]
            explained_var = self.pca_handler.pca.explained_variance_ratio_[component_idx]
            component_info = f"(EV: {explained_var*100:.2f}%)"
        else:
            # Sparse PC
            if self.pca_handler.sparse_pca is None:
                raise ValueError("No sparse components available")
                
            is_sparse = True
            pc_type = "SPC"
            # Adjust index for sparse components
            sparse_idx = component_idx - num_pca_components
            pc_num = sparse_idx + 1  # Convert to 1-based indexing for sparse
            
            if sparse_idx >= self.pca_handler.sparse_pca.n_components_:
                raise ValueError(f"Component index {component_idx} exceeds available components")
                
            component_loadings = self.pca_handler.sparse_pca.components_[sparse_idx]
            
            # Get sparsity if available
            component_info = ""
            if hasattr(self.pca_handler, 'get_sparse_component_sparsity'):
                sparse_sparsity = self.pca_handler.get_sparse_component_sparsity()
                if sparse_sparsity and sparse_idx < len(sparse_sparsity):
                    component_info += f"(Sparsity: {sparse_sparsity[sparse_idx]*100:.2f}%"
                    
                    # Get variance if available
                    if hasattr(self.pca_handler, 'get_sparse_component_variance'):
                        sparse_variance = self.pca_handler.get_sparse_component_variance()
                        if sparse_variance is not None and sparse_idx < len(sparse_variance):
                            # Label appropriately depending on whether we're using residuals or original data
                            variance_label = "REV" if self.pca_handler.pca is not None else "EV"
                            component_info += f", {variance_label}: {sparse_variance[sparse_idx]*100:.2f}%"
                            
                    component_info += ")"
        
        # Get feature names
        feature_names = self.pca_handler.common_columns
        
        # Create a DataFrame with feature names and their loadings
        loadings_df = pd.DataFrame({
            'Feature': feature_names,
            'Loading': component_loadings
        })
        
        # Sort by absolute loading value (to get most influential features)
        loadings_df['Abs_Loading'] = abs(loadings_df['Loading'])
        loadings_df = loadings_df.sort_values('Abs_Loading', ascending=False)
        
        # Get top 10 features
        top_n = min(num_features, len(loadings_df))
        top_features = loadings_df.head(top_n)
        
        # Helper function to extract context from token_mapping
        def extract_context_from_feature(feature):
            if not token_mapping or feature not in token_mapping:
                return ""
                
            # Get current token
            token = token_mapping[feature]
            
            # Format current token, handling special characters
            def format_token_display(token_str, max_len=5):
                # Handle special characters first
                formatted = token_str.replace('\n', '\\n').replace('\t', '\\t').replace('\r', '\\r')
                
                # Check if it's mostly whitespace
                if formatted.strip() == "" and len(formatted) > 2:
                    return "[whitespace]"
                    
                # Truncate if too long
                if len(formatted) > max_len:
                    return formatted[:max_len] + "..."
                    
                return formatted
            
            # Format current token
            token_text = format_token_display(token, 8)
            
            # Try to extract context from the feature name
            # Format is typically 'context_X_pos_Y'
            try:
                parts = feature.split('_pos_')
                if len(parts) == 2:
                    context_prefix = parts[0]  # e.g., "context_123"
                    pos = int(parts[1])  # e.g., 45
                    
                    # Try to get previous token
                    prev_feature = f"{context_prefix}_pos_{pos-1}"
                    prev_token = ""
                    if prev_feature in token_mapping:
                        prev_token = format_token_display(token_mapping[prev_feature], 5)
                    
                    # Try to get next token
                    next_feature = f"{context_prefix}_pos_{pos+1}"
                    next_token = ""
                    if next_feature in token_mapping:
                        next_token = format_token_display(token_mapping[next_feature], 5)
                    
                    # Only show context if we have at least one neighboring token
                    if prev_token or next_token:
                        return f" [\"{prev_token}\", <b>\"{token_text}\"</b>, \"{next_token}\"]"
            except:
                # Fall back to just showing the token
                pass
            
            # If extraction failed or no context available, just show the token
            return f" \"{token_text}\""
        
        # Create subplot titles with loading values

        def get_subplot_position(idx):
            # For a 2x5 grid, ordering goes row by row
            row = (idx % 5) + 1   # Remainder gives row (0-4, then add 1)
            col = (idx // 5) + 1  # Integer division gives column (0 or 1, then add 1)
            return row, col

        subplot_titles = [None] * top_n
        for i, (_, row) in enumerate(top_features.iterrows()):
            feature = row['Feature']
            loading = row['Loading']
            
            # Add token info if available
            token_info = extract_context_from_feature(feature)
            
            # Add this code for category annotation
            category_info = ""
            if token_data is not None and is_dual_pca:
                try:
                    # Handle feature name format for DualTrajectoryPCA (which has _few or _zero suffix)
                    feature_base = feature
                    if feature.endswith('_few') or feature.endswith('_zero'):
                        feature_base = feature[:-5]
                    
                    # Extract context_id
                    parts = feature_base.split('_pos_')[0].split('_')
                    if len(parts) >= 2:
                        context_id = int(parts[1])
                        if context_id in token_data:
                            category = token_data[context_id].get('category', 'unknown')
                            category_info = f" [{category}]"
                except:
                    pass
            
            # Update your title creation
            title = f"{i+1}. {feature}{token_info}{category_info} (Loading: {loading:.4f})"
        
            
            # Calculate the position in the subplot_titles list for row-by-row ordering
            row, col = get_subplot_position(i)
            subplot_title_list_position = 2*(row-1) + col - 1
            
            # Place the title in the correct position
            subplot_titles[subplot_title_list_position] = title
        
        fig = make_subplots(
            rows=5,
            cols=2,
            subplot_titles=subplot_titles,
            vertical_spacing=0.06,
            horizontal_spacing=0.08
        )
        
        # Helper function to get row, col from feature index (0-based)
        
        
        # Add traces for each feature and model
        for i, (_, row) in enumerate(top_features.iterrows()):
            feature = row['Feature']
            
            # Get subplot position
            subplot_row, subplot_col = get_subplot_position(i)
            
            # Add traces for each model size
            for model_size in self.model_order:
                # Get data for this model
                if model_size not in self.pca_handler.trajectory_data:
                    continue
                    
                df = self.pca_handler.trajectory_data[model_size]
                
                # Check if feature exists and filter by min_step
                if feature not in df.columns or 'step' not in df.columns:
                    continue
                    
                filtered_df = df[df['step'] >= min_step]
                
                # Create hover text
                hover_text = [f"Model: {model_size}<br>Step: {step}<br>Value: {val:.6f}" 
                            for step, val in zip(filtered_df['step'], filtered_df[feature])]
                
                # Add trace
                fig.add_trace(
                    go.Scatter(
                        x=filtered_df['step'], 
                        y=filtered_df[feature],
                        mode='lines',
                        name=f"Pythia {model_size}",
                        line=dict(color=self.model_colors[model_size], width=line_width),
                        text=hover_text,
                        hoverinfo='text',
                        showlegend=True if i == 0 else False  # Only show legend for first subplot
                    ),
                    row=subplot_row, 
                    col=subplot_col
                )
        
        # Generate title
        dataset_name = self.pca_handler.dataset_name or "Dataset"
        data_source = "Original Data" if self.pca_handler.pca is None and is_sparse else "Dataset"
        title = f"Top {top_n} Loaded Features for {pc_type}{pc_num} on {dataset_name} {component_info}"
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,  # Center the title
                y=0.99,
                font=dict(
                    family="Arial",
                    size=18,
                )
            ),
            width=width,
            height=height,
            template='plotly_white',
            showlegend=True,
            legend=dict(
                orientation="h",  # horizontal orientation
                yanchor="top",
                y=1.06,  # Position at the top
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=60, r=80, t=100, b=20)
        )
        
        # Update all x-axes to log scale
        for i in range(min(top_n, 10)):  # Maximum 10 subplots
            row, col = get_subplot_position(i)
            
            # Set x-axis to log scale and add grid
            fig.update_xaxes(
                showline=True,
                linewidth=1,
                linecolor='black',
                mirror=True,
                showticklabels=True,
                type='log',
                showgrid=True,
                gridwidth=1,
                gridcolor='LightGray',
                row=row,
                col=col,
                # range=[self.pca_handler.step_range[0], self.pca_handler.step_range[1]]
            )
            
            # Add "Training Steps" label to bottom plots
            if row == 5:
                fig.update_xaxes(title="Training Steps", row=row, col=col)
            
            # Add y-axis grid and title
            fig.update_yaxes(
                showline=True,
                linewidth=1,
                linecolor='black',
                mirror=True,
                showgrid=True,
                gridwidth=1,
                gridcolor='LightGray',
                title="Loss",
                row=row,
                col=col
            )
        
        return fig



### DM MATHEMATICS ###

def load_token_mapping_dm_math(experiment_name, dataset_name):
    tokens_path = f"/Users/liam/quests/lsoc-psych/datasets/experiments/{experiment_name}/trajectories/tokens/{dataset_name}_tokens.pkl"
    
    with open(tokens_path, 'rb') as f:
        token_data = pickle.load(f)
    
    token_mapping = {}
    
    for context_id, context_data in token_data.items():
        tokens = context_data['tokens']
        for token_idx, token_value in enumerate(tokens):
            key = f"context_{context_id}_pos_{token_idx}"
            token_mapping[key] = token_value
    
    return token_mapping, token_data


def build_category_columns(tokens_dict, answer_tokens_only=False):
    """
    Build a dictionary mapping each category to a dictionary of context IDs and 
    their associated CSV column names based on the tokens (all tokens for full context 
    or only answer tokens if answer_tokens_only is True).

    The tokens_dict is updated in-place to add 'answer_tokens' (if needed) for each context.
    """
    category_columns = {}
    for context_idx, context_data in tokens_dict.items():
        # Get category (default to 'unknown' if not provided)
        category = context_data.get('category', 'unknown')
        tokens = context_data.get('tokens', [])
        
        # If we want answer tokens only, precompute them if they are not already set.
        if answer_tokens_only:
            if 'answer_tokens' not in context_data:
                # Find the last newline index; tokens after this are considered the answer.
                last_newline = -1
                for i, token in enumerate(tokens):
                    if token == "\n":
                        last_newline = i
                # Store answer token indices in the dictionary.
                context_data['answer_tokens'] = list(range(last_newline + 1, len(tokens))) if last_newline != -1 else []
            indices = context_data['answer_tokens']
        else:
            indices = list(range(len(tokens)))
        
        # Create the list of CSV column names for this context.
        # The CSV columns are expected to have the format: "context_{context_idx}_pos_{token_idx}"
        cols = [f"context_{context_idx}_pos_{i}" for i in indices]
        
        # Organize by category and context.
        if category not in category_columns:
            category_columns[category] = {}
        category_columns[category][context_idx] = cols

    return category_columns

def get_answer_columns(tokens_dict):
    category_columns = build_category_columns(tokens_dict, answer_tokens_only=True)
    
    answer_columns = []
    for category, contexts in category_columns.items():
        for context_idx, columns in contexts.items():
            answer_columns.extend(columns)
    
    return answer_columns

def load_dual_trajectory_data(experiment_name, dataset_name, model_sizes=None, 
                             answer_only=False, num_contexts=None):
    base_path = f"/Users/liam/quests/lsoc-psych/datasets/experiments/{experiment_name}/trajectories/csv"
    _, tokens_dict = load_token_mapping_dm_math(experiment_name, dataset_name)
    
    columns_to_load = None
    if answer_only:
        columns_to_load = ['step'] + get_answer_columns(tokens_dict)
        print(f"Loading {len(columns_to_load)-1} answer columns")

    if model_sizes is None:
        pattern_few = os.path.join(base_path, f"*_{dataset_name}_few_shot.csv")
        file_paths_few = glob.glob(pattern_few)
        model_sizes_few = [os.path.basename(fp).split('_')[0] for fp in file_paths_few]
        
        pattern_zero = os.path.join(base_path, f"*_{dataset_name}_zero_shot.csv")
        file_paths_zero = glob.glob(pattern_zero)
        model_sizes_zero = [os.path.basename(fp).split('_')[0] for fp in file_paths_zero]
        
        model_sizes = [size for size in model_sizes_few if size in model_sizes_zero]
    
    print(f"Loading data for model sizes: {model_sizes}")
    trajectory_data = {}

    # Get available columns with csv package, find intersection 
    with open(file_paths_few[0], 'r') as f:
        available_columns = next(csv.reader(f))
    
    columns_to_load = [col for col in columns_to_load if col in available_columns]
    
    if num_contexts is not None and not answer_only:
        all_context_ids = sorted([int(ctx_id) for ctx_id in tokens_dict.keys()])
        selected_context_ids = all_context_ids[:num_contexts]
        
        def should_include_column(col_name):
            if col_name == 'step':
                return True
            parts = col_name.split('_')
            if len(parts) >= 4 and parts[0] == 'context':
                try:
                    context_id = int(parts[1])
                    return context_id in selected_context_ids
                except ValueError:
                    return False
            return False
    
    for model_size in tqdm(model_sizes, desc="Loading trajectory data"):
        trajectory_data[model_size] = {}
        
        few_shot_path = os.path.join(base_path, f"{model_size}_{dataset_name}_few_shot.csv")
        zero_shot_path = os.path.join(base_path, f"{model_size}_{dataset_name}_zero_shot.csv")
        
        if answer_only:
            df_few = pd.read_csv(few_shot_path, usecols=columns_to_load)
            df_zero = pd.read_csv(zero_shot_path, usecols=columns_to_load)
        elif num_contexts is not None:
            with open(few_shot_path, 'r') as f:
                header = f.readline().strip().split(',')
            filtered_cols = [col for col in header if should_include_column(col)]
            df_few = pd.read_csv(few_shot_path, usecols=filtered_cols)
            
            with open(zero_shot_path, 'r') as f:
                header = f.readline().strip().split(',')
            filtered_cols = [col for col in header if should_include_column(col)]
            df_zero = pd.read_csv(zero_shot_path, usecols=filtered_cols)
        else:
            df_few = pd.read_csv(few_shot_path)
            df_zero = pd.read_csv(zero_shot_path)
        
        few_cols_with_nan = df_few.columns[df_few.isna().any()].tolist()
        zero_cols_with_nan = df_zero.columns[df_zero.isna().any()].tolist()
        all_cols_with_nan = set(few_cols_with_nan + zero_cols_with_nan)
        
        if all_cols_with_nan:
            df_few = df_few.drop(columns=[col for col in all_cols_with_nan if col in df_few.columns])
            df_zero = df_zero.drop(columns=[col for col in all_cols_with_nan if col in df_zero.columns])
        
        few_cols = set(df_few.columns)
        zero_cols = set(df_zero.columns)
        common_cols = (few_cols.intersection(zero_cols)) - {'step'}
        
        if len(common_cols) < len(few_cols) - 1 or len(common_cols) < len(zero_cols) - 1:
            df_few = df_few[['step'] + list(common_cols)]
            df_zero = df_zero[['step'] + list(common_cols)]
        
        trajectory_data[model_size]['few_shot'] = df_few
        trajectory_data[model_size]['zero_shot'] = df_zero
    
    def model_size_to_numeric(size_str):
        if 'm' in size_str:
            return float(size_str.replace('m', '')) * 1e6
        elif 'b' in size_str:
            return float(size_str.replace('b', '')) * 1e9
        else:
            try:
                return float(size_str)
            except ValueError:
                return 0
    
    sorted_model_sizes = sorted(trajectory_data.keys(), key=model_size_to_numeric)
    ordered_trajectory_data = {model_size: trajectory_data[model_size] for model_size in sorted_model_sizes}
    
    return ordered_trajectory_data


class DualTrajectoryPCA:
    """
    Class for performing PCA on concatenated trajectory data for both few-shot and zero-shot.
    
    This class handles:
    1. Renaming columns to differentiate between few-shot and zero-shot
    2. Concatenating trajectory data from different model sizes (vertically)
    3. Horizontally concatenating few-shot and zero-shot matrices
    4. Applying PCA to the combined data
    5. Transforming the original trajectories into the PCA space
    6. Storing the transformed trajectories back in the original data structure
    """
    
    def __init__(self, trajectory_data, 
                 step_range=[None, None], 
                 n_components=10, n_sparse_components=0, 
                 scale=False,
                 sparse_pca_params=None, 
                 run_at_init=False,
                 dataset_name=None,
                 num_contexts=None):
        """
        Initialize DualTrajectoryPCA with trajectory data.
        
        Args:
            trajectory_data (dict): Dictionary where keys are model sizes and 
                                   values are dictionaries with 'few_shot' and 'zero_shot' dataframes
            step_range (list): [min_step, max_step] to include in the analysis
            n_components (int): Number of regular PCA components to use
            n_sparse_components (int): Number of sparse PCA components to extract from residuals
            scale (bool): Whether to standardize the data before PCA
            sparse_pca_params (dict, optional): Parameters for the sparse PCA
            run_at_init (bool): Whether to run the PCA pipeline during initialization
            dataset_name (str): Name of the dataset
            num_contexts (int): Number of contexts included
        """
        # Input data and parameters
        self.trajectory_data = {}
        self.step_range = step_range
        self.n_components = n_components
        self.n_sparse_components = n_sparse_components
        self.scale = scale
        self.sparse_pca_params = sparse_pca_params or {}
        self.dataset_name = dataset_name
        self.num_contexts = num_contexts
        
        # Find min_step and max_step from data
        first_model = list(trajectory_data.keys())[0]
        df_few = trajectory_data[first_model]['few_shot']
        if self.step_range[0] is None:
            min_step = df_few['step'].min()
        else:
            min_step = self.step_range[0]
            
        if self.step_range[1] is None:
            max_step = df_few['step'].max()
        else:
            max_step = self.step_range[1]
            
        self.min_step = min_step
        self.max_step = max_step
        
        # Process and rename columns in the input data
        for model_size, shot_data in trajectory_data.items():
            # Process few-shot data
            df_few = shot_data['few_shot'].copy()
            
            # Apply step range filtering
            df_few = df_few[(df_few['step'] >= min_step) & (df_few['step'] <= max_step)]
            
            # Rename columns to include "_few" suffix except for 'step'
            col_mapping = {col: f"{col}_few" for col in df_few.columns if col != 'step'}
            df_few = df_few.rename(columns=col_mapping)
            
            # Process zero-shot data
            df_zero = shot_data['zero_shot'].copy()
            
            # Apply the same step range filtering
            df_zero = df_zero[(df_zero['step'] >= min_step) & (df_zero['step'] <= max_step)]
            
            # Rename columns to include "_zero" suffix except for 'step'
            col_mapping = {col: f"{col}_zero" for col in df_zero.columns if col != 'step'}
            df_zero = df_zero.rename(columns=col_mapping)
            
            # Merge the two dataframes on 'step'
            # This creates a horizontally concatenated dataframe with both few-shot and zero-shot data
            merged_df = pd.merge(df_few, df_zero, on='step', how='inner')
            
            # Store the processed data directly in the trajectory_data dict
            self.trajectory_data[model_size] = merged_df
            
        # Store the original structure for reference if needed
        self.original_data = {}
        for model_size, shot_data in trajectory_data.items():
            self.original_data[model_size] = {
                'few_shot': shot_data['few_shot'].copy(),
                'zero_shot': shot_data['zero_shot'].copy()
            }

        # Model attributes
        self.model_sizes = list(trajectory_data.keys())
        self.pca = None
        self.sparse_pca = None
        self.scaler = None
        
        # Data containers
        self.few_shot_columns = None
        self.zero_shot_columns = None
        self.combined_columns = None
        self.concatenated_few_matrix = None
        self.concatenated_zero_matrix = None
        self.combined_matrix = None
        self.pca_residual_matrix = None
        self.row_indices = {}
        
        # Run pipeline at initialization if requested
        if run_at_init:
            self.run_pca_pipeline()
    
    def concatenate_trajectories(self):
        """
        Concatenate the horizontally-combined trajectories from all model sizes vertically.
        
        Returns:
            np.ndarray: Combined matrix for PCA
        """
        # Extract column names from the first model's data
        first_model = self.model_sizes[0]
        all_columns = list(self.trajectory_data[first_model].columns)
        step_column = ['step']
        
        # Separate few-shot and zero-shot columns
        self.few_shot_columns = [col for col in all_columns if col.endswith('_few') and col != 'step']
        self.zero_shot_columns = [col for col in all_columns if col.endswith('_zero') and col != 'step']
        self.combined_columns = self.few_shot_columns + self.zero_shot_columns
        
        print(f"Few-shot columns: {len(self.few_shot_columns)}")
        print(f"Zero-shot columns: {len(self.zero_shot_columns)}")
        print(f"Combined columns: {len(self.combined_columns)}")
        
        # Initialize data matrix and row tracking
        combined_data = []
        start_idx = 0
        
        # Process each model
        for model_size in self.model_sizes:
            # Get the combined dataframe for this model
            df = self.trajectory_data[model_size]
            
            # Extract all feature columns (both few-shot and zero-shot)
            model_data = df[self.combined_columns].values
            
            # Store row indices for this model
            end_idx = start_idx + len(model_data)
            self.row_indices[model_size] = (start_idx, end_idx)
            start_idx = end_idx
            
            # Append to the combined data
            combined_data.append(model_data)
        
        # Vertically concatenate all data
        self.combined_matrix = np.vstack(combined_data)
        
        print(f"Combined matrix shape: {self.combined_matrix.shape}")
        
        # For backward compatibility, also create separate few-shot and zero-shot matrices
        few_shot_data = []
        zero_shot_data = []
        
        for model_size in self.model_sizes:
            df = self.trajectory_data[model_size]
            few_shot_data.append(df[self.few_shot_columns].values)
            zero_shot_data.append(df[self.zero_shot_columns].values)
            
        self.concatenated_few_matrix = np.vstack(few_shot_data)
        self.concatenated_zero_matrix = np.vstack(zero_shot_data)
        
        return self.combined_matrix
    
    def fit_pca(self):
        """
        Fit PCA on the combined trajectory data.
        
        Returns:
            Tuple[Optional[PCA], Optional[SparsePCA]]: Fitted PCA and SparsePCA objects
        """
        if self.combined_matrix is None:
            self.concatenate_trajectories()
            
        # Scale the data if required
        if self.scale:
            self.scaler = StandardScaler()
            scaled_data = self.scaler.fit_transform(self.combined_matrix)
        else:
            scaled_data = self.combined_matrix
            
        # Initialize PCA and sparse PCA objects to None
        self.pca = None
        self.sparse_pca = None
        
        # Fit regular PCA if n_components > 0
        if self.n_components > 0:
            self.pca = PCA(n_components=self.n_components)
            self.pca.fit_transform(scaled_data)
            
            # Print explained variance for regular PCA
            explained_var = self.pca.explained_variance_ratio_
            cumulative_var = np.cumsum(explained_var)
            
            print(f"Regular PCA: Top {self.n_components} components explain {cumulative_var[-1]*100:.2f}% of variance")
            print(f"Individual explained variance: {explained_var}")
        
        # Fit sparse PCA if n_sparse_components > 0
        if self.n_sparse_components > 0:
            # Calculate residuals if regular PCA was performed
            if self.pca is not None:
                pca_transformed = self.pca.transform(scaled_data)
                reconstructed_data = self.pca.inverse_transform(pca_transformed)
                self.pca_residual_matrix = scaled_data - reconstructed_data
                print(f"Residual matrix shape after regular PCA: {self.pca_residual_matrix.shape}")
            else:
                # If no regular PCA was performed, use the original scaled data
                self.pca_residual_matrix = scaled_data
                print(f"Using full matrix for sparse PCA (no regular PCA performed): {self.pca_residual_matrix.shape}")
            
            # Default sparse PCA parameters
            default_sparse_params = {
                'alpha': 1.0,
                'ridge_alpha': 0.01,
                'max_iter': 1000,
                'tol': 1e-6,
                'random_state': 42
            }
            
            # Update with user-provided parameters
            sparse_params = {**default_sparse_params, **self.sparse_pca_params}
            
            # Fit sparse PCA on residuals or original data
            self.sparse_pca = SparsePCA(n_components=self.n_sparse_components, **sparse_params)
            self.sparse_pca.fit_transform(self.pca_residual_matrix)
            
            # Calculate sparsity of components
            if hasattr(self, 'get_sparse_component_sparsity'):
                component_sparsity = self.get_sparse_component_sparsity()
                print(f"Sparsity of components (fraction of zero values): {component_sparsity}")
            
        return self.pca, self.sparse_pca
    
    def transform_trajectories(self):
        """
        Transform the original trajectories into the PCA space and store them back.
        
        Returns:
            dict: Dictionary of transformed trajectories for each model size
        """
        if self.pca is None and self.sparse_pca is None:
            raise ValueError("Neither PCA nor sparse PCA fitted. Call fit_pca() first.")
            
        transformed_data = {}
        
        for model_size in self.model_sizes:
            # Get the original combined data
            df = self.trajectory_data[model_size]
            
            # Extract features (both few-shot and zero-shot)
            feature_values = df[self.combined_columns].values
            
            # Scale if needed
            if self.scaler is not None:
                processed_data = self.scaler.transform(feature_values)
            else:
                processed_data = feature_values
                
            # Initialize transformed DataFrame
            transformed_df = pd.DataFrame()
            
            # Transform using regular PCA if it was fitted
            if self.pca is not None:
                pca_transformed = self.pca.transform(processed_data)
                
                # Create column names for regular PCA components
                pca_component_cols = [f"PC{i+1}" for i in range(pca_transformed.shape[1])]
                
                # Add regular PCA components to DataFrame
                pca_df = pd.DataFrame(pca_transformed, columns=pca_component_cols)
                transformed_df = pd.concat([transformed_df, pca_df], axis=1)
                
                # Calculate residuals for sparse PCA if both were fitted
                if self.sparse_pca is not None:
                    reconstructed_data = self.pca.inverse_transform(pca_transformed)
                    model_residuals = processed_data - reconstructed_data
                else:
                    model_residuals = None
            else:
                # If no regular PCA, use original data for sparse PCA
                model_residuals = processed_data
            
            # Add sparse PCA components if fitted
            if self.sparse_pca is not None and model_residuals is not None:
                # Transform using sparse PCA
                sparse_transformed = self.sparse_pca.transform(model_residuals)
                
                # Create column names for sparse PCA components
                sparse_component_cols = [f"SPC{i+1}" for i in range(sparse_transformed.shape[1])]
                
                # Add sparse components to DataFrame
                sparse_df = pd.DataFrame(sparse_transformed, columns=sparse_component_cols)
                transformed_df = pd.concat([transformed_df, sparse_df], axis=1)
            
            # Add the step column
            transformed_df['step'] = df['step'].values
                
            # Store the transformed data
            transformed_key = f"{model_size}_transformed"
            self.trajectory_data[transformed_key] = transformed_df
            transformed_data[model_size] = transformed_df
            
        return transformed_data
    
    def run_pca_pipeline(self):
        """
        Run the complete PCA pipeline.
        
        Returns:
            dict: Dictionary of transformed trajectories
        """
        self.concatenate_trajectories()
        self.fit_pca()
        transformed_data = self.transform_trajectories()
        
        # Normalize component signs if method exists
        if hasattr(self, 'normalize_component_signs'):
            self.normalize_component_signs()
            
            # After normalization, update transformed data
            transformed_data = {}
            for model_size in self.model_sizes:
                transformed_key = f"{model_size}_transformed"
                if transformed_key in self.trajectory_data:
                    transformed_data[model_size] = self.trajectory_data[transformed_key]
        
        return transformed_data
    
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
    
    def get_feature_loadings(self, pc_index=1, top_n=20, sparse=False):
        """
        Get the top feature loadings for a specific principal component.
        
        Args:
            pc_index (int): Principal component index (1-based)
            top_n (int): Number of top features to return
            sparse (bool): Whether to use sparse PCA components
            
        Returns:
            pd.DataFrame: DataFrame with feature names and loadings
        """
        pc_idx = pc_index - 1  # Convert to 0-based indexing
        
        if sparse:
            if self.sparse_pca is None:
                raise ValueError("Sparse PCA not fitted")
            if pc_idx >= self.sparse_pca.components_.shape[0]:
                raise ValueError(f"Sparse PC index {pc_index} exceeds available components")
            component_loadings = self.sparse_pca.components_[pc_idx]
            component_type = "SPC"
        else:
            if self.pca is None:
                raise ValueError("PCA not fitted")
            if pc_idx >= self.pca.components_.shape[0]:
                raise ValueError(f"PC index {pc_index} exceeds available components")
            component_loadings = self.pca.components_[pc_idx]
            component_type = "PC"
        
        # Create DataFrame with all features and loadings
        loadings_df = pd.DataFrame({
            'Feature': self.combined_columns,
            'Loading': component_loadings
        })
        
        # Sort by absolute loading value
        loadings_df['Abs_Loading'] = abs(loadings_df['Loading'])
        loadings_df = loadings_df.sort_values('Abs_Loading', ascending=False)
        
        # Add column to indicate whether it's from few-shot or zero-shot
        loadings_df['Shot_Type'] = loadings_df['Feature'].apply(
            lambda x: 'Few-Shot' if x.endswith('_few') else 'Zero-Shot'
        )
        
        # Extract original column name without the _few or _zero suffix
        loadings_df['Original_Feature'] = loadings_df['Feature'].apply(
            lambda x: x[:-5] if x.endswith('_few') or x.endswith('_zero') else x
        )

        # Get top features by absolute loading
        top_loadings = loadings_df.head(top_n)
        
        return top_loadings
    
    def normalize_component_signs(self, reference_model=None):
        """
        Normalize PCA component signs so the reference model has positive values at the first step.
        
        Args:
            reference_model (str, optional): Model to use as reference. Default: largest model
        
        Returns:
            bool: Success status
        """
        # Select reference model (default to last/largest model)
        if reference_model is None:
            reference_model = self.model_sizes[-1]
        
        # Get transformed data for reference model
        transformed_key = f"{reference_model}_transformed"
        if transformed_key not in self.trajectory_data:
            print("Warning: Reference model not found in transformed data. Skipping sign normalization.")
            return False
        
        reference_data = self.trajectory_data[transformed_key]
        
        # Get first step row
        if 'step' not in reference_data.columns:
            print("Warning: Step column not found in reference data. Skipping sign normalization.")
            return False
            
        min_step_row = reference_data.loc[reference_data['step'].idxmin()]
        
        # Normalize regular PCA components if they exist
        if self.pca is not None:
            pc_cols = [col for col in reference_data.columns if col.startswith('PC')]
            for pc_col in pc_cols:
                pc_idx = int(pc_col[2:]) - 1  # Extract PC index (0-based)
                first_step_value = min_step_row[pc_col]
                
                # Flip sign if negative
                if first_step_value < 0:
                    self.pca.components_[pc_idx] *= -1
                    for model_size in self.model_sizes:
                        model_key = f"{model_size}_transformed"
                        if model_key in self.trajectory_data and pc_col in self.trajectory_data[model_key].columns:
                            self.trajectory_data[model_key][pc_col] *= -1
        
        # Normalize sparse PCA components if they exist
        if self.sparse_pca is not None:
            spc_cols = [col for col in reference_data.columns if col.startswith('SPC')]
            for spc_col in spc_cols:
                spc_idx = int(spc_col[3:]) - 1
                first_step_value = min_step_row[spc_col]
                
                if first_step_value < 0:
                    self.sparse_pca.components_[spc_idx] *= -1
                    for model_size in self.model_sizes:
                        model_key = f"{model_size}_transformed"
                        if model_key in self.trajectory_data and spc_col in self.trajectory_data[model_key].columns:
                            self.trajectory_data[model_key][spc_col] *= -1
        
        return True
    
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
    
    # Add compatibility property for TrajectoryPlotter
    @property
    def common_columns(self):
        if self.combined_columns is None:
            self.concatenate_trajectories()
        return self.combined_columns
    
    


    
### OTHER DATA STUFF ###


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


def compute_pc_shot_cosine_similarity(pca_handler, sparse=True, top_k=None):
    """
    Compute the cosine similarity between principal components and the feature groups
    (few-shot and zero-shot columns).
    
    Args:
        pca_handler: DualTrajectoryPCA instance with fitted PCA
        sparse (bool): Whether to use sparse PCA components (True) or regular PCA components (False)
        top_k (int, optional): Only show top k components. If None, show all.
        
    Returns:
        pd.DataFrame: Table with cosine similarities
    """
    from scipy.spatial.distance import cosine
    import numpy as np
    import pandas as pd
    
    # Make sure the PCA handler has the required attributes
    if not hasattr(pca_handler, 'few_shot_columns') or not hasattr(pca_handler, 'zero_shot_columns'):
        raise ValueError("Input doesn't appear to be a properly initialized DualTrajectoryPCA object")
    
    # Select the appropriate components based on sparse flag
    if sparse and pca_handler.sparse_pca is not None:
        components = pca_handler.sparse_pca.components_
        pc_prefix = "SPC"
    elif not sparse and pca_handler.pca is not None:
        components = pca_handler.pca.components_
        pc_prefix = "PC"
    else:
        raise ValueError(f"{'Sparse' if sparse else 'Regular'} PCA components not available")
    
    # Limit to top_k components if specified
    if top_k is not None:
        components = components[:top_k]
    
    # Get the feature lists
    few_shot_features = pca_handler.few_shot_columns
    zero_shot_features = pca_handler.zero_shot_columns
    all_features = pca_handler.combined_columns
    
    # Create masks for few-shot and zero-shot columns
    few_mask = np.array([col in few_shot_features for col in all_features])
    zero_mask = np.array([col in zero_shot_features for col in all_features])
    
    # Create unit vectors for few-shot and zero-shot domains
    # (1 where feature belongs to the domain, 0 elsewhere)
    few_vector = np.zeros(len(all_features))
    few_vector[few_mask] = 1
    few_vector = few_vector / np.sqrt(np.sum(few_vector**2))  # Normalize to unit vector
    
    zero_vector = np.zeros(len(all_features))
    zero_vector[zero_mask] = 1
    zero_vector = zero_vector / np.sqrt(np.sum(zero_vector**2))  # Normalize to unit vector
    
    # Calculate cosine similarities
    results = []
    for i, component in enumerate(components):
        # Normalize component
        component_norm = component / np.sqrt(np.sum(component**2))
        
        # Calculate cosine similarity (1 - cosine distance)
        few_sim = 1 - cosine(component_norm, few_vector)
        zero_sim = 1 - cosine(component_norm, zero_vector)
        
        # Calculate bias toward one domain vs the other
        # Positive means bias toward few-shot, negative means bias toward zero-shot
        bias = few_sim - zero_sim
        
        results.append({
            "Component": f"{pc_prefix}{i+1}",
            "Few-Shot Similarity": few_sim,
            "Zero-Shot Similarity": zero_sim,
            "Bias (Few - Zero)": bias
        })
    
    # Create DataFrame
    result_df = pd.DataFrame(results)
    
    # Format for display
    pd.set_option('display.float_format', '{:.4f}'.format)
    
    return result_df

def print_pc_shot_cosine_table(pca_handler, sparse=True, regular=True, top_k=None):
    """
    Calculate and print cosine similarities between PCs and shot types.
    
    Args:
        pca_handler: DualTrajectoryPCA instance with fitted PCA
        sparse (bool): Whether to include sparse PCA components
        regular (bool): Whether to include regular PCA components
        top_k (int, optional): Only show top k components. If None, show all.
    
    Returns:
        None (prints to console)
    """
    if regular and pca_handler.pca is not None:
        print("=" * 80)
        print("REGULAR PCA COMPONENTS - COSINE SIMILARITY WITH SHOT TYPES")
        print("=" * 80)
        df_regular = compute_pc_shot_cosine_similarity(pca_handler, sparse=False, top_k=top_k)
        # Sort by PC number (extract number from string and convert to int)
        df_regular['PC_Number'] = df_regular['Component'].str.extract(r'(\d+)').astype(int)
        df_sorted = df_regular.sort_values('PC_Number').drop(columns=['PC_Number'])
        print(df_sorted.to_string(index=False))
        print("\n")
    
    if sparse and pca_handler.sparse_pca is not None:
        print("=" * 80)
        print("SPARSE PCA COMPONENTS - COSINE SIMILARITY WITH SHOT TYPES")
        print("=" * 80)
        df_sparse = compute_pc_shot_cosine_similarity(pca_handler, sparse=True, top_k=top_k)
        # Sort by PC number (extract number from string and convert to int)
        df_sparse['PC_Number'] = df_sparse['Component'].str.extract(r'(\d+)').astype(int)
        df_sorted = df_sparse.sort_values('PC_Number').drop(columns=['PC_Number'])
        print(df_sorted.to_string(index=False))
        print("\n")