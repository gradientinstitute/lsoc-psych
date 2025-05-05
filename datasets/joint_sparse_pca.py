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
import time
import wandb
import gc
import io
import sys
import contextlib
import re


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

class FilteredStdout:
    """
    A stdout filter that only allows through specific lines
    while capturing all output for later analysis.
    """
    def __init__(self, original_stdout=sys.stdout):
        self.original_stdout = original_stdout
        self.captured_text = StringIO()
    
    def write(self, text):
        # Always capture everything for later analysis
        self.captured_text.write(text)
        
        # Only write to the original stdout if it's not a parallel processing message
        if "[Parallel" not in text and (text.strip() == "" or "Iteration" in text):
            self.original_stdout.write(text)
    
    def flush(self):
        self.original_stdout.flush()
    
    def get_captured_text(self):
        return self.captured_text.getvalue()


class SparseTrajectoryPCA:
    """
    Class for performing sparse PCA on concatenated trajectory data across multiple model sizes.
    
    This class handles:
    1. Concatenating trajectory data from different model sizes
    2. Applying sparse PCA to the concatenated data
    3. Transforming the original trajectories into the sparse PCA space
    4. Storing the transformed trajectories back in the original data structure
    """
    
    def __init__(self, trajectory_data: Dict[str, pd.DataFrame], 
             step_range=[None, None], 
             n_sparse_components: int = 10, scale: bool = False,
             sparse_pca_params: Optional[Dict] = None, run_at_init: bool = True,
             dataset_name=None):
        """
        Initialize SparseTrajectoryPCA with trajectory data.
        
        Args:
            trajectory_data (dict): Dictionary where keys are model sizes and 
                                    values are the corresponding dataframes
            step_range (list): [min_step, max_step] to include in the analysis
            n_sparse_components (int): Number of sparse PCA components to extract
            scale (bool): Whether to standardize the data before PCA
            sparse_pca_params (dict, optional): Parameters for the sparse PCA
            run_at_init (bool): Whether to run the PCA pipeline during initialization
            dataset_name (str, optional): Name of the dataset for reference
            num_contexts (int, optional): Number of contexts included in the data
        """
        # Input data and parameters
        self.trajectory_data = {}
        self.step_range = step_range
        self.n_sparse_components = n_sparse_components
        self.scale = scale
        self.dataset_name = dataset_name

        # Default sparse PCA parameters
        default_sparse_params = {
            'alpha': 1.0,  # L1 penalty parameter
            'ridge_alpha': 0.01,  # Ridge penalty parameter
            'max_iter': 1000,
            'tol': 1e-6,
            'random_state': 42, # stablisises initalisation too
            'n_jobs':-1,
            'method': 'cd' # better for dense data
        }
        
        # Update with user-provided parameters
        sparse_pca_params = sparse_pca_params if sparse_pca_params is not None else {}
        self.sparse_params = {**default_sparse_params, **sparse_pca_params}
        
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

        self.optimization_history = []

        # Model attributes
        self.model_sizes = list(trajectory_data.keys())
        self.sparse_pca = None
        self.scaler = None
        
        # Data containers
        self.common_columns = None
        self.concatenated_matrix = None
        self.raw_concatenated_matrix = None
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

    def parse_sparse_pca_output(self, output_text, sparse_pca, start_time):
        """
        Parse the verbose output from SparsePCA and extract iteration costs.
        
        Args:
            output_text (str): The captured stdout text from SparsePCA verbose output
            sparse_pca (SparsePCA): The fitted SparsePCA object
            scaled_data (np.ndarray): The data that was used for fitting
            start_time (float): The time when the fitting started
            
        Returns:
            list: List of dictionaries containing iteration metrics
        """
        iteration_costs = []
        
        # Regular expression to extract iteration and cost from output
        # Updated regex pattern to match the actual output format
        cost_pattern = re.compile(r'Iteration\s+(\d+).*?current cost\s+([\d.nan]+)', re.MULTILINE | re.DOTALL)
        matches = cost_pattern.findall(output_text)

        # Add this right after the regex pattern in fit_sparse_pca, replace the existing component_sparsity calculation
        component_sparsity = {f"sparsity_spc{i+1:02d}": 1.0 - (np.count_nonzero(self.sparse_pca.components_[i]) / len(self.sparse_pca.components_[i])) 
                              for i in range(self.n_sparse_components)}
        
        # Create the optimization history
        for iteration, cost in matches:
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Calculate sparsity (this will be final sparsity for all iterations)
            mean_sparsity = 1.0 - (np.count_nonzero(sparse_pca.components_) / sparse_pca.components_.size)
            
            iteration_costs.append({
                'iteration': int(iteration),
                'cost': float(cost),
                'elapsed_time': elapsed,
                'mean_sparsity': mean_sparsity,
                **component_sparsity,
            })
        
        return iteration_costs

    def fit_sparse_pca(self) -> SparsePCA:
        """
        Fit sparse PCA on the concatenated trajectory data with real-time filtered output.
        
        Returns:
            SparsePCA: Fitted SparsePCA object
        """
        if self.concatenated_matrix is None:
            self.concatenate_trajectories()
            
        # Scale the data if required
        if self.scale:
            self.scaler = StandardScaler()
            scaled_data = self.scaler.fit_transform(self.concatenated_matrix)
        else:
            scaled_data = self.concatenated_matrix

        # Start tracking time
        start_time = time.time()
        
        # Set up the filtered stdout
        filtered_stdout = FilteredStdout()
        original_stdout = sys.stdout
        sys.stdout = filtered_stdout
        
        try:
            # Fit sparse PCA on data
            self.sparse_pca = SparsePCA(n_components=self.n_sparse_components, **self.sparse_params)
            self.sparse_pca.fit(scaled_data)
        finally:
            # Restore stdout even if an exception occurs
            sys.stdout = original_stdout
        
        # Get the captured output text
        output_text = filtered_stdout.get_captured_text()
        
        # Parse the output to extract iteration costs
        self.optimization_history = pd.DataFrame(self.parse_sparse_pca_output(
            output_text, 
            self.sparse_pca, 
            start_time
        ))
        
        # Calculate sparsity of components
        component_sparsity = self.get_sparse_component_sparsity()
            
        print(f"Sparse PCA: {self.n_sparse_components} components extracted")
        print(f"Sparsity of components (fraction of zero values): {component_sparsity}")
        print(f"Number of iterations captured: {len(self.optimization_history)}")
            
        return self.sparse_pca
    
    def transform_trajectories(self) -> Dict[str, pd.DataFrame]:
        """
        Transform the original trajectories into the sparse PCA space
        and store them back in the trajectory data dictionary.
        
        Returns:
            dict: Dictionary of transformed trajectories for each model size
        """
        if self.sparse_pca is None:
            raise ValueError("Sparse PCA not fitted. Call fit_sparse_pca() first.")
            
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
            
            # Transform using sparse PCA
            sparse_transformed = self.sparse_pca.transform(processed_data)
            
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
        Run the complete sparse PCA pipeline using the instance parameters.
        
        Returns:
            dict: Dictionary of transformed trajectories for each model size
        """
        self.find_common_columns()
        self.concatenate_trajectories()
        self.fit_sparse_pca()
        transformed_data = self.transform_trajectories()
        self.normalize_component_signs()  # Modify the transformed data
        
        # After normalization, collect the transformed data to return
        normalized_transformed_data = {}
        for model_size in self.model_sizes:
            transformed_key = f"{model_size}_transformed"
            if transformed_key in self.trajectory_data:
                normalized_transformed_data[model_size] = self.trajectory_data[transformed_key]
        
        return normalized_transformed_data
    
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
        Estimate the variance explained by sparse PCA components.
        
        Args:
            n_samples (int): Number of samples to use for variance estimation
            
        Returns:
            np.ndarray: Array of explained variance ratios
        """
        if self.sparse_pca is None or self.concatenated_matrix is None:
            return None
        
        # Scale data if needed
        if self.scale:
            if self.scaler is None:
                scaler = StandardScaler()
                data = scaler.fit_transform(self.concatenated_matrix)
            else:
                data = self.scaler.transform(self.concatenated_matrix)
        else:
            data = self.concatenated_matrix
        
        # Use a subset of samples if the matrix is very large
        if data.shape[0] > n_samples:
            indices = np.random.choice(data.shape[0], n_samples, replace=False)
            data_subset = data[indices]
        else:
            data_subset = data
        
        # Calculate total variance
        total_variance = np.var(data_subset, axis=0).sum()
        
        # Transform the subset using sparse PCA
        transformed = self.sparse_pca.transform(data_subset)
        
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
        Normalize sparse PCA component signs so the reference model (default: largest) 
        has positive values at the first step.
        
        Args:
            reference_model (str, optional): Model to use as reference
            
        Returns:
            bool: Whether normalization was successful
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
        
        # Normalize sparse PCA components
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
        Generate a table showing loadings of specific columns on sparse PCs.
        
        Args:
            columns_of_interest (list): List of column names to include (e.g., "context_0_pos_1")
        
        Returns:
            pd.DataFrame: Table with columns as rows and components as columns
        """
        # Initialize the results DataFrame with feature names
        results = pd.DataFrame(columns_of_interest, columns=['Feature'])
        results.set_index('Feature', inplace=True)
        
        # Process sparse PCA components
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
    
    def compute_cosine_with_spc(self, columns_of_interest, spc_idx=6):
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
        
        # Get all feature names
        feature_names = self.common_columns
        
        # Create one-hot vector for columns of interest (1 at the column's position, 0 elsewhere)
        token_vector = np.zeros(len(feature_names))
        
        # Set 1 for each column of interest
        for col in columns_of_interest:
            if col in feature_names:
                idx = feature_names.index(col)
                token_vector[idx] = 1
        
        # Get the sparse PC vector (0-indexed, so SPC6 is at index 5)
        spc_vector = self.sparse_pca.components_[spc_idx-1]
        
        # Compute cosine similarity (1 - cosine distance)
        similarity = 1 - cosine(token_vector, spc_vector)
        
        return similarity
    

def run_sparse_pca_sweep(
    trajectory_data,
    experiment_name,
    dataset_name,
    alphas=[0.1, 0.5, 1.0, 2.0, 5.0],
    ridge_alphas=[0.001, 0.01, 0.1],
    num_components_list=[5, 10, 20, 50],
    models_dir="./sparse_pca_models",
    wandb_project="sparse-pca-sweep",
    wandb_entity=None  # Your wandb username or team name
):
    """
    Run a simple hyperparameter sweep for SparseTrajectoryPCA with W&B logging.
    
    Args:
        trajectory_data: Dictionary of trajectory dataframes by model size
        experiment_name: Name of the experiment
        dataset_name: Name of the dataset
        alphas: List of alpha values to try
        ridge_alphas: List of ridge_alpha values to try
        components_list: List of n_components values to try
        models_dir: Directory to save model components
        wandb_project: W&B project name
        wandb_entity: W&B entity (username or team name)
    
    Returns:
        List of run IDs that were completed
    """
    # Create output directory for model components
    os.makedirs(models_dir, exist_ok=True)
    
    # Initialize W&B
    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        config={
            "experiment": experiment_name,
            "dataset": dataset_name,
            "alphas": alphas,
            "ridge_alphas": ridge_alphas,
            "components_list": num_components_list
        },
        name=f"{experiment_name}_{dataset_name}_sweep"
    )
    
    # Total configurations for progress tracking
    total_configs = len(alphas) * len(ridge_alphas) * len(num_components_list)
    print(f"Starting hyperparameter sweep with {total_configs} configurations")
    
    # List to track completed runs
    completed_runs = []
    
    # Run sweep
    completed = 0
    
    for n_components in num_components_list:
        for alpha in alphas:
            for ridge_alpha in ridge_alphas:
                # Create run ID
                run_id = f"n{n_components:03d}_a{alpha:.1f}_r{ridge_alpha:.4f}"
                
                # Create model file path
                model_file = os.path.join(models_dir, f"{run_id}.pkl")
                
                # Check if this model file already exists (resume capability)
                if os.path.exists(model_file):
                    print(f"Model for run {run_id} already exists, skipping")
                    completed += 1
                    completed_runs.append(run_id)
                    continue
                
                print(f"Starting run {run_id}")
                start_time = time.time()
                
                # Set up the SparsePCA parameters
                sparse_pca_params = {
                    'alpha': alpha,
                    'ridge_alpha': ridge_alpha
                }
                
                # Run sparse PCA
                sparse_pca = SparseTrajectoryPCA(
                    trajectory_data=trajectory_data,
                    n_sparse_components=n_components,
                    scale=True,
                    sparse_pca_params=sparse_pca_params,
                    run_at_init=True,
                    dataset_name=dataset_name
                )
                
                # Calculate metrics
                elapsed_time = time.time() - start_time
                component_sparsity = sparse_pca.get_sparse_component_sparsity()
                mean_sparsity = np.mean(component_sparsity)
                
                # Try to get explained variance
                try:
                    explained_variance = sparse_pca.get_sparse_component_variance()
                    total_explained_variance = sum(explained_variance)
                except:
                    explained_variance = None
                    total_explained_variance = None
                
                # Get number of iterations
                n_iterations = sparse_pca.optimization_history
                
                # Save only the essential components
                sparse_results = {
                    'sparse_pca': sparse_pca.sparse_pca,  # Just the sklearn SparsePCA object
                    'scaler': sparse_pca.scaler,
                    'common_columns': sparse_pca.common_columns,
                    'model_sizes': sparse_pca.model_sizes
                }
                
                with open(model_file, 'wb') as f:
                    pickle.dump(sparse_results, f)

                elapsed_time_list = sparse_pca.optimization_history['elapsed_time'].values.tolist()
                cost_history = sparse_pca.optimization_history['cost'].values.tolist()
                
                # Log to W&B
                run_data = {
                    'alpha': alpha,
                    'ridge_alpha': ridge_alpha,
                    'n_components': n_components,
                    'elapsed_time': elapsed_time,
                    'mean_sparsity': mean_sparsity,
                    'n_iterations': n_iterations,
                    'model_file': model_file,
                    'cost_history': cost_history,
                    'elapsed_time_list': elapsed_time_list,
                }
                
                # Add component-specific metrics
                for i, sparsity in enumerate(component_sparsity):
                    run_data[f'component_{i+1}_sparsity'] = sparsity
                
                # Add explained variance if available
                if explained_variance is not None:
                    run_data['total_explained_variance'] = total_explained_variance
                    for i, var in enumerate(explained_variance):
                        run_data[f'component_{i+1}_explained_variance'] = var
                
                # Log optimization history if available
                if hasattr(sparse_pca, 'optimization_history') and sparse_pca.optimization_history is not None:
                    # Create a table for optimization history
                    columns = ["iteration", "cost", "mean_sparsity"]
                    optim_data = []
                    
                    for i, row in enumerate(sparse_pca.optimization_history.to_dict('records')):
                        optim_data.append([row.get('iteration', i), row.get('cost', 0), row.get('mean_sparsity', 0)])
                    
                    # Log as a table
                    run_data['optimization_history'] = wandb.Table(
                        columns=columns,
                        data=optim_data
                    )
                
                # Log to W&B
                wandb.log(run_data)
                
                completed += 1
                print(f"Completed run {run_id} in {elapsed_time:.2f} seconds ({completed}/{total_configs})")
                
                # Track completed runs
                completed_runs.append(run_id)
                
                # Clean up memory
                del sparse_pca
                gc.collect()
    
    print("Hyperparameter sweep completed!")
    print(f"Models saved to: {models_dir}")
    
    # Finish W&B run
    wandb.finish()
    
    return completed_runs

def load_sparse_pca_model(model_file):
    """
    Load a previously saved SparsePCA model.
    
    Args:
        model_file: Path to the saved model file
    
    Returns:
        dict: Dictionary containing the SparsePCA model and metadata
    """
    with open(model_file, 'rb') as f:
        return pickle.load(f)