import os
import pandas as pd
import numpy as np
import glob
from tqdm import tqdm
from sklearn.decomposition import PCA, SparsePCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ridge_regression
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


def block_train_test_split(n_samples, block_size, test_size=0.2, random_state=42):
    """Like a shuffled train_test_split but selects contiguous blocks to handle autocorrelation
    
    Args:
        n_samples (int): Total number of samples
        block_size (int): Size of contiguous blocks
        test_size (float): Fraction of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_indices, test_indices)
    """
    # Select a shuffled set of blocks
    n_blocks = int(np.ceil(n_samples / block_size))
    rng = np.random.RandomState(random_state)
    shuffled_blocks = rng.permutation(n_blocks)
    
    # Create a boolean array indicating which blocks are for training
    block_is_train = np.ones(n_blocks, dtype=bool)
    n_test_blocks = int(np.ceil(n_blocks * test_size))
    block_is_train[shuffled_blocks[:n_test_blocks]] = False
    
    # Map each sample to its corresponding block's train/test status
    block_indices = np.arange(n_samples) // block_size
    is_train = block_is_train[block_indices]
    
    # Get indices
    train_indices = np.where(is_train)[0]
    test_indices = np.where(~is_train)[0]
    
    return train_indices, test_indices


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
    Class for performing sparse PCA on concatenated trajectory data across multiple model sizes,
    with optional bi-cross-validation for hyperparameter selection.
    """
    
    def __init__(self, 
                 trajectory_data: Dict[str, pd.DataFrame], 
                 step_range=[None, None], 
                 n_sparse_components: int = 10, 
                 scale: bool = True,
                 sparse_pca_params: Optional[Dict] = None, 
                 cross_val_params: Optional[Dict] = None,
                 transform_trajectories: bool = True,
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
            cross_val_params (dict, optional): Parameters for bi-cross-validation.
                                              If None, no cross-validation is performed.
            transform_trajectories (bool): Whether to transform the original trajectories
            dataset_name (str, optional): Name of the dataset for reference
        """
        # Input data and parameters
        self.trajectory_data = {}
        self.step_range = step_range
        self.n_sparse_components = n_sparse_components
        self.scale = scale
        self.dataset_name = dataset_name
        self.transform_trajectories_flag = transform_trajectories

        # Default sparse PCA parameters
        default_sparse_params = {
            'alpha': 1.0,         # L1 penalty parameter
            'ridge_alpha': 0.01,  # Ridge penalty parameter
            'max_iter': 1000,
            'tol': 1e-6,
            'random_state': 42,   # Stabilizes initialization
            'n_jobs': -1,
            'method': 'cd',        # Better for dense data
            'verbose': True
        }
        
        # Update with user-provided parameters
        sparse_pca_params = sparse_pca_params if sparse_pca_params is not None else {}
        self.sparse_params = {**default_sparse_params, **sparse_pca_params}
        
        # Default cross-validation parameters
        self.use_cross_val = cross_val_params is not None
        if self.use_cross_val:
            default_cross_val_params = {
                'row_frac': 0.3,
                'col_frac': 0.3,
                'row_autocorr': 2,
                'seed': 42
            }
            self.cross_val_params = {**default_cross_val_params, **(cross_val_params or {})}
        
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

        # Results container
        self.results = {
            'optimization_history': {
                'iterations': [],
                'cost_history': [],
                'time_elapsed': []
            },
            'total_proportion_sparsity': None,
            'component_sparsity': None,
            'bi_cross_val_loss': None,
            'total_reconstruction_error': None,
            'total_variance': None
        }

        # Model attributes
        self.model_sizes = list(trajectory_data.keys())
        # Initialize SparsePCA model
        self.sparse_pca = SparsePCA(n_components=self.n_sparse_components, **self.sparse_params)
        self.scaler = None
        
        # Data containers
        self.common_columns = None
        self.concatenated_matrix = None
        self.raw_concatenated_matrix = None
        self.row_indices = None
        self.train_indices = None
        self.test_indices = None
        self.col_train_indices = None
        self.col_test_indices = None
        
        # Run the PCA pipeline
        self.run_pca_pipeline()
        
    def run_pca_pipeline(self):
        """Run the complete sparse PCA pipeline using the instance parameters."""
        # Pre-process data
        self.preprocess_data()
        
        # Fit sparse PCA (with or without cross-validation)
        self.fit_sparse_pca()
        
        # Calculate and store metrics
        self.calculate_metrics()
        
        # Transform trajectories if requested
        if self.transform_trajectories_flag:
            transformed_data = self.transform_trajectories()
            self.normalize_component_signs()
            return transformed_data
            
        return None
    
    def preprocess_data(self):
        """
        Prepare the data for sparse PCA by finding common columns,
        concatenating trajectories, and optionally setting up train-test splits.
        """
        self.find_common_columns()
        self.concatenate_trajectories()
        
        if self.use_cross_val:
            n_rows, n_cols = self.concatenated_matrix.shape
            print('concat shape')
            print(self.concatenated_matrix.shape)
            row_frac = self.cross_val_params['row_frac']
            col_frac = self.cross_val_params['col_frac']
            row_autocorr = self.cross_val_params['row_autocorr']
            seed = self.cross_val_params['seed']
            
            # Create row splits using block-aware split for temporal data
            self.train_indices, self.test_indices = block_train_test_split(
                n_rows, row_autocorr, test_size=row_frac, random_state=seed
            )
            # = r1, r0
            
            # Create column splits using standard train-test split
            self.col_train_indices, self.col_test_indices = train_test_split(
                np.arange(n_cols), test_size=col_frac, random_state=seed+1
            )

            print('col train indices')
            print(sorted(self.col_train_indices))
            print(len(self.col_train_indices))

            print('col test indices')
            print(sorted(self.col_test_indices))
            print(len(self.col_test_indices))


            # = c1, c0

            # r1, r0 = train_indices, test_indices
            # c1, c0 = col_train_indices, col_test_indices
            # X = (A B)
            #     (C D)
            # A = X[np.ix_(r0, c0)] = X[np.ix_(self.test_indices, self.col_test_indices)]
            # B = X[np.ix_(r0, c1)] = X[np.ix_(self.test_indices, self.col_train_indices)]
            # CD = X[r1] = X[self.train_indices]
            
            print(f"Cross-validation setup: {len(self.train_indices)} train rows, {len(self.test_indices)} test rows")
            print(f"Cross-validation setup: {len(self.col_train_indices)} train columns, {len(self.col_test_indices)} test columns")
    
    def find_common_columns(self):
        """Find columns that are common across all model size dataframes."""
        if not self.model_sizes:
            raise ValueError("No model sizes found in trajectory data")
            
        # Start with all columns from the first model
        common_cols = set(self.trajectory_data[self.model_sizes[0]].columns)
        
        # Intersect with columns from other models
        for model_size in self.model_sizes[1:]:
            model_cols = set(self.trajectory_data[model_size].columns)
            common_cols = common_cols.intersection(model_cols)
        
        # Convert back to list and remove 'step' if it exists (handle separately)
        common_cols = list(common_cols)
        if 'step' in common_cols:
            common_cols.remove('step')
            
        if not common_cols:
            raise ValueError("No common columns found across model sizes")
            
        print(f"Found {len(common_cols)} common columns across all {len(self.model_sizes)} model sizes")
        self.common_columns = common_cols
        return common_cols
    
    def concatenate_trajectories(self):
        """Concatenate trajectories from all model sizes into a single matrix."""
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
    
    def fit_sparse_pca(self):
        """
        Fit sparse PCA on the concatenated trajectory data.
        If cross-validation is enabled, fits on training data only.
        """
        if self.concatenated_matrix is None:
            self.concatenate_trajectories()
            
        # Scale the data if required
        if self.scale:
            self.scaler = StandardScaler()
            if self.use_cross_val:
                # If using cross-validation, fit scaler on training data only
                train_data = self.concatenated_matrix[self.train_indices]
                self.scaler.fit(train_data)
                scaled_data = self.scaler.transform(self.concatenated_matrix)
            else:
                scaled_data = self.scaler.fit_transform(self.concatenated_matrix)
        else:
            scaled_data = self.concatenated_matrix

        # Start tracking time
        start_time = time.time()
        
        # Set up the filtered stdout to capture optimization progress
        filtered_stdout = FilteredStdout()
        original_stdout = sys.stdout
        sys.stdout = filtered_stdout

        # if self.use_cross_val:
        #     # If using cross-validation, fit on training data only (CD quadrant)
        #     train_data = scaled_data[self.train_indices]
        #     print('train data shape')
        #     print(train_data.shape)
        #     self.sparse_pca.fit(train_data)
        # else:
        #     print('got here instead')
        #     # Otherwise, fit on all data
        #     self.sparse_pca.fit(scaled_data)
        
        try:
            if self.use_cross_val:
                # If using cross-validation, fit on training data only (CD quadrant)
                train_data = scaled_data[self.train_indices]
                self.sparse_pca.fit(train_data)
            else:
                # Otherwise, fit on all data
                self.sparse_pca.fit(scaled_data)
        finally:
            # Restore stdout even if an exception occurs
            sys.stdout = original_stdout
        
        # Get the captured output text
        output_text = filtered_stdout.get_captured_text()

        print(output_text)
        
        # Parse the output to extract iteration costs
        self.parse_sparse_pca_output(output_text, start_time)
        
        if self.use_cross_val:
            # Calculate the bi-cross-validation loss if needed
            self.calculate_bi_cross_val_loss(scaled_data)
            
        return self.sparse_pca
    
    def calculate_bi_cross_val_loss(self, scaled_data):
        """
        Calculate bi-cross-validation loss for the fitted sparse PCA model.
        
        Submatrix holdout validation methodology:
        X = (A B)
            (C D)
        - Train on D
        - Use components from D to score C (get latent factors)
        - Use these latent factors to reconstruct B
        - Compare reconstructed B with actual B to compute validation loss
        """
        # r1, r0 = train_indices, test_indices
        # c1, c0 = col_train_indices, col_test_indices
        # X = (A B)
        #     (C D)
        # A = X[np.ix_(r0, c0)] = X[np.ix_(self.test_indices, self.col_test_indices)]
        # B = X[np.ix_(r0, c1)] = X[np.ix_(self.test_indices, self.col_train_indices)]
        # CD = X[r1] = X[self.train_indices]

        # Extract the relevant quadrants
        # A - test rows, test columns (completely held out)
        A = scaled_data[np.ix_(self.test_indices, self.col_test_indices)]
        
        # B - test rows, train columns
        B = scaled_data[np.ix_(self.test_indices, self.col_train_indices)]
        
        # Extract factorisation components corresponding to D
        sub_components = self.sparse_pca.components_[:, self.col_train_indices]
        sub_mean = self.sparse_pca.mean_[self.col_train_indices] if hasattr(self.sparse_pca, 'mean_') else 0
        
        # Apply ridge regression to get scores for B
        B_centered = B - sub_mean
        AB_score = ridge_regression(
            sub_components.T, B_centered.T, self.sparse_pca.ridge_alpha, solver="cholesky"
        )
        
        # Extrapolate these scores to reconstruct A
        AB_pred = self.sparse_pca.inverse_transform(AB_score)
        A_pred = AB_pred[:, self.col_test_indices]

        # Calculate MSE
        mse = np.mean((A - A_pred) ** 2)
        self.results['bi_cross_val_loss'] = mse
        
        print(f"Bi-cross-validation MSE: {mse:.6f}")
        return mse
    
    def calculate_reconstruction_error(self):
        """Calculate the total reconstruction error of the sparse PCA model on the data."""

        # Get the data
        if self.scale:
            data = self.scaler.transform(self.concatenated_matrix)
        else:
            data = self.concatenated_matrix
        
        # Transform data to latent space
        latent = self.sparse_pca.transform(data)
        
        # Inverse transform to get reconstructed data
        reconstructed = self.sparse_pca.inverse_transform(latent)
        
        # Calculate MSE
        mse = np.mean((data - reconstructed) ** 2)
        return mse
    
    def calculate_metrics(self):
        """Calculate and store key metrics for the model."""
        if self.sparse_pca is None:
            return
        
        # Calculate sparsity metrics
        non_zero = np.count_nonzero(self.sparse_pca.components_)
        total_elements = self.sparse_pca.components_.size
        total_sparsity = 1.0 - (non_zero / total_elements)
        
        component_sparsity = []
        for component in self.sparse_pca.components_:
            comp_non_zero = np.count_nonzero(component)
            comp_sparsity = 1.0 - (comp_non_zero / len(component))
            component_sparsity.append(comp_sparsity)
        
        # Calculate reconstruction error
        rec_error = self.calculate_reconstruction_error()
        explained_variance = 1 - rec_error
        
        # Calculate total variance
        if self.scale:
            data = self.scaler.transform(self.concatenated_matrix)
        else:
            data = self.concatenated_matrix
            
        total_variance = np.mean((data - data.mean())**2)
        
        # Store the results
        self.results['total_proportion_sparsity'] = total_sparsity
        self.results['component_sparsity'] = component_sparsity
        self.results['explained_varianced'] = explained_variance
        self.results['total_variance'] = total_variance
        
        print(f"Sparse PCA: {self.n_sparse_components} components extracted")
        print(f"Overall sparsity: {total_sparsity:.4f} (fraction of zero values)")
        print(f"Reconstruction error: {rec_error:.6f}")
        print(f"Total data variance: {total_variance:.6f}")
    
    def transform_trajectories(self):
        """
        Transform the original trajectories into the sparse PCA space
        and store them back in the trajectory data dictionary.
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
    
    def parse_sparse_pca_output(self, output_text, start_time):
        """
        Parse the verbose output from SparsePCA and extract iteration costs.
        
        Args:
            output_text (str): The captured stdout text from SparsePCA verbose output
            start_time (float): The time when the fitting started
        """
        # Regular expression to extract iteration and cost from output
        cost_pattern = re.compile(r'Iteration\s+(\d+).*?current cost\s+([\d.nan]+)', re.MULTILINE | re.DOTALL)
        matches = cost_pattern.findall(output_text)
        
        iterations = []
        costs = []
        times = []
        
        for iteration, cost in matches:
            current_time = time.time()
            elapsed = current_time - start_time
            
            iterations.append(int(iteration))
            costs.append(float(cost))
            times.append(elapsed)
        
        self.results['optimization_history']['iterations'] = iterations
        self.results['optimization_history']['cost_history'] = costs
        self.results['optimization_history']['time_elapsed'] = times
        
        print(f"Number of iterations captured: {len(iterations)}")
        
    # HELPER FUNCTIONS
    
    def normalize_component_signs(self, reference_model=None):
        """
        Normalize sparse PCA component signs so the reference model (default: largest) 
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
        """
        results = pd.DataFrame(columns_of_interest, columns=['Feature'])
        results.set_index('Feature', inplace=True)
        
        if self.sparse_pca is not None:
            feature_names = self.common_columns
            
            for i in range(self.sparse_pca.components_.shape[0]):
                spc_col = f"SPC{i+1}"
                
                spc_loadings = {}
                for feature in columns_of_interest:
                    if feature in feature_names:
                        feature_idx = feature_names.index(feature)
                        spc_loadings[feature] = self.sparse_pca.components_[i, feature_idx]
                
                results[spc_col] = pd.Series(spc_loadings)
        
        results.reset_index(inplace=True)
        return results


def train_sparse_pca():
    """Training function for wandb sweep agent."""
    # Initialize a new wandb run
    run = wandb.init()
    
    # Get hyperparameters from wandb config
    config = wandb.config
    
    # Create run ID based on parameters
    run_id = f"{wandb.config['dataset_name']}_n={config.n_components:03d}_a={config.alpha:.1f}_r={config.ridge_alpha:.4f}"
    wandb.run.name = run_id

    # Create model file path
    model_file = os.path.join(config.models_dir, f"{run_id}.pkl")
    
    # Check if this model file already exists (resume capability)
    if os.path.exists(model_file):
        print(f"Model for run {run_id} already exists, loading metrics")
        with open(model_file, 'rb') as f:
            sparse_results = pickle.load(f)
            
            # Log all metrics from the results dictionary directly
            results_dict = sparse_results['results']
            wandb.log(results_dict)
            
            # Add elapsed time if available
            if 'elapsed_time' in sparse_results:
                wandb.log({'elapsed_time': sparse_results['elapsed_time']})
                
            return
    
    print(f"Starting run {run_id}")
    start_time = time.time()
    
    # Load trajectory data from the path specified in config
    with open(config.trajectory_data_path, 'rb') as f:
        print(f"Loading trajectory data from {config.trajectory_data_path}")
        trajectory_data = pickle.load(f)
    
    # Set up the SparsePCA parameters
    sparse_pca_params = {
        'alpha': config.alpha,
        'ridge_alpha': config.ridge_alpha,
        'max_iter': config.max_iter,
    }
    
    # Run sparse PCA
    sparse_pca = SparseTrajectoryPCA(
        trajectory_data=trajectory_data,
        n_sparse_components=config.n_components,
        scale=config.scale if hasattr(config, 'scale') else False,
        sparse_pca_params=sparse_pca_params,
        cross_val_params=config.cross_val_params if hasattr(config, 'cross_val_params') else None,
        transform_trajectories=False,
        dataset_name=config.dataset_name
    )
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Save essential components
    sparse_results = {
        'sparse_pca': sparse_pca.sparse_pca,
        'scaler': sparse_pca.scaler,
        'common_columns': sparse_pca.common_columns,
        'model_sizes': sparse_pca.model_sizes,
        'results': sparse_pca.results,
        'elapsed_time': elapsed_time
    }
    
    if not config.is_test:
        with open(model_file, 'wb') as f:
            pickle.dump(sparse_results, f)
    
    # Log all metrics directly from the results dictionary
    wandb.log(sparse_pca.results)
    
    # Add additional metadata
    wandb.log({
        'elapsed_time': elapsed_time,
        'model_file': model_file,
    })
    
    # Log optimization history as a line plot
    iterations = sparse_pca.results['optimization_history']['iterations']
    costs = sparse_pca.results['optimization_history']['cost_history']
    
    wandb.log({
        "optimization_plot": wandb.plot.line_series(
            xs=[iterations],
            ys=[costs],
            keys=["cost"],
            title="Optimization Cost History",
            xname="Iteration"
        )
    })
    
    # Clean up memory
    del sparse_pca, trajectory_data
    gc.collect()


def configure_sparse_pca_sweep(
    trajectory_data,
    experiment_name,
    dataset_name,
    alphas=[0.1, 0.5, 1.0, 2.0, 5.0],
    ridge_alphas=[0.001, 0.01, 0.1],
    num_components_list=[5, 10, 20, 50],
    max_iter=100,
    cross_val_params=None,
    models_dir="./sparse_pca_models",
    wandb_project="sparse-pca-sweep",
    wandb_entity=None,
    scale=False,
    sweep_name=None,
    is_test=False
):
    """
    Configure a native wandb sweep for SparseTrajectoryPCA.
    
    Args:
        trajectory_data: Dictionary of trajectory dataframes by model size
        experiment_name: Name of the experiment
        dataset_name: Name of the dataset
        alphas: List of alpha values to try
        ridge_alphas: List of ridge_alpha values to try
        num_components_list: List of n_components values to try
        cross_val_params: Parameters for bi-cross-validation
        models_dir: Directory to save model components
        wandb_project: W&B project name
        wandb_entity: W&B entity (username or team name)
        scale: Whether to scale the data
    
    Returns:
        str: Sweep ID
    """
    # Create output directory for model components
    os.makedirs(models_dir, exist_ok=True)
    
    # Save trajectory_data to a pickle file that can be loaded by the training function
    data_file = os.path.join(models_dir, f"{experiment_name}_{dataset_name}_trajectory_data.pkl")
    with open(data_file, 'wb') as f:
        pickle.dump(trajectory_data, f)
    
    print(f"Trajectory data saved to {data_file}")
    
    # Define the sweep configuration
    sweep_config = {
        'method': 'grid',
        'name': f"{experiment_name}_{dataset_name}_{sweep_name}",
        'metric': {
            'name': 'bi_cross_val_loss' if cross_val_params is not None else 'reconstruction_error',
            'goal': 'minimize'
        },
        'parameters': {
            'n_components': {'values': num_components_list},
            'alpha': {'values': alphas},
            'ridge_alpha': {'values': ridge_alphas},
            'experiment_name': {'value': experiment_name},
            'dataset_name': {'value': dataset_name},
            'models_dir': {'value': models_dir},
            'max_iter': {'value': max_iter},
            'trajectory_data_path': {'value': data_file},
            'scale': {'value': scale},
            'is_test': {'value': is_test}
        }
    }
    
    # Add cross_val_params if provided
    if cross_val_params is not None:
        sweep_config['parameters']['cross_val_params'] = {'value': cross_val_params}
    
    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project=wandb_project, entity=wandb_entity)
    print(f"Sweep created with ID: {sweep_id}")
    
    return sweep_id


def run_sweep_agent(sweep_id, count=None):
    """
    Run the sweep agent to execute the sweep.
    
    Args:
        sweep_id: ID of the sweep to run
        count: Number of runs to execute (None means run all configurations)
    """
    wandb.agent(sweep_id, train_sparse_pca, count=count)

  

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
