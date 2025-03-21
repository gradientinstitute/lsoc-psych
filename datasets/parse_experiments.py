import os
import pickle
import pandas as pd
import numpy as np
import glob
from tqdm import tqdm
import re

def extract_step_number(filename):
    """Extract step number from the pickle filename."""
    match = re.search(r'step(\d+)\.pkl', filename)
    if match:
        return int(match.group(1))
    return 0

def update_data_with_losses(data_dict, context_id, losses):
    """Update data dictionary with loss values for a context."""
    for pos in range(len(losses)):
        col_name = f"context_{context_id}_pos_{pos}"
        if col_name not in data_dict:
            data_dict[col_name] = []
        data_dict[col_name].append(losses[pos])
    
    return data_dict

def clean_and_save_dataframe(data_dict, output_path):
    """Clean DataFrame by removing empty/constant columns and save to CSV."""
    if not data_dict['step']:  # Skip empty datasets
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(data_dict)
    
    # Remove columns with all NaN
    for col in df.columns:
        if col != 'step':
            if df[col].isna().all():
                df = df.drop(columns=[col])
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Saved {output_path}")

def extract_tokens_from_checkpoint(pickle_file, selected_step):
    """Extract tokens from a specific checkpoint for all datasets."""
    # Extract step number from pickle filename
    step = extract_step_number(pickle_file)
    
    # Process only the selected checkpoint
    if step != selected_step:
        return None
    
    # Load data from pickle file
    with open(pickle_file, 'rb') as f:
        results = pickle.load(f)
    
    # Dictionary to store tokens for each dataset
    dataset_tokens = {}
    
    # Process regular datasets
    for dataset_name, data_items in results.items():
            
        # Initialize token dictionary for this dataset
        tokens_dict = {}
        
        # Process each context item
        for item in data_items:
            if 'context_id' in item:
                context_id = item['context_id']
            elif 'context_idx' in item:
                context_id = item['context_idx']
            tokens = item['tokens']
            tokens_dict[context_id] = tokens
        
        # Store tokens dictionary for this dataset
        dataset_tokens[dataset_name] = tokens_dict
    
    return dataset_tokens

def main():
    # Create output directory for CSVs and tokens
    exp_name = "EXP000"
    comp_path = "/Users/liam/quests/lsoc-psych/"
    base_dir = comp_path + f"datasets/experiments/{exp_name}/trajectories"
    output_dir = os.path.join(base_dir, "csv")
    tokens_dir = os.path.join(base_dir, "tokens")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tokens_dir, exist_ok=True)
    
    # Select a checkpoint step to extract tokens from
    # Choose a step that should exist for all models
    selected_step = 0  # Adjust to a step known to exist for all models
    
    # Find experiment directories in pkl folder (where scp goes)
    pkl_path = os.path.join(base_dir, "pkl")
    model_dirs = [d for d in os.listdir(pkl_path) 
                         if os.path.isdir(os.path.join(pkl_path, d))]
    
    # Flag to track if we've already extracted tokens
    tokens_extracted = False
    
    for model_size in model_dirs:
        model_path = os.path.join(pkl_path, model_size)
        print(f"Processing model: {model_size}")
        
        # Find all pickle files for this model
        pickle_files = glob.glob(os.path.join(model_path, "step*.pkl"))
        pickle_files = sorted(pickle_files, key=extract_step_number)
        
        # Data structures to hold results
        datasets_data = {}
        dm_math_zero_shot = {'step': []}
        dm_math_few_shot = {'step': []}
        
        # Process each pickle file
        for step_idx, pickle_file in enumerate(tqdm(pickle_files, desc=f"Processing {model_size} checkpoints")):
            step = extract_step_number(pickle_file)
            
            # Extract tokens from the selected checkpoint (only once)
            if not tokens_extracted and step == selected_step:
                print(f"Extracting tokens from step {selected_step}")
                dataset_tokens = extract_tokens_from_checkpoint(pickle_file, selected_step)
                
                if dataset_tokens:
                    # Save tokens for each dataset
                    for dataset_name, tokens_dict in dataset_tokens.items():
                        tokens_output_path = os.path.join(tokens_dir, f"{dataset_name}_tokens.pkl")
                        with open(tokens_output_path, 'wb') as f:
                            pickle.dump(tokens_dict, f)
                        print(f"Saved tokens for {dataset_name} to {tokens_output_path}")
                    
                    # Set flag to avoid extracting tokens again
                    tokens_extracted = True
            
            # Load data from pickle file
            with open(pickle_file, 'rb') as f:
                results = pickle.load(f)
            
            # Process dm_mathematics dataset separately
            if 'dm_mathematics' in results:
                # Add step number
                dm_math_zero_shot['step'].append(step)
                dm_math_few_shot['step'].append(step)
                
                # Process each context item
                for item in results['dm_mathematics']:
                    context_idx = item['context_idx']
                    
                    # Process zero-shot data
                    if item['loss']['zero-shot'] is not None:
                        zero_shot_losses = item['loss']['zero-shot']
                        dm_math_zero_shot = update_data_with_losses(
                            dm_math_zero_shot, context_idx, zero_shot_losses)
                    
                    # Process few-shot data
                    if item['loss']['few-shot'] is not None:
                        few_shot_losses = item['loss']['few-shot']
                        dm_math_few_shot = update_data_with_losses(
                            dm_math_few_shot, context_idx, few_shot_losses)
            
            # Process regular datasets
            for dataset_name in results:
                if dataset_name == 'dm_mathematics':
                    continue
                    
                # Initialize dataset structure if not already done
                if dataset_name not in datasets_data:
                    datasets_data[dataset_name] = {'step': []}
                
                # Add step number
                if len(datasets_data[dataset_name]['step']) <= step_idx:
                    datasets_data[dataset_name]['step'].append(step)
                
                # Process each context item
                for item in results[dataset_name]:
                    context_id = item['context_id']
                    losses = item['loss']
                    
                    # Update data dictionary with losses
                    datasets_data[dataset_name] = update_data_with_losses(
                        datasets_data[dataset_name], context_id, losses)
        
        # Create and save CSVs for regular datasets
        for dataset_name, data in datasets_data.items():
            csv_path = os.path.join(output_dir, f"{model_size}_{dataset_name}.csv")
            clean_and_save_dataframe(data, csv_path)
        
        # Create and save CSVs for dm_mathematics
        zero_shot_path = os.path.join(output_dir, f"{model_size}_dm_mathematics_zero_shot.csv")
        clean_and_save_dataframe(dm_math_zero_shot, zero_shot_path)
        
        few_shot_path = os.path.join(output_dir, f"{model_size}_dm_mathematics_few_shot.csv")
        clean_and_save_dataframe(dm_math_few_shot, few_shot_path)

if __name__ == "__main__":
    main()