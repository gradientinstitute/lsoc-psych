# Copyright (c) Gradient Institute and Timaeus. All rights reserved.
# Licensed under the Apache 2.0 License.

import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

def load_token_data(token_file_path):
    """Load token data from a pickle file."""
    with open(token_file_path, 'rb') as f:
        return pickle.load(f)

def load_losses_data(losses_file_path):
    """Load losses data from a CSV file."""
    return pd.read_csv(losses_file_path)

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

def aggregate_losses_by_category(losses_df, category_columns):
    """
    Given the losses DataFrame and the mapping of category->context->columns,
    aggregate the loss values for each step (row) and category.
    
    For each row (corresponding to a training step), for each category, we:
      - Gather all valid (non-NaN) losses from the listed columns.
      - Count the number of tokens and contexts (a context contributes if at least one loss is found).
      - Compute the mean loss.
    
    Returns a DataFrame with columns: 'category', 'step', 'mean_loss', 'token_count', 'context_count'.
    """
    rows = []
    # Iterate over each row (step) in the losses DataFrame.
    for _, row in losses_df.iterrows():
        step = row['step']
        # Process each category.
        for category, contexts in category_columns.items():
            all_losses = []
            context_count = 0
            # Each category is a dict mapping context_idx -> list of CSV columns.
            for ctx, cols in contexts.items():
                # Only use columns that are present in the DataFrame row.
                valid_cols = [col for col in cols if col in row.index]
                # Get non-NaN loss values for this context.
                context_losses = row[valid_cols].dropna().tolist() if valid_cols else []
                if context_losses:
                    context_count += 1
                    all_losses.extend(context_losses)
            if all_losses:
                mean_loss = np.mean(all_losses)
                token_count = len(all_losses)
                rows.append({
                    'category': category,
                    'step': int(step),
                    'mean_loss': mean_loss,
                    'token_count': token_count,
                    'context_count': context_count
                })
    # sort by step, ascending
    df = pd.DataFrame(rows)
    df = df.sort_values(by='step', ascending=True)
    return df

def extract_token_losses(tokens_dict, losses_df, answer_tokens_only=False):
    """
    New extraction routine which:
      1. Precomputes (or reuses) answer token indices in the tokens dictionary.
      2. Iterates over the tokens_dict to build a mapping (for the given task) of
         each category to its corresponding CSV columns.
      3. Aggregates the losses DataFrame by computing, for each step and category,
         the mean loss (and counts) over the selected columns.
    
    Returns a DataFrame with columns: 'category', 'step', 'mean_loss', 'token_count', and 'context_count'.
    """
    # Build the mapping from categories to lists of CSV columns.
    category_columns = build_category_columns(tokens_dict, answer_tokens_only)
    # Aggregate the losses using an efficient DataFrame operation.
    df = aggregate_losses_by_category(losses_df, category_columns)
    return df

def create_merged_dataframes(few_shot_full, few_shot_answer, zero_shot_full, zero_shot_answer):
    """
    Create merged dataframes that compare few-shot and zero-shot losses.
    
    Args:
        few_shot_full: DataFrame with few-shot losses for full context
        few_shot_answer: DataFrame with few-shot losses for answer tokens only
        zero_shot_full: DataFrame with zero-shot losses for full context
        zero_shot_answer: DataFrame with zero-shot losses for answer tokens only
    
    Returns:
        A tuple containing (full_merged_df, answer_merged_df)
    """
    # Ensure step is integer type in all dataframes
    few_shot_full['step'] = few_shot_full['step'].astype(int)
    few_shot_answer['step'] = few_shot_answer['step'].astype(int)
    zero_shot_full['step'] = zero_shot_full['step'].astype(int)
    zero_shot_answer['step'] = zero_shot_answer['step'].astype(int)
    
    # Create merged DataFrames for the full context difference
    few_shot_full_merged = few_shot_full.copy()
    few_shot_full_merged['id'] = few_shot_full_merged['category'] + '_' + few_shot_full_merged['step'].astype(str)
    zero_shot_full_merged = zero_shot_full.copy()
    zero_shot_full_merged['id'] = zero_shot_full_merged['category'] + '_' + zero_shot_full_merged['step'].astype(str)
    
    full_merged_df = pd.merge(
        few_shot_full_merged[['id', 'category', 'step', 'mean_loss']],
        zero_shot_full_merged[['id', 'mean_loss']],
        on='id',
        suffixes=('_few', '_zero'),
        how='outer'
    )
    full_merged_df['mean_loss_diff'] = full_merged_df['mean_loss_few'] - full_merged_df['mean_loss_zero']
    # Sort the merged dataframe by category and step (numerically)
    full_merged_df = full_merged_df.sort_values(by=['category', 'step'])
    
    # Create merged DataFrames for the answer tokens difference
    few_shot_answer_merged = few_shot_answer.copy()
    few_shot_answer_merged['id'] = few_shot_answer_merged['category'] + '_' + few_shot_answer_merged['step'].astype(str)
    zero_shot_answer_merged = zero_shot_answer.copy()
    zero_shot_answer_merged['id'] = zero_shot_answer_merged['category'] + '_' + zero_shot_answer_merged['step'].astype(str)
    
    answer_merged_df = pd.merge(
        few_shot_answer_merged[['id', 'category', 'step', 'mean_loss']],
        zero_shot_answer_merged[['id', 'mean_loss']],
        on='id',
        suffixes=('_few', '_zero'),
        how='outer'
    )
    answer_merged_df['mean_loss_diff'] = answer_merged_df['mean_loss_few'] - answer_merged_df['mean_loss_zero']
    # Sort the merged dataframe by category and step (numerically)
    answer_merged_df = answer_merged_df.sort_values(by=['category', 'step'])
    
    return full_merged_df, answer_merged_df

def process_all_data(token_data, few_shot_losses, zero_shot_losses):
    """
    Process all token and loss data to generate the required dataframes.
    
    Args:
        token_data: Dictionary with token data
        few_shot_losses: DataFrame with few-shot losses
        zero_shot_losses: DataFrame with zero-shot losses
        
    Returns:
        A dictionary containing all processed dataframes:
        {
            'few_shot_full': DataFrame,
            'few_shot_answer': DataFrame,
            'zero_shot_full': DataFrame,
            'zero_shot_answer': DataFrame,
            'full_merged': DataFrame,
            'answer_merged': DataFrame
        }
    """
    # Extract token losses for each condition
    few_shot_full = extract_token_losses(token_data, few_shot_losses, answer_tokens_only=False)
    few_shot_answer = extract_token_losses(token_data, few_shot_losses, answer_tokens_only=True)
    zero_shot_full = extract_token_losses(token_data, zero_shot_losses, answer_tokens_only=False)
    zero_shot_answer = extract_token_losses(token_data, zero_shot_losses, answer_tokens_only=True)
    
    # Create merged dataframes
    full_merged_df, answer_merged_df = create_merged_dataframes(
        few_shot_full, few_shot_answer, zero_shot_full, zero_shot_answer
    )
    
    # Return all dataframes in a dictionary
    return {
        'few_shot_full': few_shot_full,
        'few_shot_answer': few_shot_answer,
        'zero_shot_full': zero_shot_full,
        'zero_shot_answer': zero_shot_answer,
        'full_merged': full_merged_df,
        'answer_merged': answer_merged_df
    }

def create_loss_visualizations(few_shot_full, few_shot_answer, zero_shot_full, zero_shot_answer, 
                              full_merged_df=None, answer_merged_df=None, use_log_scale=True, 
                              model_size = None, axes_ranges = None):
    """
    Create a visualization of loss metrics with an option to toggle between log and linear scales.
    
    Args:
        few_shot_full: DataFrame with few-shot losses for full context
        few_shot_answer: DataFrame with few-shot losses for answer tokens only
        zero_shot_full: DataFrame with zero-shot losses for full context
        zero_shot_answer: DataFrame with zero-shot losses for answer tokens only
        full_merged_df: Optional pre-computed merged DataFrame for full context
        answer_merged_df: Optional pre-computed merged DataFrame for answer tokens
        use_log_scale: Boolean to toggle between log scale (True) and linear scale (False)
    
    Returns:
        A plotly figure with 2x3 subplots
    """
    # If merged dataframes weren't provided, create them
    if full_merged_df is None or answer_merged_df is None:
        full_merged_df, answer_merged_df = create_merged_dataframes(
            few_shot_full, few_shot_answer, zero_shot_full, zero_shot_answer
        )
    
    # Determine all categories
    all_categories = sorted(
        set(few_shot_full['category'].unique()) | 
        set(few_shot_answer['category'].unique()) | 
        set(zero_shot_full['category'].unique()) | 
        set(zero_shot_answer['category'].unique())
    )
    
    # Create a color map for plotting
    n_categories = len(all_categories)
    colors = px.colors.sample_colorscale('viridis', n_categories)
    color_map = {category: color for category, color in zip(all_categories, colors)}
    
    # Create a 2x3 subplot figure
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "Full Context: Few-Shot Losses",
            "Full Context: Zero-Shot Losses",
            "Full Context: Difference (Few - Zero)",
            "Answer Tokens: Few-Shot Losses",
            "Answer Tokens: Zero-Shot Losses",
            "Answer Tokens: Difference (Few - Zero)"
        ),
        shared_yaxes=False,
        horizontal_spacing=0.05,
        vertical_spacing=0.15
    )
    
    # Calculate min and max values for each axis to set fixed ranges
    # For full context plots
    y_min_few_full = few_shot_full['mean_loss'].min() if not few_shot_full.empty else 0
    y_max_few_full = few_shot_full['mean_loss'].max() if not few_shot_full.empty else 1
    y_min_zero_full = zero_shot_full['mean_loss'].min() if not zero_shot_full.empty else 0
    y_max_zero_full = zero_shot_full['mean_loss'].max() if not zero_shot_full.empty else 1
    
    # For answer token plots
    y_min_few_answer = few_shot_answer['mean_loss'].min() if not few_shot_answer.empty else 0
    y_max_few_answer = few_shot_answer['mean_loss'].max() if not few_shot_answer.empty else 1
    y_min_zero_answer = zero_shot_answer['mean_loss'].min() if not zero_shot_answer.empty else 0
    y_max_zero_answer = zero_shot_answer['mean_loss'].max() if not zero_shot_answer.empty else 1
    
    # For difference plots
    y_min_diff_full = full_merged_df['mean_loss_diff'].min() if not full_merged_df.empty and not full_merged_df['mean_loss_diff'].isna().all() else -1
    y_max_diff_full = full_merged_df['mean_loss_diff'].max() if not full_merged_df.empty and not full_merged_df['mean_loss_diff'].isna().all() else 1
    y_min_diff_answer = answer_merged_df['mean_loss_diff'].min() if not answer_merged_df.empty and not answer_merged_df['mean_loss_diff'].isna().all() else -1
    y_max_diff_answer = answer_merged_df['mean_loss_diff'].max() if not answer_merged_df.empty and not answer_merged_df['mean_loss_diff'].isna().all() else 1
    
    # Add a small padding to the ranges (10%)
    def add_padding(min_val, max_val, padding=0.1):
        range_val = max_val - min_val
        return min_val - range_val * padding, max_val + range_val * padding

    y_min_few_full, y_max_few_full = add_padding(y_min_few_full, y_max_few_full)
    y_min_zero_full, y_max_zero_full = add_padding(y_min_zero_full, y_max_zero_full)
    y_min_few_answer, y_max_few_answer = add_padding(y_min_few_answer, y_max_few_answer)
    y_min_zero_answer, y_max_zero_answer = add_padding(y_min_zero_answer, y_max_zero_answer)
    y_min_diff_full, y_max_diff_full = add_padding(y_min_diff_full, y_max_diff_full)
    y_min_diff_answer, y_max_diff_answer = add_padding(y_min_diff_answer, y_max_diff_answer)
    
    # Find min and max for x-axis (steps)
    all_steps = []
    for df in [few_shot_full, few_shot_answer, zero_shot_full, zero_shot_answer]:
        if not df.empty:
            all_steps.extend(df['step'].tolist())
    x_min = min(all_steps) if all_steps else 1
    x_max = max(all_steps) if all_steps else 1000
    
    # Add padding for x-axis range
    if use_log_scale:
        # For log scale, ensure min is at least 1
        x_min = max(1, x_min * 0.9)
        x_max = x_max * 1.1
    else:
        # For linear scale
        x_range = x_max - x_min
        x_min = max(0, x_min - x_range * 0.05)  # Can start at 0 for linear
        x_max = x_max + x_range * 0.05
    
    # Add traces for each category
    for category in all_categories:
        # 1. Full Context - Few-shot losses
        cat_few_full = few_shot_full[few_shot_full['category'] == category].sort_values(by='step')
        if not cat_few_full.empty:
            fig.add_trace(
                go.Scatter(
                    x=cat_few_full['step'],
                    y=cat_few_full['mean_loss'],
                    mode='lines',
                    name=category,
                    line=dict(color=color_map[category]),
                    legendgroup=category,
                    showlegend=True,
                    hovertemplate='Step: %{x}<br>Loss: %{y:.4f}<extra>' + category + '</extra>'
                ),
                row=1, col=1
            )
        
        # 2. Full Context - Zero-shot losses
        cat_zero_full = zero_shot_full[zero_shot_full['category'] == category].sort_values(by='step')
        if not cat_zero_full.empty:
            fig.add_trace(
                go.Scatter(
                    x=cat_zero_full['step'],
                    y=cat_zero_full['mean_loss'],
                    mode='lines',
                    name=category,
                    line=dict(color=color_map[category]),
                    legendgroup=category,
                    showlegend=False,
                    hovertemplate='Step: %{x}<br>Loss: %{y:.4f}<extra>' + category + '</extra>'
                ),
                row=1, col=2
            )
        
        # 3. Full Context - Difference
        cat_full_diff = full_merged_df[full_merged_df['category'] == category].sort_values(by='step')
        if not cat_full_diff.empty and not cat_full_diff['mean_loss_diff'].isna().all():
            cat_full_diff = cat_full_diff.dropna(subset=['mean_loss_diff'])
            if not cat_full_diff.empty:
                fig.add_trace(
                    go.Scatter(
                        x=cat_full_diff['step'],
                        y=cat_full_diff['mean_loss_diff'],
                        mode='lines',
                        name=category,
                        line=dict(color=color_map[category]),
                        legendgroup=category,
                        showlegend=False,
                        hovertemplate='Step: %{x}<br>Diff: %{y:.4f}<extra>' + category + '</extra>'
                    ),
                    row=1, col=3
                )
        
        # 4. Answer Tokens - Few-shot losses
        cat_few_answer = few_shot_answer[few_shot_answer['category'] == category].sort_values(by='step')
        if not cat_few_answer.empty:
            fig.add_trace(
                go.Scatter(
                    x=cat_few_answer['step'],
                    y=cat_few_answer['mean_loss'],
                    mode='lines',
                    name=category,
                    line=dict(color=color_map[category]),
                    legendgroup=category,
                    showlegend=False,
                    hovertemplate='Step: %{x}<br>Loss: %{y:.4f}<extra>' + category + '</extra>'
                ),
                row=2, col=1
            )
        
        # 5. Answer Tokens - Zero-shot losses
        cat_zero_answer = zero_shot_answer[zero_shot_answer['category'] == category].sort_values(by='step')
        if not cat_zero_answer.empty:
            fig.add_trace(
                go.Scatter(
                    x=cat_zero_answer['step'],
                    y=cat_zero_answer['mean_loss'],
                    mode='lines',
                    name=category,
                    line=dict(color=color_map[category]),
                    legendgroup=category,
                    showlegend=False,
                    hovertemplate='Step: %{x}<br>Loss: %{y:.4f}<extra>' + category + '</extra>'
                ),
                row=2, col=2
            )
        
        # 6. Answer Tokens - Difference
        cat_answer_diff = answer_merged_df[answer_merged_df['category'] == category].sort_values(by='step')
        if not cat_answer_diff.empty and not cat_answer_diff['mean_loss_diff'].isna().all():
            cat_answer_diff = cat_answer_diff.dropna(subset=['mean_loss_diff'])
            if not cat_answer_diff.empty:
                fig.add_trace(
                    go.Scatter(
                        x=cat_answer_diff['step'],
                        y=cat_answer_diff['mean_loss_diff'],
                        mode='lines',
                        name=category,
                        line=dict(color=color_map[category]),
                        legendgroup=category,
                        showlegend=False,
                        hovertemplate='Step: %{x}<br>Diff: %{y:.4f}<extra>' + category + '</extra>'
                    ),
                    row=2, col=3
                )
    
    # Add a scale toggle button
    button_log = dict(
        method="relayout",
        args=[{"xaxis.type": "log", "xaxis2.type": "log", "xaxis3.type": "log", 
               "xaxis4.type": "log", "xaxis5.type": "log", "xaxis6.type": "log"}],
        label="Log Scale"
    )
    
    button_linear = dict(
        method="relayout",
        args=[{"xaxis.type": "linear", "xaxis2.type": "linear", "xaxis3.type": "linear", 
               "xaxis4.type": "linear", "xaxis5.type": "linear", "xaxis6.type": "linear"}],
        label="Linear Scale"
    )
    
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                active=0 if use_log_scale else 1,
                showactive=True,
                buttons=[button_log, button_linear],
                x=0.9,
                y=1.1,
                xanchor="left",
                yanchor="top"
            )
        ]
    )
    
    # Update layout for the figure
    fig.update_layout(
        height=900,
        width=1800,
        title=f"Pythia {model_size} - dm_mathematics",
        plot_bgcolor='rgba(255,255,255,1)',
        paper_bgcolor='rgba(255,255,255,1)',
        hovermode='closest',
        legend=dict(
            orientation='v',
            yanchor='top',
            y=1.0,
            xanchor='left',
            x=1.02,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.1)',
            borderwidth=1,
            itemsizing='constant',
            tracegroupgap=0,
            itemclick="toggle",
            itemdoubleclick="toggleothers",
            groupclick="togglegroup"
        ),
        margin=dict(l=60, r=250, t=120, b=60),  # Increased top margin for buttons
        uirevision=True  # This helps maintain view consistency when data changes
    )

    # Set x-axis scale based on parameter
    x_scale_type = "log" if use_log_scale else "linear"
    
    # Calculate appropriate x-axis range based on scale type
    if use_log_scale:
        x_range = [np.log10(x_min), np.log10(x_max)]
    else:
        x_range = [x_min, x_max]
    
    # Update all x-axes and set fixed ranges for all plots
    for i in range(1, 3):
        for j in range(1, 4):
            # Set common x-axis range for all plots
            fig.update_xaxes(
                type=x_scale_type,
                title='Training Step',
                showgrid=True,
                gridcolor='rgba(230,230,230,0.8)',
                range=x_range,
                row=i, col=j
            )
            
            # Set y-axis titles and ranges based on the subplot
            if i == 1:  # First row (Full Context)
                if j == 1:  # Few-Shot
                    y_title = 'Mean Few-Shot Loss'
                    y_range = [y_min_few_full, y_max_few_full]
                elif j == 2:  # Zero-Shot
                    y_title = 'Mean Zero-Shot Loss'
                    y_range = [y_min_zero_full, y_max_zero_full]
                else:  # Difference
                    y_title = 'Mean Loss Difference'
                    y_range = [y_min_diff_full, y_max_diff_full]
                    # Add zero line for difference plots
                    fig.update_yaxes(
                        zeroline=True,
                        zerolinecolor='rgba(0,0,0,0.5)',
                        zerolinewidth=1,
                        row=i, col=j
                    )
            else:  # Second row (Answer Tokens)
                if j == 1:  # Few-Shot
                    y_title = 'Mean Few-Shot Loss'
                    y_range = [y_min_few_answer, y_max_few_answer]
                elif j == 2:  # Zero-Shot
                    y_title = 'Mean Zero-Shot Loss'
                    y_range = [y_min_zero_answer, y_max_zero_answer]
                else:  # Difference
                    y_title = 'Mean Loss Difference'
                    y_range = [y_min_diff_answer, y_max_diff_answer]
                    # Add zero line for difference plots
                    fig.update_yaxes(
                        zeroline=True,
                        zerolinecolor='rgba(0,0,0,0.5)',
                        zerolinewidth=1,
                        row=i, col=j
                    )
            
            fig.update_yaxes(
                title=y_title,
                showgrid=True,
                gridcolor='rgba(230,230,230,0.8)',
                range=y_range,  # Set fixed range for each y-axis
                row=i, col=j
            )

    if axes_ranges is not None:
        for (row, col), (y_min, y_max) in axes_ranges.items():
            fig.update_yaxes(range=[y_min, y_max], row = row, col = col)
    
    # Add this code inside the create_loss_visualizations function
# after all the other axes updates (near the end of the function)

    # Add thicker axis lines for all plots
    for i in range(1, 3):
        for j in range(1, 4):
            # Standard axis lines
            # fig.update_xaxes(
            #     linewidth=1.5,        # Regular axis line thickness
            #     linecolor='black',
            #     row=i, col=j
            # )
            
            fig.update_yaxes(
                linewidth=1.5,        # Regular axis line thickness
                linecolor='black',
                row=i, col=j
            )
            
            # Add prominent zeroline for all plots
            fig.update_yaxes(
                zeroline=True,
                zerolinewidth=1.5,      # Thicker zero line
                zerolinecolor='black',
                row=i, col=j
            )
    
    return fig
