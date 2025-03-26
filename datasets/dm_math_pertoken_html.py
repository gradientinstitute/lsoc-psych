import os
import pickle
import pandas as pd
import numpy as np
import glob
import re
from IPython.display import HTML, display
import csv
import json

def get_context_preview(tokens, max_words=3):
    """
    Extract the first few words from a context for display purposes.
    
    Args:
        tokens: List of tokens
        max_words: Maximum number of words to include
        
    Returns:
        String with first few words and ellipsis
    """
    # Join tokens and limit to first few words
    text = ''.join(tokens).strip()
    
    # Split by whitespace and take first few words
    words = text.split()
    preview = ' '.join(words[:max_words])
    
    # Add ellipsis if there are more words
    if len(words) > max_words:
        preview += "..."
        
    return preview

def load_token_data(tokens_dir, dataset_name):
    """
    Load token data for a specific dataset.
    
    Args:
        tokens_dir: Directory containing token pickle files
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary of token data by context ID
    """
    dataset_name="dm_mathematics"

    token_file = os.path.join(tokens_dir, f"{dataset_name}_tokens.pkl")
    
    if not os.path.exists(token_file):
        print(f"Token file not found: {token_file}")
        return {}
    
    with open(token_file, 'rb') as f:
        tokens_dict = pickle.load(f)
        
    return tokens_dict

def load_loss_data(csv_dir, model_name, dataset_name, max_contexts=None):
    """
    Load loss data for a specific model and dataset.
    
    Args:
        csv_dir: Directory containing CSV files
        model_name: Name of the model
        dataset_name: Name of the dataset
        max_contexts: Maximum number of contexts to include
        
    Returns:
        Pandas DataFrame with loss data
    """
    csv_file = os.path.join(csv_dir, f"{model_name}_{dataset_name}.csv")
    
    if not os.path.exists(csv_file):
        print(f"CSV file not found: {csv_file}")
        return pd.DataFrame()
    
    print(f"Loading loss data from {csv_file}")
    if max_contexts is None:
        print("Loading all contexts")
        return pd.read_csv(csv_file)
    
    # First read just the headers to identify context columns
    # headers = pd.read_csv(csv_file, nrows=0).columns.tolist()
    # print(headers)
    with open(csv_file, 'r') as f:
        headers = next(csv.reader(f))
    
    # Extract unique context IDs from column names
    context_pattern = re.compile(r'context_(\d+)_pos_(\d+)')
    context_ids = set()
    
    for col in headers:
        if col.startswith('context_'):
            match = context_pattern.match(col)
            if match:
                context_ids.add(int(match.group(1)))
    
    # Sort and limit context IDs
    context_ids = sorted(list(context_ids))[:max_contexts]
    print(context_ids)
    
    # Create a list of columns to load: non-context columns + only the selected contexts
    columns_to_load = [col for col in headers if not col.startswith('context_')]
    for context_id in context_ids:
        columns_to_load.extend([col for col in headers if col.startswith(f'context_{context_id}_pos_')])
    
    # Now load only the required columns
    return pd.read_csv(csv_file, usecols=columns_to_load)

def extract_context_data(df, tokens_dict, max_contexts=None, selected_steps=None):
    """
    Extract loss data for each context and organize it into a structured format.
    
    Args:
        df: DataFrame with loss data
        tokens_dict: Dictionary of tokens by context ID
        max_contexts: Maximum number of contexts to include (or None for all)
        selected_steps: List of specific steps to include (or None for all)
        
    Returns:
        Dictionary of results organized by context
    """
    results = {}
    
    # Get all steps or filter to selected steps
    all_steps = df['step'].unique().tolist()
    steps = selected_steps if selected_steps is not None else all_steps
    
    # Filter the DataFrame to only include the selected steps
    if selected_steps is not None:
        df = df[df['step'].isin(steps)]
    
    # Get all context columns
    context_columns = [col for col in df.columns if col.startswith('context_')]
    
    # Extract context IDs from column names
    context_pattern = re.compile(r'context_(\d+)_pos_(\d+)')
    context_ids = set()
    
    for col in context_columns:
        match = context_pattern.match(col)
        if match:
            context_ids.add(int(match.group(1)))
    
    # Sort context IDs
    context_ids = sorted(list(context_ids))
    
    # Limit contexts if specified - do this BEFORE processing to avoid wasted work
    if max_contexts and len(context_ids) > max_contexts:
        context_ids = context_ids[:max_contexts]
    
    # Process each context
    for context_id in context_ids:
        # Skip if we don't have tokens for this context
        if context_id not in tokens_dict:
            continue
        
        tokens = tokens_dict[context_id]
        context_preview = get_context_preview(tokens)
        
        # Initialize context data
        context_data = {
            "tokens": tokens,
            "preview": context_preview,
            "checkpoints": {}
        }
        
        # Extract columns for this context
        context_cols = [col for col in context_columns if col.startswith(f'context_{context_id}_pos_')]
        
        # For each step, collect losses for this context
        for step in steps:
            step_str = str(step)
            step_data = df[df['step'] == step]
            
            if len(step_data) == 0:
                continue
                
            losses = []
            
            # Extract loss values for each position
            for col in sorted(context_cols, key=lambda x: int(re.match(r'context_\d+_pos_(\d+)', x).group(1))):
                if col in step_data.columns:
                    loss_value = step_data[col].values[0]
                    losses.append(float(loss_value) if not pd.isna(loss_value) else None)
                
            # Store losses for this step
            if losses:
                context_data["checkpoints"][step_str] = {
                    "losses": losses
                }
        
        # Add context to results
        results[str(context_id)] = context_data
    
    return results

def create_dm_math_pertoken_html(all_models_results, selected_steps=None, model_name="", dataset_name="dm_mathematics"):
    """
    Create HTML for per-token loss visualization with interactive controls for dm_mathematics.
    
    Args:
        all_models_results: Dictionary with model sizes containing zero-shot and few-shot data
        selected_steps: List of steps to display initially (or None for all)
        model_name: Name of the model being visualized
        dataset_name: Name of the dataset being visualized
        
    Returns:
        HTML string for display
    """
    # Get all available model sizes
    model_sizes = list(all_models_results.keys())
    if not model_sizes:
        return "<div>Error: No model data available</div>"
    
    # Default to the first model size in the list
    default_model_size = model_sizes[0]
    
    # Get both types of results
    zero_shot_results = all_models_results[default_model_size]["zero_shot"]
    few_shot_results = all_models_results[default_model_size]["few_shot"]
    
    # Use whichever has data
    results = zero_shot_results if zero_shot_results else few_shot_results
    if not results:
        return "<div>Error: No data available for the selected model</div>"
    
    # Get all context indices
    context_indices = sorted(list(results.keys()), key=int)
    
    # Get available steps
    available_steps = []
    if context_indices:
        # Collect all unique steps from all contexts
        all_available_steps = set()
        for context_id in context_indices:
            all_available_steps.update(results[context_id]["checkpoints"].keys())
        # Sort steps as integers, not strings
        available_steps = sorted(list(all_available_steps), key=int)

    # If selected_steps is provided, only use those that are actually in the data
    if selected_steps is not None:
        available_steps = [step for step in selected_steps if step in available_steps]
    
    # Create context categories with context ID and preview
    categories = {}
    for context_idx in context_indices:
        if "preview" in results[context_idx]:
            preview = results[context_idx]["preview"]
            categories[context_idx] = f"Context {context_idx} - {preview}"
        else:
            categories[context_idx] = f"Context {context_idx}"
    
    # Get the first context as default
    default_context = context_indices[0] if context_indices else ""
    
    # Calculate universal max loss values per context (for both zero and few shot)
    universal_max_losses = {}
    for context_id in context_indices:
        max_loss = 0
        # Check across all model sizes
        for model_size in model_sizes:
            model_results = all_models_results[model_size]
            
            # Check zero-shot data
            if context_id in model_results["zero_shot"]:
                zero_shot_data = model_results["zero_shot"][context_id]
                for step in available_steps:
                    if step in zero_shot_data["checkpoints"]:
                        losses = zero_shot_data["checkpoints"][step]["losses"]
                        if losses:
                            valid_losses = [loss for loss in losses if loss is not None]
                            if valid_losses:
                                max_loss = max(max_loss, max(valid_losses))
            
            # Check few-shot data
            if context_id in model_results["few_shot"]:
                few_shot_data = model_results["few_shot"][context_id]
                for step in available_steps:
                    if step in few_shot_data["checkpoints"]:
                        losses = few_shot_data["checkpoints"][step]["losses"]
                        if losses:
                            valid_losses = [loss for loss in losses if loss is not None]
                            if valid_losses:
                                max_loss = max(max_loss, max(valid_losses))
                                
        universal_max_losses[context_id] = max_loss
    
    # Initialize the HTML with styling and script
    html = create_html_styles()
    
    # Add title for model and dataset
    if model_name or dataset_name:
        title_text = ""
        if model_name and dataset_name:
            title_text = f"{model_name} on {dataset_name}"
        elif model_name:
            title_text = model_name
        else:
            title_text = dataset_name
        
        html += f"""
        <div style="text-align: center; margin: 20px 0; padding: 10px; background-color: #2c3e50; color: white; font-size: 24px; font-weight: bold; border-radius: 5px;">
            {title_text}
        </div>
        """
    
    # Add controls with prompting strategy options
    html += create_dm_math_controls_html(available_steps, selected_steps, categories, model_sizes, default_model_size)
    
    # Create container for visualization
    html += '<div id="visualization-container">'
    
    # Add initial context view (first context with zero-shot data)
    if default_context:
        # Pass the universal max loss for this context
        max_loss = universal_max_losses.get(default_context, 0)
        if zero_shot_results and default_context in zero_shot_results:
            html += create_single_context_raw_html(zero_shot_results, default_context, selected_steps, max_loss)
        elif few_shot_results and default_context in few_shot_results:
            html += create_single_context_raw_html(few_shot_results, default_context, selected_steps, max_loss)
        else:
            html += '<div style="text-align: center; padding: 20px; color: #666; font-size: 16px;">No data available for this context.</div>'
    else:
        html += '<div style="text-align: center; padding: 20px; color: #666; font-size: 16px;">No contexts available.</div>'
    
    html += '</div>'
    
    # Add JavaScript for toggling between views and handling selections
    html += create_dm_math_toggle_script(all_models_results, context_indices, default_context, default_model_size, universal_max_losses)
    
    return html

def create_html_styles():
    """
    Create CSS styles for the visualization.
    
    Returns:
        HTML string with CSS styles
    """
    return """
    <style>
        .controls-container {
            margin: 20px 0;
            text-align: center;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 15px;
            width: 100%;  /* Set explicit width */
            position: sticky;  /* Add this line */
            top: 0;           /* Add this line */
            background-color: white;  /* Add this line */
            z-index: 100;     /* Add this line */
            padding: 10px 0;  /* Add this line */
        }
        .scrolling-controls {
            margin: 20px 0;
            text-align: center;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 15px;
            width: 100%;
        }

        .sticky-controls {
            position: sticky;
            top: 0;
            background-color: white;
            z-index: 100;
            padding: 10px 0;
            width: 100%;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 15px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            margin-bottom: 15px;
        }
        
        /* Add this to center the toggle group */
        .toggle-group {
            display: inline-flex;
            gap: 20px;
            background-color: #f5f5f5;
            padding: 10px 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            margin: 0 auto;  /* This centers the element */
            justify-content: center;  /* Center content inside */
        }
        
        /* Fix the model selector */
        #model-selector, 
        #model-comparison-container {
            width: 100%;
            max-width: 800px;
            margin-left: auto;
            margin-right: auto;
            margin-bottom: 15px;
            display: flex;
            justify-content: center;
        }
        
        /* The rest of your original CSS follows... */
        .toggle-label {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 16px;
            cursor: pointer;
            user-select: none;
            font-weight: 500;
            color: black;
        }
        .toggle-label input {
            margin-right: 5px;
            transform: scale(1.2);
        }
        .step-selector {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 10px;
            width: 100%;
            max-width: 800px;
            color: black;
        }
        .step-selector-title {
            font-weight: bold;
            font-size: 16px;
        }
        .step-checkboxes {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            justify-content: center;
            max-width: 800px;
            padding: 10px;
            background-color: #f5f5f5;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            max-height: 200px;
            overflow-y: auto;
        }
        .step-checkbox-label {
            display: flex;
            align-items: center;
            gap: 2px;
            font-size: 12px;
            cursor: pointer;
            user-select: none;
            padding: 3px 6px;
            background-color: white;
            border-radius: 4px;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
        }
        .step-checkbox-label:hover {
            background-color: #efefef;
        }
        .step-selector-buttons {
            display: flex;
            gap: 10px;
            justify-content: center;  /* Center buttons */
        }
        .step-selector-button {
            padding: 8px 15px;
            border: none;
            border-radius: 4px;
            background-color: #2c3e50;
            color: white;
            cursor: pointer;
            font-size: 14px;
            transition: background-color 0.2s;
        }
        .step-selector-button:hover {
            background-color: #34495e;
        }
        .context-selector {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 10px;
            margin-bottom: 15px;
            width: 100%;  /* Add width */
            max-width: 800px;  /* Add max-width to match step-selector */
        }
        .context-selector-title {
            font-weight: bold;
            font-size: 16px;
            color: black;
        }
        .context-dropdown {
            padding: 8px 15px;
            border: 1px solid #ccc;
            border-radius: 4px;
            font-size: 14px;
            min-width: 300px;
            background-color: white;
        }
        .context-container {
            margin-bottom: 60px;
            border-bottom: 2px solid #888;
            padding-bottom: 20px;
        }
        .context-title {
            font-size: 18px;
            font-weight: bold;
            margin: 20px 0 10px 0;
            padding: 8px;
            background-color: #333;
            color: white;
            border-radius: 4px;
        }
        
        /* Token grid styles for proper horizontal layout */
        .token-grid-wrapper {
            overflow-x: auto;
            width: 100%;
        }
        
        .token-grid {
            display: block;
            font-family: monospace;
            margin-top: 10px;
        }
        
        .token-row-header, .token-row {
            display: flex;
            flex-direction: row;
            flex-wrap: nowrap;
        }
        
        .header-cell, .token-cell, .step-header {
            padding: 4px 8px;
            min-width: 30px;
            white-space: nowrap;
            box-sizing: border-box;
        }
        
        .header-cell {
            font-weight: bold;
            text-align: center;
            background-color: #444444;
            color: white;
            position: sticky;
            top: 0;
            z-index: 10;
            flex: 0 0 auto;
            min-width: 30px;
            width: auto;
            padding: 4px 8px;
            box-sizing: border-box;
        }
        
        .step-header {
            font-weight: bold;
            text-align: right;
            background-color: #444444;
            color: white;
            position: sticky;
            left: 0;
            z-index: 5;
            min-width: 210px;
            width: 235px;
        }
        
        .token-header {
            text-align: center;
            padding-right: 0;
            padding-left: 10px;
        }
        
        .token-cell {
            text-align: center;
            color: black;
            flex: 0 0 auto;
            min-width: 30px;
            width: auto;
            padding: 4px 8px;
            box-sizing: border-box;
        }
        
        /* Tooltip styles */
        .tooltip {
            position: relative;
            display: inline-block;
        }
        .tooltip .tooltiptext {
            visibility: hidden;
            background-color: rgba(0, 0, 0, 0.8);
            color: white;
            text-align: center;
            padding: 8px;
            border-radius: 3px;
            position: absolute;
            z-index: 100;
            bottom: 125%;
            left: 50%;
            transform: translateX(-50%);
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 12px;
            white-space: nowrap;
            pointer-events: none;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
        }
        
        /* Fix for tooltips near the top of the screen */
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        
        /* Alternative positioning for tooltips near the top */
        .token-grid .header-cell.tooltip .tooltiptext,
        .token-grid .token-row:first-child .tooltip .tooltiptext {
            bottom: auto;
            top: 125%;
        }
        
        /* Legend styling */
        .legend-container {
            margin: 10px 0;
            display: flex;
            align-items: center;
            gap: 20px;
            font-family: sans-serif;
            font-size: 14px;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        .legend-color {
            width: 20px;
            height: 20px;
            border: 1px solid #888;
        }
        .interval-selector {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 10px;
            margin-bottom: 15px;
            width: 100%;
            max-width: 800px;
        }
        .interval-selector-title {
            font-weight: bold;
            font-size: 16px;
            color: black;
        }
        .intervals-input {
            padding: 8px 15px;
            border: 1px solid #ccc;
            border-radius: 4px;
            font-size: 14px;
            width: 80%;
            max-width: 600px;
        }
        .model-selector {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 10px;
            margin-bottom: 15px;
            width: 100%;  /* Add width */
            max-width: 800px;  /* Add max-width to match others */
        }
        .model-selector-title {
            font-weight: bold;
            font-size: 16px;
            color: black;
        }
        .model-dropdown {
            padding: 8px 15px;
            border: 1px solid #ccc;
            border-radius: 4px;
            font-size: 14px;
            min-width: 300px;
            background-color: white;
        }
        .model-comparison {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 10px;
            margin-top: 15px;
            width: 100%;
        }
        .model-comparison-selectors {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .comparison-vs {
            font-weight: bold;
            font-size: 16px;
            color: #555;
        }
        /* Legend for model differences */
        .model-diff-legend {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 15px;
            margin: 10px 0;
        }
        @media (max-width: 768px) {
            .interval-selector {
                width: 95%;
            }
            .intervals-input {
                width: 95%;
            }
        }
        /* Add to your existing styles */
        .diff-mode-buttons {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 10px;
            margin-top: 10px;
            width: 100%;
        }

        .diff-mode-toggle {
            display: flex;
            gap: 10px;
            justify-content: center;
        }

        .diff-mode-button {
            padding: 8px 15px;
            border: 1px solid #ccc;
            border-radius: 4px;
            background-color: #f5f5f5;
            color: #333;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s;
        }

        .diff-mode-button:hover {
            background-color: #e9e9e9;
        }

        .diff-mode-active {
            background-color: #2c3e50;
            color: white;
            border-color: #2c3e50;
        }
        .prompting-selector {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 10px;
            margin: 15px 0;
            width: 100%;
            max-width: 800px;
        }
        
        .prompting-selector-title {
            font-weight: bold;
            font-size: 16px;
            color: black;
        }
        
        .prompting-toggle-group {
            display: inline-flex;
            gap: 20px;
            background-color: #f5f5f5;
            padding: 10px 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
    </style>
    """

def create_dm_math_controls_html(all_steps, selected_steps, categories, model_sizes, default_model_size):
    """
    Create HTML for controls with prompting strategy toggle.
    """
    # Start with scrolling controls
    html = '''
    <!-- Regular scrolling controls -->
    <div class="scrolling-controls">
        <div class="step-selector">
            <div class="step-selector-title">Select Checkpoints to Display:</div>
            <div class="step-checkboxes">
    '''

    # Add a checkbox for each step
    for step in all_steps:
        checked = "checked" if selected_steps is None or step in selected_steps else ""
        html += f'''
        <label class="step-checkbox-label">
            <input type="checkbox" class="step-checkbox" value="{step}" {checked}>
            {step}
        </label>
        '''

    html += '''
            </div>
            
            <div class="interval-selector">
                <div class="interval-selector-title">Or select steps by interval:</div>
                <input type="text" id="intervals-input" class="intervals-input" placeholder="e.g. [0,512], [9000, 15000]">
                <button class="step-selector-button" onclick="applyIntervals()">Apply Intervals</button>
            </div>
            
            <div class="step-selector-buttons">
                <button class="step-selector-button" onclick="selectAllSteps()">Select All</button>
                <button class="step-selector-button" onclick="deselectAllSteps()">Deselect All</button>
                <button class="step-selector-button" onclick="updateVisualization()">Update View</button>
            </div>
        </div>

        <!-- Context selector now in the non-sticky part -->
        <div class="context-selector">
            <div class="context-selector-title">Select Context:</div>
            <select id="context-dropdown" class="context-dropdown" onchange="updateContext()">
    '''

    # Add contexts to dropdown
    for context_id, label in categories.items():
        html += f'<option value="{context_id}">{label}</option>'

    html += '''
            </select>
        </div>
    </div>

    <!-- Separate sticky controls - now contains view toggle, prompting strategy and model selection -->
    <div class="sticky-controls">
        <!-- Primary view toggle -->
        <div class="toggle-group">
            <label class="toggle-label">
                <input type="radio" name="viewToggle" value="raw" onchange="toggleView('raw')" checked>
                Raw Loss View
            </label>
            <label class="toggle-label">
                <input type="radio" name="viewToggle" value="diff" onchange="toggleView('diff')">
                Step-by-Step Differences
            </label>
            <label class="toggle-label">
                <input type="radio" name="viewToggle" value="promptdiff" onchange="toggleView('promptdiff')">
                Zero vs Few Shot Differences
            </label>
            <label class="toggle-label">
                <input type="radio" name="viewToggle" value="modeldiff" onchange="toggleView('modeldiff')">
                Model Size Differences
            </label>
        </div>

        <!-- Prompting strategy toggle -->
        <div id="prompting-selector" class="prompting-selector">
            <div class="prompting-selector-title">Select Prompting Strategy:</div>
            <div class="prompting-toggle-group">
                <label class="toggle-label">
                    <input type="radio" name="promptingToggle" value="zero_shot" onchange="updatePromptingStrategy()" checked>
                    Zero-Shot
                </label>
                <label class="toggle-label">
                    <input type="radio" name="promptingToggle" value="few_shot" onchange="updatePromptingStrategy()">
                    Few-Shot
                </label>
            </div>
        </div>

        <div id="model-selector" class="model-selector">
            <div class="model-selector-title">Select Model Size:</div>
            <select id="model-dropdown" class="model-dropdown" onchange="updateModelSize()">
    '''

    # Add model sizes to dropdown
    for model_size in model_sizes:
        selected = "selected" if model_size == default_model_size else ""
        html += f'<option value="{model_size}" {selected}>Model Size: {model_size}</option>'

    html += '''
            </select>
        </div>
        
        <!-- Model comparison selectors (initially hidden) -->
        <div id="model-comparison-container" style="display: none;" class="model-comparison">
            <div class="model-selector-title">Compare Models:</div>
            <div class="model-comparison-selectors">
                <select id="model-base-dropdown" class="model-dropdown">
                </select>
                <span class="comparison-vs">vs</span>
                <select id="model-compare-dropdown" class="model-dropdown">
                </select>
            </div>
        </div>
    </div>
    '''

    return html

def create_dm_math_toggle_script(all_models_results, context_indices, default_context, default_model_size, universal_max_losses=None):
    """
    Create JavaScript for toggle functionality and selection with prompting strategy support.
    """
    # Serialize the full results data for JavaScript
    import json
    
    # Optimize the JSON serialization
    optimized_results = {}
    
    for model_size, model_data in all_models_results.items():
        optimized_results[model_size] = {"zero_shot": {}, "few_shot": {}}
        
        # Process zero-shot data
        zero_shot_data = model_data.get("zero_shot", {})
        for context_id in context_indices:
            if context_id in zero_shot_data:
                context_data = zero_shot_data[context_id]
                optimized_results[model_size]["zero_shot"][context_id] = {
                    "tokens": context_data["tokens"],
                    "preview": context_data.get("preview", ""),
                    "checkpoints": context_data["checkpoints"]
                }
        
        # Process few-shot data
        few_shot_data = model_data.get("few_shot", {})
        for context_id in context_indices:
            if context_id in few_shot_data:
                context_data = few_shot_data[context_id]
                optimized_results[model_size]["few_shot"][context_id] = {
                    "tokens": context_data["tokens"],
                    "preview": context_data.get("preview", ""),
                    "checkpoints": context_data["checkpoints"]
                }

    # Serialize the universal max losses
    if universal_max_losses is None:
        universal_max_losses = {}
    universal_max_losses_json = json.dumps(universal_max_losses)
    
    # Use the optimized results for serialization
    results_json = json.dumps(optimized_results)


    return f"""
    <script>
        // Store the full results data for dynamic processing
        const fullModelResults = {results_json};
        const contextIndices = {json.dumps(context_indices)};
        const universalMaxLosses = {universal_max_losses_json};
        let currentView = 'raw';
        let currentContext = "{default_context}";
        let currentModelSize = "{default_model_size}";
        let currentPromptingStrategy = "zero_shot";
        let diffMode = 'absolute'; 
        
        // Update prompting strategy when radio buttons change
        function updatePromptingStrategy() {{
            const promptingRadios = document.getElementsByName('promptingToggle');
            for (const radio of promptingRadios) {{
                if (radio.checked) {{
                    currentPromptingStrategy = radio.value;
                    break;
                }}
            }}
            updateVisualization();
        }}
        
        // Update model size when dropdown changes
        function updateModelSize() {{
            const dropdown = document.getElementById('model-dropdown');
            currentModelSize = dropdown.value;
            updateVisualization();
        }}

        // Update when base model dropdown changes
        function updateBaseModel() {{
            updateVisualization();
        }}

        // Update when compare model dropdown changes
        function updateCompareModel() {{
            updateVisualization();
        }}

        // Parse and apply interval selections
        function applyIntervals() {{
            const intervalsInput = document.getElementById('intervals-input').value.trim();
            
            if (!intervalsInput) {{
                alert('Please enter interval ranges in the format [start,end], [start,end]');
                return;
            }}
            
            try {{
                // Parse input string to get interval ranges
                // First replace all spaces for consistent parsing
                const cleanedInput = intervalsInput.replace(/\\s+/g, '');
                
                // Split by "],["
                const intervalStrings = cleanedInput.split('],[');
                
                // Process each interval
                const intervals = [];
                
                for (let i = 0; i < intervalStrings.length; i++) {{
                    let intervalStr = intervalStrings[i];
                    
                    // Clean up brackets for first and last intervals
                    if (i === 0) {{
                        intervalStr = intervalStr.replace(/^\\[/, '');
                    }}
                    if (i === intervalStrings.length - 1) {{
                        intervalStr = intervalStr.replace(/\\]$/, '');
                    }}
                    
                    // Split by comma
                    const parts = intervalStr.split(',');
                    if (parts.length !== 2) {{
                        throw new Error('Each interval must have a start and end value');
                    }}
                    
                    const start = parseInt(parts[0]);
                    const end = parseInt(parts[1]);
                    
                    if (isNaN(start) || isNaN(end)) {{
                        throw new Error('Interval values must be integers');
                    }}
                    
                    if (start > end) {{
                        throw new Error('Start value must be less than or equal to end value');
                    }}
                    
                    intervals.push([start, end]);
                }}
                
                // Now select all checkboxes that fall within any of the intervals
                const checkboxes = document.querySelectorAll('.step-checkbox');
                
                // First uncheck all
                checkboxes.forEach(cb => cb.checked = false);
                
                // Then check those that fall within intervals
                checkboxes.forEach(cb => {{
                    const step = parseInt(cb.value);
                    
                    for (const interval of intervals) {{
                        const [start, end] = interval;
                        if (step >= start && step <= end) {{
                            cb.checked = true;
                            break;
                        }}
                    }}
                }});
                
                // Update visualization
                updateVisualization();
                
            }} catch (error) {{
                alert('Error parsing intervals: ' + error.message);
            }}
        }}
        
        // Toggle between view types
        function toggleView(viewType) {{
            currentView = viewType;
            
            // Show/hide UI elements based on view type
            const modelSelectorContainer = document.getElementById('model-selector');
            const modelComparisonContainer = document.getElementById('model-comparison-container');
            const promptingSelectorContainer = document.getElementById('prompting-selector');
            
            // Reset display for all containers
            if (modelSelectorContainer) {{
                modelSelectorContainer.style.display = 'flex';
            }}
            if (promptingSelectorContainer) {{
                promptingSelectorContainer.style.display = 'flex';
            }}
            if (modelComparisonContainer) {{
                modelComparisonContainer.style.display = 'none';
            }}
            
            // Handle view-specific UI changes
            if (viewType === 'modeldiff') {{
                // Model diff - show comparison dropdowns, hide single model selector
                if (modelSelectorContainer) {{
                    modelSelectorContainer.style.display = 'none';
                }}
                if (modelComparisonContainer) {{
                    modelComparisonContainer.style.display = 'flex';
                    
                    // Set up model comparison dropdowns
                    const modelComparisonSelectors = document.querySelector('.model-comparison-selectors');
                    if (modelComparisonSelectors) {{
                        modelComparisonSelectors.innerHTML = `
                            <div class="dropdown-with-label">
                                <span class="model-label"><b>(A)</b></span>
                                <select id="model-base-dropdown" class="model-dropdown"></select>
                            </div>
                            <span class="comparison-vs">vs</span>
                            <div class="dropdown-with-label">
                                <span class="model-label"><b>(B)</b></span>
                                <select id="model-compare-dropdown" class="model-dropdown"></select>
                            </div>
                        `;
                    }}
                    
                    // Populate the model comparison dropdowns
                    const baseDropdown = document.getElementById('model-base-dropdown');
                    const compareDropdown = document.getElementById('model-compare-dropdown');
                    
                    if (baseDropdown && compareDropdown) {{
                        baseDropdown.onchange = updateBaseModel;
                        compareDropdown.onchange = updateCompareModel;
                        // Clear existing options
                        baseDropdown.innerHTML = '';
                        compareDropdown.innerHTML = '';
                        
                        // Add options for all model sizes
                        const modelSizes = Object.keys(fullModelResults);
                        
                        modelSizes.forEach((size, index) => {{
                            const baseOption = document.createElement('option');
                            baseOption.value = size;
                            baseOption.textContent = `Model Size: ${{size}}`;
                            
                            const compareOption = document.createElement('option');
                            compareOption.value = size;
                            compareOption.textContent = `Model Size: ${{size}}`;
                            
                            baseDropdown.appendChild(baseOption);
                            compareDropdown.appendChild(compareOption);
                            
                            // Select different models by default if possible
                            if (index === 0) {{
                                baseOption.selected = true;
                            }}
                            if (index === Math.min(1, modelSizes.length - 1)) {{
                                compareOption.selected = true;
                            }}
                        }});
                    }}
                    
                    // Add diff mode buttons
                    if (!document.getElementById('diff-mode-buttons')) {{
                        const diffModeButtons = document.createElement('div');
                        diffModeButtons.id = 'diff-mode-buttons';
                        diffModeButtons.className = 'diff-mode-buttons';
                        diffModeButtons.innerHTML = `
                            <div class="model-selector-title" style="margin-top: 10px;">Difference Mode:</div>
                            <div class="diff-mode-toggle">
                                <button id="absolute-diff-btn" class="diff-mode-button diff-mode-active" onclick="setDiffMode('absolute')">
                                    Absolute (B-A)
                                </button>
                                <button id="relative-diff-btn" class="diff-mode-button" onclick="setDiffMode('relative')">
                                    Relative (B/A)
                                </button>
                            </div>
                        `;
                        modelComparisonContainer.appendChild(diffModeButtons);
                    }}
                }}
            }} else if (viewType === 'promptdiff') {{
                // For prompting diff, hide prompting strategy selector
                if (promptingSelectorContainer) {{
                    promptingSelectorContainer.style.display = 'none';
                }}
            }}
            
            updateVisualization();
        }}
        
        // Set the difference mode for model comparisons
        function setDiffMode(mode) {{
            diffMode = mode;
            
            // Update button states
            const absoluteBtn = document.getElementById('absolute-diff-btn');
            const relativeBtn = document.getElementById('relative-diff-btn');
            
            if (absoluteBtn && relativeBtn) {{
                if (mode === 'absolute') {{
                    absoluteBtn.classList.add('diff-mode-active');
                    relativeBtn.classList.remove('diff-mode-active');
                }} else {{
                    absoluteBtn.classList.remove('diff-mode-active');
                    relativeBtn.classList.add('diff-mode-active');
                }}
            }}
            
            // Update the visualization
            updateVisualization();
        }}
        
        // Get the currently selected steps
        function getSelectedSteps() {{
            const checkboxes = document.querySelectorAll('.step-checkbox:checked');
            const steps = Array.from(checkboxes).map(cb => cb.value);
            
            // Return steps sorted numerically, not lexicographically
            return steps.sort((a, b) => parseInt(a) - parseInt(b));
        }}
        
        // Select all steps
        function selectAllSteps() {{
            const checkboxes = document.querySelectorAll('.step-checkbox');
            checkboxes.forEach(cb => cb.checked = true);
        }}
        
        // Deselect all steps
        function deselectAllSteps() {{
            const checkboxes = document.querySelectorAll('.step-checkbox');
            checkboxes.forEach(cb => cb.checked = false);
        }}
        
        // Update context when dropdown changes
        function updateContext() {{
            const dropdown = document.getElementById('context-dropdown');
            currentContext = dropdown.value;
            updateVisualization();
        }}
        
        // Update the visualization based on current selections
        function updateVisualization() {{
            const container = document.getElementById('visualization-container');
            const selectedSteps = getSelectedSteps();
            
            // Validate that we have at least one step selected
            if (selectedSteps.length === 0) {{
                alert('Please select at least one step to display.');
                return;
            }}
            
            // Generate HTML based on the current view
            let html = '';
            
            if (currentView === 'modeldiff') {{
                // For model diff view, get data from both selected models
                const baseModelSize = document.getElementById('model-base-dropdown').value;
                const compareModelSize = document.getElementById('model-compare-dropdown').value;
                
                const baseModelResults = fullModelResults[baseModelSize] || {{}};
                const compareModelResults = fullModelResults[compareModelSize] || {{}};
                
                // Use the current prompting strategy
                const baseResults = baseModelResults[currentPromptingStrategy] || {{}};
                const compareResults = compareModelResults[currentPromptingStrategy] || {{}};
                
                html = createModelDiffHtml(baseResults, compareResults, currentContext, selectedSteps, 
                                          baseModelSize, compareModelSize, currentPromptingStrategy);
            }}
            else if (currentView === 'promptdiff') {{
                // For prompting diff view, compare zero-shot and few-shot for the current model
                const modelResults = fullModelResults[currentModelSize] || {{}};
                const zeroShotResults = modelResults["zero_shot"] || {{}};
                const fewShotResults = modelResults["few_shot"] || {{}};
                
                html = createPromptingDiffHtml(zeroShotResults, fewShotResults, currentContext, selectedSteps);
            }}
            else {{
                // For raw or step-diff views, use the single selected model and prompting strategy
                const modelResults = fullModelResults[currentModelSize] || {{}};
                const results = modelResults[currentPromptingStrategy] || {{}};
                
                if (currentView === 'raw') {{
                    html = createSingleContextRawHtml(results, currentContext, selectedSteps);
                }} else {{
                    html = createSingleContextStepDiffHtml(results, currentContext, selectedSteps);
                }}
            }}
            
            container.innerHTML = html;
        }}

        // Create HTML for a single context with raw losses
        function createSingleContextRawHtml(results, contextId, selectedSteps) {{
            // Get data for the context
            const contextData = results[contextId];
            if (!contextData) {{
                return `<div class="context-container">
                    <div class="context-title">Context not available for this model size and prompting strategy</div>
                </div>`;
            }}
            
            const tokens = contextData["tokens"];
            const preview = contextData["preview"] || "";
            
            // Ensure steps are sorted numerically and exist in the data
            const steps = selectedSteps
                .filter(step => contextData["checkpoints"][step] && contextData["checkpoints"][step]["losses"])
                .sort((a, b) => parseInt(a) - parseInt(b));
            
            // Find max loss for normalization
            let maxLoss = Math.min(universalMaxLosses[contextId] || 0, 15);
            
            // Start building HTML
            let html = `
                <div class="context-container">
                    <div class="context-title">Context ${{contextId}} - ${{preview}} (Raw Loss - ${{currentPromptingStrategy.replace('_', '-')}})</div>
                    <div class="legend-container">
                        <div class="legend-item">
                            <div class="legend-color" style="background-color: rgb(255, 255, 255);"></div>
                            <span>Low Loss</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background-color: rgb(255, 0, 0);"></div>
                            <span>High Loss</span>
                        </div>
                    </div>
            `;
            
            // If no steps selected or available, show a message
            if (steps.length === 0) {{
                html += '<div>No steps selected or available for this context</div></div>';
                return html;
            }}
            
            // Get first step's losses to determine token count
            const firstStep = steps[0];
            const firstLosses = contextData["checkpoints"][firstStep]["losses"];
            const tokenCount = firstLosses ? firstLosses.length : 0;
            
            // Calculate chunks for wrapping
            const tokensPerRow = 20;
            const numChunks = Math.ceil(tokenCount / tokensPerRow);
            
            // Create the wrapped token grid
            html += `<div class="token-grid-wrapper"><div class="token-grid">`;
            
            // Process tokens in chunks
            for (let chunkIdx = 0; chunkIdx < numChunks; chunkIdx++) {{
                const startIdx = chunkIdx * tokensPerRow;
                const endIdx = Math.min(startIdx + tokensPerRow, tokenCount);
                
                // Add position header row for this chunk
                html += '<div class="token-row-header">';
                html += `<div class="header-cell" style="width: 235px;">Position</div>`;
                for (let i = startIdx; i < endIdx; i++) {{
                    html += `<div class="header-cell">${{i}}</div>`;
                }}
                html += '</div>';
                
                // Add token text header row for this chunk
                html += '<div class="token-row-header">';
                html += `<div class="header-cell token-header" style="width: 235px;">Token</div>`;
                for (let i = startIdx; i < endIdx; i++) {{
                    if (i < tokens.length) {{
                        // Handle special characters
                        let token = tokens[i];
                        let displayToken = token;
                        if (displayToken === ' ') {{
                            displayToken = '␣';
                        }} else if (displayToken === '\\n') {{
                            displayToken = '\\\\n';
                        }}
                        html += `<div class="header-cell">${{displayToken}}</div>`;
                    }} else {{
                        html += `<div class="header-cell">-</div>`;
                    }}
                }}
                html += '</div>';
                
                // Add a row for each checkpoint in this chunk
                for (const step of steps) {{
                    const losses = contextData["checkpoints"][step]["losses"];
                    if (!losses) continue;
                    
                    // Add step row
                    html += '<div class="token-row">';
                    html += `<div class="step-header" style="width: 235px;">Step ${{step}}</div>`;
                    
                    // Add loss-colored cells for this chunk
                    for (let i = startIdx; i < endIdx; i++) {{
                        if (i >= losses.length) {{
                            html += `<div class="token-cell" style="background-color: #ddd;">-</div>`;
                            continue;
                        }}
                        
                        const loss = losses[i];
                        
                        // Skip null values
                        if (loss === null) {{
                            html += `
                                <div class="token-cell" style="background-color: #ddd;">
                                    -
                                </div>
                            `;
                            continue;
                        }}
                        
                        // Normalize loss for color
                        const normalizedLoss = maxLoss > 0 ? Math.min(loss / maxLoss, 1) : 0;
                        
                        // Create color from white to red based on loss
                        const r = 255;
                        const g = Math.floor(255 * (1 - normalizedLoss));
                        const b = Math.floor(255 * (1 - normalizedLoss));
                        const color = `rgb(${{r}}, ${{g}}, ${{b}})`;
                        
                        // Get token for this position
                        let displayToken = '-';
                        if (i < tokens.length) {{
                            const token = tokens[i];
                            displayToken = token;
                            if (displayToken === ' ') {{
                                displayToken = '␣';
                            }} else if (displayToken === '\\n') {{
                                displayToken = '\\\\n';
                            }}
                        }}
                        
                        html += `
                        <div class="token-cell tooltip" style="background-color:${{color}};">
                            ${{displayToken}}
                            <span class="tooltiptext">Position: ${{i}}<br>Loss: ${{loss.toFixed(4)}}</span>
                        </div>
                        `;
                    }}
                    
                    html += '</div>'; // End token-row
                }}
                
                // Add spacing between chunks
                html += '<div style="height: 20px;"></div>';
            }}
            
            html += "</div></div></div>"; // End token-grid, token-grid-wrapper, context-container
            return html;
        }}
        
        // Create HTML for a single context with step-by-step differences
        function createSingleContextStepDiffHtml(results, contextId, selectedSteps) {{
            // Get data for the context
            const contextData = results[contextId];
            if (!contextData) {{
                return `<div class="context-container">
                    <div class="context-title">Context not available for this model size and prompting strategy</div>
                </div>`;
            }}
            
            const tokens = contextData["tokens"];
            const preview = contextData["preview"] || "";
            
            // Ensure steps are filtered and sorted numerically
            const steps = selectedSteps
                .filter(step => contextData["checkpoints"][step] && contextData["checkpoints"][step]["losses"])
                .sort((a, b) => parseInt(a) - parseInt(b));
            
            // If no steps or only one step, can't show differences
            if (steps.length < 2) {{
                return `<div class="context-container">
                    <div class="context-title">Context ${{contextId}} - ${{preview}} (Step-by-Step Differences - ${{currentPromptingStrategy.replace('_', '-')}})</div>
                    <div>At least two steps must be selected to show differences.</div>
                </div>`;
            }}
            
            // Find max loss for normalization
            let maxDiff = 3.0; 
            
            // Start building HTML
            let html = `
                <div class="context-container">
                    <div class="context-title">Context ${{contextId}} - ${{preview}} (Step-by-Step Differences - ${{currentPromptingStrategy.replace('_', '-')}})</div>
                    <div class="legend-container">
                        <div class="legend-item">
                            <div class="legend-color" style="background-color: rgb(255, 255, 255);"></div>
                            <span>No change</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background-color: rgb(0, 255, 0);"></div>
                            <span>Loss decreased (improved)</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background-color: rgb(255, 0, 0);"></div>
                            <span>Loss increased (worsened)</span>
                        </div>
                    </div>
            `;
            
            // Get first step's losses to determine token count
            const firstStep = steps[0];
            const firstLosses = contextData["checkpoints"][firstStep]["losses"];
            const tokenCount = firstLosses ? firstLosses.length : 0;
            
            // Calculate chunks for wrapping
            const tokensPerRow = 20;
            const numChunks = Math.ceil(tokenCount / tokensPerRow);
            
            // Create the wrapped token grid
            html += `<div class="token-grid-wrapper"><div class="token-grid">`;
            
            // Process tokens in chunks
            for (let chunkIdx = 0; chunkIdx < numChunks; chunkIdx++) {{
                const startIdx = chunkIdx * tokensPerRow;
                const endIdx = Math.min(startIdx + tokensPerRow, tokenCount);
                
                // Add position header row for this chunk
                html += '<div class="token-row-header">';
                html += `<div class="header-cell" style="width: 235px;">Position</div>`;
                for (let i = startIdx; i < endIdx; i++) {{
                    html += `<div class="header-cell">${{i}}</div>`;
                }}
                html += '</div>';
                
                // Add token text header row for this chunk
                html += '<div class="token-row-header">';
                html += `<div class="header-cell token-header" style="width: 235px;">Token</div>`;
                for (let i = startIdx; i < endIdx; i++) {{
                    if (i < tokens.length) {{
                        // Handle special characters
                        let token = tokens[i];
                        let displayToken = token;
                        if (displayToken === ' ') {{
                            displayToken = '␣';
                        }} else if (displayToken === '\\n') {{
                            displayToken = '\\\\n';
                        }}
                        html += `<div class="header-cell">${{displayToken}}</div>`;
                    }} else {{
                        html += `<div class="header-cell">-</div>`;
                    }}
                }}
                html += '</div>';
                
                // Add a row for each step (baseline + differences)
                for (let stepIdx = 0; stepIdx < steps.length; stepIdx++) {{
                    const step = steps[stepIdx];
                    const losses = contextData["checkpoints"][step]["losses"];
                    
                    if (!losses) continue;
                    
                    // Add checkpoint step header with description
                    let stepHeader;
                    if (stepIdx === 0) {{
                        stepHeader = `Step ${{step}} (Base)`;
                    }} else {{
                        const prevStep = steps[stepIdx-1];
                        stepHeader = `Step ${{step}} - ${{prevStep}}`;
                    }}
                    
                    // Add step row 
                    html += '<div class="token-row">';
                    html += `<div class="step-header" style="width: 235px;">${{stepHeader}}</div>`;
                    
                    // For the first step, show white cells (no difference)
                    if (stepIdx === 0) {{
                        for (let i = startIdx; i < endIdx; i++) {{
                            if (i >= losses.length) {{
                                html += `<div class="token-cell" style="background-color: #ddd;">-</div>`;
                                continue;
                            }}
                            
                            const loss = losses[i];
                            
                            // Get token for this position
                            let displayToken = '-';
                            if (i < tokens.length) {{
                                const token = tokens[i];
                                displayToken = token;
                                if (displayToken === ' ') {{
                                    displayToken = '␣';
                                }} else if (displayToken === '\\n') {{
                                    displayToken = '\\\\n';
                                }}
                            }}
                            
                            // First row is all white (baseline)
                            if (loss === null) {{
                                html += `
                                    <div class="token-cell tooltip" style="background-color: #ddd;">
                                        ${{displayToken}}
                                        <span class="tooltiptext">Position: ${{i}}<br>Loss: N/A</span>
                                    </div>
                                `;
                            }} else {{
                                html += `
                                    <div class="token-cell tooltip" style="background-color: rgb(255, 255, 255);">
                                        ${{displayToken}}
                                        <span class="tooltiptext">Position: ${{i}}<br>Loss: ${{loss.toFixed(4)}}</span>
                                    </div>
                                `;
                            }}
                        }}
                    }} else {{
                        // For subsequent steps, show difference from previous step
                        const prevStep = steps[stepIdx-1];
                        const prevLosses = contextData["checkpoints"][prevStep]["losses"];
                        
                        if (!prevLosses) {{
                            // If previous losses don't exist, show gray cells
                            for (let i = startIdx; i < endIdx; i++) {{
                                if (i >= losses.length) {{
                                    html += `<div class="token-cell" style="background-color: #ddd;">-</div>`;
                                    continue;
                                }}
                                
                                // Get token for this position
                                let displayToken = '-';
                                if (i < tokens.length) {{
                                    const token = tokens[i];
                                    displayToken = token;
                                    if (displayToken === ' ') {{
                                        displayToken = '␣';
                                    }} else if (displayToken === '\\n') {{
                                        displayToken = '\\\\n';
                                    }}
                                }}
                                
                                html += `
                                    <div class="token-cell tooltip" style="background-color: #ddd;">
                                        ${{displayToken}}
                                        <span class="tooltiptext">Position: ${{i}}<br>No previous data</span>
                                    </div>
                                `;
                            }}
                        }} else {{
                            for (let i = startIdx; i < endIdx; i++) {{
                                if (i >= losses.length) {{
                                    html += `<div class="token-cell" style="background-color: #ddd;">-</div>`;
                                    continue;
                                }}
                                
                                const loss = losses[i];
                                
                                // Get token for this position
                                let displayToken = '-';
                                if (i < tokens.length) {{
                                    const token = tokens[i];
                                    displayToken = token;
                                    if (displayToken === ' ') {{
                                        displayToken = '␣';
                                    }} else if (displayToken === '\\n') {{
                                        displayToken = '\\\\n';
                                    }}
                                }}
                                
                                // Handle null values
                                if (loss === null || i >= prevLosses.length || prevLosses[i] === null) {{
                                    html += `
                                        <div class="token-cell tooltip" style="background-color: #ddd;">
                                            ${{displayToken}}
                                            <span class="tooltiptext">Position: ${{i}}<br>Missing data</span>
                                        </div>
                                    `;
                                    continue;
                                }}
                                
                                // Calculate loss difference
                                const prevLoss = prevLosses[i];
                                const diff = loss - prevLoss;
                                
                                // Use a threshold to ignore very small changes
                                const significanceThreshold = 0.001;
                                
                                // Normalize difference for color intensity
                                let normalizedDiff = 0;
                                if (maxDiff > 0 && Math.abs(diff) > significanceThreshold) {{
                                    normalizedDiff = Math.min(Math.abs(diff) / maxDiff, 1);
                                }}
                                
                                let r, g, b, arrow = "";
                                
                                // Green for improvement (decrease), red for regression (increase)
                                if (diff < -significanceThreshold) {{  // Loss decreased (improved)
                                    r = 255 * (1 - normalizedDiff);
                                    g = 255;
                                    b = 255 * (1 - normalizedDiff);
                                    arrow = "↓";
                                }} else if (diff > significanceThreshold) {{  // Loss increased (worsened)
                                    r = 255;
                                    g = 255 * (1 - normalizedDiff);
                                    b = 255 * (1 - normalizedDiff);
                                    arrow = "↑";
                                }} else {{  // No change or very small change
                                    r = 255;
                                    g = 255;
                                    b = 255;
                                    arrow = "";
                                }}
                                
                                const color = `rgb(${{r}}, ${{g}}, ${{b}})`;
                                
                                html += `
                                    <div class="token-cell tooltip" style="background-color:${{color}};">
                                        ${{displayToken}}
                                        <span class="tooltiptext">Position: ${{i}}<br>Previous loss (Step ${{prevStep}}): ${{prevLoss.toFixed(4)}}<br>Current loss (Step ${{step}}): ${{loss.toFixed(4)}}<br>Difference: ${{diff.toFixed(4)}} ${{arrow}}</span>
                                    </div>
                                `;
                            }}
                        }}
                    }}
                    
                    html += '</div>'; // End token-row
                }}
                
                // Add spacing between chunks
                html += '<div style="height: 20px;"></div>';
            }}
            
            html += "</div></div></div>"; // End token-grid, token-grid-wrapper, context-container
            return html;
        }}

        // Create HTML for model size difference visualization
        function createModelDiffHtml(baseResults, compareResults, contextId, selectedSteps, 
                                   baseModelSize, compareModelSize, promptingStrategy) {{
            // Get data for the context from both models
            const baseContextData = baseResults[contextId];
            const compareContextData = compareResults[contextId];
            
            // Check if context exists in both models
            if (!baseContextData || !compareContextData) {{
                return `<div class="context-container">
                    <div class="context-title">Context data not available for both models with the selected prompting strategy</div>
                </div>`;
            }}
            
            const tokens = baseContextData["tokens"]; // Use tokens from base model
            const preview = baseContextData["preview"] || "";
            
            // Ensure steps are filtered and exist in both model datasets
            const steps = selectedSteps
                .filter(step => 
                    baseContextData["checkpoints"][step] && 
                    baseContextData["checkpoints"][step]["losses"] &&
                    compareContextData["checkpoints"][step] && 
                    compareContextData["checkpoints"][step]["losses"]
                )
                .sort((a, b) => parseInt(a) - parseInt(b));
            
            // If no matching steps, show an error
            if (steps.length === 0) {{
                return `<div class="context-container">
                    <div class="context-title">Context ${{contextId}} - ${{preview}} (Model Differences)</div>
                    <div>No matching steps found in both models.</div>
                </div>`;
            }}

            // Set fixed maximum difference for normalization based on mode
            let maxDiff;
            if (diffMode === 'absolute') {{
                maxDiff = 3.0;  // Fixed max diff for absolute scale: [-3, 3]
            }} else {{
                maxDiff = 1.0;  // Fixed max diff for relative scale: [-100%, 100%]
            }}
            
            // Update the title text based on the difference mode and prompting strategy
            let titleText = '';
            if (diffMode === 'absolute') {{
                titleText = `Context ${{contextId}} - ${{preview}} (Model Differences: ${{compareModelSize}} - ${{baseModelSize}}, ${{promptingStrategy.replace('_', '-')}})`;
            }} else {{
                titleText = `Context ${{contextId}} - ${{preview}} (Model Differences: ${{compareModelSize}} / ${{baseModelSize}}, ${{promptingStrategy.replace('_', '-')}})`;
            }}
            
            // Start building HTML
            let html = `
                <div class="context-container">
                    <div class="context-title">${{titleText}}</div>
                    <div class="model-diff-legend">
                        <div class="legend-item">
                            <div class="legend-color" style="background-color: rgb(0, 255, 0);"></div>
                            <span>${{compareModelSize}} performs better than ${{baseModelSize}} (lower loss)</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background-color: rgb(255, 255, 255);"></div>
                            <span>No significant difference</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background-color: rgb(255, 0, 0);"></div>
                            <span>${{compareModelSize}} performs worse than ${{baseModelSize}} (higher loss)</span>
                        </div>
                    </div>
            `;
            
            // Get token count from base model's first checkpoint
            const firstStep = steps[0];
            const firstLosses = baseContextData["checkpoints"][firstStep]["losses"];
            const tokenCount = firstLosses ? firstLosses.length : 0;
            
            // Calculate chunks for wrapping
            const tokensPerRow = 20;
            const numChunks = Math.ceil(tokenCount / tokensPerRow);
            
            // Create the wrapped token grid
            html += `<div class="token-grid-wrapper"><div class="token-grid">`;
            
            // Process tokens in chunks
            for (let chunkIdx = 0; chunkIdx < numChunks; chunkIdx++) {{
                const startIdx = chunkIdx * tokensPerRow;
                const endIdx = Math.min(startIdx + tokensPerRow, tokenCount);
                
                // Add position header row for this chunk
                html += '<div class="token-row-header">';
                html += `<div class="header-cell" style="width: 235px;">Position</div>`;
                for (let i = startIdx; i < endIdx; i++) {{
                    html += `<div class="header-cell">${{i}}</div>`;
                }}
                html += '</div>';
                
                // Add token text header row for this chunk
                html += '<div class="token-row-header">';
                html += `<div class="header-cell token-header" style="width: 235px;">Token</div>`;
                for (let i = startIdx; i < endIdx; i++) {{
                    if (i < tokens.length) {{
                        // Handle special characters
                        let token = tokens[i];
                        let displayToken = token;
                        if (displayToken === ' ') {{
                            displayToken = '␣';
                        }} else if (displayToken === '\\n') {{
                            displayToken = '\\\\n';
                        }}
                        html += `<div class="header-cell">${{displayToken}}</div>`;
                    }} else {{
                        html += `<div class="header-cell">-</div>`;
                    }}
                }}
                html += '</div>';
                
                // Add a row for each step
                for (const step of steps) {{
                    const baseLosses = baseContextData["checkpoints"][step]["losses"];
                    const compareLosses = compareContextData["checkpoints"][step]["losses"];
                    
                    // Add step row
                    html += '<div class="token-row">';
                    html += `<div class="step-header" style="width: 235px;">Step ${{step}}</div>`;
                    
                    // Add diff-colored cells for this chunk
                    for (let i = startIdx; i < endIdx; i++) {{
                        // Get token for this position
                        let displayToken = '-';
                        if (i < tokens.length) {{
                            const token = tokens[i];
                            displayToken = token;
                            if (displayToken === ' ') {{
                                displayToken = '␣';
                            }} else if (displayToken === '\\n') {{
                                displayToken = '\\\\n';
                            }}
                        }}
                        
                        // Handle cases where data is missing
                        if (i >= baseLosses.length || i >= compareLosses.length || 
                            baseLosses[i] === null || compareLosses[i] === null) {{
                            html += `
                                <div class="token-cell tooltip" style="background-color: #ddd;">
                                    ${{displayToken}}
                                    <span class="tooltiptext">Position: ${{i}}<br>Missing data</span>
                                </div>
                            `;
                            continue;
                        }}
                        
                        // Calculate difference based on the current mode
                        const baseLoss = baseLosses[i];
                        const compareLoss = compareLosses[i];
                        let diff, diffDisplay, symbol = "";
                        
                        if (diffMode === 'absolute') {{
                            diff = compareLoss - baseLoss;
                            diffDisplay = diff.toFixed(4);
                        }} else {{ // relative
                            if (baseLoss === 0) {{
                                // Handle division by zero
                                if (compareLoss === 0) {{
                                    diff = 0; // Both are zero, no difference
                                    diffDisplay = "0%";
                                }} else {{
                                    diff = 1; // Base is zero but compare isn't, large difference
                                    diffDisplay = "∞"; // Infinity symbol for division by zero
                                }}
                            }} else {{
                                diff = (compareLoss / baseLoss) - 1;
                                diffDisplay = (diff * 100).toFixed(2) + "%"; // Format as percentage
                            }}
                        }}
                        
                        // Use a threshold to ignore very small changes
                        const significanceThreshold = diffMode === 'absolute' ? 0.001 : 0.01; // 1% for relative
                        
                        // Normalize difference for color intensity
                        let normalizedDiff = 0;
                        if (maxDiff > 0 && Math.abs(diff) > significanceThreshold) {{
                            normalizedDiff = Math.min(Math.abs(diff) / maxDiff, 1);
                        }}
                        
                        let r, g, b;
                        
                        // Green for improvement in compare model (lower loss), red for worse performance (higher loss)
                        if (diff < -significanceThreshold) {{  // Compare model has lower loss (better)
                            r = 255 * (1 - normalizedDiff);
                            g = 255;
                            b = 255 * (1 - normalizedDiff);
                            symbol = "↓";
                        }} else if (diff > significanceThreshold) {{  // Compare model has higher loss (worse)
                            r = 255;
                            g = 255 * (1 - normalizedDiff);
                            b = 255 * (1 - normalizedDiff);
                            symbol = "↑";
                        }} else {{  // No significant difference
                            r = 255;
                            g = 255;
                            b = 255;
                            symbol = "=";
                        }}
                        
                        const color = `rgb(${{r}}, ${{g}}, ${{b}})`;
                        
                        html += `
                            <div class="token-cell tooltip" style="background-color:${{color}};">
                                ${{displayToken}}
                                <span class="tooltiptext">
                                    Position: ${{i}}<br>
                                    ${{baseModelSize}} loss: ${{baseLoss.toFixed(4)}}<br>
                                    ${{compareModelSize}} loss: ${{compareLoss.toFixed(4)}}<br>
                                    ${{diffMode === 'absolute' ? 'Difference: ' : 'Relative Change: '}}${{diffDisplay}}
                                </span>
                            </div>
                        `;
                    }}
                    
                    html += '</div>'; // End token-row
                }}
                
                // Add spacing between chunks
                html += '<div style="height: 20px;"></div>';
            }}
            
            html += "</div></div></div>"; // End token-grid, token-grid-wrapper, context-container
            return html;
        }}
        
        // Create HTML for prompting strategy comparison (zero-shot vs few-shot)
        function createPromptingDiffHtml(zeroShotResults, fewShotResults, contextId, selectedSteps) {{
            // Get data for the context from both strategies
            const zeroShotContextData = zeroShotResults[contextId];
            const fewShotContextData = fewShotResults[contextId];
            
            // Check if context exists in both strategies
            if (!zeroShotContextData || !fewShotContextData) {{
                return `<div class="context-container">
                    <div class="context-title">Context data not available for both prompting strategies</div>
                </div>`;
            }}
            
            // Prefer zero-shot tokens for consistency
            const tokens = zeroShotContextData["tokens"]; 
            const preview = zeroShotContextData["preview"] || "";
            
            // Ensure steps are filtered and exist in both datasets
            const steps = selectedSteps
                .filter(step => 
                    zeroShotContextData["checkpoints"][step] && 
                    zeroShotContextData["checkpoints"][step]["losses"] &&
                    fewShotContextData["checkpoints"][step] && 
                    fewShotContextData["checkpoints"][step]["losses"]
                )
                .sort((a, b) => parseInt(a) - parseInt(b));
            
            // If no matching steps, show an error
            if (steps.length === 0) {{
                return `<div class="context-container">
                    <div class="context-title">Context ${{contextId}} - ${{preview}} (Prompting Strategy Differences)</div>
                    <div>No matching steps found in both zero-shot and few-shot data.</div>
                </div>`;
            }}

            // Set fixed maximum difference for normalization
            const maxDiff = 3.0;  // Fixed max diff
            
            // Start building HTML
            let html = `
                <div class="context-container">
                    <div class="context-title">Context ${{contextId}} - ${{preview}} (Prompting Strategy Differences: Few-Shot vs Zero-Shot)</div>
                    <div class="model-diff-legend">
                        <div class="legend-item">
                            <div class="legend-color" style="background-color: rgb(0, 255, 0);"></div>
                            <span>Few-shot better than zero-shot (lower loss)</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background-color: rgb(255, 255, 255);"></div>
                            <span>No significant difference</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background-color: rgb(255, 0, 0);"></div>
                            <span>Zero-shot better than few-shot (lower loss)</span>
                        </div>
                    </div>
            `;
            
            // Get token count from first checkpoint
            const firstStep = steps[0];
            const firstLosses = zeroShotContextData["checkpoints"][firstStep]["losses"];
            const tokenCount = firstLosses ? firstLosses.length : 0;
            
            // Calculate chunks for wrapping
            const tokensPerRow = 20;
            const numChunks = Math.ceil(tokenCount / tokensPerRow);
            
            // Create the wrapped token grid
            html += `<div class="token-grid-wrapper"><div class="token-grid">`;
            
            // Process tokens in chunks
            for (let chunkIdx = 0; chunkIdx < numChunks; chunkIdx++) {{
                const startIdx = chunkIdx * tokensPerRow;
                const endIdx = Math.min(startIdx + tokensPerRow, tokenCount);
                
                // Add position header row for this chunk
                html += '<div class="token-row-header">';
                html += `<div class="header-cell" style="width: 235px;">Position</div>`;
                for (let i = startIdx; i < endIdx; i++) {{
                    html += `<div class="header-cell">${{i}}</div>`;
                }}
                html += '</div>';
                
                // Add token text header row for this chunk
                html += '<div class="token-row-header">';
                html += `<div class="header-cell token-header" style="width: 235px;">Token</div>`;
                for (let i = startIdx; i < endIdx; i++) {{
                    if (i < tokens.length) {{
                        // Handle special characters
                        let token = tokens[i];
                        let displayToken = token;
                        if (displayToken === ' ') {{
                            displayToken = '␣';
                        }} else if (displayToken === '\\n') {{
                            displayToken = '\\\\n';
                        }}
                        html += `<div class="header-cell">${{displayToken}}</div>`;
                    }} else {{
                        html += `<div class="header-cell">-</div>`;
                    }}
                }}
                html += '</div>';
                
                // Add a row for each step
                for (const step of steps) {{
                    const zeroShotLosses = zeroShotContextData["checkpoints"][step]["losses"];
                    const fewShotLosses = fewShotContextData["checkpoints"][step]["losses"];
                    
                    // Add step row
                    html += '<div class="token-row">';
                    html += `<div class="step-header" style="width: 235px;">Step ${{step}} (Few-Zero)</div>`;
                    
                    // Add diff-colored cells for this chunk
                    for (let i = startIdx; i < endIdx; i++) {{
                        // Get token for this position
                        let displayToken = '-';
                        if (i < tokens.length) {{
                            const token = tokens[i];
                            displayToken = token;
                            if (displayToken === ' ') {{
                                displayToken = '␣';
                            }} else if (displayToken === '\\n') {{
                                displayToken = '\\\\n';
                            }}
                        }}
                        
                        // Handle cases where data is missing
                        if (i >= zeroShotLosses.length || i >= fewShotLosses.length || 
                            zeroShotLosses[i] === null || fewShotLosses[i] === null) {{
                            html += `
                                <div class="token-cell tooltip" style="background-color: #ddd;">
                                    ${{displayToken}}
                                    <span class="tooltiptext">Position: ${{i}}<br>Missing data</span>
                                </div>
                            `;
                            continue;
                        }}
                        
                        // Calculate the difference: few-shot minus zero-shot
                        const zeroShotLoss = zeroShotLosses[i];
                        const fewShotLoss = fewShotLosses[i];
                        const diff = fewShotLoss - zeroShotLoss;
                        
                        // Use a threshold to ignore very small changes
                        const significanceThreshold = 0.001;
                        
                        // Normalize difference for color intensity
                        let normalizedDiff = 0;
                        if (maxDiff > 0 && Math.abs(diff) > significanceThreshold) {{
                            normalizedDiff = Math.min(Math.abs(diff) / maxDiff, 1);
                        }}
                        
                        let r, g, b, symbol = "";
                        
                        // Green for few-shot better (lower loss), red for zero-shot better
                        if (diff < -significanceThreshold) {{  // Few-shot has lower loss (better)
                            r = 255 * (1 - normalizedDiff);
                            g = 255;
                            b = 255 * (1 - normalizedDiff);
                            symbol = "↓";
                        }} else if (diff > significanceThreshold) {{  // Zero-shot has lower loss (better)
                            r = 255;
                            g = 255 * (1 - normalizedDiff);
                            b = 255 * (1 - normalizedDiff);
                            symbol = "↑";
                        }} else {{  // No significant difference
                            r = 255;
                            g = 255;
                            b = 255;
                            symbol = "=";
                        }}
                        
                        const color = `rgb(${{r}}, ${{g}}, ${{b}})`;
                        
                        html += `
                            <div class="token-cell tooltip" style="background-color:${{color}};">
                                ${{displayToken}}
                                <span class="tooltiptext">
                                    Position: ${{i}}<br>
                                    Zero-shot loss: ${{zeroShotLoss.toFixed(4)}}<br>
                                    Few-shot loss: ${{fewShotLoss.toFixed(4)}}<br>
                                    Difference (F-Z): ${{diff.toFixed(4)}} ${{symbol}}
                                </span>
                            </div>
                        `;
                    }}
                    
                    html += '</div>'; // End token-row
                }}
                
                // Add spacing between chunks
                html += '<div style="height: 20px;"></div>';
            }}
            
            html += "</div></div></div>"; // End token-grid, token-grid-wrapper, context-container
            return html;
        }}

        // Initialize the visualization when the page loads
        document.addEventListener('DOMContentLoaded', function() {{
            // Set default context and model
            const contextDropdown = document.getElementById('context-dropdown');
            if (contextDropdown) {{
                contextDropdown.value = currentContext;
            }}
            
            const modelDropdown = document.getElementById('model-dropdown');
            if (modelDropdown) {{
                modelDropdown.value = currentModelSize;
            }}

            // Set default prompting strategy
            const promptingRadios = document.getElementsByName('promptingToggle');
            for (const radio of promptingRadios) {{
                if (radio.value === currentPromptingStrategy) {{
                    radio.checked = true;
                    break;
                }}
            }}
            
            // Select evenly spaced steps (maximum 15)
            const allCheckboxes = document.querySelectorAll('.step-checkbox');
            const allSteps = Array.from(allCheckboxes);
            
            // First uncheck all steps
            allSteps.forEach(cb => cb.checked = false);
            
            // If we have more than 15 steps, select evenly spaced ones
            if (allSteps.length > 15) {{
                // Calculate the step size to get approximately 15 evenly spaced steps
                const stepSize = Math.max(1, Math.floor(allSteps.length / 15));
                
                // Select every nth step
                for (let i = 0; i < allSteps.length; i += stepSize) {{
                    if (allSteps[i]) {{
                        allSteps[i].checked = true;
                    }}
                }}
                
                // Ensure we always include the first and last step for better context
                if (allSteps[0]) {{
                    allSteps[0].checked = true;
                }}
                if (allSteps[allSteps.length - 1]) {{
                    allSteps[allSteps.length - 1].checked = true;
                }}
            }} else {{
                // If we have 15 or fewer steps, select all of them
                allSteps.forEach(cb => cb.checked = true);
            }}
            
            // Update the visualization with the evenly spaced steps
            updateVisualization();
        }});
    </script>
    """

def create_single_context_raw_html(results, context_idx, selected_steps, max_loss=None):
    """
    Create initial HTML for a single context with raw losses.
    This will be displayed before JavaScript takes over.
    
    Args:
        results: Dictionary with token loss results
        context_idx: Index of the context to display
        selected_steps: List of steps to display
        
    Returns:
        HTML string with the raw loss visualization
    """
    # Get data for the specified context
    context_data = results[context_idx]
    tokens = context_data["tokens"]
    preview = context_data.get("preview", "")
    
    # Filter to only include selected steps that exist in the data and sort them numerically
    if selected_steps is None:
        # Use all available steps if none specified
        steps = sorted(list(context_data["checkpoints"].keys()), key=int)
    else:
        # Otherwise, filter to only include selected steps that exist in the data
        steps = sorted([step for step in selected_steps if step in context_data["checkpoints"]], key=int)
        
    # Find max loss across selected checkpoints for normalization
    # If max_loss not provided, calculate it from this context's data
    if max_loss is None:
        max_loss = 0
        for step in steps:
            losses = context_data["checkpoints"][step]["losses"]
            if losses:
                valid_losses = [loss for loss in losses if loss is not None]
                if valid_losses:
                    max_loss = max(max_loss, max(valid_losses))
    
    
    # Start a container for this context
    html = f"""
    <div class="context-container">
        <div class="context-title">Context {context_idx} - {preview} (Raw Loss)</div>
        <div class="legend-container">
            <div class="legend-item">
                <div class="legend-color" style="background-color: rgb(255, 255, 255);"></div>
                <span>Low Loss</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: rgb(255, 0, 0);"></div>
                <span>High Loss</span>
            </div>
        </div>
    """
    
    # If no steps selected or available, show a message
    if not steps:
        html += "<div>No steps selected or available for this context</div></div>"
        return html
    
    # Get first step's losses to determine number of tokens to display
    first_step = steps[0]
    first_losses = context_data["checkpoints"][first_step]["losses"]
    token_count = len(first_losses)
    
    # Calculate number of chunks (wrap every 20 tokens)
    tokens_per_row = 20
    num_chunks = (token_count + tokens_per_row - 1) // tokens_per_row
    
    # Create the token grid with wrapped chunks
    html += '<div class="token-grid-wrapper">'
    html += '<div class="token-grid">'
    
    # Process tokens in chunks of 20
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * tokens_per_row
        end_idx = min(start_idx + tokens_per_row, token_count)
        chunk_size = end_idx - start_idx
        
        # Add position header row for this chunk
        html += '<div class="token-row-header">'
        html += f'<div class="header-cell" style="width: 235px;">Position</div>'
        for i in range(start_idx, end_idx):
            html += f'<div class="header-cell">{i}</div>'
        html += '</div>'
        
        # Add token text header row for this chunk
        html += '<div class="token-row-header">'
        html += f'<div class="header-cell token-header" style="width: 235px;">Token</div>'
        for i in range(start_idx, end_idx):
            if i < len(tokens):
                # Handle special characters
                token = tokens[i]
                display_token = token
                if display_token == ' ':
                    display_token = '␣'
                elif display_token == '\n':
                    display_token = '\\n'
                html += f'<div class="header-cell" style="width: 30px;">{display_token}</div>'
            else:
                html += f'<div class="header-cell" style="width: 30px;">-</div>'
        html += '</div>'
        
        # Add a row for each checkpoint in this chunk
        for step in steps:
            losses = context_data["checkpoints"][step]["losses"]
            if not losses:
                continue
            
            # Add step row
            html += '<div class="token-row">'
            html += f'<div class="step-header" style="width: 235px;">Step {step}</div>'
            
            # Add loss-colored cells for this chunk
            for i in range(start_idx, end_idx):
                if i >= len(losses):
                    html += '<div class="token-cell" style="width: 30px; background-color: #ddd;">-</div>'
                    continue
                
                loss = losses[i]
                # Skip null values
                if loss is None:
                    html += f'<div class="token-cell" style="width: 30px; background-color: #ddd;">-</div>'
                    continue
                
                # Normalize loss for color
                normalized_loss = min(loss / max_loss, 1) if max_loss > 0 else 0
                
                # Create color from white to red based on loss
                r = 255
                g = int(255 * (1 - normalized_loss))
                b = int(255 * (1 - normalized_loss))
                color = f"rgb({r}, {g}, {b})"
                
                # Get token for this position
                display_token = '-'
                if i < len(tokens):
                    token = tokens[i]
                    display_token = token
                    if display_token == ' ':
                        display_token = '␣'
                    elif display_token == '\n':
                        display_token = '\\n'
                
                html += f'''
                <div class="token-cell tooltip" style="background-color:{color};">
                    {display_token}
                    <span class="tooltiptext">Position: {i}<br>Loss: {loss:.4f}</span>
                </div>
                '''
            
            html += '</div>'  # End token-row
        
        # Add spacing between chunks
        html += '<div style="height: 20px;"></div>'
    
    html += "</div></div></div>"  # End token-grid, token-grid-wrapper, context-container
    return html

def load_dm_math_data(model_sizes, base_dir=None, max_contexts=None, selected_steps=None):
    """
    Load and process the token and loss data for multiple model sizes on dm_mathematics dataset,
    including both zero-shot and few-shot versions.
    
    Args:
        model_sizes: List of model sizes to visualize
        base_dir: Base directory for trajectory data (or None for default)
        max_contexts: Maximum number of contexts to include (or None for all)
        selected_steps: List of specific steps to include (or None for all)
        
    Returns:
        Dictionary mapping model sizes to their processed results for both zero-shot and few-shot
    """
    # Set default base directory if not provided
    if base_dir is None:
        base_dir = "/path/to/dm_math/trajectories"
    
    # Define directories
    csv_dir = os.path.join(base_dir, "csv")
    tokens_dir = os.path.join(base_dir, "tokens")
    
    # Load token data for both datasets
    print('Loading token data...')
    zero_shot_tokens = load_token_data(tokens_dir, "dm_mathematics_zero_shot")
    few_shot_tokens = load_token_data(tokens_dir, "dm_mathematics_few_shot")
    
    if not zero_shot_tokens or not few_shot_tokens:
        print("Error: Missing token data for one or both datasets")
        return {}
    
    # Store results for each model size
    all_models_results = {}
    
    # Process each model size
    for model_size in model_sizes:
        model_name = f"{model_size}"
        print(f'Loading data for {model_name}...')
        
        model_results = {"zero_shot": {}, "few_shot": {}}
        
        # Load zero-shot data
        zero_shot_df = load_loss_data(csv_dir, model_name, "dm_mathematics_zero_shot", max_contexts)
        if not zero_shot_df.empty:
            zero_shot_results = extract_context_data(zero_shot_df, zero_shot_tokens, max_contexts, selected_steps)
            model_results["zero_shot"] = zero_shot_results
        else:
            print(f"Warning: No zero-shot data found for model '{model_name}'")
        
        # Load few-shot data
        few_shot_df = load_loss_data(csv_dir, model_name, "dm_mathematics_few_shot", max_contexts)
        if not few_shot_df.empty:
            few_shot_results = extract_context_data(few_shot_df, few_shot_tokens, max_contexts, selected_steps)
            model_results["few_shot"] = few_shot_results
        else:
            print(f"Warning: No few-shot data found for model '{model_name}'")
        
        # Store results for this model size
        if model_results["zero_shot"] or model_results["few_shot"]:
            all_models_results[model_size] = model_results
    
    if not all_models_results:
        print(f"Error: No valid data found for any model size")
        return {}
    
    return all_models_results


def visualize_from_loaded_data(all_models_results, dataset_name, steps=None):
    """
    Create and display a per-token loss visualization using pre-loaded data.
    
    Args:
        all_models_results: Dictionary mapping model sizes to their results
        dataset_name: Name of the dataset to visualize (for display purposes only)
        steps: List of steps to display initially (or None for all)
        
    Returns:
        IPython.display.HTML object
    """
    # Check if we have any model data
    if not all_models_results:
        return HTML("<div>Error: No model data available</div>")
    
    # If steps is None, try to get available steps from the first model's first context
    if steps is None:
        default_model = next(iter(all_models_results.values()))
        if default_model:
            first_context = next(iter(default_model.values()))
            if first_context and "checkpoints" in first_context:
                steps = list(first_context["checkpoints"].keys())
    
    # Create HTML visualization
    html_content = create_dm_math_pertoken_html(all_models_results, steps, dataset_name=dataset_name)
    
    # Return as displayable HTML
    return HTML(html_content)

def visualize_pertoken_losses(model_name, dataset_name, base_dir=None, steps=None, max_contexts=None):
    """
    Create and display a per-token loss visualization for a specific model and dataset.
    This function is kept for backward compatibility but internally uses the split functions.
    
    Args:
        model_name: Name of the model to visualize
        dataset_name: Name of the dataset to visualize
        base_dir: Base directory for trajectory data (or None for default)
        steps: List of steps to display initially (or None for all)
        max_contexts: Maximum number of contexts to include (or None for all)
        
    Returns:
        IPython.display.HTML object
    """
    # Load data
    results = load_dm_math_data(model_name, dataset_name, base_dir, max_contexts, selected_steps=steps)
    if not results:
        return HTML("<div>Error: Failed to load data</div>")
    
    # Visualize
    return visualize_from_loaded_data(results, model_name, dataset_name, steps)