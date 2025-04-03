import asyncio
import sys
import os
import modal

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Fix for binary incompatibility between numpy and pandas/torch
import streamlit as st

# Core imports needed immediately
import numpy as np 
import pandas as pd

# Lazy load other imports
@st.cache_resource
def load_ml_packages():
    import torch
    import matplotlib.pyplot as plt
    import wandb
    return torch, plt, wandb

# Configure wandb defaults
os.environ["WANDB_ENTITY"] = "smilai"
os.environ["WANDB_PROJECT"] = "sd1"

# Now import the rest of the modules
from src.data_processing.data_loader import DataLoader
from src.data_processing.transformers import DataTransformer
@st.cache_resource
def load_selected_model(model_type):
    if model_type == 'TableGAN':
        from src.models.table_gan import TableGAN
        return TableGAN
    elif model_type == 'WGAN':
        from src.models.wgan import WGAN
        return WGAN
    elif model_type == 'CGAN':
        from src.models.cgan import CGAN
        return CGAN
    elif model_type == 'TVAE':
        from src.models.tvae import TVAE
        return TVAE
    elif model_type == 'CTGAN':
        from src.models.ctgan import CTGAN
        return CTGAN  # Added import for CTGAN
from src.utils.validation import validate_data, check_column_types
from src.utils.evaluation import DataEvaluator
from src.ui import components
from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization

# Import Modal last to avoid conflicts
# This helps with the numpy._core module not found error
from src.models.modal_gan import ModalGAN

# Configure Streamlit page first, before any other operations
st.set_page_config(page_title="Synthetic Data Generator", layout="wide")

# Initialize PyTorch and configure device
from src.utils.torch_init import init_torch

device = init_torch()


# Initialize event loop for async operations if needed
def init_async():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)


# Only initialize async when needed
if 'async_loop_initialized' not in st.session_state:
    init_async()
    st.session_state['async_loop_initialized'] = True

# Store device in session state for consistency
st.session_state['device'] = device

# Initialize Modal resources
modal_gan = ModalGAN()


def main():
    st.title("Synthetic Tabular Data Generator")

    # File upload
    df, error = components.file_uploader()
    if error:
        st.error(error)
        return
    if df is None:
        return

    # Store DataFrame in session state for CGAN condition column selector
    st.session_state['uploaded_df'] = df

    # Validate data
    valid, issues = validate_data(df)
    if not valid:
        st.warning("Data validation issues found:")
        for issue in issues:
            st.write(f"- {issue}")

    # Store original column order
    original_columns = df.columns.tolist()

    # Data preview
    components.data_preview(df)

    # Missing values handling
    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        st.subheader("Missing Values Handling")
        missing_handling = st.radio(
            "How would you like to handle missing values?",
            options=[
                "Impute with mean/mode", "Drop rows with missing values",
                "Keep as is"
            ],
            index=0,
            help="Choose how to handle missing data before training the model")

        if missing_handling == "Drop rows with missing values":
            original_count = len(df)
            df = df.dropna()
            st.info(
                f"Dropped {original_count - len(df)} rows with missing values. {len(df)} rows remaining."
            )
        elif missing_handling == "Impute with mean/mode":
            st.info(
                "Missing values will be imputed during the transformation process."
            )
        # "Keep as is" doesn't require any action

    # Column type selection
    column_types = components.column_type_selector(df)

    # Validate column types
    valid, issues = check_column_types(column_types, df)
    if not valid:
        st.error("Column type validation failed:")
        for issue in issues:
            st.write(f"- {issue}")
        return

    # Transformation selection
    transformations = components.transformation_selector(column_types)

    # Model configuration
    model_config = model_config_section()

    # Add condition column selector for CGAN
    if model_config['model_type'] == 'CGAN':
        if 'uploaded_df' in st.session_state:
            model_config['condition_column'] = st.selectbox(
                "Select condition column for CGAN",
                options=st.session_state['uploaded_df'].columns.tolist(),
                help=
                "This column will be used as a condition for generating data")

            # Now add selector for specific condition values to generate
            if 'condition_column' in model_config and model_config[
                    'condition_column']:
                unique_values = st.session_state['uploaded_df'][
                    model_config['condition_column']].unique().tolist()
                model_config['condition_values'] = st.multiselect(
                    f"Select specific values from '{model_config['condition_column']}' to generate",
                    options=unique_values,
                    default=unique_values[:min(3, len(unique_values))],
                    help=
                    "CGAN will generate data only for these selected condition values"
                )

                # Add ratio selector for each selected condition value
                if model_config['condition_values']:
                    st.write(
                        "Set the proportion of samples to generate for each condition value:"
                    )
                    condition_ratios = {}
                    total_ratio = 0

                    cols = st.columns(
                        min(3, len(model_config['condition_values'])))
                    for i, value in enumerate(
                            model_config['condition_values']):
                        col_idx = i % len(cols)
                        with cols[col_idx]:
                            ratio = st.slider(
                                f"Ratio for '{value}'",
                                min_value=1,
                                max_value=10,
                                value=10 //
                                len(model_config['condition_values']),
                                help=
                                "Relative proportion of samples with this condition"
                            )
                            condition_ratios[value] = ratio
                            total_ratio += ratio

                    # Normalize ratios
                    model_config['condition_ratios'] = {
                        k: v / total_ratio
                        for k, v in condition_ratios.items()
                    }
        else:
            st.warning(
                "Please upload data first to select a condition column for CGAN"
            )

    # Add Modal training option
    use_modal = st.checkbox("Use Modal for cloud training (faster)",
                            value=True)

    # Column selection for synthetic data generation
    st.subheader("Column Selection")

    col1, col2 = st.columns(2)

    with col1:
        selection_mode = st.radio(
            "Column Selection Mode",
            options=[
                "Use all columns", "Choose columns to include",
                "Choose columns to exclude"
            ],
            index=0,
            help="Select which columns to use for synthetic data generation")

    selected_columns = original_columns.copy()

    if selection_mode == "Choose columns to include":
        with col2:
            selected_columns = st.multiselect(
                "Select columns to include",
                options=original_columns,
                default=original_columns,
                help=
                "Only these columns will be used for synthetic data generation"
            )
            if not selected_columns:
                st.warning("You must select at least one column")
    elif selection_mode == "Choose columns to exclude":
        with col2:
            excluded_columns = st.multiselect(
                "Select columns to exclude",
                options=original_columns,
                default=[],
                help=
                "These columns will be excluded from synthetic data generation"
            )
            selected_columns = [
                col for col in original_columns if col not in excluded_columns
            ]
            if not selected_columns:
                st.warning("You cannot exclude all columns")

    # Target column selection for ML evaluation
    st.subheader("ML Evaluation Settings")

    target_col = st.selectbox(
        "Select target column for ML utility evaluation",
        options=df.columns.tolist(),
        help=
        "This column will be used to evaluate how well the synthetic data preserves predictive relationships"
    )

    task_type = st.selectbox(
        "Select task type",
        options=['classification', 'regression'],
        help=
        "Choose 'classification' for categorical targets, 'regression' for continuous targets"
    )

    if st.button("Generate Synthetic Data"):
        with st.spinner("Preparing data..."):
            try:
                # Split data into train and test sets
                train_df, test_df = train_test_split(df,
                                                     test_size=0.2,
                                                     random_state=42)
                st.info(
                    f"Data split into {len(train_df)} training samples and {len(test_df)} test samples"
                )

                # Transform data
                transformer = DataTransformer()
                transformed_data = pd.DataFrame()

                # Validate that selected columns exist in the dataframe
                missing_cols = [
                    col for col in selected_columns
                    if col not in train_df.columns
                ]
                if missing_cols:
                    st.error(
                        f"The following selected columns are missing from the dataset: {', '.join(missing_cols)}"
                    )
                    return

                # Only use columns that exist in the dataframe
                valid_columns = [
                    col for col in selected_columns if col in train_df.columns
                ]

                if not valid_columns:
                    st.error("No valid columns selected for transformation")
                    return

                st.info(f"Transforming {len(valid_columns)} columns")

                for col in valid_columns:
                    try:
                        if col not in column_types:
                            st.warning(
                                f"No type specified for column '{col}'. Skipping."
                            )
                            continue

                        col_type = column_types[col]
                        if col_type == 'Continuous':
                            transformed_col = transformer.transform_continuous(
                                train_df[col],
                                transformations.get(col, 'standard'))
                            transformed_data[col] = transformed_col
                        elif col_type == 'Categorical':
                            transformed_col = transformer.transform_categorical(
                                train_df[col],
                                transformations.get(col, 'label'))
                            transformed_data[col] = transformed_col
                        elif col_type == 'Ordinal':  # Handle ordinal as continuous
                            transformed_col = transformer.transform_continuous(
                                train_df[col],
                                transformations.get(col, 'standard'))
                            transformed_data[col] = transformed_col
                        elif col_type == 'Datetime':
                            dt_features = transformer.transform_datetime(
                                train_df[col])
                            transformed_data = pd.concat(
                                [transformed_data, dt_features], axis=1)
                    except Exception as e:
                        st.error(f"Error transforming column {col}: {str(e)}")
                        st.exception(e)  # This will show the full traceback
                        return

                # If no columns were selected or all transformations failed, display an error
                if transformed_data.empty:
                    st.error(
                        "No columns were successfully transformed. Please check your data and configuration."
                    )
                    return

                st.info(
                    f"Successfully transformed {len(transformed_data.columns)} columns"
                )

                if use_modal:
                    try:
                        with st.spinner("Training model on Modal cloud..."):
                            # Identify categorical columns for special handling
                            categorical_indices = []
                            categorical_dims = {}
                            
                            for i, col in enumerate(valid_columns):
                                if col in column_types and column_types[col] == 'Categorical':
                                    categorical_indices.append(i)
                                    # Count unique values in the original column
                                    unique_values = train_df[col].nunique()
                                    categorical_dims[i] = unique_values
                            
                            # Show categorical information
                            if categorical_indices:
                                st.info(f"Identified {len(categorical_indices)} categorical columns for specialized handling")
                                
                            # Train on Modal with model persistence
                            losses = modal_gan.train(
                                transformed_data,
                                input_dim=transformed_data.shape[1],
                                hidden_dim=model_config['hidden_dim'],
                                epochs=model_config['epochs'],
                                batch_size=model_config['batch_size'],
                                model_type=model_config['model_type'],
                                load_existing=model_config.get('load_existing', False),
                                model_name=model_config.get('model_name', None),
                                fine_tune=model_config.get('fine_tune', False),
                                categorical_columns=categorical_indices if categorical_indices else None,
                                categorical_dims=categorical_dims if categorical_dims else None
                            )

                            # Generate samples using Modal with temperature control
                            synthetic_data = modal_gan.generate(
                                num_samples=len(df),
                                input_dim=transformed_data.shape[1],
                                hidden_dim=model_config['hidden_dim'],
                                model_type=model_config['model_type'],
                                model_name=model_config.get('model_name', None),
                                temperature=model_config.get('temperature', 0.8),
                                categorical_columns=categorical_indices if categorical_indices else None,
                                categorical_dims=categorical_dims if categorical_dims else None
                            )
                    except Exception as e:
                        st.error(f"Modal training failed: {str(e)}")
                        st.info("Falling back to local training...")
                        use_modal = False

                if not use_modal:
                    # Local training fallback
                    # Calculate appropriate batch size
                    num_samples = len(transformed_data)
                    if model_config['batch_size'] >= num_samples:
                        adjusted_batch_size = max(
                            32, num_samples // 4)  # Ensure at least 4 batches
                        st.warning(
                            f"Batch size ({model_config['batch_size']}) is larger than dataset size ({num_samples}). Adjusted to {adjusted_batch_size}"
                        )
                        model_config['batch_size'] = adjusted_batch_size

                    train_data = torch.FloatTensor(transformed_data.values)
                    train_loader = torch.utils.data.DataLoader(
                        train_data,
                        batch_size=model_config['batch_size'],
                        shuffle=True,
                        drop_last=True  # Drop last batch if incomplete
                    )

                    device = torch.device(
                        'cuda' if torch.cuda.is_available() else 'cpu')

                    # Initialize selected model
                    # Load packages when needed
                    torch, plt, wandb = load_ml_packages()
                    
                    # Load selected model
                    ModelClass = load_selected_model(model_config['model_type'])
                    if model_config['model_type'] == 'WGAN':
                        gan = ModelClass(input_dim=transformed_data.shape[1],
                                   hidden_dim=model_config['hidden_dim'],
                                   clip_value=model_config['clip_value'],
                                   n_critic=model_config['n_critic'],
                                   lr_g=model_config['lr_g'],
                                   lr_d=model_config['lr_d'],
                                   device=device,
                                   use_wandb=True)

                        # Run Bayesian optimization if requested
                        if model_config.get('use_bayesian_opt', False):
                            with st.spinner(
                                    "Running Bayesian hyperparameter optimization..."
                            ):
                                best_params, history = gan.optimize_hyperparameters(
                                    train_loader,
                                    n_epochs=model_config['bayes_epochs'],
                                    n_iterations=model_config[
                                        'bayes_iterations'])

                                # Display optimization results
                                st.success(
                                    "Hyperparameter optimization completed!")
                                st.write("Best Parameters:")
                                for param, value in best_params.items():
                                    st.write(f"- {param}: {value:.6f}")

                                # Show optimization history
                                st.subheader("Optimization History")
                                st.dataframe(history)
                    elif model_config['model_type'] == 'CGAN':
                        # For CGAN, we need to identify a condition column
                        if 'condition_column' in model_config and model_config[
                                'condition_column'] in df.columns:
                            condition_col = model_config['condition_column']
                            # Extract condition data
                            condition_data = transformed_data[
                                condition_col].values.reshape(-1, 1)
                            condition_dim = 1
                            main_data = transformed_data.drop(
                                columns=[condition_col])

                            # Store specific condition values for generation
                            if 'condition_values' in model_config and model_config[
                                    'condition_values']:
                                st.session_state[
                                    'condition_values'] = model_config[
                                        'condition_values']
                                st.session_state[
                                    'condition_ratios'] = model_config[
                                        'condition_ratios']
                                st.session_state[
                                    'condition_encoder'] = transformer.encoders.get(
                                        condition_col)
                                st.session_state[
                                    'condition_col'] = condition_col

                            gan = CGAN(input_dim=main_data.shape[1],
                                       condition_dim=condition_dim,
                                       hidden_dim=model_config['hidden_dim'],
                                       device=device)
                        else:
                            # Default to using first column as condition if not specified
                            condition_col = df.columns[0]
                            condition_data = transformed_data[
                                condition_col].values.reshape(-1, 1)
                            condition_dim = 1
                            main_data = transformed_data.drop(
                                columns=[condition_col])

                            gan = CGAN(input_dim=main_data.shape[1],
                                       condition_dim=condition_dim,
                                       hidden_dim=model_config['hidden_dim'],
                                       device=device)
                    elif model_config['model_type'] == 'TVAE':
                        gan = TVAE(input_dim=transformed_data.shape[1],
                                   hidden_dim=model_config['hidden_dim'],
                                   latent_dim=model_config['latent_dim'],
                                   device=device)
                    elif model_config[
                            'model_type'] == 'CTGAN':  # Added CTGAN handling
                        gan = CTGAN(input_dim=transformed_data.shape[1],
                                    hidden_dim=model_config['hidden_dim'],
                                    num_residual_blocks=model_config[
                                        'num_residual_blocks'],
                                    device=device)
                    else:  # TableGAN
                        gan = TableGAN(input_dim=transformed_data.shape[1],
                                       hidden_dim=model_config['hidden_dim'],
                                       device=device)

                    st.session_state.total_epochs = model_config['epochs']
                    # Train the model
                    with st.spinner(
                            f"Training {model_config['model_type']} model for {model_config['epochs']} epochs..."
                    ):
                        # Setup progress bar
                        progress_bar = st.progress(0)
                        epoch_status = st.empty()

                        # Train loop
                        for epoch in range(model_config['epochs']):
                            epoch_metrics = {}
                            for i, batch_data in enumerate(train_loader):
                                metrics = gan.train_step(batch_data)
                                epoch_metrics.update(metrics)

                            # Update progress
                            progress = (epoch + 1) / model_config['epochs']
                            progress_bar.progress(progress)
                            # Access appropriate metric names based on model output
                            d_loss = epoch_metrics.get(
                                'discriminator_loss',
                                epoch_metrics.get('disc_loss', 0))
                            g_loss = epoch_metrics.get(
                                'generator_loss',
                                epoch_metrics.get('gen_loss', 0))
                            epoch_status.text(
                                f"Epoch {epoch+1}/{model_config['epochs']} - Disc Loss: {d_loss:.4f}, Gen Loss: {g_loss:.4f}"
                            )

                        # Save the trained model
                        model_dir = "models"
                        os.makedirs(model_dir, exist_ok=True)
                        torch.save(
                            gan.state_dict(),
                            os.path.join(
                                model_dir,
                                f"{model_config['model_type'].lower()}_model.pt"
                            ))

                        # Finish wandb run if the model has a finish_wandb method
                        if hasattr(gan, 'finish_wandb'):
                            gan.finish_wandb()

                    # Determine number of samples to generate
                    original_size = len(df)

                    # Calculate number of rows to generate based on user settings
                    if model_config.get('use_default_row_count', True):
                        # Default: 25% of original data
                        num_rows_to_generate = max(1,
                                                   int(original_size * 0.25))
                    else:
                        if model_config.get('use_exact_row_count', False):
                            # Use exact count specified by user, but cap at original size
                            num_rows_to_generate = min(
                                model_config.get('exact_row_count', 100),
                                original_size)
                        else:
                            # Use percentage of original data
                            percentage = model_config.get(
                                'row_percentage', 25) / 100
                            num_rows_to_generate = max(
                                1, int(original_size * percentage))

                    # Display info about generation size
                    st.info(
                        f"Generating {num_rows_to_generate} rows of synthetic data ({(num_rows_to_generate/original_size*100):.1f}% of original size)"
                    )

                    if model_config[
                            'model_type'] == 'CGAN' and 'condition_values' in model_config and model_config[
                                'condition_values']:
                        # Generate data based on selected condition values with their proportions
                        condition_values = model_config['condition_values']
                        condition_ratios = model_config['condition_ratios']
                        encoder = transformer.encoders.get(
                            model_config['condition_column'])

                        total_samples = num_rows_to_generate  # Use the calculated number of rows
                        synthetic_data_list = []

                        for value in condition_values:
                            # Calculate how many samples to generate for this value
                            num_samples = int(condition_ratios[value] *
                                              total_samples)
                            if num_samples < 1:
                                num_samples = 1

                            # Encode the condition value
                            encoded_value = encoder.transform([value])[0]
                            condition_tensor = torch.full(
                                (num_samples, 1),
                                encoded_value,
                                dtype=torch.float).to(device)

                            # Generate samples for this condition
                            value_samples = gan.generate_samples(
                                num_samples, condition_tensor).cpu().numpy()

                            # Find the correct index for the condition column
                            condition_idx = -1  # Default to last column

                            # If we have transformed_columns available, find the index of the condition column
                            if model_config[
                                    'condition_column'] in transformed_data.columns:
                                condition_idx = list(
                                    transformed_data.columns).index(
                                        model_config['condition_column'])

                            # Ensure the condition column has the correct encoded value
                            value_samples[:, condition_idx] = encoded_value

                            # Store the generated samples along with their expected condition value
                            # for verification during inverse transform
                            synthetic_data_list.append(
                                (value_samples, value, condition_idx))

                        # Combine all generated samples, handling the new structure
                        all_samples = []
                        condition_values_map = {}

                        # Extract samples and build the condition mapping
                        for samples, value, idx in synthetic_data_list:
                            all_samples.append(samples)
                            # For each row in these samples, remember its condition value
                            for i in range(samples.shape[0]):
                                # Add to map: sample_index -> (condition_value, column_index)
                                condition_values_map[len(all_samples) - 1,
                                                     i] = (value, idx)

                        # Combine the samples
                        synthetic_data = np.vstack(all_samples)

                        # Create a map to preserve condition values
                        condition_map = {}
                        for batch_idx, samples in enumerate(all_samples):
                            for sample_idx in range(samples.shape[0]):
                                # Store the original condition value for this sample
                                if (batch_idx,
                                        sample_idx) in condition_values_map:
                                    value, col_idx = condition_values_map[(
                                        batch_idx, sample_idx)]
                                    condition_map[batch_idx * samples.shape[0]
                                                  + sample_idx] = (value,
                                                                   col_idx)

                        # If we didn't generate exactly the requested number of samples, adjust
                        if len(synthetic_data) != total_samples:
                            # Create a smaller set of samples if we generated too many
                            if len(synthetic_data) > total_samples:
                                indices = np.random.choice(len(synthetic_data),
                                                           total_samples,
                                                           replace=False)
                                synthetic_data = synthetic_data[indices]

                                # Update condition map for the new indices
                                new_condition_map = {}
                                for i, old_idx in enumerate(indices):
                                    if old_idx in condition_map:
                                        new_condition_map[i] = condition_map[
                                            old_idx]
                                condition_map = new_condition_map
                            # Or repeat samples if we didn't generate enough
                            else:
                                indices = np.random.choice(len(synthetic_data),
                                                           total_samples,
                                                           replace=True)
                                synthetic_data = synthetic_data[indices]

                                # Update condition map for the repeated indices
                                new_condition_map = {}
                                for i, old_idx in enumerate(indices):
                                    if old_idx in condition_map:
                                        new_condition_map[i] = condition_map[
                                            old_idx]
                                condition_map = new_condition_map
                    else:
                        # Use the calculated number of rows to generate for non-CGAN models
                        synthetic_data = gan.generate_samples(
                            num_rows_to_generate).cpu().numpy()

                # Inverse transform
                result_df = pd.DataFrame()
                col_idx = 0
                transformed_columns = []

                # Keep track of the condition column index for CGAN if used
                condition_col_idx = -1
                if model_config[
                        'model_type'] == 'CGAN' and 'condition_column' in model_config:
                    # Find the index of the condition column in the selected columns
                    if model_config['condition_column'] in selected_columns:
                        condition_col_idx = selected_columns.index(
                            model_config['condition_column'])

                for col in selected_columns:  # Use only selected columns
                    # Skip if column type isn't defined
                    if col not in column_types:
                        continue

                    col_type = column_types[col]
                    try:
                        # Special handling for condition column in CGAN
                        if model_config[
                                'model_type'] == 'CGAN' and col == model_config[
                                    'condition_column']:
                            # Create a Series for storing the condition values
                            condition_series = pd.Series(index=range(
                                len(synthetic_data)),
                                                         name=col)

                            # If using single-value mode, set all to that value
                            if 'condition_values' in model_config and len(
                                    model_config['condition_values']) == 1:
                                condition_series[:] = model_config[
                                    'condition_values'][0]
                            else:
                                # Fill with values from our condition map where available
                                for sample_idx in range(len(synthetic_data)):
                                    if sample_idx in condition_map:
                                        original_value, _ = condition_map[
                                            sample_idx]
                                        condition_series.iloc[
                                            sample_idx] = original_value
                                    else:
                                        # If no mapping (could happen after resampling), decode normally
                                        pass

                                # Fill missing values by inverse transforming
                                missing_indices = condition_series.isna()
                                if missing_indices.any():
                                    condition_series[
                                        missing_indices] = transformer.inverse_transform_categorical(
                                            pd.Series(
                                                synthetic_data[missing_indices,
                                                               col_idx],
                                                name=col))

                            # Set the result
                            result_df[col] = condition_series
                            transformed_columns.append(col)
                            col_idx += 1
                        elif col_type in ['Continuous', 'Ordinal']:
                            result_df[
                                col] = transformer.inverse_transform_continuous(
                                    pd.Series(synthetic_data[:, col_idx],
                                              name=col))
                            transformed_columns.append(col)
                            col_idx += 1
                        elif col_type == 'Categorical':
                            result_df[
                                col] = transformer.inverse_transform_categorical(
                                    pd.Series(synthetic_data[:, col_idx],
                                              name=col))
                            transformed_columns.append(col)
                            col_idx += 1
                        elif col_type == 'Datetime':
                            # Handle datetime reconstruction with error checking
                            if col_idx + 2 < synthetic_data.shape[
                                    1]:  # Make sure we have enough columns
                                year = synthetic_data[:, col_idx]
                                month = synthetic_data[:, col_idx + 1]
                                day = synthetic_data[:, col_idx + 2]
                                try:
                                    result_df[col] = pd.to_datetime(
                                        dict(year=year, month=month, day=day),
                                        errors=
                                        'coerce'  # Convert invalid dates to NaT
                                    )
                                    transformed_columns.append(col)
                                except:
                                    result_df[col] = pd.NaT
                                col_idx += 4
                            else:
                                result_df[col] = pd.NaT
                    except Exception as e:
                        st.warning(
                            f"Error transforming column {col}: {str(e)}")
                        result_df[col] = None

                # Add excluded columns back with empty/NaN values if they were in the original data
                for col in original_columns:
                    if col not in transformed_columns:
                        # Use appropriate NaN type based on column type
                        if col in column_types and column_types[
                                col] == 'Datetime':
                            result_df[col] = pd.NaT
                        else:
                            result_df[col] = None

                # Ensure all columns from original data are present in result
                for col in original_columns:
                    if col not in result_df.columns:
                        result_df[col] = None

                # Evaluate synthetic data
                st.subheader("Data Quality Evaluation")

                # Display results with progress tracking
                with st.spinner("Processing evaluation results..."):
                    # Filter DataFrames to only include selected columns
                    full_real_df = df[selected_columns].copy()
                    eval_synthetic_df = result_df[selected_columns].copy()

                    # Sample the real data to match the synthetic data size for fair comparison
                    synthetic_size = len(eval_synthetic_df)
                    if synthetic_size < len(full_real_df):
                        # Sample without replacement if we have enough real data
                        eval_real_df = full_real_df.sample(n=synthetic_size,
                                                           random_state=42)
                        st.info(
                            f"Using a sample of {synthetic_size} rows from the real dataset for fair comparison"
                        )
                    else:
                        # Use all real data if synthetic data is larger
                        eval_real_df = full_real_df.copy()

                    st.write("Real data shape (for comparison):",
                             eval_real_df.shape)
                    st.write("Synthetic data shape:", eval_synthetic_df.shape)
                    st.write("Original real data shape:", full_real_df.shape)

                    evaluator = DataEvaluator(eval_real_df, eval_synthetic_df)

                    # Run comprehensive evaluation
                    all_metrics = evaluator.evaluate_all(
                        target_column=target_col, task_type=task_type)

                    # ML utility evaluation
                    with st.expander("ML Utility Evaluation (TSTR)"):
                        if 'ml_utility' in all_metrics and isinstance(
                                all_metrics['ml_utility'], dict):
                            ml_metrics = all_metrics['ml_utility']
                            st.write(
                                "Train-Synthetic-Test-Real (TSTR) Evaluation:")
                            for metric, value in ml_metrics.items():
                                if isinstance(value, (int, float)):
                                    st.write(f"{metric}: {value:.4f}")
                                else:
                                    st.write(f"{metric}: {value}")
                        else:
                            st.write("ML utility metrics not available.")
                    
                    # AI-powered Quality Score
                    if 'quality_score_details' in all_metrics and isinstance(all_metrics['quality_score_details'], dict):
                        components.display_quality_score(all_metrics['quality_score_details'])
                    else:
                        st.error("AI Quality Score data is not available.")

                    # Advanced Evaluation Metrics
                    with st.expander("Advanced Evaluation Metrics"):
                        # Create 2 columns for better organization
                        col1, col2 = st.columns(2)

                        # Column 1: Correlation and Similarity Metrics
                        with col1:
                            st.subheader("Correlation Metrics")

                            # Correlation similarity
                            corr_sim = all_metrics.get(
                                'correlation_similarity', 0)
                            if isinstance(corr_sim, (int, float)):
                                st.metric("Correlation Similarity",
                                          f"{corr_sim:.4f}")

                            # Correlation distance metrics
                            corr_rmse = all_metrics.get(
                                'correlation_distance_rmse', 0)
                            corr_mae = all_metrics.get(
                                'correlation_distance_mae', 0)
                            if isinstance(corr_rmse,
                                          (int, float)) and isinstance(
                                              corr_mae, (int, float)):
                                st.metric("Column Correlation RMSE",
                                          f"{corr_rmse:.4f}")
                                st.metric("Column Correlation MAE",
                                          f"{corr_mae:.4f}")

                            # Overall similarity score
                            similarity_score = all_metrics.get(
                                'similarity_score', 0)
                            if isinstance(similarity_score, (int, float)):
                                st.metric("Overall Similarity Score",
                                          f"{similarity_score:.4f}",
                                          delta=None,
                                          delta_color="normal")

                        # Column 2: Nearest Neighbor Metrics
                        with col2:
                            st.subheader("Privacy Metrics")

                            # Nearest neighbor stats
                            nn_mean = all_metrics.get('nearest_neighbor_mean',
                                                      0)
                            nn_std = all_metrics.get('nearest_neighbor_std', 0)
                            if isinstance(nn_mean,
                                          (int, float)) and isinstance(
                                              nn_std, (int, float)):
                                st.metric("Nearest Neighbor Mean",
                                          f"{nn_mean:.4f}")
                                st.metric("Nearest Neighbor Std",
                                          f"{nn_std:.4f}")
                                st.info(
                                    "Lower nearest neighbor distance may indicate potential privacy concerns."
                                )



                # Display final results
                st.success("Synthetic data generated successfully!")
                st.subheader("Generated Data Preview")
                st.dataframe(result_df.head())

                # Download button
                csv = result_df.to_csv(index=False)
                st.download_button(label="Download Synthetic Data",
                                   data=csv,
                                   file_name="synthetic_data.csv",
                                   mime="text/csv")

            except Exception as e:
                st.error(f"An error occurred during data generation: {str(e)}")
                return


def model_config_section():
    st.subheader("Model Configuration")

    model_config = {}

    model_config['model_type'] = st.selectbox(
        "Select Model Type",
        options=['TableGAN', 'WGAN', 'CGAN', 'TVAE', 'CTGAN'],  # Added CTGAN
        help="Choose the type of model to use for synthetic data generation")
        
    # Model persistence options
    st.subheader("Model Persistence")
    
    # Add option for loading existing model for fine-tuning
    model_config['load_existing'] = st.checkbox(
        "Load existing model for fine-tuning",
        value=False,
        help="When checked, an existing model will be loaded for fine-tuning"
    )
    
    if model_config['load_existing']:
        # Add a dedicated section for model management
        st.subheader("Model Management")
        
        # Option to list available models
        col1, col2 = st.columns([1, 1])
        with col1:
            list_models_button = st.button("List Available Models")
        with col2:
            force_local_only = st.checkbox("List local models only", 
                              value=False, 
                              help="Check this if Modal cloud connection is not working")
        
        if list_models_button:
            with st.spinner("Retrieving available models..."):
                try:
                    # Create an expander for debug information
                    with st.expander("Model Listing Debug Info"):
                        # Add a placeholder for debug messages
                        debug_placeholder = st.empty()
                    
                    # Redirect print statements to the debug placeholder
                    import io
                    import sys
                    old_stdout = sys.stdout
                    new_stdout = io.StringIO()
                    sys.stdout = new_stdout
                    
                    # Call the list_available_models function
                    if force_local_only:
                        # Force local-only mode by triggering an exception in Modal code
                        try:
                            with modal.app.run():
                                raise Exception("Forcing local-only model listing")
                        except:
                            pass
                        
                    available_models = modal_gan.list_available_models()
                    
                    # Restore stdout and update debug info
                    sys.stdout = old_stdout
                    debug_placeholder.code(new_stdout.getvalue())
                    
                    if available_models:
                        st.success(f"Found {len(available_models)} saved models")
                        # Display models in a table with improved formatting
                        model_df = pd.DataFrame(available_models)
                        
                        # Reorder columns for better display
                        display_cols = ['filename', 'model_type', 'location', 'size_mb', 'last_modified']
                        display_cols = [col for col in display_cols if col in model_df.columns]
                        model_df = model_df[display_cols]
                        
                        # Format size to 2 decimal places
                        if 'size_mb' in model_df.columns:
                            model_df['size_mb'] = model_df['size_mb'].round(2).astype(str) + ' MB'
                        
                        st.dataframe(model_df, use_container_width=True)
                    else:
                        st.info("No saved models found. Train a model first.")
                        st.markdown("""
                        **Possible reasons:**
                        1. You haven't trained any models yet
                        2. Modal cloud storage connection issue
                        3. Local storage not properly initialized
                        """)
                except Exception as e:
                    import traceback
                    st.error(f"Failed to list models: {str(e)}")
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())
    
        # Input for model name
        model_config['model_name'] = st.text_input(
            "Model name to load/save",
            value=f"{model_config['model_type'].lower()}_model",
            help="Name of the model to load for fine-tuning or save after training (without file extension)"
        )
        
        # Checkbox for fine-tuning vs just loading
        model_config['fine_tune'] = st.checkbox(
            "Fine-tune loaded model",
            value=True,
            help="When checked, the loaded model will be fine-tuned; otherwise, it will only be used for generation"
        )
    else:
        # Default model name based on type
        model_config['model_name'] = f"{model_config['model_type'].lower()}_model"
        model_config['fine_tune'] = False
    
    # Add option for temperature parameter (controls diversity in categorical data)
    model_config['temperature'] = st.slider(
        "Temperature for categorical sampling",
        min_value=0.1,
        max_value=2.0,
        value=0.8,
        step=0.1,
        help="Higher values produce more diverse categorical outputs but may be less accurate"
    )
    
    # Add option to specify number of rows to generate
    st.subheader("Data Generation Settings")

    model_config['use_default_row_count'] = st.checkbox(
        "Use default row count (25% of original data)",
        value=False,
        help=
        "When checked, generates synthetic data with 25% of the original dataset size"
    )

    if not model_config['use_default_row_count']:
        model_config['row_percentage'] = st.slider(
            "Percentage of original data size to generate",
            min_value=5,
            max_value=100,
            value=25,
            step=5,
            help=
            "Percentage of the original dataset size to generate as synthetic data"
        )

        st.info(
            "You can also specify an exact number of rows to generate below")

        model_config['use_exact_row_count'] = st.checkbox(
            "Use exact row count instead of percentage",
            value=False,
            help=
            "When checked, allows you to specify the exact number of rows to generate"
        )

        if model_config['use_exact_row_count']:
            model_config['exact_row_count'] = st.number_input(
                "Number of rows to generate",
                min_value=1,
                max_value=
                10000,  # We'll cap this with the actual data size later
                value=100,
                step=10,
                help="The exact number of synthetic data rows to generate")

    # Add CTGAN-specific parameters
    if model_config['model_type'] == 'CTGAN':
        model_config['num_residual_blocks'] = st.slider(
            "Number of Residual Blocks",
            min_value=3,
            max_value=10,
            value=5,
            step=1,
            help="Number of residual blocks in CTGAN generator")

    # Add latent dimension parameter for TVAE
    if model_config['model_type'] == 'TVAE':
        model_config['latent_dim'] = st.slider(
            "Latent Dimension",
            min_value=32,
            max_value=256,
            value=128,
            step=32,
            help="Dimension of the latent space for TVAE")

    model_config['hidden_dim'] = st.slider(
        "Hidden Dimension",
        min_value=32,
        max_value=512,
        value=128,
        step=64,
        help="Size of hidden layers in the model")

    model_config['epochs'] = st.slider("Number of Epochs",
                                       min_value=10,
                                       max_value=5000,
                                       value=500,
                                       step=100,
                                       help="Number of training epochs")

    # Add a note about batch size requirements
    st.info(
        "Note: Batch size will be automatically adjusted if it exceeds dataset size."
    )

    model_config['batch_size'] = st.slider(
        "Batch Size",
        min_value=8,
        max_value=750,
        value=256,
        step=32,
        help=
        "Number of samples per training batch. Will be adjusted if larger than dataset size."
    )

    # Bayesian Optimization UI controls for all models
    model_config['use_bayesian_opt'] = st.checkbox(
        "Use Bayesian Hyperparameter Optimization",
        value=False,
        help=
        "Automatically optimize hyperparameters using Bayesian optimization")

    if model_config['use_bayesian_opt']:
        col1, col2 = st.columns(2)
        with col1:
            model_config['bayes_iterations'] = st.slider(
                "Optimization Iterations",
                min_value=5,
                max_value=50,
                value=10,
                step=5,
                help="Number of iterations for Bayesian optimization")
        with col2:
            model_config['bayes_epochs'] = st.slider(
                "Epochs Per Iteration",
                min_value=5,
                max_value=100,
                value=20,
                step=10,
                help="Number of epochs to train for each optimization iteration"
            )

    if model_config['model_type'] == 'WGAN':
        model_config['clip_value'] = st.slider(
            "Clip Value",
            min_value=0.01,
            max_value=0.1,
            value=0.01,
            step=0.01,
            help="Weight clipping value for WGAN")
        model_config['n_critic'] = st.slider(
            "Critic Updates",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            help="Number of critic updates per generator update")
        col1, col2 = st.columns(2)
        with col1:
            model_config['lr_g'] = st.slider(
                "Generator Learning Rate",
                min_value=0.00001,
                max_value=0.001,
                value=0.0001,
                format="%.5f",
                help="Learning rate for the generator")
        with col2:
            model_config['lr_d'] = st.slider(
                "Discriminator Learning Rate",
                min_value=0.00001,
                max_value=0.001,
                value=0.0001,
                format="%.5f",
                help="Learning rate for the discriminator/critic")

    return model_config


if __name__ == "__main__":
    main()
