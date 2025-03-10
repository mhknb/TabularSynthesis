import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import streamlit as st
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # Added matplotlib import
from src.data_processing.data_loader import DataLoader
from src.data_processing.transformers import DataTransformer
from src.models.table_gan import TableGAN
from src.models.modal_gan import ModalGAN
from src.models.wgan import WGAN
from src.models.cgan import CGAN
from src.models.tvae import TVAE
from src.utils.validation import validate_data, check_column_types
from src.utils.evaluation import DataEvaluator
from src.ui import components
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Synthetic Data Generator", layout="wide")

# Initialize Modal resources
modal_gan = ModalGAN()

def plot_training_losses():
    """Plot training losses"""
    if 'training_losses' not in st.session_state:
        return None

    losses = st.session_state.training_losses
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(losses['epochs'], losses['generator'], label='Generator Loss', color='blue')
    ax.plot(losses['epochs'], losses['discriminator'], label='Discriminator/Critic Loss', color='orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Losses')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig

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
            options=["Impute with mean/mode", "Drop rows with missing values", "Keep as is"],
            index=0,
            help="Choose how to handle missing data before training the model"
        )

        if missing_handling == "Drop rows with missing values":
            original_count = len(df)
            df = df.dropna()
            st.info(f"Dropped {original_count - len(df)} rows with missing values. {len(df)} rows remaining.")
        elif missing_handling == "Impute with mean/mode":
            st.info("Missing values will be imputed during the transformation process.")
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
    model_config = model_config_section()  # Use the newly defined function directly

    # Add condition column selector for CGAN
    if model_config['model_type'] == 'CGAN':
        if 'uploaded_df' in st.session_state:
            model_config['condition_column'] = st.selectbox(
                "Select condition column for CGAN",
                options=st.session_state['uploaded_df'].columns.tolist(),
                help="This column will be used as a condition for generating data"
            )

            # Now add selector for specific condition values to generate
            if 'condition_column' in model_config and model_config['condition_column']:
                unique_values = st.session_state['uploaded_df'][model_config['condition_column']].unique().tolist()
                model_config['condition_values'] = st.multiselect(
                    f"Select specific values from '{model_config['condition_column']}' to generate",
                    options=unique_values,
                    default=unique_values[:min(3, len(unique_values))],
                    help="CGAN will generate data only for these selected condition values"
                )

                # Add ratio selector for each selected condition value
                if model_config['condition_values']:
                    st.write("Set the proportion of samples to generate for each condition value:")
                    condition_ratios = {}
                    total_ratio = 0

                    cols = st.columns(min(3, len(model_config['condition_values'])))
                    for i, value in enumerate(model_config['condition_values']):
                        col_idx = i % len(cols)
                        with cols[col_idx]:
                            ratio = st.slider(
                                f"Ratio for '{value}'",
                                min_value=1,
                                max_value=10,
                                value=10 // len(model_config['condition_values']),
                                help="Relative proportion of samples with this condition"
                            )
                            condition_ratios[value] = ratio
                            total_ratio += ratio

                    # Normalize ratios
                    model_config['condition_ratios'] = {k: v/total_ratio for k, v in condition_ratios.items()}
        else:
            st.warning("Please upload data first to select a condition column for CGAN")

    # Add Modal training option
    use_modal = st.checkbox("Use Modal for cloud training (faster)", value=True)

    # Column selection for synthetic data generation
    st.subheader("Column Selection")

    col1, col2 = st.columns(2)

    with col1:
        selection_mode = st.radio(
            "Column Selection Mode",
            options=["Use all columns", "Choose columns to include", "Choose columns to exclude"],
            index=0,
            help="Select which columns to use for synthetic data generation"
        )

    selected_columns = original_columns.copy()

    if selection_mode == "Choose columns to include":
        with col2:
            selected_columns = st.multiselect(
                "Select columns to include",
                options=original_columns,
                default=original_columns,
                help="Only these columns will be used for synthetic data generation"
            )
            if not selected_columns:
                st.warning("You must select at least one column")
    elif selection_mode == "Choose columns to exclude":
        with col2:
            excluded_columns = st.multiselect(
                "Select columns to exclude",
                options=original_columns,
                default=[],
                help="These columns will be excluded from synthetic data generation"
            )
            selected_columns = [col for col in original_columns if col not in excluded_columns]
            if not selected_columns:
                st.warning("You cannot exclude all columns")


    # Target column selection for ML evaluation
    st.subheader("ML Evaluation Settings")

    target_col = st.selectbox(
        "Select target column for ML utility evaluation",
        options=df.columns.tolist(),
        help="This column will be used to evaluate how well the synthetic data preserves predictive relationships"
    )

    task_type = st.selectbox(
        "Select task type",
        options=['classification', 'regression'],
        help="Choose 'classification' for categorical targets, 'regression' for continuous targets"
    )

    if st.button("Generate Synthetic Data"):
        with st.spinner("Preparing data..."):
            try:
                # Split data into train and test sets
                train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
                st.info(f"Data split into {len(train_df)} training samples and {len(test_df)} test samples")

                # Transform data
                transformer = DataTransformer()
                transformed_data = pd.DataFrame()

                for col in selected_columns:  # Use only selected columns
                    try:
                        col_type = column_types[col]
                        if col_type == 'Continuous':
                            transformed_col = transformer.transform_continuous(
                                train_df[col],
                                transformations.get(col, 'standard')
                            )
                            transformed_data[col] = transformed_col
                        elif col_type == 'Categorical':
                            transformed_col = transformer.transform_categorical(
                                train_df[col],
                                transformations.get(col, 'label')
                            )
                            transformed_data[col] = transformed_col
                        elif col_type == 'Ordinal':  # Handle ordinal as continuous
                            transformed_col = transformer.transform_continuous(
                                train_df[col],
                                transformations.get(col, 'standard')
                            )
                            transformed_data[col] = transformed_col
                        elif col_type == 'Datetime':
                            dt_features = transformer.transform_datetime(train_df[col])
                            transformed_data = pd.concat([transformed_data, dt_features], axis=1)
                    except Exception as e:
                        st.error(f"Error transforming column {col}: {str(e)}")
                        return

                # If no columns were selected or all transformations failed, display an error
                if transformed_data.empty:
                    st.error("No columns were successfully transformed. Please check your data and configuration.")
                    return

                st.info(f"Successfully transformed {len(transformed_data.columns)} columns")


                if use_modal:
                    try:
                        with st.spinner("Training model on Modal cloud..."):
                            # Initialize training progress tracking
                            st.session_state.total_epochs = model_config['epochs']
                            for key in ['progress_bar', 'status_text', 'loss_chart', 'training_losses']:
                                if key in st.session_state:
                                    del st.session_state[key]

                            # Initialize progress display elements
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            loss_chart = st.empty()

                            # Initialize loss storage
                            st.session_state.training_losses = {'epochs': [], 'generator': [], 'discriminator': []}

                            # Create a placeholder for the loss plot
                            loss_plot = st.empty()

                            # Train on Modal and process logs in real-time
                            all_losses = modal_gan.train(
                                transformed_data,
                                input_dim=transformed_data.shape[1],
                                hidden_dim=model_config['hidden_dim'],
                                epochs=model_config['epochs'],
                                batch_size=model_config['batch_size']
                            )

                            # Process each epoch's losses
                            for epoch, loss_dict in all_losses:
                                # Update progress bar
                                progress = (epoch + 1) / model_config['epochs']
                                progress_bar.progress(progress)

                                # Update loss tracking
                                st.session_state.training_losses['epochs'].append(epoch)
                                st.session_state.training_losses['generator'].append(loss_dict['generator_loss'])
                                st.session_state.training_losses['discriminator'].append(loss_dict['discriminator_loss'])

                                # Update status text
                                status_text.text(
                                    f"Epoch {epoch + 1}/{model_config['epochs']}: "
                                    f"Generator Loss: {loss_dict['generator_loss']:.4f}, "
                                    f"Discriminator Loss: {loss_dict['discriminator_loss']:.4f}"
                                )

                                # Update loss plot
                                fig, ax = plt.subplots(figsize=(10, 4))
                                ax.plot(st.session_state.training_losses['epochs'],
                                      st.session_state.training_losses['generator'],
                                      label='Generator Loss', color='blue')
                                ax.plot(st.session_state.training_losses['epochs'],
                                      st.session_state.training_losses['discriminator'],
                                      label='Discriminator/Critic Loss', color='orange')
                                ax.set_xlabel('Epoch')
                                ax.set_ylabel('Loss')
                                ax.set_title('Training Losses')
                                ax.legend()
                                ax.grid(True, alpha=0.3)
                                loss_plot.pyplot(fig)
                                plt.close(fig)

                            # Generate samples using Modal
                            synthetic_data = modal_gan.generate(
                                num_samples=len(df),
                                input_dim=transformed_data.shape[1],
                                hidden_dim=model_config['hidden_dim']
                            )
                    except Exception as e:
                        st.error(f"Modal training failed: {str(e)}")
                        st.info("Falling back to local training...")
                        use_modal = False

                if not use_modal:
                    # Local training fallback
                    train_data = torch.FloatTensor(transformed_data.values)
                    train_loader = torch.utils.data.DataLoader(
                        train_data, 
                        batch_size=model_config['batch_size'],
                        shuffle=True
                    )

                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                    # Initialize selected model
                    if model_config['model_type'] == 'WGAN':
                        gan = WGAN(
                            input_dim=transformed_data.shape[1],
                            hidden_dim=model_config['hidden_dim'],
                            clip_value=model_config['clip_value'],
                            n_critic=model_config['n_critic'],
                            device=device
                        )
                    elif model_config['model_type'] == 'CGAN':
                        # For CGAN, we need to identify a condition column
                        if 'condition_column' in model_config and model_config['condition_column'] in df.columns:
                            condition_col = model_config['condition_column']
                            # Extract condition data
                            condition_data = transformed_data[condition_col].values.reshape(-1, 1)
                            condition_dim = 1
                            main_data = transformed_data.drop(columns=[condition_col])

                            # Store specific condition values for generation
                            if 'condition_values' in model_config and model_config['condition_values']:
                                st.session_state['condition_values'] = model_config['condition_values']
                                st.session_state['condition_ratios'] = model_config['condition_ratios']
                                st.session_state['condition_encoder'] = transformer.encoders.get(condition_col)
                                st.session_state['condition_col'] = condition_col

                            gan = CGAN(
                                input_dim=main_data.shape[1],
                                condition_dim=condition_dim,
                                hidden_dim=model_config['hidden_dim'],
                                device=device
                            )
                        else:
                            # Default to using first column as condition if not specified
                            condition_col = df.columns[0]
                            condition_data = transformed_data[condition_col].values.reshape(-1, 1)
                            condition_dim = 1
                            main_data = transformed_data.drop(columns=[condition_col])

                            gan = CGAN(
                                input_dim=main_data.shape[1],
                                condition_dim=condition_dim,
                                hidden_dim=model_config['hidden_dim'],
                                device=device
                            )
                    elif model_config['model_type'] == 'TVAE':
                        gan = TVAE(
                            input_dim=transformed_data.shape[1],
                            hidden_dim=model_config['hidden_dim'],
                            latent_dim=model_config['latent_dim'],
                            device=device
                        )
                    else:  # TableGAN
                        gan = TableGAN(
                            input_dim=transformed_data.shape[1],
                            hidden_dim=model_config['hidden_dim'],
                            device=device
                        )

                    # Initialize training progress tracking
                    st.session_state.total_epochs = model_config['epochs']

                    # Clear previous training state
                    for key in ['progress_bar', 'status_text', 'loss_chart', 'training_losses']:
                        if key in st.session_state:
                            del st.session_state[key]

                    # Create placeholder for loss plot
                    loss_plot_placeholder = st.empty()

                    # Initialize loss storage
                    st.session_state.training_losses = {'epochs': [], 'generator': [], 'discriminator': []}
                    losses = gan.train(train_loader, model_config['epochs'], components.training_progress)


                    # Final loss plot
                    st.subheader("Training Progress")
                    final_loss_plot = components.plot_training_losses()
                    if final_loss_plot:
                        st.pyplot(final_loss_plot)


                    if model_config['model_type'] == 'CGAN' and 'condition_values' in model_config and model_config['condition_values']:
                        # Generate data based on selected condition values with their proportions
                        condition_values = model_config['condition_values']
                        condition_ratios = model_config['condition_ratios']
                        encoder = transformer.encoders.get(model_config['condition_column'])

                        total_samples = len(df)
                        synthetic_data_list = []

                        for value in condition_values:
                            # Calculate how many samples to generate for this value
                            num_samples = int(condition_ratios[value] * total_samples)
                            if num_samples < 1:
                                num_samples = 1

                            # Encode the condition value
                            encoded_value = encoder.transform([value])[0]
                            condition_tensor = torch.full((num_samples, 1), encoded_value, dtype=torch.float).to(device)

                            # Generate samples for this condition
                            value_samples = gan.generate_samples(num_samples, condition_tensor).cpu().numpy()

                            # Ensure the condition column has the correct encoded value
                            # We need to replace the last column with the encoded condition value
                            value_samples[:, -1] = encoded_value

                            synthetic_data_list.append(value_samples)

                        # Combine all generated samples
                        synthetic_data = np.vstack(synthetic_data_list)

                        # If we didn't generate exactly the requested number of samples, adjust
                        if len(synthetic_data) != total_samples:
                            indices = np.random.choice(len(synthetic_data), total_samples, replace=len(synthetic_data) < total_samples)
                            synthetic_data = synthetic_data[indices]
                    else:
                        synthetic_data = gan.generate_samples(len(df)).cpu().numpy()

                    # Display training losses plot



                # Inverse transform
                result_df = pd.DataFrame()
                col_idx = 0
                for col in selected_columns:  # Use only selected columns
                    col_type = column_types[col]
                    if col_type in ['Continuous', 'Ordinal']:
                        result_df[col] = transformer.inverse_transform_continuous(
                            pd.Series(synthetic_data[:, col_idx], name=col)
                        )
                        col_idx += 1
                    elif col_type == 'Categorical':
                        result_df[col] = transformer.inverse_transform_categorical(
                            pd.Series(synthetic_data[:, col_idx], name=col)
                        )
                        col_idx += 1
                    elif col_type == 'Datetime':
                        # Handle datetime reconstruction
                        year = synthetic_data[:, col_idx]
                        month = synthetic_data[:, col_idx + 1]
                        day = synthetic_data[:, col_idx + 2]
                        result_df[col] = pd.to_datetime(
                            dict(year=year, month=month, day=day)
                        )
                        col_idx += 4

                # Add excluded columns back with empty/NaN values if they were in the original data
                for col in original_columns:
                    if col not in selected_columns:
                        result_df[col] = None

                # Evaluate synthetic data
                st.subheader("Data Quality Evaluation")
                evaluator = DataEvaluator(df, result_df)

                # Statistical tests
                with st.expander("Statistical Similarity Tests"):
                    stats = evaluator.statistical_similarity()
                    for col, values in stats.items():
                        st.write(f"{col}: {values:.4f}")

                # Correlation similarity
                with st.expander("Correlation Matrix Similarity"):
                    corr_sim = evaluator.correlation_similarity()
                    st.write(f"Correlation Similarity Score: {corr_sim:.4f}")

                # Column statistics
                with st.expander("Column-wise Statistics Comparison"):
                    col_stats = evaluator.column_statistics()
                    st.dataframe(col_stats)

                # ML utility evaluation
                with st.expander("ML Utility Evaluation (TSTR)"):
                    ml_metrics = evaluator.evaluate_ml_utility(
                        target_column=target_col,
                        task_type=task_type
                    )
                    st.write("Train-Synthetic-Test-Real (TSTR) Evaluation:")
                    for metric, value in ml_metrics.items():
                        st.write(f"{metric}: {value:.4f}")

                # Distribution plots
                with st.expander("Distribution Comparisons"):
                    st.subheader("Cumulative Distribution Plots")
                    fig_cumulative = evaluator.plot_cumulative_distributions()
                    if fig_cumulative:
                        st.pyplot(fig_cumulative)

                    st.subheader("Density Distribution Plots")
                    fig_density = evaluator.plot_distributions()
                    st.pyplot(fig_density)

                # Add new divergence metrics section
                with st.expander("Distribution Divergence Metrics"):
                    metrics = evaluator.evaluate_all(target_column=target_col, task_type=task_type)
                    st.subheader("Jensen-Shannon Divergence (JSD)")
                    st.write("JSD measures the similarity between probability distributions (0 = identical, 1 = completely different)")
                    for col, value in metrics['divergence_metrics'].items():
                        if 'jsd' in col:
                            st.write(f"{col.replace('_jsd', '')}: {value:.4f}")

                    st.subheader("Wasserstein Distance (WD)")
                    st.write("WD measures the minimum 'cost' of transforming one distribution into another")
                    for col, value in metrics['divergence_metrics'].items():
                        if 'wasserstein' in col:
                            st.write(f"{col.replace('_wasserstein', '')}: {value:.4f}")

                # Display results
                st.success("Synthetic data generated successfully!")
                st.subheader("Generated Data Preview")
                st.dataframe(result_df.head())

                # Download button
                csv = result_df.to_csv(index=False)
                st.download_button(
                    label="Download Synthetic Data",
                    data=csv,
                    file_name="synthetic_data.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"An error occurred during data generation: {str(e)}")
                return

def model_config_section():
    st.subheader("Model Configuration")

    model_config = {}

    model_config['model_type'] = st.selectbox(
        "Select Model Type",
        options=['TableGAN', 'WGAN', 'CGAN', 'TVAE'],  # Added TVAE
        help="Choose the type of model to use for synthetic data generation"
    )

    # Add latent dimension parameter for TVAE
    if model_config['model_type'] == 'TVAE':
        model_config['latent_dim'] = st.slider(
            "Latent Dimension",
            min_value=32,
            max_value=256,
            value=128,
            step=32,
            help="Dimension of the latent space for TVAE"
        )

    model_config['hidden_dim'] = st.slider(
        "Hidden Dimension",
        min_value=64,
        max_value=512,
        value=256,
        step=64,
        help="Size of hidden layers in the model"
    )

    model_config['epochs'] = st.slider(
        "Number of Epochs",
        min_value=10,
        max_value=500,
        value=100,
        step=10,
        help="Number of training epochs"
    )

    model_config['batch_size'] = st.slider(
        "Batch Size",
        min_value=32,
        max_value=256,
        value=64,
        step=32,
        help="Number of samples per training batch"
    )

    if model_config['model_type'] == 'WGAN':
        model_config['clip_value'] = st.slider(
            "Clip Value",
            min_value=0.01,
            max_value=0.1,
            value=0.01,
            step=0.01,
            help="Weight clipping value for WGAN"
        )
        model_config['n_critic'] = st.slider(
            "Critic Updates",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            help="Number of critic updates per generator update"
        )

    return model_config

if __name__ == "__main__":
    main()