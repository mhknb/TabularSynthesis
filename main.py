import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import streamlit as st
import torch
import pandas as pd
import numpy as np
from src.data_processing.data_loader import DataLoader
from src.data_processing.transformers import DataTransformer
from src.models.table_gan import TableGAN
from src.models.modal_gan import ModalGAN
from src.models.wgan import WGAN
from src.models.cgan import CGAN
from src.utils.validation import validate_data, check_column_types
from src.utils.evaluation import DataEvaluator
from src.ui import components
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Synthetic Data Generator", layout="wide")

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
    model_config = components.model_config_section()

    # Add Modal training option
    use_modal = st.checkbox("Use Modal for cloud training (faster)", value=True)

    # Target column selection for ML evaluation
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
            # Split data into train and test sets
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            st.info(f"Data split into {len(train_df)} training samples and {len(test_df)} test samples")

            # Transform data
            transformer = DataTransformer()
            transformed_data = pd.DataFrame()

            for col in original_columns:  # Use original column order
                col_type = column_types[col]
                if col_type == 'Continuous':
                    transformed_col = transformer.transform_continuous(
                        train_df[col], 
                        transformations.get(col, 'standard')  # Default to standard scaling
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

            if use_modal:
                try:
                    with st.spinner("Training model on Modal cloud..."):
                        # Train on Modal
                        losses = modal_gan.train(
                            transformed_data,
                            input_dim=transformed_data.shape[1],
                            hidden_dim=model_config['hidden_dim'],
                            epochs=model_config['epochs'],
                            batch_size=model_config['batch_size']
                        )

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
                else:  # TableGAN
                    gan = TableGAN(
                        input_dim=transformed_data.shape[1],
                        hidden_dim=model_config['hidden_dim'],
                        device=device
                    )

                st.session_state.total_epochs = model_config['epochs']
                losses = gan.train(train_loader, model_config['epochs'], components.training_progress)
                synthetic_data = gan.generate_samples(len(df)).cpu().numpy()

            # Inverse transform
            result_df = pd.DataFrame()
            col_idx = 0
            for col in original_columns:  # Use original column order
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
                fig = evaluator.plot_distributions()
                st.pyplot(fig)

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

if __name__ == "__main__":
    main()