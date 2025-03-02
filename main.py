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
from src.utils.validation import validate_data, check_column_types
from src.ui import components
import io

st.set_page_config(page_title="Synthetic Data Generator", layout="wide")

def main():
    st.title("Synthetic Tabular Data Generator")

    # File upload
    df, error = components.file_uploader()
    if error:
        st.error(error)
        return
    if df is None:
        return

    # Validate data
    valid, issues = validate_data(df)
    if not valid:
        st.warning("Data validation issues found:")
        for issue in issues:
            st.write(f"- {issue}")

    # Data preview
    components.data_preview(df)

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

    if st.button("Generate Synthetic Data"):
        with st.spinner("Preparing data..."):
            # Transform data
            transformer = DataTransformer()
            transformed_data = pd.DataFrame()

            for col, col_type in column_types.items():
                if col_type == 'Continuous':
                    transformed_data[col] = transformer.transform_continuous(
                        df[col], 
                        transformations.get(col, 'minmax')
                    )
                elif col_type == 'Categorical':
                    transformed_data[col] = transformer.transform_categorical(
                        df[col], 
                        transformations.get(col, 'label')
                    )
                elif col_type == 'Datetime':
                    dt_features = transformer.transform_datetime(df[col])
                    transformed_data = pd.concat([transformed_data, dt_features], axis=1)

            # Prepare data for training
            train_data = torch.FloatTensor(transformed_data.values)
            train_loader = torch.utils.data.DataLoader(
                train_data, 
                batch_size=model_config['batch_size'],
                shuffle=True
            )

            # Initialize and train GAN
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            gan = TableGAN(
                input_dim=transformed_data.shape[1],
                hidden_dim=model_config['hidden_dim'],
                device=device
            )

            st.session_state.total_epochs = model_config['epochs']
            losses = gan.train(train_loader, model_config['epochs'], components.training_progress)

            # Generate synthetic data
            num_samples = len(df)
            synthetic_data = gan.generate_samples(num_samples).cpu().numpy()

            # Inverse transform
            result_df = pd.DataFrame()
            col_idx = 0

            for col, col_type in column_types.items():
                if col_type == 'Continuous':
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