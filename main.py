import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import streamlit as st
import torch
import pandas as pd
import numpy as np
from src.data_processing.data_loader import DataLoader
from src.data_processing.advanced_transformations import AdvancedTransformer
from src.models.table_gan import TableGAN
from src.models.modal_gan import ModalGAN
from src.utils.validation import validate_data, check_column_types
from src.ui import components

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

    # Advanced transformation configuration
    transform_config = components.advanced_transformation_options()

    # Model configuration
    model_config = {
        'hidden_dim': st.slider("Hidden Layer Dimension", 64, 512, 256, 64),
        'batch_size': st.slider("Batch Size", 16, 256, 64, 16),
        'epochs': st.slider("Number of Epochs", 10, 1000, 100, 10),
        'learning_rate': st.select_slider(
            "Learning Rate",
            options=[0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005],
            value=0.0002
        )
    }

    # Add Modal training option
    use_modal = st.checkbox("Use Modal for cloud training (faster)", value=True)

    if st.button("Generate Synthetic Data"):
        with st.spinner("Preparing data..."):
            # Initialize transformer
            transformer = AdvancedTransformer()

            # Apply advanced transformations
            transformed_data = transformer.fit_transform(df, transform_config)

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
                gan = TableGAN(
                    input_dim=transformed_data.shape[1],
                    hidden_dim=model_config['hidden_dim'],
                    device=device
                )

                st.session_state.total_epochs = model_config['epochs']
                losses = gan.train(train_loader, model_config['epochs'], components.training_progress)
                synthetic_data = gan.generate_samples(len(df)).cpu().numpy()

            # Inverse transform the generated data
            result_df = transformer.inverse_transform(pd.DataFrame(synthetic_data))

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