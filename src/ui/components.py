import streamlit as st
import pandas as pd
from typing import Tuple, Optional
import io

def file_uploader() -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Create file upload widget and handle uploaded file"""
    st.header("Data Upload")
    uploaded_file = st.file_uploader(
        "Choose a file", 
        type=['csv', 'xlsx', 'xls', 'parquet'],
        help="Upload your tabular data file (CSV, Excel, or Parquet format)"
    )

    if uploaded_file is not None:
        from src.data_processing.data_loader import DataLoader
        return DataLoader.load_data(uploaded_file, uploaded_file.name)
    return None, None

def data_preview(df: pd.DataFrame):
    """Display data preview with basic statistics"""
    st.header("Data Preview")

    st.subheader("Sample Data")
    st.dataframe(df.head())

    st.subheader("Data Info")
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

    st.subheader("Basic Statistics")
    st.write(df.describe())

def column_type_selector(df: pd.DataFrame):
    """Create column type selection interface"""
    st.header("Column Configuration")

    from src.data_processing.data_loader import DataLoader
    inferred_types = DataLoader.infer_column_types(df)

    column_types = {}
    for col in df.columns:
        column_types[col] = st.selectbox(
            f"Select type for '{col}'",
            options=['Continuous', 'Categorical', 'Datetime', 'Ordinal'],
            index=['Continuous', 'Categorical', 'Datetime', 'Ordinal'].index(inferred_types[col])
        )
    return column_types

def transformation_selector(column_types: dict):
    """Create transformation selection interface"""
    st.header("Transformation Settings")

    transformations = {}
    for col, col_type in column_types.items():
        st.subheader(f"Column: {col}")

        if col_type == 'Continuous':
            transformations[col] = st.selectbox(
                f"Scaling method for '{col}'",
                options=['robust', 'standard', 'minmax'],
                key=f"transform_{col}"
            )
        elif col_type == 'Categorical':
            transformations[col] = st.selectbox(
                f"Encoding method for '{col}'",
                options=['binary', 'onehot', 'label'],
                key=f"transform_{col}"
            )

    return transformations

def model_config_section():
    """Create model configuration interface"""
    st.header("Model Configuration")

    config = {
        'model_type': st.selectbox(
            "Select GAN Model",
            options=['TableGAN', 'WGAN', 'CGAN', 'TVAE'],  # Added TVAE to options
            help="Choose the type of model to use for synthetic data generation"
        ),
        'hidden_dim': st.slider("Hidden Layer Dimension", 64, 512, 256, 64),
        'batch_size': st.slider("Batch Size", 16, 256, 64, 16),
        'epochs': st.slider("Number of Epochs", 10, 1000, 100, 10),
        'learning_rate': st.select_slider(
            "Learning Rate",
            options=[0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005],
            value=0.0002
        )
    }

    # TVAE specific parameters
    if config['model_type'] == 'TVAE':
        config['latent_dim'] = st.slider(
            "Latent Dimension",
            min_value=32,
            max_value=256,
            value=128,
            step=32,
            help="Dimension of the latent space for TVAE"
        )

    # WGAN specific parameters
    if config['model_type'] == 'WGAN':
        config.update({
            'clip_value': st.slider(
                "Weight Clipping Value",
                0.001, 0.1, 0.01, 0.001,
                help="Maximum allowed weight value in the critic (WGAN specific)"
            ),
            'n_critic': st.slider(
                "Number of Critic Updates",
                1, 10, 5, 1,
                help="Number of critic updates per generator update (WGAN specific)"
            )
        })

    return config

def training_progress(epoch: int, losses: dict):
    """Update training progress"""
    progress_bar = st.progress(0)
    status_text = st.empty()

    progress_bar.progress(epoch / st.session_state.total_epochs)
    if 'critic_loss' in losses:
        status_text.text(f"Epoch {epoch}: Generator Loss: {losses['generator_loss']:.4f}, "
                        f"Critic Loss: {losses['critic_loss']:.4f}")
    else:
        status_text.text(f"Epoch {epoch}: Generator Loss: {losses['generator_loss']:.4f}, "
                        f"Discriminator Loss: {losses['discriminator_loss']:.4f}")