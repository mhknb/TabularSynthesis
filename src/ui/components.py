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

def transformation_selector(column_types):
    """Component for selecting data transformations"""

    st.subheader("Data Transformations")

    transformations = {}

    with st.expander("Configure Transformations"):
        for col, col_type in column_types.items():
            if col_type == 'Continuous':
                transformations[col] = st.selectbox(
                    f"Transform '{col}'",
                    options=['standard', 'minmax', 'robust', 'none'],
                    key=f"transform_{col}"
                )
            elif col_type == 'Categorical':
                transformations[col] = st.selectbox(
                    f"Encode '{col}'",
                    options=['label', 'onehot', 'none'],
                    key=f"transform_{col}"
                )

    return transformations

def outlier_detection_config():
    """Component for configuring outlier detection and removal"""

    st.subheader("Outlier Detection")

    with st.expander("Configure Outlier Detection"):
        use_outlier_detection = st.checkbox("Detect and remove outliers", value=True,
                                           help="Remove outliers from training data to improve model quality")

        if use_outlier_detection:
            outlier_method = st.selectbox(
                "Outlier detection method",
                options=['iqr', 'zscore', 'isolation_forest'],
                help="IQR: Interquartile Range method, Z-score: Statistical method, Isolation Forest: Machine learning method"
            )

            if outlier_method == 'iqr':
                k_value = st.slider("IQR multiplier (k)", 
                                   min_value=1.0, max_value=3.0, value=1.5, step=0.1,
                                   help="Higher values are more permissive (fewer outliers detected)")
                params = {'method': outlier_method, 'k': k_value}

            elif outlier_method == 'zscore':
                z_threshold = st.slider("Z-score threshold", 
                                       min_value=2.0, max_value=5.0, value=3.0, step=0.1,
                                       help="Higher values are more permissive (fewer outliers detected)")
                params = {'method': outlier_method, 'z_threshold': z_threshold}

            elif outlier_method == 'isolation_forest':
                contamination = st.slider("Contamination (expected proportion of outliers)", 
                                         min_value=0.01, max_value=0.1, value=0.05, step=0.01,
                                         help="Higher values detect more outliers")
                params = {'method': outlier_method, 'contamination': contamination}
        else:
            params = {'method': None}

    return params

def model_config_section():
    """Create model configuration interface"""
    st.header("Model Configuration")

    config = {
        'model_type': st.selectbox(
            "Select GAN Model",
            options=['TableGAN', 'WGAN', 'CGAN'],
            help="Choose the type of GAN model to use for synthetic data generation"
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
    # CGAN specific parameters
    elif config['model_type'] == 'CGAN':
        config.update({
            'condition_column': st.selectbox(
                "Condition Column",
                options=[None] + list(st.session_state.get('uploaded_df', pd.DataFrame()).columns),
                help="Column to use as a condition for generating data (CGAN specific)"
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