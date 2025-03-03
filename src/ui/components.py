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

def advanced_transformation_options():
    """Create advanced transformation configuration interface"""
    st.header("Advanced Transformation Settings")

    config = {
        'scaling_method': st.selectbox(
            "Scaling Method",
            options=['minmax', 'standard', 'robust', 'power', 'quantile'],
            help="Choose the scaling method for numerical features"
        ),
        'encoding_method': st.selectbox(
            "Categorical Encoding",
            options=['label', 'onehot', 'target'],
            help="Choose the encoding method for categorical features"
        ),
        'handle_missing': st.checkbox(
            "Handle Missing Values",
            value=True,
            help="Automatically handle missing values in the dataset"
        ),
        'handle_outliers': st.checkbox(
            "Handle Outliers",
            value=True,
            help="Detect and handle outliers in numerical features"
        ),
        'feature_engineering': st.multiselect(
            "Feature Engineering Operations",
            options=['polynomial', 'log', 'interaction'],
            default=[],
            help="Select additional feature engineering operations"
        )
    }

    return config

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

def training_progress(epoch: int, losses: dict):
    """Update training progress"""
    progress_bar = st.progress(0)
    status_text = st.empty()

    progress_bar.progress(epoch / st.session_state.total_epochs)
    status_text.text(f"Epoch {epoch}: Generator Loss: {losses['generator_loss']:.4f}, "
                    f"Discriminator Loss: {losses['discriminator_loss']:.4f}")