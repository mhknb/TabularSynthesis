import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, Any
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
                options=['standard', 'robust', 'minmax'], 
                key=f"transform_{col}"
            )
        elif col_type == 'Categorical':
            transformations[col] = st.selectbox(
                f"Encoding method for '{col}'",
                options=['onehot', 'label'],  # Changed order to make 'onehot' the default
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
                        
def display_quality_score(quality_score_details: Dict[str, Any]):
    """
    Display AI-powered quality score indicator with radar chart and recommendations
    
    Args:
        quality_score_details: Dictionary containing quality score details from DataEvaluator
    """
    st.header("📊 AI Quality Assessment")
    
    if not quality_score_details or 'overall_score' not in quality_score_details:
        st.warning("Quality score data is not available.")
        return
    
    # Extract scores and labels
    overall_score = quality_score_details.get('overall_score', 0)
    overall_label = quality_score_details.get('overall_label', 'N/A')
    category_scores = quality_score_details.get('category_scores', {})
    category_labels = quality_score_details.get('category_labels', {})
    improvement = quality_score_details.get('improvement', {})
    
    # Create columns for layout
    col1, col2 = st.columns([1, 2])
    
    # Column 1: Overall score display
    with col1:
        st.subheader("Overall Quality")
        
        # Display score in a gauge-like format with color
        score_color = get_score_color(overall_score)
        st.markdown(
            f"""
            <div style="background-color: #f0f0f0; border-radius: 10px; padding: 10px; text-align: center;">
                <h1 style="color: {score_color}; font-size: 3em; margin: 0;">{overall_score:.2f}</h1>
                <p style="font-size: 1.5em; margin: 0;">{overall_label}</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Show improvement recommendation
        st.subheader("Recommended Focus")
        st.info(f"**{improvement.get('focus_area', 'N/A')}**: {improvement.get('advice', 'N/A')}")
    
    # Column 2: Radar chart for category scores
    with col2:
        st.subheader("Quality Dimensions")
        
        # Create radar chart
        fig = create_radar_chart(category_scores)
        st.pyplot(fig)
        
        # Display category scores in a table
        st.markdown("#### Dimension Scores")
        score_data = {
            "Dimension": list(category_scores.keys()),
            "Score": [f"{score:.2f}" for score in category_scores.values()],
            "Rating": [category_labels.get(cat, "N/A") for cat in category_scores.keys()]
        }
        score_df = pd.DataFrame(score_data)
        st.dataframe(score_df, hide_index=True)
        
def create_radar_chart(category_scores: Dict[str, float]):
    """Create a radar chart for quality score dimensions"""
    categories = list(category_scores.keys())
    categories = [cat.capitalize() for cat in categories]
    
    values = list(category_scores.values())
    
    # Number of variables
    N = len(categories)
    
    # What will be the angle of each axis in the plot
    angles = [n / N * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Values need to be repeated to close the loop
    values += values[:1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], categories, color='grey', size=12)
    
    # Draw the chart
    ax.plot(angles, values, linewidth=2, linestyle='solid', color='#4CAF50')
    ax.fill(angles, values, color='#4CAF50', alpha=0.25)
    
    # Fix axis to go from 0 to 1
    ax.set_ylim(0, 1)
    
    # Add grid lines and labels
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=10)
    plt.ylim(0, 1)
    
    # Add a title
    plt.title("Quality Dimensions", size=15, y=1.1)
    
    return fig

def get_score_color(score: float) -> str:
    """Return appropriate color based on score"""
    if score >= 0.8:
        return "#2E7D32"  # Dark green
    elif score >= 0.6:
        return "#4CAF50"  # Green
    elif score >= 0.4:
        return "#FF9800"  # Orange
    else:
        return "#F44336"  # Red