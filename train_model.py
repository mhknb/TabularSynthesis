"""
Script to train a model from a dataset and save it to be used for fine-tuning
"""
import pandas as pd
import torch
import numpy as np
from src.models.modal_gan import ModalGAN
from src.data_processing.transformers import DataTransformer
import os

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

def train_and_save_model():
    # Load the dataset
    data = pd.read_csv('attached_assets/cataneo2.csv')
    
    print(f"Loaded dataset with shape: {data.shape}")
    print("Columns:", list(data.columns))
    
    # Identify numerical and categorical columns
    numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in data.columns if col not in numerical_cols]
    
    print(f"Numerical columns: {len(numerical_cols)}")
    print(f"Categorical columns: {len(categorical_cols)}")
    
    # Transform data for model training
    transformer = DataTransformer()
    
    # Process numerical columns
    for col in numerical_cols:
        data[col] = transformer.transform_continuous(data[col], method='standard')
    
    # Process categorical columns
    categorical_indices = []
    categorical_dims = {}
    
    for i, col in enumerate(categorical_cols):
        data[col] = transformer.transform_categorical(data[col], method='label')
        col_idx = data.columns.get_loc(col)
        categorical_indices.append(col_idx)
        # Get number of unique values for this column
        categorical_dims[col_idx] = len(data[col].unique())
    
    # Initialize ModalGAN
    modal_gan = ModalGAN()
    
    # Configure training parameters
    input_dim = data.shape[1]
    hidden_dim = 256
    epochs = 30
    batch_size = 64
    model_type = 'TableGAN'  # You can change to 'CGAN', 'WGAN', 'TVAE', or 'CTGAN'
    model_name = f"{model_type.lower()}_base_model"
    
    print(f"Training {model_type} model with input dimension {input_dim}")
    print(f"Model will be saved as {model_name}.pt")
    
    # Train the model
    try:
        modal_gan.train(
            data=data,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            epochs=epochs,
            batch_size=batch_size,
            model_type=model_type,
            model_name=model_name,
            categorical_columns=categorical_indices,
            categorical_dims=categorical_dims
        )
        print(f"Model training complete. Model saved as {model_name}.pt")
        
        # List available models to verify
        models = modal_gan.list_available_models()
        print(f"Available models: {models}")
        
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        
if __name__ == "__main__":
    train_and_save_model()