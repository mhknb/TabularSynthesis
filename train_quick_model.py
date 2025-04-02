"""
Script to train a very small model from a dataset and save it to be used for fine-tuning
"""
import pandas as pd
import torch
import numpy as np
from src.models.table_gan import TableGAN
from src.data_processing.transformers import DataTransformer
import os

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

def train_and_save_model():
    # Load the dataset
    data = pd.read_csv('attached_assets/cataneo2.csv')
    
    # Use only a small subset (first 500 rows) for faster training
    data = data.head(500)
    
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
    
    print("Categorical indices:", categorical_indices)
    print("Categorical dimensions:", categorical_dims)
    
    # Configure training parameters
    input_dim = data.shape[1]
    hidden_dim = 64  # Much smaller for faster training
    epochs = 2  # Minimal epochs for testing
    batch_size = 32  # Smaller batch size
    model_name = "tablegan_base_model"
    
    print(f"Training TableGAN model with input dimension {input_dim}")
    print(f"Model will be saved as {model_name}.pt")
    
    # Initialize the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    gan = TableGAN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        device=device,
        categorical_columns=categorical_indices,
        categorical_dims=categorical_dims
    )
    
    # Convert data to tensor
    train_data = torch.FloatTensor(data.values)
    train_loader = torch.utils.data.DataLoader(
        train_data, 
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    
    # Train the model
    try:
        print("Starting training...")
        history = []
        
        # Use a simplified training approach
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            epoch_losses = []
            
            for batch_idx, batch in enumerate(train_loader):
                metrics = gan.train_step(batch)
                epoch_losses.append(metrics)
                if batch_idx % 5 == 0:
                    print(f"  Batch {batch_idx}, G Loss: {metrics['generator_loss']:.4f}, D Loss: {metrics['discriminator_loss']:.4f}")
            
            # Print epoch summary
            avg_g_loss = sum(d['generator_loss'] for d in epoch_losses) / len(epoch_losses)
            avg_d_loss = sum(d['discriminator_loss'] for d in epoch_losses) / len(epoch_losses)
            print(f"  Epoch {epoch+1} summary - G Loss: {avg_g_loss:.4f}, D Loss: {avg_d_loss:.4f}")
            
            history.append({
                'epoch': epoch,
                'generator_loss': avg_g_loss,
                'discriminator_loss': avg_d_loss
            })
        
        print("Training complete")
        
        # Save the model
        model_path = f"models/{model_name}.pt"
        torch.save(gan.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
        # Also save with timestamp for versioning
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        versioned_path = f"models/{model_name}_{timestamp}.pt"
        torch.save(gan.state_dict(), versioned_path)
        print(f"Model saved to {versioned_path}")
        
        # List files in the models directory to confirm
        print("Models directory contents:")
        print(os.listdir('models'))
        
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        
if __name__ == "__main__":
    train_and_save_model()