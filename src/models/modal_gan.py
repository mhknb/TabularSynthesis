import modal
from src.models.table_gan import TableGAN
import torch
import pandas as pd
import numpy as np

# Create Modal stub
stub = modal.Stub("synthetic-data-generator")

# Create Modal volume for model persistence
volume = modal.Volume().persist("gan-model-vol")

@stub.cls(
    gpu="T4",
    volumes={"/model": volume},
    image=modal.Image.debian_slim().pip_install(["torch", "numpy", "pandas"])
)
class ModalGAN:
    def __init__(self):
        self.model_path = "/model/table_gan.pt"
        
    @modal.method()
    def train(self, data: pd.DataFrame, input_dim: int, hidden_dim: int, epochs: int, batch_size: int):
        """Train GAN model using Modal"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        gan = TableGAN(input_dim=input_dim, hidden_dim=hidden_dim, device=device)
        
        # Convert data to tensor
        train_data = torch.FloatTensor(data.values)
        train_loader = torch.utils.data.DataLoader(
            train_data, 
            batch_size=batch_size,
            shuffle=True
        )
        
        # Train model
        losses = []
        for epoch in range(epochs):
            epoch_losses = []
            for batch in train_loader:
                loss_dict = gan.train_step(batch)
                epoch_losses.append(loss_dict)
                
            avg_losses = {k: sum(d[k] for d in epoch_losses) / len(epoch_losses) 
                         for k in epoch_losses[0].keys()}
            losses.append(avg_losses)
            
        # Save model
        torch.save(gan.state_dict(), self.model_path)
        return losses
        
    @modal.method()
    def generate(self, num_samples: int, input_dim: int, hidden_dim: int) -> np.ndarray:
        """Generate synthetic samples using Modal"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        gan = TableGAN(input_dim=input_dim, hidden_dim=hidden_dim, device=device)
        
        # Load saved model if exists
        try:
            gan.load_state_dict(torch.load(self.model_path))
        except:
            raise ValueError("No trained model found. Please train the model first.")
            
        # Generate samples
        with torch.no_grad():
            noise = torch.randn(num_samples, input_dim).to(device)
            synthetic_data = gan.generator(noise)
            return synthetic_data.cpu().numpy()
