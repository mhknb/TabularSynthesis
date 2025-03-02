import modal
from src.models.table_gan import TableGAN
import torch
import pandas as pd
import numpy as np

# Initialize Modal volume
volume = modal.Volume.from_name("gan-model-vol")

# Create Modal image with required dependencies
image = modal.Image.debian_slim().pip_install(["torch", "numpy", "pandas"])

# Create Modal app with proper initialization
app = modal.App()

@app.function(
    gpu="T4",
    volumes={"/model": volume},
    image=image
)
async def train_gan_remote(data: pd.DataFrame, input_dim: int, hidden_dim: int, epochs: int, batch_size: int):
    """Train GAN model using Modal remote execution"""
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
    torch.save(gan.state_dict(), "/model/table_gan.pt")
    return losses

@app.function(
    gpu="T4",
    volumes={"/model": volume},
    image=image
)
async def generate_samples_remote(num_samples: int, input_dim: int, hidden_dim: int) -> np.ndarray:
    """Generate synthetic samples using Modal remote execution"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gan = TableGAN(input_dim=input_dim, hidden_dim=hidden_dim, device=device)

    # Load saved model if exists
    try:
        gan.load_state_dict(torch.load("/model/table_gan.pt"))
    except Exception as e:
        raise ValueError(f"Failed to load model: {str(e)}")

    # Generate samples
    with torch.no_grad():
        noise = torch.randn(num_samples, input_dim).to(device)
        synthetic_data = gan.generator(noise)
        return synthetic_data.cpu().numpy()

class ModalGAN:
    """Class for managing Modal GAN operations"""

    def __init__(self):
        """Initialize Modal app"""
        try:
            modal.init()
            self.app = app
        except Exception as e:
            print(f"Modal initialization warning: {str(e)}")

    def train(self, data: pd.DataFrame, input_dim: int, hidden_dim: int, epochs: int, batch_size: int):
        """Train GAN model using Modal"""
        try:
            return train_gan_remote.remote(data, input_dim, hidden_dim, epochs, batch_size)
        except Exception as e:
            raise RuntimeError(f"Modal training failed: {str(e)}")

    def generate(self, num_samples: int, input_dim: int, hidden_dim: int) -> np.ndarray:
        """Generate synthetic samples using Modal"""
        try:
            return generate_samples_remote.remote(num_samples, input_dim, hidden_dim)
        except Exception as e:
            raise RuntimeError(f"Modal generation failed: {str(e)}")