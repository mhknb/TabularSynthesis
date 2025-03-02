import modal
from src.models.table_gan import TableGAN
import torch
import pandas as pd
import numpy as np

# Initialize Modal and create app
stub = modal.Stub("synthetic-data-generator")
volume = modal.Volume.from_name("gan-model-vol")
image = modal.Image.debian_slim().pip_install(["torch", "numpy", "pandas"])

# Initialize Modal at module level
try:
    if not stub.is_initialized():
        modal.init()
except Exception as e:
    print(f"Modal initialization warning (non-critical): {str(e)}")

@stub.function(
    gpu="T4",
    volumes={"/model": volume},
    image=image
)
def train_gan_async(data: pd.DataFrame, input_dim: int, hidden_dim: int, epochs: int, batch_size: int):
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
    torch.save(gan.state_dict(), "/model/table_gan.pt")
    return losses

@stub.function(
    gpu="T4",
    volumes={"/model": volume},
    image=image
)
def generate_samples_async(num_samples: int, input_dim: int, hidden_dim: int) -> np.ndarray:
    """Generate synthetic samples using Modal"""
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
    """Simple wrapper for Modal GAN operations"""

    def train(self, data: pd.DataFrame, input_dim: int, hidden_dim: int, epochs: int, batch_size: int):
        """Train GAN model using Modal"""
        try:
            return train_gan_async.remote(data, input_dim, hidden_dim, epochs, batch_size)
        except Exception as e:
            raise RuntimeError(f"Modal training failed: {str(e)}")

    def generate(self, num_samples: int, input_dim: int, hidden_dim: int) -> np.ndarray:
        """Generate synthetic samples using Modal"""
        try:
            return generate_samples_async.remote(num_samples, input_dim, hidden_dim)
        except Exception as e:
            raise RuntimeError(f"Modal generation failed: {str(e)}")