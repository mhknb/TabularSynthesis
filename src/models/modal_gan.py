import modal
from src.models.table_gan import TableGAN
import torch
import pandas as pd
import numpy as np

# Create Modal stub
stub = modal.Stub("synthetic-data-generator")

# Create Modal volume for model persistence
volume = modal.Volume.from_name("gan-model-vol")

# Define model path constant
MODEL_PATH = "/model/table_gan.pt"

# Define training function in global scope
@stub.function(
    gpu="T4",
    volumes={"/model": volume},
    image=modal.Image.debian_slim().pip_install(["torch", "numpy", "pandas"])
)
def train_gan(data: pd.DataFrame, input_dim: int, hidden_dim: int, epochs: int, batch_size: int):
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
    torch.save(gan.state_dict(), MODEL_PATH)
    return losses

# Define generation function in global scope
@stub.function(
    gpu="T4",
    volumes={"/model": volume},
    image=modal.Image.debian_slim().pip_install(["torch", "numpy", "pandas"])
)
def generate_samples(num_samples: int, input_dim: int, hidden_dim: int) -> np.ndarray:
    """Generate synthetic samples using Modal"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gan = TableGAN(input_dim=input_dim, hidden_dim=hidden_dim, device=device)

    # Load saved model if exists
    try:
        gan.load_state_dict(torch.load(MODEL_PATH))
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
        try:
            modal.setup()
        except Exception as e:
            print(f"Modal setup error: {str(e)}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def train(self, data: pd.DataFrame, input_dim: int, hidden_dim: int, epochs: int, batch_size: int):
        """Train GAN model using Modal"""
        return train_gan.remote(data, input_dim, hidden_dim, epochs, batch_size)

    def generate(self, num_samples: int, input_dim: int, hidden_dim: int) -> np.ndarray:
        """Generate synthetic samples using Modal"""
        return generate_samples.remote(num_samples, input_dim, hidden_dim)