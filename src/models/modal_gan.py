import modal
import torch
import pandas as pd
import numpy as np
from src.models.table_gan import TableGAN
import json
import logging

# Define app and shared resources at module level
app = modal.App("synthetic-data-generator")
volume = modal.Volume.from_name("gan-model-vol", create_if_missing=True)
image = modal.Image.debian_slim().pip_install(["torch", "numpy", "pandas"])

@app.function(
    gpu="T4",
    volumes={"/model": volume},
    image=image,
    timeout=1800
)
def train_gan_remote(data: pd.DataFrame, input_dim: int, hidden_dim: int, epochs: int, batch_size: int):
    """Train GAN model using Modal remote execution with structured logging"""
    import sys
    import logging
    import json

    # Configure logging
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gan = TableGAN(input_dim=input_dim, hidden_dim=hidden_dim, device=device)

    train_data = torch.FloatTensor(data.values)
    train_loader = torch.utils.data.DataLoader(
        train_data, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    all_losses = []
    best_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        epoch_generator_loss = 0.0
        epoch_discriminator_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            loss_dict = gan.train_step(batch)
            epoch_generator_loss += loss_dict['generator_loss']
            epoch_discriminator_loss += loss_dict.get('discriminator_loss', loss_dict.get('critic_loss', 0.0))
            num_batches += 1

        # Calculate average losses
        avg_generator_loss = epoch_generator_loss / num_batches
        avg_discriminator_loss = epoch_discriminator_loss / num_batches

        # Store epoch results
        epoch_result = {
            'epoch': epoch,
            'generator_loss': avg_generator_loss,
            'discriminator_loss': avg_discriminator_loss
        }
        all_losses.append(epoch_result)

        # Print progress that can be captured from Modal output
        print(json.dumps({
            'type': 'epoch_update',
            'epoch': epoch,
            'total_epochs': epochs,
            'generator_loss': avg_generator_loss,
            'discriminator_loss': avg_discriminator_loss
        }))

        # Early stopping check
        current_loss = avg_generator_loss + avg_discriminator_loss
        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0
            torch.save(gan.state_dict(), "/model/table_gan.pt")
            volume.commit()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(json.dumps({
                    'type': 'early_stopping',
                    'epoch': epoch,
                    'message': 'Early stopping triggered'
                }))
                break

    return all_losses

@app.function(
    gpu="T4",
    volumes={"/model": volume},
    image=image,
    timeout=600
)
def generate_samples_remote(num_samples: int, input_dim: int, hidden_dim: int) -> np.ndarray:
    """Generate synthetic samples using Modal remote execution"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gan = TableGAN(input_dim=input_dim, hidden_dim=hidden_dim, device=device)

    try:
        gan.load_state_dict(torch.load("/model/table_gan.pt"))
    except Exception as e:
        raise ValueError(f"Failed to load model: {str(e)}")

    # Generate in batches
    batch_size = 1000
    num_batches = (num_samples + batch_size - 1) // batch_size
    synthetic_data_list = []

    with torch.no_grad():
        for i in range(num_batches):
            current_batch_size = min(batch_size, num_samples - i * batch_size)
            noise = torch.randn(current_batch_size, input_dim).to(device)
            batch_data = gan.generator(noise)
            synthetic_data_list.append(batch_data.cpu().numpy())

    return np.concatenate(synthetic_data_list)

class ModalGAN:
    """Class for managing Modal GAN operations"""

    def train(self, data: pd.DataFrame, input_dim: int, hidden_dim: int, epochs: int, batch_size: int):
        """Train GAN model using Modal with real-time log processing"""
        try:
            with app.run():
                # Start training and get the output directly
                output = train_gan_remote.remote(data, input_dim, hidden_dim, epochs, batch_size).get()

                # Process output and extract epoch information
                reformatted_losses = []
                for loss_dict in output:
                    reformatted_losses.append((
                        loss_dict['epoch'],
                        {
                            'generator_loss': loss_dict['generator_loss'],
                            'discriminator_loss': loss_dict['discriminator_loss']
                        }
                    ))

                return reformatted_losses

        except Exception as e:
            if "timeout" in str(e).lower():
                raise RuntimeError("Modal training exceeded time limit. Try reducing epochs or batch size.")
            raise RuntimeError(f"Modal training failed: {str(e)}")

    def generate(self, num_samples: int, input_dim: int, hidden_dim: int) -> np.ndarray:
        """Generate synthetic samples using Modal"""
        try:
            with app.run():
                return generate_samples_remote.remote(num_samples, input_dim, hidden_dim).get()
        except Exception as e:
            raise RuntimeError(f"Modal generation failed: {str(e)}")