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

    # Configure logging with JSON format
    class JsonFormatter(logging.Formatter):
        def format(self, record):
            if isinstance(record.msg, dict):
                return json.dumps(record.msg)
            return json.dumps({
                'message': record.getMessage(),
                'level': record.levelname,
                'time': self.formatTime(record)
            })

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
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

        # Log progress as structured JSON
        logging.info({
            'type': 'epoch_update',
            'epoch': epoch,
            'total_epochs': epochs,
            'generator_loss': avg_generator_loss,
            'discriminator_loss': avg_discriminator_loss
        })

        # Store epoch results
        epoch_result = {
            'epoch': epoch,
            'generator_loss': avg_generator_loss,
            'discriminator_loss': avg_discriminator_loss
        }
        all_losses.append(epoch_result)

        # Early stopping check
        current_loss = avg_generator_loss + avg_discriminator_loss
        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0
            torch.save(gan.state_dict(), "/model/table_gan.pt")
            volume.commit()
            logging.info({
                'type': 'model_save',
                'epoch': epoch,
                'loss': current_loss
            })
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info({
                    'type': 'early_stopping',
                    'epoch': epoch,
                    'message': 'Early stopping triggered'
                })
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
            with app.run() as app_ctx:
                reformatted_losses = []

                # Stream logs during training
                with app_ctx.logs() as logs:
                    train_future = train_gan_remote.remote(data, input_dim, hidden_dim, epochs, batch_size)

                    # Process logs in real-time
                    for log_entry in logs:
                        try:
                            log_data = json.loads(log_entry)
                            if isinstance(log_data, dict) and log_data.get('type') == 'epoch_update':
                                # Format the loss data for training progress
                                epoch = log_data['epoch']
                                losses = {
                                    'generator_loss': log_data['generator_loss'],
                                    'discriminator_loss': log_data['discriminator_loss']
                                }
                                reformatted_losses.append((epoch, losses))

                        except json.JSONDecodeError:
                            continue

                    # Wait for training to complete
                    train_future.get()

                return reformatted_losses

        except Exception as e:
            if "timeout" in str(e).lower():
                raise RuntimeError("Modal training exceeded time limit. Try reducing epochs or batch size.")
            raise RuntimeError(f"Modal training failed: {str(e)}")

    def generate(self, num_samples: int, input_dim: int, hidden_dim: int) -> np.ndarray:
        """Generate synthetic samples using Modal"""
        try:
            with app.run():
                return generate_samples_remote.remote(num_samples, input_dim, hidden_dim)
        except Exception as e:
            raise RuntimeError(f"Modal generation failed: {str(e)}")