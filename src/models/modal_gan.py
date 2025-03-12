import modal
import torch
import pandas as pd
import numpy as np
from src.models.table_gan import TableGAN
import os
import wandb
import time

# Define app and shared resources at module level
app = modal.App(
    "synthetic-data-generator",
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("wandb-secret")  # Add WandB secret
    ]
)
volume = modal.Volume.from_name("gan-model-vol", create_if_missing=True)
image = modal.Image.debian_slim().pip_install(["torch", "numpy", "pandas", "wandb"])

@app.function(
    gpu="T4",
    volumes={"/model": volume},
    image=image,
    timeout=1800  # 30 minutes
)
def train_gan_remote(data: pd.DataFrame, input_dim: int, hidden_dim: int, epochs: int, batch_size: int, model_type: str):
    """Train GAN model using Modal remote execution with proper batch handling and WandB logging"""
    # Initialize wandb
    try:
        wandb.init(
            project="synthetic-data-generator",
            name=f"{model_type}-run-{int(time.time())}",
            config={
                "model_type": model_type,
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "epochs": epochs,
                "batch_size": batch_size
            }
        )
    except Exception as e:
        print(f"Warning: WandB initialization failed: {str(e)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gan = TableGAN(input_dim=input_dim, hidden_dim=hidden_dim, device=device, min_batch_size=2, use_wandb=True)

    # Convert data to tensor and optimize batch size
    train_data = torch.FloatTensor(data.values)

    # Ensure batch size is at least min_batch_size (2) and not larger than dataset
    batch_size = max(gan.min_batch_size, min(batch_size, len(train_data) // 4))

    train_loader = torch.utils.data.DataLoader(
        train_data, 
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,  # Drop last batch if incomplete
        num_workers=2
    )

    # Initialize training metrics
    best_loss = float('inf')
    patience = 5
    patience_counter = 0
    losses = []

    # Training loop with improved batch handling
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        gan.g_optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )

    total_steps = 0
    for epoch in range(epochs):
        epoch_losses = []
        for batch_idx, batch in enumerate(train_loader):
            try:
                loss_dict = gan.train_step(batch, current_step=total_steps)
                epoch_losses.append(loss_dict)
                total_steps += 1

                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}")

            except ValueError as e:
                print(f"Warning: Skipping batch due to size issue: {str(e)}")
                continue
            except Exception as e:
                print(f"Error in batch processing: {str(e)}")
                continue

        if not epoch_losses:
            print("No valid batches in epoch, stopping training")
            break

        # Calculate average loss
        avg_losses = {k: sum(d[k] for d in epoch_losses) / len(epoch_losses) 
                     for k in epoch_losses[0].keys()}
        losses.append(avg_losses)

        current_loss = sum(avg_losses.values())
        scheduler.step(current_loss)

        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0
            print(f"Epoch {epoch+1}: New best model found")
            try:
                torch.save(gan.state_dict(), "/model/table_gan.pt")
                volume.commit()
            except Exception as e:
                print(f"Warning: Failed to save model: {str(e)}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Finish WandB run
    try:
        wandb.finish()
    except Exception as e:
        print(f"Warning: Error finishing WandB run: {str(e)}")

    return losses

@app.function(
    gpu="T4",
    volumes={"/model": volume},
    image=image,
    timeout=600
)
def generate_samples_remote(num_samples: int, input_dim: int, hidden_dim: int) -> np.ndarray:
    """Generate synthetic samples using Modal remote execution"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gan = TableGAN(input_dim=input_dim, hidden_dim=hidden_dim, device=device, min_batch_size=2)

    try:
        gan.load_state_dict(torch.load("/model/table_gan.pt"))
    except Exception as e:
        raise ValueError(f"Failed to load model: {str(e)}")

    try:
        synthetic_data = gan.generate_samples(num_samples).cpu().numpy()
        return synthetic_data
    except Exception as e:
        raise RuntimeError(f"Failed to generate samples: {str(e)}")

class ModalGAN:
    """Class for managing Modal GAN operations"""

    def train(self, data: pd.DataFrame, input_dim: int, hidden_dim: int, epochs: int, batch_size: int, model_type: str = 'TableGAN'):
        """Train GAN model using Modal"""
        try:
            with app.run():
                return train_gan_remote.remote(data, input_dim, hidden_dim, epochs, batch_size, model_type)
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