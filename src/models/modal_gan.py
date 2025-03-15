"""GAN model training and generation using Modal cloud execution"""
import modal
import torch
import pandas as pd
import numpy as np
from src.models.table_gan import TableGAN
import wandb
import time
import os

# Define app and shared resources at module level
app = modal.App()
volume = modal.Volume.from_name("gan-model-vol", create_if_missing=True)
image = modal.Image.debian_slim().pip_install(["torch", "numpy", "pandas", "wandb"])

@app.function(
    gpu="T4",
    volumes={"/model": volume},
    image=image,
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=1800
)
def train_gan_remote(data: pd.DataFrame, input_dim: int, hidden_dim: int, epochs: int, batch_size: int, model_type: str):
    """Train GAN model using Modal remote execution with proper batch handling and WandB logging"""
    # Clear any existing model file first
    model_path = "/model/table_gan.pt"
    if os.path.exists(model_path):
        os.remove(model_path)
        print("Removed existing model file")

    # Initialize wandb
    wandb.init(project="sd1")
    wandb.config = {
        "model_type": model_type,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "epochs": epochs,
        "batch_size": batch_size,
        "environment": "modal-cloud"
    }

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        gan = TableGAN(input_dim=input_dim, hidden_dim=hidden_dim, device=device)

        # Convert data to tensor and optimize batch size
        train_data = torch.FloatTensor(data.values)
        batch_size = min(batch_size, len(train_data) // 4)

        train_loader = torch.utils.data.DataLoader(
            train_data, 
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=2
        )

        # Training metrics
        best_loss = float('inf')
        patience = 5
        patience_counter = 0
        total_steps = 0

        # Training loop
        for epoch in range(epochs):
            epoch_losses = []
            for batch_idx, batch in enumerate(train_loader):
                try:
                    metrics = gan.train_step(batch)
                    epoch_losses.append(metrics)
                    total_steps += 1

                    # Log metrics every few batches
                    if batch_idx % 10 == 0:
                        print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}")
                        wandb.log(metrics)

                except Exception as e:
                    print(f"Error in batch processing: {str(e)}")
                    continue

            if not epoch_losses:
                print("No valid batches in epoch, stopping training")
                break

            # Calculate and log epoch metrics
            avg_losses = {k: sum(d[k] for d in epoch_losses) / len(epoch_losses) 
                         for k in epoch_losses[0].keys()}
            current_loss = sum(avg_losses.values())

            wandb.log({
                "epoch": epoch,
                "average_loss": current_loss,
                **avg_losses
            })

            # Save best model with retries
            if current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0

                # Try saving model multiple times
                max_retries = 3
                for retry in range(max_retries):
                    try:
                        # Save model state with dimensions
                        model_save_data = {
                            'state_dict': gan.state_dict(),
                            'input_dim': input_dim,
                            'hidden_dim': hidden_dim,
                            'model_type': model_type
                        }
                        torch.save(model_save_data, model_path)

                        # Verify the save was successful
                        if os.path.exists(model_path):
                            try:
                                # Try loading the model to verify it's valid
                                checkpoint = torch.load(model_path)
                                if all(k in checkpoint for k in ['state_dict', 'input_dim', 'hidden_dim']):
                                    print(f"Successfully saved and verified model with input_dim={input_dim}, hidden_dim={hidden_dim}")
                                    volume.commit()
                                    break
                            except Exception as ve:
                                print(f"Model verification failed on attempt {retry + 1}: {str(ve)}")
                                if retry == max_retries - 1:
                                    raise
                                continue
                    except Exception as e:
                        print(f"Failed to save model on attempt {retry + 1}: {str(e)}")
                        if retry == max_retries - 1:
                            raise
                        time.sleep(1)  # Wait before retrying
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

    except Exception as e:
        print(f"Training error: {str(e)}")
        wandb.finish()
        raise e

    finally:
        wandb.finish()

    return avg_losses if 'avg_losses' in locals() else None

@app.function(
    gpu="T4",
    volumes={"/model": volume},
    image=image,
    timeout=600
)
def generate_samples_remote(num_samples: int, input_dim: int, hidden_dim: int) -> np.ndarray:
    """Generate synthetic samples using Modal remote execution"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "/model/table_gan.pt"

    try:
        # Check if model exists and is valid
        if not os.path.exists(model_path):
            raise RuntimeError("No saved model found")

        try:
            # Load and validate model
            checkpoint = torch.load(model_path)
            if not all(k in checkpoint for k in ['state_dict', 'input_dim', 'hidden_dim']):
                raise RuntimeError("Invalid model checkpoint structure")

            saved_input_dim = checkpoint['input_dim']
            saved_hidden_dim = checkpoint['hidden_dim']

            # Strict dimension checking
            if saved_input_dim != input_dim:
                raise RuntimeError(f"Input dimension mismatch: model={saved_input_dim}, requested={input_dim}")
            if saved_hidden_dim != hidden_dim:
                raise RuntimeError(f"Hidden dimension mismatch: model={saved_hidden_dim}, requested={hidden_dim}")

            # Initialize and load model
            print(f"Loading model with input_dim={input_dim}, hidden_dim={hidden_dim}")
            gan = TableGAN(input_dim=input_dim, hidden_dim=hidden_dim, device=device)
            gan.load_state_dict(checkpoint['state_dict'])
            gan.eval()  # Set to evaluation mode

            # Generate samples
            with torch.no_grad():
                synthetic_data = gan.generate_samples(num_samples).cpu().numpy()

            # Validate output
            if not (0 <= synthetic_data.all() <= 1):
                print("Warning: Generated data contains values outside [0,1] range")
                synthetic_data = np.clip(synthetic_data, 0, 1)

            return synthetic_data

        except Exception as e:
            print(f"Error loading or using model: {str(e)}")
            raise RuntimeError(f"Failed to use saved model: {str(e)}")

    except Exception as e:
        print(f"Generation error: {str(e)}")
        raise RuntimeError(f"Failed to generate samples: {str(e)}")

class ModalGAN:
    """Class for managing Modal GAN operations"""

    def __init__(self):
        self.input_dim = None
        self.hidden_dim = None

    def train(self, data: pd.DataFrame, input_dim: int, hidden_dim: int, epochs: int, batch_size: int, model_type: str = 'TableGAN'):
        """Train GAN model using Modal"""
        try:
            print(f"Starting Modal training with input_dim={input_dim}, hidden_dim={hidden_dim}")
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            with app.run():
                return train_gan_remote.remote(data, input_dim, hidden_dim, epochs, batch_size, model_type)
        except Exception as e:
            if "timeout" in str(e).lower():
                raise RuntimeError("Modal training exceeded time limit. Try reducing epochs or batch size.")
            raise RuntimeError(f"Modal training failed: {str(e)}")

    def generate(self, num_samples: int, input_dim: int, hidden_dim: int) -> np.ndarray:
        """Generate synthetic samples using Modal"""
        try:
            print(f"Generating samples with input_dim={input_dim}, hidden_dim={hidden_dim}")
            with app.run():
                return generate_samples_remote.remote(num_samples, input_dim, hidden_dim)
        except Exception as e:
            raise RuntimeError(f"Modal generation failed: {str(e)}")