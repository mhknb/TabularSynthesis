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
volume = modal.Volume.persisted("gan-model-storage")
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
    # Create model directory structure
    model_dir = "/model/gan_models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "latest_model.pt")
    print(f"Using model directory: {model_dir}")
    print(f"Model will be saved to: {model_path}")

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

        # Training loop
        for epoch in range(epochs):
            epoch_losses = []
            for batch_idx, batch in enumerate(train_loader):
                try:
                    metrics = gan.train_step(batch)
                    epoch_losses.append(metrics)

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

            # Log to wandb
            wandb.log({
                "epoch": epoch,
                "average_loss": current_loss,
                **avg_losses
            })

            # Save model if improved
            if current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0

                # Save model state
                try:
                    print(f"Saving improved model (loss: {current_loss:.4f})")

                    # Prepare model data
                    model_data = {
                        'state_dict': gan.state_dict(),
                        'input_dim': input_dim,
                        'hidden_dim': hidden_dim,
                        'model_type': model_type,
                        'epoch': epoch,
                        'loss': current_loss
                    }

                    # Save to temporary file first
                    temp_path = os.path.join(model_dir, "temp_model.pt")
                    torch.save(model_data, temp_path)
                    print(f"Saved temporary model to {temp_path}")

                    # Verify the save was successful
                    try:
                        # Test load the model
                        test_load = torch.load(temp_path)
                        if all(k in test_load for k in ['state_dict', 'input_dim', 'hidden_dim']):
                            # If verification successful, move to final location
                            os.replace(temp_path, model_path)
                            print(f"Successfully saved and verified model at epoch {epoch+1}")
                            volume.commit()  # Commit changes to volume
                            print("Volume changes committed")
                        else:
                            raise ValueError("Saved model file is incomplete")
                    except Exception as ve:
                        print(f"Model verification failed: {str(ve)}")
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        raise
                except Exception as e:
                    print(f"Failed to save model: {str(e)}")
                    raise
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

    # Final verification
    if os.path.exists(model_path):
        print(f"Training completed, model saved at: {model_path}")
        file_size = os.path.getsize(model_path)
        print(f"Model file size: {file_size} bytes")
    else:
        print("Warning: Model file not found after training")

    return avg_losses if 'avg_losses' in locals() else None

@app.function(
    gpu="T4",
    volumes={"/model": volume},
    image=image,
    timeout=600
)
def generate_samples_remote(num_samples: int, input_dim: int, hidden_dim: int) -> np.ndarray:
    """Generate synthetic samples using Modal remote execution"""
    model_dir = "/model/gan_models"
    model_path = os.path.join(model_dir, "latest_model.pt")
    print(f"Looking for model at: {model_path}")

    try:
        # Verify model directory exists
        if not os.path.exists(model_dir):
            raise RuntimeError(f"Model directory not found: {model_dir}")

        # Check if model exists
        if not os.path.exists(model_path):
            raise RuntimeError(f"No saved model found at {model_path}")

        print(f"Found model file (size: {os.path.getsize(model_path)} bytes)")

        # Set up device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Load and verify model
        print("Loading model...")
        checkpoint = torch.load(model_path)

        # Verify model contents
        for key in ['state_dict', 'input_dim', 'hidden_dim']:
            if key not in checkpoint:
                raise ValueError(f"Missing key in model checkpoint: {key}")

        # Verify dimensions
        if checkpoint['input_dim'] != input_dim:
            raise ValueError(f"Input dimension mismatch: saved={checkpoint['input_dim']}, requested={input_dim}")
        if checkpoint['hidden_dim'] != hidden_dim:
            raise ValueError(f"Hidden dimension mismatch: saved={checkpoint['hidden_dim']}, requested={hidden_dim}")

        # Initialize model
        print(f"Initializing model with input_dim={input_dim}, hidden_dim={hidden_dim}")
        gan = TableGAN(input_dim=input_dim, hidden_dim=hidden_dim, device=device)
        gan.load_state_dict(checkpoint['state_dict'])
        gan.eval()

        # Generate samples
        print(f"Generating {num_samples} samples...")
        with torch.no_grad():
            synthetic_data = gan.generate_samples(num_samples).cpu().numpy()

        # Validate output
        print("Validating generated data...")
        if not np.all((synthetic_data >= 0) & (synthetic_data <= 1)):
            print("Clipping generated data to [0,1] range")
            synthetic_data = np.clip(synthetic_data, 0, 1)

        return synthetic_data

    except Exception as e:
        error_msg = f"Failed to generate samples: {str(e)}"
        print(error_msg)
        raise RuntimeError(error_msg)

class ModalGAN:
    """Class for managing Modal GAN operations"""

    def __init__(self):
        self.input_dim = None
        self.hidden_dim = None

    def train(self, data: pd.DataFrame, input_dim: int, hidden_dim: int, epochs: int, batch_size: int, model_type: str = 'TableGAN'):
        """Train GAN model using Modal"""
        try:
            print(f"Starting Modal training with input_dim={input_dim}")
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