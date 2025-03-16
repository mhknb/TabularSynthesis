import modal
import torch
import pandas as pd
import numpy as np
from src.models.table_gan import TableGAN
import wandb
import time

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
def train_gan_remote(data, input_dim: int, hidden_dim: int, epochs: int, batch_size: int, model_type: str):
    """Train GAN model using Modal remote execution with proper batch handling and WandB logging"""
    # Initialize wandb first, following the example pattern
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
        gan = TableGAN(input_dim=input_dim, hidden_dim=hidden_dim, device=device, min_batch_size=2)

        # Convert data from dict format back to DataFrame if needed
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict):
                data = pd.DataFrame(data)
            else:
                # Handle case where data is a list of lists
                data = pd.DataFrame(data)
        elif not isinstance(data, pd.DataFrame):
            print(f"Warning: Unexpected data type: {type(data)}")
            data = pd.DataFrame(data)
        
        # Convert data to tensor and optimize batch size
        train_data = torch.FloatTensor(data.values)
        batch_size = max(gan.min_batch_size, min(batch_size, len(train_data) // 4))

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

            # Save best model
            if current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0
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

    except Exception as e:
        print(f"Training error: {str(e)}")
        wandb.finish()
        raise e

    finally:
        # Always ensure wandb run is finished
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
    gan = TableGAN(input_dim=input_dim, hidden_dim=hidden_dim, device=device, min_batch_size=2)

    try:
        gan.load_state_dict(torch.load("/model/table_gan.pt"))
        synthetic_data = gan.generate_samples(num_samples).cpu().numpy()
        return synthetic_data
    except Exception as e:
        raise RuntimeError(f"Failed to generate samples: {str(e)}")

class ModalGAN:
    """Class for managing Modal GAN operations"""

    def train(self, data: pd.DataFrame, input_dim: int, hidden_dim: int, epochs: int, batch_size: int, model_type: str = 'TableGAN'):
        """Train GAN model using Modal"""
        try:
            # Convert data to a list of dictionaries for better serialization
            serializable_data = data.reset_index(drop=True).to_dict('records')
            
            with app.run():
                # Inside Modal, we'll convert back to DataFrame
                return train_gan_remote.remote(serializable_data, input_dim, hidden_dim, epochs, batch_size, model_type)
        except Exception as e:
            if "timeout" in str(e).lower():
                print("Modal training exceeded time limit. Try reducing epochs or batch size.")
                print("Falling back to local training with reduced epochs...")
                epochs = max(5, epochs // 4)  # Reduce epochs for local training
            else:
                print(f"Modal training failed: {str(e)}")
                print("Falling back to local training...")
            
            # Attempt local training as fallback
            try:
                # Initialize wandb
                import wandb
                wandb.init(project="sd1")
                wandb.config = {
                    "model_type": model_type,
                    "input_dim": input_dim,
                    "hidden_dim": hidden_dim,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "environment": "local-fallback"
                }
                
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                # Instantiate the right model
                if model_type == 'TableGAN':
                    from src.models.table_gan import TableGAN
                    gan = TableGAN(input_dim=input_dim, hidden_dim=hidden_dim, device=device)
                elif model_type == 'WGAN':
                    from src.models.wgan import WGAN
                    gan = WGAN(input_dim=input_dim, hidden_dim=hidden_dim, device=device)
                else:
                    # Default to TableGAN for unsupported model types
                    from src.models.table_gan import TableGAN
                    gan = TableGAN(input_dim=input_dim, hidden_dim=hidden_dim, device=device)
                
                # Create data loader
                train_data = torch.FloatTensor(data.values)
                batch_size = max(2, min(batch_size, len(train_data) // 4))
                train_loader = torch.utils.data.DataLoader(
                    train_data, 
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=True,
                    num_workers=0  # Use 0 for local training to avoid potential issues
                )
                
                # Training
                losses = gan.train(train_loader, epochs)
                wandb.finish()
                return losses[-1] if losses else None
                
            except Exception as inner_e:
                raise RuntimeError(f"Both remote and local training failed: {str(e)} -> {str(inner_e)}")

    def generate(self, num_samples: int, input_dim: int, hidden_dim: int) -> np.ndarray:
        """Generate synthetic samples using Modal"""
        try:
            with app.run():
                # Get result and convert if needed (numpy arrays serialize differently through Modal)
                result = generate_samples_remote.remote(num_samples, input_dim, hidden_dim)
                # Ensure we have a numpy array (Modal might convert it to a list)
                if not isinstance(result, np.ndarray) and isinstance(result, list):
                    result = np.array(result)
                return result
        except Exception as e:
            # Fall back to local generation if Modal fails
            print(f"Modal generation failed: {str(e)}")
            print("Falling back to local training...")
            try:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                gan = TableGAN(input_dim=input_dim, hidden_dim=hidden_dim, device=device)
                return gan.generate_samples(num_samples).cpu().numpy()
            except Exception as inner_e:
                raise RuntimeError(f"Both remote and local generation failed: {str(e)} -> {str(inner_e)}")