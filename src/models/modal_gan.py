import modal
import torch
import pandas as pd
import numpy as np
import json
from src.models.table_gan import TableGAN
import wandb
import time

# Define app and shared resources at module level
app = modal.App()
volume = modal.Volume.from_name("gan-model-vol", create_if_missing=True)
# Create base image with required dependencies and proper sequence for local modules
# First install all dependencies
image = modal.Image.debian_slim().pip_install(["torch", "numpy", "pandas", "wandb", "tarsafe"])
# Add the src directory to make models accessible from remote functions with copy=True
image = image.add_local_dir("./src", "/root/src", copy=True)

# Add helper functions to handle numpy serialization
def serialize_numpy(data):
    """Serialize numpy array to list to avoid numpy._core serialization issues"""
    if isinstance(data, np.ndarray):
        return data.tolist()
    return data

def deserialize_numpy(data):
    """Convert list back to numpy array"""
    if isinstance(data, list):
        return np.array(data)
    return data

# Create a final image with all dependencies properly prepared
final_image = image.pip_install("tarsafe")

@app.function(
    gpu="T4",
    volumes={"/model": volume},
    image=final_image,
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=1800
)
def train_gan_remote(data_list, input_dim: int, hidden_dim: int, epochs: int, batch_size: int, model_type: str):
    """Train GAN model using Modal remote execution with proper batch handling and WandB logging"""
    # Import all necessary libraries first
    import pandas as pd
    import numpy as np
    import torch
    import wandb
    import sys
    
    # Add paths to system path to ensure modules can be found
    sys.path.append('/root')
    sys.path.append('/root/src')
    
    try:
        # Try importing directly
        from src.models.table_gan import TableGAN
    except ImportError:
        # Fall back to importing from models directly
        try:
            from models.table_gan import TableGAN
        except ImportError:
            # Try one more approach
            print("Attempting to find TableGAN in alternate locations...")
            import os
            print(f"Current directory: {os.getcwd()}")
            print(f"Directory contents: {os.listdir('.')}")
            if os.path.exists('/root/src'):
                print(f"/root/src contents: {os.listdir('/root/src')}")
                if os.path.exists('/root/src/models'):
                    print(f"/root/src/models contents: {os.listdir('/root/src/models')}")
            raise
    
    # Convert data_list back to dataframe if needed
    if isinstance(data_list, list):
        data = pd.DataFrame(data_list)
    else:
        data = data_list
        
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
    image=final_image,
    timeout=600
)
def generate_samples_remote(num_samples: int, input_dim: int, hidden_dim: int):
    """Generate synthetic samples using Modal remote execution"""
    # Import all necessary libraries first
    import torch
    import numpy as np
    import sys
    
    # Add paths to system path to ensure modules can be found
    sys.path.append('/root')
    sys.path.append('/root/src')
    
    try:
        # Try importing directly
        from src.models.table_gan import TableGAN
    except ImportError:
        # Fall back to importing from models directly
        try:
            from models.table_gan import TableGAN
        except ImportError:
            # Try one more approach
            print("Attempting to find TableGAN in alternate locations...")
            import os
            print(f"Current directory: {os.getcwd()}")
            print(f"Directory contents: {os.listdir('.')}")
            if os.path.exists('/root/src'):
                print(f"/root/src contents: {os.listdir('/root/src')}")
                if os.path.exists('/root/src/models'):
                    print(f"/root/src/models contents: {os.listdir('/root/src/models')}")
            raise
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gan = TableGAN(input_dim=input_dim, hidden_dim=hidden_dim, device=device, min_batch_size=2)

    try:
        gan.load_state_dict(torch.load("/model/table_gan.pt"))
        synthetic_data = gan.generate_samples(num_samples).cpu().numpy()
        # Serialize numpy array to list to avoid serialization issues
        return serialize_numpy(synthetic_data)
    except Exception as e:
        import traceback
        print(f"Failed to generate samples: {str(e)}")
        print(traceback.format_exc())
        raise RuntimeError(f"Failed to generate samples: {str(e)}")

class ModalGAN:
    """Class for managing Modal GAN operations"""

    def train(self, data: pd.DataFrame, input_dim: int, hidden_dim: int, epochs: int, batch_size: int, model_type: str = 'TableGAN'):
        """Train GAN model using Modal"""
        try:
            # Serialize the DataFrame to avoid numpy._core serialization issues
            serialized_data = serialize_numpy(data.values.tolist())
            
            with app.run():
                result = train_gan_remote.remote(serialized_data, input_dim, hidden_dim, epochs, batch_size, model_type)
                return result
        except Exception as e:
            import traceback
            if "timeout" in str(e).lower():
                print("Modal training exceeded time limit. Trying with reduced parameters.")
                raise RuntimeError("Modal training exceeded time limit. Try reducing epochs or batch size.")
            print(f"Modal training failed: {str(e)}")
            print(traceback.format_exc())
            raise RuntimeError(f"Modal training failed: {str(e)}")

    def generate(self, num_samples: int, input_dim: int, hidden_dim: int) -> np.ndarray:
        """Generate synthetic samples using Modal"""
        try:
            with app.run():
                serialized_data = generate_samples_remote.remote(num_samples, input_dim, hidden_dim)
                # Convert the serialized data back to numpy array
                return deserialize_numpy(serialized_data)
        except Exception as e:
            print(f"Modal generation failed: {str(e)}")
            # Fall back to local generation if modal fails
            print("Falling back to local training...")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Make sure to import TableGAN properly
            from src.models.table_gan import TableGAN
            
            gan = TableGAN(input_dim=input_dim, hidden_dim=hidden_dim, device=device)
            # Generate data locally
            synthetic_data = gan.generate_samples(num_samples).cpu().numpy()
            return synthetic_data