import modal
import torch
import pandas as pd
import numpy as np
import os
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
def train_gan_remote(data, input_dim: int, hidden_dim: int, epochs: int, batch_size: int, model_type: str,
                    load_existing: bool = False, model_name: str = None, fine_tune: bool = False,
                    categorical_columns=None, categorical_dims=None):
    """
    Train GAN model using Modal remote execution with proper batch handling and WandB logging

    Args:
        data: The training data
        input_dim: Input dimension for the model
        hidden_dim: Hidden dimension size
        epochs: Number of training epochs
        batch_size: Batch size for training
        model_type: Type of GAN model to use ('TableGAN', 'WGAN', 'CGAN', 'TVAE', 'CTGAN')
        load_existing: Whether to load an existing model
        model_name: Name of the model file to save/load (without extension)
        fine_tune: If True, fine-tune an existing model; if False, train from scratch
        categorical_columns: List of indices for categorical columns
        categorical_dims: Dictionary mapping column indices to number of categories
    """
    # Generate a default model name if none provided
    if model_name is None:
        model_name = f"{model_type.lower()}_model"

    model_path = f"/model/{model_name}.pt"

    # Initialize wandb with more detailed config
    run_name = f"{model_type}_{time.strftime('%Y%m%d_%H%M%S')}"
    wandb.init(project="synthetic_data_generation", name=run_name)
    wandb.config = {
        "model_type": model_type,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "epochs": epochs,
        "batch_size": batch_size,
        "environment": "modal-cloud",
        "fine_tuning": fine_tune,
        "categorical_columns": categorical_columns,
        "categorical_dims": categorical_dims
    }

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Instantiate the appropriate model type
        if model_type == 'TableGAN':
            from src.models.table_gan import TableGAN
            gan = TableGAN(
                input_dim=input_dim, 
                hidden_dim=hidden_dim, 
                device=device, 
                min_batch_size=2,
                categorical_columns=categorical_columns,
                categorical_dims=categorical_dims
            )
        elif model_type == 'WGAN':
            from src.models.wgan import WGAN
            gan = WGAN(input_dim=input_dim, hidden_dim=hidden_dim, device=device)
        elif model_type == 'CGAN':
            from src.models.cgan import CGAN
            # Default condition_dim to 0 if not provided
            condition_dim = input_dim // 2  # Arbitrary default
            gan = CGAN(input_dim=input_dim, condition_dim=condition_dim, hidden_dim=hidden_dim, device=device)
        elif model_type == 'TVAE':
            from src.models.tvae import TVAE
            latent_dim = min(input_dim * 2, 128)  # Reasonable default for latent dimension
            gan = TVAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, device=device)
        elif model_type == 'CTGAN':
            from src.models.ctgan import CTGAN
            gan = CTGAN(input_dim=input_dim, hidden_dim=hidden_dim, device=device)
        else:
            # Default to TableGAN if unsupported model type
            from src.models.table_gan import TableGAN
            gan = TableGAN(input_dim=input_dim, hidden_dim=hidden_dim, device=device, min_batch_size=2)
            model_type = 'TableGAN'  # Reset to ensure consistency

        # Load existing model if requested
        if load_existing or fine_tune:
            try:
                print(f"Loading existing model from {model_path}")
                gan.load_state_dict(torch.load(model_path))
                print("Model loaded successfully")

                # Log the loading
                wandb.log({"model_loaded": True, "model_path": model_path})
            except Exception as load_error:
                print(f"Error loading model: {str(load_error)}")
                if fine_tune:
                    print("Fine-tuning requested but model couldn't be loaded. Training from scratch.")
                    wandb.log({"model_loaded": False, "fine_tune_fallback": "train_from_scratch"})
                else:
                    # Just log the error, but continue with new model
                    wandb.log({"model_loaded": False, "error": str(load_error)})

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
                    # Save model with type-specific name
                    save_path = f"/model/{model_name}.pt"
                    torch.save(gan.state_dict(), save_path)
                    print(f"Saved model to {save_path}")

                    # Also save a copy with timestamp for versioning
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    versioned_path = f"/model/{model_name}_{timestamp}.pt"
                    torch.save(gan.state_dict(), versioned_path)
                    print(f"Saved versioned model to {versioned_path}")

                    volume.commit()

                    # Log the saving event
                    wandb.log({
                        "model_saved": True,
                        "model_path": model_path,
                        "versioned_path": versioned_path,
                        "best_loss": best_loss
                    })
                except Exception as e:
                    print(f"Warning: Failed to save model: {str(e)}")
                    wandb.log({"model_saved": False, "error": str(e)})
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

    except Exception as e:
        print(f"Training error: {str(e)}")
        wandb.log({"training_error": str(e)})
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
def generate_samples_remote(num_samples: int, input_dim: int, hidden_dim: int, 
                           model_type: str = 'TableGAN', model_name: str = None,
                           temperature: float = 0.8, categorical_columns=None, 
                           categorical_dims=None) -> np.ndarray:
    """
    Generate synthetic samples using Modal remote execution

    Args:
        num_samples: Number of samples to generate
        input_dim: Input dimension for the model
        hidden_dim: Hidden dimension size
        model_type: Type of GAN model to use ('TableGAN', 'WGAN', 'CGAN', 'TVAE', 'CTGAN')
        model_name: Name of the saved model file to load (without extension)
        temperature: Temperature parameter for categorical sampling (higher = more diversity)
        categorical_columns: List of indices for categorical columns
        categorical_dims: Dictionary mapping column indices to number of categories
    """
    # Generate a default model name if none provided
    if model_name is None:
        model_name = f"{model_type.lower()}_model"

    model_path = f"/model/{model_name}.pt"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Instantiate the appropriate model type with the same logic as in training
    if model_type == 'TableGAN':
        from src.models.table_gan import TableGAN
        gan = TableGAN(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            device=device, 
            min_batch_size=2,
            categorical_columns=categorical_columns,
            categorical_dims=categorical_dims
        )
    elif model_type == 'WGAN':
        from src.models.wgan import WGAN
        gan = WGAN(input_dim=input_dim, hidden_dim=hidden_dim, device=device)
    elif model_type == 'CGAN':
        from src.models.cgan import CGAN
        # Default condition_dim to 0 if not provided
        condition_dim = input_dim // 2
        gan = CGAN(input_dim=input_dim, condition_dim=condition_dim, hidden_dim=hidden_dim, device=device)
    elif model_type == 'TVAE':
        from src.models.tvae import TVAE
        latent_dim = min(input_dim * 2, 128)
        gan = TVAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, device=device)
    elif model_type == 'CTGAN':
        from src.models.ctgan import CTGAN
        gan = CTGAN(input_dim=input_dim, hidden_dim=hidden_dim, device=device)
    else:
        # Default to TableGAN if unsupported model type
        from src.models.table_gan import TableGAN
        gan = TableGAN(input_dim=input_dim, hidden_dim=hidden_dim, device=device, min_batch_size=2)

    try:
        print(f"Loading model from {model_path}")
        gan.load_state_dict(torch.load(model_path))
        print("Model loaded successfully")

        # Generate samples with temperature parameter (for models that support it)
        if hasattr(gan, 'generate_samples') and 'temperature' in gan.generate_samples.__code__.co_varnames:
            print(f"Generating samples with temperature={temperature}")
            synthetic_data = gan.generate_samples(num_samples, temperature=temperature).cpu().detach().numpy()
        else:
            print(f"Generating samples (model doesn't support temperature parameter)")
            synthetic_data = gan.generate_samples(num_samples).cpu().detach().numpy()

        print(f"Generated {len(synthetic_data)} samples")
        return synthetic_data
    except Exception as e:
        print(f"Error generating samples: {str(e)}")
        raise RuntimeError(f"Failed to generate samples: {str(e)}")

class ModalGAN:
    """Class for managing Modal GAN operations with model persistence"""

    def train(self, data: pd.DataFrame, input_dim: int, hidden_dim: int, epochs: int, batch_size: int, 
              model_type: str = 'TableGAN', load_existing: bool = False, model_name: str = None, 
              fine_tune: bool = False, categorical_columns=None, categorical_dims=None):
        """
        Train GAN model using Modal with option to load existing model and fine-tune

        Args:
            data: Training data as DataFrame
            input_dim: Input dimension for the model
            hidden_dim: Hidden dimension size
            epochs: Number of training epochs
            batch_size: Batch size for training
            model_type: Type of GAN model to use ('TableGAN', 'WGAN', 'CGAN', 'TVAE', 'CTGAN')
            load_existing: Whether to load an existing model
            model_name: Name of the model file to save/load (without extension)
            fine_tune: If True, fine-tune an existing model; if False, train from scratch
            categorical_columns: List of indices for categorical columns
            categorical_dims: Dictionary mapping column indices to number of categories
        """
        try:
            # Convert data to a list of dictionaries for better serialization
            serializable_data = data.reset_index(drop=True).to_dict('records')

            # Generate a default model name if none provided
            if model_name is None:
                model_name = f"{model_type.lower()}_model"

            print(f"Training {model_type} model with Modal {'(fine-tuning)' if fine_tune else ''}")
            print(f"Model will be saved as {model_name}.pt")

            if categorical_columns:
                print(f"Using categorical columns: {categorical_columns}")
                print(f"Categorical dimensions: {categorical_dims}")

            with app.run():
                # Inside Modal, we'll convert back to DataFrame and pass all parameters
                return train_gan_remote.remote(
                    serializable_data, 
                    input_dim, 
                    hidden_dim, 
                    epochs, 
                    batch_size, 
                    model_type,
                    load_existing,
                    model_name,
                    fine_tune,
                    categorical_columns,
                    categorical_dims
                )
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
                wandb.init(project="synthetic_data_generation")
                wandb.config = {
                    "model_type": model_type,
                    "input_dim": input_dim,
                    "hidden_dim": hidden_dim,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "environment": "local-fallback",
                    "fine_tune": fine_tune,
                    "categorical_columns": categorical_columns,
                    "categorical_dims": categorical_dims
                }

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                # Instantiate the right model with categorical support if needed
                if model_type == 'TableGAN':
                    from src.models.table_gan import TableGAN
                    gan = TableGAN(
                        input_dim=input_dim, 
                        hidden_dim=hidden_dim, 
                        device=device,
                        categorical_columns=categorical_columns,
                        categorical_dims=categorical_dims
                    )
                elif model_type == 'WGAN':
                    from src.models.wgan import WGAN
                    gan = WGAN(input_dim=input_dim, hidden_dim=hidden_dim, device=device)
                elif model_type == 'CGAN':
                    from src.models.cgan import CGAN
                    condition_dim = input_dim // 2
                    gan = CGAN(input_dim=input_dim, condition_dim=condition_dim, hidden_dim=hidden_dim, device=device)
                elif model_type == 'TVAE':
                    from src.models.tvae import TVAE
                    latent_dim = min(input_dim * 2, 128)
                    gan = TVAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, device=device)
                elif model_type == 'CTGAN':
                    from src.models.ctgan import CTGAN
                    gan = CTGAN(input_dim=input_dim, hidden_dim=hidden_dim, device=device)
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

                # Save locally as fallback
                import os
                os.makedirs('models', exist_ok=True)
                local_path = f"models/{model_name}.pt"
                torch.save(gan.state_dict(), local_path)
                print(f"Model saved locally to {local_path}")

                wandb.finish()
                return losses[-1] if losses else None

            except Exception as inner_e:
                raise RuntimeError(f"Both remote and local training failed: {str(e)} -> {str(inner_e)}")

    def generate(self, num_samples: int, input_dim: int, hidden_dim: int, model_type: str = 'TableGAN', 
                model_name: str = None, temperature: float = 0.8, categorical_columns=None, 
                categorical_dims=None) -> np.ndarray:
        """
        Generate synthetic samples using Modal with specific model and parameters

        Args:
            num_samples: Number of samples to generate
            input_dim: Input dimension for the model
            hidden_dim: Hidden dimension size
            model_type: Type of GAN model to use ('TableGAN', 'WGAN', 'CGAN', 'TVAE', 'CTGAN')
            model_name: Name of the model file to load (without extension)
            temperature: Temperature parameter for categorical sampling (higher = more diversity)
            categorical_columns: List of indices for categorical columns
            categorical_dims: Dictionary mapping column indices to number of categories
        """
        # Generate a default model name if none provided
        if model_name is None:
            model_name = f"{model_type.lower()}_model"

        print(f"Generating {num_samples} samples using {model_type} model ({model_name})")
        if categorical_columns:
            print(f"Using temperature={temperature} for categorical sampling")

        try:
            with app.run():
                # Get result with extended parameters
                result = generate_samples_remote.remote(
                    num_samples, 
                    input_dim, 
                    hidden_dim,
                    model_type,
                    model_name,
                    temperature,
                    categorical_columns,
                    categorical_dims
                )
                # Ensure we have a numpy array (Modal might convert it to a list)
                if not isinstance(result, np.ndarray) and isinstance(result, list):
                    result = np.array(result)
                return result
        except Exception as e:
            # Fall back to local generation if Modal fails
            print(f"Modal generation failed: {str(e)}")
            print("Falling back to local generation...")
            try:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                # Instantiate the right model with the same logic as in training
                if model_type == 'TableGAN':
                    from src.models.table_gan import TableGAN
                    gan = TableGAN(
                        input_dim=input_dim, 
                        hidden_dim=hidden_dim, 
                        device=device,
                        categorical_columns=categorical_columns,
                        categorical_dims=categorical_dims
                    )
                elif model_type == 'WGAN':
                    from src.models.wgan import WGAN
                    gan = WGAN(input_dim=input_dim, hidden_dim=hidden_dim, device=device)
                elif model_type == 'CGAN':
                    from src.models.cgan import CGAN
                    condition_dim = input_dim // 2
                    gan = CGAN(input_dim=input_dim, condition_dim=condition_dim, hidden_dim=hidden_dim, device=device)
                elif model_type == 'TVAE':
                    from src.models.tvae import TVAE
                    latent_dim = min(input_dim * 2, 128)
                    gan = TVAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, device=device)
                elif model_type == 'CTGAN':
                    from src.models.ctgan import CTGAN
                    gan = CTGAN(input_dim=input_dim, hidden_dim=hidden_dim, device=device)
                else:
                    # Default to TableGAN if unsupported model type
                    from src.models.table_gan import TableGAN
                    gan = TableGAN(input_dim=input_dim, hidden_dim=hidden_dim, device=device)

                # Try to load local model
                local_path = f"models/{model_name}.pt"
                if os.path.exists(local_path):
                    print(f"Loading local model from {local_path}")
                    gan.load_state_dict(torch.load(local_path))

                # Generate samples with temperature parameter if supported
                if hasattr(gan, 'generate_samples') and 'temperature' in gan.generate_samples.__code__.co_varnames:
                    return gan.generate_samples(num_samples, temperature=temperature).cpu().detach().numpy()
                else:
                    return gan.generate_samples(num_samples).cpu().detach().numpy()

            except Exception as inner_e:
                raise RuntimeError(f"Both remote and local generation failed: {str(e)} -> {str(inner_e)}")

    def list_available_models(self):
        """List all available models saved in the Modal volume"""
        try:
            with app.run():
                @app.function(volumes={"/model": volume})
                def list_models():
                    import os
                    model_files = [f for f in os.listdir("/model") if f.endswith(".pt")]
                    model_info = []
                    for model_file in model_files:
                        # Get file size and modification time
                        stats = os.stat(f"/model/{model_file}")
                        size_mb = stats.st_size / (1024 * 1024)
                        mod_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stats.st_mtime))

                        # Extract model type from filename
                        model_type = "unknown"
                        for t in ['tablegan', 'wgan', 'cgan', 'tvae', 'ctgan']:
                            if t in model_file.lower():
                                model_type = t.upper()
                                break

                        model_info.append({
                            "filename": model_file,
                            "model_type": model_type,
                            "size_mb": round(size_mb, 2),
                            "last_modified": mod_time,
                        })

                    return model_info

                return list_models.remote()

        except Exception as e:
            print(f"Failed to list models: {str(e)}")
            # Try to list local models as fallback
            try:
                import os
                if os.path.exists("models"):
                    model_files = [f for f in os.listdir("models") if f.endswith(".pt")]
                    return [{"filename": f, "location": "local"} for f in model_files]
                else:
                    return []
            except:
                return []

    def delete_model(self, model_name: str):
        """Delete a model from the Modal volume"""
        if not model_name.endswith(".pt"):
            model_name += ".pt"

        try:
            with app.run():
                @app.function(volumes={"/model": volume})
                def delete_model_file(filename):
                    import os
                    file_path = f"/model/{filename}"
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        volume.commit()
                        return True
                    return False

                result = delete_model_file.remote(model_name)
                if result:
                    print(f"Successfully deleted model {model_name}")
                else:
                    print(f"Model {model_name} not found")
                return result

        except Exception as e:
            print(f"Failed to delete model: {str(e)}")
            return False