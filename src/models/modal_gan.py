import modal
import torch
import pandas as pd
import numpy as np
from src.models.table_gan import TableGAN

# Define app and shared resources at module level
app = modal.App("synthetic-data-generator")

# Create volume with create_if_missing flag
volume = modal.Volume.from_name("gan-model-vol", create_if_missing=True)

# Create Modal image with required dependencies
image = modal.Image.debian_slim().pip_install(["torch", "numpy", "pandas"])

@app.function(
    gpu="T4",
    volumes={"/model": volume},
    image=image,
    timeout=1800  # Increase timeout to 30 minutes
)
def train_gan_remote(data: pd.DataFrame, input_dim: int, hidden_dim: int, epochs: int, batch_size: int):
    """Train GAN model using Modal remote execution"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gan = TableGAN(input_dim=input_dim, hidden_dim=hidden_dim, device=device)

    # Convert data to tensor and optimize batch size
    train_data = torch.FloatTensor(data.values)
    batch_size = min(batch_size, len(train_data) // 4)  # Ensure reasonable batch size
    train_loader = torch.utils.data.DataLoader(
        train_data, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=2  # Parallelize data loading
    )

    # Initialize training metrics
    best_loss = float('inf')
    patience = 5
    patience_counter = 0
    all_losses = []  # Store all losses for return

    # Training loop with improved early stopping and adaptive learning rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        gan.g_optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )

    # Initialize metrics tracking
    best_fid_score = float('inf')  # Using loss as a proxy for FID score
    min_delta = 0.001  # Minimum improvement required

    for epoch in range(epochs):
        epoch_losses = []
        batch_generator_loss = 0
        batch_discriminator_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            loss_dict = gan.train_step(batch)
            epoch_losses.append(loss_dict)

            # Accumulate batch losses
            batch_generator_loss += loss_dict['generator_loss']
            batch_discriminator_loss += loss_dict.get('discriminator_loss', loss_dict.get('critic_loss', 0.0))
            num_batches += 1

            # Print progress every 10 batches
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}")

            # Add gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(gan.generator.parameters(), max_norm=1.0)
            if hasattr(gan, 'discriminator'):
                torch.nn.utils.clip_grad_norm_(gan.discriminator.parameters(), max_norm=1.0)
            elif hasattr(gan, 'critic'):
                torch.nn.utils.clip_grad_norm_(gan.critic.parameters(), max_norm=1.0)

        # Calculate average losses for the epoch
        avg_generator_loss = batch_generator_loss / num_batches
        avg_discriminator_loss = batch_discriminator_loss / num_batches

        # Store epoch results
        epoch_result = {
            'epoch': epoch,
            'generator_loss': avg_generator_loss,
            'discriminator_loss': avg_discriminator_loss
        }
        all_losses.append(epoch_result)

        # Calculate proxy metric for quality
        current_loss = avg_generator_loss + avg_discriminator_loss

        # Update learning rate based on performance
        scheduler.step(current_loss)

        # Check improvement
        is_improvement = (best_fid_score - current_loss) > min_delta

        if is_improvement:
            best_fid_score = current_loss
            best_loss = current_loss
            patience_counter = 0
            print(f"Epoch {epoch+1}: Improved model quality, saving model")
            # Save best model
            try:
                torch.save(gan.state_dict(), "/model/table_gan.pt")
                volume.commit()
            except Exception as e:
                print(f"Warning: Failed to save model: {str(e)}")
        else:
            patience_counter += 1
            print(f"Epoch {epoch+1}: No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Generate sample data every few epochs to check quality visually
        if epoch % 5 == 0 and epoch > 0:
            print("Generating sample data for quality check...")
            with torch.no_grad():
                sample_noise = torch.randn(min(10, batch_size), gan.input_dim).to(device)
                sample_data = gan.generator(sample_noise).cpu().numpy()
                print(f"Sample data range: {sample_data.min()} to {sample_data.max()}")

    return all_losses  # Return all losses for visualization

@app.function(
    gpu="T4",
    volumes={"/model": volume},
    image=image,
    timeout=600  # 10 minutes timeout for generation
)
def generate_samples_remote(num_samples: int, input_dim: int, hidden_dim: int) -> np.ndarray:
    """Generate synthetic samples using Modal remote execution"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gan = TableGAN(input_dim=input_dim, hidden_dim=hidden_dim, device=device)

    try:
        gan.load_state_dict(torch.load("/model/table_gan.pt"))
    except Exception as e:
        raise ValueError(f"Failed to load model: {str(e)}")

    # Generate in batches to avoid memory issues
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
        """Train GAN model using Modal"""
        try:
            with app.run():
                all_losses = train_gan_remote.remote(data, input_dim, hidden_dim, epochs, batch_size)

                # Convert list of loss dictionaries to format expected by training_progress
                reformatted_losses = []
                for loss_dict in all_losses:
                    epoch = loss_dict['epoch']
                    # Create loss dictionary in the format expected by training_progress
                    training_loss = {
                        'generator_loss': loss_dict.get('generator_loss', 0.0),
                        'discriminator_loss': loss_dict.get('discriminator_loss', 
                                                         loss_dict.get('critic_loss', 0.0))
                    }
                    reformatted_losses.append((epoch, training_loss))

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