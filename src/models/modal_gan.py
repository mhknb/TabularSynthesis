
import modal
import torch
import pandas as pd
import numpy as np
from src.models.table_gan import TableGAN
import torch.nn as nn

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
def train_gan_remote(data, input_dim, hidden_dim, epochs, batch_size):
    """Remote function to train GAN model on Modal"""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import json
    import os
    
    # Convert data to tensor
    train_data = torch.FloatTensor(data.values)
    
    # Define generator based on TableGAN architecture
    generator = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        
        nn.Linear(hidden_dim, hidden_dim * 2),
        nn.BatchNorm1d(hidden_dim * 2),
        nn.ReLU(),
        
        nn.Linear(hidden_dim * 2, hidden_dim * 2),
        nn.BatchNorm1d(hidden_dim * 2),
        nn.ReLU(),
        
        nn.Linear(hidden_dim * 2, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        
        nn.Linear(hidden_dim, input_dim),
        nn.Tanh()  # Output in range [-1, 1]
    )
    
    # Define discriminator
    discriminator = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.3),
        
        nn.Linear(hidden_dim, hidden_dim * 2),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.3),

        nn.Linear(hidden_dim * 2, hidden_dim),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.3),

        nn.Linear(hidden_dim, 1)
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator.to(device)
    discriminator.to(device)
    
    # Optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # Loss
    criterion = nn.BCEWithLogitsLoss()
    
    # Training
    losses = []
    
    # Save dimensions for future use
    dims = {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim
    }
    with open('/model/dims.json', 'w') as f:
        json.dump(dims, f)
    
    for epoch in range(epochs):
        g_epoch_loss = 0
        d_epoch_loss = 0
        
        # Shuffle data
        indices = torch.randperm(train_data.size(0))
        batches = [indices[i:i+batch_size] for i in range(0, len(indices), batch_size)]
        
        for batch_idx in batches:
            # Get real batch
            real_batch = train_data[batch_idx].to(device)
            batch_size = real_batch.size(0)
            
            # Train discriminator
            d_optimizer.zero_grad()
            
            # Real data
            real_labels = torch.ones(batch_size, 1).to(device)
            output_real = discriminator(real_batch)
            d_loss_real = criterion(output_real, real_labels)
            
            # Fake data
            noise = torch.randn(batch_size, input_dim).to(device)
            fake_data = generator(noise)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            output_fake = discriminator(fake_data.detach())
            d_loss_fake = criterion(output_fake, fake_labels)
            
            # Combine losses
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            
            # Train generator
            g_optimizer.zero_grad()
            output_fake = discriminator(fake_data)
            g_loss = criterion(output_fake, real_labels)  # We want the discriminator to think these are real
            g_loss.backward()
            g_optimizer.step()
            
            g_epoch_loss += g_loss.item() * batch_size
            d_epoch_loss += d_loss.item() * batch_size
        
        g_epoch_loss /= len(train_data)
        d_epoch_loss /= len(train_data)
        losses.append((g_epoch_loss, d_epoch_loss))
        print(f"Epoch {epoch+1}/{epochs}, G Loss: {g_epoch_loss:.4f}, D Loss: {d_epoch_loss:.4f}")
    
    # Save the model
    torch.save(generator.state_dict(), '/model/generator.pth')
    torch.save(discriminator.state_dict(), '/model/discriminator.pth')
    
    return losses

@app.function(
    gpu="T4",
    volumes={"/model": volume},
    image=image
)
def generate_samples_remote(num_samples, input_dim=None, hidden_dim=None):
    """Remote function to generate samples using trained GAN"""
    import torch
    import torch.nn as nn
    import json
    import os
    
    # If dimensions not provided, load from saved file
    if input_dim is None or hidden_dim is None:
        try:
            with open('/model/dims.json', 'r') as f:
                dims = json.load(f)
                input_dim = dims.get('input_dim')
                hidden_dim = dims.get('hidden_dim')
                
            if input_dim is None or hidden_dim is None:
                raise ValueError("Could not load dimensions from saved file")
        except Exception as e:
            raise RuntimeError(f"Failed to load model dimensions: {str(e)}")
    
    # Define generator with same architecture as training
    generator = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        
        nn.Linear(hidden_dim, hidden_dim * 2),
        nn.BatchNorm1d(hidden_dim * 2),
        nn.ReLU(),
        
        nn.Linear(hidden_dim * 2, hidden_dim * 2),
        nn.BatchNorm1d(hidden_dim * 2),
        nn.ReLU(),
        
        nn.Linear(hidden_dim * 2, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        
        nn.Linear(hidden_dim, input_dim),
        nn.Tanh()
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator.to(device)
    
    # Load the model
    try:
        generator.load_state_dict(torch.load('/model/generator.pth'))
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")
    
    generator.eval()
    
    with torch.no_grad():
        noise = torch.randn(num_samples, input_dim).to(device)
        fake_data = generator(noise).cpu().numpy()
    
    return fake_data

class ModalGAN:
    """Class for managing Modal GAN operations"""

    def __init__(self):
        """Initialize Modal GAN interface"""
        # Store dimensions used in training for consistency
        self.input_dim = None
        self.hidden_dim = None

    def train(self, data: pd.DataFrame, input_dim: int, hidden_dim: int, epochs: int, batch_size: int):
        """Train GAN model using Modal"""
        try:
            with app.run():
                self.input_dim = input_dim #Store input dim
                self.hidden_dim = hidden_dim #Store hidden dim
                return train_gan_remote.remote(data, input_dim, hidden_dim, epochs, batch_size)
        except Exception as e:
            if "timeout" in str(e).lower():
                raise RuntimeError("Modal training exceeded time limit. Try reducing epochs or batch size.")
            raise RuntimeError(f"Modal training failed: {str(e)}")

    def generate(self, num_samples: int, input_dim: int =None, hidden_dim: int =None) -> np.ndarray:
        """Generate synthetic samples using Modal"""
        # Use stored dimensions if not provided
        input_dim = input_dim if input_dim is not None else self.input_dim
        hidden_dim = hidden_dim if hidden_dim is not None else self.hidden_dim

        if input_dim is None or hidden_dim is None:
            raise ValueError("Input and hidden dimensions must be specified or trained model must be loaded.")

        try:
            with app.run():
                return generate_samples_remote.remote(num_samples, input_dim, hidden_dim)
        except Exception as e:
            raise RuntimeError(f"Modal generation failed: {str(e)}")
