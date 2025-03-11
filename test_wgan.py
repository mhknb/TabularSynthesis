
import torch
import pandas as pd
import numpy as np
from src.models.wgan import WGAN
import time
import wandb
import os

print("Testing WGAN model...")

# Make sure WandB is set up
os.environ["WANDB_ENTITY"] = "smilai"
os.environ["WANDB_PROJECT"] = "sd1"

try:
    # Create sample data
    n_samples = 100
    n_features = 10
    synthetic_data = np.random.randn(n_samples, n_features)
    
    # Convert to tensor
    train_data = torch.FloatTensor(synthetic_data)
    batch_size = 32
    
    # Create data loader
    train_loader = torch.utils.data.DataLoader(
        train_data, 
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    
    # Initialize WGAN
    device = torch.device('cpu')
    wgan = WGAN(
        input_dim=n_features,
        hidden_dim=64,
        clip_value=0.01,
        n_critic=5,
        lr_g=0.0001,
        lr_d=0.0001,
        device=device,
        use_wandb=True
    )
    
    print(f"WGAN initialized successfully with {n_features} input dimensions")
    
    # Train for a few steps
    n_epochs = 3
    
    print(f"Starting training for {n_epochs} epochs...")
    for epoch in range(n_epochs):
        epoch_losses = []
        for i, batch_data in enumerate(train_loader):
            # Perform one training step
            metrics = wgan.train_step(batch_data, current_step=epoch * len(train_loader) + i)
            
            # Print progress
            if i % 2 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}, Batch {i+1}/{len(train_loader)}, "
                      f"Gen Loss: {metrics['gen_loss']:.4f}, Disc Loss: {metrics['disc_loss']:.4f}")
            
            epoch_losses.append(metrics)
    
    # Generate samples
    print("Generating samples...")
    samples = wgan.generate_samples(10)
    print(f"Generated samples shape: {samples.shape}")
    
    # Finish wandb run
    wgan.finish_wandb()
    
    print("WGAN test completed successfully!")
    
except Exception as e:
    print(f"Error testing WGAN: {e}")
    import traceback
    traceback.print_exc()
