"""
Test script for the improved TableGAN model with relationship-aware components
and categorical data handling
"""
import torch
import pandas as pd
import numpy as np
from src.models.table_gan import TableGAN
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

print("Testing TableGAN model with relationship-aware components and categorical data handling...")

try:
    # Load the continuous data example
    data_path = 'attached_assets/Continuous Data Example from GAN.ipynb.csv'
    df = pd.read_csv(data_path, header=0)
    print(f"Loaded data with shape: {df.shape}")
    
    # Show first few rows
    print("First 5 rows:")
    print(df.head())
    
    # Analyze column relationships
    col_names = df.columns
    for i in range(len(col_names)-1):
        ratio = df[col_names[i+1]] / df[col_names[i]]
        mean_ratio = ratio.mean()
        std_ratio = ratio.std()
        print(f"Mean ratio {col_names[i+1]}/{col_names[i]}: {mean_ratio:.4f} (std: {std_ratio:.6f})")
    
    # Convert to tensor
    train_data = torch.FloatTensor(df.values)
    batch_size = 32
    
    # Create data loader
    train_loader = torch.utils.data.DataLoader(
        train_data, 
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )
    
    # Initialize TableGAN
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize with default relationship weights
    input_dim = df.shape[1]
    table_gan = TableGAN(
        input_dim=input_dim,
        hidden_dim=128,
        device=device,
        min_batch_size=4
    )
    
    # Set initial loss weights
    table_gan.alpha = 1.0  # Weight for adversarial loss
    table_gan.beta = 10.0  # Weight for relationship loss
    table_gan.gamma = 0.1  # Weight for feature matching loss
    
    print(f"TableGAN initialized with {input_dim} input dimensions")
    print(f"Initial loss weights: alpha={table_gan.alpha}, beta={table_gan.beta}, gamma={table_gan.gamma}")
    
    # Run hyperparameter optimization
    print("Running hyperparameter optimization...")
    best_params, history_df = table_gan.optimize_hyperparameters(
        train_loader=train_loader,
        n_epochs=20,
        n_iterations=5
    )
    
    print(f"Best parameters: {best_params}")
    
    # Train for a few epochs
    n_epochs = 100
    total_steps = 0
    
    print(f"Starting training for {n_epochs} epochs...")
    metrics_history = []
    
    for epoch in range(n_epochs):
        epoch_losses = []
        for i, batch_data in enumerate(train_loader):
            # Perform one training step
            metrics = table_gan.train_step(batch_data)
            total_steps += 1
            epoch_losses.append(metrics)
        
        # Compute average losses for the epoch
        avg_gen_loss = np.mean([m['generator_loss'] for m in epoch_losses])
        avg_disc_loss = np.mean([m['discriminator_loss'] for m in epoch_losses])
        avg_relation_loss = np.mean([m.get('relationship_loss', 0) for m in epoch_losses])
        
        metrics_history.append({
            'epoch': epoch,
            'generator_loss': avg_gen_loss,
            'discriminator_loss': avg_disc_loss,
            'relationship_loss': avg_relation_loss
        })
        
        # Print progress every 10 epochs
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch+1}/{n_epochs}, "
                  f"Gen Loss: {avg_gen_loss:.4f}, Disc Loss: {avg_disc_loss:.4f}, "
                  f"Relation Loss: {avg_relation_loss:.4f}")
    
    # Generate samples
    print("Generating samples...")
    num_samples = 1000
    synthetic_tensor = table_gan.generate_samples(num_samples)
    synthetic_data = synthetic_tensor.cpu().detach().numpy()
    
    # Convert to DataFrame
    synthetic_df = pd.DataFrame(synthetic_data, columns=df.columns)
    print(f"Generated synthetic data with shape: {synthetic_df.shape}")
    print("First 5 synthetic samples:")
    print(synthetic_df.head())
    
    # Analyze synthetic data relationships
    print("\nAnalyzing synthetic data relationships:")
    for i in range(len(col_names)-1):
        ratio = synthetic_df[col_names[i+1]] / synthetic_df[col_names[i]]
        mean_ratio = ratio.mean()
        std_ratio = ratio.std()
        print(f"Mean ratio {col_names[i+1]}/{col_names[i]}: {mean_ratio:.4f} (std: {std_ratio:.6f})")
    
    # Plot comparison of real vs synthetic data
    plt.figure(figsize=(12, 8))
    
    # Plot real data distributions
    for i, col in enumerate(df.columns):
        plt.subplot(2, len(df.columns), i+1)
        plt.hist(df[col], bins=30, alpha=0.7, label='Real')
        plt.title(f'Real {col}')
        plt.grid(True, alpha=0.3)
    
    # Plot synthetic data distributions
    for i, col in enumerate(df.columns):
        plt.subplot(2, len(df.columns), i+1+len(df.columns))
        plt.hist(synthetic_df[col], bins=30, alpha=0.7, label='Synthetic')
        plt.title(f'Synthetic {col}')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('table_gan_test_result.png')
    print("Saved distribution comparison plot to 'table_gan_test_result.png'")
    
    # Save loss history plot
    plt.figure(figsize=(10, 6))
    epochs = [m['epoch'] for m in metrics_history]
    plt.plot(epochs, [m['generator_loss'] for m in metrics_history], label='Generator Loss')
    plt.plot(epochs, [m['discriminator_loss'] for m in metrics_history], label='Discriminator Loss')
    plt.plot(epochs, [m['relationship_loss'] for m in metrics_history], label='Relationship Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title('Training Loss History')
    plt.savefig('table_gan_loss_history.png')
    print("Saved loss history plot to 'table_gan_loss_history.png'")
    
    print("TableGAN continuous data test completed successfully!")
    
    #########################################################################
    # Now, test with categorical data
    #########################################################################
    
    print("\n\n" + "="*80)
    print("Testing TableGAN with categorical data...")
    print("="*80 + "\n")
    
    try:
        # Create a synthetic dataset with both continuous and categorical columns
        np.random.seed(42)
        n_samples = 1000
        
        # Continuous columns
        continuous_data = np.random.randn(n_samples, 2) * 5
        
        # Categorical columns - 3 categories (0, 1, 2)
        categorical_col1 = np.random.randint(0, 3, size=(n_samples, 1))
        categorical_col2 = np.random.randint(0, 2, size=(n_samples, 1))  # Binary category
        
        # Combine the data
        mixed_data = np.hstack([continuous_data, categorical_col1, categorical_col2])
        
        # Create DataFrame
        mixed_df = pd.DataFrame(
            mixed_data, 
            columns=['continuous1', 'continuous2', 'categorical3', 'categorical4']
        )
        
        print("Created mixed dataset with 2 continuous and 2 categorical columns:")
        print(mixed_df.head())
        
        # Show data statistics
        print("\nData statistics:")
        print(mixed_df.describe())
        
        # Show categorical distribution
        print("\nCategorical distributions:")
        for col in ['categorical3', 'categorical4']:
            print(f"{col} value counts:")
            print(mixed_df[col].value_counts())
        
        # Convert to tensor for training
        mixed_tensor = torch.FloatTensor(mixed_df.values)
        
        # Create data loader
        mixed_loader = torch.utils.data.DataLoader(
            mixed_tensor, 
            batch_size=32,
            shuffle=True,
            drop_last=False
        )
        
        # Set up TableGAN with categorical columns
        categorical_columns = [2, 3]  # 0-indexed column positions
        categorical_dims = {2: 3, 3: 2}  # Number of categories for each column
        
        mixed_gan = TableGAN(
            input_dim=mixed_df.shape[1],
            hidden_dim=128,
            device=device,
            min_batch_size=4,
            categorical_columns=categorical_columns,
            categorical_dims=categorical_dims
        )
        
        print(f"Initialized TableGAN with categorical columns: {categorical_columns}")
        print(f"Number of categories per column: {categorical_dims}")
        
        # Train for a few epochs
        n_epochs = 50
        print(f"Training for {n_epochs} epochs...")
        
        for epoch in range(n_epochs):
            epoch_losses = []
            for batch_data in mixed_loader:
                metrics = mixed_gan.train_step(batch_data)
                epoch_losses.append(metrics)
            
            # Only print every 10 epochs
            if epoch % 10 == 0 or epoch == n_epochs - 1:
                avg_gen_loss = np.mean([m['generator_loss'] for m in epoch_losses])
                avg_disc_loss = np.mean([m['discriminator_loss'] for m in epoch_losses])
                # Check if entropy loss is available
                if 'entropy_loss' in epoch_losses[0]:
                    avg_entropy = np.mean([m.get('entropy_loss', 0) for m in epoch_losses])
                    print(f"Epoch {epoch+1}/{n_epochs}, Gen: {avg_gen_loss:.4f}, Disc: {avg_disc_loss:.4f}, Entropy: {avg_entropy:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{n_epochs}, Gen: {avg_gen_loss:.4f}, Disc: {avg_disc_loss:.4f}")
        
        # Generate synthetic samples with various temperatures
        temperatures = [0.2, 0.5, 0.8, 1.0]
        fig, axes = plt.subplots(len(temperatures), 2, figsize=(12, 4*len(temperatures)))
        
        for i, temp in enumerate(temperatures):
            print(f"Generating samples with temperature={temp}...")
            synth_tensor = mixed_gan.generate_samples(num_samples=1000, temperature=temp)
            synth_array = synth_tensor.cpu().detach().numpy()
            synth_df = pd.DataFrame(synth_array, columns=mixed_df.columns)
            
            # Round categorical columns to nearest integers
            synth_df['categorical3'] = np.round(synth_df['categorical3']).astype(int)
            synth_df['categorical4'] = np.round(synth_df['categorical4']).astype(int)
            
            # Show value counts
            cat3_counts = synth_df['categorical3'].value_counts().sort_index()
            cat4_counts = synth_df['categorical4'].value_counts().sort_index()
            
            print(f"Temperature {temp} categorical3 counts: {dict(cat3_counts)}")
            print(f"Temperature {temp} categorical4 counts: {dict(cat4_counts)}")
            
            # Plot categorical distributions
            cat3_counts.plot.bar(ax=axes[i,0], alpha=0.7)
            axes[i,0].set_title(f'categorical3 (temp={temp})')
            axes[i,0].grid(alpha=0.3)
            
            cat4_counts.plot.bar(ax=axes[i,1], alpha=0.7)
            axes[i,1].set_title(f'categorical4 (temp={temp})')
            axes[i,1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('table_gan_categorical_temperature.png')
        print("Saved categorical temperature plot to 'table_gan_categorical_temperature.png'")
        
        # Plot comparison of continuous columns
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Real continuous distributions
        for i, col in enumerate(['continuous1', 'continuous2']):
            sns.kdeplot(mixed_df[col], ax=axes[i], label='Real', fill=True, alpha=0.3)
        
        # Synthetic continuous distributions for different temperatures
        for temp in temperatures:
            synth_tensor = mixed_gan.generate_samples(num_samples=1000, temperature=temp)
            synth_array = synth_tensor.cpu().detach().numpy()
            synth_df = pd.DataFrame(synth_array, columns=mixed_df.columns)
            
            for i, col in enumerate(['continuous1', 'continuous2']):
                sns.kdeplot(synth_df[col], ax=axes[i], label=f'Temp={temp}', alpha=0.6)
        
        axes[0].set_title('continuous1')
        axes[0].grid(alpha=0.3)
        axes[0].legend()
        
        axes[1].set_title('continuous2')
        axes[1].grid(alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig('table_gan_continuous_temperature.png')
        print("Saved continuous temperature plot to 'table_gan_continuous_temperature.png'")
        
        print("CategoryGAN mixed data test completed successfully!")
        
    except Exception as e:
        print(f"Error testing CategoryGAN with mixed data: {e}")
        import traceback
        traceback.print_exc()
    
except Exception as e:
    print(f"Error testing TableGAN: {e}")
    import traceback
    traceback.print_exc()