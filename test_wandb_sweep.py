import modal
import os
import time
import torch
import numpy as np
from src.models.tvae import TVAE
import wandb
import matplotlib.pyplot as plt

# Create Modal app
app = modal.App()

def train_model(config=None):
    """Training function to be called by wandb.agent"""
    import wandb

    # Initialize a new wandb run
    with wandb.init(project="sd1", name=f"tvae-sweep-{int(time.time())}", config=config) as run:
        config = wandb.config
        print(f"Starting training run {run.id} with config:", config)

        # Create a small synthetic dataset for testing
        input_dim = 10
        n_samples = 1000
        test_data = torch.randn(n_samples, input_dim)

        # Split data for evaluation
        train_size = int(0.8 * len(test_data))
        train_data = test_data[:train_size]
        val_data = test_data[train_size:]

        # Initialize model with config parameters
        model = TVAE(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            latent_dim=config.latent_dim,
            device='cpu'
        )

        # Training loop
        for epoch in range(config.epochs):
            epoch_metrics = {'epoch': epoch}
            batches = torch.split(train_data, config.batch_size)

            # Training metrics
            train_loss = 0
            for batch in batches:
                metrics = model.train_step(batch)
                train_loss += metrics['total_loss']

            avg_train_loss = train_loss / len(batches)

            # Validation and generation of synthetic data
            with torch.no_grad():
                # Generate synthetic samples
                synthetic_data = model.generate_samples(len(val_data))

                # Calculate validation loss
                val_metrics = model.train_step(val_data)
                val_loss = val_metrics['total_loss']

                # Calculate statistical similarity
                real_mean = val_data.mean(dim=0)
                real_std = val_data.std(dim=0)
                synth_mean = synthetic_data.mean(dim=0)
                synth_std = synthetic_data.std(dim=0)

                mean_diff = torch.mean(torch.abs(real_mean - synth_mean))
                std_diff = torch.mean(torch.abs(real_std - synth_std))

                # Create distribution plots
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))

                # Real data distribution
                axes[0].hist(val_data[:, 0].numpy(), bins=30, alpha=0.5, label='Real')
                axes[0].set_title('Real Data Distribution')
                axes[0].legend()

                # Synthetic data distribution
                axes[1].hist(synthetic_data[:, 0].numpy(), bins=30, alpha=0.5, label='Synthetic')
                axes[1].set_title('Synthetic Data Distribution')
                axes[1].legend()

                # Log metrics and plots to wandb
                metrics_dict = {
                    'train/loss': avg_train_loss,
                    'val/loss': val_loss,
                    'evaluation/mean_difference': mean_diff.item(),
                    'evaluation/std_difference': std_diff.item(),
                    'epoch': epoch,
                    'distributions': wandb.Image(fig),
                }

                # Add hyperparameters for tracking
                metrics_dict.update({
                    'hyperparameters/learning_rate': config.learning_rate,
                    'hyperparameters/latent_dim': config.latent_dim,
                    'hyperparameters/hidden_dim': config.hidden_dim,
                    'hyperparameters/batch_size': config.batch_size,
                })

                wandb.log(metrics_dict)
                plt.close(fig)

            print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}")

        print(f"Training completed. Final validation loss: {val_loss:.4f}")
        return val_loss  # Return validation loss for sweep optimization

@app.function(
    image=modal.Image.debian_slim()
        .pip_install(["wandb", "torch", "numpy", "matplotlib"])
        .add_local_dir("src", "/root/src")
        .add_local_python_source("sitecustomize", "src"),
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def run_sweep():
    """Run WandB sweep with Bayesian optimization"""
    import wandb

    # Define sweep config
    sweep_config = {
        'method': 'bayes',
        'name': 'tvae_optimization',
        'metric': {
            'name': 'val/loss',
            'goal': 'minimize'
        },
        'parameters': {
            'learning_rate': {
                'distribution': 'log_uniform',
                'min': float(1e-5),
                'max': float(1e-2)
            },
            'hidden_dim': {
                'values': [64, 128, 256, 512]
            },
            'latent_dim': {
                'values': [32, 64, 128, 256]
            },
            'batch_size': {
                'values': [32, 64, 128, 256]
            },
            'epochs': {
                'values': [10, 20, 30]
            }
        }
    }

    try:
        # First check if API key is present
        api_key = os.environ.get('WANDB_API_KEY')
        print(f"WANDB_API_KEY present in environment: {'Yes' if api_key else 'No'}")

        if not api_key:
            raise ValueError("WANDB_API_KEY not found in environment variables. Please ensure the Modal secret is properly configured.")

        # Initialize WandB
        print("Creating WandB sweep...")
        sweep_id = wandb.sweep(sweep=sweep_config, project="sd1")
        print(f"Created sweep with ID: {sweep_id}")

        # Start sweep agent
        print("Starting sweep agent...")
        wandb.agent(sweep_id, function=train_model, count=8)
        print("Sweep agent completed")

    except Exception as e:
        print(f"Error during sweep: {str(e)}")
        import traceback
        print("Full error traceback:")
        print(traceback.format_exc())
        raise e

    print("Sweep completed successfully")

if __name__ == "__main__":
    print("Starting WandB Modal test...")
    try:
        with app.run():
            run_sweep.remote()
            print("Modal function executed successfully")
    except Exception as e:
        print(f"Error running Modal function: {str(e)}")
        import traceback
        print(traceback.format_exc())
    print("Test completed.")