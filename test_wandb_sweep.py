import modal
import os
import time
import torch
import numpy as np
from src.models.tvae import TVAE

# Create Modal app
app = modal.App()

def train_model(config=None):
    """Training function to be called by wandb.agent"""
    import wandb

    # Initialize a new wandb run
    with wandb.init(config=config):
        config = wandb.config
        wandb.log({"Start": "Training started"}) #Added verbose logging

        # Create a small synthetic dataset for testing
        input_dim = 10
        n_samples = 1000
        test_data = torch.randn(n_samples, input_dim)

        # Initialize model with config parameters
        model = TVAE(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            latent_dim=config.latent_dim,
            device='cpu'
        )

        # Create optimizer with config learning rate
        optimizer = torch.optim.Adam(
            list(model.encoder.parameters()) + list(model.decoder.parameters()),
            lr=config.learning_rate
        )

        # Training loop
        for epoch in range(config.epochs):
            total_loss = 0
            batches = torch.split(test_data, config.batch_size)

            for batch in batches:
                metrics = model.train_step(batch)
                total_loss += metrics['total_loss']

                # Log metrics to wandb
                wandb.log({
                    'epoch': epoch,
                    'batch_loss': metrics['total_loss'],
                    'reconstruction_loss': metrics['reconstruction_loss'],
                    'kl_loss': metrics['kl_loss']
                })

            # Log epoch metrics
            avg_loss = total_loss / len(batches)
            wandb.log({
                'epoch': epoch,
                'average_loss': avg_loss
            })
        wandb.log({"End": "Training ended"}) #Added verbose logging

@app.function(
    image=modal.Image.debian_slim()
        .pip_install(["wandb", "torch", "numpy"])
        .add_local_dir("src", "/root/src")
        .add_local_python_source("sitecustomize", "src"),  # Add Python modules as suggested
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
            'name': 'average_loss',
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