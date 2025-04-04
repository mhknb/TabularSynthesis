import modal
import os
import time

# Create Modal app
app = modal.App()

@app.function(
    image=modal.Image.debian_slim().pip_install("wandb"),
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def test_wandb():
    """Test WandB integration with Modal"""
    import wandb

    # First check if API key is present
    api_key = os.environ.get('WANDB_API_KEY')
    print(f"WANDB_API_KEY present in environment: {'Yes' if api_key else 'No'}")

    if not api_key:
        raise ValueError("WANDB_API_KEY not found in environment variables. Please ensure the Modal secret is properly configured.")

    try:
        # Initialize wandb with the same project name as our models
        print("Initializing WandB...")
        run = wandb.init(project="sd1", name=f"test-run-{int(time.time())}")
        print(f"WandB initialized successfully. Run ID: {run.id}")

        # Configure some test parameters
        wandb.config = {
            "test_param": "testing modal integration",
            "environment": "modal-cloud"
        }
        print("WandB config set")

        # Simulate a training loop
        print("Starting test metrics logging...")
        for i in range(10):
            metrics = {
                'test_loss': 1.0 / (i + 1),
                'test_accuracy': i / 10.0,
                'iteration': i
            }
            wandb.log(metrics)
            print(f"Logged metrics for iteration {i}: {metrics}")
            time.sleep(1)  # Small delay to see metrics flow

        print("Test completed successfully")
    except Exception as e:
        print(f"Error during WandB test: {str(e)}")
        import traceback
        print("Full error traceback:")
        print(traceback.format_exc())
        raise e
    finally:
        if wandb.run is not None:
            print("Finishing WandB run...")
            wandb.finish()
            print("WandB run finished")

if __name__ == "__main__":
    print("Starting WandB Modal test...")
    try:
        with app.run():
            test_wandb.remote()
            print("Modal function executed successfully")
    except Exception as e:
        print(f"Error running Modal function: {str(e)}")
        import traceback
        print(traceback.format_exc())
    print("Test completed.")