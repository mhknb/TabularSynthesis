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

    # Set flag to determine if WandB should be used
    use_wandb = True if api_key else False
    
    if not api_key:
        print("WANDB_API_KEY not found in environment variables.")
        print("Running in anonymous mode with limited functionality.")

    try:
        if use_wandb:
            # Initialize wandb with the same project name as our models
            print("Initializing WandB...")
            # Explicitly login with API key to avoid no-tty issues
            wandb.login(key=api_key)
            run = wandb.init(project="sd1", name=f"test-run-{int(time.time())}")
            print(f"WandB initialized successfully. Run ID: {run.id}")

            # Configure some test parameters
            wandb.config = {
                "test_param": "testing modal integration",
                "environment": "modal-cloud"
            }
            print("WandB config set")
        else:
            print("Running without WandB integration")

        # Simulate a training loop
        print("Starting test metrics logging...")
        for i in range(10):
            metrics = {
                'test_loss': 1.0 / (i + 1),
                'test_accuracy': i / 10.0,
                'iteration': i
            }
            
            if use_wandb:
                wandb.log(metrics)
                
            print(f"Metrics for iteration {i}: {metrics}")
            time.sleep(1)  # Small delay to see metrics flow

        print("Test completed successfully")
    except Exception as e:
        print(f"Error during test: {str(e)}")
        import traceback
        print("Full error traceback:")
        print(traceback.format_exc())
        raise e
    finally:
        if use_wandb and wandb.run is not None:
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