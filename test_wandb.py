
import os
import wandb
import time
import torch
import numpy as np

# Configure wandb with your credentials
os.environ["WANDB_ENTITY"] = "smilai"
os.environ["WANDB_PROJECT"] = "sd1"
os.environ["WANDB_API_KEY"] = "74b2d1e3cfbeefbf3732f36430b2508f51bb0c34"

print(f"Testing WandB connection...")

try:
    # Login to wandb
    wandb.login(key=os.environ["WANDB_API_KEY"])
    
    # Print user info
    user = wandb.api.viewer()
    print(f"Logged in as: {user.get('entity', 'unknown')}")
    print(f"WandB Teams: {[team['name'] for team in user.get('teams', [])]}")
    
    # Initialize a test run with reinit=True to ensure a new run is created
    run = wandb.init(
        project="sd1",
        entity="smilai",
        name=f"test-run-{int(time.time())}",
        config={"test": True},
        reinit=True
    )
    
    # Create and log some dummy model data
    dummy_model = torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 1)
    )
    
    # Log some test metrics
    for i in range(10):
        dummy_loss = 1.0 - (i * 0.1)
        dummy_acc = i * 0.1
        
        wandb.log({
            "loss": dummy_loss,
            "accuracy": dummy_acc,
            "step": i
        })
        print(f"Logged metrics - Step {i}, Loss: {dummy_loss}, Accuracy: {dummy_acc}")
        time.sleep(1)
    
    # Log a simple plot
    wandb.log({"chart": wandb.plot.line_series(
        xs=[[i for i in range(10)]], 
        ys=[[np.random.rand() for _ in range(10)]], 
        keys=["random values"], 
        title="Random Values Plot"
    )})
    
    # Finish the run
    wandb.finish()
    
    print(f"Test complete! Please check https://wandb.ai/smilai/sd1 to see your test run.")
    print(f"Run URL: {run.get_url()}")
    
except Exception as e:
    print(f"Error testing wandb: {e}")
    import traceback
    traceback.print_exc()
