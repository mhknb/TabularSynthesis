
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
    
    # Print teams safely - inspecting the actual structure first
    teams = user.get('teams', [])
    if teams:
        if isinstance(teams, list):
            # Try to safely extract team names
            team_names = []
            for team in teams:
                if isinstance(team, dict) and 'name' in team:
                    team_names.append(team['name'])
                elif hasattr(team, 'name'):
                    team_names.append(team.name)
            print(f"WandB Teams: {team_names}")
        else:
            print(f"Teams data structure: {type(teams)}")
    else:
        print("No teams found")
    
    # Initialize a test run
    run = wandb.init(
        project="sd1",
        entity="smilai",
        name=f"test-run-{int(time.time())}",
        config={"test": True},
        reinit=True
    )
    
    # Log some test metrics
    for i in range(5):
        dummy_loss = 1.0 - (i * 0.1)
        dummy_acc = i * 0.1
        
        wandb.log({
            "loss": dummy_loss,
            "accuracy": dummy_acc,
            "step": i
        })
        print(f"Logged metrics - Step {i}, Loss: {dummy_loss}, Accuracy: {dummy_acc}")
    
    # Finish the run
    wandb.finish()
    
    print(f"Test complete! Please check https://wandb.ai/smilai/sd1 to see your test run.")
    if hasattr(run, 'get_url'):
        print(f"Run URL: {run.get_url()}")
    
except Exception as e:
    print(f"Error testing wandb: {e}")
    import traceback
    traceback.print_exc()
