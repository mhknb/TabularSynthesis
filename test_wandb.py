"""
Simple test for WandB compatibility
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb

print("Python packages imported successfully")

def test_basic_wandb():
    """Test basic wandb functionality"""
    try:
        # Initialize wandb run
        run = wandb.init(project="test-project", name="test-run")
        print(f"WandB initialized successfully with run ID: {run.id}")
        
        # Log some metrics
        for i in range(10):
            metrics = {"accuracy": i / 10, "loss": 1.0 - i / 10}
            wandb.log(metrics)
            print(f"Logged metrics: {metrics}")
        
        # Finish the run
        wandb.finish()
        print("WandB run completed successfully")
        return True
    except Exception as e:
        print(f"Error in WandB test: {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting WandB test...")
    test_basic_wandb()
    print("Test completed")