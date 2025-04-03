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
    """Test basic wandb functionality with graceful fallback"""
    
    # Check if API key is available
    api_key = os.environ.get('WANDB_API_KEY')
    print(f"WANDB_API_KEY present: {'Yes' if api_key else 'No'}")
    
    # Set flag to determine if WandB should be used
    use_wandb = True if api_key else False
    
    if not api_key:
        print("WANDB_API_KEY not found. Running in limited functionality mode.")
    
    try:
        if use_wandb:
            # Initialize wandb run
            run = wandb.init(project="test-project", name="test-run")
            print(f"WandB initialized successfully with run ID: {run.id}")
        else:
            print("Skipping WandB initialization (running in offline mode)")
        
        # Log some metrics
        for i in range(10):
            metrics = {"accuracy": i / 10, "loss": 1.0 - i / 10}
            
            if use_wandb:
                wandb.log(metrics)
                
            print(f"Metrics: {metrics}")
        
        # Finish the run if initialized
        if use_wandb:
            wandb.finish()
            print("WandB run completed successfully")
        else:
            print("Test completed (without WandB)")
            
        return True
    except Exception as e:
        print(f"Error in test: {str(e)}")
        
        # Try to clean up wandb run if it was created
        if use_wandb and 'run' in locals() and run is not None:
            try:
                wandb.finish()
            except:
                pass
                
        return False

if __name__ == "__main__":
    print("Starting WandB test...")
    test_basic_wandb()
    print("Test completed")