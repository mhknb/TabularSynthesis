"""
Script to upload a local model to the Modal volume
"""
import os
import modal
import shutil

# Define app and shared resources
app = modal.App()
volume = modal.Volume.from_name("gan-model-vol", create_if_missing=True)

@app.function(volumes={"/model": volume})
def upload_model(model_name):
    """Upload a model from local models directory to Modal volume"""
    import os
    import shutil
    from pathlib import Path
    
    # Get the absolute path to where the file should be
    local_path = str(Path(__file__).parent / "models" / model_name)
    remote_path = f"/model/{model_name}"
    
    print(f"Checking for model at: {local_path}")
    
    if not os.path.exists(local_path):
        return False, f"Local model not found at {local_path}"
    
    # Copy the file to the volume
    try:
        with open(local_path, 'rb') as src_file:
            with open(remote_path, 'wb') as dst_file:
                shutil.copyfileobj(src_file, dst_file)
        return True, f"Model uploaded to {remote_path}"
    except Exception as e:
        return False, f"Error copying file: {str(e)}"

def main():
    try:
        # List local models
        if not os.path.exists('models'):
            print("No local models directory found")
            return
        
        local_models = os.listdir('models')
        if not local_models:
            print("No local models found")
            return
        
        print("Available local models:")
        for i, model in enumerate(local_models):
            print(f"{i+1}. {model}")
        
        # Upload all models
        for model in local_models:
            print(f"Uploading {model}...")
            with app.run():
                success, message = upload_model.remote(model)
            
            if success:
                print(f"✓ Success: {message}")
            else:
                print(f"✗ Failed: {message}")
                
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()