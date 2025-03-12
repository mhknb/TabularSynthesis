import modal

app = modal.App()

@app.function(
    image=modal.Image.debian_slim().pip_install("wandb"),
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def test_wandb():
    import wandb
    
    wandb.init(project="sd1")
    wandb.config = {"test": "Modal integration test"}
    
    # Log a simple metric
    for i in range(5):
        wandb.log({"test_metric": i})
    
    wandb.finish()

if __name__ == "__main__":
    with app.run():
        test_wandb.remote()
