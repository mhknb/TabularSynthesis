
import pandas as pd
import torch
from src.models.modal_gan import ModalGAN

# Load your data
data = pd.read_csv("attached_assets/sample_dataset.csv")

# Initialize ModalGAN
gan = ModalGAN()

# Train the model
# You can choose model_type from: 'TableGAN', 'TVAE', 'CGAN', 'CTGAN', 'WGAN'
gan.train(
    data=data,
    input_dim=data.shape[1],  # Number of columns
    hidden_dim=256,           # Hidden dimension size
    epochs=100,               # Number of training epochs
    batch_size=32,           # Batch size
    model_type='TableGAN',    # Model type to use
    model_name='my_model'     # Name to save the model as
)
