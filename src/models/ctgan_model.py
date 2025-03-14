import torch
import numpy as np
from ctgan import CTGAN
from src.models.base_gan import BaseGAN

class CTGANModel(BaseGAN):
    """CTGAN implementation for tabular data"""

    def __init__(self, input_dim: int, hidden_dim: int = 256, epochs=100, device: str = 'cpu'):
        super().__init__(input_dim, device)
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.model = CTGAN(
            epochs=epochs,
            embedding_dim=hidden_dim,
            verbose=True
        )
        self.is_fitted = False

    def train_step(self, real_data: torch.Tensor) -> dict:
        """Perform one training step"""
        # Convert torch tensor to numpy for CTGAN
        data = real_data.cpu().numpy()
        
        if not self.is_fitted:
            # CTGAN expects a DataFrame, so we'll create column names
            import pandas as pd
            columns = [f'col_{i}' for i in range(self.input_dim)]
            train_data = pd.DataFrame(data, columns=columns)
            
            # Train the model
            self.model.fit(train_data)
            self.is_fitted = True
            
            # Return metrics
            return {
                'status': 'Training completed',
                'epochs': self.epochs,
                'input_dim': self.input_dim
            }
        
        return {
            'status': 'Model already trained',
            'epochs': self.epochs,
            'input_dim': self.input_dim
        }

    def generate_samples(self, num_samples: int) -> torch.Tensor:
        """Generate synthetic samples"""
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before generating samples")
        
        # Generate synthetic data
        synthetic_data = self.model.sample(num_samples)
        
        # Convert back to torch tensor
        return torch.tensor(synthetic_data.values, dtype=torch.float32).to(self.device)

    def state_dict(self):
        """Get state dict for model persistence"""
        import pickle
        model_bytes = pickle.dumps(self.model)
        return {
            'model_bytes': model_bytes,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'epochs': self.epochs,
            'device': self.device,
            'is_fitted': self.is_fitted
        }

    def load_state_dict(self, state_dict):
        """Load state dict for model persistence"""
        import pickle
        self.model = pickle.loads(state_dict['model_bytes'])
        self.input_dim = state_dict['input_dim']
        self.hidden_dim = state_dict['hidden_dim']
        self.epochs = state_dict['epochs']
        self.device = state_dict['device']
        self.is_fitted = state_dict['is_fitted']
