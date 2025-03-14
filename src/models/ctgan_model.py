
import torch
from ctgan import CTGAN
from src.models.base_gan import BaseGAN

class CTGANModel(BaseGAN):
    """CTGAN implementation for tabular data"""
    
    def __init__(self, input_dim: int, device: str = 'cpu', epochs: int = 300):
        super().__init__(input_dim, device)
        self.epochs = epochs
        self.model = CTGAN(epochs=epochs)
        
    def fit(self, data, discrete_columns=None):
        """Fit the CTGAN model"""
        if discrete_columns is None:
            discrete_columns = []
        self.model.fit(data, discrete_columns)
        
    def generate_samples(self, num_samples: int) -> torch.Tensor:
        """Generate synthetic samples"""
        synthetic_data = self.model.sample(num_samples)
        return torch.tensor(synthetic_data.values, dtype=torch.float32)
    
    def state_dict(self):
        """Get model state"""
        return {
            'model_state': self.model.get_state_dict(),
            'input_dim': self.input_dim,
            'device': self.device,
            'epochs': self.epochs
        }
        
    def load_state_dict(self, state_dict):
        """Load model state"""
        self.model.set_state_dict(state_dict['model_state'])
        self.input_dim = state_dict['input_dim']
        self.device = state_dict.get('device', 'cpu')
        self.epochs = state_dict.get('epochs', 300)
