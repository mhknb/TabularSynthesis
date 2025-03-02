from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple

class BaseGAN(ABC):
    """Abstract base class for GAN implementations"""
    
    def __init__(self, input_dim: int, device: str = 'cpu'):
        self.input_dim = input_dim
        self.device = device
        self.generator = None
        self.discriminator = None
        
    @abstractmethod
    def build_generator(self) -> nn.Module:
        """Build the generator network"""
        pass
    
    @abstractmethod
    def build_discriminator(self) -> nn.Module:
        """Build the discriminator network"""
        pass
    
    @abstractmethod
    def train_step(self, real_data: torch.Tensor) -> Dict[str, float]:
        """Perform one training step"""
        pass
    
    def generate_samples(self, num_samples: int) -> torch.Tensor:
        """Generate synthetic samples"""
        with torch.no_grad():
            noise = torch.randn(num_samples, self.input_dim).to(self.device)
            return self.generator(noise)
    
    def train(self, dataloader: torch.utils.data.DataLoader, 
              epochs: int, 
              callback=None) -> list:
        """Train the GAN model"""
        losses = []
        for epoch in range(epochs):
            epoch_losses = []
            for batch in dataloader:
                loss_dict = self.train_step(batch)
                epoch_losses.append(loss_dict)
                
            # Calculate average losses for epoch
            avg_losses = {k: sum(d[k] for d in epoch_losses) / len(epoch_losses) 
                         for k in epoch_losses[0].keys()}
            losses.append(avg_losses)
            
            if callback:
                callback(epoch, avg_losses)
                
        return losses
